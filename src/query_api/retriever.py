"""
pgvector retrieval layer for ExpertiseRAG.

Replaces the Bedrock Knowledge Base Retrieve API with direct queries against
the PostgreSQL pgvector chunks table, using Bedrock Titan V2 embeddings for
semantic similarity.

Supports multiple retrieval strategies via retrieve_with_strategy():
  - semantic_search   : pgvector cosine similarity (default)
  - graph_first       : same as semantic, but graph expansion is forced
  - keyword_boosted   : semantic + keyword-overlap reranking
  - hybrid            : semantic primary + secondary pgvector pass merged

The RAGRouter selects the strategy at query time; this module executes it.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

from models import RetrievalConfig, RetrievedChunk, RAGStrategyLabel, get_source_weight

logger = logging.getLogger(__name__)


def _get_shared_module(name: str):
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    import importlib
    return importlib.import_module(f"shared.{name}")


# ─────────────────────────────────────────────────────────────────────────────
# Core pgvector retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(
    question: str,
    knowledge_base_id: str = "",  # kept for API compatibility; unused
    top_k: int = 10,
    min_score: float = 0.0,
    doc_type_filter: str | None = None,
    strategy: str = "",
) -> list[RetrievedChunk]:
    """
    Embed the question with Titan V2 and run a pgvector cosine similarity query.

    Evidence weighting (same thresholds as before):
      - architecture.md, CLAUDE.md → weight 1.0 (highest)
      - PlantUML-derived summaries → weight 0.8
      - code / README               → weight 0.6
      - resume                      → weight 0.3

    Chunks are sorted by effective_score (cosine_similarity × source_weight).
    """
    embedder = _get_shared_module("embedder")
    db_clients = _get_shared_module("db_clients")

    embedding = embedder.embed_text(question)
    if embedding is None:
        logger.error("Failed to embed question – cannot retrieve chunks")
        return []

    conn = db_clients.get_pg_connection()

    # Retrieve child chunks preferentially (more precise); fall back to all
    # chunks for strategies that need broader context.
    base_query = """
        SELECT
            id::TEXT,
            content,
            source_file,
            doc_type,
            strategy,
            parent_content,
            is_child,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM chunks
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    where_parts = ["embedding IS NOT NULL"]
    params: list[Any] = [embedding, embedding, top_k * 2]  # over-fetch for dedup

    if doc_type_filter:
        where_parts.append("doc_type = %s")
        params.insert(2, doc_type_filter)  # insert before LIMIT param

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
    query = base_query.format(where_clause=where_clause)

    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description]
    except Exception as exc:
        logger.error("pgvector retrieve failed: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return []

    chunks: list[RetrievedChunk] = []
    for row in rows:
        r = dict(zip(columns, row))
        score = float(r.get("score", 0.0))
        source_file = r.get("source_file", "")
        metadata = r.get("metadata") or {}
        if r.get("doc_type"):
            metadata["doc_type"] = r["doc_type"]
        if r.get("strategy"):
            metadata["strategy"] = r["strategy"]
        if r.get("is_child"):
            metadata["is_child"] = r["is_child"]

        source_uri = f"pgvector://{source_file}"
        chunk = RetrievedChunk(
            content=r.get("content", ""),
            score=score,
            source_uri=source_uri,
            source_weight=get_source_weight(source_file, strategy),
            metadata=metadata,
        )
        if chunk.effective_score >= min_score:
            chunks.append(chunk)

    # Sort by effective score (cosine similarity × authority weight)
    chunks.sort(key=lambda c: c.effective_score, reverse=True)
    chunks = chunks[:top_k]

    logger.info(
        "pgvector retrieved %d chunks (top effective score: %.3f)",
        len(chunks),
        chunks[0].effective_score if chunks else 0.0,
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy-aware retrieval dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_with_strategy(
    question: str,
    knowledge_base_id: str = "",  # kept for API compatibility; unused
    config: RetrievalConfig = None,
    top_k: int = 10,
    min_score: float = 0.0,
    neptune_graph_id: str = "",   # kept for API compatibility; unused
) -> list[RetrievedChunk]:
    """
    Retrieve chunks using the strategy specified in config.

    Dispatches to the appropriate combination of:
      - pgvector semantic search (always the primary source)
      - Keyword-overlap reranking (keyword_boosted / hybrid)
      - Secondary pgvector pass for alternative doc types (hybrid)

    Returns a deduplicated, ranked list of RetrievedChunk objects.
    """
    if config is None:
        from models import RetrievalConfig as _RC, RAGStrategyLabel as _RSL
        config = _RC(strategy=_RSL.SEMANTIC)

    strategy = config.strategy

    # Primary: pgvector semantic search with strategy-aware source weights.
    # PDF/resume chunks score higher for keyword_boosted (project/credential);
    # authoritative MD files score higher for graph_first (skill/architecture).
    chunks = retrieve_chunks(question, top_k=top_k, min_score=min_score, strategy=strategy)

    # Keyword-boosted reranking
    if config.boost_keywords and strategy in (
        RAGStrategyLabel.KEYWORD_BOOSTED, RAGStrategyLabel.HYBRID
    ):
        chunks = _keyword_boost_rerank(chunks, question)

    # Hybrid: secondary pgvector pass over a complementary doc_type
    if config.use_neptune_chunks and strategy == RAGStrategyLabel.HYBRID:
        secondary = _retrieve_secondary(question, top_k=top_k // 2)
        chunks = _merge_results(chunks, secondary, max_total=top_k)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Keyword-boosted reranking
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "and", "or",
    "but", "nor", "so", "yet", "both", "either", "neither", "not", "only",
    "own", "same", "than", "too", "very", "just", "this", "that", "what",
    "which", "who", "how", "when", "where", "why",
})


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful lowercase tokens, removing stopwords."""
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _keyword_overlap_score(chunk_text: str, question_keywords: set[str]) -> float:
    """Fraction of question keywords present in the chunk text. Returns 0–1."""
    if not question_keywords:
        return 0.0
    chunk_keywords = _extract_keywords(chunk_text)
    overlap = len(question_keywords & chunk_keywords)
    return overlap / len(question_keywords)


def _keyword_boost_rerank(
    chunks: list[RetrievedChunk],
    question: str,
    keyword_weight: float = 0.25,
) -> list[RetrievedChunk]:
    """
    Rerank chunks by blending semantic effective_score with keyword overlap.

    boosted_score = (1 - keyword_weight) × effective_score
                  + keyword_weight       × keyword_overlap_score
    """
    q_keywords = _extract_keywords(question)
    if not q_keywords:
        return chunks

    def _sort_key(chunk: RetrievedChunk) -> float:
        kw_score = _keyword_overlap_score(chunk.content, q_keywords)
        return (1 - keyword_weight) * chunk.effective_score + keyword_weight * kw_score

    reranked = sorted(chunks, key=_sort_key, reverse=True)
    logger.info("Keyword-boost reranking applied to %d chunks", len(reranked))
    return reranked


# ─────────────────────────────────────────────────────────────────────────────
# Secondary pgvector pass (hybrid strategy)
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve_secondary(
    question: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Secondary pgvector retrieval that targets parent (non-child) chunks only.
    Used by the hybrid strategy to complement child-chunk primary results
    with broader context chunks.
    """
    embedder = _get_shared_module("embedder")
    db_clients = _get_shared_module("db_clients")

    embedding = embedder.embed_text(question)
    if embedding is None:
        return []

    query = """
        SELECT
            id::TEXT,
            content,
            source_file,
            doc_type,
            strategy,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
          AND (is_child = FALSE OR is_child IS NULL)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    conn = db_clients.get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, [embedding, embedding, top_k])
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description]
    except Exception as exc:
        logger.warning("Secondary pgvector pass failed (non-fatal): %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return []

    chunks = []
    for row in rows:
        r = dict(zip(columns, row))
        source_file = r.get("source_file", "")
        metadata = r.get("metadata") or {}
        metadata["source"] = "pgvector_secondary"
        chunks.append(RetrievedChunk(
            content=r.get("content", ""),
            score=float(r.get("score", 0.0)),
            source_uri=f"pgvector://{source_file}",
            source_weight=get_source_weight(source_file),
            metadata=metadata,
        ))

    logger.info("Secondary pgvector pass returned %d chunks", len(chunks))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Result merging (hybrid strategy)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_results(
    primary: list[RetrievedChunk],
    secondary: list[RetrievedChunk],
    max_total: int = 10,
) -> list[RetrievedChunk]:
    """
    Merge primary and secondary chunk lists, dedup, and re-rank.

    Secondary chunks are included only if their content is not near-duplicate
    of primary chunks (Jaccard threshold 0.85).
    """
    merged = list(primary)
    primary_prefixes = [c.content[:150] for c in primary]

    for chunk in secondary:
        prefix = chunk.content[:150]
        is_dup = any(
            _prefix_similarity(prefix, p) >= 0.85
            for p in primary_prefixes
        )
        if not is_dup:
            merged.append(chunk)
            primary_prefixes.append(prefix)

    merged.sort(key=lambda c: c.effective_score, reverse=True)
    return merged[:max_total]


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_chunks(chunks: list[RetrievedChunk], threshold: float = 0.92) -> list[RetrievedChunk]:
    """
    Remove near-duplicate chunks based on content similarity (prefix overlap heuristic).
    A full semantic dedup would require embeddings; this is a fast approximation.
    """
    seen: list[str] = []
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        prefix = chunk.content[:150]
        is_dup = any(
            _prefix_similarity(prefix, s) >= threshold
            for s in seen
        )
        if not is_dup:
            seen.append(prefix)
            unique.append(chunk)
    return unique


def _prefix_similarity(a: str, b: str) -> float:
    """Jaccard similarity of word sets for fast near-dup detection."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
