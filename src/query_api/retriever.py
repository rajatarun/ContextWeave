"""
Bedrock Knowledge Base retrieval layer for ExpertiseRAG.

Wraps the Bedrock Agent Runtime `retrieve` API and applies
source-authority weighting to the returned chunks.

Supports multiple retrieval strategies via retrieve_with_strategy():
  - semantic_search   : pure Bedrock KB vector search (default)
  - graph_first       : same as semantic, but graph expansion is forced
  - keyword_boosted   : semantic + keyword-overlap reranking
  - hybrid            : semantic + Neptune vector chunks merged

The RAGRouter selects the strategy at query time; this module executes it.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import boto3
from botocore.exceptions import ClientError

from models import RetrievalConfig, RetrievedChunk, RAGStrategyLabel, get_source_weight

logger = logging.getLogger(__name__)

# Lazy clients – reused across warm Lambda invocations
_BEDROCK_AGENT_RUNTIME: Any = None
_NEPTUNE_CLIENT: Any = None


def _get_bedrock_client() -> Any:
    global _BEDROCK_AGENT_RUNTIME
    if _BEDROCK_AGENT_RUNTIME is None:
        _BEDROCK_AGENT_RUNTIME = boto3.client(
            "bedrock-agent-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _BEDROCK_AGENT_RUNTIME


def _get_neptune_client() -> Any:
    global _NEPTUNE_CLIENT
    if _NEPTUNE_CLIENT is None:
        _NEPTUNE_CLIENT = boto3.client(
            "neptune-graph",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _NEPTUNE_CLIENT


# ─────────────────────────────────────────────────────────────────────────────
# Core Bedrock KB retrieval (unchanged semantic baseline)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(
    question: str,
    knowledge_base_id: str,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """
    Call Bedrock Knowledge Base Retrieve API and return weighted chunks.

    Evidence weighting:
      - architecture.md, CLAUDE.md → weight 1.0 (highest)
      - PlantUML-derived summaries → weight 0.8
      - code / README               → weight 0.6
      - articles / blog             → weight 0.5
      - resume                      → weight 0.3

    Chunks are sorted by effective_score (raw score × source_weight).
    """
    client = _get_bedrock_client()

    try:
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "SEMANTIC",
                }
            },
        )
    except ClientError as exc:
        logger.error("Bedrock Retrieve failed: %s", exc)
        raise

    chunks: list[RetrievedChunk] = []

    for result in response.get("retrievalResults", []):
        content = result.get("content", {}).get("text", "")
        score = float(result.get("score", 0.0))
        location = result.get("location", {})
        source_uri = (
            location.get("s3Location", {}).get("uri", "")
            or location.get("uri", "")
        )
        metadata = result.get("metadata", {})

        # Extract source file name for weighting
        source_file = source_uri.split("/")[-1] if source_uri else ""
        source_weight = get_source_weight(source_file)

        chunk = RetrievedChunk(
            content=content,
            score=score,
            source_uri=source_uri,
            source_weight=source_weight,
            metadata=metadata,
        )

        if chunk.effective_score >= min_score:
            chunks.append(chunk)

    # Sort by effective score (semantic score × authority weight)
    chunks.sort(key=lambda c: c.effective_score, reverse=True)

    logger.info(
        "Retrieved %d chunks (top effective score: %.3f)",
        len(chunks),
        chunks[0].effective_score if chunks else 0.0,
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy-aware retrieval dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_with_strategy(
    question: str,
    knowledge_base_id: str,
    config: RetrievalConfig,
    top_k: int = 10,
    min_score: float = 0.0,
    neptune_graph_id: str = "",
) -> list[RetrievedChunk]:
    """
    Retrieve chunks using the strategy specified in config.

    Dispatches to the appropriate combination of:
      - Bedrock KB semantic search (always the primary source)
      - Keyword-overlap reranking (keyword_boosted / hybrid)
      - Neptune vector search (hybrid only)

    Returns a deduplicated, ranked list of RetrievedChunk objects.
    """
    # Primary: always start with Bedrock KB semantic search
    chunks = retrieve_chunks(question, knowledge_base_id, top_k=top_k, min_score=min_score)

    strategy = config.strategy

    # Keyword-boosted reranking
    if config.boost_keywords and strategy in (
        RAGStrategyLabel.KEYWORD_BOOSTED, RAGStrategyLabel.HYBRID
    ):
        chunks = _keyword_boost_rerank(chunks, question)

    # Hybrid: also search Neptune vector store and merge
    if config.use_neptune_chunks and strategy == RAGStrategyLabel.HYBRID:
        gid = neptune_graph_id or os.environ.get("NEPTUNE_GRAPH_ID", "")
        if gid:
            neptune_chunks = _retrieve_from_neptune(question, gid, top_k=top_k // 2)
            chunks = _merge_results(chunks, neptune_chunks, max_total=top_k)

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
    """
    Fraction of question keywords present in the chunk text.
    Returns 0–1.
    """
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

    The chunk's score attribute is NOT mutated; reranking is done via sort key.
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
# Neptune vector search (hybrid strategy)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_question(question: str) -> list[float] | None:
    """
    Generate a 1024-dim Titan Text Embeddings V2 vector for the question.
    Returns None if embedding fails (non-fatal).
    """
    import json
    import boto3 as _boto3

    try:
        bedrock = _boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        body = json.dumps({"inputText": question, "dimensions": 1024, "normalize": True})
        resp = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        return json.loads(resp["body"].read())["embedding"]
    except Exception as exc:
        logger.warning("Embedding failed (non-fatal): %s", exc)
        return None


def _retrieve_from_neptune(
    question: str,
    graph_id: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Vector search over ChunkVector nodes stored in Neptune Analytics.
    These are alternative-strategy chunks embedded during preprocessing.

    Returns RetrievedChunk objects with source_uri prefixed 'neptune://'.
    """
    import json

    embedding = _embed_question(question)
    if not embedding:
        return []

    client = _get_neptune_client()
    query = """
    CALL neptune.algo.vectors.topKByEmbedding(
        $embedding,
        {topK: $top_k, concurrency: 2}
    )
    YIELD node, score
    WHERE node:ChunkVector
    RETURN
        node.text        AS text,
        node.doc_type    AS doc_type,
        node.strategy    AS strategy,
        node.doc_node_id AS doc_node_id,
        score
    """
    try:
        kwargs: dict[str, Any] = {
            "graphIdentifier": graph_id,
            "queryString": query,
            "language": "OPEN_CYPHER",
            "parameters": json.dumps({"embedding": embedding, "top_k": top_k}),
        }
        response = client.execute_query(**kwargs)
        payload = response.get("payload")
        raw = payload.read() if hasattr(payload, "read") else (payload or b"{}")
        rows = json.loads(raw).get("results", [])
    except Exception as exc:
        logger.warning("Neptune vector search failed (non-fatal): %s", exc)
        return []

    chunks = []
    for row in rows:
        text = row.get("text", "")
        score = float(row.get("score", 0.0))
        strategy = row.get("strategy", "unknown")
        doc_node_id = row.get("doc_node_id", "")
        source_uri = f"neptune://{doc_node_id}/{strategy}"
        # Neptune chunks have weight 0.6 by default (similar to code/README)
        chunks.append(RetrievedChunk(
            content=text,
            score=score,
            source_uri=source_uri,
            source_weight=0.6,
            metadata={"strategy": strategy, "doc_node_id": doc_node_id, "source": "neptune"},
        ))

    logger.info("Neptune vector search returned %d chunks", len(chunks))
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
    Merge Bedrock KB chunks with Neptune vector chunks, dedup, and re-rank.

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
        # Check if similar prefix already seen
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
