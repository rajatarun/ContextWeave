"""
Bedrock Knowledge Base retrieval layer for ExpertiseRAG.

Wraps the Bedrock Agent Runtime `retrieve` API and applies
source-authority weighting to the returned chunks.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.models import RetrievedChunk, get_source_weight

logger = logging.getLogger(__name__)

# Lazy client – reused across warm Lambda invocations
_BEDROCK_AGENT_RUNTIME: Any = None


def _get_client() -> Any:
    global _BEDROCK_AGENT_RUNTIME
    if _BEDROCK_AGENT_RUNTIME is None:
        _BEDROCK_AGENT_RUNTIME = boto3.client(
            "bedrock-agent-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _BEDROCK_AGENT_RUNTIME


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve
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
    client = _get_client()

    try:
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "HYBRID",  # semantic + keyword
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
