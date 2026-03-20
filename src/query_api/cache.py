"""
Cache-Augmented Generation (CAG) for ExpertiseRAG.

Uses the existing PostgreSQL/pgvector store to cache full query responses
keyed by the question's Titan V2 embedding. A cosine-similarity threshold
(default 0.95) treats near-identical questions as cache hits, skipping the
entire classify → route → retrieve → expand → synthesize pipeline.

No new AWS services required — zero additional monthly cost.

Environment variables:
  CACHE_SIMILARITY_THRESHOLD  float  default 0.95   cosine threshold for hit
  CACHE_TTL_DAYS              int    default 7       days before entry expires
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.95"))
CACHE_TTL_DAYS = int(os.environ.get("CACHE_TTL_DAYS", "7"))

# Questions containing these patterns reference time-sensitive context
# and should never be served from cache.
_TIME_SENSITIVE = re.compile(
    r"\b(today|yesterday|this week|this month|this year|currently|right now|"
    r"latest|recent|just|now|at the moment|as of|current)\b",
    re.IGNORECASE,
)


def is_time_sensitive(question: str) -> bool:
    """Return True if the question should bypass the cache."""
    return bool(_TIME_SENSITIVE.search(question))


def check_cache(embedding: list[float], conn: Any) -> dict | None:
    """
    Look up a semantically similar cached response.

    Searches query_cache for an unexpired entry whose question_embedding has
    cosine similarity ≥ SIMILARITY_THRESHOLD with the given embedding.
    Increments hit_count on a match and returns the stored response dict.
    Returns None on a cache miss or any DB error (non-fatal).
    """
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, response_json
                FROM query_cache
                WHERE expires_at > NOW()
                  AND 1 - (question_embedding <=> %s::vector) >= %s
                ORDER BY question_embedding <=> %s::vector
                LIMIT 1
                """,
                (embedding_str, SIMILARITY_THRESHOLD, embedding_str),
            )
            row = cur.fetchone()
            if row is None:
                return None

            cache_id, response_json = row

            # Fire-and-forget hit count increment — ignore failures
            try:
                cur.execute(
                    "UPDATE query_cache SET hit_count = hit_count + 1 WHERE id = %s",
                    (cache_id,),
                )
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass

            return response_json if isinstance(response_json, dict) else json.loads(response_json)

    except Exception as exc:
        logger.warning("Cache lookup failed (non-fatal): %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def write_cache(
    embedding: list[float],
    response: dict,
    question_type: str,
    conn: Any,
) -> None:
    """
    Store a query response in the cache with CACHE_TTL_DAYS expiry.

    Skips the insert if a near-duplicate entry already exists (prevents
    redundant writes when the same FAQ is asked rapidly).
    Any DB error is logged and silently swallowed — caching is non-critical.
    """
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    # Strip fields that shouldn't be replayed from cache
    cacheable = {k: v for k, v in response.items() if k != "latencyMs"}

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_cache
                    (question_embedding, response_json, question_type, expires_at)
                SELECT %s::vector, %s::jsonb, %s, NOW() + INTERVAL '%s days'
                WHERE NOT EXISTS (
                    SELECT 1 FROM query_cache
                    WHERE expires_at > NOW()
                      AND 1 - (question_embedding <=> %s::vector) >= %s
                )
                """,
                (
                    embedding_str,
                    json.dumps(cacheable),
                    question_type,
                    CACHE_TTL_DAYS,
                    embedding_str,
                    SIMILARITY_THRESHOLD,
                ),
            )
        conn.commit()
        logger.info(
            "Cache write: question_type=%s ttl=%d days threshold=%.2f",
            question_type, CACHE_TTL_DAYS, SIMILARITY_THRESHOLD,
        )
    except Exception as exc:
        logger.warning("Cache write failed (non-fatal): %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
