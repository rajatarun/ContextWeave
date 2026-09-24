"""Ingest a health document from the health bucket into the health store.

Deliberately *not* the expertise preprocessor. That one classifies document
type, extracts expertise signals and writes entities to Memgraph and Neptune --
shared stores with no per-document delete. Running a medical record through it
would put rows in three places, one of which could never be taken back.

This does four things and nothing else: read, chunk, embed, write. No graph, no
routing analysis, no expertise signals.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any
from urllib.parse import unquote_plus

from .layout import shared_module

health_db = shared_module("health_db")

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Plain text only. A PDF or an image would need an extractor, and a half-read
# record that silently loses half its content is worse here than a refusal.
SUPPORTED_SUFFIXES = (".txt", ".md", ".markdown", ".json", ".csv")

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


def doc_id_for(key: str) -> str:
    """Stable per key, so re-uploading a corrected record replaces it."""
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def chunk(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than the window, or this never advances")
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def ingest_object(bucket: str, key: str, *, s3=None, embed=None) -> dict:
    if not key.lower().endswith(SUPPORTED_SUFFIXES):
        logger.warning("skipping %s: unsupported type for the health store", key)
        return {"ingested": 0, "skipped": key, "reason": "unsupported file type"}

    if s3 is None:
        import boto3
        s3 = boto3.client("s3")

    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "replace")
    pieces = chunk(body)
    if not pieces:
        return {"ingested": 0, "skipped": key, "reason": "empty document"}

    # Injected like `s3`, and imported lazily otherwise: it pulls in Bedrock
    # and the observability wrapper, and a *withdrawal* needs none of that.
    # Deleting a record must not depend on the embedding stack being healthy.
    if embed is None:
        embed = shared_module("embedder").embed_texts

    vectors = embed(pieces)
    rows = [(c, v) for c, v in zip(pieces, vectors) if v is not None]
    if not rows:
        # Every embedding failed. Writing nothing and reporting success would
        # leave a record the person believes is searchable and is not.
        raise RuntimeError(f"no embeddings returned for {key}; nothing was stored")

    doc_id = doc_id_for(key)
    health_db.init_schema()
    written = health_db.replace_document(doc_id, key, rows, {"chunks": len(rows)})
    # The key is logged, never the content: this log is the one place a medical
    # record could leak into CloudWatch, where it would outlive any delete.
    logger.info("health document ingested doc_id=%s chunks=%d", doc_id, written)
    return {"ingested": written, "docId": doc_id, "sourceKey": key}


def lambda_handler(event: dict, _context: Any = None) -> dict:
    """S3 events from the health bucket, created and removed alike."""
    detail = event.get("detail") or {}
    bucket = (detail.get("bucket") or {}).get("name", "")
    key = unquote_plus((detail.get("object") or {}).get("key", ""))
    if not bucket or not key:
        logger.warning("event carried no bucket/key")
        return {"ingested": 0}

    if event.get("detail-type") == "Object Deleted":
        # Removing the file removes the record. Without this the only way to
        # unpublish a health document would be to know its doc id and call the
        # API, and deleting the object would leave it searchable.
        removed = health_db.delete_document(doc_id_for(key))
        logger.info("health document withdrawn doc_id=%s chunks=%d", doc_id_for(key), removed)
        return {"deleted": removed}

    return ingest_object(bucket, key)
