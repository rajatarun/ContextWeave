"""
Bedrock Titan Text Embeddings V2 wrapper for ExpertiseRAG.

Generates 1024-dimensional semantic embeddings used for:
  - Storing chunk embeddings in pgvector during preprocessing
  - Embedding query questions during retrieval

The Bedrock client is cached at module level for Lambda warm-start reuse.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_BEDROCK_RUNTIME: Any = None
_MODEL_ID = "amazon.titan-embed-text-v2:0"
_DIMENSIONS = 1024


def _get_bedrock_client() -> Any:
    global _BEDROCK_RUNTIME
    if _BEDROCK_RUNTIME is None:
        import boto3
        _BEDROCK_RUNTIME = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _BEDROCK_RUNTIME


def embed_text(text: str) -> list[float] | None:
    """
    Generate a 1024-dim Titan Text Embeddings V2 vector for a single text.

    Returns None if embedding fails (caller should handle gracefully).
    Text is truncated to 8192 characters to stay within model limits.
    """
    if not text or not text.strip():
        return None

    # Titan V2 context window is ~8192 tokens; truncate conservatively
    truncated = text[:8000]

    try:
        client = _get_bedrock_client()
        body = json.dumps({
            "inputText": truncated,
            "dimensions": _DIMENSIONS,
            "normalize": True,
        })
        response = client.invoke_model(
            modelId=_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as exc:
        logger.warning("Embedding failed (non-fatal): %s", exc)
        return None


def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """
    Generate embeddings for a list of texts.

    Calls embed_text() for each item individually (Titan V2 has no batch API).
    Returns a list of the same length; failed embeddings are None.
    """
    return [embed_text(t) for t in texts]
