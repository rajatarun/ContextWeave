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

from botocore.exceptions import EndpointConnectionError

logger = logging.getLogger(__name__)

_BEDROCK_RUNTIME: Any = None
_MODEL_ID = "amazon.titan-embed-text-v2:0"
_DIMENSIONS = 1024
_CUSTOM_BEDROCK_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
_USING_CUSTOM_ENDPOINT = bool(_CUSTOM_BEDROCK_ENDPOINT)


def _get_bedrock_client(*, force_default_endpoint: bool = False) -> Any:
    global _BEDROCK_RUNTIME
    global _USING_CUSTOM_ENDPOINT
    use_custom_endpoint = bool(_CUSTOM_BEDROCK_ENDPOINT) and not force_default_endpoint
    if _BEDROCK_RUNTIME is None or _USING_CUSTOM_ENDPOINT != use_custom_endpoint:
        import boto3
        client_kwargs: dict[str, Any] = {
            "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        }
        if use_custom_endpoint:
            client_kwargs["endpoint_url"] = _CUSTOM_BEDROCK_ENDPOINT
        _BEDROCK_RUNTIME = boto3.client(
            "bedrock-runtime",
            **client_kwargs,
        )
        _USING_CUSTOM_ENDPOINT = use_custom_endpoint
    return _BEDROCK_RUNTIME


def _invoke_model(client: Any, body: str) -> list[float]:
    response = client.invoke_model(
        modelId=_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


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
        body = json.dumps({
            "inputText": truncated,
            "dimensions": _DIMENSIONS,
            "normalize": True,
        })
        client = _get_bedrock_client()
        try:
            return _invoke_model(client, body)
        except EndpointConnectionError as exc:
            if _USING_CUSTOM_ENDPOINT:
                logger.warning(
                    "Embedding failed via AWS_ENDPOINT_URL_BEDROCK_RUNTIME=%s (%s). "
                    "Retrying Bedrock call with default endpoint.",
                    _CUSTOM_BEDROCK_ENDPOINT,
                    exc,
                )
                fallback_client = _get_bedrock_client(force_default_endpoint=True)
                return _invoke_model(fallback_client, body)
            raise
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
