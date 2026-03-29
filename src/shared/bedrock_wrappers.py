from __future__ import annotations

from typing import Any


def invoke_model_request(
    runtime_client: Any,
    *,
    model_id: str,
    body: str,
    content_type: str | None = None,
    accept: str | None = None,
) -> Any:
    """Raw Bedrock invoke_model provider call."""
    return runtime_client.invoke_model(
        modelId=model_id,
        contentType=content_type,
        accept=accept,
        body=body,
    )


def converse_request(
    runtime_client: Any,
    *,
    model_id: str,
    request_body: dict[str, Any],
) -> Any:
    """Raw Bedrock converse provider call."""
    return runtime_client.converse(modelId=model_id, **request_body)
