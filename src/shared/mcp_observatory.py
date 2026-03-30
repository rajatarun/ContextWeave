from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

import boto3
from mcp_observatory.instrument import instrument_wrapper_api

from .bedrock_wrappers import converse_request, invoke_model_request

log = logging.getLogger(__name__)
_wrapper = instrument_wrapper_api("contextweave-bedrock")
_ddb_table = None
_TTL_SECONDS = 90 * 24 * 60 * 60


def _get_ddb_table():
    global _ddb_table
    table_name = os.environ.get("OBSERVATORY_METRICS_TABLE")
    if not table_name:
        return None
    if _ddb_table is None:
        _ddb_table = boto3.resource("dynamodb").Table(table_name)
    return _ddb_table


def _to_decimal(value: float | int | str | None) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(round(float(value), 8)))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _push_metric(operation: str, span: Any, decision: Any, extra: dict[str, Any]) -> None:
    table = _get_ddb_table()
    if table is None:
        return

    try:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        expiry = int(time.time()) + _TTL_SECONDS

        item = {
            "pk": f"OBSERVATORY#{operation}",
            "sk": f"{now_iso}#{span.trace_id}",
            "trace_id": span.trace_id,
            "operation": operation,
            "timestamp": now_iso,
            "prompt_tokens": Decimal(int(getattr(span, "prompt_tokens", 0) or 0)),
            "completion_tokens": Decimal(int(getattr(span, "completion_tokens", 0) or 0)),
            "cost_usd": _to_decimal(getattr(span, "cost_usd", 0.0)),
            "decision": getattr(decision, "action", "unknown"),
            "decision_reason": getattr(decision, "reason", None) or "none",
            # Hallucination metrics
            "hallucination_risk_score": _to_decimal(getattr(span, "hallucination_risk_score", None)),
            "hallucination_risk_level": str(getattr(span, "hallucination_risk_level", None) or "unknown"),
            # Composite risk metrics
            "composite_risk_score": _to_decimal(getattr(span, "composite_risk_score", None)),
            "composite_risk_level": str(getattr(span, "composite_risk_level", None) or "unknown"),
            # Policy gate
            "policy_gate": str(getattr(span, "policy_decision", None) or "unknown"),
            "ttl": Decimal(expiry),
        }

        item.update({k: str(v) if isinstance(v, float) else v for k, v in extra.items()})
        table.put_item(Item=item)
    except Exception as exc:
        log.warning("observatory_metric_write_failed", extra={"err": str(exc)})


def _run_observed_call(
    *,
    operation: str,
    source: str,
    model: str,
    prompt: str,
    input_payload: dict[str, Any],
    call: Callable[[], Any],
    log_extra: dict[str, Any],
    metric_extra: dict[str, Any],
) -> Any:
    result = asyncio.run(
        _wrapper.invoke(
            source=source,
            model=model,
            prompt=prompt,
            input_payload=input_payload,
            call=call,
        )
    )

    log.info(
        "mcp_observatory",
        extra={
            "operation": operation,
            **log_extra,
            "trace_id": result.span.trace_id,
            "prompt_tokens": result.span.prompt_tokens,
            "completion_tokens": result.span.completion_tokens,
            "cost_usd": result.span.cost_usd,
            "decision": result.decision.action,
            "decision_reason": result.decision.reason,
            "hallucination_risk_score": getattr(result.span, "hallucination_risk_score", None),
            "hallucination_risk_level": getattr(result.span, "hallucination_risk_level", None),
            "composite_risk_score": getattr(result.span, "composite_risk_score", None),
            "composite_risk_level": getattr(result.span, "composite_risk_level", None),
            "policy_gate": getattr(result.span, "policy_decision", None),
        },
    )

    _push_metric(operation, result.span, result.decision, metric_extra)
    return result.output


def observe_model_request(
    *,
    runtime_client: Any,
    model_id: str,
    body: str,
    content_type: str | None = None,
    accept: str | None = None,
) -> Any:
    return _run_observed_call(
        operation="invoke_model",
        source="model",
        model=model_id,
        prompt=body,
        input_payload={
            "model_id": model_id,
            "content_type": content_type,
            "accept": accept,
        },
        call=lambda: invoke_model_request(
            runtime_client,
            model_id=model_id,
            body=body,
            content_type=content_type,
            accept=accept,
        ),
        log_extra={
            "model_id": model_id,
            "body_len": len(body),
        },
        metric_extra={
            "model_id": model_id,
            "body_len": Decimal(len(body)),
        },
    )


def observe_converse_request(
    *,
    runtime_client: Any,
    model_id: str,
    prompt: str,
    request_body: dict[str, Any],
    source: str,
    operation: str,
) -> Any:
    return _run_observed_call(
        operation="invoke_model",
        source=source,
        model=model_id,
        prompt=prompt,
        input_payload={
            "model_id": model_id,
            "request_keys": sorted(request_body.keys()),
        },
        call=lambda: converse_request(
            runtime_client,
            model_id=model_id,
            request_body=request_body,
        ),
        log_extra={
            "model_id": model_id,
            "body_len": len(prompt),
            "pipeline_operation": operation,
        },
        metric_extra={
            "model_id": model_id,
            "body_len": Decimal(len(prompt)),
            "pipeline_operation": operation,
        },
    )
