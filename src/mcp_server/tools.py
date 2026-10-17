"""MCP tools that wrap the existing HTTP handlers.

Each tool builds the same API Gateway event the route already receives and
calls that route's Lambda handler. Validation, status codes, and response
bodies stay in those handlers.

Health-record tools (``health_query``, ``list_health_documents``,
``delete_health_document``) call ``health_api.handler``, which is the function
behind the API's custom authorizer on ``/health/*``. They are advertised only
when that surface is deployed: the template sets ``HEALTH_POSTGRES_SECRET_ARN``
only under ``HealthEnabled``, the same condition that creates those routes and
requires the authorizer. The ``/health/*`` routes themselves are unchanged and
still carry the authorizer. ``GET /health`` is the expertise liveness check,
not the health-record store.
"""
from __future__ import annotations

import importlib
import json
import os
from typing import Any, Callable

PROTOCOL_VERSION = "2025-06-18"

HEALTH_TOOL_NAMES = frozenset({
    "health_query",
    "list_health_documents",
    "delete_health_document",
})


class InvalidParams(Exception):
    """JSON-RPC -32602. The message is safe to return to the caller."""


def health_store_enabled() -> bool:
    return bool((os.environ.get("HEALTH_POSTGRES_SECRET_ARN") or "").strip())


def _query_lambda() -> Callable:
    return importlib.import_module("query_api.handler").lambda_handler


def _health_lambda() -> Callable:
    return importlib.import_module("health_api.handler").lambda_handler


def _child_event(
    parent: dict,
    *,
    method: str,
    path: str,
    body: dict | None = None,
    query: dict | None = None,
) -> dict:
    parent_ctx = dict(parent.get("requestContext") or {})
    # Do not forward the MCP bearer token, or any authorizer context that
    # belongs to a different route, into the downstream handler.
    parent_ctx.pop("authorizer", None)
    http = dict(parent_ctx.get("http") or {})
    http["method"] = method
    http["path"] = path
    parent_ctx["http"] = http
    event: dict[str, Any] = {
        "requestContext": parent_ctx,
        "rawPath": path,
        "headers": {},
        "isBase64Encoded": False,
    }
    if body is not None:
        event["body"] = json.dumps(body)
    if query:
        event["queryStringParameters"] = {
            key: str(value) for key, value in query.items() if value is not None and str(value) != ""
        }
    return event


def _pick(arguments: dict, keys: tuple[str, ...]) -> dict:
    return {key: arguments[key] for key in keys if key in arguments and arguments[key] is not None}


def _call_query_expertise(arguments: dict, event: dict) -> dict:
    body = _pick(arguments, ("question", "topK", "includeGraphExpansion", "minConfidence", "questionType"))
    return _query_lambda()(_child_event(event, method="POST", path="/query-expertise", body=body), None)


def _call_submit_feedback(arguments: dict, event: dict) -> dict:
    body = _pick(arguments, ("queryId", "rating"))
    return _query_lambda()(_child_event(event, method="POST", path="/feedback", body=body), None)


def _call_list_routing_decisions(arguments: dict, event: dict) -> dict:
    query = _pick(arguments, ("mode", "questionType", "strategy", "since", "limit", "offset"))
    return _query_lambda()(
        _child_event(event, method="GET", path="/routing-decisions", query=query),
        None,
    )


def _call_health(arguments: dict, event: dict) -> dict:
    return _query_lambda()(_child_event(event, method="GET", path="/health"), None)


def _call_get_agent_card(arguments: dict, event: dict) -> dict:
    return _query_lambda()(
        _child_event(event, method="GET", path="/.well-known/agent-card.json"),
        None,
    )


def _call_health_query(arguments: dict, event: dict) -> dict:
    body = _pick(arguments, ("question", "topK"))
    return _health_lambda()(_child_event(event, method="POST", path="/health/query", body=body), None)


def _call_list_health_documents(arguments: dict, event: dict) -> dict:
    return _health_lambda()(_child_event(event, method="GET", path="/health/documents"), None)


def _call_delete_health_document(arguments: dict, event: dict) -> dict:
    doc_id = arguments.get("docId")
    if not isinstance(doc_id, str) or not doc_id.strip() or "/" in doc_id:
        raise InvalidParams("docId is required and must be a single path segment")
    path = "/health/documents/" + doc_id.strip()
    return _health_lambda()(_child_event(event, method="DELETE", path=path), None)


def _object_schema(properties: dict, required: list[str] | None = None) -> dict:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


_READ = {"readOnlyHint": True, "destructiveHint": False}
_WRITE = {"readOnlyHint": False, "destructiveHint": False}
_DELETE = {"readOnlyHint": False, "destructiveHint": True, "title": "Delete health document"}

_EMPTY = _object_schema({})

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "query_expertise",
        "health": False,
        "description": (
            "Answer a question about this developer's professional expertise from the "
            "GraphRAG corpus. Wraps POST /query-expertise. Returns the answer, sources, "
            "inferred skills, confidence, question type, and the routing decision. "
            "Optional fields match that endpoint: topK (1-50), includeGraphExpansion, "
            "minConfidence, and questionType."
        ),
        "inputSchema": _object_schema(
            {
                "question": {
                    "type": "string",
                    "description": "The expertise question to answer. Required, at most 2000 characters.",
                },
                "topK": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of chunks to retrieve. Defaults to 10.",
                },
                "includeGraphExpansion": {
                    "type": "boolean",
                    "description": "Expand the knowledge graph around the retrieved chunks.",
                },
                "minConfidence": {
                    "type": "number",
                    "description": "Minimum retrieval score. Defaults to 0.3.",
                },
                "questionType": {
                    "type": "string",
                    "description": "Skip classification and force this question type.",
                },
            },
            ["question"],
        ),
        "annotations": dict(_WRITE),
        "call": _call_query_expertise,
    },
    {
        "name": "submit_feedback",
        "health": False,
        "description": (
            "Rate an answer so the router can learn from it. Wraps POST /feedback. "
            "rating is up, down, neutral, true, false, or a number from 0 to 1. "
            "queryId comes from a previous query_expertise result."
        ),
        "inputSchema": _object_schema(
            {
                "queryId": {"type": "string", "description": "queryId of the answer to rate."},
                "rating": {
                    "description": "up, down, neutral, true, false, or a number in [0, 1].",
                    "oneOf": [
                        {"type": "string", "enum": ["up", "down", "neutral"]},
                        {"type": "boolean"},
                        {"type": "number", "minimum": 0, "maximum": 1},
                    ],
                },
            },
            ["queryId", "rating"],
        ),
        "annotations": dict(_WRITE),
        "call": _call_submit_feedback,
    },
    {
        "name": "list_routing_decisions",
        "health": False,
        "description": (
            "Read the routing decision log. Wraps GET /routing-decisions. "
            "mode=list returns rows, newest first. mode=summary groups by question "
            "type and strategy and reports meanAbsDiff. Read-only; the log has no "
            "question text and no answer text."
        ),
        "inputSchema": _object_schema(
            {
                "mode": {
                    "type": "string",
                    "enum": ["list", "summary"],
                    "description": "list (default) or summary.",
                },
                "questionType": {"type": "string", "description": "Exact question-type filter."},
                "strategy": {"type": "string", "description": "Exact strategy filter."},
                "since": {"type": "string", "description": "ISO 8601 lower bound on created_at."},
                "limit": {"type": "integer", "description": "Page size for list mode (1-1000)."},
                "offset": {"type": "integer", "description": "Page offset for list mode."},
            },
        ),
        "annotations": dict(_READ),
        "call": _call_list_routing_decisions,
    },
    {
        "name": "health",
        "health": False,
        "description": (
            "Liveness plus routing-graph learning-loop health. Wraps GET /health. "
            "This is not the health-record store; record tools are health_query, "
            "list_health_documents, and delete_health_document."
        ),
        "inputSchema": _EMPTY,
        "annotations": dict(_READ),
        "call": _call_health,
    },
    {
        "name": "health_query",
        "health": True,
        "description": (
            "Ask a question of the uploaded health records. Wraps POST /health/query, "
            "which stays behind the API's custom authorizer for direct HTTP calls. "
            "An empty retrieval is not answered from general knowledge, nothing is "
            "cached, and database errors are not echoed."
        ),
        "inputSchema": _object_schema(
            {
                "question": {"type": "string", "description": "Question to answer from the records."},
                "topK": {"type": "integer", "minimum": 1, "description": "Number of excerpts. Defaults to 6."},
            },
            ["question"],
        ),
        "annotations": dict(_READ),
        "call": _call_health_query,
    },
    {
        "name": "list_health_documents",
        "health": True,
        "description": (
            "List health-record documents without their content. Wraps "
            "GET /health/documents, which stays behind the API's custom authorizer "
            "for direct HTTP calls."
        ),
        "inputSchema": _EMPTY,
        "annotations": dict(_READ),
        "call": _call_list_health_documents,
    },
    {
        "name": "delete_health_document",
        "health": True,
        "description": (
            "Delete one health-record document and report how many rows were removed. "
            "Wraps DELETE /health/documents/{docId}, which stays behind the API's "
            "custom authorizer for direct HTTP calls. A document that was not in the "
            "store is reported as not found rather than as deleted."
        ),
        "inputSchema": _object_schema(
            {
                "docId": {"type": "string", "description": "Document id to delete. A single path segment."},
            },
            ["docId"],
        ),
        "annotations": dict(_DELETE),
        "call": _call_delete_health_document,
    },
    {
        "name": "get_agent_card",
        "health": False,
        "description": (
            "Return the agent card published at GET /.well-known/agent-card.json, "
            "including the interface URL and the skills this deployment serves."
        ),
        "inputSchema": _EMPTY,
        "annotations": dict(_READ),
        "call": _call_get_agent_card,
    },
]

_BY_NAME = {spec["name"]: spec for spec in TOOL_SPECS}


def list_tools() -> list[dict]:
    enabled = health_store_enabled()
    listed = []
    for spec in TOOL_SPECS:
        if spec["health"] and not enabled:
            continue
        listed.append({
            "name": spec["name"],
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
            "annotations": spec["annotations"],
        })
    return listed


def http_to_tool_result(response: dict) -> dict:
    status = int(response.get("statusCode") or 500)
    raw = response.get("body") or ""
    if response.get("isBase64Encoded") and isinstance(raw, str):
        import base64
        raw = base64.b64decode(raw).decode("utf-8")
    text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "isError": status >= 400,
    }
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        result["structuredContent"] = parsed
    return result


def call_tool(name: str, arguments: Any, event: dict) -> dict:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise InvalidParams("arguments must be an object")
    spec = _BY_NAME.get(name)
    if spec is None or (spec["health"] and not health_store_enabled()):
        raise InvalidParams(f"unknown tool: {name}")
    response = spec["call"](arguments, event)
    if not isinstance(response, dict) or "statusCode" not in response:
        raise RuntimeError("tool handler returned an unexpected response")
    return http_to_tool_result(response)
