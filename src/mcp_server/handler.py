"""Streamable HTTP MCP endpoint (protocol 2025-06-18).

POST /mcp
    One JSON-RPC message per request: initialize, ping, tools/list, tools/call.
    Notifications, including notifications/initialized, get HTTP 202 and an
    empty body. JSON-RPC batches are rejected.

GET /mcp
    405 Method Not Allowed. This server does not open a server-initiated SSE
    stream; that is one of the two responses the transport allows.

Every method requires ``Authorization: Bearer <token>``. The token is compared
with the value named by ``MCP_AUTH_TOKEN_SECRET_NAME``. Failure is HTTP 401,
and the body does not say whether the header was missing, malformed, or wrong.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from . import auth, tools

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

PROTOCOL_VERSION = tools.PROTOCOL_VERSION
SUPPORTED_PROTOCOL_VERSIONS = frozenset({PROTOCOL_VERSION, "2025-03-26"})

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,Accept,MCP-Protocol-Version,Mcp-Session-Id",
    "Access-Control-Allow-Methods": "POST,GET,OPTIONS",
}


def _headers(**extra: str) -> dict[str, str]:
    base = {
        "Content-Type": "application/json",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
        **_CORS_HEADERS,
    }
    base.update(extra)
    return base


def _http(status: int, body: Any, **header_overrides: str) -> dict:
    payload = "" if body is None else json.dumps(body)
    headers = _headers(**header_overrides)
    if body is None:
        headers.pop("Content-Type", None)
    return {"statusCode": status, "headers": headers, "body": payload}


def _rpc_result(req_id: Any, result: dict) -> dict:
    return _http(200, {"jsonrpc": "2.0", "id": req_id, "result": result})


def _rpc_error(req_id: Any, code: int, message: str, *, status: int = 200) -> dict:
    return _http(status, {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    })


def _unauthorized() -> dict:
    # One body for every failure. Missing, malformed, and wrong tokens look the same.
    return _http(
        401,
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32001, "message": "Unauthorized"},
        },
        **{"WWW-Authenticate": 'Bearer realm="contextweave"'},
    )


def _header(event: dict, name: str) -> str:
    headers = event.get("headers") or {}
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == name.lower() and isinstance(value, str):
            return value
    return ""


def _parse_message(event: dict) -> tuple[dict | None, dict | None]:
    raw = event.get("body") or ""
    if event.get("isBase64Encoded") and isinstance(raw, str) and raw:
        import base64
        try:
            raw = base64.b64decode(raw).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None, _rpc_error(None, -32700, "Parse error", status=400)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if not str(raw).strip():
        return None, _rpc_error(None, -32700, "Parse error", status=400)
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        return None, _rpc_error(None, -32700, "Parse error", status=400)
    if isinstance(message, list):
        return None, _rpc_error(None, -32600, "JSON-RPC batches are not supported", status=400)
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        req_id = message.get("id") if isinstance(message, dict) else None
        return None, _rpc_error(req_id, -32600, "Invalid Request", status=400)
    return message, None


def _check_protocol_header(event: dict, req_id: Any) -> dict | None:
    version = _header(event, "MCP-Protocol-Version").strip()
    if version and version not in SUPPORTED_PROTOCOL_VERSIONS:
        return _rpc_error(
            req_id,
            -32600,
            f"Unsupported MCP-Protocol-Version. Supported: {PROTOCOL_VERSION}",
            status=400,
        )
    return None


def _initialize(req_id: Any, params: dict) -> dict:
    requested = str((params or {}).get("protocolVersion") or PROTOCOL_VERSION)
    # Reply with the version this server speaks. A client that cannot use it disconnects.
    _ = requested
    return _rpc_result(req_id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {
            "name": "ContextWeave",
            "version": os.environ.get("MCP_SERVER_VERSION", "1.0.0"),
        },
        "instructions": (
            "Tools wrap this deployment's HTTP API. query_expertise answers questions "
            "about professional expertise. submit_feedback rates an answer by queryId. "
            "list_routing_decisions reads the routing log. health is the liveness check. "
            "get_agent_card returns the agent card. Health-record tools are present only "
            "when the health store is enabled, and direct HTTP calls to /health/* still "
            "use that API's custom authorizer."
        ),
    })


def _tools_list(req_id: Any, params: dict) -> dict:
    if (params or {}).get("cursor"):
        return _rpc_error(req_id, -32602, "unknown cursor")
    return _rpc_result(req_id, {"tools": tools.list_tools()})


def _tools_call(req_id: Any, params: dict, event: dict) -> dict:
    params = params or {}
    name = params.get("name")
    if not isinstance(name, str) or not name:
        return _rpc_error(req_id, -32602, "name is required")
    try:
        result = tools.call_tool(name, params.get("arguments"), event)
    except tools.InvalidParams as exc:
        return _rpc_error(req_id, -32602, str(exc))
    except Exception:
        # No exception text: a database error from the health store can carry a row.
        logger.exception("mcp tool call failed name=%s", name)
        result = {
            "content": [{"type": "text", "text": "The tool call could not be completed."}],
            "isError": True,
        }
    return _rpc_result(req_id, result)


def _dispatch(message: dict, event: dict) -> dict:
    method = message["method"]
    req_id = message.get("id")
    params = message.get("params") or {}
    if not isinstance(params, dict):
        if "id" not in message:
            return _http(202, None)
        return _rpc_error(req_id, -32602, "params must be an object")

    if "id" not in message:
        # Notification. Nothing on this server needs one, including initialized.
        logger.info("mcp notification=%s", method)
        return _http(202, None)

    logger.info("mcp method=%s", method)
    rejected = _check_protocol_header(event, req_id)
    if rejected is not None:
        return rejected

    if method == "initialize":
        return _initialize(req_id, params)
    if method == "ping":
        return _rpc_result(req_id, {})
    if method == "tools/list":
        return _tools_list(req_id, params)
    if method == "tools/call":
        logger.info("mcp tools/call name=%s", params.get("name"))
        return _tools_call(req_id, params, event)
    return _rpc_error(req_id, -32601, f"Method not found: {method}")


def lambda_handler(event: dict, _context: Any = None) -> dict:
    method = ((event.get("requestContext") or {}).get("http") or {}).get("method", "")
    method = str(method or "GET").upper()
    presented = auth.extract_bearer(event.get("headers"))
    expected = auth.read_configured_token()
    if not auth.credentials_ok(presented, expected):
        return _unauthorized()

    if method == "GET":
        return _http(
            405,
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32601,
                    "message": "SSE streams are not offered on GET /mcp. Send JSON-RPC with POST.",
                },
            },
            **{"Allow": "POST"},
        )
    if method != "POST":
        return _http(
            405,
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32601, "message": "Method not allowed"},
            },
            **{"Allow": "POST"},
        )

    message, error = _parse_message(event)
    if error is not None:
        return error
    assert message is not None
    return _dispatch(message, event)
