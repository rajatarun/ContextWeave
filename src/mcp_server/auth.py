"""Bearer-token check for POST /mcp and GET /mcp.

The token value is never in source or in the template. ``MCP_AUTH_TOKEN_SECRET_NAME``
is the name of either an AWS Secrets Manager secret or an SSM SecureString
parameter:

* a name starting with ``/``, or an ``ssm`` ARN, is read with ``GetParameter``
  (decrypted)
* any other name, or a Secrets Manager ARN, is read with ``GetSecretValue``

A JSON secret may hold the token under ``token``, ``bearerToken``,
``bearer_token``, or ``value``. Anything else is the raw string.

Comparison is constant-time on SHA-256 digests so a length mismatch is not a
fast path and ``hmac.compare_digest`` does not raise. A missing name, a failed
lookup, a missing header, or a mismatch all fail closed the same way.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

SECRET_NAME_ENV = "MCP_AUTH_TOKEN_SECRET_NAME"
_CACHE_TTL_SECONDS = 60
_MAX_PRESENTED_LEN = 4096
_JSON_KEYS = ("token", "bearerToken", "bearer_token", "value")

_cache: tuple[str, str, float] | None = None


def clear_cache() -> None:
    global _cache
    _cache = None


def tokens_equal(presented: str, expected: str) -> bool:
    """Constant-time equality for two strings that may differ in length."""
    presented_digest = hashlib.sha256(presented.encode("utf-8")).digest()
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(presented_digest, expected_digest)


def extract_bearer(headers: dict | None) -> str | None:
    """Return the bearer token, or None when the header is absent or not Bearer."""
    if not headers:
        return None
    raw = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "authorization" and isinstance(value, str):
            raw = value
            break
    if raw is None:
        return None
    scheme, separator, rest = raw.strip().partition(" ")
    if separator != " " or scheme.lower() != "bearer":
        return None
    token = rest.strip()
    if not token or len(token) > _MAX_PRESENTED_LEN:
        return None
    return token


def credentials_ok(presented: str | None, expected: str | None) -> bool:
    """True only when both sides are non-empty and equal.

    The digest compare always runs, including when one side is missing, so the
    failure path does not return before the compare.
    """
    presented_s = presented if isinstance(presented, str) else ""
    expected_s = expected if isinstance(expected, str) else ""
    match = tokens_equal(presented_s, expected_s)
    if not presented_s or not expected_s:
        return False
    return match


def token_from_secret_string(raw: str | None) -> str | None:
    """Accept a raw token or a small JSON object that carries one."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text[0] != "{":
        return text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, str) and data.strip():
        return data.strip()
    if isinstance(data, dict):
        for key in _JSON_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _uses_ssm(name: str) -> bool:
    lowered = name.lower()
    return name.startswith("/") or ":ssm:" in lowered


def _client(service: str, injected: Any) -> Any:
    if injected is not None:
        return injected
    import boto3
    return boto3.client(service)


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str((response.get("Error") or {}).get("Code") or "")
    return ""


def _read_ssm(name: str, client: Any) -> str | None:
    response = client.get_parameter(Name=name, WithDecryption=True)
    parameter = response.get("Parameter") or {}
    return token_from_secret_string(parameter.get("Value"))


def _read_secrets_manager(name: str, client: Any) -> str | None:
    response = client.get_secret_value(SecretId=name)
    if response.get("SecretString") is not None:
        return token_from_secret_string(response.get("SecretString"))
    binary = response.get("SecretBinary")
    if isinstance(binary, (bytes, bytearray)):
        return token_from_secret_string(binary.decode("utf-8"))
    return None


def read_configured_token(*, secrets_client: Any = None, ssm_client: Any = None, now: float | None = None) -> str | None:
    """Load the configured token, or None when it cannot be read.

    None fails closed. Lookup errors are logged by type and code only — the
    token itself is never written to the log.
    """
    global _cache
    name = (os.environ.get(SECRET_NAME_ENV) or "").strip()
    if not name:
        return None

    current = now if now is not None else time.monotonic()
    if _cache is not None and _cache[0] == name and _cache[2] > current:
        return _cache[1]

    try:
        if _uses_ssm(name):
            token = _read_ssm(name, _client("ssm", ssm_client))
        else:
            token = _read_secrets_manager(name, _client("secretsmanager", secrets_client))
    except Exception as exc:
        logger.warning(
            "mcp auth token lookup failed type=%s code=%s",
            type(exc).__name__,
            _error_code(exc),
        )
        return None

    if token:
        _cache = (name, token, current + _CACHE_TTL_SECONDS)
    return token
