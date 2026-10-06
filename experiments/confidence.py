"""Verbalized self-confidence.

Two readings of the same reply are stored.

* Robust (the ``self`` signal). The first JSON object in the reply, after
  markdown fences are stripped. Prose before or after that object is ignored.
  A confidence in [0, 1] is an observation. A missing or unusable confidence,
  a reply with no JSON object, a failed call, and a truncated call are not
  observations: the stored robust value is null, not a fallback constant.
  The saved answer is the object's ``answer`` field when that field is a
  string. It is never the raw reply.

* Deployed strict (the ``self_with_fallbacks`` row). The whole reply, after
  the same fence strip, must be one JSON object. This is
  ``synthesizer._parse_model_response`` and ``_confidence_from``. Omitted,
  unparseable, and failed calls store 0.7, 0.5, and 0.0. Those constants are
  what the deployed synthesizer would have put on the response. They are not
  the robust signal.

Truncation (the generator hit ``maxTokens``) is neither reading. Both values
are null and both statuses are ``truncated``.
"""
from __future__ import annotations

import json
import re
from typing import Any

# These three constants are the deployed fallbacks. tests lock them to
# synthesizer._OMITTED_CONFIDENCE, synthesizer._UNPARSED_CONFIDENCE, and the
# failed-call value 0.0.
OMITTED = 0.7
UNPARSEABLE = 0.5
FAILED = 0.0

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$", re.MULTILINE)


def _strip_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    return text.strip()


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Whole-reply JSON parse, matching synthesizer._parse_model_response.

    Failure is marked ``_unparsed``. The raw reply is kept on this dict only
    so the strict status can be named. Callers that save an answer must use
    ``parse_self_confidence`` and must not copy this raw text.
    """
    text = _strip_fences(raw_text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"answer": raw_text, "_unparsed": True}
    if not isinstance(parsed, dict):
        return {"answer": raw_text, "_unparsed": True}
    return parsed


def first_json_object(raw_text: str) -> dict[str, Any] | None:
    """First JSON object in the reply. Prose on either side is skipped."""
    text = _strip_fences(raw_text)
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            return None
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = end if end > start else start + 1
    return None


def _answer_field(obj: dict[str, Any] | None) -> str:
    if not isinstance(obj, dict):
        return ""
    raw = obj.get("answer")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return str(raw)
    return ""


def _confidence_number(obj: dict[str, Any] | None) -> float | None:
    if not isinstance(obj, dict) or obj.get("_unparsed"):
        return None
    raw = obj.get("confidence")
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= value <= 1.0) or value != value:
        return None
    return value


def status_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    """Mirror synthesizer._confidence_from, and name the four statuses.

    The numeric value here is the deployed fallback when the model did not
    report a confidence. It is not, by itself, an observation.
    """
    if parsed.get("_unparsed"):
        return {"value": UNPARSEABLE, "reported": False, "status": "unparseable"}
    number = _confidence_number(parsed)
    if number is None:
        return {"value": OMITTED, "reported": False, "status": "omitted"}
    return {"value": number, "reported": True, "status": "ok"}


def parse_self_confidence(
    raw_text: str | None,
    *,
    call_failed: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    if call_failed:
        return {
            "value": None,
            "reported": False,
            "status": "failed",
            "answer": "",
            "deployed_value": FAILED,
            "deployed_reported": False,
            "deployed_status": "failed",
        }
    obj = first_json_object(raw_text or "")
    answer = _answer_field(obj)
    if truncated:
        return {
            "value": None,
            "reported": False,
            "status": "truncated",
            "answer": answer,
            "deployed_value": None,
            "deployed_reported": False,
            "deployed_status": "truncated",
        }
    number = _confidence_number(obj)
    if obj is None:
        robust = {"value": None, "reported": False, "status": "unparseable"}
    elif number is None:
        robust = {"value": None, "reported": False, "status": "omitted"}
    else:
        robust = {"value": number, "reported": True, "status": "ok"}
    deployed = status_from_parsed(parse_model_json(raw_text or ""))
    robust["answer"] = answer
    robust["deployed_value"] = deployed["value"]
    robust["deployed_reported"] = deployed["reported"]
    robust["deployed_status"] = deployed["status"]
    return robust
