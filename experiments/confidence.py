"""Verbalized self-confidence, parsed the same way as the query synthesizer.

The status is one of ``ok``, ``omitted``, ``unparseable``, ``failed``,
``truncated``. Truncation is the generator hitting its output-token cap.
That row is not an observation and it is not given a fallback constant.
Fallback numbers are recorded on the response and are not observations:
the skip-unobserved reward leaves them out, and only the fallback replay
row substitutes them.
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


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Same fence-stripping JSON parse as synthesizer._parse_model_response."""
    text = (raw_text or "").strip()
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"answer": raw_text, "_unparsed": True}
    if not isinstance(parsed, dict):
        return {"answer": raw_text, "_unparsed": True}
    return parsed


def status_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    """Mirror synthesizer._confidence_from, and name the four statuses."""
    if parsed.get("_unparsed"):
        return {"value": UNPARSEABLE, "reported": False, "status": "unparseable"}
    raw = parsed.get("confidence")
    if isinstance(raw, bool) or raw is None:
        return {"value": OMITTED, "reported": False, "status": "omitted"}
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return {"value": OMITTED, "reported": False, "status": "omitted"}
    if not (0.0 <= value <= 1.0) or value != value:
        return {"value": OMITTED, "reported": False, "status": "omitted"}
    return {"value": value, "reported": True, "status": "ok"}


def parse_self_confidence(
    raw_text: str | None,
    *,
    call_failed: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    if call_failed:
        return {"value": FAILED, "reported": False, "status": "failed", "answer": ""}
    parsed = parse_model_json(raw_text or "")
    answer = parsed.get("answer")
    if not isinstance(answer, str):
        answer = raw_text or ""
    if truncated:
        # The cap cut the response. A confidence parsed from that text would
        # be a number the model did not finish saying.
        return {"value": None, "reported": False, "status": "truncated", "answer": answer}
    out = status_from_parsed(parsed)
    out["answer"] = answer
    return out
