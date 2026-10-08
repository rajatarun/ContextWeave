"""Verbalized self-confidence.

Two readings of the same reply are stored.

* Robust (the ``self`` signal). The first JSON object in the reply, after
  markdown fences are stripped. Prose before or after that object is ignored.
  A confidence in [0, 1] is an observation. A missing or unusable confidence,
  a reply with no JSON object, and a failed call are not observations: the
  stored robust value is null, not a fallback constant. The saved answer is
  the object's ``answer`` field when that field is a string. It is never the
  raw reply.

  A reply that starts with a complete JSON object and then runs into the
  output-token cap is still that object. Robust status is ``ok`` when the
  confidence is in [0, 1], and ``trailing_truncated`` is true. Only a JSON
  object that is itself cut off or broken is status ``truncated``, with a
  null value. A reply with no JSON whose first line is ``insufficient
  evidence`` is an abstention: the answer is that phrase, the robust value
  is null, and the status is ``omitted``.

* Deployed strict (the ``self_with_fallbacks`` row). The whole reply, after
  the same fence strip, must be one JSON object. This is
  ``synthesizer._parse_model_response`` and ``_confidence_from``. Omitted,
  unparseable, and failed calls store 0.7, 0.5, and 0.0. Those constants are
  what the deployed synthesizer would have put on the response. They are not
  the robust signal. A broken object cut off by the token cap is not given
  the unparseable constant: both readings are ``truncated`` and null.
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
# The generator's fixed abstain phrase, written as the first line and then
# explained in prose, with no JSON object around it.
_BARE_ABSTAIN_LINE = re.compile(r"(?i)^insufficient evidence\s*[.!]?\s*$")


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


def _claim_field(obj: dict[str, Any] | None) -> str | None:
    """The generator's ``claim`` string, or null when the object has none.

    An empty string is a claim the model sent empty. A missing field is null.
    Callers do not copy the answer into the claim.
    """
    if not isinstance(obj, dict):
        return None
    raw = obj.get("claim")
    if isinstance(raw, str):
        return raw.strip()
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


def bare_abstention_answer(raw_text: str) -> str | None:
    """Canonical abstain phrase when the reply leads with it and has no JSON.

    The first line is the phrase, optional trailing punctuation, and nothing
    else. Later lines may be prose. The returned answer is exactly
    ``insufficient evidence`` so abstention scoring sees the fixed phrase.
    """
    text = _strip_fences(raw_text or "")
    if not text:
        return None
    first, _, _rest = text.partition("\n")
    if not _BARE_ABSTAIN_LINE.match(first.strip()):
        return None
    return "insufficient evidence"


def _with_deployed(robust: dict[str, Any], raw_text: str, answer: str, trailing: bool) -> dict[str, Any]:
    deployed = status_from_parsed(parse_model_json(raw_text or ""))
    robust["answer"] = answer
    robust["trailing_truncated"] = trailing
    robust["deployed_value"] = deployed["value"]
    robust["deployed_reported"] = deployed["reported"]
    robust["deployed_status"] = deployed["status"]
    return robust


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
            "claim": None,
            "trailing_truncated": False,
            "deployed_value": FAILED,
            "deployed_reported": False,
            "deployed_status": "failed",
        }
    text = raw_text or ""
    obj = first_json_object(text)
    if obj is None:
        bare = bare_abstention_answer(text)
        if bare is not None:
            # The model abstained in prose and did not report a confidence.
            # That is the omitted case, including when later prose hit the cap.
            return _with_deployed(
                {"value": None, "reported": False, "status": "omitted", "claim": None},
                text, bare, False,
            )
        if truncated:
            return {
                "value": None,
                "reported": False,
                "status": "truncated",
                "answer": "",
                "claim": None,
                "trailing_truncated": False,
                "deployed_value": None,
                "deployed_reported": False,
                "deployed_status": "truncated",
            }
        return _with_deployed(
            {"value": None, "reported": False, "status": "unparseable", "claim": None},
            text, "", False,
        )
    answer = _answer_field(obj)
    number = _confidence_number(obj)
    claim = _claim_field(obj)
    if number is None:
        robust = {"value": None, "reported": False, "status": "omitted", "claim": claim}
    else:
        robust = {"value": number, "reported": True, "status": "ok", "claim": claim}
    # A complete object is an observation even when prose after it was cut off.
    return _with_deployed(robust, text, answer, bool(truncated))
