"""On-demand adjudication of token F1 inside the closed band.

The model is ``openai.gpt-oss-120b-1:0``. Temperature is 0, so the call has
no random draw. The run seed is still stored on the artifact and on every
row. Each row stores the system prompt, the user prompt, and the raw reply.
A reply that is not ``{"correct": true}`` or ``{"correct": false}`` is
``unparseable`` and is not turned into a 0 or a 1.

The spend cap is checked before the call, from the shared ledger. A call
under the minimum batch size is still one Converse request: this stage does
not submit a batch job.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from experiments.common import (
    ProtocolError, append_jsonl, cost_usd, done_keys, estimate_tokens, price_for,
)
from experiments.generate_stage import (
    FATAL_CODES, THROTTLE_CODES, _error_code, _error_message, converse_once,
)
from experiments.ledger import Budget

ADJUDICATOR_MODEL_ID = "openai.gpt-oss-120b-1:0"

ADJUDICATOR_SYSTEM = (
    "You decide whether a candidate answer matches a gold answer for a question. "
    'Reply with JSON only: {"correct": true} or {"correct": false}. '
    "true means the candidate expresses the same answer as the gold. "
    "false means it does not."
)


class AdjudicatorAccessDenied(ProtocolError):
    """The adjudicator model rejected the credentials or the model grant."""


def require_adjudicator(cfg: dict[str, Any]) -> str:
    model_id = cfg["adjudicator_model_id"]
    if model_id != ADJUDICATOR_MODEL_ID:
        raise ProtocolError(
            f"adjudicator_model_id is {model_id!r}. "
            f"This stage calls {ADJUDICATOR_MODEL_ID}."
        )
    temperature = cfg["adjudicator_temperature"]
    if temperature != 0 and temperature != 0.0:
        raise ProtocolError(
            f"adjudicator_temperature is {temperature!r}. "
            "Adjudication runs at temperature 0 so the call has no random draw."
        )
    price_for(cfg, model_id, "on_demand")
    return model_id


def user_prompt(question: str, answer: str, gold_answers: Sequence[str]) -> str:
    if gold_answers:
        gold = "\n".join(f"- {item}" for item in gold_answers)
    else:
        gold = "- (no gold answer; the question is unanswerable)"
    return f"Question: {question}\nGold answers:\n{gold}\nCandidate: {answer}\n"


def parse_adjudication(text: str) -> bool | None:
    """The boolean ``correct`` field of the first JSON object, or None."""
    start = (text or "").find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or type(obj.get("correct")) is not bool:
        return None
    return obj["correct"]


def _text_and_usage(resp: dict[str, Any]) -> tuple[str, int, int]:
    parts = []
    for block in resp.get("output", {}).get("message", {}).get("content", []):
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    usage = resp.get("usage") or {}
    in_tok, out_tok = usage.get("inputTokens"), usage.get("outputTokens")
    if in_tok is None or out_tok is None:
        raise ProtocolError("adjudicator response had no usage tokens. Refusing to invent a cost.")
    return "".join(parts), int(in_tok), int(out_tok)


def call_adjudicator(
    client: Any,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    *,
    sleep: Callable[[float], None],
    max_attempts: int = 6,
) -> tuple[str, int, int, str | None]:
    """Return raw text, input tokens, output tokens, and a failure reason.

    A failure reason of ``access_denied`` is raised as
    :class:`AdjudicatorAccessDenied` before a row is written.
    ``validation_exception`` returns empty usage. Any other failure stops.
    """
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            resp = converse_once(client, model_id, system, user, max_tokens, 0.0)
            raw, in_tok, out_tok = _text_and_usage(resp)
            return raw, in_tok, out_tok, None
        except ProtocolError:
            raise
        except Exception as exc:
            code = _error_code(exc)
            if code in FATAL_CODES or "AccessDenied" in code:
                raise AdjudicatorAccessDenied(
                    f"adjudicator model access not yet granted ({code}): {_error_message(exc)}"
                ) from exc
            if code in THROTTLE_CODES and attempt + 1 < max_attempts:
                sleep(delay)
                delay *= 2
                continue
            if code == "ValidationException":
                return _error_message(exc), 0, 0, "validation_exception"
            if code in THROTTLE_CODES:
                raise ProtocolError(
                    f"adjudicator still throttled after {max_attempts} attempts: {code}"
                ) from exc
            raise ProtocolError(f"unhandled adjudicator error ({code}): {_error_message(exc)}") from exc
    raise ProtocolError("adjudicator retries exhausted")


def adjudication_record(
    row: dict[str, Any],
    *,
    seed: int,
    model_id: str,
    system: str,
    user: str,
    raw: str,
    value: int | None,
    reason: str | None,
    in_tok: int,
    out_tok: int,
    usd: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    price = price_for(cfg, model_id, "on_demand")
    return {
        "qid": row["qid"],
        "dataset": row["dataset"],
        "arm": row["arm"],
        "question_type": row["question_type"],
        "f1": row["f1"],
        "value": value,
        "reason": reason,
        "seed": seed,
        "model_id": model_id,
        "temperature": 0,
        "system_prompt": system,
        "prompt": user,
        "raw_response": raw,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd": usd,
        "pricing": "on_demand",
        "price_usd_per_million": {"input": float(price["input"]), "output": float(price["output"])},
    }


def score_reply(raw: str, failure_reason: str | None) -> tuple[int | None, str | None]:
    if failure_reason:
        return None, failure_reason
    parsed = parse_adjudication(raw)
    if parsed is None:
        return None, "unparseable"
    return (1 if parsed else 0), None
