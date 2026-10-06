"""Generate an answer for every (question, arm) with Bedrock Converse.

Self-confidence is the generator's verbalized number. The robust reading
(first JSON object) is the ``self`` signal. The strict whole-reply reading
is ``synthesizer._confidence_from`` and is stored beside it for the
fallback row. Status is ``ok``, ``omitted``, ``unparseable``, ``failed``,
or ``truncated``. A complete object whose trailing prose hit the output-token
cap stays ``ok`` and sets ``trailing_truncated``. A JSON object cut off
mid-token is ``truncated`` and is not stored as a fallback confidence. A
bare ``insufficient evidence`` line with no JSON is an abstention with
status ``omitted`` and a null confidence. The saved answer is the JSON
``answer`` field, or that phrase, never the raw reply.

``--dry-run`` builds every prompt from the retrieval file and reports an
input-token estimate plus an output-token upper bound (max output tokens on
every call). It does not call the model.

A missing price, a missing credential on a real run, or ``AccessDenied`` stops
the process. ``ValidationException`` is recorded on that row. Five identical
validation failures in a row abort the run so a bad request shape is not
written out as eighteen thousand per-row errors.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from experiments.common import (
    ProtocolError, append_jsonl, artifact_meta, cost_usd, done_keys,
    estimate_tokens, price_for, read_jsonl, validate_rows, write_json,
    RETRIEVAL_FIELDS,
)
from experiments.confidence import parse_self_confidence
from experiments.ledger import SPEND_CAP_REASON, Budget

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "generator_system.txt"
THROTTLE_CODES = {
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "ModelTimeoutException", "Throttling",
}
FATAL_CODES = {
    "AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException",
    "InvalidSignatureException", "NoCredentialsError",
}


def system_prompt() -> str:
    if not PROMPT_PATH.is_file():
        raise ProtocolError(f"generator prompt missing: {PROMPT_PATH}")
    return PROMPT_PATH.read_text()


def build_user_message(question: str, passages: list[dict[str, Any]]) -> str:
    blocks = []
    for i, passage in enumerate(passages, start=1):
        title = passage.get("title") or ""
        header = f"[{i}]" + (f" {title}" if title else "")
        blocks.append(f"{header}\n{passage['text'].strip()}")
    evidence = "\n\n".join(blocks) if blocks else "(no passages)"
    return f"PASSAGES:\n{evidence}\n\nQUESTION:\n{question}"


def prompt_token_estimate(system: str, user: str) -> int:
    # Converse sends system and user as separate blocks. Estimate each, then sum.
    return estimate_tokens(system) + estimate_tokens(user)


class SpendCap(Exception):
    pass


def dry_run(cfg: dict[str, Any], retrieval_rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    price_for(cfg, model_id)
    system = system_prompt()
    n = 0
    input_tokens = 0
    for row in retrieval_rows:
        user = build_user_message(row["question"], row["passages"])
        input_tokens += prompt_token_estimate(system, user)
        n += 1
    max_out = int(cfg["generator_max_output_tokens"])
    output_tokens = n * max_out
    usd = cost_usd(cfg, model_id, input_tokens, output_tokens)
    return {
        "model_id": model_id,
        "region": cfg["region"],
        "temperature": cfg["temperature"],
        "n_calls": n,
        "input_tokens_estimate": input_tokens,
        "output_tokens_upper_bound": output_tokens,
        "generator_max_output_tokens": max_out,
        "usd_upper_bound": usd,
        "estimator": "ceil(utf-8 bytes / 4) summed over the system prompt and the user message",
        "output_policy": "upper bound charges generator_max_output_tokens on every call",
        "called_model": False,
    }


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", type(exc).__name__))
    return type(exc).__name__


def _error_message(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        msg = str(response.get("Error", {}).get("Message", exc))
    else:
        msg = str(exc)
    if len(msg) > 2000:
        return msg[:2000] + "…[truncated]"
    return msg


def converse_once(client: Any, model_id: str, system: str, user: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    return client.converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )


def _text_and_usage(resp: dict[str, Any]) -> tuple[str, int | None, int | None, str | None]:
    parts = []
    for block in resp.get("output", {}).get("message", {}).get("content", []):
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    usage = resp.get("usage") or {}
    in_tok = usage.get("inputTokens")
    out_tok = usage.get("outputTokens")
    stop = resp.get("stopReason")
    return "".join(parts), in_tok, out_tok, stop if isinstance(stop, str) else None


def _stop_summary(reason: str, pending: list[dict[str, Any]], n_written: int, n_skipped: int, budget: Budget) -> dict[str, Any]:
    return {
        "written": n_written,
        "skipped": n_skipped,
        "spent_usd": budget.global_spent,
        "stage_spent_usd": budget.stage_spent,
        "stopped": True,
        "stop_reason": reason,
        "pending": [
            {"qid": row["qid"], "arm": row["arm"], "dataset": row.get("dataset")}
            for row in pending
        ],
    }


def generate_rows(
    cfg: dict[str, Any],
    retrieval_rows: list[dict[str, Any]],
    out_path: Path,
    budget: Budget,
    *,
    model_id: str,
    client: Any,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 6,
) -> dict[str, Any]:
    price_for(cfg, model_id)
    system = system_prompt()
    done = done_keys(out_path, ("qid", "arm"))
    max_out = int(cfg["generator_max_output_tokens"])
    temperature = float(cfg["temperature"])
    n_written = 0
    n_skipped = 0
    recent_validation: list[str] = []
    for index, row in enumerate(retrieval_rows):
        key = (row["qid"], row["arm"])
        if key in done:
            n_skipped += 1
            continue
        user = build_user_message(row["question"], row["passages"])
        est_in = prompt_token_estimate(system, user)
        est_cost = cost_usd(cfg, model_id, est_in, max_out)
        reason = budget.blocking_reason(est_cost)
        if reason:
            pending = [
                item for item in retrieval_rows[index:]
                if (item["qid"], item["arm"]) not in done
            ]
            return _stop_summary(reason, pending, n_written, n_skipped, budget)
        raw = ""
        in_tok = out_tok = None
        stop_reason = None
        error = None
        call_failed = False
        delay = 1.0
        for attempt in range(max_attempts):
            try:
                resp = converse_once(client, model_id, system, user, max_out, temperature)
                raw, in_tok, out_tok, stop_reason = _text_and_usage(resp)
                break
            except Exception as exc:
                code = _error_code(exc)
                if code in FATAL_CODES or "AccessDenied" in code:
                    raise ProtocolError(
                        f"fatal Bedrock error ({code}) on qid={row['qid']} arm={row['arm']}: "
                        f"{_error_message(exc)}"
                    ) from exc
                if code in THROTTLE_CODES and attempt + 1 < max_attempts:
                    sleep(delay)
                    delay *= 2
                    continue
                if code == "ValidationException":
                    error = {"type": code, "message": _error_message(exc)}
                    call_failed = True
                    recent_validation.append(error["message"])
                    break
                if code in THROTTLE_CODES:
                    raise ProtocolError(
                        f"still throttled after {max_attempts} attempts on qid={row['qid']}: {code}"
                    ) from exc
                raise ProtocolError(
                    f"unhandled Bedrock error ({code}) on qid={row['qid']} arm={row['arm']}: "
                    f"{_error_message(exc)}"
                ) from exc
        else:
            raise ProtocolError(f"exhausted retries on qid={row['qid']} arm={row['arm']}")
        if len(recent_validation) >= 5 and len(set(recent_validation[-5:])) == 1:
            raise ProtocolError(
                "five identical ValidationException messages in a row. "
                "This is a request-shape failure, not a per-row data error. Stopped: "
                + recent_validation[-1]
            )
        if error is None:
            recent_validation.clear()
        if in_tok is None or out_tok is None:
            if error is None:
                raise ProtocolError(
                    f"Bedrock response for qid={row['qid']} arm={row['arm']} had no usage tokens. "
                    "Refusing to invent a cost."
                )
            in_tok, out_tok = 0, 0
        usd = cost_usd(cfg, model_id, int(in_tok), int(out_tok))
        truncated = stop_reason == "max_tokens"
        parsed = parse_self_confidence(raw, call_failed=call_failed, truncated=truncated)
        record = {
            "qid": row["qid"],
            "dataset": row["dataset"],
            "arm": row["arm"],
            "question_type": row["question_type"],
            "answer": parsed["answer"],
            "raw_response": raw,
            "self_confidence": parsed["value"],
            "self_reported": parsed["reported"],
            "self_status": parsed["status"],
            "deployed_self_confidence": parsed["deployed_value"],
            "deployed_self_reported": parsed["deployed_reported"],
            "deployed_self_status": parsed["deployed_status"],
            "trailing_truncated": bool(parsed["trailing_truncated"]),
            "stop_reason": stop_reason,
            "input_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "usd": usd,
            "model_id": model_id,
            "temperature": temperature,
            "region": cfg["region"],
            "error": error,
        }
        append_jsonl(out_path, record)
        done.add(key)
        budget.record({
            "qid": row["qid"], "arm": row["arm"], "dataset": row.get("dataset"),
            "usd": usd, "input_tokens": int(in_tok), "output_tokens": int(out_tok),
            "model_id": model_id,
        })
        n_written += 1
        overrun = budget.blocking_reason(0.0)
        if overrun and budget.global_spent > budget.total_usd_cap + 1e-12:
            # The call was under the estimate and still crossed the cap.
            # Keep the row. Do not start another one.
            pending = [
                item for item in retrieval_rows[index + 1:]
                if (item["qid"], item["arm"]) not in done
            ]
            return _stop_summary(
                f"{SPEND_CAP_REASON}: a call under the pre-call estimate crossed the cap "
                f"(ledger ${budget.global_spent:.6f}, total cap ${budget.total_usd_cap:.6f})",
                pending, n_written, n_skipped, budget,
            )
    return {
        "written": n_written,
        "skipped": n_skipped,
        "spent_usd": budget.global_spent,
        "stage_spent_usd": budget.stage_spent,
        "stopped": False,
        "stop_reason": None,
        "pending": [],
    }


def make_client(region: str) -> Any:
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    except ImportError as exc:
        raise ProtocolError("boto3 is not installed; pip install -r experiments/requirements.txt") from exc
    try:
        session = boto3.Session(region_name=region)
        if session.get_credentials() is None:
            raise ProtocolError(
                f"no AWS credentials for region {region}. Generation was not started."
            )
        return session.client("bedrock-runtime", region_name=region)
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise ProtocolError(f"no AWS credentials: {exc}") from exc


def load_retrieval(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    validate_rows(rows, RETRIEVAL_FIELDS + ("question",), path)
    return rows


def write_dry_run_artifact(cfg: dict[str, Any], seed: int, estimate: dict[str, Any], path: Path) -> None:
    body = artifact_meta(cfg, seed, stage="generate_dry_run", estimate=estimate, prompts={"generator_system": system_prompt()})
    write_json(path, body)
