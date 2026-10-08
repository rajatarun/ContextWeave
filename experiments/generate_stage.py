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

from experiments.bedrock_batch import (
    TERMINAL_BAD, download_output_lines, load_state, model_family, model_input,
    parsed_output_rows, poll_job, record_id_for, run_job,
)
from experiments.common import (
    GENERATION_SCHEMA_VERSION, ProtocolError, append_jsonl, artifact_meta, cost_usd,
    done_keys, estimate_tokens, price_for, read_jsonl, validate_rows, write_json,
    RETRIEVAL_FIELDS,
)
from experiments.confidence import parse_self_confidence
from experiments.labels import abstention_outcome
from experiments.ledger import SPEND_CAP_REASON, Budget
from experiments.metrics import is_abstention

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
    on_demand = cost_usd(cfg, model_id, input_tokens, output_tokens, "on_demand")
    batch = cost_usd(cfg, model_id, input_tokens, output_tokens, "batch")
    pricing = cfg.get("inference_mode") or "on_demand"
    usd = batch if pricing == "batch" else on_demand
    return {
        "model_id": model_id,
        "region": cfg["region"],
        "temperature": cfg["temperature"],
        "pricing": pricing,
        "n_calls": n,
        "input_tokens_estimate": input_tokens,
        "output_tokens_upper_bound": output_tokens,
        "generator_max_output_tokens": max_out,
        "usd_upper_bound": usd,
        "on_demand_usd_upper_bound": on_demand,
        "batch_usd_upper_bound": batch,
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


def _refuse_foreign_generation(path: Path) -> None:
    """v1 generation rows have no schema_version 2 and are not resumed."""
    try:
        rows = read_jsonl(path)
    except ProtocolError:
        return
    for i, row in enumerate(rows, 1):
        if row.get("schema_version") != GENERATION_SCHEMA_VERSION:
            raise ProtocolError(
                f"{path}:{i}: generation row is not schema_version {GENERATION_SCHEMA_VERSION}. "
                "v1 generation rows are not reused. Move the file aside and rerun."
            )


def _abstention_fields(row: dict[str, Any], parsed: dict[str, Any], call_failed: bool) -> dict[str, Any]:
    missing = [key for key in ("unanswerable", "source_retrieved") if key not in row]
    if missing:
        raise ProtocolError(
            f"qid={row.get('qid')} arm={row.get('arm')}: retrieval row is missing {missing}. "
            "Abstention labels need source_retrieved and unanswerable. Refusing to guess."
        )
    abstained = (
        not call_failed
        and parsed["status"] in ("ok", "omitted")
        and is_abstention(parsed["answer"])
    )
    return abstention_outcome(
        abstained=abstained,
        unanswerable=bool(row["unanswerable"]),
        source_retrieved=bool(row["source_retrieved"]),
    )


def _price_fields(cfg: dict[str, Any], model_id: str, pricing: str) -> dict[str, Any]:
    price = price_for(cfg, model_id, pricing)
    return {
        "pricing": pricing,
        "price_usd_per_million": {"input": float(price["input"]), "output": float(price["output"])},
    }


def _generation_record(
    cfg: dict[str, Any],
    row: dict[str, Any],
    parsed: dict[str, Any],
    *,
    model_id: str,
    temperature: float,
    raw: str,
    stop_reason: str | None,
    in_tok: int,
    out_tok: int,
    usd: float,
    error: dict[str, Any] | None,
    pricing: str,
    usage_observed: bool,
) -> dict[str, Any]:
    return {
        "qid": row["qid"],
        "dataset": row["dataset"],
        "arm": row["arm"],
        "question_type": row["question_type"],
        "answer": parsed["answer"],
        "claim": parsed["claim"],
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
        "usage_observed": usage_observed,
        "usd": usd,
        "model_id": model_id,
        "temperature": temperature,
        "region": cfg["region"],
        "error": error,
        "schema_version": GENERATION_SCHEMA_VERSION,
        **_abstention_fields(row, parsed, call_failed=parsed["status"] == "failed"),
        **_price_fields(cfg, model_id, pricing),
    }


def _ledger_row(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": row["qid"], "arm": row["arm"], "dataset": row.get("dataset"),
        "usd": record["usd"], "input_tokens": record["input_tokens"],
        "output_tokens": record["output_tokens"], "model_id": record["model_id"],
        "pricing": record["pricing"],
        "price_usd_per_million": record["price_usd_per_million"],
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
    pricing: str = "on_demand",
) -> dict[str, Any]:
    price_for(cfg, model_id, pricing)
    _refuse_foreign_generation(out_path)
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
        est_cost = cost_usd(cfg, model_id, est_in, max_out, pricing)
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
        usd = cost_usd(cfg, model_id, int(in_tok), int(out_tok), pricing)
        truncated = stop_reason == "max_tokens"
        parsed = parse_self_confidence(raw, call_failed=call_failed, truncated=truncated)
        record = _generation_record(
            cfg, row, parsed, model_id=model_id, temperature=temperature,
            raw=raw, stop_reason=stop_reason, in_tok=int(in_tok), out_tok=int(out_tok),
            usd=usd, error=error, pricing=pricing, usage_observed=error is None,
        )
        append_jsonl(out_path, record)
        done.add(key)
        budget.record(_ledger_row(row, record))
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
                f"({budget.cap_label()})",
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


def _append_from_output(
    cfg: dict[str, Any],
    row: dict[str, Any],
    parsed_out: dict[str, Any],
    *,
    model_id: str,
    temperature: float,
    out_path: Path,
    done: set[tuple],
    budget: Budget,
) -> None:
    call_failed = parsed_out["error"] is not None
    stop = parsed_out["stop_reason"]
    truncated = isinstance(stop, str) and stop in {"max_tokens", "max_gen_len"}
    parsed = parse_self_confidence(
        parsed_out["text"], call_failed=call_failed, truncated=truncated and not call_failed,
    )
    record = _generation_record(
        cfg, row, parsed, model_id=model_id, temperature=temperature,
        raw=parsed_out["text"], stop_reason=stop if isinstance(stop, str) else None,
        in_tok=int(parsed_out["input_tokens"]), out_tok=int(parsed_out["output_tokens"]),
        usd=cost_usd(
            cfg, model_id, int(parsed_out["input_tokens"]), int(parsed_out["output_tokens"]), "batch",
        ),
        error=parsed_out["error"], pricing="batch",
        usage_observed=bool(parsed_out["usage_observed"]),
    )
    append_jsonl(out_path, record)
    done.add((row["qid"], row["arm"]))
    budget.record(_ledger_row(row, record))


def _recover_batch_states(
    cfg: dict[str, Any],
    retrieval_rows: list[dict[str, Any]],
    out_path: Path,
    budget: Budget,
    *,
    model_id: str,
    temperature: float,
    done: set[tuple],
    s3: Any,
    bedrock: Any,
    sleep: Callable[[float], None],
    state_dir: Path,
) -> int:
    """Finish jobs whose ARN is already stored and whose rows are not on disk."""
    if not state_dir.is_dir():
        return 0
    index = {(row["qid"], row["arm"]): row for row in retrieval_rows}
    poll_seconds = float(cfg["batch"]["poll_seconds"])
    max_polls = max(1, int(float(cfg["batch"]["timeout_hours"]) * 3600.0 / poll_seconds) + 1)
    written = 0
    for path in sorted(state_dir.glob("cw*.json")):
        state = load_state(path)
        if not state or state.get("model_id") != model_id or not state.get("job_arn"):
            continue
        missing = []
        for rid, meta in (state.get("records") or {}).items():
            key = (meta.get("qid"), meta.get("arm"))
            if key not in done:
                missing.append((str(rid), key))
        if not missing:
            continue
        job = poll_job(bedrock, state["job_arn"], sleep, poll_seconds, max_polls)
        status = str(job.get("status") or "")
        if status in TERMINAL_BAD:
            message = job.get("message") or status
            raise ProtocolError(f"batch job {state['job_arn']} ended {status}: {message}")
        lines = download_output_lines(s3, state["bucket"], state["s3_output_prefix"])
        parsed = parsed_output_rows(state["family"], lines)
        for rid, key in missing:
            if rid not in parsed:
                raise ProtocolError(
                    f"batch job {state['job_arn']} returned no output for recordId {rid}. "
                    "Refusing to invent that row."
                )
            row = index.get(key)
            if row is None:
                raise ProtocolError(
                    f"batch state {path.name} names qid={key[0]} arm={key[1]}, "
                    "which is not in this retrieval file"
                )
            _append_from_output(
                cfg, row, parsed[rid], model_id=model_id, temperature=temperature,
                out_path=out_path, done=done, budget=budget,
            )
            written += 1
    return written


def generate_rows_batch(
    cfg: dict[str, Any],
    retrieval_rows: list[dict[str, Any]],
    out_path: Path,
    budget: Budget,
    *,
    model_id: str,
    s3: Any,
    bedrock: Any,
    seed: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """One or more batch jobs for the rows not already on disk.

    The projected batch cost of a job is checked against the spend cap before
    the job is created. A job smaller than ``batch.min_records`` is not
    submitted and is not padded.
    """
    price_for(cfg, model_id, "batch")
    _refuse_foreign_generation(out_path)
    system = system_prompt()
    family = model_family(model_id)
    done = done_keys(out_path, ("qid", "arm"))
    n_skipped = sum(1 for row in retrieval_rows if (row["qid"], row["arm"]) in done)
    max_out = int(cfg["generator_max_output_tokens"])
    temperature = float(cfg["temperature"])
    min_records = int(cfg["batch"]["min_records"])
    state_dir = out_path.parent / "batch"
    state_dir.mkdir(parents=True, exist_ok=True)
    n_written = _recover_batch_states(
        cfg, retrieval_rows, out_path, budget, model_id=model_id, temperature=temperature,
        done=done, s3=s3, bedrock=bedrock, sleep=sleep, state_dir=state_dir,
    )
    pending = [row for row in retrieval_rows if (row["qid"], row["arm"]) not in done]

    def upper_bound(row: dict[str, Any]) -> tuple[float, str]:
        user = build_user_message(row["question"], row["passages"])
        est_in = prompt_token_estimate(system, user)
        return cost_usd(cfg, model_id, est_in, max_out, "batch"), user

    while pending:
        chosen: list[dict[str, Any]] = []
        users: list[str] = []
        projected = 0.0
        for row in pending:
            cost, user = upper_bound(row)
            if budget.blocking_reason(projected + cost):
                break
            chosen.append(row)
            users.append(user)
            projected += cost
        if not chosen:
            return _stop_summary(
                budget.blocking_reason(upper_bound(pending[0])[0]) or SPEND_CAP_REASON,
                pending, n_written, n_skipped, budget,
            )
        if len(chosen) < min_records:
            return _stop_summary(
                f"batch job would have {len(chosen)} records and batch.min_records is {min_records}. "
                "Refusing to pad the job or to switch to on-demand. "
                "Pass --inference-mode on_demand for these rows.",
                pending, n_written, n_skipped, budget,
            )
        blocked = budget.blocking_reason(projected)
        if blocked:
            return _stop_summary(blocked, pending, n_written, n_skipped, budget)
        records = []
        seen: dict[str, tuple] = {}
        for row, user in zip(chosen, users):
            rid = record_id_for(str(row["dataset"]), str(row["qid"]), str(row["arm"]), model_id)
            key = (row["qid"], row["arm"])
            if rid in seen:
                raise ProtocolError(
                    f"batch recordId {rid} collides for {seen[rid]} and {key}. "
                    "Refusing to merge the rows."
                )
            seen[rid] = key
            records.append({
                "recordId": rid,
                "modelInput": model_input(family, system, user, max_out, temperature),
                "meta": {"qid": row["qid"], "arm": row["arm"], "dataset": row["dataset"]},
            })
        result = run_job(
            cfg=cfg, model_id=model_id, records=records, state_dir=state_dir,
            s3=s3, bedrock=bedrock, sleep=sleep, seed=seed,
        )
        by_key = {(row["qid"], row["arm"]): row for row in chosen}
        for rid, key in seen.items():
            _append_from_output(
                cfg, by_key[key], result["rows"][rid], model_id=model_id,
                temperature=temperature, out_path=out_path, done=done, budget=budget,
            )
            n_written += 1
        chosen_keys = set(seen.values())
        pending = [row for row in pending if (row["qid"], row["arm"]) not in chosen_keys]
    return {
        "written": n_written,
        "skipped": n_skipped,
        "spent_usd": budget.global_spent,
        "stage_spent_usd": budget.stage_spent,
        "stopped": False,
        "stop_reason": None,
        "pending": [],
    }


def make_batch_clients(region: str) -> tuple[Any, Any]:
    """S3 and Bedrock control-plane clients. Missing credentials stop the run."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    except ImportError as exc:
        raise ProtocolError("boto3 is not installed; pip install -r experiments/requirements.txt") from exc
    try:
        session = boto3.Session(region_name=region)
        if session.get_credentials() is None:
            raise ProtocolError(
                f"no AWS credentials for region {region}. Batch generation was not started."
            )
        return (
            session.client("s3", region_name=region),
            session.client("bedrock", region_name=region),
        )
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise ProtocolError(f"no AWS credentials: {exc}") from exc


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
