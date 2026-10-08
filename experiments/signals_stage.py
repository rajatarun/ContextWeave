"""Lexical grounding, NLI grounding, and the sampled judge.

Lexical support and the judge prompt are the deployed ones
(``verified_reward.lexical_support``, ``verified_reward.JUDGE_PROMPT``).
The experiment scores the stored claim against each passage and against each
pair of passages. Numbers in the claim must occur verbatim. A null claim,
an empty retrieval, and an abstention are missing rather than a score.

NLI is ``cross-encoder/nli-deberta-v3-small``. Each passage and each pair is
split into sentence windows that fit the token limit. A window that does not
fit is cut to the limit and the cut is counted on the row.

The experiment judge draws a seeded prefix at ``v2.judge_sample_rate``.
``judge_sampled`` remains the deployed hash sample and does not use the seed.
Every arm of a sampled question is judged. A question outside the sample is
written with reason ``not_sampled``.

The judge model must differ from the generator model unless ``--allow-same-judge``
is passed. That flag is stored on the artifact either way.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from experiments.bedrock_batch import (
    TERMINAL_BAD, load_state, model_family, model_input, parsed_output_rows,
    poll_job, record_id_for, run_job, download_output_lines,
)
from experiments.common import (
    ProtocolError, append_jsonl, cost_usd, done_keys, estimate_tokens, price_for,
    snapshot_revision,
)
from experiments.ledger import SPEND_CAP_REASON, Budget

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))
sys.path.insert(0, str(_ROOT / "scripts"))

import verified_reward as V  # noqa: E402

from experiments.generate_stage import (
    FATAL_CODES, THROTTLE_CODES, _error_code, _error_message,
)


class JudgeAccessDenied(ProtocolError):
    """The judge model rejected the credentials or the model grant."""

JUDGE_PROMPT = V.JUDGE_PROMPT


def judge_sampled(qid: str, rate: float) -> bool:
    return V.should_judge(qid, rate)


def lexical_value(answer: str, passages: Sequence[str], threshold: float, weight: float) -> tuple[float | None, str | None, dict[str, Any]]:
    signal = V.grounding_signal(answer, passages, verifier=V.lexical_support, threshold=threshold, weight=weight)
    reason = signal.detail.get("reason") if signal.value is None else None
    return signal.value, reason, signal.detail


def nli_value(answer: str, passages: Sequence[str], verifier: V.Verifier, threshold: float, weight: float) -> tuple[float | None, str | None, dict[str, Any]]:
    signal = V.grounding_signal(answer, passages, verifier=verifier, threshold=threshold, weight=weight)
    reason = signal.detail.get("reason") if signal.value is None else None
    return signal.value, reason, signal.detail


def load_nli(model_name: str) -> tuple[V.Verifier, dict[str, Any]]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ProtocolError(
            "sentence-transformers is not installed; pip install -r experiments/requirements.txt"
        ) from exc
    model = CrossEncoder(model_name, device="cpu")
    labels = [str(l).lower() for l in getattr(model.config, "id2label", {}).values()] or [
        "contradiction", "entailment", "neutral",
    ]
    if "entailment" not in labels:
        raise ProtocolError(f"NLI model {model_name} has no entailment label: {labels}")
    ent = labels.index("entailment")
    revision = None
    vocab = None
    try:
        revision = model.model.config._commit_hash
    except Exception:
        revision = None
    tok = getattr(model, "tokenizer", None)
    init = getattr(tok, "init_kwargs", None) if tok is not None else None
    if isinstance(init, dict):
        vocab = init.get("vocab_file")
    revision = snapshot_revision(revision, vocab)

    def verify(claim: str, passage: str) -> float:
        probs = model.predict([(passage, claim)], apply_softmax=True)[0]
        return float(probs[ent])

    return verify, {"nli_model": model_name, "revision": revision, "labels": labels}


def check_judge_model(judge_model: str, generator_model: str, allow_same: bool) -> None:
    if judge_model == generator_model and not allow_same:
        raise ProtocolError(
            f"judge model {judge_model} equals the generator model. "
            "Pass --allow-same-judge to run anyway. Refusing to grade a model with itself."
        )


def judge_one(
    client: Any,
    model_id: str,
    question: str,
    answer: str,
    passages: Sequence[str],
    max_tokens: int,
    *,
    sleep: Callable[[float], None],
    max_attempts: int = 6,
) -> tuple[float | None, str | None, str, int, int]:
    built = build_judge_prompt(question, answer, passages)
    if built["prompt"] is None:
        return None, built["reason"], "", 0, 0
    prompt = built["prompt"]
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
            raw = ""
            for block in resp.get("output", {}).get("message", {}).get("content", []):
                if isinstance(block, dict) and "text" in block:
                    raw += block["text"]
            usage = resp.get("usage") or {}
            in_tok, out_tok = usage.get("inputTokens"), usage.get("outputTokens")
            if in_tok is None or out_tok is None:
                raise ProtocolError(
                    "judge response had no usage tokens. Refusing to invent a cost."
                )
            score = V.parse_judge_score(raw)
            if score is None:
                return None, "unparseable", raw, int(in_tok), int(out_tok)
            return score, None, raw, int(in_tok), int(out_tok)
        except Exception as exc:
            code = _error_code(exc)
            if code in FATAL_CODES or "AccessDenied" in code:
                raise JudgeAccessDenied(
                    f"judge model access not yet granted ({code}): {_error_message(exc)}"
                ) from exc
            if code in THROTTLE_CODES and attempt + 1 < max_attempts:
                sleep(delay)
                delay *= 2
                continue
            if code == "ValidationException":
                return None, "validation_exception", _error_message(exc), 0, 0
            if code in THROTTLE_CODES:
                raise ProtocolError(f"judge still throttled after {max_attempts} attempts: {code}") from exc
            raise ProtocolError(f"unhandled judge error ({code}): {_error_message(exc)}") from exc
    raise ProtocolError("judge retries exhausted")


def signal_row(base: dict[str, Any], value: float | None, reason: str | None, **extra: Any) -> dict[str, Any]:
    row = {
        "qid": base["qid"],
        "dataset": base["dataset"],
        "arm": base["arm"],
        "question_type": base["question_type"],
        "value": value,
        "reason": reason,
    }
    row.update(extra)
    return row


EVIDENCE_CHAR_LIMIT = 12000


def build_judge_prompt(question: str, answer: str, passages: Sequence[str]) -> dict[str, Any]:
    """Deployed judge prompt. A cut at the character limit is recorded."""
    texts = [p.strip() for p in passages if isinstance(p, str) and p.strip()]
    body = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(texts, start=1))
    truncated = len(body) > EVIDENCE_CHAR_LIMIT
    evidence = body[:EVIDENCE_CHAR_LIMIT]
    if not evidence:
        return {
            "prompt": None,
            "reason": "no_passages",
            "evidence_truncated": False,
            "n_evidence_chars": 0,
        }
    return {
        "prompt": JUDGE_PROMPT.format(question=question, evidence=evidence, answer=answer),
        "reason": None,
        "evidence_truncated": truncated,
        "n_evidence_chars": len(evidence),
    }


def load_nli_tools(model_name: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Entailment probability, a token counter, a truncator, and model info.

    The counter and truncator use the model's tokenizer with special tokens
    off, so a window can be cut before the encoder sees it.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ProtocolError(
            "sentence-transformers is not installed; pip install -r experiments/requirements.txt"
        ) from exc
    model = CrossEncoder(model_name, device="cpu")
    labels = [str(label).lower() for label in getattr(model.config, "id2label", {}).values()] or [
        "contradiction", "entailment", "neutral",
    ]
    if "entailment" not in labels:
        raise ProtocolError(f"NLI model {model_name} has no entailment label: {labels}")
    ent = labels.index("entailment")
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ProtocolError(f"NLI model {model_name} has no tokenizer. Refusing to window by characters.")
    revision = None
    vocab = None
    try:
        revision = model.model.config._commit_hash
    except Exception:
        revision = None
    init = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init, dict):
        vocab = init.get("vocab_file")
    revision = snapshot_revision(revision, vocab)
    reported = getattr(tokenizer, "model_max_length", None)

    def predict(claim: str, passage: str) -> float:
        probs = model.predict([(passage, claim)], apply_softmax=True)[0]
        return float(probs[ent])

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        return len(encoded["input_ids"])

    def truncate(text: str, n: int) -> str:
        if n <= 0 or not text:
            return ""
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        ids = encoded["input_ids"][:n]
        return tokenizer.decode(ids, skip_special_tokens=True)

    info = {
        "nli_model": model_name,
        "revision": revision,
        "labels": labels,
        "model_max_length": reported if isinstance(reported, int) and reported < 10**6 else None,
    }
    return predict, count_tokens, truncate, info


def nli_limit(model_max_length: int | None, configured: int) -> int:
    """The window limit is the smaller of the model max and the configured cap."""
    if configured < 8:
        raise ProtocolError(f"nli_max_tokens {configured} is below 8")
    if isinstance(model_max_length, int) and 8 <= model_max_length < configured:
        return model_max_length
    return configured


def price_fields(cfg: dict[str, Any], model_id: str, pricing: str) -> dict[str, Any]:
    price = price_for(cfg, model_id, pricing)
    return {
        "pricing": pricing,
        "price_usd_per_million": {"input": float(price["input"]), "output": float(price["output"])},
    }


def judge_rows_batch(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    out_path: Path,
    budget: Budget,
    *,
    model_id: str,
    s3: Any,
    bedrock: Any,
    seed: int,
    sampled: set[str],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """Write unsampled rows, then one or more Llama batch jobs for the rest.

    The projected batch cost is checked before a job is created. A job smaller
    than ``batch.min_records`` is not submitted and is not padded.
    """
    price_for(cfg, model_id, "batch")
    family = model_family(model_id)
    done = done_keys(out_path, ("qid", "arm"))
    n_skipped = sum(1 for row in rows if (row["qid"], row["arm"]) in done)
    max_out = int(cfg["judge_max_output_tokens"])
    min_records = int(cfg["batch"]["min_records"])
    poll_seconds = float(cfg["batch"]["poll_seconds"])
    if poll_seconds <= 0:
        raise ProtocolError("batch.poll_seconds must be positive")
    state_dir = out_path.parent / "batch"
    state_dir.mkdir(parents=True, exist_ok=True)
    n_written = _recover_judge_states(
        cfg, rows, out_path, budget, model_id=model_id, done=done,
        s3=s3, bedrock=bedrock, sleep=sleep, state_dir=state_dir,
    )
    pending: list[dict[str, Any]] = []
    for row in rows:
        key = (row["qid"], row["arm"])
        if key in done:
            continue
        if row["qid"] not in sampled:
            _append_judge(
                out_path, budget, row, None, "not_sampled", "",
                in_tok=0, out_tok=0, usage_observed=False, error=None,
                model_id=model_id, cfg=cfg, pricing="batch", sampled_flag=False,
                evidence_truncated=False, n_evidence_chars=0, record_ledger=False,
            )
            done.add(key)
            n_written += 1
            continue
        pending.append(row)

    def upper_bound(row: dict[str, Any]) -> tuple[float, str | None, dict[str, Any]]:
        built = build_judge_prompt(row["question"], row["answer"], row["passages"])
        if built["prompt"] is None:
            return 0.0, None, built
        return cost_usd(cfg, model_id, estimate_tokens(built["prompt"]), max_out, "batch"), built["prompt"], built

    while pending:
        empty = [row for row in pending if upper_bound(row)[1] is None]
        for row in empty:
            built = upper_bound(row)[2]
            _append_judge(
                out_path, budget, row, None, built["reason"], "",
                in_tok=0, out_tok=0, usage_observed=False, error=None,
                model_id=model_id, cfg=cfg, pricing="batch", sampled_flag=True,
                evidence_truncated=False, n_evidence_chars=0, record_ledger=False,
            )
            done.add((row["qid"], row["arm"]))
            n_written += 1
        pending = [row for row in pending if upper_bound(row)[1] is not None]
        if not pending:
            break
        chosen: list[dict[str, Any]] = []
        prompts: list[str] = []
        built_rows: list[dict[str, Any]] = []
        projected = 0.0
        for row in pending:
            cost, prompt, built = upper_bound(row)
            if budget.blocking_reason(projected + cost):
                break
            chosen.append(row)
            prompts.append(prompt or "")
            built_rows.append(built)
            projected += cost
        if not chosen:
            return _judge_stop(
                budget.blocking_reason(upper_bound(pending[0])[0]) or SPEND_CAP_REASON,
                pending, n_written, n_skipped, budget,
            )
        if len(chosen) < min_records:
            return _judge_stop(
                f"batch job would have {len(chosen)} records and batch.min_records is {min_records}. "
                "Refusing to pad the job or to switch to on-demand. "
                "Pass --inference-mode on_demand for these rows.",
                pending, n_written, n_skipped, budget,
            )
        records = []
        seen: dict[str, tuple] = {}
        for row, prompt in zip(chosen, prompts):
            rid = record_id_for(str(row["dataset"]), str(row["qid"]), str(row["arm"]), model_id, "judge")
            key = (row["qid"], row["arm"])
            if rid in seen:
                raise ProtocolError(
                    f"batch recordId {rid} collides for {seen[rid]} and {key}. Refusing to merge the rows."
                )
            seen[rid] = key
            records.append({
                "recordId": rid,
                "modelInput": model_input(family, "", prompt, max_out, 0.0),
                "meta": {"qid": row["qid"], "arm": row["arm"], "dataset": row["dataset"]},
            })
        result = run_job(
            cfg=cfg, model_id=model_id, records=records, state_dir=state_dir,
            s3=s3, bedrock=bedrock, sleep=sleep, seed=seed,
        )
        by_key = {(row["qid"], row["arm"]): (row, built) for row, built in zip(chosen, built_rows)}
        for rid, key in seen.items():
            row, built = by_key[key]
            _append_from_judge_output(
                cfg, row, result["rows"][rid], built, model_id=model_id,
                out_path=out_path, done=done, budget=budget,
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


def _judge_stop(reason: str, pending: list[dict[str, Any]], n_written: int, n_skipped: int, budget: Budget) -> dict[str, Any]:
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


def _append_judge(
    out_path: Path,
    budget: Budget,
    row: dict[str, Any],
    value: float | None,
    reason: str | None,
    raw: str,
    *,
    in_tok: int,
    out_tok: int,
    usage_observed: bool,
    error: dict[str, Any] | None,
    model_id: str,
    cfg: dict[str, Any],
    pricing: str,
    sampled_flag: bool,
    evidence_truncated: bool,
    n_evidence_chars: int,
    record_ledger: bool,
) -> None:
    usd = cost_usd(cfg, model_id, int(in_tok), int(out_tok), pricing) if record_ledger or in_tok or out_tok else 0.0
    body = signal_row(
        row, value, reason,
        judge_sampled=sampled_flag,
        judge_model_id=model_id,
        raw_response=raw,
        input_tokens=int(in_tok),
        output_tokens=int(out_tok),
        usage_observed=usage_observed,
        usd=usd,
        evidence_truncated=evidence_truncated,
        n_evidence_chars=n_evidence_chars,
        error=error,
        **price_fields(cfg, model_id, pricing),
    )
    append_jsonl(out_path, body)
    if record_ledger:
        budget.record({
            "qid": row["qid"], "arm": row["arm"], "dataset": row.get("dataset"),
            "usd": usd, "input_tokens": int(in_tok), "output_tokens": int(out_tok),
            "model_id": model_id, **price_fields(cfg, model_id, pricing),
        })


def _append_from_judge_output(
    cfg: dict[str, Any],
    row: dict[str, Any],
    parsed_out: dict[str, Any],
    built: dict[str, Any],
    *,
    model_id: str,
    out_path: Path,
    done: set[tuple],
    budget: Budget,
) -> None:
    call_failed = parsed_out["error"] is not None
    raw = parsed_out["text"] or ""
    if call_failed:
        value, reason = None, "validation_exception" if (parsed_out["error"] or {}).get("type") == "ValidationException" else "call_failed"
    else:
        value = V.parse_judge_score(raw)
        reason = None if value is not None else "unparseable"
    _append_judge(
        out_path, budget, row, value, reason, raw,
        in_tok=int(parsed_out["input_tokens"]), out_tok=int(parsed_out["output_tokens"]),
        usage_observed=bool(parsed_out["usage_observed"]), error=parsed_out["error"],
        model_id=model_id, cfg=cfg, pricing="batch", sampled_flag=True,
        evidence_truncated=bool(built["evidence_truncated"]),
        n_evidence_chars=int(built["n_evidence_chars"]),
        record_ledger=True,
    )
    done.add((row["qid"], row["arm"]))


def _recover_judge_states(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    out_path: Path,
    budget: Budget,
    *,
    model_id: str,
    done: set[tuple],
    s3: Any,
    bedrock: Any,
    sleep: Callable[[float], None],
    state_dir: Path,
) -> int:
    if not state_dir.is_dir():
        return 0
    index = {(row["qid"], row["arm"]): row for row in rows}
    poll_seconds = float(cfg["batch"]["poll_seconds"])
    max_polls = max(1, int(float(cfg["batch"]["timeout_hours"]) * 3600.0 / poll_seconds) + 1)
    written = 0
    for path in sorted(state_dir.glob("cw*.json")):
        state = load_state(path)
        if not state or state.get("model_id") != model_id or not state.get("job_arn"):
            continue
        if state.get("stage") not in (None, "judge"):
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
                    "which is not in this judge input"
                )
            built = build_judge_prompt(row["question"], row["answer"], row["passages"])
            _append_from_judge_output(
                cfg, row, parsed[rid], built, model_id=model_id,
                out_path=out_path, done=done, budget=budget,
            )
            written += 1
    return written
