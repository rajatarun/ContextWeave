#!/usr/bin/env python3
"""Compute one signal family per invocation.

``lexical`` and ``nli`` score the stored claim against single passages and
against pairs. ``nli`` windows each text to the token limit and records every
cut. ``judge`` draws a seeded prefix at ``v2.judge_sample_rate`` and calls
Llama on demand or as a Bedrock batch job. ``self_percentile``, ``logistic``,
and ``oracle`` do not call a model. Rows already stored for a per-row signal
are skipped. Percentile and logistic are functions of the whole log, so a
partial file is refused.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (
    DATASETS, RESULTS, ProtocolError, append_jsonl, artifact_meta, cost_usd,
    done_keys, estimate_tokens, in_sample, load_config, price_for, read_jsonl,
    sample_qids_by_dataset, stage_jsonl, validate_rows, write_json,
)
from experiments.generate_stage import make_batch_clients, make_client
from experiments.judge_access import clear_marker, write_marker
from experiments.ledger import SPEND_CAP_REASON, Budget
from experiments.metrics import score_answer
from experiments.records import load_joined
from experiments.signal_score import (
    assign_self_percentiles, lexical_unit, logistic_rows, oracle_correct,
    oracle_retrieval, score_stored_claim, seeded_ids, stream_seed, windowed_unit,
)
from experiments.signals_stage import (
    JUDGE_PROMPT, JudgeAccessDenied, _append_judge, build_judge_prompt,
    check_judge_model, judge_one, judge_rows_batch, load_nli_tools, nli_limit,
    signal_row,
)


def _bases(results: Path, cfg: dict[str, Any], names: list[str]) -> list[dict]:
    return [r for r in load_joined(results, cfg, names, attach_signals=False)]


def _rate(cfg: dict[str, Any], key: str) -> float:
    value = cfg["v2"][key]
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"v2.{key} is {value!r}. Refusing to guess a fraction.") from exc
    if rate < 0 or rate > 1:
        raise ProtocolError(f"v2.{key} must be in [0, 1], got {rate}")
    return rate


def _sample_block(qids: list[str], rate: float, seed: int, stream: str) -> dict[str, Any]:
    stream_value = stream_seed(seed, stream)
    chosen = seeded_ids(qids, rate, stream_value)
    return {
        "rate": rate,
        "seed": seed,
        "stream": stream,
        "stream_seed": stream_value,
        "n_questions": len(set(qids)),
        "n_sampled": len(chosen),
        "sampled_qids": chosen,
        "rule": (
            "sorted question ids, random.Random(stream_seed).shuffle, "
            "first take_count(n, rate) with half-up rounding"
        ),
    }


def _refuse_partial(path: Path, wanted: set[tuple]) -> bool:
    """True when ``path`` already holds exactly ``wanted`` and the stage can stop.

    A file that holds some other set is refused. Percentile and logistic
    change when the row set changes, so appending would score a different
    function than the one already on disk.
    """
    if not path.is_file():
        return False
    rows = read_jsonl(path)
    have = {(row["qid"], row["arm"]) for row in rows}
    if have == wanted:
        return True
    raise ProtocolError(
        f"{path.name} has {len(have)} rows and this run has {len(wanted)}. "
        "This signal is a function of the whole log. Move the file aside and rerun."
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        append_jsonl(path, row)


def _question_rows(results: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for name in names:
        for row in read_jsonl(results / "samples" / f"{name}.jsonl"):
            out[row["qid"]] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "signal",
        choices=("lexical", "nli", "judge", "self_percentile", "logistic", "oracle"),
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--judge-model-id", default=None)
    ap.add_argument("--allow-same-judge", action="store_true")
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--total-usd-cap", type=float, default=None,
                    help="shared generation+judge ceiling (default: config total_usd_cap)")
    ap.add_argument("--dry-run", action="store_true", help="judge only: count sampled calls, do not call the model")
    ap.add_argument("--inference-mode", choices=("on_demand", "batch"), default=None,
                    help="judge only: on_demand calls Converse; batch submits a Bedrock job")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    names = [p.strip() for p in args.datasets.split(",") if p.strip()]
    try:
        rows = _bases(args.results, cfg, names)
        threshold = float(cfg["grounding_threshold"])
        if args.signal == "lexical":
            path = stage_jsonl(args.results / "signals", "lexical")
            done = done_keys(path, ("qid", "arm"))
            n = 0
            for row in rows:
                if (row["qid"], row["arm"]) in done:
                    continue
                scored = score_stored_claim(row.get("claim"), row["passages"], lexical_unit, threshold)
                append_jsonl(path, signal_row(
                    row, scored["value"], scored["reason"], detail=scored["detail"], threshold=threshold,
                    scored_field="claim",
                ))
                n += 1
            meta = artifact_meta(
                cfg, seed, stage="signals_lexical", written=n, tau=threshold, scored_field="claim",
                support="single passage or pair",
            )
            write_json(args.results / "signals" / "lexical_meta.json", meta)
            print(f"lexical: wrote {n} rows")
            return 0
        if args.signal == "nli":
            predict, count_tokens, truncate, info = load_nli_tools(cfg["nli_model"])
            limit = nli_limit(info.get("model_max_length"), int(cfg["nli_max_tokens"]))
            special = int(cfg["nli_special_tokens"])
            unit = windowed_unit(predict, count_tokens, truncate, limit, special)
            path = stage_jsonl(args.results / "signals", "nli")
            done = done_keys(path, ("qid", "arm"))
            n = 0
            for row in rows:
                if (row["qid"], row["arm"]) in done:
                    continue
                scored = score_stored_claim(row.get("claim"), row["passages"], unit, threshold)
                append_jsonl(path, signal_row(
                    row, scored["value"], scored["reason"], detail=scored["detail"], threshold=threshold,
                    nli_model=info["nli_model"], scored_field="claim",
                    n_truncated_windows=(scored["detail"] or {}).get("n_truncated_windows", 0),
                ))
                n += 1
            stored = read_jsonl(path) if path.is_file() else []
            n_truncated = sum(int(row.get("n_truncated_windows") or 0) for row in stored)
            n_rows_truncated = sum(1 for row in stored if int(row.get("n_truncated_windows") or 0) > 0)
            meta = artifact_meta(
                cfg, seed, stage="signals_nli", written=n, nli=info, tau=threshold,
                nli_max_tokens=limit, nli_special_tokens=special, scored_field="claim",
                n_truncated_windows=n_truncated, n_rows_with_truncation=n_rows_truncated,
            )
            write_json(args.results / "signals" / "nli_meta.json", meta)
            print(f"nli: wrote {n} rows, {n_truncated} truncated windows")
            return 0
        if args.signal == "self_percentile":
            path = stage_jsonl(args.results / "signals", "self_percentile")
            wanted = {(row["qid"], row["arm"]) for row in rows}
            if _refuse_partial(path, wanted):
                print("self_percentile: file already matches this log")
                return 0
            written = assign_self_percentiles(rows)
            _write_rows(path, written)
            meta = artifact_meta(
                cfg, seed, stage="signals_self_percentile", written=len(written),
                peers="other ok confidences in the same dataset", ties="half",
                deterministic=True,
            )
            write_json(args.results / "signals" / "self_percentile_meta.json", meta)
            print(f"self_percentile: wrote {len(written)} rows")
            return 0
        if args.signal == "oracle":
            return _oracle(args.results, cfg, seed, names, rows)
        if args.signal == "logistic":
            return _logistic(args.results, cfg, seed, names, rows)
        return _judge(args, cfg, seed, names, rows)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2


def _oracle(results: Path, cfg: dict[str, Any], seed: int, names: list[str], rows: list[dict]) -> int:
    questions = _question_rows(results, names)
    retrieval = {}
    for name in names:
        for row in read_jsonl(results / "retrieval" / f"{name}.jsonl"):
            retrieval[(row["qid"], row["arm"])] = row
    correct_path = stage_jsonl(results / "signals", "oracle_correct")
    retrieval_path = stage_jsonl(results / "signals", "oracle_retrieval")
    done_correct = done_keys(correct_path, ("qid", "arm"))
    done_retrieval = done_keys(retrieval_path, ("qid", "arm"))
    n = 0
    for row in rows:
        key = (row["qid"], row["arm"])
        question = questions.get(row["qid"])
        if question is None:
            raise ProtocolError(f"oracle: {row['qid']} is not in the sample")
        if key not in done_correct:
            if "unanswerable" not in question or "gold_answers" not in question:
                raise ProtocolError(f"oracle: sample row {row['qid']} has no gold label")
            value = oracle_correct(row["answer"], question["gold_answers"], bool(question["unanswerable"]))
            append_jsonl(correct_path, signal_row(
                row, value, None, label="token_f1_correct",
            ))
            n += 1
        if key not in done_retrieval:
            source = retrieval.get(key)
            if source is None or "source_retrieved" not in source:
                raise ProtocolError(
                    f"oracle: retrieval row {key} has no source_retrieved. Refusing to guess."
                )
            value = oracle_retrieval(bool(source["source_retrieved"]))
            append_jsonl(retrieval_path, signal_row(
                row, value, None, label="source_retrieved",
                gold_in_top_k=source.get("gold_in_top_k"),
            ))
    meta = artifact_meta(
        cfg, seed, stage="signals_oracle", written=n, deterministic=True,
        oracle_correct="token F1 correct. Joined correct applies abstention labels and adjudication on top of this bit.",
        oracle_retrieval="1 when every source passage is in the generator top-k",
    )
    write_json(results / "signals" / "oracle_meta.json", meta)
    print(f"oracle: wrote {n} new correct rows")
    return 0


def _logistic(results: Path, cfg: dict[str, Any], seed: int, names: list[str], rows: list[dict]) -> int:
    lexical = _signal_index(results, "lexical")
    nli = _signal_index(results, "nli")
    questions = _question_rows(results, names)
    prepared = []
    for row in rows:
        key = (row["qid"], row["arm"])
        if key not in lexical or key not in nli:
            raise ProtocolError(
                f"logistic: missing lexical or nli for {key}. Run those stages first."
            )
        question = questions[row["qid"]]
        prepared.append({
            **row,
            "lexical": lexical[key]["value"],
            "nli": nli[key]["value"],
            "correct": score_answer(
                row["answer"], question["gold_answers"], bool(question["unanswerable"]),
            )["correct"],
        })
    path = stage_jsonl(results / "signals", "logistic")
    wanted = {(row["qid"], row["arm"]) for row in prepared}
    if _refuse_partial(path, wanted):
        print("logistic: file already matches this log")
        return 0
    fraction = _rate(cfg, "held_out_fraction")
    block = _sample_block([row["qid"] for row in prepared], fraction, seed, "holdout")
    written, fit = logistic_rows(prepared, holdout_ids=set(block["sampled_qids"]))
    _write_rows(path, written)
    meta = artifact_meta(
        cfg, seed, stage="signals_logistic", written=len(written),
        holdout=block, fit=fit, label="token_f1_correct",
    )
    write_json(results / "signals" / "logistic_meta.json", meta)
    n_scored = sum(1 for row in written if row["fold"] == "holdout" and row["value"] is not None)
    print(f"logistic: wrote {len(written)} rows, {n_scored} holdout scores")
    return 0


def _signal_index(results: Path, stem: str) -> dict[tuple, dict[str, Any]]:
    path = stage_jsonl(results / "signals", stem)
    if not path.is_file():
        raise ProtocolError(f"logistic needs {path.name}. Run the {stem} stage first.")
    out = {}
    for row in read_jsonl(path):
        out[(row["qid"], row["arm"])] = row
    return out


def _judge(args: argparse.Namespace, cfg: dict[str, Any], seed: int, names: list[str], rows: list[dict]) -> int:
    judge_model = args.judge_model_id or cfg["judge_model_id"]
    inference_mode = args.inference_mode or cfg["inference_mode"]
    price_for(cfg, judge_model, "batch" if inference_mode == "batch" else "on_demand")
    gen_ids = set()
    allowed = sample_qids_by_dataset(args.results, names)
    for name in names:
        for row in read_jsonl(args.results / "generation" / f"{name}.jsonl"):
            if not in_sample(row, allowed):
                continue
            validate_rows([row], ("model_id",), args.results / "generation" / f"{name}.jsonl")
            gen_ids.add(row["model_id"])
    if len(gen_ids) != 1:
        raise ProtocolError(f"generation rows use more than one model id: {sorted(gen_ids)}")
    check_judge_model(judge_model, next(iter(gen_ids)), args.allow_same_judge)
    rate = _rate(cfg, "judge_sample_rate")
    block = _sample_block([row["qid"] for row in rows], rate, seed, "judge")
    sampled = set(block["sampled_qids"])
    path = stage_jsonl(args.results / "signals", "judge")
    if path.is_file():
        for existing in read_jsonl(path):
            if "judge_sampled" not in existing:
                continue
            expected = existing["qid"] in sampled
            if bool(existing["judge_sampled"]) != expected:
                raise ProtocolError(
                    f"judge sample for {existing['qid']} changed under seed {seed}. "
                    "Move the judge file aside and rerun."
                )
    if args.dry_run:
        n_calls = sum(1 for row in rows if row["qid"] in sampled and (row["qid"], row["arm"]) not in done_keys(path, ("qid", "arm")))
        print(f"judge sample rate {rate}: {len(sampled)} question ids, {n_calls} arm calls")
        print("dry-run does not estimate judge tokens: pass --max-usd on the real run")
        meta = artifact_meta(
            cfg, seed, stage="signals_judge_dry_run", judge_model_id=judge_model,
            allow_same_judge=bool(args.allow_same_judge), sample=block,
            n_calls=n_calls, prompt=JUDGE_PROMPT, called_model=False,
            inference_mode=inference_mode,
        )
        write_json(args.results / "signals" / "judge_dry_run.json", meta)
        return 0
    if args.max_usd is None:
        raise ProtocolError("--max-usd is required for judge calls")
    total_cap = float(cfg["total_usd_cap"]) if args.total_usd_cap is None else args.total_usd_cap
    budget = Budget(args.results, "judge", args.max_usd, total_cap)
    if inference_mode == "batch":
        s3, bedrock = make_batch_clients(cfg["region"])
        summary = judge_rows_batch(
            cfg, rows, path, budget, model_id=judge_model, s3=s3, bedrock=bedrock,
            seed=seed, sampled=sampled, sleep=time.sleep,
        )
        n = summary["written"]
        stop_reason = summary["stop_reason"]
        pending = summary["pending"]
    else:
        n, stop_reason, pending, code = _judge_on_demand(
            cfg, rows, path, budget, model_id=judge_model, sampled=sampled, max_usd_pricing="on_demand",
        )
        if code:
            return code
    pending_path = args.results / "signals" / "judge_pending.json"
    if pending:
        write_json(pending_path, {"reason": SPEND_CAP_REASON, "detail": stop_reason, "rows": pending})
        print(f"spend cap: kept {n} judge rows, {len(pending)} left pending")
        print(stop_reason)
    elif pending_path.is_file():
        pending_path.unlink()
    if not stop_reason:
        clear_marker(args.results)
    meta = artifact_meta(
        cfg, seed, stage="signals_judge", judge_model_id=judge_model,
        allow_same_judge=bool(args.allow_same_judge), generator_model_id=next(iter(gen_ids)),
        prompt=JUDGE_PROMPT, sample=block, written=n, total_usd_cap=total_cap,
        max_usd=args.max_usd, stopped=bool(stop_reason), stop_reason=stop_reason,
        n_pending=len(pending), ledger_usd=budget.global_spent, inference_mode=inference_mode,
    )
    write_json(args.results / "signals" / "judge_meta.json", meta)
    print(f"judge: wrote {n} rows ({len(sampled)} sampled qids)")
    return 0


def _judge_on_demand(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
    budget: Budget,
    *,
    model_id: str,
    sampled: set[str],
    max_usd_pricing: str,
) -> tuple[int, str | None, list[dict], int]:
    client = make_client(cfg["region"])
    done = done_keys(path, ("qid", "arm"))
    max_out = int(cfg["judge_max_output_tokens"])
    n = 0
    stop_reason = None
    pending: list[dict] = []
    for index, row in enumerate(rows):
        key = (row["qid"], row["arm"])
        if key in done:
            continue
        built = build_judge_prompt(row["question"], row["answer"], row["passages"])
        if row["qid"] not in sampled:
            _append_judge(
                path, budget, row, None, "not_sampled", "",
                in_tok=0, out_tok=0, usage_observed=False, error=None,
                model_id=model_id, cfg=cfg, pricing=max_usd_pricing, sampled_flag=False,
                evidence_truncated=False, n_evidence_chars=int(built["n_evidence_chars"]),
                record_ledger=False,
            )
            done.add(key)
            n += 1
            continue
        if built["prompt"] is None:
            _append_judge(
                path, budget, row, None, built["reason"], "",
                in_tok=0, out_tok=0, usage_observed=False, error=None,
                model_id=model_id, cfg=cfg, pricing=max_usd_pricing, sampled_flag=True,
                evidence_truncated=False, n_evidence_chars=0, record_ledger=False,
            )
            done.add(key)
            n += 1
            continue
        est = cost_usd(cfg, model_id, estimate_tokens(built["prompt"]), max_out, max_usd_pricing)
        blocked = budget.blocking_reason(est)
        if blocked:
            stop_reason = blocked
            for item in rows[index:]:
                if (item["qid"], item["arm"]) not in done:
                    pending.append({"qid": item["qid"], "arm": item["arm"], "dataset": item["dataset"]})
            break
        try:
            value, reason, raw, in_tok, out_tok = judge_one(
                client, model_id, row["question"], row["answer"], row["passages"],
                max_out, sleep=time.sleep,
            )
        except JudgeAccessDenied as exc:
            write_marker(budget.results, str(exc))
            print(exc, file=sys.stderr)
            return n, None, [], 2
        _append_judge(
            path, budget, row, value, reason, raw,
            in_tok=in_tok, out_tok=out_tok, usage_observed=reason != "validation_exception", error=None,
            model_id=model_id, cfg=cfg, pricing=max_usd_pricing, sampled_flag=True,
            evidence_truncated=bool(built["evidence_truncated"]),
            n_evidence_chars=int(built["n_evidence_chars"]),
            record_ledger=True,
        )
        done.add(key)
        n += 1
        if budget.global_spent > budget.total_usd_cap + 1e-12 or budget.stage_spent > budget.max_usd + 1e-12:
            stop_reason = (
                f"{SPEND_CAP_REASON}: a call under the pre-call estimate crossed the cap "
                f"(ledger ${budget.global_spent:.6f}, total cap ${budget.total_usd_cap:.6f})"
            )
            for item in rows[index + 1:]:
                if (item["qid"], item["arm"]) not in done:
                    pending.append({"qid": item["qid"], "arm": item["arm"], "dataset": item["dataset"]})
            break
    return n, stop_reason, pending, 0


if __name__ == "__main__":
    raise SystemExit(main())
