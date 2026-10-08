#!/usr/bin/env python3
"""Adjudicate token F1 inside the closed band with gpt-oss-120b, on demand.

Rows already stored are skipped. The seed is logged. Prompts and raw replies
are stored on each row. ``--dry-run`` counts the band and prices an upper
bound. It does not call the model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.adjudicate_stage import (
    ADJUDICATOR_SYSTEM, AdjudicatorAccessDenied, adjudication_record, call_adjudicator,
    require_adjudicator, score_reply, user_prompt,
)
from experiments.common import (
    DATASETS, RESULTS, ProtocolError, append_jsonl, artifact_meta, cost_usd, done_keys,
    estimate_tokens, in_sample, load_config, read_jsonl, sample_qids_by_dataset,
    stage_jsonl, validate_rows, write_json,
)
from experiments.generate_stage import make_client
from experiments.labels import decide_correctness
from experiments.ledger import SPEND_CAP_REASON, budget_for, planned_output_tokens
from experiments.records import _pending_keys


def _band_rows(results: Path, cfg: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    low = float(cfg["v2"]["adjudication_f1_low"])
    high = float(cfg["v2"]["adjudication_f1_high"])
    allowed = sample_qids_by_dataset(results, names)
    questions = []
    for name in names:
        path = results / "samples" / f"{name}.jsonl"
        rows = [row for row in read_jsonl(path) if in_sample(row, allowed)]
        validate_rows(rows, ("qid", "question", "gold_answers", "unanswerable", "yes_no"), path)
        questions.extend(rows)
    gen_pending = _pending_keys(results / "generation" / "pending.json")
    retrieval = {}
    generation = {}
    for name in names:
        for row in read_jsonl(results / "retrieval" / f"{name}.jsonl"):
            if in_sample(row, allowed):
                retrieval[(row["qid"], row["arm"])] = row
        try:
            generated = read_jsonl(results / "generation" / f"{name}.jsonl")
        except ProtocolError:
            if not gen_pending:
                raise
            generated = []
        for row in generated:
            if in_sample(row, allowed):
                generation[(row["qid"], row["arm"])] = row
    arms = ("semantic_search", "graph_first", "keyword_boosted", "hybrid")
    chosen = []
    for question in questions:
        keys = [(question["qid"], arm) for arm in arms]
        if any(key not in retrieval for key in keys):
            missing = [key for key in keys if key not in retrieval]
            raise ProtocolError(f"missing retrieval for {missing[0]}")
        if any(key not in generation for key in keys):
            absent = [key for key in keys if key not in generation]
            if all(key in gen_pending for key in absent):
                continue
            raise ProtocolError(f"missing generation for {absent[0]}")
        for arm in arms:
            key = (question["qid"], arm)
            gen = generation[key]
            ret = retrieval[key]
            for flag in ("gold_in_top_k", "source_retrieved"):
                if flag not in ret:
                    raise ProtocolError(f"retrieval row {key} has no {flag}")
            decision = decide_correctness(
                answer=gen["answer"],
                gold_answers=question["gold_answers"],
                unanswerable=bool(question["unanswerable"]),
                self_status=gen["self_status"],
                source_retrieved=bool(ret["source_retrieved"]),
                gold_in_top_k=bool(ret["gold_in_top_k"]),
                yes_no=bool(question["yes_no"]),
                f1_low=low,
                f1_high=high,
                stored_label=gen.get("abstention_label"),
                stored_counts=gen.get("abstention_counts_correct"),
                check_stored="abstention_label" in gen,
            )
            if not decision["in_f1_band"]:
                continue
            chosen.append({
                "qid": question["qid"],
                "dataset": question["dataset"],
                "arm": arm,
                "question_type": question["question_type"],
                "question": question["question"],
                "answer": gen["answer"],
                "gold_answers": list(question["gold_answers"]),
                "f1": decision["f1"],
            })
    chosen.sort(key=lambda row: (row["dataset"], row["qid"], row["arm"]))
    return chosen


def _refuse_seed_mismatch(path: Path, seed: int) -> None:
    if not path.is_file():
        return
    for row in read_jsonl(path):
        if "seed" not in row:
            raise ProtocolError(f"{path.name} has a row with no seed. Move the file aside and rerun.")
        if int(row["seed"]) != int(seed):
            raise ProtocolError(
                f"{path.name} was written under seed {row['seed']}. "
                f"This run uses seed {seed}. Move the file aside and rerun."
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--total-usd-cap", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    try:
        model_id = require_adjudicator(cfg)
        rows = _band_rows(args.results, cfg, names)
        path = stage_jsonl(args.results / "adjudication", "adjudication")
        _refuse_seed_mismatch(path, seed)
        done = done_keys(path, ("qid", "arm"))
        pending_rows = [row for row in rows if (row["qid"], row["arm"]) not in done]
        max_out = int(cfg["adjudicator_max_output_tokens"])
        planned_out, planned_source = planned_output_tokens(cfg, model_id, results=args.results)
        if args.dry_run:
            usd = 0.0
            listed = []
            for row in pending_rows:
                prompt = user_prompt(row["question"], row["answer"], row["gold_answers"])
                usd += cost_usd(
                    cfg, model_id, estimate_tokens(ADJUDICATOR_SYSTEM) + estimate_tokens(prompt),
                    planned_out, "on_demand",
                )
                listed.append({
                    "qid": row["qid"], "arm": row["arm"], "dataset": row["dataset"], "f1": row["f1"],
                })
            meta = artifact_meta(
                cfg, seed, stage="adjudication_dry_run", model_id=model_id,
                system_prompt=ADJUDICATOR_SYSTEM, called_model=False, deterministic=True,
                temperature=0, n_calls=len(pending_rows), usd_upper_bound=usd,
                expected_output_tokens_per_call=planned_out,
                output_token_source=planned_source,
                estimator=(
                    f"ceil_utf8_bytes_div_4 plus {planned_out:g} output tokens ({planned_source}). "
                    f"adjudicator_max_output_tokens {max_out} is the request cap and is not priced."
                ),
                f1_low=float(cfg["v2"]["adjudication_f1_low"]),
                f1_high=float(cfg["v2"]["adjudication_f1_high"]),
                rows=listed,
            )
            out = args.results / "adjudication" / "dry_run.json"
            write_json(out, meta)
            print(f"adjudication dry-run: {len(pending_rows)} calls, upper bound ${usd:.6f}")
            print(f"wrote {out}")
            return 0
        if args.max_usd is None:
            raise ProtocolError("--max-usd is required for adjudication. Use --dry-run to estimate first.")
        total_cap = float(cfg["total_usd_cap"]) if args.total_usd_cap is None else args.total_usd_cap
        budget = budget_for(args.results, "adjudicate", args.max_usd, cfg, args.total_usd_cap)
        client = make_client(cfg["region"])
        n = 0
        n_unparsed = 0
        stop_reason = None
        pending: list[dict] = []
        for index, row in enumerate(pending_rows):
            prompt = user_prompt(row["question"], row["answer"], row["gold_answers"])
            planned_out, _source = planned_output_tokens(cfg, model_id, budget=budget)
            est = cost_usd(
                cfg, model_id, estimate_tokens(ADJUDICATOR_SYSTEM) + estimate_tokens(prompt),
                planned_out, "on_demand",
            )
            blocked = budget.blocking_reason(est)
            if blocked:
                stop_reason = blocked
                pending = [
                    {"qid": item["qid"], "arm": item["arm"], "dataset": item["dataset"]}
                    for item in pending_rows[index:]
                ]
                break
            try:
                raw, in_tok, out_tok, failure = call_adjudicator(
                    client, model_id, ADJUDICATOR_SYSTEM, prompt, max_out, sleep=time.sleep,
                )
            except AdjudicatorAccessDenied as exc:
                write_json(
                    args.results / "adjudication" / "adjudicator_unavailable.json",
                    {"reason": "adjudicator model access not yet granted", "detail": str(exc)},
                )
                print(exc, file=sys.stderr)
                return 2
            value, reason = score_reply(raw, failure)
            usd = cost_usd(cfg, model_id, in_tok, out_tok, "on_demand")
            record = adjudication_record(
                row, seed=seed, model_id=model_id, system=ADJUDICATOR_SYSTEM, user=prompt,
                raw=raw, value=value, reason=reason, in_tok=in_tok, out_tok=out_tok, usd=usd, cfg=cfg,
            )
            append_jsonl(path, record)
            budget.record({
                "qid": row["qid"], "arm": row["arm"], "dataset": row["dataset"],
                "usd": usd, "input_tokens": in_tok, "output_tokens": out_tok,
                "usage_observed": failure is None,
                "model_id": model_id, "pricing": "on_demand",
                "price_usd_per_million": record["price_usd_per_million"],
            })
            n += 1
            if reason:
                n_unparsed += 1
        pending_path = args.results / "adjudication" / "pending.json"
        if pending:
            write_json(pending_path, {"reason": SPEND_CAP_REASON, "detail": stop_reason, "rows": pending})
            print(f"spend cap: kept {n} adjudication rows, {len(pending)} left pending")
            print(stop_reason)
        elif pending_path.is_file():
            pending_path.unlink()
        meta = artifact_meta(
            cfg, seed, stage="adjudication", model_id=model_id,
            system_prompt=ADJUDICATOR_SYSTEM, called_model=True, deterministic=True,
            temperature=0, written=n, n_unparsed=n_unparsed, n_band=len(rows),
            f1_low=float(cfg["v2"]["adjudication_f1_low"]),
            f1_high=float(cfg["v2"]["adjudication_f1_high"]),
            stopped=bool(stop_reason), stop_reason=stop_reason, n_pending=len(pending),
            max_usd=args.max_usd, total_usd_cap=total_cap, ledger_usd=budget.global_spent,
        )
        write_json(args.results / "adjudication" / "adjudication_meta.json", meta)
        print(f"adjudication: wrote {n} rows ({len(rows)} in band)")
        if n_unparsed:
            print(
                f"{n_unparsed} adjudication replies were unparsed. "
                "Joined correctness will stop on those rows.",
                file=sys.stderr,
            )
            return 2
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
