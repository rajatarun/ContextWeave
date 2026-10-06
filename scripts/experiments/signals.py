#!/usr/bin/env python3
"""Compute lexical grounding, NLI grounding, or the hash-sampled judge.

One signal per invocation. Rows already stored for (qid, arm) are skipped.
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
from experiments.generate_stage import make_client
from experiments.judge_access import clear_marker, write_marker
from experiments.ledger import SPEND_CAP_REASON, Budget
from experiments.records import load_joined
from experiments.signals_stage import (
    JUDGE_PROMPT, JudgeAccessDenied, check_judge_model, judge_one, judge_sampled,
    lexical_value, load_nli, nli_value, signal_row,
)


def _bases(results: Path, cfg: dict[str, Any], names: list[str]) -> list[dict]:
    # Local import type: records.load_joined requires generation. For lexical
    # and nli that is correct. Re-read here so a missing generation file errors.
    return [r for r in load_joined(results, cfg, names, attach_signals=False)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("signal", choices=("lexical", "nli", "judge"))
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
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    names = [p.strip() for p in args.datasets.split(",") if p.strip()]
    try:
        rows = _bases(args.results, cfg, names)
        threshold = float(cfg["grounding_threshold"])
        weight = float(cfg["grounding_weight"])
        if args.signal == "lexical":
            path = stage_jsonl(args.results / "signals", "lexical")
            done = done_keys(path, ("qid", "arm"))
            n = 0
            for row in rows:
                if (row["qid"], row["arm"]) in done:
                    continue
                value, reason, detail = lexical_value(row["answer"], row["passages"], threshold, weight)
                append_jsonl(path, signal_row(row, value, reason, detail=detail, threshold=threshold))
                n += 1
            meta = artifact_meta(cfg, seed, stage="signals_lexical", written=n, tau=threshold)
            write_json(args.results / "signals" / "lexical_meta.json", meta)
            print(f"lexical: wrote {n} rows")
            return 0
        if args.signal == "nli":
            verifier, info = load_nli(cfg["nli_model"])
            path = stage_jsonl(args.results / "signals", "nli")
            done = done_keys(path, ("qid", "arm"))
            n = 0
            for row in rows:
                if (row["qid"], row["arm"]) in done:
                    continue
                value, reason, detail = nli_value(row["answer"], row["passages"], verifier, threshold, weight)
                append_jsonl(path, signal_row(row, value, reason, detail=detail, threshold=threshold, nli_model=info["nli_model"]))
                n += 1
            meta = artifact_meta(cfg, seed, stage="signals_nli", written=n, nli=info, tau=threshold)
            write_json(args.results / "signals" / "nli_meta.json", meta)
            print(f"nli: wrote {n} rows")
            return 0
        judge_model = args.judge_model_id or cfg["judge_model_id"]
        price_for(cfg, judge_model)
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
        rate = float(cfg["judge_sample_rate"])
        sampled = {r["qid"] for r in rows if judge_sampled(r["qid"], rate)}
        if args.dry_run:
            n_calls = sum(1 for r in rows if r["qid"] in sampled)
            print(f"judge sample rate {rate}: {len(sampled)} question ids, {n_calls} arm calls")
            print("dry-run does not estimate judge tokens: the prompt includes the answer, and this flag does not call the model")
            meta = artifact_meta(
                cfg, seed, stage="signals_judge_dry_run", judge_model_id=judge_model,
                allow_same_judge=bool(args.allow_same_judge), n_qids_sampled=len(sampled),
                n_calls=n_calls, prompt=JUDGE_PROMPT, called_model=False,
            )
            write_json(args.results / "signals" / "judge_dry_run.json", meta)
            return 0
        if args.max_usd is None:
            raise ProtocolError("--max-usd is required for judge calls")
        total_cap = float(cfg["total_usd_cap"]) if args.total_usd_cap is None else args.total_usd_cap
        budget = Budget(args.results, "judge", args.max_usd, total_cap)
        client = make_client(cfg["region"])
        path = stage_jsonl(args.results / "signals", "judge")
        done = done_keys(path, ("qid", "arm"))
        max_out = int(cfg["judge_max_output_tokens"])
        n = 0
        stop_reason = None
        pending: list[dict] = []
        for index, row in enumerate(rows):
            if (row["qid"], row["arm"]) in done:
                continue
            if row["qid"] not in sampled:
                append_jsonl(path, signal_row(row, None, "not_sampled", judge_sampled=False, judge_model_id=judge_model))
                done.add((row["qid"], row["arm"]))
                n += 1
                continue
            evidence = "\n\n".join(row["passages"])[:12000]
            prompt = JUDGE_PROMPT.format(question=row["question"], evidence=evidence, answer=row["answer"])
            est = cost_usd(cfg, judge_model, estimate_tokens(prompt), max_out)
            blocked = budget.blocking_reason(est)
            if blocked:
                stop_reason = blocked
                for item in rows[index:]:
                    if (item["qid"], item["arm"]) not in done:
                        pending.append({"qid": item["qid"], "arm": item["arm"], "dataset": item["dataset"]})
                break
            try:
                value, reason, raw, in_tok, out_tok = judge_one(
                    client, judge_model, row["question"], row["answer"], row["passages"],
                    max_out, sleep=time.sleep,
                )
            except JudgeAccessDenied as exc:
                write_marker(args.results, str(exc))
                print(exc, file=sys.stderr)
                return 2
            append_jsonl(path, signal_row(
                row, value, reason, judge_sampled=True, judge_model_id=judge_model, raw_response=raw,
            ))
            done.add((row["qid"], row["arm"]))
            usd = cost_usd(cfg, judge_model, in_tok, out_tok)
            budget.record({
                "qid": row["qid"], "arm": row["arm"], "dataset": row["dataset"],
                "usd": usd, "input_tokens": in_tok, "output_tokens": out_tok,
                "model_id": judge_model,
            })
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
            prompt=JUDGE_PROMPT, sample_rate=rate, hash="sha256(qid) first 8 hex / 2**32 < rate",
            written=n, total_usd_cap=total_cap, max_usd=args.max_usd,
            stopped=bool(stop_reason), stop_reason=stop_reason, n_pending=len(pending),
            ledger_usd=budget.global_spent,
        )
        write_json(args.results / "signals" / "judge_meta.json", meta)
        print(f"judge: wrote {n} rows ({len(sampled)} sampled qids)")
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
