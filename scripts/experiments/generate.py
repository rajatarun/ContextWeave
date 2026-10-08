#!/usr/bin/env python3
"""Generate answers for every (question, arm), or estimate the cost with --dry-run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (
    DATASETS, RESULTS, ProtocolError, artifact_meta, done_keys, in_sample, load_config,
    sample_qids_by_dataset, stage_jsonl, write_json,
)
from experiments.generate_stage import (
    dry_run, generate_rows, generate_rows_batch, load_retrieval, make_batch_clients,
    make_client, system_prompt, write_dry_run_artifact,
)
from experiments.ledger import SPEND_CAP_REASON, budget_for


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--total-usd-cap", type=float, default=None,
                    help="shared generation+judge ceiling (default: config total_usd_cap)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inference-mode", choices=("on_demand", "batch"), default=None,
                    help="on_demand calls Converse per row; batch submits a Bedrock job")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.region:
        cfg = {**cfg, "region": args.region}
    seed = cfg["seed"] if args.seed is None else args.seed
    model_id = args.model_id or cfg["generator_model_id"]
    inference_mode = args.inference_mode or cfg["inference_mode"]
    cfg = {**cfg, "inference_mode": inference_mode}
    names = [p.strip() for p in args.datasets.split(",") if p.strip()]
    try:
        allowed = sample_qids_by_dataset(args.results, names)
        rows = []
        for name in names:
            loaded = load_retrieval(args.results / "retrieval" / f"{name}.jsonl")
            kept = [row for row in loaded if in_sample(row, allowed)]
            left_out = len(loaded) - len(kept)
            if left_out:
                print(f"{name}: left {left_out} retrieval rows outside the sample", flush=True)
            rows.extend(kept)
        if args.dry_run:
            estimate = dry_run(cfg, rows, model_id, results=args.results)
            path = args.results / "generation" / "dry_run_cost.json"
            write_dry_run_artifact(cfg, seed, estimate, path)
            print(f"calls: {estimate['n_calls']}")
            print(f"input tokens (estimate): {estimate['input_tokens_estimate']}")
            print(f"output tokens (expected): {estimate['output_tokens_estimate']}")
            print(f"USD (expected): {estimate['usd_upper_bound']:.6f}")
            print(estimate["output_policy"])
            print(f"estimator: {estimate['estimator']}")
            print(f"wrote {path}")
            return 0
        if args.max_usd is None:
            raise ProtocolError("--max-usd is required for a real generation run. Use --dry-run to estimate first.")
        total_cap = float(cfg["total_usd_cap"]) if args.total_usd_cap is None else args.total_usd_cap
        budget = budget_for(args.results, "generate", args.max_usd, cfg, args.total_usd_cap)
        client = None
        s3 = bedrock = None
        if inference_mode == "batch":
            s3, bedrock = make_batch_clients(cfg["region"])
        else:
            client = make_client(cfg["region"])
        pending: list[dict] = []
        stop_reason = None
        total_written = 0
        for i, name in enumerate(names):
            subset = [r for r in rows if r["dataset"] == name]
            out = stage_jsonl(args.results / "generation", name)
            if inference_mode == "batch":
                summary = generate_rows_batch(
                    cfg, subset, out, budget, model_id=model_id, s3=s3, bedrock=bedrock,
                    seed=seed,
                )
            else:
                summary = generate_rows(
                    cfg, subset, out, budget, model_id=model_id, client=client,
                )
            total_written += summary["written"]
            print(
                f"{name}: wrote {summary['written']} skipped {summary['skipped']} "
                f"cap total ${summary['spent_usd']:.6f}"
            )
            if summary["stopped"]:
                stop_reason = summary["stop_reason"]
                pending.extend(summary["pending"])
                for later in names[i + 1:]:
                    already = done_keys(stage_jsonl(args.results / "generation", later), ("qid", "arm"))
                    pending.extend(
                        {"qid": r["qid"], "arm": r["arm"], "dataset": r["dataset"]}
                        for r in rows
                        if r["dataset"] == later and (r["qid"], r["arm"]) not in already
                    )
                break
        pending_path = args.results / "generation" / "pending.json"
        if pending:
            write_json(pending_path, {"reason": SPEND_CAP_REASON, "detail": stop_reason, "rows": pending})
            print(f"spend cap: kept {total_written} rows, {len(pending)} left pending")
            print(stop_reason)
        elif pending_path.is_file():
            pending_path.unlink()
        meta = artifact_meta(
            cfg, seed, stage="generate", model_id=model_id, inference_mode=inference_mode,
            prompts={"generator_system": system_prompt()}, called_model=True,
            total_usd_cap=total_cap, max_usd=args.max_usd,
            stopped=bool(stop_reason), stop_reason=stop_reason, n_pending=len(pending),
        )
        write_json(args.results / "generation" / "generation_meta.json", meta)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
