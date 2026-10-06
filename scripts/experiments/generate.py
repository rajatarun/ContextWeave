#!/usr/bin/env python3
"""Generate answers for every (question, arm), or estimate the cost with --dry-run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import DATASETS, RESULTS, ProtocolError, artifact_meta, load_config, write_json
from experiments.generate_stage import (
    dry_run, generate_rows, load_retrieval, make_client, system_prompt, write_dry_run_artifact,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.region:
        cfg = {**cfg, "region": args.region}
    seed = cfg["seed"] if args.seed is None else args.seed
    model_id = args.model_id or cfg["generator_model_id"]
    names = [p.strip() for p in args.datasets.split(",") if p.strip()]
    try:
        rows = []
        for name in names:
            rows.extend(load_retrieval(args.results / "retrieval" / f"{name}.jsonl"))
        if args.dry_run:
            estimate = dry_run(cfg, rows, model_id)
            path = args.results / "generation" / "dry_run_cost.json"
            write_dry_run_artifact(cfg, seed, estimate, path)
            print(f"calls: {estimate['n_calls']}")
            print(f"input tokens (estimate): {estimate['input_tokens_estimate']}")
            print(f"output tokens (upper bound): {estimate['output_tokens_upper_bound']}")
            print(f"USD upper bound: {estimate['usd_upper_bound']:.6f}")
            print(f"estimator: {estimate['estimator']}")
            print(f"wrote {path}")
            return 0
        if args.max_usd is None:
            raise ProtocolError("--max-usd is required for a real generation run. Use --dry-run to estimate first.")
        client = make_client(cfg["region"])
        total_written = 0
        for name in names:
            subset = [r for r in rows if r["dataset"] == name]
            summary = generate_rows(
                cfg, subset,
                args.results / "generation" / f"{name}.jsonl",
                args.results / "generation" / "cost_log.jsonl",
                model_id=model_id, max_usd=args.max_usd, client=client,
            )
            total_written += summary["written"]
            print(f"{name}: wrote {summary['written']} skipped {summary['skipped']} spent ${summary['spent_usd']:.6f}")
        meta = artifact_meta(
            cfg, seed, stage="generate", model_id=model_id,
            prompts={"generator_system": system_prompt()}, called_model=True,
        )
        write_json(args.results / "generation" / "generation_meta.json", meta)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
