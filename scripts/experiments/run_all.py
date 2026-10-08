#!/usr/bin/env python3
"""Run every experiment stage in order.

The cost projection is printed first. ``--project-only`` stops there.
A real run needs ``--max-usd``. Without it the model stages are not started.
``--dry-run`` passes ``--dry-run`` through to the stages that call a model.
A nonzero exit from a stage stops the runner.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import DATASETS, RESULTS, ProtocolError, load_config
from experiments.cost_projection import format_projection, project
from experiments.runner import stage_commands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--judge-rate", type=float, default=None)
    ap.add_argument("--from-results", type=Path, default=None)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--band-fraction", type=float, default=None)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--total-usd-cap", type=float, default=None)
    ap.add_argument("--inference-mode", choices=("on_demand", "batch"), default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--project-only", action="store_true")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        seed = cfg["seed"] if args.seed is None else args.seed
        n = cfg["n_per_dataset"] if args.n is None else args.n
        rate = cfg["v2"]["judge_sample_rate"] if args.judge_rate is None else args.judge_rate
        cap = float(cfg["total_usd_cap"]) if args.total_usd_cap is None else float(args.total_usd_cap)
        already = float(cfg["already_spent_usd"])
        print(
            f"spend cap: total_usd_cap {cap:.2f}, already_spent_usd {already:.2f} "
            f"(AWS bill, prior ledgers), remaining {cap - already:.2f} before this results ledger"
        )
        print(f"inference_mode: {args.inference_mode or cfg['inference_mode']}")
        if args.from_results is not None:
            prompt_results, require = args.from_results, True
        else:
            prompt_results, require = args.results, False
        text = format_projection(project(
            cfg, int(n), float(rate),
            results=prompt_results, ledger_path=args.ledger, band_fraction=args.band_fraction,
            seed=int(seed), require_results=require,
        ))
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(text, end="")
    if args.project_only:
        return 0
    if args.max_usd is None and not args.dry_run:
        print(
            "Pass --max-usd to run the model stages. The projection above did not call a model.",
            file=sys.stderr,
        )
        return 2
    commands = stage_commands(
        results=args.results,
        seed=seed,
        max_usd=args.max_usd,
        total_usd_cap=args.total_usd_cap,
        inference_mode=args.inference_mode or cfg["inference_mode"],
        dry_run=args.dry_run,
        datasets=args.datasets,
    )
    for cmd in commands:
        print("+ " + " ".join(cmd), flush=True)
        code = subprocess.call(cmd)
        if code != 0:
            print(f"stage exited {code}: {' '.join(cmd)}", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
