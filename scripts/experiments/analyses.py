#!/usr/bin/env python3
"""Slope estimates, the per-arm conditional check, and missingness rates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analyses_stage import analyse
from experiments.assumed import assumed_slopes
from experiments.common import DATASETS, RESULTS, ProtocolError, artifact_meta, load_config, write_json
from experiments.records import load_joined


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    n_boot = cfg["bootstrap_samples"] if args.bootstrap is None else args.bootstrap
    names = [p.strip() for p in args.datasets.split(",") if p.strip()]
    try:
        for name in names:
            for fname in ("lexical.jsonl", "nli.jsonl", "judge.jsonl"):
                path = args.results / "signals" / fname
                if not path.is_file():
                    raise ProtocolError(
                        f"analyses needs {path}. A missing signal is left pending rather than scored as all-missing."
                    )
        rows = load_joined(args.results, cfg, names)
        body = artifact_meta(
            cfg, seed, stage="analyses", bootstrap_samples=n_boot,
            assumed_simulation=assumed_slopes(),
            **analyse(rows, seed, n_boot),
        )
        path = args.results / "analyses" / "analyses.json"
        write_json(path, body)
        print(f"wrote {path}")
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
