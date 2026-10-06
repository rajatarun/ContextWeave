#!/usr/bin/env python3
"""Download SQuAD 2.0, HotpotQA, and the MRQA Natural Questions dev file, and write a seeded sample.

The sample is the first ``--n-per-dataset`` questions after sorting by qid
and shuffling with ``random.Random(seed)``. A smaller n with the same seed
is that prefix. The seed is stored on the sample manifest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import DATASETS, RESULTS, ProtocolError, load_config
from experiments.data import build_sample


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-per-dataset", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    n = cfg["n_per_dataset"] if args.n_per_dataset is None else args.n_per_dataset
    names = tuple(p.strip() for p in args.datasets.split(",") if p.strip())
    try:
        manifest = build_sample(cfg, args.results, n, seed, names)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"wrote {args.results / 'samples' / 'sample_manifest.json'}")
    for name, count in manifest["counts"].items():
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
