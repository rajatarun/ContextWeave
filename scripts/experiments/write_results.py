#!/usr/bin/env python3
"""Write FINDINGS.md and PENDING.md strictly from results/ artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import RESULTS, ProtocolError, load_config
from experiments.reporting import render, write_assumptions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    try:
        write_assumptions(args.results / "simulation_assumptions.json", cfg, seed)
        findings, pending = render(args.results)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "FINDINGS.md").write_text(findings)
    (args.results / "PENDING.md").write_text(pending)
    print(f"wrote {args.results / 'FINDINGS.md'}")
    print(f"wrote {args.results / 'PENDING.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
