#!/usr/bin/env python3
"""Print projected Bedrock spend for the configured n, before any model call.

``--n`` and ``--judge-rate`` override the config for this print. Pass
``--from-results`` to price the prompts in a sample and retrieval directory.
When that flag is omitted, files under ``--results`` are used if every dataset
has both, and otherwise the input estimate stays on the template. ``--ledger``
supplies output-token means. A ledger that is not on disk, or a model with
no numeric rows, leaves output tokens at ``expected_output_tokens`` and the
print says so. ``--band-fraction`` overrides
``v2.adjudication_band_fraction``. Pass ``--write`` to store
``results/cost_projection.json``. The file is a projection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import RESULTS, ProtocolError, artifact_meta, load_config, write_json
from experiments.cost_projection import format_projection, project


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="questions per dataset (default: config n_per_dataset)")
    ap.add_argument("--judge-rate", type=float, default=None, help="default: v2.judge_sample_rate")
    ap.add_argument(
        "--from-results", type=Path, default=None,
        help="sample and retrieval directory. Missing files stop the run.",
    )
    ap.add_argument(
        "--ledger", type=Path, default=None,
        help="prior cost ledger with output_tokens per model. Missing file uses expected_output_tokens.",
    )
    ap.add_argument(
        "--band-fraction", type=float, default=None,
        help="assumed share of judged rows priced for adjudication (default: config)",
    )
    ap.add_argument("--write", action="store_true", help="write results/cost_projection.json")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        seed = cfg["seed"] if args.seed is None else args.seed
        n = cfg["n_per_dataset"] if args.n is None else args.n
        rate = cfg["v2"]["judge_sample_rate"] if args.judge_rate is None else args.judge_rate
        if args.from_results is not None:
            results, require = args.from_results, True
        else:
            results, require = args.results, False
        body = project(
            cfg, int(n), float(rate),
            results=results, ledger_path=args.ledger, band_fraction=args.band_fraction,
            seed=int(seed), require_results=require,
        )
        text = format_projection(body)
        print(text, end="")
        if args.write:
            path = args.results / "cost_projection.json"
            write_json(path, artifact_meta(cfg, seed, stage="cost_projection", **body))
            print(f"wrote {path}")
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
