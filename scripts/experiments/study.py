#!/usr/bin/env python3
"""Information ratios, Thompson streams, judge coverage, drift, and assumption checks.

Reads the joined log. Subset, agreement, judge, and Nova files are optional:
a missing one is a null block with a reason. Defaults are ``v2.replay_seeds``
(500), ``v2.replay_rounds`` (10000), ``v2.judge_coverage``, and
``v2.drift_discounts``. Overrides are logged on the artifact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (
    DATASETS, RESULTS, ProtocolError, artifact_meta, load_config, read_json, read_jsonl,
    write_json,
)
from experiments.study_stage import (
    DEFINITIONS, assumption_checks, configured_study, judge_coverage_weighted,
    ratios_by_dataset, run_streams, score_second_generator, seed_list,
)
from experiments.records import load_joined
from experiments.subset_stage import expected_keys


def _floats(text: str | None, fallback: list[float]) -> list[float]:
    if text is None:
        return list(fallback)
    return [float(part) for part in text.split(",") if part.strip()]


def _optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


def _questions(results: Path, names: list[str]) -> dict[str, dict]:
    out = {}
    for name in names:
        for row in read_jsonl(results / "samples" / f"{name}.jsonl"):
            out[str(row["qid"])] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", type=int, default=None, help="count of seeds: seed, seed+1, ...")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--discounts", default=None, help="comma-separated drift discounts")
    ap.add_argument("--coverage", default=None, help="comma-separated judge coverage rates")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        seed = cfg["seed"] if args.seed is None else args.seed
        names = [part.strip() for part in args.datasets.split(",") if part.strip()]
        settings = configured_study(cfg)
        n_seeds = settings["replay_seeds"] if args.seeds is None else args.seeds
        rounds = settings["replay_rounds"] if args.rounds is None else args.rounds
        discounts = _floats(args.discounts, settings["drift_discounts"])
        coverage = _floats(args.coverage, settings["judge_coverage"])
        n_boot = cfg["bootstrap_samples"] if args.bootstrap is None else args.bootstrap
        seeds = seed_list(seed, int(n_seeds))
        rows = load_joined(args.results, cfg, names)
        information = ratios_by_dataset(rows)
        streams = run_streams(rows, seeds, int(rounds), discounts)
        coverage_body = judge_coverage_weighted(rows, coverage, seed, cfg)
        assumptions = assumption_checks(rows, seed, int(n_boot))

        subset = _optional_json(args.results / "subset" / "subset.json")
        agreement = _optional_json(args.results / "subset" / "agreement.json")
        if agreement is None:
            agreement_block = {"reason": "results/subset/agreement.json is missing", "datasets": None}
        else:
            agreement_block = {
                "reason": None,
                "source": "results/subset/agreement.json",
                "partial": agreement.get("partial"),
                "datasets": agreement.get("datasets"),
                "definition": agreement.get("definition"),
            }
        nova_path = args.results / "subset" / "nova" / "generations.jsonl"
        if subset is None:
            nova_block = {"reason": "results/subset/subset.json is missing", "datasets": None}
        else:
            try:
                nova_rows = read_jsonl(nova_path)
            except ProtocolError:
                nova_block = {"reason": f"{nova_path} is missing", "datasets": None}
            else:
                questions = _questions(args.results, names)
                keys = [
                    key for key in expected_keys(subset)
                    if key[0] in names
                ]
                nova_block = score_second_generator(nova_rows, questions, keys)

        body = artifact_meta(
            cfg, seed, stage="study",
            definitions=DEFINITIONS,
            overrides={
                "seeds": args.seeds,
                "rounds": args.rounds,
                "bootstrap": args.bootstrap,
                "discounts": args.discounts,
                "coverage": args.coverage,
            },
            seed_start=seed,
            seed_count=len(seeds),
            seeds=seeds,
            rounds=int(rounds),
            information=information,
            thompson=streams,
            judge_coverage=coverage_body,
            assumption=assumptions,
            agreement=agreement_block,
            nova=nova_block,
        )
        path = args.results / "study" / "study.json"
        write_json(path, body)
        print(f"seeds: {len(seeds)} rounds: {rounds}")
        print(f"wrote {path}")
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
