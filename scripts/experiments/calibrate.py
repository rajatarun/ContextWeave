#!/usr/bin/env python3
"""Calibration harness over logged signals. Refuses to invent a signal that was not written."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.calibrate_stage import SIGNALS, calibrate_all
from experiments.common import DATASETS, RESULTS, ProtocolError, artifact_meta, load_config, write_json
from experiments.records import load_joined


def _signal_frame(rows: list[dict], signal: str) -> list[dict] | None:
    frame = []
    for row in rows:
        if signal == "self":
            value = row["self_confidence"] if row["self_status"] == "ok" else None
            reason = None if value is not None else row["self_status"]
        else:
            reason = row.get(f"{signal}_reason")
            if reason == "signal_file_missing":
                return None
            value = row.get(signal)
        frame.append({
            "qid": row["qid"], "arm": row["arm"], "f1": row["f1"],
            "correct": row["correct"], "value": value, "signal": signal, "reason": reason,
        })
    return frame


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
        gen = args.results / "generation" / f"{names[0]}.jsonl"
        if not gen.is_file():
            raise ProtocolError(f"generation output not found: {gen}")
        joined = load_joined(args.results, cfg, names)
        present = []
        for signal in SIGNALS:
            if _signal_frame(joined, signal) is None:
                print(f"{signal}: signal file missing, left out of this artifact")
                continue
            present.append(signal)
        by_dataset: dict[str, list[dict]] = defaultdict(list)
        q_dataset = {r["qid"]: r["dataset"] for r in joined}
        for signal in present:
            frame = _signal_frame(joined, signal)
            assert frame is not None
            for row in frame:
                row["dataset"] = q_dataset[row["qid"]]
                by_dataset[row["dataset"]].append(row)
        if "self" not in present:
            raise ProtocolError("generation rows did not yield a self signal")
        result = calibrate_all(
            by_dataset, seed, n_boot, float(cfg["high_coverage"]), float(cfg["low_auroc_max"]),
        )
        body = artifact_meta(
            cfg, seed, stage="calibration", bootstrap_samples=n_boot,
            signals_present=present, **result,
        )
        path = args.results / "calibration" / "calibration.json"
        write_json(path, body)
        print(f"prediction: {result['prediction']['verdict']}")
        print(f"wrote {path}")
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
