"""Per-dataset, per-signal calibration with cluster-bootstrap intervals.

Metrics are coverage, Brier, 10-bin ECE, AUROC (F1 >= 0.5), Spearman with F1,
both strategy rankings, and Kendall tau between them. Bootstrap resamples
question ids and recomputes each scalar. Ranking agreement itself is reported
as the point estimate; its interval is the interval on Kendall tau.

The judge value is the stored grounding score. It is paired with token F1
as recorded. An abstention the judge scored 1.0 is left at 1.0 when
correctness is 0.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from experiments.metrics import cluster_bootstrap, signal_metrics
from experiments.prediction import dataset_verdict, overall_verdict

SIGNALS = ("self", "lexical_grounding", "nli_grounding", "judge")


def _pick(stat_name: str):
    def fn(rows: Sequence[dict[str, Any]]) -> float | None:
        val = signal_metrics(rows).get(stat_name)
        if isinstance(val, bool) or val is None:
            return None
        return float(val)
    return fn


def calibrate_signal(rows: Sequence[dict[str, Any]], rng: random.Random, n_boot: int) -> dict[str, Any]:
    point = signal_metrics(rows)
    intervals = {}
    for name in ("coverage", "brier", "ece", "auroc", "spearman", "kendall_tau"):
        # Independent streams so adding a metric does not move the others' intervals.
        sub = random.Random(rng.randrange(1 << 30))
        intervals[name] = cluster_bootstrap(rows, _pick(name), sub, n_boot)
    point["ci"] = intervals
    return point


def calibrate_dataset(rows: Sequence[dict[str, Any]], seed: int, n_boot: int,
                      high_coverage: float, low_auroc_max: float) -> dict[str, Any]:
    by_signal: dict[str, list[dict[str, Any]]] = {s: [] for s in SIGNALS}
    for row in rows:
        by_signal[row["signal"]].append(row)
    rng = random.Random(seed)
    out = {}
    for signal in SIGNALS:
        group = by_signal[signal]
        if not group:
            out[signal] = None
            continue
        out[signal] = calibrate_signal(group, random.Random(rng.randrange(1 << 30)), n_boot)
    verdict = dataset_verdict(out.get("self"), out.get("lexical_grounding"), high_coverage, low_auroc_max)
    return {"signals": out, "prediction": verdict}


def calibrate_all(by_dataset: dict[str, list[dict[str, Any]]], seed: int, n_boot: int,
                  high_coverage: float, low_auroc_max: float) -> dict[str, Any]:
    per = {
        name: calibrate_dataset(rows, seed, n_boot, high_coverage, low_auroc_max)
        for name, rows in by_dataset.items()
    }
    return {
        "datasets": per,
        "prediction": overall_verdict({name: block["prediction"] for name, block in per.items()}),
    }
