"""Three checks that reuse the logged signals, plus the slope estimates.

(a) Assumption check. For each signal, arm, and binary outcome Y, estimate
    E[R | Y, arm, R observed] with a cluster-bootstrap interval. Arm-dependent
    offsets are the pattern behind the confounding result: the same Y carrying
    a different expected reward on different arms.

(b) Missingness. Per arm and per signal, the rate m_a at which the signal is
    unobserved, and the reason counts. Accuracy (mean binary correctness) on
    rounds where the signal is missing and on rounds where it is observed.

(c) The normalized-self definition used by replay is recorded here so the
    scale-objection baseline has one written definition. The replay itself
    computes the values.

Slopes c0 = E[R | Y=0, R observed], c1 = E[R | Y=1, R observed], s = c1 - c0,
per dataset and per signal, with bootstrap intervals. The simulation's assumed
slopes are written beside them by the caller.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Sequence

from experiments.metrics import cluster_bootstrap, conditional_mean
from experiments.replay_stage import NORMALIZED_SELF_DEFINITION

SIGNALS = ("self", "lexical_grounding", "nli_grounding", "judge")


def _signal_rows(rows: Sequence[dict[str, Any]], signal: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        value = row.get(signal)
        reason = row.get(f"{signal}_reason")
        if signal == "self":
            value = row["self_confidence"] if row["self_status"] == "ok" else None
            reason = None if value is not None else row["self_status"]
        out.append({
            "qid": row["qid"],
            "arm": row["arm"],
            "correct": int(row["correct"]),
            "value": value,
            "reason": reason,
        })
    return out


def _slope_block(rows: Sequence[dict[str, Any]], rng: random.Random, n_boot: int) -> dict[str, Any]:
    def c0(sample: Sequence[dict[str, Any]]) -> float | None:
        return conditional_mean(sample, 0)

    def c1(sample: Sequence[dict[str, Any]]) -> float | None:
        return conditional_mean(sample, 1)

    def slope(sample: Sequence[dict[str, Any]]) -> float | None:
        a, b = c0(sample), c1(sample)
        if a is None or b is None:
            return None
        return b - a

    return {
        "c0": cluster_bootstrap(rows, c0, random.Random(rng.randrange(1 << 30)), n_boot),
        "c1": cluster_bootstrap(rows, c1, random.Random(rng.randrange(1 << 30)), n_boot),
        "s": cluster_bootstrap(rows, slope, random.Random(rng.randrange(1 << 30)), n_boot),
        "n_y0": sum(1 for r in rows if r["value"] is not None and r["correct"] == 0),
        "n_y1": sum(1 for r in rows if r["value"] is not None and r["correct"] == 1),
    }


def _arm_conditional(rows: Sequence[dict[str, Any]], rng: random.Random, n_boot: int) -> dict[str, Any]:
    arms = sorted({r["arm"] for r in rows})
    out: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        out[arm] = {}
        for y in (0, 1):
            def stat(sample: Sequence[dict[str, Any]], y: int = y) -> float | None:
                return conditional_mean([r for r in sample if r["arm"] == arm], y)
            out[arm][f"y{y}"] = cluster_bootstrap(arm_rows, stat, random.Random(rng.randrange(1 << 30)), n_boot)
            out[arm][f"n{y}"] = sum(1 for r in arm_rows if r["value"] is not None and r["correct"] == y)
    # Non-overlapping intervals at the same Y are the observable form of an arm offset.
    flags = []
    for y in (0, 1):
        intervals = []
        for arm in arms:
            block = out[arm][f"y{y}"]
            if block["lo"] is None or block["hi"] is None:
                continue
            intervals.append((arm, block["lo"], block["hi"]))
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                a, alo, ahi = intervals[i]
                b, blo, bhi = intervals[j]
                if ahi < blo or bhi < alo:
                    flags.append({"y": y, "arms": [a, b]})
    return {"by_arm": out, "nonoverlapping_pairs": flags}


def _missingness(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    out = {}
    for arm, group in sorted(by_arm.items()):
        n = len(group)
        missing = [r for r in group if r["value"] is None]
        observed = [r for r in group if r["value"] is not None]
        def acc(xs: list[dict[str, Any]]) -> float | None:
            if not xs:
                return None
            return sum(r["correct"] for r in xs) / len(xs)
        out[arm] = {
            "n": n,
            "n_missing": len(missing),
            "m": (len(missing) / n) if n else None,
            "reasons": dict(Counter(r["reason"] or "observed" for r in group)),
            "accuracy_when_missing": acc(missing),
            "accuracy_when_observed": acc(observed),
        }
    return out


def analyse(rows: Sequence[dict[str, Any]], seed: int, n_boot: int) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    rng = random.Random(seed)
    datasets: dict[str, Any] = {}
    for name, group in sorted(by_dataset.items()):
        signals: dict[str, Any] = {}
        for signal in SIGNALS:
            srows = _signal_rows(group, signal)
            sub = random.Random(rng.randrange(1 << 30))
            signals[signal] = {
                "slope": _slope_block(srows, sub, n_boot),
                "assumption": _arm_conditional(srows, sub, n_boot),
                "missingness": _missingness(srows),
            }
        datasets[name] = signals
    return {
        "datasets": datasets,
        "normalized_self_definition": NORMALIZED_SELF_DEFINITION,
    }
