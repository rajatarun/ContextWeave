"""Correctness and calibration metrics.

Token F1, abstention, Brier, 10-bin ECE, AUROC and Spearman are the functions
in ``scripts/verified_reward_bench.py``. Brier and ECE use that harness's
continuous correctness target (token F1, or 1/0 when the question is
unanswerable). AUROC binarises the same target at 0.5.
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src" / "query_api"))

import verified_reward as V  # noqa: E402
import verified_reward_bench as B  # noqa: E402

ABSTAIN_PATTERN = V._ABSTAIN_RE.pattern
F1_CORRECT_THRESHOLD = 0.5


def is_abstention(answer: str) -> bool:
    """True when the answer is empty after SQuAD normalisation, or matches the
    abstention phrases the claim splitter already drops (insufficient evidence,
    cannot answer, and the other phrases in ``verified_reward._ABSTAIN_RE``).
    """
    return B.is_abstention(answer)


def score_answer(answer: str, gold_answers: Sequence[str], unanswerable: bool) -> dict[str, Any]:
    """Return token-F1 correctness and the binary label.

    Unanswerable questions (SQuAD 2.0, gold list empty) are correct exactly
    when the answer abstains. Answerable questions are correct when token F1
    against the best gold answer is at least 0.5. An abstention on an
    answerable question scores 0.
    """
    gold = list(gold_answers)
    if unanswerable and gold:
        raise ValueError("unanswerable question carried gold answers")
    if unanswerable:
        gold = []
    f1 = B.correctness(answer, gold)
    return {
        "f1": f1,
        "correct": 1 if f1 >= F1_CORRECT_THRESHOLD else 0,
        "abstained": is_abstention(answer),
    }


def kendall_tau(ranking_a: Sequence[str], ranking_b: Sequence[str]) -> float | None:
    """Kendall tau-a on the shared items, best-first rankings."""
    items = [x for x in ranking_a if x in set(ranking_b)]
    if len(items) < 2:
        return None
    rank_b = {x: i for i, x in enumerate(ranking_b)}
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if rank_b[items[i]] < rank_b[items[j]]:
                conc += 1
            elif rank_b[items[i]] > rank_b[items[j]]:
                disc += 1
    total = conc + disc
    if total == 0:
        return None
    return (conc - disc) / total


def _mean(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def strategy_order(rows: Sequence[dict[str, Any]], value_key: str) -> list[str]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is None:
            continue
        buckets[row["arm"]].append(float(row[value_key]))
    return sorted(buckets, key=lambda arm: (-(sum(buckets[arm]) / len(buckets[arm])), arm))


def signal_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Harness metrics for one signal. ``rows`` have arm, f1, correct, value."""
    n = len(rows)
    observed = [r for r in rows if r.get("value") is not None]
    pairs = [(float(r["value"]), float(r["f1"])) for r in observed]
    truth = strategy_order(rows, "f1")
    sig = strategy_order(observed, "value")
    comparable = [a for a in truth if a in sig]
    sig_comp = [a for a in sig if a in comparable]
    return {
        "n": n,
        "n_observed": len(observed),
        "coverage": (len(observed) / n) if n else None,
        "brier": B.brier(pairs),
        "ece": B.ece(pairs),
        "auroc": B.auroc(pairs, threshold=F1_CORRECT_THRESHOLD),
        "spearman": B.spearman(pairs),
        "correctness_ranking": truth,
        "signal_ranking": sig,
        "kendall_tau": kendall_tau(sig_comp, comparable) if comparable else None,
        "rank_agrees": (sig_comp == comparable) if comparable else None,
        "strategy_mean_f1": {
            arm: _mean([float(r["f1"]) for r in rows if r["arm"] == arm])
            for arm in sorted({r["arm"] for r in rows})
        },
        "strategy_mean_signal": {
            arm: _mean([float(r["value"]) for r in observed if r["arm"] == arm])
            for arm in sorted({r["arm"] for r in observed})
        },
    }


def _percentile(sorted_stats: Sequence[float], p: float) -> float:
    if not sorted_stats:
        raise ValueError("empty bootstrap")
    idx = int(round(p * (len(sorted_stats) - 1)))
    idx = min(max(idx, 0), len(sorted_stats) - 1)
    return sorted_stats[idx]


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    stat_fn: Callable[[Sequence[dict[str, Any]]], float | None],
    rng: random.Random,
    n_boot: int,
    cluster_key: str = "qid",
) -> dict[str, Any]:
    """Resample whole questions, then recompute ``stat_fn``. Percentile 95% CI."""
    clusters: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row[cluster_key]].append(row)
    keys = list(clusters)
    point = stat_fn(rows)
    if not keys or n_boot <= 0:
        return {"estimate": point, "lo": None, "hi": None, "n_boot": 0, "n_clusters": len(keys)}
    stats: list[float] = []
    for _ in range(n_boot):
        chosen = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        sample = [rec for key in chosen for rec in clusters[key]]
        val = stat_fn(sample)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            stats.append(float(val))
    stats.sort()
    if not stats:
        return {"estimate": point, "lo": None, "hi": None, "n_boot": 0, "n_clusters": len(keys)}
    return {
        "estimate": point,
        "lo": _percentile(stats, 0.025),
        "hi": _percentile(stats, 0.975),
        "n_boot": n_boot,
        "n_clusters": len(keys),
    }


def conditional_mean(rows: Sequence[dict[str, Any]], y: int) -> float | None:
    xs = [float(r["value"]) for r in rows if r.get("value") is not None and int(r["correct"]) == y]
    return _mean(xs)
