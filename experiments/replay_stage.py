"""Replay the deployed Thompson-sampling router on logged rewards.

Selection is ``rag_router.select_strategy`` (one Beta draw per arm, tie-break
by the deployed priority). The update is the deployed fractional step
``alpha += R``, ``beta += 1 - R``. When ``R`` is missing the posterior is left
alone. The Bernoulli-trick variant draws ``B ~ Bernoulli(R)`` and applies the
same update to ``B``.

All four arms were logged, so the replay is on-policy with respect to the
rewards: the policy only updates the arm it selects, and the reward it sees
for that arm is the logged one. Question order is a seeded permutation.
Posteriors are per (strategy, question type), started from the deployed
``general`` prior in ``models.ROUTING_PRIORS`` scaled by ``ROUTER_PRIOR_STRENGTH``,
because these QA types have no seeded edge of their own.

Regret:
* ``pseudo_regret`` sums ``mu*(type) - mu(selected, type)`` where ``mu`` is the
  arm's mean binary correctness on that question type over the whole log.
* ``realized_regret`` sums ``max_arm Y - Y_selected`` on that question.

``normalized_self`` is a causal per-arm rank. At the moment arm ``a`` is
selected for question type ``q``, let ``H`` be the self-confidences observed
on earlier selections of ``(q, a)`` in this replay (other arms, and later
questions, are not used). The reward is the fraction of ``H`` strictly below
the current value, plus half the ties, divided by ``|H|``. The first selection
of an arm has an empty ``H`` and the reward is missing, so the posterior is
not updated. The value is appended to ``H`` after the reward is computed.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))
sys.path.insert(0, str(_ROOT / "src" / "shared"))

import rag_router as R  # noqa: E402
from models import RAGStrategyLabel, ROUTING_PRIORS  # noqa: E402

from experiments.common import ARMS

REWARDS = (
    "self", "self_with_fallbacks", "lexical_grounding", "verified",
    "verified_plus_self", "normalized_self", "oracle",
)
UPDATES = ("fractional", "bernoulli")

NORMALIZED_SELF_DEFINITION = (
    "Causal per-arm rank of verbalized self-confidence. History H is the "
    "self-confidences seen on earlier selections of this (question type, arm) "
    "in this replay only. Reward = (|{h in H: h < c}| + 0.5 |{h in H: h = c}|) / |H|. "
    "Empty H yields a missing reward and no posterior update. Logged confidences "
    "from arms that were not selected are not used."
)


def general_priors() -> dict[str, float]:
    return {arm: float(ROUTING_PRIORS[(RAGStrategyLabel(arm), "general")]) for arm in ARMS}


def empirical_mu(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """question_type -> arm -> mean binary correctness."""
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[row["question_type"]][row["arm"]].append(int(row["correct"]))
    out: dict[str, dict[str, float]] = {}
    for qt, arms in buckets.items():
        out[qt] = {arm: sum(v) / len(v) for arm, v in arms.items()}
    return out


class _Router:
    def __init__(self, seed: int, question_types: Sequence[str], priors: dict[str, float], strength: float):
        self._prop = R._PROPENSITY_SAMPLES
        R._PROPENSITY_SAMPLES = 0
        self._py_random = random.getstate()
        self.rng_bern = random.Random(seed + 1_000_003)
        self.store = {
            qt: {arm: [priors[arm] * strength, (1.0 - priors[arm]) * strength] for arm in ARMS}
            for qt in question_types
        }
        self._prev = R._run_query
        R._run_query = self._query  # type: ignore[assignment]
        random.seed(seed)

    def _query(self, query: str, params: dict | None = None):
        if "RETURN r.label AS strategy" in query:
            qt = (params or {})["question_type"]
            return [
                {"strategy": arm, "weight": None, "alpha": ab[0], "beta": ab[1]}
                for arm, ab in self.store[qt].items()
            ]
        return []

    def select(self, question_type: str) -> str:
        return R.select_strategy(question_type, explore=True).strategy

    def update(self, question_type: str, arm: str, reward: float | None, mode: str) -> float | None:
        if reward is None:
            return None
        applied = float(reward)
        if mode == "bernoulli":
            applied = 1.0 if self.rng_bern.random() < applied else 0.0
        elif mode != "fractional":
            raise ValueError(mode)
        a, b = self.store[question_type][arm]
        self.store[question_type][arm] = [a + applied, b + (1.0 - applied)]
        return applied

    def close(self) -> None:
        R._run_query = self._prev
        R._PROPENSITY_SAMPLES = self._prop
        random.setstate(self._py_random)


def _best_arms(mu: dict[str, float]) -> set[str]:
    if not mu:
        return set()
    top = max(mu.values())
    return {arm for arm, m in mu.items() if m == top}


def normalized_self_reward(history: list[float], value: float | None) -> float | None:
    if value is None or not history:
        return None
    less = sum(1 for h in history if h < value)
    ties = sum(1 for h in history if h == value)
    return (less + 0.5 * ties) / len(history)


def run_one(
    questions: Sequence[dict[str, Any]],
    by_qid: dict[str, dict[str, dict[str, Any]]],
    reward_name: str,
    update: str,
    seed: int,
    mu: dict[str, dict[str, float]],
    priors: dict[str, float],
    strength: float,
) -> dict[str, Any]:
    order_rng = random.Random(seed)
    order = list(questions)
    order_rng.shuffle(order)
    qtypes = sorted({q["question_type"] for q in questions})
    router = _Router(seed, qtypes, priors, strength)
    history: dict[tuple[str, str], list[float]] = defaultdict(list)
    pseudo = 0.0
    realized = 0.0
    best_hits = 0
    curve = []
    try:
        for t, q in enumerate(order, start=1):
            qt = q["question_type"]
            arm = router.select(qt)
            logged = by_qid[q["qid"]][arm]
            if reward_name == "normalized_self":
                key = (qt, arm)
                reward = normalized_self_reward(history[key], logged["rewards"]["self"])
                if logged["rewards"]["self"] is not None:
                    history[key].append(float(logged["rewards"]["self"]))
            else:
                reward = logged["rewards"][reward_name]
            applied = router.update(qt, arm, reward, update)
            mu_t = mu[qt]
            best = max(mu_t.values())
            pseudo += best - mu_t[arm]
            ys = {a: int(by_qid[q["qid"]][a]["correct"]) for a in by_qid[q["qid"]]}
            realized += max(ys.values()) - ys[arm]
            if arm in _best_arms(mu_t):
                best_hits += 1
            curve.append({
                "t": t,
                "cumulative_pseudo_regret": pseudo,
                "cumulative_realized_regret": realized,
                "selected_arm": arm,
                "best_arm_share": best_hits / t,
                "applied_reward": applied,
            })
    finally:
        router.close()
    return {
        "pseudo_regret": pseudo,
        "realized_regret": realized,
        "best_arm_share": best_hits / len(order) if order else None,
        "curve": curve,
        "n": len(order),
    }
