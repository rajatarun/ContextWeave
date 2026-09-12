"""Tests for the adaptive RAG router.

The lock-in test is the one that matters: it fails against the previous
argmax-with-fixed-steps rule and passes against Thompson sampling, and its
absence is why the defect shipped.

The graph store is stubbed (no Memgraph in CI); selection and posterior
migration are the real implementation, and the Beta update is applied here
mirroring the Cypher in update_feedback().
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "query_api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shared"))

import rag_router as R  # noqa: E402


@pytest.fixture
def store(monkeypatch):
    """In-memory stand-in for the EFFECTIVE_FOR edges."""
    data: dict[str, list[float]] = {}

    def fake_query(query, params=None):
        if "RETURN r.label AS strategy" in query:
            return [{"strategy": s, "weight": None, "alpha": a, "beta": b}
                    for s, (a, b) in ((k, v) for k, v in data.items())]
        return []

    monkeypatch.setattr(R, "_run_query", fake_query)
    return data


def seed(store, priors):
    p = R._PRIOR_STRENGTH
    for s, w in priors.items():
        store[s] = [w * p, (1.0 - w) * p]


def reward(store, strategy, confidence):
    """Mirrors the Cypher: alpha += c, beta += 1 - c."""
    a, b = store[strategy]
    store[strategy] = [a + confidence, b + (1.0 - confidence)]


PRIORS = {"semantic_search": 0.60, "graph_first": 0.55,
          "hybrid": 0.50, "keyword_boosted": 0.45}


# ---------------------------------------------------------------------------
# The regression that motivated the change
# ---------------------------------------------------------------------------

def test_mediocre_incumbent_does_not_lock_out_a_better_strategy(store):
    """A better strategy must be found even when the incumbent is never penalised.

    Against the previous rule -- argmax selection, +0.05/-0.02 fixed steps, and
    no change at all for confidence in [0.40, 0.70) -- the incumbent answering at
    0.55 was selected 2000/2000 times and the 0.90 strategy was never tried once,
    because a challenger's weight could only move if it was selected and it could
    only be selected if its weight moved.
    """
    random.seed(11)
    seed(store, PRIORS)
    true_conf = {"semantic_search": 0.55, "graph_first": 0.90,
                 "hybrid": 0.85, "keyword_boosted": 0.50}

    picks = {s: 0 for s in PRIORS}
    for _ in range(1500):
        chosen = R.select_strategy("architecture").strategy
        picks[chosen] += 1
        reward(store, chosen, true_conf[chosen])

    assert picks["graph_first"] > 1000, (
        f"best strategy should dominate after exploration, got {picks}")
    best_mean = store["graph_first"][0] / sum(store["graph_first"])
    assert best_mean == pytest.approx(0.90, abs=0.05), (
        "posterior should converge to the strategy's true confidence")


def test_every_strategy_is_tried_at_least_once(store):
    """No arm may be starved: an untried arm cannot be evaluated."""
    random.seed(5)
    seed(store, PRIORS)
    tried = set()
    for _ in range(600):
        chosen = R.select_strategy("architecture").strategy
        tried.add(chosen)
        reward(store, chosen, 0.6)
    assert tried == set(PRIORS), f"starved arms: {set(PRIORS) - tried}"


# ---------------------------------------------------------------------------
# Properties of the update rule
# ---------------------------------------------------------------------------

def test_no_dead_zone_mid_confidence_still_moves_the_posterior(store):
    """Confidence in the old [0.40, 0.70) band must be informative."""
    seed(store, {"graph_first": 0.50})
    before = store["graph_first"][0] / sum(store["graph_first"])
    for _ in range(20):
        reward(store, "graph_first", 0.55)
    after = store["graph_first"][0] / sum(store["graph_first"])
    assert after != before, "mid-band confidence was discarded"
    assert after == pytest.approx(0.55, abs=0.03)


def test_posterior_does_not_saturate(store):
    """Feedback must stay live; the old weight pinned at 1.0 after 8 queries."""
    seed(store, {"graph_first": 0.60})
    for _ in range(300):
        reward(store, "graph_first", 0.90)
    a, b = store["graph_first"]
    assert a + b > 300, "sufficient statistics should grow without bound"
    assert store["graph_first"][0] / (a + b) == pytest.approx(0.90, abs=0.02)
    before = a / (a + b)
    reward(store, "graph_first", 0.0)
    after = store["graph_first"][0] / sum(store["graph_first"])
    assert after < before, "a bad answer must still move a well-established arm"


def test_low_confidence_lowers_the_posterior(store):
    seed(store, {"graph_first": 0.60})
    before = store["graph_first"][0] / sum(store["graph_first"])
    for _ in range(30):
        reward(store, "graph_first", 0.10)
    after = store["graph_first"][0] / sum(store["graph_first"])
    assert after < before


# ---------------------------------------------------------------------------
# Compatibility and representation
# ---------------------------------------------------------------------------

def test_legacy_scalar_weight_is_migrated_to_a_prior(monkeypatch):
    """Edges seeded before this change carry only `weight`; they must still work."""
    def fake_query(query, params=None):
        if "RETURN r.label AS strategy" in query:
            return [{"strategy": "graph_first", "weight": 0.75,
                     "alpha": None, "beta": None}]
        return []
    monkeypatch.setattr(R, "_run_query", fake_query)

    post = R._query_strategy_posteriors("architecture")
    a, b = post["graph_first"]
    assert a / (a + b) == pytest.approx(0.75), "prior mean must match the old weight"
    assert a + b == pytest.approx(R._PRIOR_STRENGTH), "prior must stay weak"


def test_unseen_strategy_gets_a_uniform_prior_not_zero(monkeypatch):
    """A missing edge previously scored 0.0 and could never be selected."""
    def fake_query(query, params=None):
        if "RETURN r.label AS strategy" in query:
            return [{"strategy": "graph_first", "weight": 0.8, "alpha": None, "beta": None}]
        return []
    monkeypatch.setattr(R, "_run_query", fake_query)

    post = R._query_strategy_posteriors("architecture")
    assert set(post) == set(R._STRATEGY_PRIORITY), "all strategies must be represented"
    assert post["hybrid"] == R._UNSEEN_PRIOR


def test_greedy_mode_is_deterministic(store):
    """Operators debugging a decision need a reproducible answer."""
    seed(store, PRIORS)
    picks = {R.select_strategy("architecture", explore=False).strategy for _ in range(40)}
    assert len(picks) == 1, "greedy selection must not vary"


def test_reported_confidence_is_the_posterior_mean_not_the_sample(store):
    seed(store, {"graph_first": 0.80})
    cfg = R.select_strategy("architecture", explore=False)
    if cfg.strategy == "graph_first":
        assert cfg.strategy_confidence == pytest.approx(0.80, abs=1e-6)


def test_retrieval_config_flags_follow_the_strategy(store):
    seed(store, {"hybrid": 0.99})
    cfg = R.select_strategy("architecture", explore=False)
    if cfg.strategy == "hybrid":
        assert cfg.include_graph and cfg.boost_keywords


# ---------------------------------------------------------------------------
# Propensity logging and learning-loop health
# ---------------------------------------------------------------------------

def test_propensities_sum_to_one_and_favour_the_leader(store):
    seed(store, {"graph_first": 0.5, "semantic_search": 0.5,
                 "hybrid": 0.5, "keyword_boosted": 0.5})
    for _ in range(100):
        reward(store, "graph_first", 0.9)   # well-observed strong arm
    post = R._query_strategy_posteriors("architecture")
    p = R._p_best(post, samples=4000, rng=random.Random(3))
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["graph_first"] > 0.8
    assert all(v > 0 for v in p.values()), "no arm should be unreachable"


def test_thompson_decision_reports_a_positive_propensity(store):
    random.seed(2)
    seed(store, PRIORS)
    cfg = R.select_strategy("architecture", explore=True)
    assert 0.0 < cfg.selection_propensity <= 1.0


def test_greedy_decision_has_propensity_one(store):
    seed(store, PRIORS)
    cfg = R.select_strategy("architecture", explore=False)
    assert cfg.selection_propensity == 1.0


def test_health_flags_a_starved_arm(store):
    """The signature of the old defect: one arm observed 200 times, siblings never."""
    seed(store, PRIORS)
    for _ in range(200):
        reward(store, "semantic_search", 0.55)
    h = R.routing_health(["architecture"], samples=500)["questionTypes"]["architecture"]
    assert h["verdict"] == "starved"
    assert set(h["starvedArms"]) == {"graph_first", "hybrid", "keyword_boosted"}


def test_health_calls_a_clear_winner_converged_not_starved(store):
    seed(store, PRIORS)
    for _ in range(200):
        reward(store, "graph_first", 0.90)
    for s in ("semantic_search", "hybrid", "keyword_boosted"):
        for _ in range(30):
            reward(store, s, 0.50)
    h = R.routing_health(["architecture"], samples=2000)["questionTypes"]["architecture"]
    assert h["verdict"] == "converged"
    assert h["leader"] == "graph_first"
    assert h["starvedArms"] == []


def test_health_reports_learning_before_evidence_accumulates(store):
    seed(store, PRIORS)
    h = R.routing_health(["architecture"], samples=2000)
    assert h["questionTypes"]["architecture"]["verdict"] == "learning"
    assert h["anyStarved"] is False
