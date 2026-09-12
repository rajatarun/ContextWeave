"""Off-policy evaluation of routing policies from the decision log.

The estimator is only as good as the logged propensity, so the central test
generates a log with the *deployed* Thompson router (its own propensity
estimate included) and checks that reweighting recovers each arm's known
true reward. The log is the one artefact the router produces that lets a
policy be judged without being deployed; these tests are what make that
claim more than an assertion.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import routing_offpolicy_eval as O  # noqa: E402

TRUTH = {"graph_first": 0.90, "hybrid": 0.85, "semantic_search": 0.55, "keyword_boosted": 0.50}


@pytest.fixture(scope="module")
def log():
    return O.simulate_log(TRUTH, n=2500, seed=7)


def test_log_is_evaluable_and_propensities_are_probabilities(log):
    O.check_evaluable(log)
    assert all(0.0 < r.propensity <= 1.0 for r in log)
    assert any(r.propensity < 1.0 for r in log), "Thompson decisions must not all be certain"


def test_snips_recovers_the_true_value_of_each_fixed_policy(log):
    """Reweighting a Thompson log must estimate what 'always X' would have scored."""
    for arm, truth in TRUTH.items():
        _, pi = O.parse_policy(f"fixed:{arm}")
        e = O.evaluate(log, pi, reps=200, seed=1)
        assert e.matched > 0, f"{arm} never selected; cannot be evaluated"
        # Rarely-chosen arms have few rows and wider error; tolerance reflects that.
        tol = 0.04 if e.matched > 200 else 0.12
        assert abs(e.snips - truth) < tol, (arm, e.snips, truth, e.matched)
        assert e.snips_ci[0] <= truth + 0.02 and e.snips_ci[1] >= truth - 0.02, (arm, e.snips_ci, truth)


def test_snips_separates_good_arms_from_poor_ones(log):
    """The two arms the router settled on must be ranked above the two it abandoned.

    Ordering *within* the abandoned pair (0.55 vs 0.50) is not required: they
    were each selected only a handful of times, and an estimate from four rows
    cannot resolve a gap of 0.05. Plain IPW is worse still there -- its weight
    normalisation assumes coverage the log does not have -- which is why the
    tool tells operators to read SNIPS first.
    """
    ests = {}
    for arm in TRUTH:
        _, pi = O.parse_policy(f"fixed:{arm}")
        ests[arm] = O.evaluate(log, pi, reps=50).snips
    assert ests["graph_first"] > ests["hybrid"]
    assert min(ests["graph_first"], ests["hybrid"]) > max(ests["semantic_search"], ests["keyword_boosted"])


def test_logging_policy_value_is_the_mean_reward(log):
    e = O.evaluate(log, None, reps=50)
    assert e.ipw == pytest.approx(sum(r.reward for r in log) / len(log))
    assert e.ess == pytest.approx(len(log))


def test_greedy_log_is_refused():
    """Every propensity 1.0 means no counterfactual information; say so, do not guess."""
    rows = [O.Record("architecture", "graph_first", 1.0, 0.8) for _ in range(50)]
    with pytest.raises(SystemExit, match="greedy"):
        O.check_evaluable(rows)


def test_map_policy_undefined_for_a_type_contributes_nothing():
    _, pi = O.parse_policy("map:architecture=graph_first")
    rows = [O.Record("architecture", "graph_first", 0.5, 0.9),
            O.Record("project", "hybrid", 0.5, 0.2)]
    e = O.evaluate(rows, pi, reps=10)
    assert e.matched == 1
    assert e.dm is None or e.dm == pytest.approx(0.45)  # only the defined type counts


def test_direct_method_is_none_when_arm_unseen_for_type():
    _, pi = O.parse_policy("fixed:keyword_boosted")
    rows = [O.Record("architecture", "graph_first", 0.5, 0.9)]
    assert O.direct_method(rows, pi) is None


def test_weight_clipping_bounds_max_weight():
    rows = [O.Record("architecture", "hybrid", 0.01, 0.9),
            O.Record("architecture", "graph_first", 0.99, 0.9)]
    _, pi = O.parse_policy("fixed:hybrid")
    assert O.evaluate(rows, pi, reps=5).max_weight == pytest.approx(100.0)
    assert O.evaluate(rows, pi, clip=10.0, reps=5).max_weight == pytest.approx(10.0)
