"""
Adaptive RAG Router

Query-time agent that selects the best RAG retrieval strategy for a given
question type by consulting the Memgraph routing graph.

Decision logic:
  1. Query Memgraph for the EFFECTIVE_FOR posterior connecting RAGStrategy nodes
     to the current question type (stored as a DocumentType proxy).
  2. Draw one Thompson sample per strategy and pick the highest.
  3. Return a RetrievalConfig that configures the retriever accordingly.

Feedback loop:
  After each query, call update_feedback() to fold the synthesis confidence into
  that strategy's Beta posterior.

Why Thompson sampling rather than argmax over a scalar weight
-------------------------------------------------------------
The previous rule selected by ``argmax`` over a scalar weight and updated only
the *selected* strategy's weight by a fixed step (+0.05 above confidence 0.70,
-0.02 below 0.40, nothing in between). That is a bandit with no exploration, and
it could demote a bad incumbent but never promote a challenger, because a
challenger was never selected and so never evaluated. Simulating that rule over
2000 queries: with an incumbent whose confidence lands in the [0.40, 0.70) dead
zone -- 30% of the confidence range -- a strategy that would have answered at
confidence 0.90 was selected *0 times*, and no weight in the graph ever moved.
A router that has stopped learning and one that has converged emit identical
logs, so the failure is silent in production.

Four properties of this implementation fix that:

  * **Exploration.** Selection draws from each strategy's posterior, so an arm is
    tried in proportion to the probability that it is best. No epsilon schedule.
  * **No dead zone.** Confidence enters as a fractional Bernoulli reward
    (alpha += c, beta += 1 - c), so every observation is informative.
  * **No ceiling.** alpha and beta are unbounded sufficient statistics, so
    feedback never becomes a no-op the way a weight pinned at 1.0 did.
  * **"Never tried" is representable.** Beta(1,1) is distinguishable from a
    well-observed mediocre arm; a single scalar could not tell them apart.

The scalar ``weight`` is still maintained on the edge as the posterior mean, so
the routing policy remains readable as a small table and existing queries and
dashboards keep working.
"""
from __future__ import annotations

import logging
import os
import random
import sys

from models import RAGStrategyLabel, RetrievalConfig, ROUTING_PRIORS

logger = logging.getLogger(__name__)

# Strength of the seeded scalar weight when it is converted into a Beta prior.
# Low on purpose: it should express the operator's initial belief without taking
# many observations to overturn. At 4.0 a seeded weight of 0.6 becomes
# Beta(2.4, 1.6), which roughly ten real observations will dominate.
_PRIOR_STRENGTH = float(os.environ.get("ROUTER_PRIOR_STRENGTH", "4.0"))

# Posterior for a strategy that has no EFFECTIVE_FOR edge at all. Beta(1,1) is
# uniform: maximally uncertain, so Thompson sampling will occasionally try it.
# Under the old rule a missing edge scored 0.0 and could never be selected,
# which made "not yet seeded" indistinguishable from "known to be useless".
_UNSEEN_PRIOR = (1.0, 1.0)

# "thompson" (default) samples the posterior; "greedy" takes the posterior mean
# and is intended for reproducible tests and for operators debugging a decision.
_EXPLORATION = os.environ.get("ROUTER_EXPLORATION", "thompson").strip().lower()

# Monte Carlo draws used to estimate the propensity of the selected strategy.
# Measured (scripts/bench_router.py): 4 arms x 200 draws costs ~1.1 ms median
# in pure Python, against ~0.014 ms for the Thompson draw itself; 500 draws
# cost ~2.7 ms. 200 puts the standard error at <= 0.035 for a propensity near
# 0.5, which is adequate for off-policy weighting, and the whole selection
# step stays three orders of magnitude below the synthesis call it precedes.
_PROPENSITY_SAMPLES = int(os.environ.get("ROUTER_PROPENSITY_SAMPLES", "200"))

# Health thresholds. An arm is *starved* when it has essentially no observations
# while a sibling has many: that pattern is exactly what the old argmax rule
# produced, and Thompson sampling should make it impossible, so seeing it in
# production means exploration is off or feedback is not being written.
_STARVED_MAX_OBS = 3.0
_STARVED_SIBLING_MIN_OBS = 50.0
# Posterior probability that one arm is best above which we call a question
# type converged rather than still learning.
_CONVERGED_P_BEST = 0.95

QUESTION_TYPES: list[str] = [
    "skill_depth", "architecture", "project", "comparison", "credential", "general",
]

# Fallback priority when Memgraph is unavailable (or graph has no data yet)
_STRATEGY_PRIORITY: list[str] = [
    RAGStrategyLabel.GRAPH_FIRST,
    RAGStrategyLabel.HYBRID,
    RAGStrategyLabel.KEYWORD_BOOSTED,
    RAGStrategyLabel.SEMANTIC,
]


def _get_db_clients():
    """Lazily import db_clients from the shared module."""
    shared_dir = os.path.join(os.path.dirname(__file__), "..", "shared")
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    import importlib
    return importlib.import_module("db_clients")


def _run_query(query: str, parameters: dict | None = None) -> list[dict]:
    """Execute an openCypher query against Memgraph; returns [] on any error."""
    try:
        db = _get_db_clients()
        return db.run_graph_query(query, parameters)
    except Exception as exc:
        logger.warning("Memgraph routing query error: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy selection
# ─────────────────────────────────────────────────────────────────────────────

def _query_strategy_posteriors(question_type: str) -> dict[str, tuple[float, float]]:
    """Fetch each strategy's Beta posterior for the given question type.

    Edges seeded before this change carry only a scalar ``weight``; those are
    migrated on read into a weak Beta prior rather than requiring a migration
    pass, so an existing deployment keeps its operator-set priors.

    Returns a mapping of strategy label -> (alpha, beta). Strategies with no
    edge are returned with the uniform prior so they remain explorable.
    """
    query = """
    MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
    WHERE d.question_type = $question_type
    RETURN r.label AS strategy,
           e.weight AS weight,
           e.alpha  AS alpha,
           e.beta   AS beta
    """
    rows = _run_query(query, {"question_type": question_type})

    posteriors: dict[str, tuple[float, float]] = {}
    for row in rows:
        label = row.get("strategy")
        if label is None:
            continue
        alpha, beta = row.get("alpha"), row.get("beta")
        if alpha is None or beta is None:
            weight = row.get("weight")
            weight = 0.5 if weight is None else float(weight)
            weight = min(1.0, max(0.0, weight))
            alpha = weight * _PRIOR_STRENGTH
            beta = (1.0 - weight) * _PRIOR_STRENGTH
        # Beta is undefined at zero, so keep both shape parameters positive.
        posteriors[label] = (max(float(alpha), 1e-3), max(float(beta), 1e-3))

    for label in _STRATEGY_PRIORITY:
        posteriors.setdefault(label, _UNSEEN_PRIOR)
    return posteriors


def _posterior_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _posterior_sd(alpha: float, beta: float) -> float:
    n = alpha + beta
    return (alpha * beta / (n * n * (n + 1.0))) ** 0.5


def _p_best(
    posteriors: dict[str, tuple[float, float]],
    samples: int,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Monte Carlo estimate of P(arm has the largest posterior draw), per arm.

    Under Thompson sampling this *is* the selection probability, so the value
    for the chosen arm is its propensity. Ties are broken by _STRATEGY_PRIORITY
    exactly as select_strategy() does, so the estimate matches the policy.
    """
    draw = (rng or random).betavariate
    wins = {s: 0 for s in posteriors}
    # Iterate in priority order so that on an exact tie the first (highest
    # priority) arm keeps the win, matching max() over _STRATEGY_PRIORITY.
    order = {s: i for i, s in enumerate(_STRATEGY_PRIORITY)}
    arms = sorted(posteriors.items(), key=lambda kv: order.get(kv[0], len(order)))
    for _ in range(samples):
        best, best_v = None, -1.0
        for s, (a, b) in arms:
            v = draw(a, b)
            if v > best_v:
                best, best_v = s, v
        wins[best] += 1
    return {s: w / samples for s, w in wins.items()}


def select_strategy(
    question_type: str,
    graph_id: str | None = None,  # kept for API compatibility; unused (Memgraph uses env)
    explore: bool | None = None,
) -> RetrievalConfig:
    """
    Choose a RAG strategy for the given question type by Thompson sampling.

    One value is drawn from each strategy's Beta posterior and the largest wins,
    so a strategy is selected roughly in proportion to the probability that it is
    the best one -- which means a strategy that has never been tried still gets
    tried. Under the previous ``argmax`` rule it could not: only the incumbent
    was ever selected, only the selected arm was ever updated, and a challenger's
    weight therefore never moved.

    Args:
        question_type: Classified question type string (e.g. "architecture").
        graph_id:      Ignored (kept for backward compatibility).
        explore:       Force sampling on/off. Defaults to the ROUTER_EXPLORATION
                       environment setting; pass False for a reproducible
                       decision (posterior mean) when debugging.

    Returns:
        RetrievalConfig describing which retrieval approach to use.
    """
    posteriors = _query_strategy_posteriors(question_type)

    use_sampling = (_EXPLORATION == "thompson") if explore is None else bool(explore)
    if use_sampling:
        scores = {s: random.betavariate(a, b) for s, (a, b) in posteriors.items()}
    else:
        scores = {s: _posterior_mean(a, b) for s, (a, b) in posteriors.items()}

    # Ties break by the fixed priority order, as before.
    best_strategy = max(_STRATEGY_PRIORITY, key=lambda s: scores.get(s, 0.0))
    alpha, beta = posteriors[best_strategy]
    posterior_mean = _posterior_mean(alpha, beta)
    observations = alpha + beta - _PRIOR_STRENGTH

    # Propensity of the decision just made. Greedy is deterministic, so its
    # chosen arm has propensity 1 and every other arm 0; anything logged under
    # greedy therefore cannot be reweighted to evaluate another policy, which
    # is one more reason it is not the default.
    if use_sampling and _PROPENSITY_SAMPLES > 0:
        propensity = _p_best(posteriors, _PROPENSITY_SAMPLES).get(best_strategy, 0.0)
        # The arm was in fact selected, so its propensity cannot be 0; clamp so
        # an unlucky Monte Carlo estimate never produces an infinite IPW weight.
        propensity = max(propensity, 1.0 / _PROPENSITY_SAMPLES)
    else:
        propensity = 1.0

    logger.info(
        "Routing decision: strategy=%s mean=%.3f n~%.1f propensity=%.3f mode=%s "
        "question_type=%s posteriors=%s",
        best_strategy, posterior_mean, max(observations, 0.0), propensity,
        "thompson" if use_sampling else "greedy", question_type,
        {s: (round(a, 2), round(b, 2)) for s, (a, b) in posteriors.items()},
    )

    return RetrievalConfig(
        strategy=best_strategy,
        include_graph=(best_strategy in (RAGStrategyLabel.GRAPH_FIRST, RAGStrategyLabel.HYBRID)),
        boost_keywords=(best_strategy in (RAGStrategyLabel.KEYWORD_BOOSTED, RAGStrategyLabel.HYBRID)),
        use_neptune_chunks=(best_strategy == RAGStrategyLabel.HYBRID),
        # Report the posterior mean, not the sample: the sample drove exploration,
        # the mean is the actual estimate of how good this strategy is.
        strategy_confidence=posterior_mean,
        selection_propensity=propensity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health: is the router learning, converged, or stuck?
# ─────────────────────────────────────────────────────────────────────────────

def routing_health(
    question_types: list[str] | None = None,
    samples: int = 2000,
) -> dict:
    """Report, per question type, whether routing is learning, converged, or stuck.

    The defect this guards against was silent: a router that had stopped
    learning and one that had converged produced the same logs, because both
    showed one strategy selected every time. The two are distinguishable from
    the posteriors, not from the selections:

      converged  one arm is best with posterior probability >= 0.95 and the
                 others have actually been observed. Concentrated selection is
                 the correct behaviour here.
      learning   no arm is yet that clearly best; exploration is still paying.
      starved    an arm has essentially no observations while a sibling has
                 many. Thompson sampling cannot produce this pattern -- an
                 unobserved arm keeps a wide posterior and keeps being drawn --
                 so it means exploration is disabled or feedback is not being
                 written. This is the signature of the old argmax rule.

    The verdict is data an operator can alert on; the old scalar table could
    not express it.
    """
    report: dict[str, dict] = {}
    for qt in question_types or QUESTION_TYPES:
        posteriors = _query_strategy_posteriors(qt)
        p_best = _p_best(posteriors, samples)
        arms = {}
        for s, (a, b) in posteriors.items():
            n = max(a + b - _PRIOR_STRENGTH, 0.0) if (a, b) != _UNSEEN_PRIOR else 0.0
            arms[s] = {
                "alpha": round(a, 3), "beta": round(b, 3),
                "mean": round(_posterior_mean(a, b), 4),
                "sd": round(_posterior_sd(a, b), 4),
                "observations": round(n, 1),
                "pBest": round(p_best.get(s, 0.0), 4),
            }
        leader = max(arms, key=lambda s: arms[s]["pBest"])
        most_obs = max(v["observations"] for v in arms.values())
        starved = sorted(
            s for s, v in arms.items()
            if v["observations"] <= _STARVED_MAX_OBS and most_obs >= _STARVED_SIBLING_MIN_OBS
        )
        if starved:
            verdict = "starved"
        elif arms[leader]["pBest"] >= _CONVERGED_P_BEST:
            verdict = "converged"
        else:
            verdict = "learning"
        report[qt] = {
            "verdict": verdict,
            "leader": leader,
            "leaderPBest": arms[leader]["pBest"],
            "starvedArms": starved,
            "arms": arms,
        }
    return {
        "exploration": _EXPLORATION,
        "priorStrength": _PRIOR_STRENGTH,
        "questionTypes": report,
        "anyStarved": any(r["verdict"] == "starved" for r in report.values()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feedback / learning
# ─────────────────────────────────────────────────────────────────────────────

def update_feedback(
    strategy: str,
    question_type: str,
    confidence: float,
    graph_id: str | None = None,  # kept for API compatibility; unused
    weight: float = 1.0,
    source: str = "self",
) -> dict | None:
    """
    Fold a reward into this strategy's Beta posterior.

    The reward is treated as a fractional Bernoulli observation::

        alpha += weight * c
        beta  += weight * (1 - c)

    so every observation moves the posterior. ``source`` is ``"self"`` for
    the synthesiser's own confidence (the default, weight 1) and ``"human"``
    for a rating submitted against the answer (see feedback.py), which is
    counted separately on the edge as ``human_feedback_count`` so the two
    reward sources can be compared. Returns the updated posterior, or None if
    no edge was found. The previous rule applied a fixed
    +0.05 above 0.70 and -0.02 below 0.40 and *nothing at all* in between; that
    dead band covered 30% of the confidence range, and an incumbent sitting in it
    was locked in permanently because its weight never changed and no challenger
    was ever selected to earn one.

    alpha and beta are unbounded, so feedback never saturates the way a weight
    clipped at 1.0 did -- which previously turned the loop into a no-op after
    about eight confident answers.

    The scalar ``weight`` is rewritten as the posterior mean so the policy stays
    readable as a table and existing queries keep working.

    Args:
        strategy:      RAGStrategyLabel value of the strategy that was used.
        question_type: Question type that was answered.
        confidence:    Synthesis confidence score (0-1); clamped.
        graph_id:      Ignored (kept for backward compatibility).
    """
    reward = min(1.0, max(0.0, float(confidence)))
    weight = max(0.0, float(weight))
    is_human = 1 if source == "human" else 0

    query = """
    MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType)
    WHERE d.question_type = $question_type
    SET e.alpha = coalesce(e.alpha, coalesce(e.weight, 0.5) * $prior_strength)
                  + $w * $reward,
        e.beta  = coalesce(e.beta, (1.0 - coalesce(e.weight, 0.5)) * $prior_strength)
                  + $w * (1.0 - $reward),
        e.feedback_count = coalesce(e.feedback_count, 0) + 1,
        e.human_feedback_count = coalesce(e.human_feedback_count, 0) + $human
    SET e.weight = toFloat(e.alpha / (e.alpha + e.beta))
    RETURN e.alpha AS alpha, e.beta AS beta, e.weight AS weight,
           e.feedback_count AS n, e.human_feedback_count AS n_human
    """
    rows = _run_query(query, {
        "strategy":       strategy,
        "question_type":  question_type,
        "reward":         reward,
        "w":              weight,
        "human":          is_human,
        "prior_strength": _PRIOR_STRENGTH,
    })
    if rows:
        row = rows[0]
        logger.info(
            "Routing feedback applied: source=%s strategy=%s qt=%s reward=%.2f weight=%.1f "
            "-> Beta(%.2f, %.2f) mean=%.3f n=%s n_human=%s",
            source, strategy, question_type, reward, weight,
            row.get("alpha", 0.0), row.get("beta", 0.0),
            row.get("weight", 0.0), row.get("n", 0), row.get("n_human", 0),
        )
        return {
            "alpha": row.get("alpha"), "beta": row.get("beta"),
            "mean": row.get("weight"), "n": row.get("n"), "nHuman": row.get("n_human"),
        }
    logger.debug(
        "Routing feedback skipped (no EFFECTIVE_FOR edge found): strategy=%s qt=%s",
        strategy, question_type,
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Routing graph seed (called from ingestion_trigger on first deploy)
# ─────────────────────────────────────────────────────────────────────────────

def reset_routing_graph(graph_id: str = "") -> dict[str, int]:
    """
    Force-reset all EFFECTIVE_FOR edge weights back to their initial prior values,
    overwriting any weight drift caused by the feedback loop.

    Unlike seed_routing_graph (which uses ON CREATE SET and skips existing edges),
    this function unconditionally SET the weight on every edge.

    Use this after confirming ingestion is complete but confidence is still low
    due to weights being eroded before data was available.
    """
    edges_reset = 0

    for (strategy, question_type), weight in ROUTING_PRIORS.items():
        q = """
        MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType {question_type: $question_type})
        SET e.weight = $weight, e.feedback_count = 0, e.reset_at = timestamp()
        RETURN e.weight AS w
        """
        rows = _run_query(q, {
            "strategy":      strategy,
            "question_type": question_type,
            "weight":        weight,
        })
        if rows:
            edges_reset += 1
            logger.info("Reset weight: %s → %s = %.2f", strategy, question_type, weight)
        else:
            logger.warning("No edge found to reset: %s → %s", strategy, question_type)

    logger.info("Routing graph reset: %d edges restored to priors", edges_reset)
    return {"edges_reset": edges_reset}


def seed_routing_graph(graph_id: str = "") -> dict[str, int]:
    """
    Idempotently create RAGStrategy and DocumentType proxy nodes plus
    EFFECTIVE_FOR edges with initial prior weights in Memgraph.

    graph_id is accepted for backward compatibility but is not used;
    Memgraph connection is resolved via MEMGRAPH_SECRET_ARN / MEMGRAPH_HOST.

    Returns counts of nodes/edges written.
    """
    question_types = QUESTION_TYPES
    strategies = [s.value for s in RAGStrategyLabel]

    nodes_written = 0
    edges_written = 0

    # Upsert RAGStrategy nodes
    for strategy in strategies:
        q = """
        MERGE (r:RAGStrategy {label: $label})
        ON CREATE SET r.created = timestamp()
        RETURN r.label AS label
        """
        rows = _run_query(q, {"label": strategy})
        if rows:
            nodes_written += 1

    # Upsert DocumentType proxy nodes (one per question_type)
    for qt in question_types:
        q = """
        MERGE (d:DocumentType {question_type: $question_type})
        ON CREATE SET d.label = $question_type, d.created = timestamp()
        RETURN d.question_type AS qt
        """
        rows = _run_query(q, {"question_type": qt})
        if rows:
            nodes_written += 1

    # Upsert EFFECTIVE_FOR edges with prior weights
    for (strategy, question_type), weight in ROUTING_PRIORS.items():
        q = """
        MATCH (r:RAGStrategy {label: $strategy})
        MATCH (d:DocumentType {question_type: $question_type})
        MERGE (r)-[e:EFFECTIVE_FOR]->(d)
        ON CREATE SET e.weight = $weight, e.feedback_count = 0, e.seeded = true
        RETURN e.weight AS w
        """
        rows = _run_query(q, {
            "strategy":      strategy,
            "question_type": question_type,
            "weight":        weight,
        })
        if rows:
            edges_written += 1

    logger.info(
        "Routing graph seeded: %d nodes, %d edges",
        nodes_written, edges_written,
    )
    return {"nodes": nodes_written, "edges": edges_written}
