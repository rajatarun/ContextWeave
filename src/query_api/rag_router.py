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

    logger.info(
        "Routing decision: strategy=%s mean=%.3f n~%.1f mode=%s question_type=%s "
        "posteriors=%s",
        best_strategy, posterior_mean, max(observations, 0.0),
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feedback / learning
# ─────────────────────────────────────────────────────────────────────────────

def update_feedback(
    strategy: str,
    question_type: str,
    confidence: float,
    graph_id: str | None = None,  # kept for API compatibility; unused
) -> None:
    """
    Fold the synthesis confidence into this strategy's Beta posterior.

    Confidence is treated as a fractional Bernoulli reward::

        alpha += c
        beta  += 1 - c

    so every observation moves the posterior. The previous rule applied a fixed
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

    query = """
    MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType)
    WHERE d.question_type = $question_type
    SET e.alpha = coalesce(e.alpha, coalesce(e.weight, 0.5) * $prior_strength) + $reward,
        e.beta  = coalesce(e.beta, (1.0 - coalesce(e.weight, 0.5)) * $prior_strength)
                  + (1.0 - $reward),
        e.feedback_count = coalesce(e.feedback_count, 0) + 1
    SET e.weight = toFloat(e.alpha / (e.alpha + e.beta))
    RETURN e.alpha AS alpha, e.beta AS beta, e.weight AS weight,
           e.feedback_count AS n
    """
    rows = _run_query(query, {
        "strategy":       strategy,
        "question_type":  question_type,
        "reward":         reward,
        "prior_strength": _PRIOR_STRENGTH,
    })
    if rows:
        row = rows[0]
        logger.info(
            "Routing feedback applied: strategy=%s qt=%s confidence=%.2f "
            "-> Beta(%.2f, %.2f) mean=%.3f n=%s",
            strategy, question_type, reward,
            row.get("alpha", 0.0), row.get("beta", 0.0),
            row.get("weight", 0.0), row.get("n", 0),
        )
    else:
        logger.debug(
            "Routing feedback skipped (no EFFECTIVE_FOR edge found): strategy=%s qt=%s",
            strategy, question_type,
        )


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
    question_types = ["skill_depth", "architecture", "project", "comparison", "credential", "general"]
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
