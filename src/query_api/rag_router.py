"""
Adaptive RAG Router

Query-time agent that selects the best RAG retrieval strategy for a given
question type by consulting the Neptune routing graph.

Decision logic:
  1. Query Neptune for EFFECTIVE_FOR edge weights connecting RAGStrategy nodes
     to the current question type (stored as a DocumentType proxy).
  2. Pick the strategy with the highest weight (ties broken by priority order).
  3. Return a RetrievalConfig that configures the retriever accordingly.

Feedback loop:
  After each query, call update_feedback() to adjust edge weights in Neptune
  based on the synthesis confidence score. This is how the graph improves.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

from models import RAGStrategyLabel, RetrievalConfig, ROUTING_PRIORS

logger = logging.getLogger(__name__)

_NEPTUNE_CLIENT: Any = None
NEPTUNE_GRAPH_ID = os.environ.get("NEPTUNE_GRAPH_ID", "")

# Confidence threshold above which we reinforce, below which we penalise
_REINFORCE_THRESHOLD = 0.70
_PENALISE_THRESHOLD = 0.40
_REINFORCE_DELTA = 0.05
_PENALISE_DELTA = -0.02
_WEIGHT_FLOOR = 0.10
_WEIGHT_CEIL = 1.00

# Adversarial guardrails for the penalty path
_WARMUP_MIN_FEEDBACK       = 5   # edges protected until this many feedback events
_CONSECUTIVE_LOW_THRESHOLD = 3   # consecutive low-confidence queries before penalising
_MIN_PENALTY_INTERVAL      = 3   # min feedback_count gap between consecutive penalties

# Fallback priority when Neptune is unavailable (or graph has no data yet)
_STRATEGY_PRIORITY: list[str] = [
    RAGStrategyLabel.GRAPH_FIRST,
    RAGStrategyLabel.HYBRID,
    RAGStrategyLabel.KEYWORD_BOOSTED,
    RAGStrategyLabel.SEMANTIC,
]


def _get_neptune_client() -> Any:
    global _NEPTUNE_CLIENT
    if _NEPTUNE_CLIENT is None:
        _NEPTUNE_CLIENT = boto3.client(
            "neptune-graph",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _NEPTUNE_CLIENT


def _run_query(graph_id: str, query: str, parameters: dict | None = None) -> list[dict]:
    """Execute an openCypher query; returns [] on any error (non-fatal)."""
    client = _get_neptune_client()
    kwargs: dict[str, Any] = {
        "graphIdentifier": graph_id,
        "queryString": query,
        "language": "OPEN_CYPHER",
    }
    if parameters:
        kwargs["parameters"] = json.dumps(parameters)
    try:
        response = client.execute_query(**kwargs)
        payload = response.get("payload")
        raw = payload.read() if hasattr(payload, "read") else (payload or b"{}")
        return json.loads(raw).get("results", [])
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        logger.warning("Neptune routing query skipped (%s): %s", code, exc)
        return []
    except Exception as exc:
        logger.warning("Neptune routing query error: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy selection
# ─────────────────────────────────────────────────────────────────────────────

def _query_strategy_weights(graph_id: str, question_type: str) -> dict[str, float]:
    """
    Fetch EFFECTIVE_FOR edge weights from Neptune for the given question_type.

    Returns a dict mapping RAGStrategy label → weight.
    Falls back to ROUTING_PRIORS when Neptune has no data.
    """
    query = """
    MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
    WHERE d.question_type = $question_type
    RETURN r.label AS strategy, e.weight AS weight
    """
    rows = _run_query(graph_id, query, {"question_type": question_type})

    if rows:
        return {row["strategy"]: float(row["weight"]) for row in rows}

    # Fall back to hard-coded priors
    return {
        strategy.value: ROUTING_PRIORS.get((strategy, question_type), 0.50)
        for strategy in RAGStrategyLabel
    }


def select_strategy(
    question_type: str,
    graph_id: str | None = None,
) -> RetrievalConfig:
    """
    Choose the best RAG strategy for the given question type.

    Args:
        question_type: Classified question type string (e.g. "architecture").
        graph_id:      Neptune graph ID (defaults to env var).

    Returns:
        RetrievalConfig describing which retrieval approach to use.
    """
    gid = graph_id or NEPTUNE_GRAPH_ID

    weights: dict[str, float] = {}
    if gid:
        weights = _query_strategy_weights(gid, question_type)
    else:
        weights = {
            strategy.value: ROUTING_PRIORS.get((strategy, question_type), 0.50)
            for strategy in RAGStrategyLabel
        }

    # Pick strategy with highest weight; use priority order for ties
    best_strategy = max(
        _STRATEGY_PRIORITY,
        key=lambda s: weights.get(s, 0.0),
    )
    best_weight = weights.get(best_strategy, 0.5)

    logger.info(
        "Routing decision: strategy=%s weight=%.3f question_type=%s weights=%s",
        best_strategy, best_weight, question_type, weights,
    )

    return RetrievalConfig(
        strategy=best_strategy,
        include_graph=(best_strategy in (RAGStrategyLabel.GRAPH_FIRST, RAGStrategyLabel.HYBRID)),
        boost_keywords=(best_strategy in (RAGStrategyLabel.KEYWORD_BOOSTED, RAGStrategyLabel.HYBRID)),
        use_neptune_chunks=(best_strategy == RAGStrategyLabel.HYBRID),
        strategy_confidence=best_weight,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feedback / learning
# ─────────────────────────────────────────────────────────────────────────────

def _read_edge_props(graph_id: str, strategy: str, question_type: str) -> dict:
    """Read guardrail-relevant properties from the EFFECTIVE_FOR edge."""
    query = """
    MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType {question_type: $question_type})
    RETURN e.feedback_count        AS feedback_count,
           e.consecutive_low_count AS consecutive_low_count,
           e.last_penalty_at_count AS last_penalty_at_count
    """
    rows = _run_query(graph_id, query, {"strategy": strategy, "question_type": question_type})
    if rows:
        r = rows[0]
        return {
            "feedback_count":        int(r.get("feedback_count") or 0),
            "consecutive_low_count": int(r.get("consecutive_low_count") or 0),
            "last_penalty_at_count": int(r.get("last_penalty_at_count") or 0),
        }
    return {"feedback_count": 0, "consecutive_low_count": 0, "last_penalty_at_count": 0}


def _check_penalty_guardrails(edge_props: dict, new_consecutive: int) -> tuple[bool, str]:
    """
    Return (should_apply_penalty, reason_if_blocked).

    All three guardrails must pass for a penalty to be applied:
      1. Warmup  – edge must have ≥ _WARMUP_MIN_FEEDBACK total feedback events.
      2. Consecutive – must be ≥ _CONSECUTIVE_LOW_THRESHOLD consecutive low-confidence queries.
      3. Rate limit – at least _MIN_PENALTY_INTERVAL feedback events must have elapsed
                      since the last applied penalty.
    """
    feedback_count = edge_props["feedback_count"]
    last_penalty   = edge_props["last_penalty_at_count"]

    if feedback_count < _WARMUP_MIN_FEEDBACK:
        return False, f"warmup (feedback_count={feedback_count} < {_WARMUP_MIN_FEEDBACK})"

    if new_consecutive < _CONSECUTIVE_LOW_THRESHOLD:
        return False, f"consecutive_low={new_consecutive} < {_CONSECUTIVE_LOW_THRESHOLD}"

    gap = feedback_count - last_penalty
    if gap < _MIN_PENALTY_INTERVAL:
        return False, f"rate_limit (gap={gap} < {_MIN_PENALTY_INTERVAL})"

    return True, ""


def update_feedback(
    strategy: str,
    question_type: str,
    confidence: float,
    graph_id: str | None = None,
) -> None:
    """
    Adjust the EFFECTIVE_FOR edge weight in Neptune based on synthesis confidence.

    Core learning loop with adversarial guardrails:

      confidence ≥ 0.70 → reinforce (+0.05, capped at 1.0); resets consecutive_low_count
      confidence < 0.40 → penalty path (guarded – see below)
      otherwise         → no change

    Penalty guardrails (all must pass before -0.02 is applied):
      1. Warmup:      edge must have ≥ _WARMUP_MIN_FEEDBACK total feedback events
      2. Consecutive: ≥ _CONSECUTIVE_LOW_THRESHOLD consecutive low-confidence queries
      3. Rate limit:  ≥ _MIN_PENALTY_INTERVAL feedback events since last penalty

    When guardrails block a penalty, consecutive_low_count is still incremented so
    the streak is tracked. When a penalty is applied, consecutive_low_count resets
    to 0 and last_penalty_at_count is updated.

    Args:
        strategy:      RAGStrategyLabel value of the strategy that was used.
        question_type: Question type that was answered.
        confidence:    Synthesis confidence score (0–1).
        graph_id:      Neptune graph ID.
    """
    gid = graph_id or NEPTUNE_GRAPH_ID
    if not gid:
        return

    # ── Reinforce path ──────────────────────────────────────────────────────
    if confidence >= _REINFORCE_THRESHOLD:
        query = """
        MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType {question_type: $question_type})
        SET e.weight = toFloat(
                CASE WHEN e.weight + $delta > $ceil THEN $ceil ELSE e.weight + $delta END
            ),
            e.consecutive_low_count = 0,
            e.feedback_count = coalesce(e.feedback_count, 0) + 1
        RETURN e.weight AS new_weight
        """
        rows = _run_query(gid, query, {
            "strategy":      strategy,
            "question_type": question_type,
            "delta":         _REINFORCE_DELTA,
            "ceil":          _WEIGHT_CEIL,
        })
        if rows:
            logger.info(
                "Routing feedback applied: strategy=%s qt=%s confidence=%.2f delta=%+.2f new_weight=%.3f",
                strategy, question_type, confidence, _REINFORCE_DELTA, rows[0].get("new_weight", 0),
            )
        else:
            logger.debug(
                "Routing feedback skipped (no EFFECTIVE_FOR edge found): strategy=%s qt=%s",
                strategy, question_type,
            )
        return

    # ── Neutral zone ─────────────────────────────────────────────────────────
    if confidence >= _PENALISE_THRESHOLD:
        return  # 0.40 ≤ confidence < 0.70 – no weight change

    # ── Penalty path (with guardrails) ───────────────────────────────────────
    edge_props   = _read_edge_props(gid, strategy, question_type)
    new_consec   = edge_props["consecutive_low_count"] + 1
    new_fc       = edge_props["feedback_count"] + 1

    apply, reason = _check_penalty_guardrails(edge_props, new_consec)

    if not apply:
        # Guardrail blocked: track streak but do not change weight
        logger.info(
            "Routing penalty blocked by guardrail [%s]: strategy=%s qt=%s confidence=%.2f consecutive=%d",
            reason, strategy, question_type, confidence, new_consec,
        )
        _run_query(gid, """
        MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType {question_type: $question_type})
        SET e.consecutive_low_count = $new_consec,
            e.feedback_count        = $new_fc
        RETURN e.weight AS new_weight
        """, {
            "strategy":      strategy,
            "question_type": question_type,
            "new_consec":    new_consec,
            "new_fc":        new_fc,
        })
        return

    # All guardrails passed – apply penalty and reset tracking state
    rows = _run_query(gid, """
    MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType {question_type: $question_type})
    SET e.weight = toFloat(
            CASE WHEN e.weight + $delta < $floor THEN $floor ELSE e.weight + $delta END
        ),
        e.consecutive_low_count = 0,
        e.last_penalty_at_count = $new_fc,
        e.feedback_count        = $new_fc
    RETURN e.weight AS new_weight
    """, {
        "strategy":      strategy,
        "question_type": question_type,
        "delta":         _PENALISE_DELTA,
        "floor":         _WEIGHT_FLOOR,
        "new_fc":        new_fc,
    })
    if rows:
        logger.info(
            "Routing feedback applied: strategy=%s qt=%s confidence=%.2f delta=%+.2f new_weight=%.3f",
            strategy, question_type, confidence, _PENALISE_DELTA, rows[0].get("new_weight", 0),
        )
    else:
        logger.debug(
            "Routing feedback skipped (no EFFECTIVE_FOR edge found): strategy=%s qt=%s",
            strategy, question_type,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Routing graph seed (called from ingestion_trigger on first deploy)
# ─────────────────────────────────────────────────────────────────────────────

def seed_routing_graph(graph_id: str) -> dict[str, int]:
    """
    Idempotently create RAGStrategy and DocumentType proxy nodes plus
    EFFECTIVE_FOR edges with initial prior weights.

    A DocumentType node is created per question_type so the EFFECTIVE_FOR
    relationship can be updated independently per question category.

    Returns counts of nodes/edges written.
    """
    if not graph_id:
        logger.warning("seed_routing_graph: no graph_id provided")
        return {"nodes": 0, "edges": 0}

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
        rows = _run_query(graph_id, q, {"label": strategy})
        if rows:
            nodes_written += 1

    # Upsert DocumentType proxy nodes (one per question_type)
    for qt in question_types:
        q = """
        MERGE (d:DocumentType {question_type: $question_type})
        ON CREATE SET d.label = $question_type, d.created = timestamp()
        RETURN d.question_type AS qt
        """
        rows = _run_query(graph_id, q, {"question_type": qt})
        if rows:
            nodes_written += 1

    # Upsert EFFECTIVE_FOR edges with prior weights
    for (strategy, question_type), weight in ROUTING_PRIORS.items():
        q = """
        MATCH (r:RAGStrategy {label: $strategy})
        MATCH (d:DocumentType {question_type: $question_type})
        MERGE (r)-[e:EFFECTIVE_FOR]->(d)
        ON CREATE SET e.weight = $weight, e.feedback_count = 0, e.seeded = true,
                      e.consecutive_low_count = 0, e.last_penalty_at_count = 0
        RETURN e.weight AS w
        """
        rows = _run_query(graph_id, q, {
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
