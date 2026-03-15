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

def update_feedback(
    strategy: str,
    question_type: str,
    confidence: float,
    graph_id: str | None = None,
) -> None:
    """
    Adjust the EFFECTIVE_FOR edge weight in Neptune based on synthesis confidence.

    This is the core learning loop:
      confidence ≥ 0.70 → reinforce (+0.05, capped at 1.0)
      confidence < 0.40 → penalise  (-0.02, floored at 0.1)
      otherwise         → no change

    Args:
        strategy:      RAGStrategyLabel value of the strategy that was used.
        question_type: Question type that was answered.
        confidence:    Synthesis confidence score (0–1).
        graph_id:      Neptune graph ID.
    """
    gid = graph_id or NEPTUNE_GRAPH_ID
    if not gid:
        return

    if confidence >= _REINFORCE_THRESHOLD:
        delta = _REINFORCE_DELTA
    elif confidence < _PENALISE_THRESHOLD:
        delta = _PENALISE_DELTA
    else:
        return  # neutral zone – no update

    query = """
    MATCH (r:RAGStrategy {label: $strategy})-[e:EFFECTIVE_FOR]->(d:DocumentType {question_type: $question_type})
    SET e.weight = toFloat(
        CASE
            WHEN e.weight + $delta > $ceil  THEN $ceil
            WHEN e.weight + $delta < $floor THEN $floor
            ELSE e.weight + $delta
        END
    ),
    e.feedback_count = coalesce(e.feedback_count, 0) + 1
    RETURN e.weight AS new_weight
    """
    rows = _run_query(gid, query, {
        "strategy":      strategy,
        "question_type": question_type,
        "delta":         delta,
        "ceil":          _WEIGHT_CEIL,
        "floor":         _WEIGHT_FLOOR,
    })
    if rows:
        logger.info(
            "Routing feedback applied: strategy=%s qt=%s confidence=%.2f delta=%+.2f new_weight=%.3f",
            strategy, question_type, confidence, delta, rows[0].get("new_weight", 0),
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
        ON CREATE SET e.weight = $weight, e.feedback_count = 0, e.seeded = true
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
