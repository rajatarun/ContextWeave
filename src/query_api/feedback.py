"""
Independent reward for the routing loop.

The router learns from the synthesiser's self-assessed confidence, which is
the model's opinion of its own output. A confidently wrong answer therefore
reinforces the strategy that produced it, and no amount of exploration
changes that: exploration fixes the search, not the objective.

This module adds a second reward source that does not come from the model.
Every answered query is recorded against a ``queryId`` together with the
routing decision that produced it. A caller who has seen the answer can
later POST a rating for that ``queryId``; the rating is folded into the same
Beta posterior as the self-assessment, with a configurable weight, and
counted separately so the two sources can be compared.

Why the same posterior rather than a separate one: the routing decision is
one decision, and what we want to know is which strategy produces answers
people find useful. Self-confidence is a cheap, always-available proxy for
that; a human rating is an expensive, occasional measurement of it. Folding
both in with the human weighted higher (default 2x) lets the rare direct
measurement correct the frequent proxy without waiting for enough ratings
to learn from ratings alone.

What this does not do: it does not calibrate the self-confidence. Comparing
``routing_decisions.confidence`` against ``rating`` over time is how an
operator would find out whether the proxy is worth anything; the table holds
both columns for that purpose.

Environment
-----------
  ROUTER_HUMAN_FEEDBACK_WEIGHT   float  default 2.0   observations one rating counts as
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

HUMAN_WEIGHT = float(os.environ.get("ROUTER_HUMAN_FEEDBACK_WEIGHT", "2.0"))

_RATING_WORDS = {
    "up": 1.0, "good": 1.0, "helpful": 1.0, "yes": 1.0, "correct": 1.0,
    "down": 0.0, "bad": 0.0, "unhelpful": 0.0, "no": 0.0, "wrong": 0.0,
    "neutral": 0.5, "partial": 0.5, "mixed": 0.5,
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    query_id        TEXT PRIMARY KEY,
    question_type   TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    propensity      DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rating          DOUBLE PRECISION,
    rated_at        TIMESTAMPTZ
)
"""


class FeedbackError(Exception):
    """Carries the HTTP status the handler should return."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def rating_to_reward(rating: Any) -> float | None:
    """Map a caller's rating to a reward in [0, 1]; None if unrecognised.

    Accepts the words in _RATING_WORDS, booleans, or a number in [0, 1].
    A number outside [0, 1] is rejected rather than clamped: a caller sending
    a 5-star scale by mistake should hear about it, not have 5 read as 1.0.
    """
    if isinstance(rating, bool):
        return 1.0 if rating else 0.0
    if isinstance(rating, (int, float)):
        r = float(rating)
        return r if 0.0 <= r <= 1.0 else None
    if isinstance(rating, str):
        return _RATING_WORDS.get(rating.strip().lower())
    return None


def ensure_schema(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def record_decision(
    conn: Any,
    *,
    query_id: str,
    question_type: str,
    strategy: str,
    propensity: float | None,
    confidence: float | None,
) -> None:
    """Persist the routing decision behind an answer so a later rating can reach it."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute(
            """
            INSERT INTO routing_decisions (query_id, question_type, strategy, propensity, confidence)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (query_id) DO NOTHING
            """,
            (query_id, question_type, strategy, propensity, confidence),
        )
    conn.commit()


def apply_rating(conn: Any, *, query_id: str, rating: Any, weight: float = HUMAN_WEIGHT) -> dict:
    """Fold a rating for ``query_id`` into the routing posterior; one rating per query.

    Raises FeedbackError(400) for an unusable rating, (404) for an unknown
    query, (409) if the query was already rated. Returns what was applied.
    """
    reward = rating_to_reward(rating)
    if reward is None:
        raise FeedbackError(400, "rating must be up/down/neutral, true/false, or a number in [0, 1]")
    if not isinstance(query_id, str) or not query_id.strip():
        raise FeedbackError(400, "queryId is required")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT question_type, strategy, confidence, rated_at FROM routing_decisions WHERE query_id = %s",
            (query_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise FeedbackError(404, "unknown queryId (answers served before feedback was enabled, "
                                     "or older than the decision log, cannot be rated)")
        question_type, strategy, self_confidence, rated_at = row
        if rated_at is not None:
            raise FeedbackError(409, "this queryId has already been rated")

        # Import here so the module stays importable in tests without the router's
        # sys.path arrangement; both live in the same directory in Lambda.
        from rag_router import update_feedback
        posterior = update_feedback(
            strategy=strategy,
            question_type=question_type,
            confidence=reward,
            weight=weight,
            source="human",
        )

        cur.execute(
            "UPDATE routing_decisions SET rating = %s, rated_at = NOW() WHERE query_id = %s",
            (reward, query_id),
        )
    conn.commit()

    logger.info(
        "Human feedback applied: query=%s strategy=%s qt=%s reward=%.2f weight=%.1f self_conf=%s",
        query_id, strategy, question_type, reward, weight, self_confidence,
    )
    return {
        "queryId": query_id,
        "strategy": strategy,
        "questionType": question_type,
        "reward": reward,
        "weight": weight,
        "selfConfidence": self_confidence,
        "posterior": posterior,
    }
