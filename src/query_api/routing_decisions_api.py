"""
Read side of the routing decision log.

``feedback.py`` writes one row per answered query into ``routing_decisions``:
the strategy that was selected, the probability Thompson sampling had of
selecting it, the synthesiser's self-assessed confidence, and — if someone
later rated the answer — the rating. CLAUDE.md calls that table "the data an
operator needs to check whether self-confidence predicts ratings at all".

Nothing could read it. This module exposes it over HTTP so the platform
console (and TeamWeave's ``/observability`` aggregator) can ask the question
the table was built to answer: **does the model's opinion of its own answer
agree with what people think of it?**

The summary mode answers that directly per (question type, strategy) with
``meanAbsDiff`` — ``AVG(ABS(confidence - rating))`` over the decisions that
have both. Near 0 means self-confidence tracks ratings and is worth using as
the cheap always-available reward; near 0.5 means the router has been
learning from a proxy that measures nothing.

Read-only. No PII: the table holds query ids, strategy labels and two
scores, never the question, the answer or anything about the caller.

Endpoint
--------
  GET /routing-decisions?mode=list|summary
                        &questionType=<exact>&strategy=<exact>&since=<iso8601>
                        &limit=<1..1000>&offset=<n>          (list mode only)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import feedback as _feedback

logger = logging.getLogger(__name__)

MODES = ("list", "summary")

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
MIN_LIMIT = 1

# Floats are rounded before they leave the process: these are display
# statistics, and 0.19999999999999996 in a dashboard helps nobody.
_ROUND_DP = 4

_LIST_COLUMNS = (
    "query_id, question_type, strategy, propensity, confidence, rating, created_at, rated_at"
)


class RoutingDecisionsError(Exception):
    """Carries the HTTP status the handler should return.

    Mirrors feedback.FeedbackError so the Lambda maps both the same way.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ─────────────────────────────────────────────────────────────────────────────
# Parameter parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_int(raw: Any, name: str) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        raise RoutingDecisionsError(400, f"{name} must be an integer")


def _parse_since(raw: Any) -> datetime:
    """Parse an ISO 8601 timestamp, accepting a trailing 'Z' for UTC."""
    text = str(raw).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise RoutingDecisionsError(
            400, "since must be an ISO 8601 timestamp, e.g. 2026-09-13T18:00:00Z"
        )


def parse_params(query_params: dict | None) -> dict:
    """Validate the query string into the arguments the fetchers take.

    Out-of-range ``limit`` is clamped rather than refused — a caller asking
    for 10000 rows wants as many as we will give, not an error. A ``limit``
    that is not a number at all is a mistake and is refused.

    ``questionType`` is *not* checked against rag_router.QUESTION_TYPES:
    POST /query-expertise lets a caller pass an arbitrary ``questionType``
    which is stored verbatim, so rejecting anything outside the canonical
    list would make rows that genuinely exist unreachable. An unknown value
    simply matches nothing.
    """
    params = query_params or {}

    mode = (params.get("mode") or "list").strip()
    if mode not in MODES:
        raise RoutingDecisionsError(400, "mode must be 'list' or 'summary'")

    question_type = params.get("questionType")
    strategy = params.get("strategy")
    since = _parse_since(params["since"]) if params.get("since") else None

    limit = DEFAULT_LIMIT
    if params.get("limit") is not None and str(params.get("limit")).strip() != "":
        limit = max(MIN_LIMIT, min(MAX_LIMIT, _parse_int(params["limit"], "limit")))

    offset = 0
    if params.get("offset") is not None and str(params.get("offset")).strip() != "":
        offset = max(0, _parse_int(params["offset"], "offset"))

    return {
        "mode": mode,
        "question_type": question_type or None,
        "strategy": strategy or None,
        "since": since,
        "limit": limit,
        "offset": offset,
    }


def _where(question_type: str | None, strategy: str | None, since: Any) -> tuple[str, list]:
    """Build the shared WHERE clause; both modes filter identically."""
    clauses: list[str] = []
    values: list[Any] = []
    if question_type is not None:
        clauses.append("question_type = %s")
        values.append(question_type)
    if strategy is not None:
        clauses.append("strategy = %s")
        values.append(strategy)
    if since is not None:
        clauses.append("created_at >= %s")
        values.append(since)
    return ("WHERE " + " AND ".join(clauses) if clauses else ""), values


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _round(value: Any) -> float | None:
    """Round a nullable float for output; None stays None.

    A group with no rated decisions must report null, not 0.0 — a spurious
    zero would read as "perfect agreement" on exactly the groups where
    nothing is known.
    """
    return None if value is None else round(float(value), _ROUND_DP)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _decision_row(row: tuple) -> dict:
    query_id, question_type, strategy, propensity, confidence, rating, created_at, rated_at = row
    return {
        "queryId": query_id,
        "questionType": question_type,
        "strategy": strategy,
        "propensity": _round(propensity),
        "confidence": _round(confidence),
        "rating": _round(rating),
        "createdAt": _iso(created_at),
        "ratedAt": _iso(rated_at),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────────────────────────────────────

def fetch_list(conn: Any, *, question_type=None, strategy=None, since=None,
               limit: int = DEFAULT_LIMIT, offset: int = 0, **_ignored) -> dict:
    """Most recent decisions first, one page at a time.

    ``count`` is the number of decisions matching the filters, *not* the
    number returned in this page — a pager needs the total to know whether
    there is a next page.
    """
    where, values = _where(question_type, strategy, since)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_LIST_COLUMNS} FROM routing_decisions {where} "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            tuple(values + [limit, offset]),
        )
        items = [_decision_row(r) for r in cur.fetchall()]

        cur.execute(f"SELECT COUNT(*) FROM routing_decisions {where}", tuple(values))
        total = cur.fetchone()[0]

    return {"items": items, "count": int(total)}


def fetch_summary(conn: Any, *, question_type=None, strategy=None, since=None, **_ignored) -> dict:
    """Agreement between self-confidence and human rating, per (type, strategy).

    The aggregates rely on SQL NULL semantics rather than filtering by hand:

      * ``AVG(confidence)`` skips unreported confidences (the router does not
        learn from those either) and is NULL only if the group has none.
      * ``AVG(rating)`` is NULL for a group nobody has rated.
      * ``ABS(confidence - rating)`` is NULL unless *both* are present, so
        ``AVG`` of it is the mean over rated decisions only, and NULL when
        there are none. That is the definition we want, not an average over
        rows where one side was silently read as zero.
    """
    where, values = _where(question_type, strategy, since)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT question_type, strategy, "
            "COUNT(*), "
            "COUNT(rating), "
            "AVG(confidence), "
            "AVG(rating), "
            "AVG(ABS(confidence - rating)) "
            f"FROM routing_decisions {where} "
            "GROUP BY question_type, strategy "
            "ORDER BY question_type, strategy",
            tuple(values),
        )
        rows = cur.fetchall()

    groups = []
    total_count = 0
    total_rated = 0
    for qt, strat, count, rated_count, avg_conf, avg_rating, mean_abs_diff in rows:
        count = int(count)
        rated_count = int(rated_count)
        total_count += count
        total_rated += rated_count
        groups.append({
            "questionType": qt,
            "strategy": strat,
            "count": count,
            "ratedCount": rated_count,
            "avgConfidence": _round(avg_conf),
            "avgRating": _round(avg_rating),
            "meanAbsDiff": _round(mean_abs_diff),
        })

    return {"groups": groups, "totalCount": total_count, "totalRated": total_rated}


def handle(conn: Any, query_params: dict | None) -> dict:
    """Entry point for GET /routing-decisions.

    Raises RoutingDecisionsError(400) for an unusable query string.
    """
    params = parse_params(query_params)

    # CREATE TABLE IF NOT EXISTS, shared with the writer, so a read against a
    # stack that has answered nothing yet returns an empty result rather than
    # failing on a missing relation.
    _feedback.ensure_schema(conn)

    mode = params.pop("mode")
    result = fetch_summary(conn, **params) if mode == "summary" else fetch_list(conn, **params)
    logger.info("routing-decisions mode=%s params=%s", mode, params)
    return result
