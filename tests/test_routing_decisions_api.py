"""GET /routing-decisions: the decision log, readable.

The table exists to answer one question — does the synthesiser's opinion of
its own answer agree with what people thought of it — and until now nothing
could ask it. What is under test is that contract: the filters select what
they claim, the pager totals are honest, and ``meanAbsDiff`` is the mean over
*rated* decisions and null (never 0.0) when there are none.

The connection is a FakeConn in the shape of the one in test_feedback.py,
but backed by in-memory sqlite rather than a hand-rolled dict. The logic
under test is largely SQL — AVG skipping NULLs, ABS(a - b) vanishing unless
both sides are present, GROUP BY, ORDER BY, LIMIT/OFFSET — and a fake that
reimplements those in Python would only test the fake. sqlite shares
Postgres' NULL-aggregate semantics (AVG ignores NULLs and is NULL over zero
rows), so the real query strings run here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "query_api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shared"))

import routing_decisions_api as RD  # noqa: E402

# sqlite has no NOW() and no TIMESTAMPTZ default, so the table is declared
# here rather than from feedback.SCHEMA_SQL. Columns and order match it;
# test_schema_is_not_redefined pins that the module itself does not carry a
# second CREATE TABLE.
_DDL = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    query_id        TEXT PRIMARY KEY,
    question_type   TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    propensity      REAL,
    confidence      REAL,
    created_at      TEXT NOT NULL,
    rating          REAL,
    rated_at        TEXT
)
"""


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self._cur = db.sqlite.cursor()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.db.executed.append((" ".join(sql.split()), params))
        # psycopg2 paramstyle -> sqlite paramstyle; nothing else is rewritten.
        self._cur.execute(sql.replace("%s", "?"), tuple(params or ()))

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()


class FakeConn:
    def __init__(self):
        self.sqlite = sqlite3.connect(":memory:")
        self.sqlite.execute(_DDL)
        self.executed: list = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def add(self, query_id, question_type, strategy, *, propensity=0.5,
            confidence=None, created_at="2026-09-01T00:00:00+00:00",
            rating=None, rated_at=None):
        self.sqlite.execute(
            "INSERT INTO routing_decisions VALUES (?,?,?,?,?,?,?,?)",
            (query_id, question_type, strategy, propensity, confidence,
             created_at, rating, rated_at),
        )
        return self


@pytest.fixture
def conn(monkeypatch):
    """A connection whose ensure_schema is a no-op (sqlite cannot run the real DDL)."""
    monkeypatch.setattr(RD._feedback, "ensure_schema", lambda c: None)
    return FakeConn()


# ─────────────────────────────────────────────────────────────────────────────
# Empty table
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_table_lists_nothing_rather_than_failing(conn):
    assert RD.handle(conn, {}) == {"items": [], "count": 0}


def test_empty_table_summarises_to_no_groups(conn):
    assert RD.handle(conn, {"mode": "summary"}) == {
        "groups": [], "totalCount": 0, "totalRated": 0,
    }


def test_schema_is_created_before_reading_and_is_not_redefined(monkeypatch):
    """A fresh stack has answered nothing; the read must not 500 on a missing table."""
    calls = []
    monkeypatch.setattr(RD._feedback, "ensure_schema", lambda c: calls.append(c))
    c = FakeConn()
    RD.handle(c, {})
    assert calls == [c], "the read must reuse feedback.ensure_schema"

    source = open(os.path.join(os.path.dirname(__file__), "..", "src", "query_api",
                               "routing_decisions_api.py")).read()
    code = [ln for ln in source.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("CREATE TABLE" in ln for ln in code), \
        "the table is defined once, in feedback.py"


# ─────────────────────────────────────────────────────────────────────────────
# List mode
# ─────────────────────────────────────────────────────────────────────────────

def test_list_returns_newest_first_with_nulls_preserved(conn):
    conn.add("q1", "architecture", "graph_first", propensity=0.31, confidence=0.82,
             created_at="2026-09-01T00:00:00+00:00", rating=1.0,
             rated_at="2026-09-01T00:05:00+00:00")
    conn.add("q2", "general", "semantic_search", confidence=0.4,
             created_at="2026-09-02T00:00:00+00:00")

    out = RD.handle(conn, {})
    assert out["count"] == 2
    assert [i["queryId"] for i in out["items"]] == ["q2", "q1"], "ORDER BY created_at DESC"

    assert out["items"][0]["rating"] is None and out["items"][0]["ratedAt"] is None
    assert out["items"][1] == {
        "queryId": "q1", "questionType": "architecture", "strategy": "graph_first",
        "propensity": 0.31, "confidence": 0.82, "rating": 1.0,
        "createdAt": "2026-09-01T00:00:00+00:00", "ratedAt": "2026-09-01T00:05:00+00:00",
    }


def test_list_filters_by_question_type_and_strategy(conn):
    conn.add("a", "architecture", "graph_first")
    conn.add("b", "architecture", "hybrid")
    conn.add("c", "general", "graph_first")

    assert {i["queryId"] for i in RD.handle(conn, {"questionType": "architecture"})["items"]} == {"a", "b"}
    assert {i["queryId"] for i in RD.handle(conn, {"strategy": "graph_first"})["items"]} == {"a", "c"}

    both = RD.handle(conn, {"questionType": "architecture", "strategy": "graph_first"})
    assert [i["queryId"] for i in both["items"]] == ["a"] and both["count"] == 1


def test_unknown_question_type_matches_nothing_rather_than_erroring(conn):
    """A caller may pass any questionType to /query-expertise; it is stored verbatim.

    Validating the filter against the canonical list would hide rows that
    genuinely exist, so an unrecognised value is a filter, not an error.
    """
    conn.add("a", "bespoke_type", "hybrid")
    assert RD.handle(conn, {"questionType": "nonexistent"})["count"] == 0
    assert RD.handle(conn, {"questionType": "bespoke_type"})["count"] == 1


def test_list_filters_by_since_inclusively(conn):
    conn.add("old", "general", "hybrid", created_at="2026-09-01T00:00:00+00:00")
    conn.add("edge", "general", "hybrid", created_at="2026-09-02T00:00:00+00:00")
    conn.add("new", "general", "hybrid", created_at="2026-09-03T00:00:00+00:00")

    out = RD.handle(conn, {"since": "2026-09-02T00:00:00+00:00"})
    assert {i["queryId"] for i in out["items"]} == {"edge", "new"}, "since is >=, not >"
    assert RD.handle(conn, {"since": "2026-09-02T00:00:00Z"})["count"] == 2, "trailing Z is UTC"


def test_pagination_pages_without_lying_about_the_total(conn):
    for n in range(5):
        conn.add(f"q{n}", "general", "hybrid", created_at=f"2026-09-0{n + 1}T00:00:00+00:00")

    first = RD.handle(conn, {"limit": "2"})
    assert [i["queryId"] for i in first["items"]] == ["q4", "q3"]
    assert first["count"] == 5, "count is the total matching, not the page size"

    second = RD.handle(conn, {"limit": "2", "offset": "2"})
    assert [i["queryId"] for i in second["items"]] == ["q2", "q1"]
    assert second["count"] == 5

    assert RD.handle(conn, {"offset": "99"})["items"] == []


def test_limit_is_clamped_not_refused():
    assert RD.parse_params({"limit": "100000"})["limit"] == RD.MAX_LIMIT
    assert RD.parse_params({"limit": "0"})["limit"] == RD.MIN_LIMIT
    assert RD.parse_params({"limit": "-5"})["limit"] == RD.MIN_LIMIT
    assert RD.parse_params({"offset": "-5"})["offset"] == 0
    assert RD.parse_params({})["limit"] == RD.DEFAULT_LIMIT
    assert RD.parse_params({})["offset"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Summary mode — the calibration question the table exists for
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_groups_by_question_type_and_strategy(conn):
    conn.add("a", "architecture", "graph_first", confidence=0.8)
    conn.add("b", "architecture", "graph_first", confidence=0.6)
    conn.add("c", "architecture", "hybrid", confidence=0.5)
    conn.add("d", "general", "graph_first", confidence=0.5)

    out = RD.handle(conn, {"mode": "summary"})
    assert [(g["questionType"], g["strategy"], g["count"]) for g in out["groups"]] == [
        ("architecture", "graph_first", 2),
        ("architecture", "hybrid", 1),
        ("general", "graph_first", 1),
    ]
    assert out["totalCount"] == 4 and out["totalRated"] == 0


def test_mean_abs_diff_is_computed_only_over_rated_decisions(conn):
    """Hand-computed: the unrated row must not dilute the mean toward zero.

    rated:   |0.9 - 1.0| = 0.1
             |0.8 - 0.0| = 0.8
             |0.5 - 0.5| = 0.0   → mean = 0.9 / 3 = 0.3
    unrated: confidence 0.2, no rating — counted in count and avgConfidence,
             excluded from ratedCount, avgRating and meanAbsDiff.
    """
    conn.add("r1", "architecture", "graph_first", confidence=0.9, rating=1.0, rated_at="t")
    conn.add("r2", "architecture", "graph_first", confidence=0.8, rating=0.0, rated_at="t")
    conn.add("r3", "architecture", "graph_first", confidence=0.5, rating=0.5, rated_at="t")
    conn.add("u1", "architecture", "graph_first", confidence=0.2)

    (group,) = RD.handle(conn, {"mode": "summary"})["groups"]
    assert group["count"] == 4 and group["ratedCount"] == 3
    assert group["meanAbsDiff"] == 0.3
    assert group["avgRating"] == 0.5, "(1.0 + 0.0 + 0.5) / 3"
    assert group["avgConfidence"] == 0.6, "(0.9 + 0.8 + 0.5 + 0.2) / 4 — all reported confidences"


def test_unrated_group_reports_null_not_a_spurious_zero(conn):
    """0.0 would read as perfect agreement on a group nobody has rated."""
    conn.add("a", "general", "hybrid", confidence=0.7)
    conn.add("b", "general", "hybrid", confidence=0.9)

    (group,) = RD.handle(conn, {"mode": "summary"})["groups"]
    assert group["ratedCount"] == 0
    assert group["meanAbsDiff"] is None
    assert group["avgRating"] is None
    assert group["avgConfidence"] == 0.8, "self-confidence is still known without ratings"


def test_unreported_confidence_is_excluded_from_both_averages(conn):
    """confidence is NULL when the model did not report one; it is not an observation."""
    conn.add("a", "general", "hybrid", confidence=None, rating=1.0, rated_at="t")
    conn.add("b", "general", "hybrid", confidence=0.6, rating=1.0, rated_at="t")

    (group,) = RD.handle(conn, {"mode": "summary"})["groups"]
    assert group["count"] == 2 and group["ratedCount"] == 2
    assert group["avgConfidence"] == 0.6, "the NULL confidence is skipped, not read as 0"
    assert group["meanAbsDiff"] == 0.4, "only the row with both sides contributes"


def test_summary_honours_the_same_filters_as_list(conn):
    conn.add("a", "architecture", "hybrid", confidence=0.9, rating=0.9, rated_at="t",
             created_at="2026-09-01T00:00:00+00:00")
    conn.add("b", "general", "hybrid", confidence=0.1, rating=1.0, rated_at="t",
             created_at="2026-09-05T00:00:00+00:00")

    out = RD.handle(conn, {"mode": "summary", "questionType": "architecture"})
    assert out["totalCount"] == 1 and out["groups"][0]["questionType"] == "architecture"

    out = RD.handle(conn, {"mode": "summary", "since": "2026-09-03T00:00:00Z"})
    assert out["totalCount"] == 1 and out["groups"][0]["questionType"] == "general"


def test_floats_are_rounded_to_four_places(conn):
    conn.add("a", "general", "hybrid", propensity=1 / 3, confidence=0.1, rating=0.35,
             rated_at="t", created_at="2026-09-01T00:00:00+00:00")
    conn.add("b", "general", "hybrid", confidence=0.2, rating=0.35,
             rated_at="t", created_at="2026-09-02T00:00:00+00:00")

    by_id = {i["queryId"]: i for i in RD.handle(conn, {})["items"]}
    assert by_id["a"]["propensity"] == 0.3333
    # |0.1-0.35| = 0.25, |0.2-0.35| = 0.15 -> 0.2, but in binary float 0.20000000000000004
    assert RD.handle(conn, {"mode": "summary"})["groups"][0]["meanAbsDiff"] == 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Bad input
# ─────────────────────────────────────────────────────────────────────────────

def test_bad_mode_is_400(conn):
    with pytest.raises(RD.RoutingDecisionsError) as e:
        RD.handle(conn, {"mode": "aggregate"})
    assert e.value.status == 400
    assert e.value.message == "mode must be 'list' or 'summary'"
    assert conn.executed == [], "validation happens before any query"


def test_malformed_since_is_400_with_an_example(conn):
    with pytest.raises(RD.RoutingDecisionsError) as e:
        RD.handle(conn, {"since": "last tuesday"})
    assert e.value.status == 400 and "ISO 8601" in e.value.message
    assert conn.executed == []


def test_malformed_limit_and_offset_are_400(conn):
    for params, name in (({"limit": "many"}, "limit"), ({"offset": "later"}, "offset")):
        with pytest.raises(RD.RoutingDecisionsError) as e:
            RD.handle(conn, params)
        assert e.value.status == 400 and name in e.value.message


def test_empty_mode_falls_back_to_list(conn):
    """API Gateway omits absent params; an explicitly blank one is not an error."""
    assert RD.parse_params({"mode": ""})["mode"] == "list"
    assert RD.parse_params(None)["mode"] == "list"


# ─────────────────────────────────────────────────────────────────────────────
# Lambda wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_handler_routes_get_routing_decisions(monkeypatch):
    """The Lambda entry point reaches the module and maps its status codes."""
    import types
    # The handler's import chain reaches boto3/botocore, which are Lambda-runtime
    # dependencies not installed for unit tests. Stub the names it touches.
    for name in ("boto3", "botocore", "botocore.exceptions", "botocore.config",
                 "mcp_observatory", "mcp_observatory.instrument"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["botocore.exceptions"].ClientError = type("ClientError", (Exception,), {})
    sys.modules["botocore.config"].Config = type("Config", (), {"__init__": lambda self, **kw: None})
    sys.modules["boto3"].client = lambda *a, **kw: None
    sys.modules["mcp_observatory.instrument"].instrument_wrapper_api = lambda *a, **kw: None
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import handler as H  # noqa: E402

    monkeypatch.setattr(RD._feedback, "ensure_schema", lambda c: None)
    conn = FakeConn().add("q1", "architecture", "graph_first", confidence=0.8,
                          rating=1.0, rated_at="t")
    monkeypatch.setattr(H, "_db_clients",
                        lambda: types.SimpleNamespace(get_pg_connection=lambda: conn))

    def get(qs):
        return H.lambda_handler(
            {"requestContext": {"http": {"method": "GET"}},
             "rawPath": "/routing-decisions", "queryStringParameters": qs}, None)

    resp = get(None)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["count"] == 1

    resp = get({"mode": "summary"})
    assert json.loads(resp["body"])["groups"][0]["meanAbsDiff"] == 0.2

    resp = get({"mode": "nope"})
    assert resp["statusCode"] == 400
    assert json.loads(resp["body"]) == {"error": "mode must be 'list' or 'summary'"}
