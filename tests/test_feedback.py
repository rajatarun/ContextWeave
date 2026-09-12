"""POST /feedback: a reward for the router that is not the model's opinion of itself.

The database and the graph are both stubbed. What is under test is the
contract: a rating reaches the posterior of the strategy that produced the
answer, with the configured weight, exactly once, and is refused otherwise.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "query_api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shared"))

import feedback as F  # noqa: E402
import rag_router as R  # noqa: E402


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.db.executed.append((" ".join(sql.split()), params))
        s = sql.strip().upper()
        if s.startswith("INSERT"):
            qid = params[0]
            self.db.rows.setdefault(qid, {
                "question_type": params[1], "strategy": params[2],
                "propensity": params[3], "confidence": params[4], "rating": None, "rated_at": None,
            })
        elif s.startswith("SELECT"):
            r = self.db.rows.get(params[0])
            self._row = None if r is None else (r["question_type"], r["strategy"], r["confidence"], r["rated_at"])
        elif s.startswith("UPDATE"):
            self.db.rows[params[1]]["rating"] = params[0]
            self.db.rows[params[1]]["rated_at"] = "now"

    def fetchone(self):
        return self._row


class FakeConn:
    def __init__(self):
        self.rows: dict = {}
        self.executed: list = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


@pytest.fixture
def graph(monkeypatch):
    """Capture the parameters update_feedback sends to the graph and return a posterior."""
    calls = []

    def fake_query(query, params=None):
        if "SET e.alpha" in query:
            calls.append(params)
            return [{"alpha": 3.0, "beta": 2.0, "weight": 0.6, "n": 5, "n_human": 1}]
        return []

    monkeypatch.setattr(R, "_run_query", fake_query)
    return calls


def test_rating_vocabulary():
    assert F.rating_to_reward("up") == 1.0 and F.rating_to_reward("DOWN") == 0.0
    assert F.rating_to_reward("neutral") == 0.5
    assert F.rating_to_reward(True) == 1.0 and F.rating_to_reward(False) == 0.0
    assert F.rating_to_reward(0.25) == 0.25
    assert F.rating_to_reward(5) is None, "a 5-star value must be rejected, not clamped to 1.0"
    assert F.rating_to_reward("meh") is None and F.rating_to_reward(None) is None


def test_rating_reaches_the_posterior_of_the_strategy_that_answered(graph):
    conn = FakeConn()
    F.record_decision(conn, query_id="q1", question_type="architecture", strategy="graph_first",
                      propensity=0.7, confidence=0.9)
    out = F.apply_rating(conn, query_id="q1", rating="down")
    assert len(graph) == 1
    p = graph[0]
    assert p["strategy"] == "graph_first" and p["question_type"] == "architecture"
    assert p["reward"] == 0.0 and p["human"] == 1 and p["w"] == F.HUMAN_WEIGHT
    assert out["selfConfidence"] == 0.9 and out["reward"] == 0.0
    assert out["posterior"]["nHuman"] == 1
    assert conn.rows["q1"]["rating"] == 0.0 and conn.rows["q1"]["rated_at"] is not None


def test_self_feedback_still_has_weight_one_and_is_not_human(graph):
    R.update_feedback(strategy="hybrid", question_type="project", confidence=0.8)
    assert graph[0]["w"] == 1.0 and graph[0]["human"] == 0


def test_human_weight_is_configurable_per_call(graph):
    conn = FakeConn()
    F.record_decision(conn, query_id="q2", question_type="general", strategy="semantic_search",
                      propensity=0.5, confidence=0.5)
    F.apply_rating(conn, query_id="q2", rating="up", weight=3.5)
    assert graph[0]["w"] == 3.5 and graph[0]["reward"] == 1.0


def test_unknown_query_is_404_and_writes_nothing(graph):
    conn = FakeConn()
    with pytest.raises(F.FeedbackError) as e:
        F.apply_rating(conn, query_id="missing", rating="up")
    assert e.value.status == 404 and graph == []


def test_second_rating_is_409_and_does_not_move_the_posterior_again(graph):
    conn = FakeConn()
    F.record_decision(conn, query_id="q3", question_type="comparison", strategy="hybrid",
                      propensity=0.4, confidence=0.7)
    F.apply_rating(conn, query_id="q3", rating="up")
    with pytest.raises(F.FeedbackError) as e:
        F.apply_rating(conn, query_id="q3", rating="down")
    assert e.value.status == 409
    assert len(graph) == 1, "the posterior must move exactly once per answer"


def test_bad_rating_is_400_before_touching_anything(graph):
    conn = FakeConn()
    with pytest.raises(F.FeedbackError) as e:
        F.apply_rating(conn, query_id="q", rating="five stars")
    assert e.value.status == 400 and conn.executed == [] and graph == []


def test_record_decision_is_idempotent_on_query_id():
    conn = FakeConn()
    for _ in range(2):
        F.record_decision(conn, query_id="dup", question_type="t", strategy="s", propensity=0.5, confidence=0.5)
    assert len(conn.rows) == 1
    assert any("ON CONFLICT (query_id) DO NOTHING" in sql for sql, _ in conn.executed)


def test_handler_routes_post_feedback(monkeypatch):
    """The Lambda entry point maps FeedbackError.status to the HTTP status."""
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

    monkeypatch.setattr(H, "_db_clients", lambda: types.SimpleNamespace(get_pg_connection=lambda: FakeConn()))
    event = {"requestContext": {"http": {"method": "POST"}}, "rawPath": "/feedback",
             "body": json.dumps({"queryId": "nope", "rating": "up"})}
    resp = H.lambda_handler(event, None)
    assert resp["statusCode"] == 404
    assert "unknown queryId" in json.loads(resp["body"])["details"]

    event["body"] = json.dumps({"queryId": "x", "rating": "five"})
    assert H.lambda_handler(event, None)["statusCode"] == 400

    event["body"] = "{not json"
    assert H.lambda_handler(event, None)["statusCode"] == 400
