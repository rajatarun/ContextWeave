"""ContextWeave's real handlers against contracts/contextweave_http_api.json.

``contracts/contextweave_http_api.json`` pins the response shape of the
endpoints TeamWeave consumes (see its own ``description`` for why: each side
used to test against its own private idea of the other, so a field rename on
either end left both suites green and broke production). This test drives
its assertions from that file's key lists -- never from keys retyped here --
so that editing the contract is what changes what this test enforces.

GET /health and GET /routing-decisions (mode=summary and mode=list) are
invoked through the real Lambda entry point, ``src/query_api/handler.py``.
Module stubbing for boto3/mcp_observatory follows the existing pattern in
tests/test_routing_decisions_api.py::test_handler_routes_get_routing_decisions.
The routing-decisions fixtures reuse that file's sqlite-backed FakeConn
rather than a new fake, per the task.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "contracts" / "contextweave_http_api.json"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "query_api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shared"))
sys.path.insert(0, os.path.dirname(__file__))  # reuse test_routing_decisions_api's FakeConn

from test_routing_decisions_api import FakeConn  # noqa: E402
import routing_decisions_api as RD  # noqa: E402


def _load_contract() -> dict:
    with open(CONTRACT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def handler_module(monkeypatch):
    """Import src/query_api/handler.py with only Lambda-runtime deps stubbed.

    Mirrors tests/test_routing_decisions_api.py::test_handler_routes_get_routing_decisions
    exactly: boto3/botocore/mcp_observatory are unit-test-environment stand-ins
    (not part of what's under test), everything else -- graph_expander,
    rag_router, retriever, synthesizer, routing_decisions_api -- is the real
    module.
    """
    for name in ("boto3", "botocore", "botocore.exceptions", "botocore.config",
                 "mcp_observatory", "mcp_observatory.instrument"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["botocore.exceptions"].ClientError = type("ClientError", (Exception,), {})
    sys.modules["botocore.config"].Config = type("Config", (), {"__init__": lambda self, **kw: None})
    sys.modules["boto3"].client = lambda *a, **kw: None
    sys.modules["mcp_observatory.instrument"].instrument_wrapper_api = lambda *a, **kw: None
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import importlib
    import handler as H
    importlib.reload(H)
    return H


def _get(handler_module, path: str, query_params: dict | None = None) -> dict:
    resp = handler_module.lambda_handler(
        {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": path,
            "queryStringParameters": query_params,
        },
        None,
    )
    return json.loads(resp["body"])


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_response_has_contract_required_top_level_keys(handler_module):
    contract = _load_contract()
    endpoint = contract["endpoints"]["GET /health"]

    body = _get(handler_module, "/health")

    for key in endpoint["required_top_level_keys"]:
        assert key in body, f"GET /health response is missing required key {key!r}: {body!r}"


# ─────────────────────────────────────────────────────────────────────────────
# GET /routing-decisions?mode=summary
# ─────────────────────────────────────────────────────────────────────────────

def test_routing_decisions_summary_has_contract_required_keys(handler_module, monkeypatch):
    contract = _load_contract()
    endpoint = contract["endpoints"]["GET /routing-decisions?mode=summary"]

    monkeypatch.setattr(RD._feedback, "ensure_schema", lambda c: None)
    conn = (
        FakeConn()
        .add("a", "architecture", "graph_first", confidence=0.8, rating=1.0, rated_at="t")
        .add("b", "general", "semantic_search", confidence=0.6)  # ratedCount == 0 group
    )
    monkeypatch.setattr(
        handler_module, "_db_clients",
        lambda: types.SimpleNamespace(get_pg_connection=lambda: conn),
    )

    body = _get(handler_module, "/routing-decisions", {"mode": "summary"})

    for key in endpoint["required_top_level_keys"]:
        assert key in body, f"summary response missing top-level key {key!r}: {body!r}"

    assert body["groups"], "expected at least one group from the seeded fake data"
    for group in body["groups"]:
        for key in endpoint["required_group_keys"]:
            assert key in group, f"summary group missing key {key!r}: {group!r}"

    # Documented null semantics: a group with ratedCount == 0 must report
    # avgRating/meanAbsDiff as None, never 0.0 (a spurious zero would read as
    # perfect agreement on exactly the group where nothing is known).
    unrated = next(g for g in body["groups"] if g["ratedCount"] == 0)
    assert unrated["avgRating"] is None
    assert unrated["meanAbsDiff"] is None


# ─────────────────────────────────────────────────────────────────────────────
# GET /routing-decisions?mode=list
# ─────────────────────────────────────────────────────────────────────────────

def test_routing_decisions_list_has_contract_required_keys(handler_module, monkeypatch):
    contract = _load_contract()
    endpoint = contract["endpoints"]["GET /routing-decisions?mode=list"]

    monkeypatch.setattr(RD._feedback, "ensure_schema", lambda c: None)
    conn = FakeConn().add(
        "q1", "architecture", "graph_first", propensity=0.31, confidence=0.82,
        rating=1.0, rated_at="2026-09-13T18:05:00+00:00",
        created_at="2026-09-13T18:00:00+00:00",
    )
    monkeypatch.setattr(
        handler_module, "_db_clients",
        lambda: types.SimpleNamespace(get_pg_connection=lambda: conn),
    )

    body = _get(handler_module, "/routing-decisions", {"mode": "list"})

    for key in endpoint["required_top_level_keys"]:
        assert key in body, f"list response missing top-level key {key!r}: {body!r}"

    assert body["items"], "expected at least one item from the seeded fake data"
    for item in body["items"]:
        for key in endpoint["required_item_keys"]:
            assert key in item, f"list item missing key {key!r}: {item!r}"
