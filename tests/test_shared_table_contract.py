"""ContextWeave's writer against the shared OBSERVATORY_METRICS table contract.

``src/shared/mcp_observatory.py`` writes span rows to a DynamoDB table shared
with other weave services (see ``contracts/observatory_metrics_item.json``).
Nothing here can see the readers' code, and the readers cannot see this
writer's -- the only thing keeping the two mutually legible is that both
sides check themselves against the vendored contract in ``contracts/``.

A sibling repo was found writing ``PK``/``SK`` (upper case) inside a bare
``except: pass`` -- DynamoDB's PutItem rejects that spelling with a
ValidationException, and the swallowed exception meant every write silently
vanished while the service kept reporting success. This test exercises the
*real* writer function (not a hand-built dict standing in for one) so that
kind of drift fails here instead of in production.

Mocking follows tests/test_mcp_observatory.py's existing style: stub the
``mcp_observatory`` package (the third-party instrumentation library) and
``boto3.resource`` before importing ``shared.mcp_observatory``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _contracts_on_path():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    yield


from contracts.conformance import check_item, load_contract  # noqa: E402


class _FakeSpan:
    trace_id = "trace-abc"
    prompt_tokens = 11
    completion_tokens = 22
    cost_usd = 0.0123
    hallucination_risk_score = 0.1
    hallucination_risk_level = "low"
    composite_risk_score = 0.2
    composite_risk_level = "low"
    policy_decision = "allow"


class _FakeDecision:
    action = "allow"
    reason = "ok"


class _FakeWrapper:
    """Mirrors FakeWrapper in test_mcp_observatory.py."""

    async def invoke(self, *, source, model, prompt, input_payload, call):
        output = call()
        return SimpleNamespace(output=output, span=_FakeSpan(), decision=_FakeDecision())


class _FakeDDBTable:
    """Mirrors FakeDDBTable in test_mcp_observatory.py; captures the raw put_item."""

    def __init__(self):
        self.items: list[dict] = []

    def put_item(self, Item):
        self.items.append(Item)


class _FakeRuntime:
    def converse(self, **kwargs):
        return {"output": {"message": {"content": [{"text": "hello"}]}}}

    def invoke_model(self, **kwargs):
        return {"body": SimpleNamespace(read=lambda: b'{"ok": true}')}


def _load_writer(monkeypatch, *, table_name: str = "obs-table"):
    """Import shared.mcp_observatory with boto3/mcp_observatory mocked out,
    exactly as tests/test_mcp_observatory.py does, and return the module plus
    the FakeDDBTable it will write into.
    """
    fake_wrapper = _FakeWrapper()
    fake_table = _FakeDDBTable()

    instrument_mod = ModuleType("mcp_observatory.instrument")
    instrument_mod.instrument_wrapper_api = lambda _name: fake_wrapper

    pkg = ModuleType("mcp_observatory")
    pkg.instrument = instrument_mod

    monkeypatch.setitem(sys.modules, "mcp_observatory", pkg)
    monkeypatch.setitem(sys.modules, "mcp_observatory.instrument", instrument_mod)

    import boto3

    class FakeResource:
        def Table(self, _):
            return fake_table

    monkeypatch.setattr(boto3, "resource", lambda _svc: FakeResource(), raising=False)

    monkeypatch.setenv("OBSERVATORY_METRICS_TABLE", table_name)

    src_root = str(REPO_ROOT / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    mod = importlib.import_module("shared.mcp_observatory")
    mod = importlib.reload(mod)
    return mod, fake_table


def test_real_writer_produces_a_conforming_item(monkeypatch):
    """Exercise the real observe_converse_request -> _push_metric path and
    check the exact item it put against the vendored contract.
    """
    mod, table = _load_writer(monkeypatch)
    runtime = _FakeRuntime()

    mod.observe_converse_request(
        runtime_client=runtime,
        model_id="us.amazon.nova-pro-v1:0",
        prompt="what does this developer know about AWS?",
        request_body={"messages": []},
        source="synthesis",
        operation="synthesize_answer",
    )

    assert len(table.items) == 1
    item = table.items[0]

    problems = check_item(item, load_contract())
    assert problems == [], problems


def test_key_attributes_are_lower_case_pk_sk(monkeypatch):
    """I1: the table's key attributes are 'pk'/'sk', not 'PK'/'SK'.

    A writer using the upper-case spelling has DynamoDB's PutItem reject the
    item with a ValidationException; if that call sits inside a bare
    ``except: pass`` the write vanishes silently and the service keeps
    reporting success. Pinning the exact attribute names here is what keeps
    that regression from recurring in this writer.
    """
    mod, table = _load_writer(monkeypatch)
    runtime = _FakeRuntime()

    mod.observe_model_request(runtime_client=runtime, model_id="m", body="{}")

    item = table.items[0]
    assert "pk" in item and "sk" in item
    assert "PK" not in item and "SK" not in item
    assert isinstance(item["pk"], str) and isinstance(item["sk"], str)


def test_written_item_carries_the_span_timeline_index_keys(monkeypatch):
    """Contract v2.0.0: reads go through the SpanTimelineIndex GSI, not pk.

    A GSI indexes only items carrying both of its key attributes, so this
    replaces the old pk/readers_for reachability check (superseded -- I5 in
    the v2 contract) with an assertion on the two attributes that now decide
    whether a dashboard ever sees this row: span_date must be present and
    must agree with timestamp's date, exactly as I6/I7 require.
    """
    contract = load_contract()
    mod, table = _load_writer(monkeypatch)
    runtime = _FakeRuntime()

    mod.observe_model_request(runtime_client=runtime, model_id="m", body="{}")

    item = table.items[0]
    gsi = contract["gsi"]
    assert gsi["partition_key"] in item
    assert gsi["sort_key"] in item
    assert item["span_date"] == item["timestamp"][:10]
    assert check_item(item, contract) == []
