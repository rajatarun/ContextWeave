from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from decimal import Decimal
from types import ModuleType, SimpleNamespace


@dataclass
class FakeSpan:
    trace_id: str = "trace-123"
    prompt_tokens: int = 11
    completion_tokens: int = 22
    cost_usd: float = 0.0123


@dataclass
class FakeDecision:
    action: str = "allow"
    reason: str = "ok"


class FakeWrapper:
    def __init__(self):
        self.calls = []

    async def invoke(self, *, source, model, prompt, input_payload, call):
        self.calls.append(
            {
                "source": source,
                "model": model,
                "prompt": prompt,
                "input_payload": input_payload,
            }
        )
        output = call()
        return SimpleNamespace(output=output, span=FakeSpan(), decision=FakeDecision())


class FakeDDBTable:
    def __init__(self, should_fail: bool = False):
        self.items = []
        self.should_fail = should_fail

    def put_item(self, Item):
        if self.should_fail:
            raise RuntimeError("ddb down")
        self.items.append(Item)


class FakeRuntime:
    def __init__(self):
        self.invoke_model_calls = []
        self.converse_calls = []
        self.raise_error = None

    def invoke_model(self, **kwargs):
        if self.raise_error:
            raise self.raise_error
        self.invoke_model_calls.append(kwargs)
        return {"body": SimpleNamespace(read=lambda: b'{"ok": true}')}

    def converse(self, **kwargs):
        if self.raise_error:
            raise self.raise_error
        self.converse_calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": "general"}]}}}


def _load_module(monkeypatch, *, table_name: str | None = "obs-table", ddb_fail: bool = False):
    fake_wrapper = FakeWrapper()
    fake_table = FakeDDBTable(should_fail=ddb_fail)

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

    monkeypatch.setattr(boto3, "resource", lambda _svc: FakeResource())

    if table_name is None:
        monkeypatch.delenv("OBSERVATORY_METRICS_TABLE", raising=False)
    else:
        monkeypatch.setenv("OBSERVATORY_METRICS_TABLE", table_name)

    src_root = str(Path(__file__).resolve().parents[1] / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    mod = importlib.import_module("shared.mcp_observatory")
    mod = importlib.reload(mod)
    return mod, fake_wrapper, fake_table


def test_observe_model_request_returns_result_output_and_invokes_wrapper(monkeypatch):
    mod, wrapper, table = _load_module(monkeypatch)
    runtime = FakeRuntime()

    result = mod.observe_model_request(
        runtime_client=runtime,
        model_id="amazon.titan-embed-text-v2:0",
        body='{"inputText":"hello"}',
        content_type="application/json",
        accept="application/json",
    )

    assert result["body"].read() == b'{"ok": true}'
    assert wrapper.calls[0]["source"] == "model"
    assert wrapper.calls[0]["model"] == "amazon.titan-embed-text-v2:0"
    assert wrapper.calls[0]["prompt"] == '{"inputText":"hello"}'
    assert len(table.items) == 1


def test_structured_log_and_metric_shape(monkeypatch):
    mod, _, table = _load_module(monkeypatch)
    runtime = FakeRuntime()
    logged = {}

    def fake_info(_msg, *, extra):
        logged.update(extra)

    monkeypatch.setattr(mod.log, "info", fake_info)

    mod.observe_converse_request(
        runtime_client=runtime,
        model_id="us.amazon.nova-pro-v1:0",
        prompt="my prompt",
        request_body={"messages": []},
        source="synthesis",
        operation="synthesize_answer",
    )

    assert logged["trace_id"] == "trace-123"
    assert logged["prompt_tokens"] == 11
    assert logged["completion_tokens"] == 22
    assert logged["cost_usd"] == 0.0123
    assert logged["decision"] == "allow"
    assert logged["decision_reason"] == "ok"

    item = table.items[0]
    assert item["pk"] == "OBSERVATORY#synthesize_answer"
    assert item["trace_id"] == "trace-123"
    assert item["decision"] == "allow"
    assert "#trace-123" in item["sk"]


def test_skip_write_when_table_env_absent(monkeypatch):
    mod, _, table = _load_module(monkeypatch, table_name=None)
    runtime = FakeRuntime()

    mod.observe_model_request(runtime_client=runtime, model_id="m", body="{}")

    assert table.items == []


def test_swallow_ddb_failures(monkeypatch):
    mod, _, _ = _load_module(monkeypatch, ddb_fail=True)
    runtime = FakeRuntime()
    warnings = []
    monkeypatch.setattr(mod.log, "warning", lambda *args, **kwargs: warnings.append((args, kwargs)))

    mod.observe_model_request(runtime_client=runtime, model_id="m", body="{}")

    assert warnings


def test_ttl_is_approximately_90_days(monkeypatch):
    mod, _, table = _load_module(monkeypatch)
    runtime = FakeRuntime()
    before = int(time.time())

    mod.observe_model_request(runtime_client=runtime, model_id="m", body="{}")

    ttl = int(table.items[0]["ttl"])
    delta = ttl - before
    assert 89 * 24 * 60 * 60 <= delta <= 91 * 24 * 60 * 60


def test_model_exceptions_are_reraised(monkeypatch):
    mod, _, _ = _load_module(monkeypatch)
    runtime = FakeRuntime()
    runtime.raise_error = RuntimeError("invoke failed")

    try:
        mod.observe_model_request(runtime_client=runtime, model_id="m", body="{}")
        assert False, "expected exception"
    except RuntimeError as exc:
        assert str(exc) == "invoke failed"
