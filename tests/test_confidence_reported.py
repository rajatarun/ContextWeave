"""The router must learn only from confidences the model actually reported.

Three fallbacks used to reach the posterior as observations: 0.7 when the
model omitted the field, 0.5 when its output was not JSON, 0.0 when the call
failed. Each is a constant the code chose. These tests pin the flag that
keeps them out of the learning loop.
"""
from __future__ import annotations

import math
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "query_api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shared"))


@pytest.fixture(scope="module")
def S():
    """Import synthesizer with its Lambda-runtime imports stubbed, then undo it.

    This used to stub with `sys.modules.setdefault(name, ModuleType(name))` and
    never restore, which left a bare `boto3` module in sys.modules for the rest
    of the session. tests/test_mcp_observatory.py does `import boto3` and
    `monkeypatch.setattr(boto3, "resource", ...)`, so when it ran after this
    file it failed six times with "<module 'boto3'> has no attribute
    'resource'" -- a real-looking error in a file whose own code is fine.
    Running that file alone passed; running the suite failed.

    pytest.MonkeyPatch() rather than the `monkeypatch` fixture because that one
    is function-scoped and this fixture is module-scoped; `undo()` in the
    teardown restores whatever was there before, stub or real.
    """
    mp = pytest.MonkeyPatch()
    # synthesizer imports boto3/botocore and the observatory shim at module load.
    for name in ("boto3", "botocore", "botocore.exceptions", "botocore.config",
                 "mcp_observatory", "mcp_observatory.instrument"):
        if name not in sys.modules:
            mp.setitem(sys.modules, name, types.ModuleType(name))
    mp.setattr(sys.modules["botocore.exceptions"], "ClientError",
               type("ClientError", (Exception,), {}), raising=False)
    mp.setattr(sys.modules["botocore.config"], "Config",
               type("Config", (), {"__init__": lambda self, **kw: None}), raising=False)
    mp.setattr(sys.modules["boto3"], "client", lambda *a, **kw: None, raising=False)
    mp.setattr(sys.modules["mcp_observatory.instrument"], "instrument_wrapper_api",
               lambda *a, **kw: None, raising=False)

    had_synthesizer = "synthesizer" in sys.modules
    import synthesizer
    yield synthesizer

    # synthesizer closed over the stubs at import time, so leaving it cached
    # would hand the next importer a module wired to modules that no longer
    # exist. Drop it only if this fixture is what imported it.
    if not had_synthesizer:
        sys.modules.pop("synthesizer", None)
    mp.undo()


def test_reported_number_in_range_is_reported(S):
    assert S._confidence_from({"confidence": 0.83}) == (0.83, True)
    assert S._confidence_from({"confidence": "0.4"}) == (0.4, True)
    assert S._confidence_from({"confidence": 0}) == (0.0, True)
    assert S._confidence_from({"confidence": 1}) == (1.0, True)


def test_omitted_field_is_a_fallback_not_an_observation(S):
    value, reported = S._confidence_from({"answer": "x"})
    assert value == S._OMITTED_CONFIDENCE and reported is False


def test_unparseable_output_is_a_fallback_not_an_observation(S):
    parsed = S._parse_model_response("this is not json {")
    assert parsed["_unparsed"] is True and "confidence" not in parsed
    value, reported = S._confidence_from(parsed)
    assert value == S._UNPARSED_CONFIDENCE and reported is False


@pytest.mark.parametrize("bad", ["high", None, True, 1.7, -0.1, float("nan"), [0.5]])
def test_garbage_confidence_is_not_reported_and_does_not_crash(S, bad):
    value, reported = S._confidence_from({"confidence": bad})
    assert reported is False and 0.0 <= value <= 1.0 and not math.isnan(value)


def test_response_dict_carries_the_flag(S):
    from models import QueryResponse
    r = QueryResponse(answer="a", sources=[], inferred_skills=[], repeated_patterns=[],
                      confidence=0.7, question_type="general", confidence_reported=False)
    assert r.to_dict()["confidenceReported"] is False
    assert QueryResponse(answer="a", sources=[], inferred_skills=[], repeated_patterns=[],
                         confidence=0.9, question_type="general").to_dict()["confidenceReported"] is True
