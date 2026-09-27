"""The routing reward is built from checkable signals, and missing stays missing.

What is under test is the contract the handler relies on:
  * grounding measures support by the retrieved passages, not the model's opinion;
  * a signal that was not observed is None, and a reward made of nothing is None,
    so the router is left alone rather than trained on a default;
  * the mode decides what the reward is made of, never what gets recorded;
  * the judge is sampled deterministically, and a failed judge is not a reward;
  * the handler folds the verified reward -- not the self-confidence -- into the
    posterior, and skips the update when no signal was observed.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "query_api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import verified_reward as V  # noqa: E402

PASSAGE = (
    "Tarun designed the ContextWeave routing layer on AWS Neptune Analytics. "
    "The router uses Thompson sampling over Beta posteriors per question type."
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("ROUTER_"):
            monkeypatch.delenv(k, raising=False)


# ── grounding ────────────────────────────────────────────────────────────────

def test_supported_answer_grounds_fully():
    s = V.grounding_signal("The router uses Thompson sampling over Beta posteriors.", [PASSAGE])
    assert s.value == 1.0 and s.detail["claims"] == 1


def test_unsupported_claim_lowers_grounding_whatever_the_model_thinks():
    answer = ("The router uses Thompson sampling over Beta posteriors. "
              "It was deployed on Kubernetes across twelve Azure regions in 2019.")
    s = V.grounding_signal(answer, [PASSAGE])
    assert s.value == 0.5 and s.detail["supported"] == 1


def test_a_number_the_passages_never_mention_is_not_supported():
    s = V.grounding_signal("Tarun designed the routing layer serving 400 teams.", [
        "Tarun designed the routing layer serving 40 teams."])
    assert s.value == 0.0


def test_no_passages_and_no_claims_are_not_observations():
    assert V.grounding_signal("Anything at all is claimed here.", []).value is None
    assert V.grounding_signal("Anything at all is claimed here.", ["", "  "]).value is None
    abstain = "The evidence is insufficient to answer this question."
    assert V.grounding_signal(abstain, [PASSAGE]).value is None


def test_verifier_is_pluggable():
    s = V.grounding_signal("Completely different words entirely here.", [PASSAGE],
                           verifier=lambda claim, passage: 1.0)
    assert s.value == 1.0


# ── combine / missing is missing ─────────────────────────────────────────────

def test_combine_ignores_unobserved_and_returns_none_for_nothing():
    assert V.combine([V.RewardSignal("a", None, 1.0), V.RewardSignal("b", 0.9, 0.0)]) == (None, [])
    reward, used = V.combine([V.RewardSignal("grounding", 0.5, 1.0), V.RewardSignal("judge", 1.0, 2.0),
                              V.RewardSignal("self", None, 1.0)])
    assert used == ["grounding", "judge"] and reward == pytest.approx(2.5 / 3)


def test_fallback_self_confidence_is_never_a_signal():
    assert V.self_signal(0.7, reported=False).value is None
    assert V.self_signal(None, reported=True).value is None
    assert V.self_signal(1.4, reported=True).value == 1.0


# ── modes ────────────────────────────────────────────────────────────────────

def _assess(**kw):
    base = dict(query_id="q", question="How does routing work?",
                answer="The router uses Thompson sampling over Beta posteriors.",
                passages=[PASSAGE], self_confidence=0.2, self_reported=True)
    base.update(kw)
    return V.assess(**base)


def test_verified_mode_ignores_self_confidence_but_records_it():
    a = _assess(mode="verified")
    assert a.reward == 1.0 and a.sources == ["grounding"]
    assert a.value_of("self") == 0.2, "self-confidence must still be recorded for calibration"


def test_self_mode_is_the_old_behaviour_and_still_records_grounding():
    a = _assess(mode="self")
    assert a.reward == 0.2 and a.sources == ["self"] and a.value_of("grounding") == 1.0


def test_verified_plus_self_averages_both():
    a = _assess(mode="verified+self")
    assert a.reward == pytest.approx(0.6) and set(a.sources) == {"self", "grounding"}


def test_nothing_observed_means_no_reward():
    a = _assess(mode="verified", passages=[])
    assert a.reward is None and a.sources == []


def test_unknown_mode_falls_back_to_verified_not_self(monkeypatch):
    monkeypatch.setenv("ROUTER_REWARD_SOURCE", "selff")
    assert V.reward_mode() == "verified"


# ── judge ────────────────────────────────────────────────────────────────────

def test_judge_sampling_is_deterministic_and_roughly_the_rate():
    ids = [f"query-{i}" for i in range(4000)]
    picked = [q for q in ids if V.should_judge(q, 0.1)]
    assert picked == [q for q in ids if V.should_judge(q, 0.1)]
    assert 300 < len(picked) < 500
    assert not V.should_judge("q", 0.0) and V.should_judge("q", 1.0)


def test_judge_reply_parsing_refuses_to_guess():
    assert V.parse_judge_score('{"score": 0.8, "unsupported_claims": []}') == 0.8
    assert V.parse_judge_score('Sure! ```json\n{"score": 1}\n```') == 1.0
    for bad in ("", "no json", '{"score": 7}', '{"score": true}', '{"other": 1}'):
        assert V.parse_judge_score(bad) is None, bad


def test_judge_joins_the_reward_on_sampled_queries(monkeypatch):
    monkeypatch.setenv("ROUTER_JUDGE_SAMPLE_RATE", "1")
    a = _assess(mode="verified", judge_invoke=lambda prompt: '{"score": 0.4}')
    assert a.value_of("judge") == 0.4
    assert a.reward == pytest.approx((1.0 * 1 + 0.4 * 2) / 3)


def test_failed_judge_is_not_a_reward(monkeypatch):
    monkeypatch.setenv("ROUTER_JUDGE_SAMPLE_RATE", "1")

    def boom(prompt):
        raise RuntimeError("throttled")

    a = _assess(mode="verified", judge_invoke=boom)
    assert a.value_of("judge") is None and a.sources == ["grounding"]


def test_judge_is_not_called_when_unsampled_or_in_self_mode(monkeypatch):
    calls = []
    judge = lambda p: calls.append(p) or '{"score": 1}'  # noqa: E731
    _assess(mode="verified", judge_invoke=judge)  # rate defaults to 0
    monkeypatch.setenv("ROUTER_JUDGE_SAMPLE_RATE", "1")
    _assess(mode="self", judge_invoke=judge)
    assert calls == []


def test_judge_prompt_carries_the_evidence_not_the_self_confidence(monkeypatch):
    monkeypatch.setenv("ROUTER_JUDGE_SAMPLE_RATE", "1")
    seen = []
    _assess(mode="verified", self_confidence=0.123, judge_invoke=lambda p: seen.append(p) or '{"score": 1}')
    assert PASSAGE in seen[0] and "0.123" not in seen[0]


# ── handler wiring ───────────────────────────────────────────────────────────

_HANDLER_MODULES = ("handler", "synthesizer", "retriever", "graph_expander", "rag_router",
                    "cache", "feedback", "routing_decisions_api", "agent_card", "models")


@pytest.fixture(scope="module")
def H():
    """Import the real handler with its Lambda-runtime imports stubbed, then undo it.

    Same approach as tests/test_confidence_reported.py: stub only what is
    absent, restore afterwards, and drop the modules this fixture imported so
    none of them stays wired to a stub for the rest of the session.
    """
    import types
    mp = pytest.MonkeyPatch()
    for name in ("boto3", "botocore", "botocore.exceptions", "botocore.config",
                 "mcp_observatory", "mcp_observatory.instrument"):
        if name not in sys.modules:
            mp.setitem(sys.modules, name, types.ModuleType(name))
    mp.setattr(sys.modules["botocore.exceptions"], "ClientError",
               type("ClientError", (Exception,), {}), raising=False)
    mp.setattr(sys.modules["botocore.config"], "Config",
               type("Config", (), {"__init__": lambda self, **kw: None}), raising=False)
    mp.setattr(sys.modules["boto3"], "client", lambda *a, **kw: None, raising=False)
    mp.setattr(sys.modules["boto3"], "resource", lambda *a, **kw: None, raising=False)
    mp.setattr(sys.modules["mcp_observatory.instrument"], "instrument_wrapper_api",
               lambda *a, **kw: None, raising=False)
    already = {m for m in _HANDLER_MODULES if m in sys.modules}
    import handler
    yield handler
    for m in _HANDLER_MODULES:
        if m not in already:
            sys.modules.pop(m, None)
    mp.undo()


@pytest.fixture
def pipeline(monkeypatch, H):
    from models import QueryRequest, QueryResponse, RetrievedChunk

    updates, records = [], []
    monkeypatch.setattr(H, "NEPTUNE_GRAPH_ID", "g-1")
    monkeypatch.setattr(H._cache, "is_time_sensitive", lambda q: True)
    monkeypatch.setattr(H, "classify_question", lambda q: "architecture")
    monkeypatch.setattr(H, "select_strategy", lambda **kw: SimpleNamespace(
        strategy="graph_first", include_graph=False, boost_keywords=False, use_neptune_chunks=False,
        strategy_confidence=0.6, selection_propensity=0.5))
    monkeypatch.setattr(H, "retrieve_with_strategy", lambda **kw: [
        RetrievedChunk(content=PASSAGE, score=0.9, source_uri="s3://x/a.md")])
    monkeypatch.setattr(H, "deduplicate_chunks", lambda c: c)
    monkeypatch.setattr(H, "update_feedback", lambda **kw: updates.append(kw))
    monkeypatch.setattr(H._feedback, "record_decision", lambda conn, **kw: records.append(kw))
    monkeypatch.setattr(H, "_db_clients", lambda: SimpleNamespace(get_pg_connection=lambda: None))

    def run(answer, confidence=0.95, reported=True, chunks=None):
        if chunks is not None:
            monkeypatch.setattr(H, "retrieve_with_strategy", lambda **kw: chunks)
        monkeypatch.setattr(H, "synthesize_answer", lambda **kw: QueryResponse(
            answer=answer, sources=[], inferred_skills=[], repeated_patterns=[],
            confidence=confidence, confidence_reported=reported, question_type="architecture",
            graph_entities_used=[], retrieval_count=1, model_id="m"))
        out = H._run_query_pipeline(QueryRequest(question="How does routing work?",
                                                 include_graph_expansion=False))
        return out, updates, records

    return run


def test_confident_but_ungrounded_answer_is_not_rewarded_as_confident(pipeline):
    out, updates, records = pipeline("It was deployed on Kubernetes across twelve Azure regions in 2019.",
                                     confidence=0.95)
    assert len(updates) == 1
    assert updates[0]["confidence"] == 0.0 and updates[0]["source"] == "verified"
    assert records[0]["confidence"] == 0.95 and records[0]["grounding"] == 0.0
    assert records[0]["reward"] == 0.0 and records[0]["reward_mode"] == "verified"
    assert out["routingDecision"]["reward"]["sources"] == ["grounding"]


def test_no_observed_signal_leaves_the_router_alone(pipeline):
    out, updates, records = pipeline("Insufficient evidence to answer.", chunks=[])
    assert updates == []
    assert records[0]["reward"] is None and records[0]["grounding"] is None
