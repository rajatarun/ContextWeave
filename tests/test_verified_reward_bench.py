"""The offline benchmark's metrics mean what its docstring says they mean."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verified_reward_bench.py"
_spec = importlib.util.spec_from_file_location("verified_reward_bench", _PATH)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def test_squad_f1_and_abstention():
    assert B.token_f1("the Neptune Analytics graph", "Neptune Analytics") == pytest.approx(0.8)
    assert B.correctness("Neptune Analytics", ["Amazon Neptune Analytics", "Neptune"]) == pytest.approx(0.8)
    assert B.correctness("Insufficient evidence to answer.", []) == 1.0
    assert B.correctness("Paris", []) == 0.0
    assert B.correctness("Insufficient evidence to answer.", ["Paris"]) == 0.0


def test_metrics_on_known_values():
    perfect = [(1.0, 1.0), (0.0, 0.0)] * 5
    assert B.brier(perfect) == 0.0 and B.ece(perfect) == 0.0 and B.auroc(perfect) == 1.0
    inverted = [(0.0, 1.0), (1.0, 0.0)] * 5
    assert B.auroc(inverted) == 0.0 and B.spearman(inverted) == pytest.approx(-1.0)
    constant = [(0.9, 1.0), (0.9, 0.0)] * 5
    assert B.auroc(constant) == 0.5 and B.ece(constant) == pytest.approx(0.4)
    assert B.brier([]) is None and B.auroc([(0.5, 1.0)]) is None


P = "The router uses Thompson sampling over Beta posteriors, and ContextWeave stores chunks in pgvector."


def test_calibrate_finds_a_confident_signal_that_cannot_rank_strategies(tmp_path):
    rows = []
    for i in range(6):
        # graph_first answers correctly and grounded; semantic_search answers wrong
        # and ungrounded. The model is equally confident about both.
        rows.append({"id": f"g{i}", "strategy": "graph_first", "passages": [P],
                     "answer": "Thompson sampling over Beta posteriors", "self_confidence": 0.9,
                     "gold_answers": ["Thompson sampling over Beta posteriors"]})
        rows.append({"id": f"s{i}", "strategy": "semantic_search", "passages": [P],
                     "answer": "An epsilon greedy schedule tuned weekly by hand", "self_confidence": 0.9,
                     "gold_answers": ["Thompson sampling over Beta posteriors"]})
    rows.append({"id": "missing", "strategy": "hybrid", "passages": [P], "answer": "pgvector",
                 "self_confidence": None, "gold_answers": ["pgvector"]})
    f = tmp_path / "runs.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))

    out = B.calibrate(B.load_records(str(f)))
    self_m, ground_m = out["signals"]["self"], out["signals"]["grounding"]
    assert self_m["coverage"] < 1.0, "an unreported confidence must be missing, not a number"
    assert self_m["auroc"] == 0.5, "a constant confidence cannot separate right from wrong"
    assert ground_m["auroc"] == 1.0
    assert ground_m["bestStrategyAgrees"] is True
    assert out["signals"]["judge"]["coverage"] == 0.0


def test_simulation_is_reproducible():
    kw = dict(overconfidence=0.7, self_bias=0.25, grounding_noise=0.2, judge_rate=0.05, judge_noise=0.1)
    assert B.simulate(2, 100, **kw) == B.simulate(2, 100, **kw)
