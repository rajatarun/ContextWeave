"""Fixtures for the real-data routing protocol. Nothing here is a result."""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "query_api"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from experiments.analyses_stage import analyse
from experiments.assumed import SELF_C1, SELF_LOW_MEAN, assumed_slopes
from experiments.calibrate_stage import calibrate_dataset
from experiments.common import ProtocolError, load_config, price_for, validate_rows
from experiments.confidence import FAILED, OMITTED, UNPARSEABLE, parse_self_confidence
from experiments.data import SAMPLE_ORDER, chunk_windows, sample_rows, seeded_order
from experiments.generate_stage import build_user_message, dry_run, generate_rows, system_prompt
from experiments.metrics import is_abstention, kendall_tau, score_answer, signal_metrics
from experiments.prediction import dataset_verdict, overall_verdict
from experiments.records import build_rewards
from experiments.reporting import render
from experiments.replay_stage import (
    _Router, empirical_mu, general_priors, normalized_self_reward, run_one,
)
from experiments.retrieve_stage import bm25_scores, graph_scores, rank_pool
from experiments.signals_stage import check_judge_model, judge_sampled, lexical_value
import rag_router as R
import verified_reward as V


def test_snapshot_revision_reads_commit_or_cache_path():
    from experiments.common import snapshot_revision
    commit = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    path = f"/home/x/.cache/huggingface/hub/models--x/snapshots/{commit}/vocab.txt"
    assert snapshot_revision(commit) == commit
    assert snapshot_revision(None, path) == commit
    assert snapshot_revision("sentence-transformers/all-MiniLM-L6-v2") is None


def test_f1_and_abstention_detector():
    scored = score_answer("Paris", ["Paris"], False)
    assert scored["f1"] == 1.0 and scored["correct"] == 1
    partial = score_answer("Paris city", ["Paris"], False)
    assert partial["f1"] == pytest.approx(2 / 3) and partial["correct"] == 1
    wrong = score_answer("Rome", ["Paris"], False)
    assert wrong["f1"] == 0.0 and wrong["correct"] == 0
    # Boundary: F1 just below 0.5 is incorrect. "aa bb" vs "aa" is 2/3; "aa bb cc" vs "aa" is 0.5.
    boundary = score_answer("aa bb cc", ["aa"], False)
    assert boundary["f1"] == pytest.approx(0.5) and boundary["correct"] == 1
    below = score_answer("aa bb cc dd", ["aa"], False)
    assert below["f1"] < 0.5 and below["correct"] == 0
    assert is_abstention("Insufficient evidence to answer this.")
    assert is_abstention("")
    assert is_abstention("   ")
    assert not is_abstention("Paris")
    assert not is_abstention("November 11, 1901")
    long = (
        "Painting, poetry, and calligraphy were often practiced together "
        "by scholar officials in imperial China as the related arts of the literati class."
    )
    span = score_answer(long, ["painting, poetry, and calligraphy"], False)
    assert span["f1"] < 0.5 and span["correct"] == 0 and span["gold_contained"] == 1
    exact = score_answer("painting, poetry, and calligraphy", ["painting, poetry, and calligraphy"], False)
    assert exact["f1"] == 1.0 and exact["correct"] == 1 and exact["gold_contained"] == 1
    assert score_answer("sculpture and music", ["painting, poetry, and calligraphy"], False)["gold_contained"] == 0
    unans_ok = score_answer("Insufficient evidence to answer.", [], True)
    unans_bad = score_answer("Paris", [], True)
    assert unans_ok["correct"] == 1 and unans_bad["correct"] == 0
    answerable_abstain = score_answer("Insufficient evidence to answer.", ["Paris"], False)
    assert answerable_abstain["correct"] == 0


def test_lexical_tau_and_verbatim_numbers():
    high, reason, _ = lexical_value("alpha beta gamma delta epsilon", ["alpha beta gamma delta epsilon extra"], 0.6, 1.0)
    mid, _, _ = lexical_value("alpha beta gamma delta epsilon", ["alpha beta gamma other words here"], 0.6, 1.0)
    low, _, _ = lexical_value("alpha beta gamma delta epsilon", ["alpha beta other words here now"], 0.6, 1.0)
    assert high == 1.0 and mid == 1.0 and low == 0.0
    # 2 of 5 content tokens is below tau, so the single claim is unsupported.
    assert V.lexical_support("alpha beta gamma delta epsilon", "alpha beta other words here now") == pytest.approx(0.4)
    numbered, nreason, _ = lexical_value(
        "alpha beta counted 400 units total",
        ["alpha beta counted 40 units total"],
        0.6, 1.0,
    )
    assert numbered == 0.0
    assert V.lexical_support("alpha beta counted 400 units total", "alpha beta counted 40 units total") == 0.0
    missing, mreason, _ = lexical_value("alpha beta gamma", [], 0.6, 1.0)
    assert missing is None and mreason == "no_passages"
    abstained, areason, _ = lexical_value("Insufficient evidence to answer.", ["alpha beta gamma"], 0.6, 1.0)
    assert abstained is None and areason == "no_claims"
    # An environment threshold must not override the explicit tau.
    import os
    os.environ["ROUTER_GROUNDING_THRESHOLD"] = "0.1"
    try:
        still, _, _ = lexical_value("alpha beta gamma delta epsilon", ["alpha beta other words here now"], 0.6, 1.0)
    finally:
        os.environ.pop("ROUTER_GROUNDING_THRESHOLD", None)
    assert still == 0.0


def test_hash_sample_is_stable_and_ignores_seed():
    qid = "squad-5733be284776f41900661182"
    rate = 0.05
    digest = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    assert judge_sampled(qid, rate) is (digest < rate)
    assert judge_sampled(qid, rate) is judge_sampled(qid, rate)
    chosen = [f"q{i}" for i in range(500) if judge_sampled(f"q{i}", rate)]
    assert chosen == [f"q{i}" for i in range(500) if judge_sampled(f"q{i}", rate)]
    assert judge_sampled("", rate) is False


def test_calibration_metrics_and_ranking():
    rows = []
    for i in range(8):
        rows.append({"qid": f"g{i}", "arm": "graph_first", "f1": 1.0, "correct": 1, "value": 0.9})
        rows.append({"qid": f"s{i}", "arm": "semantic_search", "f1": 0.0, "correct": 0, "value": 0.1})
    metrics = signal_metrics(rows)
    assert metrics["coverage"] == 1.0
    assert metrics["brier"] == pytest.approx(0.01)
    assert metrics["auroc"] == 1.0
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["rank_agrees"] is True
    assert metrics["kendall_tau"] == pytest.approx(1.0)
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)
    flat = []
    for i in range(4):
        flat.append({"qid": f"a{i}", "arm": "semantic_search", "f1": 1.0, "correct": 1, "value": 0.9})
        flat.append({"qid": f"b{i}", "arm": "keyword_boosted", "f1": 0.0, "correct": 0, "value": 0.9})
    flat_m = signal_metrics(flat)
    assert flat_m["auroc"] == pytest.approx(0.5)
    assert flat_m["rank_agrees"] is False
    assert signal_metrics([{"qid": "only", "arm": "semantic_search", "f1": 1.0, "correct": 1, "value": None}])["coverage"] == 0.0
    point = calibrate_dataset(
        [{"qid": f"q{i}", "arm": "semantic_search", "signal": "self", "f1": float(i % 2), "correct": i % 2, "value": 0.2 * (i % 2)}
         for i in range(10)],
        seed=3, n_boot=30, high_coverage=0.8, low_auroc_max=0.6,
    )
    again = calibrate_dataset(
        [{"qid": f"q{i}", "arm": "semantic_search", "signal": "self", "f1": float(i % 2), "correct": i % 2, "value": 0.2 * (i % 2)}
         for i in range(10)],
        seed=3, n_boot=30, high_coverage=0.8, low_auroc_max=0.6,
    )
    assert point["signals"]["self"]["ci"]["brier"] == again["signals"]["self"]["ci"]["brier"]


def test_prediction_verdicts():
    good_self = {"coverage": 0.95, "auroc": 0.52, "rank_agrees": False}
    good_ground = {"coverage": 0.4, "auroc": 0.8, "rank_agrees": True}
    assert dataset_verdict(good_self, good_ground, 0.8, 0.6)["verdict"] == "supported"
    bad = dataset_verdict(good_self, {**good_ground, "rank_agrees": False}, 0.8, 0.6)
    assert bad["verdict"] == "contradicted"
    pending = dataset_verdict({"coverage": 0.9, "auroc": None, "rank_agrees": False}, good_ground, 0.8, 0.6)
    assert pending["verdict"] == "pending"
    per = {d: {"verdict": "supported"} for d in ("squad", "hotpot", "nq")}
    assert overall_verdict(per)["verdict"] == "supported"
    assert overall_verdict({"squad": {"verdict": "supported"}})["verdict"] == "pending"
    mixed = {**per, "nq": {"verdict": "contradicted"}}
    assert overall_verdict(mixed)["verdict"] == "contradicted"


def _capital_entities(text: str) -> list[str]:
    return re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b", text or "")


def test_graph_and_bm25_rank_the_supporting_passage_first():
    passages = [
        {"id": "a", "text": "Ada Lovelace met Charles Babbage in London."},
        {"id": "b", "text": "Charles Babbage built the Analytical Engine."},
        {"id": "c", "text": "Bananas are yellow fruit from tropical farms."},
    ]
    scores = graph_scores("What did Ada Lovelace do?", passages, _capital_entities)
    order = sorted(range(3), key=lambda i: (-scores[i], passages[i]["id"]))
    assert passages[order[0]]["id"] == "a"
    assert passages[order[-1]]["id"] == "c"
    bm = bm25_scores("bananas fruit", passages, 1.5, 0.75)
    assert passages[max(range(3), key=lambda i: bm[i])]["id"] == "c"

    def embed(texts):
        out = []
        for text in texts:
            vec = [0.0, 0.0, 0.0, 0.0]
            for i, ch in enumerate(text.lower()):
                vec[i % 4] += (ord(ch) % 5) / 10.0
            out.append(vec)
        return out

    ranked, meta = rank_pool("bananas", passages, embed, top_k=2, entity_fn=_capital_entities)
    assert meta["score_normalization"] == "per_query_minmax"
    assert set(ranked) == {"semantic_search", "graph_first", "keyword_boosted", "hybrid"}
    assert len(ranked["semantic_search"]) == 2
    assert ranked["semantic_search"][0]["rank"] == 1
    assert "raw_score" in ranked["semantic_search"][0]


def _arm_row(qid, arm, correct, status, confidence, grounding=None, judge=None):
    return {
        "qid": qid, "arm": arm, "correct": correct, "self_status": status,
        "self_confidence": confidence, "lexical_grounding": grounding, "judge": judge,
    }


def test_rewards_treat_unobserved_self_as_missing_and_fallbacks_as_constants():
    cfg = load_config()
    ok = build_rewards(_arm_row("q", "semantic_search", 1, "ok", 0.4, 0.5, None), cfg)
    assert ok["self"] == 0.4 and ok["self_with_fallbacks"] == 0.4
    omitted = build_rewards(_arm_row("q", "semantic_search", 1, "omitted", OMITTED, None, None), cfg)
    assert omitted["self"] is None and omitted["self_with_fallbacks"] == 0.7
    bad = build_rewards(_arm_row("q", "semantic_search", 0, "unparseable", UNPARSEABLE), cfg)
    assert bad["self_with_fallbacks"] == 0.5
    failed = build_rewards(_arm_row("q", "semantic_search", 0, "failed", FAILED), cfg)
    assert failed["self"] is None and failed["self_with_fallbacks"] == 0.0
    # Verified reward is the weighted mean, and a missing judge leaves grounding.
    both = build_rewards(_arm_row("q", "graph_first", 1, "ok", 0.2, 0.5, 1.0), cfg)
    assert both["verified"] == pytest.approx((1.0 * 0.5 + 2.0 * 1.0) / 3.0)
    with pytest.raises(ProtocolError):
        build_rewards(_arm_row("q", "semantic_search", 1, "omitted", 0.2), cfg)


def test_replay_is_deterministic_and_skips_missing_rewards():
    cfg = load_config()
    priors = general_priors()
    strength = float(R._PRIOR_STRENGTH)
    arms = ["semantic_search", "graph_first", "keyword_boosted", "hybrid"]
    questions = [{"qid": f"q{i}", "question_type": "answerable"} for i in range(12)]
    by_qid = {}
    for i, q in enumerate(questions):
        by_qid[q["qid"]] = {}
        for arm in arms:
            correct = 1 if arm == "graph_first" else 0
            status = "omitted" if arm == "semantic_search" else "ok"
            confidence = OMITTED if status == "omitted" else (0.95 if arm == "graph_first" else 0.4)
            row = _arm_row(q["qid"], arm, correct, status, confidence, grounding=float(correct), judge=None)
            by_qid[q["qid"]][arm] = {"correct": correct, "rewards": build_rewards(row, cfg)}
    mu = empirical_mu([
        {"question_type": "answerable", "arm": arm, "correct": by_qid["q0"][arm]["correct"]}
        for arm in arms
    ])
    # empirical_mu needs one row per question; the constant pattern makes any question enough
    # only if every question agrees, which it does. Rebuild from all rows.
    mu = empirical_mu([
        {"question_type": "answerable", "arm": arm, "correct": by_qid[q["qid"]][arm]["correct"]}
        for q in questions for arm in arms
    ])
    a = run_one(questions, by_qid, "self", "fractional", 7, mu, priors, strength)
    b = run_one(questions, by_qid, "self", "fractional", 7, mu, priors, strength)
    assert a["curve"] == b["curve"]
    c = run_one(questions, by_qid, "self_with_fallbacks", "fractional", 7, mu, priors, strength)
    assert [p["applied_reward"] for p in a["curve"]] != [p["applied_reward"] for p in c["curve"]]
    d = run_one(questions, by_qid, "self", "bernoulli", 7, mu, priors, strength)
    e = run_one(questions, by_qid, "self", "bernoulli", 7, mu, priors, strength)
    assert d["curve"] == e["curve"]
    router = _Router(1, ["answerable"], priors, strength)
    try:
        before = list(router.store["answerable"]["semantic_search"])
        assert router.update("answerable", "semantic_search", None, "fractional") is None
        assert router.store["answerable"]["semantic_search"] == before
        router.update("answerable", "semantic_search", 0.0, "fractional")
        assert router.store["answerable"]["semantic_search"] != before
    finally:
        router.close()


def test_normalized_self_is_causal():
    assert normalized_self_reward([], 0.9) is None
    assert normalized_self_reward([0.2, 0.4], 0.9) == 1.0
    assert normalized_self_reward([0.2, 0.9], 0.4) == pytest.approx(0.5)
    cfg = load_config()
    priors = general_priors()
    strength = float(R._PRIOR_STRENGTH)
    arms = ["semantic_search", "graph_first", "keyword_boosted", "hybrid"]

    def build(last_self: float):
        questions = [{"qid": f"q{i}", "question_type": "nq"} for i in range(4)]
        by_qid = {}
        for i, q in enumerate(questions):
            by_qid[q["qid"]] = {}
            for arm in arms:
                conf = last_self if i == 3 else 0.3 + 0.1 * i
                row = _arm_row(q["qid"], arm, 1, "ok", conf, 0.5, None)
                by_qid[q["qid"]][arm] = {"correct": 1, "rewards": build_rewards(row, cfg)}
        mu = empirical_mu([
            {"question_type": "nq", "arm": arm, "correct": 1} for arm in arms for _ in questions
        ])
        return questions, by_qid, mu

    q1, b1, mu = build(0.2)
    q2, b2, _ = build(0.99)
    r1 = run_one(q1, b1, "normalized_self", "fractional", 4, mu, priors, strength)
    r2 = run_one(q2, b2, "normalized_self", "fractional", 4, mu, priors, strength)
    order = [{"qid": f"q{i}", "question_type": "nq"} for i in range(4)]
    random.Random(4).shuffle(order)
    idx = next(i for i, q in enumerate(order) if q["qid"] == "q3")
    assert idx > 0
    assert r1["curve"][:idx] == r2["curve"][:idx]


def test_missingness_and_slopes_do_not_fill_none_with_zero():
    rows = []
    for i in range(6):
        for arm, status, conf, correct in (
            ("semantic_search", "omitted", OMITTED, 0),
            ("graph_first", "ok", 0.8, 1),
        ):
            rows.append({
                "qid": f"q{i}", "dataset": "squad", "arm": arm, "correct": correct,
                "self_status": status, "self_confidence": conf,
                "lexical_grounding": None if arm == "semantic_search" else 1.0,
                "lexical_grounding_reason": "no_claims" if arm == "semantic_search" else None,
                "nli_grounding": None, "nli_grounding_reason": "signal_file_missing",
                "judge": None, "judge_reason": "not_sampled",
            })
    out = analyse(rows, seed=1, n_boot=20, f1_low=0.2, f1_high=0.8)
    self_m = out["datasets"]["squad"]["self"]["missingness"]
    assert self_m["semantic_search"]["m"] == 1.0
    assert self_m["graph_first"]["m"] == 0.0
    assert self_m["semantic_search"]["accuracy_when_observed"] is None
    slope = out["datasets"]["squad"]["self"]["slope"]
    # Only the observed graph_first rows (Y=1) contribute; Y=0 is entirely missing.
    assert slope["c0"]["estimate"] is None
    assert slope["c1"]["estimate"] == pytest.approx(0.8)
    assert slope["s"]["estimate"] is None


def test_results_writer_leaves_missing_artifacts_pending(tmp_path: Path):
    findings, pending = render(tmp_path)
    assert "pending" in findings
    assert "results/calibration/calibration.json is missing" in pending
    assert "usd_upper_bound" in pending
    # A present number is copied; a sibling dataset stays pending.
    cal = {
        "datasets": {
            "squad": {
                "signals": {
                    "self": {
                        "coverage": 0.91, "brier": 0.2, "ece": 0.1, "auroc": 0.55,
                        "spearman": 0.1, "kendall_tau": 0.0, "rank_agrees": False,
                        "correctness_ranking": ["graph_first", "semantic_search"],
                        "signal_ranking": ["semantic_search", "graph_first"],
                        "ci": {
                            "coverage": {"estimate": 0.91, "lo": 0.8, "hi": 0.95},
                            "brier": {"estimate": 0.2, "lo": 0.1, "hi": 0.3},
                            "ece": {"estimate": 0.1, "lo": 0.0, "hi": 0.2},
                            "auroc": {"estimate": 0.55, "lo": 0.4, "hi": 0.6},
                            "spearman": {"estimate": 0.1, "lo": -0.1, "hi": 0.2},
                            "kendall_tau": {"estimate": 0.0, "lo": -1.0, "hi": 1.0},
                        },
                    }
                },
                "prediction": {"verdict": "pending"},
            }
        },
        "prediction": {"verdict": "pending"},
    }
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "calibration.json").write_text(json.dumps(cal))
    findings, pending = render(tmp_path)
    assert "0.9100" in findings
    assert "hotpot self coverage" in pending


def test_dry_run_does_not_call_a_model_and_price_table_is_closed():
    cfg = load_config()
    rows = [{
        "qid": "q", "dataset": "squad", "arm": "semantic_search", "question": "Where?",
        "passages": [{"id": "p", "title": "T", "text": "A short passage."}],
    }]
    estimate = dry_run(cfg, rows, cfg["generator_model_id"])
    assert estimate["called_model"] is False
    assert estimate["n_calls"] == 1
    assert estimate["input_tokens_estimate"] > 0
    assert estimate["usd_upper_bound"] > 0
    haiku = price_for(cfg, cfg["generator_model_id"])
    assert haiku["input"] == pytest.approx(1.10)
    assert haiku["output"] == pytest.approx(5.50)
    assert "input 1.00 / output 5.00" in haiku["source"]
    llama = price_for(cfg, cfg["judge_model_id"])
    assert llama["input"] == pytest.approx(0.72)
    assert llama["output"] == pytest.approx(0.72)
    comment = (ROOT / "experiments" / "config.yaml").read_text()
    assert "input 1.00 / output 5.00" in comment
    assert "input 1.10 / output 5.50" in comment
    with pytest.raises(ProtocolError):
        price_for(cfg, "unknown-model")
    user = build_user_message("Where?", rows[0]["passages"])
    assert "A short passage." in user and "Where?" in user
    assert "confidence" in system_prompt()


def test_generate_records_validation_and_stops_on_access_denied(tmp_path: Path):
    from experiments.ledger import Budget
    cfg = load_config()
    rows = [
        {
            "qid": "q1", "dataset": "squad", "arm": arm, "question_type": "answerable",
            "question": "Where?", "passages": [{"title": "", "text": "Paris is the capital."}],
            "unanswerable": False, "source_retrieved": True,
        }
        for arm in ("semantic_search", "graph_first")
    ]

    class Client:
        def __init__(self):
            self.calls = 0

        def converse(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _client_error("ValidationException", "bad prompt")
            raise _client_error("AccessDeniedException", "no")

    client = Client()
    budget = Budget(tmp_path, "generate", max_usd=10, total_usd_cap=30)
    with pytest.raises(ProtocolError, match="AccessDenied"):
        generate_rows(cfg, rows, tmp_path / "g.jsonl", budget,
                      model_id=cfg["generator_model_id"], client=client, sleep=lambda s: None)
    written = [json.loads(line) for line in (tmp_path / "g.jsonl").read_text().splitlines()]
    assert written[0]["self_status"] == "failed"
    assert written[0]["error"]["type"] == "ValidationException"


def test_same_judge_is_refused_and_schema_and_sample_are_strict():
    with pytest.raises(ProtocolError):
        check_judge_model("same", "same", False)
    check_judge_model("same", "same", True)
    check_judge_model("judge", "generator", False)
    with pytest.raises(ProtocolError):
        validate_rows([{"qid": "q"}], ("qid", "arm"), Path("x.jsonl"))
    rows = [{"qid": f"q{i:02d}", "dataset": "squad"} for i in range(5)]
    a = sample_rows(rows, 3, 11, "squad")
    b = sample_rows(rows, 3, 11, "squad")
    assert [r["qid"] for r in a] == [r["qid"] for r in b]
    assert len(a) == 3
    order = seeded_order(rows, 11, "squad")
    assert {r["qid"] for r in a} == {r["qid"] for r in order[:3]}
    wider = sample_rows(rows, 4, 11, "squad")
    assert {r["qid"] for r in a} < {r["qid"] for r in wider}
    with pytest.raises(ProtocolError):
        sample_rows(rows, 9, 1, "squad")
    assert chunk_windows("abcdefghij", 4, 1)[0] == "abcd"
    assert len(chunk_windows("short", 800, 200)) == 1


def test_assumed_slopes_match_the_simulation_source():
    text = (ROOT / "scripts" / "verified_reward_bench.py").read_text()
    assert "rng.gauss(0.88, 0.06)" in text
    assert "0.55 + self_bias" in text
    assert "gauss(0.8 if correct else 0.3" in text
    regret = (ROOT / "scripts" / "routing_regret_sim.py").read_text()
    assert "NOISE_K = 20.0" in regret
    slopes = assumed_slopes()
    assert slopes["self"]["c1"] == SELF_C1
    assert slopes["self"]["c0_prime"] == SELF_LOW_MEAN
    assert slopes["self"]["s"] == pytest.approx((1 - 0.7) * (0.88 - 0.80))
    assert slopes["grounding"]["s"] == pytest.approx(0.5)
    assert slopes["routing_regret_sim"]["s_self"] is None


def test_confidence_statuses_match_synthesizer_constants():
    ok = parse_self_confidence('{"answer": "Paris", "confidence": 0.25}')
    assert ok["status"] == "ok" and ok["value"] == 0.25 and ok["deployed_status"] == "ok"
    omitted = parse_self_confidence('{"answer": "Paris"}')
    assert omitted["status"] == "omitted" and omitted["value"] is None and omitted["reported"] is False
    assert omitted["deployed_status"] == "omitted" and omitted["deployed_value"] == OMITTED
    assert omitted["answer"] == "Paris"
    bad = parse_self_confidence("not json")
    assert bad["status"] == "unparseable" and bad["value"] is None and bad["answer"] == ""
    assert bad["deployed_status"] == "unparseable" and bad["deployed_value"] == UNPARSEABLE
    failed = parse_self_confidence(None, call_failed=True)
    assert failed["status"] == "failed" and failed["value"] is None and failed["answer"] == ""
    assert failed["deployed_status"] == "failed" and failed["deployed_value"] == FAILED
    assert ok["trailing_truncated"] is False
    assert failed["trailing_truncated"] is False
    src = (ROOT / "src" / "query_api" / "synthesizer.py").read_text()
    assert "_OMITTED_CONFIDENCE = 0.7" in src
    assert "_UNPARSED_CONFIDENCE = 0.5" in src


def test_prose_around_json_keeps_the_answer_field_and_the_deployed_fallback():
    raw = (
        '{"answer": "November 11, 1901", "confidence": 0.95} '
        "The date is given in the passage."
    )
    parsed = parse_self_confidence(raw)
    assert parsed["answer"] == "November 11, 1901"
    assert parsed["answer"] != raw
    assert parsed["status"] == "ok" and parsed["value"] == 0.95 and parsed["reported"] is True
    assert parsed["deployed_status"] == "unparseable" and parsed["deployed_value"] == UNPARSEABLE
    leading = parse_self_confidence(
        'The answer is {"answer": "November 11, 1901", "confidence": 0.95}.'
    )
    assert leading["answer"] == "November 11, 1901" and leading["status"] == "ok"
    assert leading["deployed_status"] == "unparseable"
    row = {
        "qid": "q", "arm": "semantic_search", "correct": 1,
        "self_status": parsed["status"], "self_confidence": parsed["value"],
        "deployed_self_status": parsed["deployed_status"],
        "deployed_self_confidence": parsed["deployed_value"],
        "lexical_grounding": 1.0, "judge": None, "judge_reason": "not_sampled",
    }
    rewards = build_rewards(row, load_config())
    assert rewards["self"] == 0.95
    assert rewards["self_with_fallbacks"] == 0.5
    # The smoke failure mode, repeated: prose around an object must not become the answer.
    wrapped = [
        f'Note. {{"answer": "span {i}", "confidence": 0.8}} Done.'
        for i in range(19)
    ] + ["no json object here at all"]
    robust_ok = sum(parse_self_confidence(text)["status"] == "ok" for text in wrapped)
    assert robust_ok / len(wrapped) >= 0.95
    assert all(parse_self_confidence(text)["answer"] != text for text in wrapped)


def test_complete_json_cut_off_after_the_object_keeps_the_confidence():
    """Smoke shape: a finished object, then prose that hits max_tokens."""
    raw = (
        '{"answer": "insufficient evidence", "confidence": 0.0}\n\n'
        "The passages do not contain"
    )
    parsed = parse_self_confidence(raw, truncated=True)
    assert parsed["status"] == "ok" and parsed["value"] == 0.0 and parsed["reported"] is True
    assert parsed["answer"] == "insufficient evidence"
    assert parsed["trailing_truncated"] is True
    assert parsed["deployed_status"] == "unparseable" and parsed["deployed_value"] == UNPARSEABLE
    assert is_abstention(parsed["answer"])
    # The object filled the cap exactly, with no trailing prose.
    exact = parse_self_confidence('{"answer": "Paris", "confidence": 0.9}', truncated=True)
    assert exact["status"] == "ok" and exact["value"] == 0.9 and exact["trailing_truncated"] is True
    assert exact["deployed_status"] == "ok" and exact["deployed_value"] == 0.9
    row = {
        "qid": "q", "arm": "semantic_search", "correct": 0,
        "self_status": parsed["status"], "self_confidence": parsed["value"],
        "deployed_self_status": parsed["deployed_status"],
        "deployed_self_confidence": parsed["deployed_value"],
        "trailing_truncated": parsed["trailing_truncated"],
        "lexical_grounding": None, "judge": None, "judge_reason": "not_sampled",
    }
    rewards = build_rewards(row, load_config())
    assert rewards["self"] == 0.0
    assert rewards["self_with_fallbacks"] == UNPARSEABLE


def test_bare_abstention_without_json_is_omitted_and_scored_as_abstention():
    raw = "insufficient evidence\n\nThe passages do not contain information about the date."
    parsed = parse_self_confidence(raw)
    assert parsed["answer"] == "insufficient evidence"
    assert parsed["answer"] != raw
    assert parsed["status"] == "omitted" and parsed["value"] is None and parsed["reported"] is False
    assert parsed["trailing_truncated"] is False
    assert parsed["deployed_status"] == "unparseable" and parsed["deployed_value"] == UNPARSEABLE
    assert is_abstention(parsed["answer"])
    scored = score_answer(parsed["answer"], [], True)
    assert scored["correct"] == 1 and scored["abstained"] is True
    titled = parse_self_confidence("Insufficient evidence.\n\nThe passages do not contain the span.")
    assert titled["answer"] == "insufficient evidence" and titled["status"] == "omitted"
    row = {
        "qid": "q", "arm": "semantic_search", "correct": 1,
        "self_status": parsed["status"], "self_confidence": parsed["value"],
        "deployed_self_status": parsed["deployed_status"],
        "deployed_self_confidence": parsed["deployed_value"],
        "lexical_grounding": None, "judge": None, "judge_reason": "not_sampled",
    }
    rewards = build_rewards(row, load_config())
    assert rewards["self"] is None
    assert rewards["self_with_fallbacks"] == UNPARSEABLE


def test_judge_score_is_compared_to_correctness_as_stored():
    """A grounding score of 1.0 on an abstention is not rewritten to match F1."""
    rows = [
        {"qid": "q1", "arm": "semantic_search", "f1": 0.0, "correct": 0, "value": 1.0},
        {"qid": "q2", "arm": "semantic_search", "f1": 1.0, "correct": 1, "value": 1.0},
    ]
    metrics = signal_metrics(rows)
    assert metrics["n_observed"] == 2
    assert metrics["brier"] == 0.5
    prompt = (ROOT / "experiments" / "prompts" / "generator_system.txt").read_text()
    assert '"claim": "<one sentence>"' in prompt
    assert 'When you abstain, the claim is "The passages do not contain the answer."' in prompt
    assert "nothing before or after it" in prompt
    assert "answer yes or no" in prompt
    assert (
        "The confidence field is your own verbalized estimate of the probability "
        "that the answer is correct."
    ) in prompt
    assert "Use 0 when" not in prompt
    assert "taken directly" not in prompt
    assert load_config()["generator_max_output_tokens"] == 256
    assert load_config()["n_per_dataset"] == 1100


def _client_error(code: str, message: str):
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        class ClientError(Exception):
            def __init__(self, response, operation_name):
                super().__init__(message)
                self.response = response
        return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")
    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _script(name: str):
    import importlib.util
    path = ROOT / "scripts" / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exp_script_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gold_window(pid: str, source: str, text: str) -> dict:
    return {"id": pid, "title": "", "text": text, "source": source, "role": "gold"}


def _nq_row(qid: str, question: str, pool: list[dict]) -> dict:
    return {
        "qid": qid, "dataset": "nq", "question": question, "question_type": "nq",
        "gold_answers": ["answer"], "unanswerable": False, "pool": pool,
    }


def test_nq_pools_keep_gold_and_fill_with_deterministic_hard_negatives():
    from experiments.data import _record, expand_nq_pools
    rows = [
        _nq_row("q1", "zzzz", [_gold_window("p1", "s1", "alpha beta")]),
        _nq_row("q2", "zzzz", [_gold_window("p2", "s2", "gamma delta")]),
        _nq_row("q3", "zzzz", [_gold_window("p3", "s3", "epsilon zeta")]),
    ]
    again = json.loads(json.dumps(rows))
    info = expand_nq_pools(rows, 3, 1.5, 0.75)
    expand_nq_pools(again, 3, 1.5, 0.75)
    assert info["pool_size_min"] == 3
    assert info["hard_negatives_added"] == 6
    for left, right in zip(rows, again):
        assert [p["id"] for p in left["pool"]] == [p["id"] for p in right["pool"]]
    gold, *negs = rows[0]["pool"]
    assert gold["role"] == "gold" and gold["text"] == "alpha beta" and gold["source"] == "s1"
    assert [p["role"] for p in negs] == ["hard_negative", "hard_negative"]
    assert {p["source"] for p in negs} == {"s2", "s3"}
    # No query term is in any window, so every score is 0 and the id breaks the tie.
    assert [p["id"] for p in negs] == sorted(p["id"] for p in negs)
    wide = [
        _nq_row("q9", "zzzz", [_gold_window(f"g{i}", "s9", f"gold window {i}") for i in range(4)]),
        _nq_row("q2", "zzzz", [_gold_window("p2", "s2", "gamma delta")]),
        _nq_row("q3", "zzzz", [_gold_window("p3", "s3", "epsilon zeta")]),
    ]
    kept = expand_nq_pools(wide, 3, 1.5, 0.75)
    assert len(wide[0]["pool"]) == 4
    assert all(p["role"] == "gold" for p in wide[0]["pool"])
    assert kept["pool_size_max"] == 4
    with pytest.raises(ProtocolError, match="hard-negative"):
        expand_nq_pools([_nq_row("only", "zzzz", [_gold_window("p1", "s1", "alpha")])], 3, 1.5, 0.75)
    shared = [_gold_window("p1", "s1", "alpha beta")]
    copied = _record("nq", "q1", "zzzz", "nq", ["x"], False, shared)
    copied["pool"].append({"id": "extra"})
    assert len(shared) == 1


def test_new_stage_files_are_gzipped_and_both_spellings_are_refused(tmp_path: Path):
    from experiments.common import append_jsonl, read_jsonl, stage_jsonl
    fresh = stage_jsonl(tmp_path, "rows")
    assert fresh.name == "rows.jsonl.gz"
    append_jsonl(fresh, {"qid": "a", "arm": "semantic_search"})
    append_jsonl(fresh, {"qid": "b", "arm": "hybrid"})
    assert read_jsonl(tmp_path / "rows.jsonl") == [
        {"qid": "a", "arm": "semantic_search"},
        {"qid": "b", "arm": "hybrid"},
    ]
    kept = tmp_path / "squad.jsonl"
    kept.write_text('{"qid": "s"}\n')
    assert stage_jsonl(tmp_path, "squad") == kept
    (tmp_path / "squad.jsonl.gz").write_bytes(b"")
    with pytest.raises(ProtocolError, match="both"):
        stage_jsonl(tmp_path, "squad")


def test_shared_ledger_blocks_the_judge_on_the_global_cap(tmp_path: Path):
    from experiments.ledger import Budget
    generation = Budget(tmp_path, "generate", max_usd=27, total_usd_cap=30)
    generation.record({
        "qid": "q", "arm": "semantic_search", "dataset": "nq",
        "usd": 28.0, "input_tokens": 1, "output_tokens": 1, "model_id": "m",
    })
    judge = Budget(tmp_path, "judge", max_usd=3, total_usd_cap=30)
    reason = judge.blocking_reason(2.5)
    assert reason is not None and "total cap" in reason
    assert judge.blocking_reason(1.0) is None


def test_truncation_is_recorded_and_is_not_the_unparseable_fallback(tmp_path: Path):
    broken = '{"answer": "Paris", "confidence": 0.9'
    parsed = parse_self_confidence(broken, truncated=True)
    assert parsed["status"] == "truncated" and parsed["value"] is None and parsed["reported"] is False
    assert parsed["trailing_truncated"] is False
    assert parsed["deployed_status"] == "truncated" and parsed["deployed_value"] is None
    row = {
        "qid": "q", "arm": "semantic_search", "self_status": "truncated",
        "self_confidence": None, "lexical_grounding": 1.0, "judge": None,
        "judge_reason": "not_sampled", "correct": 1,
    }
    rewards = build_rewards(row, load_config())
    assert rewards["self"] is None and rewards["self_with_fallbacks"] is None
    cfg = load_config()

    class Client:
        def converse(self, **kwargs):
            assert kwargs["inferenceConfig"]["maxTokens"] == 256
            return {
                "stopReason": "max_tokens",
                "output": {"message": {"content": [{"text": broken}]}},
                "usage": {"inputTokens": 12, "outputTokens": 256},
            }

    from experiments.ledger import Budget
    retrieval = [{
        "qid": "q1", "dataset": "squad", "arm": "semantic_search", "question_type": "answerable",
        "question": "Where?", "passages": [{"title": "", "text": "Paris is the capital."}],
        "unanswerable": False, "source_retrieved": True,
    }]
    budget = Budget(tmp_path, "generate", max_usd=10, total_usd_cap=30)
    summary = generate_rows(
        cfg, retrieval, tmp_path / "g.jsonl", budget,
        model_id=cfg["generator_model_id"], client=Client(), sleep=lambda _s: None,
    )
    assert summary["stopped"] is False and summary["written"] == 1
    written = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])
    assert written["self_status"] == "truncated"
    assert written["self_confidence"] is None
    assert written["trailing_truncated"] is False
    assert written["stop_reason"] == "max_tokens"
    assert written["self_confidence"] != UNPARSEABLE
    assert written["deployed_self_status"] == "truncated"
    assert written["deployed_self_confidence"] is None


def test_spend_cap_stops_cleanly_and_lists_the_rest_pending(tmp_path: Path):
    from experiments.common import cost_usd, read_jsonl
    from experiments.generate_stage import build_user_message, prompt_token_estimate
    from experiments.ledger import Budget, SPEND_CAP_REASON
    cfg = load_config()
    passages = [{"title": "", "text": "Paris is the capital."}]
    rows = [
        {
            "qid": "q1", "dataset": "squad", "arm": arm, "question_type": "answerable",
            "question": "Where?", "passages": passages,
            "unanswerable": False, "source_retrieved": True,
        }
        for arm in ("semantic_search", "graph_first")
    ]
    user = build_user_message(rows[0]["question"], passages)
    in_tok = prompt_token_estimate(system_prompt(), user)
    out_tok = int(cfg["generator_max_output_tokens"])
    one_call = cost_usd(cfg, cfg["generator_model_id"], in_tok, out_tok)

    class Client:
        def converse(self, **kwargs):
            return {
                "stopReason": "end_turn",
                "output": {"message": {"content": [{"text": '{"answer": "Paris", "confidence": 0.4}'}]}},
                "usage": {"inputTokens": in_tok, "outputTokens": out_tok},
            }

    _write_jsonl(tmp_path / "samples" / "squad.jsonl", [{
        "qid": "q1", "dataset": "squad", "question": "Where?",
        "question_type": "answerable", "gold_answers": ["Paris"], "unanswerable": False,
        "pool": [],
    }])
    _write_jsonl(tmp_path / "retrieval" / "squad.jsonl", rows)
    gen = _script("generate")
    gen.make_client = lambda region: Client()
    code = gen.main([
        "--results", str(tmp_path), "--datasets", "squad", "--seed", "0",
        "--max-usd", f"{one_call * 1.5:.10f}", "--total-usd-cap", "30",
        "--inference-mode", "on_demand",
    ])
    assert code == 0
    written = read_jsonl(tmp_path / "generation" / "squad.jsonl")
    assert len(written) == 1 and written[0]["arm"] == "semantic_search"
    pending = json.loads((tmp_path / "generation" / "pending.json").read_text())
    assert pending["reason"] == SPEND_CAP_REASON
    assert pending["rows"] == [{"qid": "q1", "arm": "graph_first", "dataset": "squad"}]
    direct = Budget(tmp_path / "other", "generate", max_usd=one_call * 1.5, total_usd_cap=30)
    summary = generate_rows(
        cfg, rows, tmp_path / "other" / "g.jsonl", direct,
        model_id=cfg["generator_model_id"], client=Client(), sleep=lambda _s: None,
    )
    assert summary["stopped"] is True
    assert [item["arm"] for item in summary["pending"]] == ["graph_first"]


def _mini_squad(tmp_path: Path) -> None:
    cfg = load_config()
    question = {
        "qid": "q1", "dataset": "squad", "question": "Where is the capital?",
        "question_type": "squad", "gold_answers": ["Paris"], "unanswerable": False,
        "source_passage_ids": ["p"], "yes_no": False, "schema_version": 2,
    }
    _write_jsonl(tmp_path / "samples" / "squad.jsonl", [question])
    passages = [{"id": "p", "title": "France", "text": "Paris is the capital of France."}]
    _write_jsonl(tmp_path / "samples" / "squad.passages.jsonl", passages)
    retrieval = []
    generation = []
    for arm in ("semantic_search", "graph_first", "keyword_boosted", "hybrid"):
        retrieval.append({
            "qid": "q1", "dataset": "squad", "arm": arm, "question_type": "squad",
            "question": question["question"], "passages": passages, "pool_size": 1,
            "gold_in_top_k": True, "source_retrieved": True, "unanswerable": False,
            "source_passage_ids": ["p"],
        })
        generation.append({
            "qid": "q1", "dataset": "squad", "arm": arm, "question_type": "answerable",
            "answer": "Paris",
            "raw_response": '{"answer": "Paris", "confidence": 0.8}',
            "self_confidence": 0.8, "self_reported": True, "self_status": "ok",
            "input_tokens": 10, "output_tokens": 20,
            "model_id": cfg["generator_model_id"], "error": None,
        })
    _write_jsonl(tmp_path / "retrieval" / "squad.jsonl", retrieval)
    _write_jsonl(tmp_path / "generation" / "squad.jsonl", generation)


def test_judge_access_denied_stays_pending_downstream(tmp_path: Path, capsys):
    from experiments.judge_access import JUDGE_ACCESS_REASON
    _mini_squad(tmp_path)
    signals = _script("signals")
    assert signals.main(["lexical", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    from experiments.common import read_jsonl
    lexical = read_jsonl(tmp_path / "signals" / "lexical.jsonl")
    _write_jsonl(tmp_path / "signals" / "nli.jsonl", [
        {"qid": row["qid"], "dataset": row["dataset"], "arm": row["arm"],
         "question_type": row["question_type"], "value": 0.4, "reason": None}
        for row in lexical
    ])

    class Client:
        def converse(self, **kwargs):
            raise _client_error("AccessDeniedException", "no grant")

    signals.make_client = lambda region: Client()
    import yaml
    cfg = yaml.safe_load((ROOT / "experiments" / "config.yaml").read_text())
    cfg["judge_sample_rate"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    code = signals.main([
        "judge", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0",
        "--config", str(config_path), "--max-usd", "3", "--total-usd-cap", "30",
        "--inference-mode", "on_demand",
    ])
    assert code == 2
    assert JUDGE_ACCESS_REASON in capsys.readouterr().err
    marker = json.loads((tmp_path / "signals" / "judge_unavailable.json").read_text())
    assert marker["reason"] == JUDGE_ACCESS_REASON
    for name, argv in (
        ("calibrate", ["--datasets", "squad", "--bootstrap", "20"]),
        ("replay", ["--datasets", "squad", "--seeds", "0"]),
        ("analyses", ["--datasets", "squad", "--bootstrap", "20"]),
        ("write_results", []),
    ):
        assert _script(name).main(
            ["--results", str(tmp_path), "--seed", "0", *argv]
        ) == 0
    findings = (tmp_path / "FINDINGS.md").read_text()
    pending = (tmp_path / "PENDING.md").read_text()
    assert JUDGE_ACCESS_REASON in findings
    assert JUDGE_ACCESS_REASON in pending
    assert "grounding judgement" in findings
    assert "stays 1.0 when correctness is 0" in findings
    assert "bare abstention" in findings
    calibration = json.loads((tmp_path / "calibration" / "calibration.json").read_text())
    assert calibration["unavailable_signals"]["judge"] == JUDGE_ACCESS_REASON
    replay = json.loads((tmp_path / "replay" / "replay_summary.json").read_text())
    assert "verified" not in replay["datasets"]["squad"]
    assert replay["unavailable_rewards"]["verified"] == JUDGE_ACCESS_REASON


def test_claim_verdicts_follow_the_artifact_rules(tmp_path: Path):
    from experiments.claims import claim_verdicts
    from experiments.common import write_json
    empty = {item["id"]: item["verdict"] for item in claim_verdicts(tmp_path, 0.5)}
    assert set(empty.values()) == {"open"}

    def interval(estimate: float) -> dict:
        return {"estimate": estimate, "lo": estimate - 0.01, "hi": estimate + 0.01}

    def analyses(pairs: list, reversed_order: bool, slopes: tuple[float, float], combo: tuple[float, float, float]) -> dict:
        datasets = {}
        for name in ("squad", "hotpot", "nq"):
            datasets[name] = {
                "self": {
                    "assumption": {
                        "nonoverlapping_pairs": pairs,
                        "by_arm": {
                            "semantic_search": {"y0": interval(0.4), "y1": interval(0.8)},
                            "graph_first": {"y0": interval(0.4), "y1": interval(0.8)},
                        },
                    },
                    "slope": {"s": interval(slopes[0])},
                },
                "lexical_grounding": {"slope": {"s": interval(slopes[1])}},
                "fallback": {
                    "order_reversed": reversed_order,
                    "order_skip": ["graph_first", "semantic_search"],
                    "order_fallback": (
                        ["semantic_search", "graph_first"] if reversed_order
                        else ["graph_first", "semantic_search"]
                    ),
                },
                "combination": {
                    "grounding": {"s": interval(combo[0])},
                    "judge": {"s": interval(combo[1])},
                    "verified": {"s": interval(combo[2])},
                },
            }
        return {"datasets": datasets}

    def calibration(agree: bool, verdict: str) -> dict:
        return {
            "prediction": {"verdict": verdict},
            "datasets": {
                name: {"signals": {"self": {"rank_agrees": agree}}}
                for name in ("squad", "hotpot", "nq")
            },
        }

    def replay(self_r: float, norm_r: float, ground_r: float) -> dict:
        def block(value: float) -> dict:
            return {"fractional": {"pseudo_regret_mean": value}}
        return {
            "datasets": {
                name: {
                    "self": block(self_r),
                    "normalized_self": block(norm_r),
                    "lexical_grounding": block(ground_r),
                }
                for name in ("squad", "hotpot", "nq")
            }
        }

    root = tmp_path / "supported"
    write_json(root / "analyses" / "analyses.json", analyses([], True, (0.1, 0.5), (0.4, 0.2, 0.3)))
    write_json(root / "calibration" / "calibration.json", calibration(False, "supported"))
    write_json(root / "replay" / "replay_summary.json", replay(10.0, 6.0, 2.0))
    by_id = {item["id"]: item["verdict"] for item in claim_verdicts(root, 0.5)}
    assert by_id == {
        "Assumption 1": "supported",
        "Proposition 1": "supported",
        "Proposition 2": "supported",
        "Theorem 1 scale objection": "supported",
        "Proposition 3": "supported",
        "Section 6 prediction": "supported",
    }
    denied = tmp_path / "denied"
    write_json(denied / "analyses" / "analyses.json", analyses(
        [{"y": 1, "arms": ["a", "b"]}], False, (0.1, 0.5), (0.4, 0.2, -0.1),
    ))
    write_json(denied / "calibration" / "calibration.json", calibration(True, "contradicted"))
    write_json(denied / "replay" / "replay_summary.json", replay(10.0, 9.5, 2.0))
    write_json(denied / "signals" / "judge_unavailable.json", {
        "reason": "judge model access not yet granted", "detail": "AccessDenied",
    })
    by_id = {item["id"]: item for item in claim_verdicts(denied, 0.5)}
    assert by_id["Assumption 1"]["verdict"] == "contradicted"
    assert by_id["Proposition 1"]["verdict"] == "contradicted"
    assert by_id["Proposition 2"]["verdict"] == "contradicted"
    assert by_id["Theorem 1 scale objection"]["verdict"] == "contradicted"
    assert by_id["Proposition 3"]["verdict"] == "open"
    assert by_id["Proposition 3"]["because"] == "judge model access not yet granted"
    assert by_id["Section 6 prediction"]["verdict"] == "contradicted"
    combo_bad = tmp_path / "combo"
    write_json(combo_bad / "analyses" / "analyses.json", analyses([], False, (0.1, 0.5), (0.4, 0.2, -0.1)))
    write_json(combo_bad / "calibration" / "calibration.json", calibration(True, "pending"))
    by_id = {item["id"]: item["verdict"] for item in claim_verdicts(combo_bad, 0.5)}
    assert by_id["Proposition 3"] == "contradicted"
    assert by_id["Section 6 prediction"] == "open"


def test_post_generation_stages_do_not_fetch_datasets():
    files = [
        "experiments/calibrate_stage.py",
        "experiments/replay_stage.py",
        "experiments/analyses_stage.py",
        "experiments/reporting.py",
        "experiments/records.py",
        "experiments/claims.py",
        "scripts/experiments/calibrate.py",
        "scripts/experiments/replay.py",
        "scripts/experiments/analyses.py",
        "scripts/experiments/write_results.py",
    ]
    for rel in files:
        text = (ROOT / rel).read_text()
        assert "urllib" not in text, rel
        assert "load_nq" not in text, rel
        assert "import datasets" not in text, rel
        assert "from datasets" not in text, rel
    judge = (ROOT / "experiments" / "signals_stage.py").read_text()
    assert "sentence_transformers" in judge
    assert "urllib" not in judge


def test_twelve_hundred_is_the_prefix_of_the_fifteen_hundred_draw():
    """1200 with a seed is the first 1200 of the 1500 draw with that seed."""
    rows = [{"qid": f"q{i:05d}", "dataset": "squad"} for i in range(2000)]
    order = seeded_order(rows, 0, "squad")
    large = sample_rows(rows, 1500, 0, "squad")
    small = sample_rows(rows, 1200, 0, "squad")
    assert {r["qid"] for r in large} == {r["qid"] for r in order[:1500]}
    assert {r["qid"] for r in small} == {r["qid"] for r in order[:1200]}
    assert {r["qid"] for r in small} < {r["qid"] for r in large}
    assert [r["qid"] for r in small] == sorted(r["qid"] for r in order[:1200])


def _shared_bundle(dataset: str, n: int) -> tuple[list[dict], list[dict]]:
    passages = [
        {"id": f"{dataset[0]}p{i:05d}", "title": dataset, "text": f"{dataset} passage {i} alpha"}
        for i in range(n)
    ]
    questions = []
    for i, passage in enumerate(passages):
        qid = f"{dataset[0]}{i:05d}"
        questions.append({
            "qid": qid, "dataset": dataset, "question": f"question {qid}",
            "question_type": dataset, "gold_answers": ["answer"], "unanswerable": False,
            "source_passage_ids": [passage["id"]], "yes_no": False, "hotpot_type": None,
            "schema_version": 2,
        })
    return questions, passages


def _install_shared_loaders(monkeypatch, bundles: dict):
    import json as jsonlib
    from experiments import data as data_mod

    def _copy(name):
        questions, passages = bundles[name]
        return (
            jsonlib.loads(jsonlib.dumps(questions)),
            jsonlib.loads(jsonlib.dumps(passages)),
            {"name": name},
        )

    monkeypatch.setitem(data_mod.LOADERS, "squad", lambda _cfg: _copy("squad"))
    monkeypatch.setitem(data_mod.LOADERS, "hotpot", lambda _cfg: _copy("hotpot"))
    monkeypatch.setattr(data_mod, "load_nq", lambda _cfg, _cache: _copy("nq"))


def test_smaller_n_keeps_the_seeded_prefix_and_the_shared_collection(tmp_path, monkeypatch):
    """A same-seed shrink keeps written question rows and the shared collection."""
    from experiments import data as data_mod
    from experiments.common import read_jsonl

    cfg = load_config()
    bundles = {name: _shared_bundle(name, 6) for name in ("squad", "hotpot", "nq")}
    _install_shared_loaders(monkeypatch, bundles)
    names = ("squad", "hotpot", "nq")
    first = data_mod.build_sample(cfg, tmp_path, 4, 0, names)
    assert first["schema_version"] == 2
    assert first["seed"] == 0
    assert first["sampling"]["seed"] == 0
    assert first["sampling"]["order"] == SAMPLE_ORDER
    assert first["sampling"]["prefix_of_n"] is None
    assert first["collection"] == "shared_dev_split"
    before = {row["qid"]: row for row in read_jsonl(tmp_path / "samples" / "nq.jsonl")}
    assert len(before) == 4
    assert all(row["question_type"] == "nq" and row["schema_version"] == 2 for row in before.values())
    passages_before = read_jsonl(tmp_path / "samples" / "nq.passages.jsonl")
    assert len(passages_before) == 6
    assert {row["source_passage_ids"][0] for row in before.values()} <= {p["id"] for p in passages_before}

    second = data_mod.build_sample(cfg, tmp_path, 2, 0, names)
    assert second["sampling"]["prefix_of_n"] == {"squad": 4, "hotpot": 4, "nq": 4}
    assert second["n_per_dataset"] == 2
    assert second["datasets"]["nq"]["rows_kept_from_previous_sample"] is True
    order = seeded_order(bundles["nq"][0], 0, "nq")
    kept = read_jsonl(tmp_path / "samples" / "nq.jsonl")
    kept_ids = [row["qid"] for row in kept]
    assert kept_ids == sorted(row["qid"] for row in order[:2])
    for row in kept:
        assert row == before[row["qid"]]
    assert read_jsonl(tmp_path / "samples" / "nq.passages.jsonl") == passages_before
    with pytest.raises(ProtocolError, match="question ids changed"):
        data_mod.build_sample(cfg, tmp_path, 4, 0, names)
    still = [row["qid"] for row in read_jsonl(tmp_path / "samples" / "nq.jsonl")]
    assert still == kept_ids
    manifest = json.loads((tmp_path / "samples" / "sample_manifest.json").read_text())
    manifest.pop("schema_version")
    (tmp_path / "samples" / "sample_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ProtocolError, match="schema_version 2"):
        data_mod.build_sample(cfg, tmp_path, 2, 0, names)


def test_eleven_hundred_keeps_the_twelve_hundred_prefix_of_the_fifteen_hundred_draw(tmp_path, monkeypatch):
    """1100 is a same-seed shrink of 1200, which is a same-seed shrink of 1500."""
    from experiments import data as data_mod
    from experiments.common import read_jsonl

    rows = [{"qid": f"q{i:05d}", "dataset": "squad"} for i in range(2000)]
    order = [row["qid"] for row in seeded_order(rows, 0, "squad")]
    assert [row["qid"] for row in sample_rows(rows, 1500, 0, "squad")] == sorted(order[:1500])
    assert [row["qid"] for row in sample_rows(rows, 1200, 0, "squad")] == sorted(order[:1200])
    assert [row["qid"] for row in sample_rows(rows, 1100, 0, "squad")] == sorted(order[:1100])
    assert set(order[:1100]) < set(order[:1200]) < set(order[:1500])

    cfg = load_config()
    bundles = {name: _shared_bundle(name, 1600) for name in ("squad", "hotpot", "nq")}
    _install_shared_loaders(monkeypatch, bundles)
    names = ("squad", "hotpot", "nq")
    data_mod.build_sample(cfg, tmp_path, 1500, 0, names)
    original = {
        name: {row["qid"]: row for row in read_jsonl(tmp_path / "samples" / f"{name}.jsonl")}
        for name in names
    }
    passages = {
        name: read_jsonl(tmp_path / "samples" / f"{name}.passages.jsonl")
        for name in names
    }
    assert {len(table) for table in original.values()} == {1500}
    assert all(len(table) == 1600 for table in passages.values())

    second = data_mod.build_sample(cfg, tmp_path, 1200, 0, names)
    assert second["n_per_dataset"] == 1200
    assert second["sampling"]["prefix_of_n"] == {"squad": 1500, "hotpot": 1500, "nq": 1500}
    assert second["datasets"]["nq"]["rows_kept_from_previous_sample"] is True
    mid = {
        name: {row["qid"]: row for row in read_jsonl(tmp_path / "samples" / f"{name}.jsonl")}
        for name in names
    }
    for name in names:
        assert len(mid[name]) == 1200
        assert set(mid[name]) < set(original[name])
        for qid, row in mid[name].items():
            assert row == original[name][qid]
        assert read_jsonl(tmp_path / "samples" / f"{name}.passages.jsonl") == passages[name]

    third = data_mod.build_sample(cfg, tmp_path, 1100, 0, names)
    assert third["n_per_dataset"] == 1100
    assert third["sampling"]["prefix_of_n"] == {"squad": 1200, "hotpot": 1200, "nq": 1200}
    assert third["datasets"]["nq"]["rows_kept_from_previous_sample"] is True
    final = {
        name: {row["qid"]: row for row in read_jsonl(tmp_path / "samples" / f"{name}.jsonl")}
        for name in names
    }
    for name in names:
        assert len(final[name]) == 1100
        assert set(final[name]) < set(mid[name])
        for qid, row in final[name].items():
            assert row == original[name][qid]
            assert "pool" not in row
        assert read_jsonl(tmp_path / "samples" / f"{name}.passages.jsonl") == passages[name]
    nq_order = [row["qid"] for row in seeded_order(bundles["nq"][0], 0, "nq")]
    assert sorted(final["nq"]) == sorted(nq_order[:1100])
    with pytest.raises(ProtocolError, match="question ids changed"):
        data_mod.build_sample(cfg, tmp_path, 1200, 0, names)
    still = [row["qid"] for row in read_jsonl(tmp_path / "samples" / "nq.jsonl")]
    assert still == sorted(final["nq"])


def test_generation_and_judge_record_usd_at_the_config_price(tmp_path: Path):
    """New paid rows bill the token counts at the price in the config."""
    from experiments.common import cost_usd, read_jsonl
    from experiments.ledger import Budget

    cfg = load_config()
    passages = [{"title": "", "text": "Paris is the capital."}]
    retrieval = [{
        "qid": "q1", "dataset": "squad", "arm": "semantic_search", "question_type": "answerable",
        "question": "Where?", "passages": passages,
        "unanswerable": False, "source_retrieved": True,
    }]

    class GenClient:
        def converse(self, **kwargs):
            return {
                "stopReason": "end_turn",
                "output": {"message": {"content": [{"text": '{"answer": "Paris", "confidence": 0.4}'}]}},
                "usage": {"inputTokens": 1000, "outputTokens": 50},
            }

    gen_budget = Budget(tmp_path / "gen", "generate", max_usd=10, total_usd_cap=30)
    summary = generate_rows(
        cfg, retrieval, tmp_path / "gen" / "g.jsonl", gen_budget,
        model_id=cfg["generator_model_id"], client=GenClient(), sleep=lambda _s: None,
    )
    assert summary["written"] == 1
    expected_gen = cost_usd(cfg, cfg["generator_model_id"], 1000, 50)
    written = json.loads((tmp_path / "gen" / "g.jsonl").read_text().splitlines()[0])
    assert written["usd"] == pytest.approx(expected_gen)
    assert written["input_tokens"] == 1000 and written["output_tokens"] == 50
    gen_ledger = read_jsonl(tmp_path / "gen" / "cost_ledger.jsonl")
    assert gen_ledger[0]["usd"] == pytest.approx(expected_gen)
    assert gen_budget.global_spent == pytest.approx(expected_gen)

    _mini_squad(tmp_path / "judge")
    signals = _script("signals")

    class JudgeClient:
        def converse(self, **kwargs):
            return {
                "stopReason": "end_turn",
                "output": {"message": {"content": [{"text": '{"score": 0.5}'}]}},
                "usage": {"inputTokens": 80, "outputTokens": 12},
            }

    signals.make_client = lambda region: JudgeClient()
    import yaml
    priced = load_config()
    priced["judge_sample_rate"] = 1
    config_path = tmp_path / "judge-config.yaml"
    config_path.write_text(yaml.safe_dump(priced))
    code = signals.main([
        "judge", "--results", str(tmp_path / "judge"), "--datasets", "squad", "--seed", "0",
        "--config", str(config_path), "--max-usd", "3", "--total-usd-cap", "30",
        "--inference-mode", "on_demand",
    ])
    assert code == 0
    expected_judge = cost_usd(priced, priced["judge_model_id"], 80, 12)
    judge_ledger = read_jsonl(tmp_path / "judge" / "cost_ledger.jsonl")
    assert len(judge_ledger) == 4
    assert all(row["stage"] == "judge" for row in judge_ledger)
    assert all(row["usd"] == pytest.approx(expected_judge) for row in judge_ledger)
    assert all(row["input_tokens"] == 80 and row["output_tokens"] == 12 for row in judge_ledger)


def test_reprice_ledger_rewrites_usd_and_the_cap_reads_it(tmp_path: Path, capsys):
    from experiments.common import cost_usd
    from experiments.ledger import spent_usd

    cfg = load_config()
    haiku = cfg["generator_model_id"]
    llama = cfg["judge_model_id"]
    rows = [
        {
            "stage": "generate", "qid": "q1", "model_id": haiku,
            "input_tokens": 1_000_000, "output_tokens": 1_000_000, "usd": 6.0,
        },
        {
            "stage": "judge", "qid": "q2", "model_id": llama,
            "input_tokens": 1_000_000, "output_tokens": 0, "usd": 0.72,
        },
        {
            "stage": "generate", "qid": "q3", "model_id": haiku,
            "input_tokens": 0, "output_tokens": 0, "usd": 0.5,
        },
    ]
    path = tmp_path / "cost_ledger.jsonl.gz"
    _write_gzip_jsonl(path, rows)
    mod = _script("reprice_ledger")
    mod.now_iso = lambda: "2026-10-07T12:00:00+00:00"
    assert mod.DEFAULT_LEDGER.name == "cost_ledger.jsonl.gz"
    assert mod.main(["--ledger", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "old total USD: 7.220000" in printed
    assert "new total USD: 7.320000" in printed
    assert f"wrote {path}" in printed
    assert not list(tmp_path.glob(".*.tmp"))
    from experiments.common import read_jsonl
    rewritten = read_jsonl(path)
    haiku_price = {"input": pytest.approx(1.10), "output": pytest.approx(5.50)}
    assert rewritten[0]["usd_at_logged_price"] == 6.0
    assert rewritten[0]["usd"] == pytest.approx(cost_usd(cfg, haiku, 1_000_000, 1_000_000))
    assert rewritten[0]["usd"] == pytest.approx(6.60)
    assert rewritten[0]["price_usd_per_million"]["input"] == haiku_price["input"]
    assert rewritten[0]["price_usd_per_million"]["output"] == haiku_price["output"]
    assert rewritten[0]["repriced_at"] == "2026-10-07T12:00:00+00:00"
    assert rewritten[0]["stage"] == "generate"
    assert rewritten[1]["usd"] == pytest.approx(0.72)
    assert rewritten[1]["usd_at_logged_price"] == 0.72
    assert rewritten[1]["price_usd_per_million"]["input"] == pytest.approx(0.72)
    assert rewritten[2]["usd"] == 0.0
    assert rewritten[2]["usd_at_logged_price"] == 0.5
    assert spent_usd(tmp_path) == pytest.approx(7.32)

    mod.now_iso = lambda: "2026-10-08T00:00:00+00:00"
    assert mod.main(["--ledger", str(path)]) == 0
    again = read_jsonl(path)
    assert again[0]["usd_at_logged_price"] == 6.0
    assert again[0]["usd"] == pytest.approx(6.60)
    assert again[0]["repriced_at"] == "2026-10-08T00:00:00+00:00"
    assert again[2]["usd_at_logged_price"] == 0.5

    plain = tmp_path / "plain.jsonl"
    _write_jsonl(plain, [rows[1]])
    assert mod.main(["--ledger", str(plain)]) == 0
    assert json.loads(plain.read_text().splitlines()[0])["usd"] == pytest.approx(0.72)

    broken = tmp_path / "broken.jsonl.gz"
    before = _write_gzip_jsonl(broken, [
        rows[0],
        {"stage": "generate", "qid": "q9", "model_id": haiku, "input_tokens": 10, "usd": 1.0},
    ])
    code = mod.main(["--ledger", str(broken)])
    err = capsys.readouterr().err
    assert code == 2
    assert "output_tokens" in err
    assert broken.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
    assert mod.main(["--ledger", str(tmp_path / "missing.jsonl.gz")]) == 2


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> bytes:
    import gzip
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    with gzip.open(path, "wb") as fh:
        fh.write(payload)
    return path.read_bytes()


def test_stages_ignore_retrieval_rows_outside_the_sample(tmp_path):
    """Generation and the stages that join records stay inside results/samples/."""
    from experiments.common import read_jsonl
    from experiments.records import load_joined

    _mini_squad(tmp_path)
    passages = [{"id": "p", "title": "France", "text": "Paris is the capital of France."}]
    extra_retrieval = []
    extra_generation = []
    for arm in ("semantic_search", "graph_first", "keyword_boosted", "hybrid"):
        extra_retrieval.append({
            "qid": "q_extra", "dataset": "squad", "arm": arm, "question_type": "answerable",
            "question": "Outside the sample?", "passages": passages, "pool_size": 1,
        })
        extra_generation.append({
            "qid": "q_extra", "dataset": "squad", "arm": arm, "question_type": "answerable",
            "answer": "no", "raw_response": '{"answer": "no", "confidence": 0.2}',
            "self_confidence": 0.2, "self_reported": True, "self_status": "ok",
            "input_tokens": 10, "output_tokens": 20,
            "model_id": "some-other-model", "error": None,
        })
    retrieval = read_jsonl(tmp_path / "retrieval" / "squad.jsonl") + extra_retrieval
    generation = read_jsonl(tmp_path / "generation" / "squad.jsonl") + extra_generation
    _write_jsonl(tmp_path / "retrieval" / "squad.jsonl", retrieval)
    _write_jsonl(tmp_path / "generation" / "squad.jsonl", generation)

    gen = _script("generate")
    assert gen.main(["--results", str(tmp_path), "--datasets", "squad", "--seed", "0", "--dry-run"]) == 0
    estimate = json.loads((tmp_path / "generation" / "dry_run_cost.json").read_text())
    assert estimate["estimate"]["n_calls"] == 4

    signals = _script("signals")
    assert signals.main(["lexical", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    lexical = read_jsonl(tmp_path / "signals" / "lexical.jsonl")
    assert {row["qid"] for row in lexical} == {"q1"}
    assert len(lexical) == 4

    joined = load_joined(tmp_path, load_config(), ["squad"])
    assert {row["qid"] for row in joined} == {"q1"}
    assert len(joined) == 4

    retrieve = _script("retrieve")
    retrieve.load_embedder = lambda model: (lambda texts: [[0.0] for _ in texts], {"revision": "test"})
    retrieve.load_entities = lambda model: (lambda text: [], {"spacy_model": model})
    assert retrieve.main(["--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    stats = json.loads((tmp_path / "retrieval" / "retrieval_stats.json").read_text())
    assert stats["datasets"]["squad"]["n_questions"] == 1
    assert stats["seed"] == 0
