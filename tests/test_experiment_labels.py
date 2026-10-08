"""Joined correctness, on-demand adjudication, and the HotpotQA yes/no split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.adjudicate_stage import ADJUDICATOR_MODEL_ID, parse_adjudication, user_prompt
from experiments.analyses_stage import hotpot_yes_no
from experiments.common import ProtocolError, load_config, read_jsonl
from experiments.labels import decide_correctness, f1_band
from experiments.metrics import score_answer
from experiments.reporting import render

_ARMS = ("semantic_search", "graph_first", "keyword_boosted", "hybrid")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _question(qid: str, gold: list[str], unanswerable: bool, yes_no: bool = False, dataset: str = "squad") -> dict:
    return {
        "qid": qid, "dataset": dataset, "question": f"Question {qid}?",
        "question_type": dataset, "gold_answers": gold, "unanswerable": unanswerable,
        "source_passage_ids": ["p"], "yes_no": yes_no, "schema_version": 2,
    }


def _arm_rows(question: dict, answer: str, *, source_retrieved: bool, gold_in_top_k: bool, status: str = "ok") -> tuple[list[dict], list[dict]]:
    retrieval = []
    generation = []
    for arm in _ARMS:
        retrieval.append({
            "qid": question["qid"], "dataset": question["dataset"], "arm": arm,
            "question_type": question["question_type"], "question": question["question"],
            "passages": [{"id": "p", "title": "", "text": "aa is here."}],
            "pool_size": 1, "gold_in_top_k": gold_in_top_k, "source_retrieved": source_retrieved,
            "unanswerable": question["unanswerable"], "source_passage_ids": ["p"],
        })
        generation.append({
            "qid": question["qid"], "dataset": question["dataset"], "arm": arm,
            "question_type": question["question_type"], "answer": answer,
            "raw_response": json.dumps({"answer": answer, "confidence": 0.4}),
            "self_confidence": 0.4, "self_reported": True, "self_status": status,
            "input_tokens": 10, "output_tokens": 5,
            "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "error": None,
        })
    return retrieval, generation


def _corpus(tmp_path: Path) -> dict[str, dict]:
    band = _question("q-band", ["aa"], False)
    exact = _question("q-exact", ["Paris"], False)
    miss = _question("q-miss", [], True)
    questions = [band, exact, miss]
    retrieval = []
    generation = []
    for question, answer, retrieved, gold_hit in (
        (band, "aa bb cc", True, True),
        (exact, "Paris", True, True),
        (miss, "insufficient evidence", False, False),
    ):
        ret, gen = _arm_rows(question, answer, source_retrieved=retrieved, gold_in_top_k=gold_hit)
        retrieval.extend(ret)
        generation.extend(gen)
    _write(tmp_path / "samples" / "squad.jsonl", questions)
    _write(tmp_path / "retrieval" / "squad.jsonl", retrieval)
    _write(tmp_path / "generation" / "squad.jsonl", generation)
    scored = score_answer("aa bb cc", ["aa"], False)
    assert f1_band(scored["f1"], 0.2, 0.8)
    assert scored["correct"] == 1
    return {"band": band, "exact": exact, "miss": miss}


def _script():
    import importlib.util
    path = ROOT / "scripts" / "experiments" / "adjudicate.py"
    spec = importlib.util.spec_from_file_location("exp_script_adjudicate_labels", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Client:
    def __init__(self, text: str = '{"correct": false}'):
        self.calls = []
        self.text = text

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 100, "outputTokens": 10},
            "stopReason": "end_turn",
        }


class _Denied(Exception):
    response = {"Error": {"Code": "AccessDeniedException", "Message": "no grant"}}


def test_joined_correctness_uses_abstention_and_the_adjudicator():
    assert f1_band(0.2, 0.2, 0.8) and f1_band(0.8, 0.2, 0.8)
    assert f1_band(0.19, 0.2, 0.8) is False and f1_band(0.81, 0.2, 0.8) is False
    base = dict(
        answer="insufficient evidence", gold_answers=[], unanswerable=True,
        self_status="ok", source_retrieved=False, gold_in_top_k=False, yes_no=False,
        f1_low=0.2, f1_high=0.8,
    )
    miss = decide_correctness(**base)
    assert miss["abstention_label"] == "abstain_retrieval_miss"
    assert miss["correct"] == 0 and miss["correct_source"] == "abstention"
    assert miss["token_f1_correct"] == 1
    credited = decide_correctness(**{**base, "source_retrieved": True, "gold_in_top_k": True})
    assert credited["correct"] == 1 and credited["abstention_label"] == "abstain_correct"
    wrong = decide_correctness(
        **{**base, "gold_answers": ["Paris"], "unanswerable": False,
           "source_retrieved": True, "gold_in_top_k": True},
    )
    assert wrong["correct"] == 0 and wrong["abstention_label"] == "abstain_incorrect"
    exact = decide_correctness(
        answer="Paris", gold_answers=["Paris"], unanswerable=False, self_status="ok",
        source_retrieved=True, gold_in_top_k=True, yes_no=True, f1_low=0.2, f1_high=0.8,
    )
    assert exact["correct"] == 1 and exact["correct_source"] == "token_f1" and exact["yes_no"] is True
    partial = decide_correctness(
        answer="aa bb cc", gold_answers=["aa"], unanswerable=False, self_status="ok",
        source_retrieved=True, gold_in_top_k=True, yes_no=False, f1_low=0.2, f1_high=0.8,
    )
    assert partial["in_f1_band"] is True and partial["correct"] is None
    decided = decide_correctness(
        answer="aa bb cc", gold_answers=["aa"], unanswerable=False, self_status="ok",
        source_retrieved=True, gold_in_top_k=True, yes_no=False, f1_low=0.2, f1_high=0.8,
        adjudication={"qid": "q", "arm": "semantic_search", "value": 0, "reason": None},
    )
    assert decided["correct"] == 0 and decided["correct_source"] == "adjudication"
    assert decided["token_f1_correct"] == 1
    with pytest.raises(ProtocolError, match="gold_in_top_k"):
        decide_correctness(**{**base, "source_retrieved": True, "gold_in_top_k": False})
    with pytest.raises(ProtocolError, match="disagrees"):
        decide_correctness(**base, check_stored=True, stored_label="abstain_correct", stored_counts=True)
    assert parse_adjudication('note {"correct": true}') is True
    assert parse_adjudication('{"correct": 1}') is None


def test_adjudication_is_on_demand_capped_logged_and_seeded(tmp_path, monkeypatch):
    _corpus(tmp_path)
    script = _script()
    client = _Client()
    script.make_client = lambda region: client
    assert script.main(["--dry-run", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    dry = json.loads((tmp_path / "adjudication" / "dry_run.json").read_text())
    assert dry["seed"] == 0 and dry["called_model"] is False and dry["deterministic"] is True
    assert dry["model_id"] == ADJUDICATOR_MODEL_ID
    assert dry["n_calls"] == 4
    assert {row["qid"] for row in dry["rows"]} == {"q-band"}
    assert client.calls == []

    assert script.main([
        "--results", str(tmp_path), "--datasets", "squad", "--seed", "0",
        "--max-usd", "1", "--total-usd-cap", "30",
    ]) == 0
    assert len(client.calls) == 4
    for call in client.calls:
        assert call["modelId"] == ADJUDICATOR_MODEL_ID
        assert call["inferenceConfig"]["temperature"] == 0.0
        assert "239571291755" not in json.dumps(call)
    written = read_jsonl(tmp_path / "adjudication" / "adjudication.jsonl")
    assert len(written) == 4
    assert {row["value"] for row in written} == {0}
    assert {row["seed"] for row in written} == {0}
    assert {row["reason"] for row in written} == {None}
    prompt = user_prompt("Question q-band?", "aa bb cc", ["aa"])
    assert written[0]["prompt"] == prompt
    assert written[0]["system_prompt"]
    assert written[0]["raw_response"] == '{"correct": false}'
    assert written[0]["usd"] == pytest.approx((100 * 0.15 + 10 * 0.60) / 1_000_000)
    meta = json.loads((tmp_path / "adjudication" / "adjudication_meta.json").read_text())
    assert meta["seed"] == 0 and meta["temperature"] == 0 and meta["called_model"] is True

    assert script.main([
        "--results", str(tmp_path), "--datasets", "squad", "--seed", "0",
        "--max-usd", "1", "--total-usd-cap", "30",
    ]) == 0
    assert len(client.calls) == 4

    from experiments.records import load_joined
    joined = {row["qid"]: row for row in load_joined(tmp_path, load_config(), ["squad"])}
    assert joined["q-band"]["correct"] == 0
    assert joined["q-band"]["token_f1_correct"] == 1
    assert joined["q-band"]["correct_source"] == "adjudication"
    assert joined["q-exact"]["correct"] == 1 and joined["q-exact"]["correct_source"] == "token_f1"
    assert joined["q-miss"]["correct"] == 0
    assert joined["q-miss"]["correct_source"] == "abstention"
    assert joined["q-miss"]["abstention_label"] == "abstain_retrieval_miss"
    assert joined["q-miss"]["source_retrieved"] is False
    assert joined["q-miss"]["gold_in_top_k"] is False


def test_adjudication_cap_access_denied_and_unparsed_stop(tmp_path):
    _corpus(tmp_path)
    script = _script()
    capped = _Client()
    script.make_client = lambda region: capped
    assert script.main([
        "--results", str(tmp_path), "--datasets", "squad", "--seed", "0",
        "--max-usd", "0.00000001", "--total-usd-cap", "30",
    ]) == 0
    assert capped.calls == []
    pending = json.loads((tmp_path / "adjudication" / "pending.json").read_text())
    assert len(pending["rows"]) == 4
    from experiments.records import load_joined
    joined = load_joined(tmp_path, load_config(), ["squad"])
    assert {row["qid"] for row in joined} == {"q-exact", "q-miss"}

    denied = _Client()
    denied.converse = lambda **kwargs: (_ for _ in ()).throw(_Denied())
    script.make_client = lambda region: denied
    (tmp_path / "adjudication" / "pending.json").unlink()
    assert script.main([
        "--results", str(tmp_path), "--datasets", "squad", "--seed", "0",
        "--max-usd", "1", "--total-usd-cap", "30",
    ]) == 2
    marker = json.loads((tmp_path / "adjudication" / "adjudicator_unavailable.json").read_text())
    assert "access" in marker["reason"]
    assert not (tmp_path / "adjudication" / "adjudication.jsonl").exists()
    assert not list((tmp_path / "adjudication").glob("adjudication.jsonl*"))

    garbled = _Client(text="not json")
    script.make_client = lambda region: garbled
    assert script.main([
        "--results", str(tmp_path), "--datasets", "squad", "--seed", "0",
        "--max-usd", "1", "--total-usd-cap", "30",
    ]) == 2
    rows = read_jsonl(tmp_path / "adjudication" / "adjudication.jsonl")
    assert rows[0]["reason"] == "unparseable" and rows[0]["value"] is None
    assert rows[0]["raw_response"] == "not json"
    assert rows[0]["prompt"]
    with pytest.raises(ProtocolError, match="unparseable"):
        load_joined(tmp_path, load_config(), ["squad"])


def test_hotpot_yes_no_is_reported_apart_from_other_hotpot_questions(tmp_path):
    def row(qid: str, yes_no: bool, correct: int, retrieved: bool) -> dict:
        return {
            "qid": qid, "dataset": "hotpot", "arm": "semantic_search", "yes_no": yes_no,
            "correct": correct, "token_f1_correct": 1 - correct, "gold_in_top_k": retrieved,
            "source_retrieved": retrieved, "abstention_label": None,
        }
    block = hotpot_yes_no([
        row("y1", True, 1, True),
        row("y2", True, 1, True),
        row("o1", False, 0, False),
        row("o2", False, 0, True),
    ])
    assert block["yes_no"]["n"] == 2 and block["yes_no"]["correct_rate"] == 1
    assert block["yes_no"]["source_retrieved_rate"] == 1
    assert block["other"]["n"] == 2 and block["other"]["correct_rate"] == 0
    assert block["other"]["gold_in_top_k_rate"] == 0.5
    assert block["other"]["token_f1_rate"] == 1
    analyses = {
        "datasets": {
            "hotpot": {
                "yes_no": block,
                "correctness_sensitivity": {
                    "primary_rule": "joined rule from the artifact",
                    "primary_rate": 0.5,
                    "secondary_rate": None,
                },
            },
        },
    }
    (tmp_path / "analyses").mkdir()
    (tmp_path / "analyses" / "analyses.json").write_text(json.dumps(analyses))
    findings, _pending = render(tmp_path)
    assert "HotpotQA yes/no" in findings
    assert "joined rule from the artifact" in findings
    assert "| yes/no | 2 | 1.0000 |" in findings
    assert "| other HotpotQA | 2 | 0.0000 |" in findings
