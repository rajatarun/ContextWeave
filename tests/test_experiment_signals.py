"""Claim grounding, seeded judge sample, percentile, logistic, and oracles."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import ProtocolError, load_config, read_jsonl
from experiments.ledger import Budget
from experiments.signal_score import (
    assign_self_percentiles, fit_logistic, lexical_unit, logistic_rows, oracle_correct,
    oracle_retrieval, pack_windows, predict_logistic, score_stored_claim, seeded_ids,
    self_percentile, stream_seed, take_count, windowed_unit,
)
from experiments.signals_stage import build_judge_prompt, judge_rows_batch, nli_limit

_ROLE = "arn:aws:iam::example/batch-role"
_BUCKET = "example-batch-bucket"
_JOB_ARN = "arn:aws:bedrock:us-east-1:example:model-invocation-job/signals"


def _count(text: str) -> int:
    return len(text.split()) if text and text.strip() else 0


def _truncate(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def test_seeded_prefix_rounds_half_up_and_keeps_the_smaller_rate():
    assert take_count(10, 0.6) == 6
    assert take_count(1, 0.6) == 1
    assert take_count(5, 0.2) == 1
    assert take_count(0, 0.6) == 0
    ids = [f"q{i:02d}" for i in range(10)]
    stream = stream_seed(0, "judge")
    assert stream_seed(0, "judge") == stream
    assert stream_seed(0, "holdout") != stream
    small = seeded_ids(ids, 0.2, stream)
    large = seeded_ids(ids, 0.6, stream)
    full = seeded_ids(ids, 1.0, stream)
    assert small == large[:len(small)]
    assert large == full[:len(large)]
    assert seeded_ids(ids, 0.6, stream) == large


def test_stored_claim_is_supported_by_a_pair_when_neither_passage_reaches_tau():
    claim = "alpha beta gamma delta epsilon"
    passages = ["alpha beta extra words", "gamma delta extra words"]
    scored = score_stored_claim(claim, passages, lexical_unit, 0.6)
    assert scored["value"] == 1.0
    assert scored["detail"]["supported_by_single"] == 0
    assert scored["detail"]["supported_by_pair"] == 1
    numbered = score_stored_claim(
        "alpha beta counted 400 units total",
        ["alpha beta counted 40 units total"],
        lexical_unit, 0.6,
    )
    assert numbered["value"] == 0.0
    assert score_stored_claim(None, passages, lexical_unit, 0.6)["reason"] == "no_claim"
    assert score_stored_claim("   ", passages, lexical_unit, 0.6)["reason"] == "no_claim"
    abstained = score_stored_claim(
        "The passages do not contain the answer.", passages, lexical_unit, 0.6,
    )
    assert abstained["value"] is None and abstained["reason"] == "no_claims"
    assert score_stored_claim(claim, [], lexical_unit, 0.6)["reason"] == "no_passages"


def test_nli_windows_cut_past_the_limit_and_record_the_cut():
    packed = pack_windows(
        "word " * 30 + "MARK",
        "one two",
        _count, _truncate, limit=10, special_tokens=2,
    )
    assert packed["n_truncated"] >= 1
    assert packed["windows"]
    assert all(_count(window["text"]) <= 10 - 2 - 2 for window in packed["windows"])
    assert all("MARK" not in window["text"] for window in packed["windows"])

    def predict(claim: str, window: str) -> float:
        return 1.0 if "MARK" in window else 0.0

    unit = windowed_unit(predict, _count, _truncate, 10, 2)
    hidden, trunc, n_windows = unit("one two", "word " * 30 + "MARK")
    assert hidden == 0.0 and trunc >= 1 and n_windows >= 1
    visible, visible_trunc, _ = unit("one two", "MARK is here.")
    assert visible == 1.0 and visible_trunc == 0
    assert nli_limit(128, 512) == 128
    assert nli_limit(None, 512) == 512
    assert nli_limit(2048, 512) == 512
    scored = score_stored_claim("one two three", ["MARK is here."], unit, 0.6)
    assert scored["value"] == 1.0
    assert scored["detail"]["n_truncated_windows"] == 0


def test_self_percentile_uses_other_rows_in_the_dataset_and_counts_ties_half():
    assert self_percentile(0.8, [0.2]) == 1.0
    assert self_percentile(0.2, [0.8]) == 0.0
    assert self_percentile(0.5, []) is None
    rows = [
        {"qid": "a", "dataset": "squad", "arm": "semantic_search", "question_type": "squad",
         "self_status": "ok", "self_confidence": 0.5},
        {"qid": "b", "dataset": "squad", "arm": "semantic_search", "question_type": "squad",
         "self_status": "ok", "self_confidence": 0.5},
        {"qid": "c", "dataset": "squad", "arm": "semantic_search", "question_type": "squad",
         "self_status": "ok", "self_confidence": 0.9},
        {"qid": "d", "dataset": "hotpot", "arm": "semantic_search", "question_type": "hotpot",
         "self_status": "ok", "self_confidence": 0.1},
        {"qid": "e", "dataset": "squad", "arm": "graph_first", "question_type": "squad",
         "self_status": "omitted", "self_confidence": None},
    ]
    scored = {row["qid"]: row for row in assign_self_percentiles(rows)}
    assert scored["a"]["value"] == pytest.approx(0.25)
    assert scored["b"]["value"] == pytest.approx(0.25)
    assert scored["c"]["value"] == pytest.approx(1.0)
    assert scored["d"]["reason"] == "no_peers" and scored["d"]["value"] is None
    assert scored["e"]["reason"] == "omitted" and scored["e"]["value"] is None


def test_logistic_scores_only_the_holdout_and_refuses_one_class():
    rows = []
    for i in range(8):
        rows.append({
            "qid": f"q{i}", "dataset": "squad", "arm": "semantic_search", "question_type": "squad",
            "self_status": "ok",
            # Three columns that are not affine copies of each other, so the
            # design matrix has a pivot. Correctness still rises with the index.
            "self_confidence": 0.2 + 0.08 * i,
            "lexical": 0.15 + 0.1 * (i % 3),
            "nli": 0.12 + (0.55 if i >= 4 else 0.0) + 0.03 * (i % 2),
            "correct": 0 if i < 4 else 1,
        })
    holdout = set(seeded_ids([row["qid"] for row in rows], 0.2, stream_seed(0, "holdout")))
    written, fit = logistic_rows(rows, holdout_ids=holdout)
    by_qid = {row["qid"]: row for row in written}
    assert fit["features"] == ["self", "lexical", "nli"]
    assert fit["coefficients"]["squad"]["n_fit"] == 8 - len(holdout)
    for qid, row in by_qid.items():
        if qid in holdout:
            assert row["fold"] == "holdout" and row["value"] is not None
            assert 0.0 <= row["value"] <= 1.0
        else:
            assert row["fold"] == "fit" and row["value"] is None and row["reason"] == "fit_fold"
    high_ids = [qid for qid in holdout if int(qid[1:]) >= 4]
    low_ids = [qid for qid in holdout if int(qid[1:]) < 4]
    if high_ids and low_ids:
        assert by_qid[high_ids[0]]["value"] > by_qid[low_ids[0]]["value"]
    beta = fit_logistic([[0.1, 0.1, 0.1], [0.2, 0.2, 0.1], [0.8, 0.9, 0.7], [0.9, 0.8, 0.9]], [0, 0, 1, 1])
    assert predict_logistic(beta, [0.9, 0.9, 0.9]) > predict_logistic(beta, [0.1, 0.1, 0.1])
    with pytest.raises(ProtocolError, match="one class"):
        fit_logistic([[0.1], [0.2], [0.3], [0.4]], [1, 1, 1, 1])
    with pytest.raises(ProtocolError, match="collinear"):
        fit_logistic([[0.1, 0.1], [0.2, 0.2], [0.8, 0.8], [0.9, 0.9]], [0, 0, 1, 1])
    missing = dict(rows[0], qid="m", lexical=None)
    mixed, _ = logistic_rows([missing, *rows], holdout_ids=holdout)
    assert next(row for row in mixed if row["qid"] == "m")["reason"] == "feature_missing"


def test_oracles_follow_token_f1_and_source_retrieved():
    assert oracle_correct("Paris", ["Paris"], False) == 1
    assert oracle_correct("Rome", ["Paris"], False) == 0
    assert oracle_correct("insufficient evidence", [], True) == 1
    assert oracle_retrieval(True) == 1
    assert oracle_retrieval(False) == 0


def test_judge_evidence_cut_is_recorded():
    long = "x" * 13000
    built = build_judge_prompt("Where?", "Paris", [long])
    assert built["evidence_truncated"] is True
    assert built["n_evidence_chars"] == 12000
    assert "Paris" in built["prompt"]
    empty = build_judge_prompt("Where?", "Paris", ["  "])
    assert empty["prompt"] is None and empty["reason"] == "no_passages"


def _batch_cfg() -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["batch"]["role_arn"] = _ROLE
    cfg["batch"]["bucket"] = _BUCKET
    cfg["batch"]["min_records"] = 1
    cfg["v2"]["judge_sample_rate"] = 1
    return cfg


def _judge_row(qid: str, arm: str = "semantic_search") -> dict:
    return {
        "qid": qid, "dataset": "squad", "arm": arm, "question_type": "squad",
        "question": "Where?", "answer": "Paris",
        "passages": ["Paris is the capital."],
    }


class _Store:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}


class _S3:
    def __init__(self, store: _Store):
        self.store = store
        self.puts = 0

    def put_object(self, Bucket, Key, Body):
        self.puts += 1
        data = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        self.store.objects[(Bucket, Key)] = data

    def list_objects_v2(self, **kwargs):
        keys = [
            key for (bucket, key) in self.store.objects
            if bucket == kwargs["Bucket"] and key.startswith(kwargs["Prefix"])
        ]
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def get_object(self, Bucket, Key):
        data = self.store.objects[(Bucket, Key)]

        class Body:
            def read(self_inner):
                return data

        return {"Body": Body()}


class _Bedrock:
    def __init__(self, store: _Store):
        self.store = store
        self.creates = 0

    def create_model_invocation_job(self, **kwargs):
        self.creates += 1
        assert kwargs["roleArn"] == _ROLE
        assert "239571291755" not in json.dumps(kwargs)
        assert kwargs["modelInvocationType"] == "InvokeModel"
        return {"jobArn": _JOB_ARN}

    def get_model_invocation_job(self, jobIdentifier):
        self._write_output()
        return {"status": "Completed"}

    def _write_output(self):
        inputs = [key for (_bucket, key) in self.store.objects if key.endswith("/input.jsonl")]
        raw = self.store.objects[(_BUCKET, inputs[0])].decode("utf-8")
        lines = []
        for line in raw.splitlines():
            row = json.loads(line)
            assert "Where?" in row["modelInput"]["prompt"]
            lines.append(json.dumps({
                "recordId": row["recordId"],
                "modelOutput": {
                    "generation": '{"score": 0.25}',
                    "prompt_token_count": 30,
                    "generation_token_count": 6,
                    "stop_reason": "stop",
                },
            }))
        out_key = inputs[0].replace("/input.jsonl", "/out/output.jsonl.out")
        self.store.objects[(_BUCKET, out_key)] = ("\n".join(lines) + "\n").encode("utf-8")


def test_judge_batch_respects_the_minimum_and_the_cap(tmp_path):
    cfg = _batch_cfg()
    cfg["batch"]["min_records"] = 100

    class Bedrock:
        def create_model_invocation_job(self, **_kwargs):
            raise AssertionError("create_model_invocation_job must not be called")

    class S3:
        def put_object(self, **_kwargs):
            raise AssertionError("the job must not be uploaded")

    budget = Budget(tmp_path, "judge", max_usd=10, total_usd_cap=30)
    summary = judge_rows_batch(
        cfg, [_judge_row("q1")], tmp_path / "judge.jsonl", budget,
        model_id=cfg["judge_model_id"], s3=S3(), bedrock=Bedrock(), seed=0,
        sampled={"q1"}, sleep=lambda _s: None,
    )
    assert summary["stopped"] is True
    assert "min_records is 100" in summary["stop_reason"]
    assert "--inference-mode on_demand" in summary["stop_reason"]

    cfg["batch"]["min_records"] = 1
    cap_budget = Budget(tmp_path / "cap", "judge", max_usd=1e-6, total_usd_cap=30)
    capped = judge_rows_batch(
        cfg, [_judge_row("q1")], tmp_path / "cap" / "judge.jsonl", cap_budget,
        model_id=cfg["judge_model_id"], s3=S3(), bedrock=Bedrock(), seed=0,
        sampled={"q1"}, sleep=lambda _s: None,
    )
    assert capped["stopped"] is True
    assert "spend cap reached" in capped["stop_reason"]


def test_judge_batch_resumes_the_stored_job(tmp_path):
    cfg = _batch_cfg()
    store = _Store()
    s3 = _S3(store)
    bedrock = _Bedrock(store)
    out = tmp_path / "judge.jsonl"
    budget = Budget(tmp_path, "judge", max_usd=10, total_usd_cap=30)
    rows = [_judge_row("q1", arm) for arm in ("semantic_search", "graph_first")]
    first = judge_rows_batch(
        cfg, rows, out, budget, model_id=cfg["judge_model_id"], s3=s3, bedrock=bedrock,
        seed=7, sampled={"q1"}, sleep=lambda _s: None,
    )
    assert first["written"] == 2 and first["stopped"] is False
    assert bedrock.creates == 1
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert {row["value"] for row in written} == {0.25}
    assert {row["pricing"] for row in written} == {"batch"}
    out.unlink()
    second = judge_rows_batch(
        cfg, rows, out, Budget(tmp_path, "judge", max_usd=10, total_usd_cap=30),
        model_id=cfg["judge_model_id"], s3=s3, bedrock=bedrock, seed=7,
        sampled={"q1"}, sleep=lambda _s: None,
    )
    assert second["written"] == 2
    assert bedrock.creates == 1
    assert s3.puts == 1


def _script(name: str):
    import importlib.util
    path = ROOT / "scripts" / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exp_script_{name}_signals", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mini(tmp_path: Path, claims: dict[str, str]) -> None:
    cfg = load_config()
    arms = ("semantic_search", "graph_first", "keyword_boosted", "hybrid")
    questions = []
    retrieval = []
    generation = []
    for i, (qid, claim) in enumerate(claims.items()):
        gold = "Paris" if i % 2 == 0 else "Rome"
        # Odd rows answer the other city, so the retrieval oracle and the
        # correctness oracle disagree on q1.
        answer = gold if i % 2 == 0 else "Paris"
        questions.append({
            "qid": qid, "dataset": "squad", "question": f"Where {qid}?",
            "question_type": "squad", "gold_answers": [gold], "unanswerable": False,
            "source_passage_ids": ["p"], "yes_no": False, "schema_version": 2,
        })
        for arm in arms:
            retrieval.append({
                "qid": qid, "dataset": "squad", "arm": arm, "question_type": "squad",
                "question": f"Where {qid}?",
                "passages": [{"id": "p", "title": "", "text": f"{gold} is the capital of France."}],
                "pool_size": 1, "gold_in_top_k": True, "source_retrieved": i % 2 == 0,
                "unanswerable": False, "source_passage_ids": ["p"],
            })
            generation.append({
                "qid": qid, "dataset": "squad", "arm": arm, "question_type": "squad",
                "answer": answer, "claim": claim,
                "raw_response": json.dumps({"answer": answer, "claim": claim, "confidence": 0.2 + i / 10}),
                "self_confidence": round(0.15 + 0.1 * i, 2),
                "self_reported": True, "self_status": "ok",
                "input_tokens": 10, "output_tokens": 5,
                "model_id": cfg["generator_model_id"], "error": None, "schema_version": 2,
            })
    (tmp_path / "samples").mkdir()
    (tmp_path / "retrieval").mkdir()
    (tmp_path / "generation").mkdir()
    for folder, rows, name in (
        ("samples", questions, "squad.jsonl"),
        ("retrieval", retrieval, "squad.jsonl"),
        ("generation", generation, "squad.jsonl"),
    ):
        path = tmp_path / folder / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "samples" / "squad.passages.jsonl").write_text(
        json.dumps({"id": "p", "title": "", "text": "Paris is the capital."}) + "\n"
    )


def test_signal_commands_write_claim_scores_percentile_oracles_and_the_holdout(tmp_path, monkeypatch):
    def _claim(i: int) -> str:
        gold = "Paris" if i % 2 == 0 else "Rome"
        supported = f"{gold} is the capital of France."
        entailed = "zebra quilt mosaic stands."
        neither = "unrelated pottery kiln vessel."
        # Lexical support and NLI support land on different sentences, so the
        # two stored fractions are not the same column.
        pattern = (
            supported,
            entailed,
            f"{supported} {neither}",
            f"{entailed} {neither}",
            f"{supported} {entailed}",
            neither,
            supported,
            f"{supported} {entailed} {neither}",
        )
        return pattern[i]

    claims = {f"q{i}": _claim(i) for i in range(8)}
    _mini(tmp_path, claims)
    signals = _script("signals")

    def predict(claim: str, window: str) -> float:
        return 0.9 if "zebra" in claim else 0.2

    def count(text: str) -> int:
        return _count(text)

    signals.load_nli_tools = lambda _model: (
        predict, count, _truncate,
        {"nli_model": "test-nli", "revision": "abc", "labels": ["entailment"], "model_max_length": 512},
    )
    assert signals.main(["lexical", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    lexical = read_jsonl(tmp_path / "signals" / "lexical.jsonl")
    assert len(lexical) == 32
    lexical_by_qid = {}
    for row in lexical:
        assert row["reason"] is None
        lexical_by_qid.setdefault(row["qid"], set()).add(row["value"])
    assert lexical_by_qid["q0"] == {1.0}
    assert lexical_by_qid["q1"] == {0.0}
    assert lexical_by_qid["q2"] == {0.5}
    assert signals.main(["nli", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    nli_meta = json.loads((tmp_path / "signals" / "nli_meta.json").read_text())
    assert nli_meta["seed"] == 0
    assert nli_meta["nli_max_tokens"] == 512
    assert nli_meta["n_truncated_windows"] == 0
    assert signals.main(["self_percentile", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    percentile = read_jsonl(tmp_path / "signals" / "self_percentile.jsonl")
    assert len(percentile) == 32
    assert all(row["value"] is not None for row in percentile)
    assert signals.main(["oracle", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    correct = {row["qid"]: row["value"] for row in read_jsonl(tmp_path / "signals" / "oracle_correct.jsonl")}
    retrieved = {row["qid"]: row["value"] for row in read_jsonl(tmp_path / "signals" / "oracle_retrieval.jsonl")}
    assert correct["q0"] == 1 and correct["q1"] == 0
    assert retrieved["q0"] == 1 and retrieved["q1"] == 0
    assert signals.main(["logistic", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"]) == 0
    logistic = read_jsonl(tmp_path / "signals" / "logistic.jsonl")
    meta = json.loads((tmp_path / "signals" / "logistic_meta.json").read_text())
    assert meta["holdout"]["stream"] == "holdout"
    assert meta["holdout"]["seed"] == 0
    holdout = set(meta["holdout"]["sampled_qids"])
    assert holdout
    for row in logistic:
        if row["qid"] in holdout:
            assert row["fold"] == "holdout" and row["value"] is not None
        else:
            assert row["reason"] == "fit_fold"
    again = signals.main(["logistic", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"])
    assert again == 0
    (tmp_path / "signals" / "logistic.jsonl.gz").unlink()
    # A partial file is a different function of the log.
    partial = tmp_path / "signals" / "self_percentile.jsonl.gz"
    # The percentile file is whatever stage_jsonl chose. Remove and rewrite a partial plain file.
    for path in (tmp_path / "signals").glob("self_percentile.jsonl*"):
        path.unlink()
    (tmp_path / "signals" / "self_percentile.jsonl").write_text(json.dumps({
        "qid": "q0", "arm": "semantic_search", "value": 0.5, "reason": None,
    }) + "\n")
    refused = signals.main(["self_percentile", "--results", str(tmp_path), "--datasets", "squad", "--seed", "0"])
    assert refused == 2
