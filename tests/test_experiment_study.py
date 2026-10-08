"""Subset study, projection, and the runner. No live AWS calls."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import ProtocolError, load_config, read_jsonl, write_json
from experiments.cost_projection import format_projection, project
from experiments.reporting import render
from experiments.runner import stage_commands
from experiments.study_stage import (
    beta_update, configured_study, gaussian_update, information_ratios,
    judge_coverage_weighted, run_streams, seed_list,
)
from experiments.subset_stage import (
    agreement_scores, choose_ids, choose_subset, require_haiku_settings, score_agreement,
)

_ROLE = "arn:aws:iam::example/batch-role"
_BUCKET = "example-batch-bucket"
_JOB = "arn:aws:bedrock:us-east-1:example:model-invocation-job/study"
_ANSWER = '{"answer": "Paris", "claim": "Paris is the capital.", "confidence": 0.4}'


def _script(name: str, filename: str):
    path = ROOT / "scripts" / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arms(qid: str) -> list[dict]:
    rows = []
    for arm in ("semantic_search", "graph_first", "keyword_boosted", "hybrid"):
        rows.append({
            "qid": qid,
            "dataset": "squad",
            "arm": arm,
            "question_type": "squad",
            "question": "Where?",
            "passages": [{"id": "p", "title": "", "text": "Paris is the capital."}],
            "unanswerable": False,
            "source_retrieved": True,
            "gold_in_top_k": True,
        })
    return rows


class _Client:
    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        assert "239571291755" not in json.dumps(kwargs)
        return {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": _ANSWER}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }


class _Store:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}


class _S3:
    def __init__(self, store: _Store):
        self.store = store

    def put_object(self, Bucket, Key, Body):
        data = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        self.store.objects[(Bucket, Key)] = data
        assert "239571291755" not in data.decode("utf-8")

    def list_objects_v2(self, **kwargs):
        bucket = kwargs["Bucket"]
        prefix = kwargs["Prefix"]
        keys = [key for (b, key) in self.store.objects if b == bucket and key.startswith(prefix)]
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
        self.creates = []

    def create_model_invocation_job(self, **kwargs):
        self.creates.append(kwargs)
        assert "239571291755" not in json.dumps(kwargs)
        return {"jobArn": _JOB}

    def get_model_invocation_job(self, jobIdentifier):
        assert jobIdentifier == _JOB
        self._write_output()
        return {"status": "Completed"}

    def _write_output(self):
        inputs = [key for (_bucket, key) in self.store.objects if key.endswith("/input.jsonl")]
        assert len(inputs) == 1
        raw = self.store.objects[(_BUCKET, inputs[0])].decode("utf-8")
        lines = []
        for line in raw.splitlines():
            row = json.loads(line)
            assert row["modelInput"]["schemaVersion"] == "messages-v1"
            assert row["modelInput"]["inferenceConfig"]["temperature"] == 0.0
            lines.append(json.dumps({
                "recordId": row["recordId"],
                "modelOutput": {
                    "output": {"message": {"content": [{"text": _ANSWER}]}},
                    "usage": {"inputTokens": 11, "outputTokens": 7},
                    "stopReason": "end_turn",
                },
            }))
        out_key = inputs[0].replace("/input.jsonl", "/out/output.jsonl.out")
        self.store.objects[(_BUCKET, out_key)] = ("\n".join(lines) + "\n").encode("utf-8")


def _write_subset(results: Path, qid: str = "q1") -> None:
    (results / "samples").mkdir(parents=True)
    (results / "samples" / "squad.jsonl").write_text(json.dumps({
        "qid": qid, "dataset": "squad", "question": "Where?", "question_type": "squad",
        "gold_answers": ["Paris"], "unanswerable": False, "yes_no": False,
    }) + "\n")
    retrieval = results / "retrieval"
    retrieval.mkdir(parents=True)
    with (retrieval / "squad.jsonl").open("w") as fh:
        for row in _arms(qid):
            fh.write(json.dumps(row) + "\n")
    manifest = choose_subset({"squad": [qid]}, 1, 0)
    write_json(results / "subset" / "subset.json", manifest)


def test_subset_is_a_prefix_and_does_not_pad():
    ids = ["d", "b", "a", "c"]
    small = choose_ids(ids, 2, 0, "squad")
    large = choose_ids(ids, 3, 0, "squad")
    assert large["qids"][:2] == small["qids"]
    assert small["stream"] == "subset:squad"
    assert small["n"] == 2
    with pytest.raises(ProtocolError, match="Refusing to pad"):
        choose_ids(ids, 5, 0, "squad")
    other = choose_ids(ids, 2, 0, "hotpot")
    assert other["stream_seed"] != small["stream_seed"]


def test_agreement_modal_fraction_and_partial_set():
    scores = agreement_scores(["Paris", "paris", "Lyon", "Paris.", "Paris"])
    assert scores["modal_fraction"] == pytest.approx(0.8)
    assert scores["pairwise_agreement"] == pytest.approx(0.6)
    keys = [("squad", "q1", "semantic_search")]
    complete = score_agreement(keys, [{("q1", "semantic_search"): {"answer": "Paris"}}] * 5)
    assert complete["partial"] is False
    assert complete["datasets"]["squad"]["mean_modal_fraction"] == pytest.approx(1.0)
    partial = score_agreement(keys, [{("q1", "semantic_search"): {"answer": "Paris"}}] * 4 + [{}])
    assert partial["partial"] is True
    assert partial["datasets"]["squad"]["mean_modal_fraction"] is None
    assert partial["rows"] == []
    assert partial["n_incomplete"] == 1


def test_haiku_temperature_is_required():
    cfg = copy.deepcopy(load_config())
    require_haiku_settings(cfg)
    cfg["v2"]["subset_temperature"] = 0
    with pytest.raises(ProtocolError, match="temperature"):
        require_haiku_settings(cfg)


def test_haiku_on_demand_samples_are_separate_files(tmp_path, monkeypatch):
    _write_subset(tmp_path)
    client = _Client()
    mod = _script("subset_haiku_cli", "subset.py")
    monkeypatch.setattr(mod, "make_client", lambda region: client)
    code = mod.main([
        "haiku", "--results", str(tmp_path), "--seed", "0", "--datasets", "squad",
        "--max-usd", "10", "--inference-mode", "on_demand",
    ])
    assert code == 0
    assert len(client.calls) == 20
    assert {call["inferenceConfig"]["temperature"] for call in client.calls} == {1.0}
    assert {call["modelId"] for call in client.calls} == {load_config()["generator_model_id"]}
    for index in range(5):
        rows = read_jsonl(tmp_path / "subset" / "haiku" / f"sample_{index}" / "generations.jsonl")
        assert len(rows) == 4
        assert {row["temperature"] for row in rows} == {1.0}
        meta = json.loads((tmp_path / "subset" / "haiku" / f"sample_{index}" / "meta.json").read_text())
        assert meta["sample_index"] == index
        assert meta["seed"] == 0
        assert meta["bedrock_seed_parameter"] is False
    code = mod.main(["agreement", "--results", str(tmp_path), "--seed", "0", "--datasets", "squad"])
    assert code == 0
    agreement = json.loads((tmp_path / "subset" / "agreement.json").read_text())
    assert agreement["datasets"]["squad"]["mean_modal_fraction"] == pytest.approx(1.0)
    assert agreement["partial"] is False


def test_nova_on_demand_and_batch(tmp_path, monkeypatch):
    _write_subset(tmp_path)
    client = _Client()
    mod = _script("subset_nova_cli", "subset.py")
    monkeypatch.setattr(mod, "make_client", lambda region: client)
    code = mod.main([
        "nova", "--results", str(tmp_path), "--seed", "3", "--datasets", "squad",
        "--max-usd", "10", "--inference-mode", "on_demand",
    ])
    assert code == 0
    assert len(client.calls) == 4
    assert {call["modelId"] for call in client.calls} == {"us.amazon.nova-pro-v1:0"}
    assert {call["inferenceConfig"]["temperature"] for call in client.calls} == {0.0}
    rows = read_jsonl(tmp_path / "subset" / "nova" / "generations.jsonl")
    assert len(rows) == 4
    assert {row["model_id"] for row in rows} == {"us.amazon.nova-pro-v1:0"}
    assert {row["pricing"] for row in rows} == {"on_demand"}

    batch_root = tmp_path / "batch"
    _write_subset(batch_root)
    cfg = copy.deepcopy(load_config())
    cfg["batch"]["role_arn"] = _ROLE
    cfg["batch"]["bucket"] = _BUCKET
    cfg["batch"]["min_records"] = 1
    cfg_path = tmp_path / "batch-config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    store = _Store()
    bedrock = _Bedrock(store)
    monkeypatch.setattr(mod, "make_batch_clients", lambda region: (_S3(store), bedrock))
    code = mod.main([
        "nova", "--config", str(cfg_path), "--results", str(batch_root), "--seed", "3",
        "--datasets", "squad", "--max-usd", "10", "--inference-mode", "batch",
    ])
    assert code == 0
    assert len(bedrock.creates) == 1
    assert bedrock.creates[0]["modelId"] == "us.amazon.nova-pro-v1:0"
    written = read_jsonl(batch_root / "subset" / "nova" / "generations.jsonl")
    assert len(written) == 4
    assert {row["pricing"] for row in written} == {"batch"}
    assert written[0]["price_usd_per_million"] == {"input": 0.40, "output": 1.60}


def test_information_ratios_match_the_hand_calculation():
    pairs = [(0, 0.0), (0, 0.0), (1, 1.0), (1, 1.0)]
    ratios = information_ratios(pairs)
    assert ratios["s"] == pytest.approx(1.0)
    assert ratios["m"] == pytest.approx(0.5)
    assert ratios["var_r"] == pytest.approx(1.0 / 3.0)
    assert ratios["s2_over_var_r"] == pytest.approx(3.0)
    assert ratios["s2_over_m_1m"] == pytest.approx(4.0)
    assert ratios["information_per_round"] == pytest.approx(12.0)
    flat = information_ratios([(0, 0.5), (1, 0.5)])
    assert flat["s2_over_var_r"] is None
    assert "Var(R) is 0" in flat["null_reasons"]["s2_over_var_r"]
    edge = information_ratios([(1, 0.2), (1, 0.4)])
    assert edge["s2_over_m_1m"] is None
    assert edge["m"] == pytest.approx(1.0)


def test_beta_and_gaussian_updates_and_stream_determinism():
    assert beta_update(0.0, 0.0, 1.0, 0.5) == (1.0, 0.0)
    assert beta_update(1.0, 0.0, 0.0, 0.5) == (0.5, 1.0)
    state = {"n_obs": 0.0, "mean_obs": 0.0, "m2": 0.0}
    gaussian_update(state, 1.0, 0.5)
    gaussian_update(state, 0.0, 0.5)
    assert state["n_obs"] == pytest.approx(1.5)
    assert state["mean_obs"] == pytest.approx(1.0 / 3.0)
    assert state["m2"] == pytest.approx(1.0 / 3.0)
    rows = []
    for qid, correct in (("q1", 1), ("q2", 0)):
        for arm in ("semantic_search", "graph_first", "keyword_boosted", "hybrid"):
            rows.append({
                "qid": qid, "dataset": "squad", "arm": arm, "question_type": "squad",
                "correct": correct,
            })
    seeds = seed_list(0, 2)
    assert seeds == [0, 1]
    first = run_streams(rows, seeds, 20, [0.99])
    second = run_streams(rows, seeds, 20, [0.99])
    assert first["streams"] == second["streams"]
    labels = {(block["policy"], block["discount_label"]) for block in first["streams"]}
    assert labels == {
        ("beta", "none"), ("beta", "0.99"),
        ("gaussian", "none"), ("gaussian", "0.99"),
    }
    assert all(block["n_seeds"] == 2 and block["n_rounds"] == 20 for block in first["streams"])
    settings = configured_study(load_config())
    assert settings["replay_seeds"] == 500
    assert settings["replay_rounds"] == 10000
    assert settings["drift_discounts"] == [0.99, 0.995, 0.999]
    assert settings["judge_coverage"] == [0.01, 0.05, 0.20, 1.0]
    assert load_config()["replay_seeds"] == [0, 1, 2, 3, 4]


def test_judge_coverage_prefix_and_absent_judge():
    rows = []
    for index in range(4):
        rows.append({
            "qid": f"q{index}", "dataset": "squad", "arm": "semantic_search",
            "correct": index % 2, "lexical_grounding": 0.2, "judge": 0.9,
            "judge_reason": None,
        })
    body = judge_coverage_weighted(rows, [0.5, 1.0], 0, load_config())
    low, high = body["rates"]
    assert low["sampled_qids"] == high["sampled_qids"][:len(low["sampled_qids"])]
    assert high["n_sampled"] == 4
    assert low["verified"]["n"] > 0
    assert low["judge"]["n"] == low["n_sampled"]
    for row in rows:
        row["judge"] = None
        row["judge_reason"] = "signal_file_missing"
    absent = judge_coverage_weighted(rows, [0.05], 0, load_config())
    assert absent["judge_observed"] is False
    assert "absent" in absent["reason"]
    assert absent["rates"][0]["verified"] is None
    assert absent["rates"][0]["judge"] is None


def test_projection_for_1100_and_judge_60_percent():
    body = project(load_config(), 1100, 0.6)
    by_name = {stage["stage"]: stage for stage in body["stages"]}
    assert by_name["generate"]["calls"] == 13200
    assert by_name["judge"]["calls"] == 7920
    assert by_name["subset_haiku"]["calls"] == 15000
    assert by_name["subset_nova"]["calls"] == 3000
    assert by_name["adjudicate"]["calls"] is None
    assert by_name["adjudicate"]["on_demand_usd"] is None
    assert by_name["lexical"]["calls"] == 0
    assert by_name["generate"]["on_demand_usd"] > by_name["generate"]["batch_usd"] > 0
    assert by_name["subset_nova"]["on_demand_usd"] > by_name["subset_nova"]["batch_usd"] > 0
    text = format_projection(body)
    assert "calls=13200" in text
    assert "calls=7920" in text
    assert "calls=15000" in text
    assert "calls=3000" in text
    assert "adjudicate\tcalls=pending" in text
    assert "239571291755" not in text
    assert "called_model: false" in text
    assert body["kind"] == "projection"


def test_runner_prints_projection_and_refuses_to_spend(tmp_path, capsys):
    mod = _script("run_all_cli", "run_all.py")
    code = mod.main(["--project-only", "--n", "1100", "--judge-rate", "0.6", "--results", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "calls=13200" in out and "calls=pending" in out
    code = mod.main(["--results", str(tmp_path), "--n", "1100", "--judge-rate", "0.6"])
    assert code == 2
    captured = capsys.readouterr()
    assert "calls=7920" in captured.out
    assert "--max-usd" in captured.err
    commands = stage_commands(
        results=tmp_path, seed=0, max_usd=5, total_usd_cap=None,
        inference_mode="batch", dry_run=False, datasets=None,
    )
    haiku = next(cmd for cmd in commands if "subset.py" in cmd[1] and "haiku" in cmd)
    nova = next(cmd for cmd in commands if "subset.py" in cmd[1] and "nova" in cmd)
    replay = next(cmd for cmd in commands if cmd[1].endswith("replay.py"))
    assert "--max-usd" in haiku and "--inference-mode" in haiku and "batch" in haiku
    assert "--inference-mode" in nova
    assert "--seeds" not in replay
    assert "239571291755" not in json.dumps(commands)


def test_findings_pending_until_artifacts_exist_and_then_copies_them(tmp_path):
    findings, pending = render(tmp_path)
    assert "Study artifact is missing" in findings
    assert "results/study/study.json is missing" in pending
    assert "results/cost_projection.json is missing" in pending
    assert "results/subset/agreement.json is missing" in pending
    body = project(load_config(), 1100, 0.6)
    write_json(tmp_path / "cost_projection.json", body)
    write_json(tmp_path / "study" / "study.json", {
        "definitions": {
            "m": "Mean of joined correct Y among rows where R is observed.",
            "s2_over_var_r": "s^2 / Var(R).",
            "s2_over_m_1m": "s^2 / (m(1-m)).",
            "information_per_round": "s^2 / residual.",
            "judge_coverage": "Seeded judge coverage.",
        },
        "information": {
            "squad": {"oracle": {
                "n": 8, "m": 0.5, "s2_over_var_r": 3.5, "s2_over_m_1m": 4.0,
                "information_per_round": 28.0, "null_reasons": {},
            }},
        },
        "thompson": {
            "definitions": {"streams": "Questions are drawn with replacement."},
            "streams": [{
                "policy": "beta", "discount_label": "none", "n_seeds": 2, "n_rounds": 12,
                "pseudo_regret_mean": 1.5, "realized_regret_mean": 0.25, "best_arm_share_mean": 0.5,
            }],
        },
        "judge_coverage": {
            "reason": "judge values are absent (signal_file_missing)",
            "rates": [{
                "rate": 0.01, "verified": None, "judge": None,
                "reason": "judge values are absent (signal_file_missing)",
            }],
        },
        "assumption": {
            "definition": "Cluster-bootstrap intervals.",
            "datasets": {"squad": {"self": {"n_nonoverlapping_pairs": 0}}},
        },
        "nova": {"reason": "results/subset/nova/generations.jsonl is missing", "datasets": None},
    })
    findings, pending = render(tmp_path)
    assert "28.0000" in findings
    assert "1.5000" in findings
    assert "13200" in findings
    assert "Study artifact is missing" not in findings
    assert "nova" in pending
    assert "judge values are absent" in pending


def test_study_cli_on_a_joined_log(tmp_path):
    _write_subset(tmp_path, "q1")
    extra = {
        "qid": "q2", "dataset": "squad", "question": "Where else?", "question_type": "squad",
        "gold_answers": ["Paris"], "unanswerable": False, "yes_no": False,
    }
    with (tmp_path / "samples" / "squad.jsonl").open("a") as fh:
        fh.write(json.dumps(extra) + "\n")
    with (tmp_path / "retrieval" / "squad.jsonl").open("a") as fh:
        for row in _arms("q2"):
            fh.write(json.dumps(row) + "\n")
    gen = tmp_path / "generation"
    gen.mkdir()
    with (gen / "squad.jsonl").open("w") as fh:
        for qid, answer in (("q1", "Paris"), ("q2", "Rome")):
            for arm in ("semantic_search", "graph_first", "keyword_boosted", "hybrid"):
                fh.write(json.dumps({
                    "qid": qid, "dataset": "squad", "arm": arm, "answer": answer,
                    "self_confidence": 0.8, "self_status": "ok", "self_reported": True,
                }) + "\n")
    mod = _script("study_cli", "study.py")
    code = mod.main([
        "--results", str(tmp_path), "--seed", "0", "--datasets", "squad",
        "--seeds", "2", "--rounds", "12", "--bootstrap", "4",
        "--discounts", "0.99", "--coverage", "0.25,1",
    ])
    assert code == 0
    body = json.loads((tmp_path / "study" / "study.json").read_text())
    assert body["seed_count"] == 2
    assert body["rounds"] == 12
    assert body["seeds"] == [0, 1]
    oracle = body["information"]["all"]["oracle"]
    assert oracle["n"] == 8
    assert oracle["information_per_round"] == pytest.approx(28.0)
    assert "absent" in body["judge_coverage"]["reason"]
    assert body["nova"]["datasets"] is None
    assert body["assumption"]["datasets"]["squad"]["self"]["n_nonoverlapping_pairs"] == 0
    assert "239571291755" not in json.dumps(body["thompson"]["prior_weights"])
