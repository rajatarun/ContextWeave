"""v2 sampling, labels, graph arm, and batch generation. No live AWS calls."""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bedrock_batch import (
    build_jsonl, create_or_find_job, model_family, model_input, parsed_output_rows,
    resolve_batch_location,
)
from experiments.common import ProtocolError, cost_usd, load_config, price_for
from experiments.data import hotpot_collection, nq_collection, squad_collection
from experiments.generate_stage import generate_rows, generate_rows_batch
from experiments.graph_rank import build_graph, graph_or_fallback, load_spacy_entities
from experiments.labels import (
    ABSTAIN_CORRECT, ABSTAIN_INCORRECT, ABSTAIN_RETRIEVAL_MISS, abstention_outcome,
    gold_in_top_k, retrieval_label, source_was_retrieved,
)
from experiments.ledger import Budget
from experiments.retrieve_stage import graph_scores, rank_pool, tokenize

_CAP = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
_ROLE = "arn:aws:iam::example/batch-role"
_BUCKET = "example-batch-bucket"
_JOB_ARN = "arn:aws:bedrock:us-east-1:example:model-invocation-job/v2test"


def _caps(text: str) -> list[str]:
    return _CAP.findall(text or "")


def _batch_cfg() -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["batch"]["role_arn"] = _ROLE
    cfg["batch"]["bucket"] = _BUCKET
    cfg["batch"]["min_records"] = 1
    return cfg


def _retrieval(qid: str, *, unanswerable: bool, source_retrieved: bool, answerable_text: str = "Paris") -> dict:
    return {
        "qid": qid,
        "dataset": "squad",
        "arm": "semantic_search",
        "question_type": "squad",
        "question": "Where?",
        "passages": [{"id": "p", "title": "", "text": f"{answerable_text} is the capital."}],
        "unanswerable": unanswerable,
        "source_retrieved": source_retrieved,
    }


class _Reply:
    def __init__(self, text: str):
        self.text = text

    def converse(self, **_kwargs):
        return {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }


class _Validation:
    def converse(self, **_kwargs):
        exc = RuntimeError("bad request")
        exc.response = {"Error": {"Code": "ValidationException", "Message": "bad request"}}
        raise exc


def _reprice():
    path = ROOT / "scripts" / "experiments" / "reprice_ledger.py"
    spec = importlib.util.spec_from_file_location("exp_script_reprice_ledger_v2", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_abstention_is_correct_only_when_the_source_was_retrieved_and_unanswerable():
    quiet = abstention_outcome(abstained=False, unanswerable=True, source_retrieved=True)
    assert quiet == {"abstention_label": None, "abstention_counts_correct": None}

    correct = abstention_outcome(abstained=True, unanswerable=True, source_retrieved=True)
    assert correct["abstention_label"] == ABSTAIN_CORRECT
    assert correct["abstention_counts_correct"] is True

    miss = abstention_outcome(abstained=True, unanswerable=True, source_retrieved=False)
    assert miss["abstention_label"] == ABSTAIN_RETRIEVAL_MISS
    assert miss["abstention_counts_correct"] is False

    # A missing source is a retrieval miss even when the question is answerable.
    miss_answerable = abstention_outcome(abstained=True, unanswerable=False, source_retrieved=False)
    assert miss_answerable["abstention_label"] == ABSTAIN_RETRIEVAL_MISS
    assert miss_answerable["abstention_counts_correct"] is False

    wrong = abstention_outcome(abstained=True, unanswerable=False, source_retrieved=True)
    assert wrong["abstention_label"] == ABSTAIN_INCORRECT
    assert wrong["abstention_counts_correct"] is False


def test_gold_in_top_k_is_any_source_and_source_retrieved_is_every_source():
    sources = ["a", "b"]
    assert gold_in_top_k(sources, ["a", "x"]) is True
    assert source_was_retrieved(sources, ["a", "x"]) is False
    assert source_was_retrieved(sources, ["b", "a", "c"]) is True
    label = retrieval_label(sources, ["a"])
    assert label == {
        "n_source_passages": 2,
        "n_source_in_top_k": 1,
        "gold_in_top_k": True,
        "source_retrieved": False,
    }
    full = retrieval_label(["only"], ["only", "other"])
    assert full["gold_in_top_k"] is True and full["source_retrieved"] is True
    with pytest.raises(ProtocolError, match="non-empty"):
        gold_in_top_k([], ["a"])


def test_shared_collections_mark_source_passages():
    squad_rows = [
        {"id": "q1", "title": "Paris", "context": "Paris is the capital.",
         "question": "What is the capital?", "answers": {"text": ["Paris"]}},
        {"id": "q2", "title": "Paris", "context": "Paris is the capital.",
         "question": "Who founded it?", "answers": {"text": []}},
        {"id": "q3", "title": "Paris", "context": "The river is the Seine.",
         "question": "Which river?", "answers": {"text": ["Seine"]}},
    ]
    questions, passages = squad_collection(squad_rows)
    by_id = {row["qid"]: row for row in questions}
    assert len(passages) == 2
    assert by_id["q1"]["question_type"] == "squad"
    assert by_id["q1"]["unanswerable"] is False
    assert by_id["q2"]["unanswerable"] is True and by_id["q2"]["gold_answers"] == []
    assert by_id["q1"]["source_passage_ids"] == by_id["q2"]["source_passage_ids"]
    assert by_id["q1"]["source_passage_ids"] != by_id["q3"]["source_passage_ids"]
    assert set(by_id["q1"]["source_passage_ids"]) <= {p["id"] for p in passages}

    hotpot_row = {
        "id": "h1", "type": "comparison", "question": "Are they the same?", "answer": "no",
        "context": {
            "title": ["Alpha", "Beta", "Gamma"],
            "sentences": [["Alpha text."], ["Beta text."], ["Gamma text."]],
        },
        "supporting_facts": {"title": ["Alpha", "Gamma", "Alpha"]},
    }
    hot_q, hot_p = hotpot_collection([hotpot_row])
    assert hot_q[0]["question_type"] == "hotpot"
    assert hot_q[0]["hotpot_type"] == "comparison"
    assert hot_q[0]["yes_no"] is True
    assert len(hot_q[0]["source_passage_ids"]) == 2
    assert len(hot_p) == 3
    assert set(hot_q[0]["source_passage_ids"]) <= {p["id"] for p in hot_p}
    with pytest.raises(ProtocolError, match="not in the context"):
        hotpot_collection([{
            **hotpot_row,
            "supporting_facts": {"title": ["Missing"]},
        }])

    context = "GOLDANSWER " + ("x" * 200)
    nq_q, nq_p, n_contexts = nq_collection([{
        "context": context,
        "qas": [{"qid": "n1", "question": "What marker?", "answers": ["GOLDANSWER"]}],
    }], 50, 10)
    assert n_contexts == 1
    assert len(nq_p) > 1
    assert nq_q[0]["question_type"] == "nq" and nq_q[0]["unanswerable"] is False
    source = set(nq_q[0]["source_passage_ids"])
    texts = {p["id"]: p["text"] for p in nq_p}
    assert source and source < set(texts)
    assert all("GOLDANSWER" in texts[pid] for pid in source)
    assert any("GOLDANSWER" not in text for text in texts.values())
    with pytest.raises(ProtocolError, match="no gold answer"):
        nq_collection([{"context": "Paris.", "qas": [{"qid": "n2", "question": "?", "answers": []}]}], 80, 10)
    with pytest.raises(ProtocolError, match="no window contains"):
        nq_collection([{
            "context": "The capital is Paris.",
            "qas": [{"qid": "n3", "question": "?", "answers": ["Berlin"]}],
        }], 80, 10)


def test_pagerank_uses_inverse_df_and_falls_back_without_seeds():
    passages = [
        {"id": "p1", "text": "Ada wrote the notes on the engine."},
        {"id": "p2", "text": "Ada met Babbage in London."},
        {"id": "p3", "text": "The river flooded the plain."},
    ]
    graph = build_graph(passages, _caps)
    assert graph["edge_weight"] == "1/df"
    assert graph["df"]["ada"] == 2
    assert graph["adj"]["p:p1"]["e:ada"] == pytest.approx(0.5)
    assert graph["adj"]["p:p2"]["e:ada"] == pytest.approx(0.5)
    assert graph["adj"]["p:p2"]["e:babbage"] == pytest.approx(1.0)
    scores = graph_scores("What did Ada write", passages, _caps)
    assert scores[0] > scores[2]
    assert scores[1] > scores[2]
    assert graph_scores("What did Ada write", passages, _caps) == scores

    empty = lambda _text: []
    fallback, source = graph_or_fallback(
        "zzzz qq", passages, empty, [0.1, 0.4, 0.2],
        build_graph(passages, empty), tokenize,
        damping=0.85, max_iter=40, tol=1e-10,
    )
    assert source == "vector_fallback"
    assert fallback == [0.1, 0.4, 0.2]


def test_minmax_stores_the_raw_score_and_does_not_reorder(tmp_path):
    del tmp_path

    def embed(texts):
        out = []
        for i, text in enumerate(texts):
            if i == 0 or "alpha" in text:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    passages = [
        {"id": "b", "title": "", "text": "beta only"},
        {"id": "a", "title": "", "text": "alpha hit"},
    ]
    ranked, meta = rank_pool("alpha", passages, embed, 2, entity_fn=lambda _text: [])
    assert meta["score_normalization"] == "per_query_minmax"
    assert meta["graph_score_source"] == "vector_fallback"
    assert meta["pagerank"]["deterministic"] is True
    assert meta["pagerank"]["seed"] is None
    semantic = ranked["semantic_search"]
    assert [row["id"] for row in semantic] == ["a", "b"]
    assert semantic[0]["raw_score"] == pytest.approx(1.0)
    assert semantic[0]["score"] == pytest.approx(1.0)
    assert semantic[1]["raw_score"] == pytest.approx(0.0)
    assert semantic[1]["score"] == pytest.approx(0.0)


def test_missing_spacy_model_stops(monkeypatch):
    with pytest.raises(ProtocolError, match="spacy_model is empty"):
        load_spacy_entities("")
    try:
        import spacy
    except ImportError:
        with pytest.raises(ProtocolError, match="spacy is not installed"):
            load_spacy_entities("en_core_web_sm")
        return

    def boom(_name):
        raise OSError("model missing")

    monkeypatch.setattr(spacy, "load", boom)
    with pytest.raises(ProtocolError, match="not installed"):
        load_spacy_entities("en_core_web_sm")


def test_batch_jsonl_uses_each_model_invoke_body():
    haiku = model_input("anthropic", "sys", "user text", 256, 0.0)
    assert haiku["anthropic_version"] == "bedrock-2023-05-31"
    assert haiku["max_tokens"] == 256
    assert haiku["messages"][0]["content"][0] == {"type": "text", "text": "user text"}
    llama = model_input("llama", "sys", "user text", 256, 0.0)
    assert llama["prompt"].startswith("<|begin_of_text|>")
    assert "sys" in llama["prompt"] and llama["max_gen_len"] == 256
    nova = model_input("nova", "sys", "user text", 256, 0.0)
    assert nova["schemaVersion"] == "messages-v1"
    assert nova["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.0}
    assert nova["system"] == [{"text": "sys"}]
    payload = build_jsonl([
        {"recordId": "bbb", "modelInput": haiku, "meta": {"drop": True}},
        {"recordId": "aaa", "modelInput": llama},
    ])
    lines = [json.loads(line) for line in payload.splitlines()]
    assert [line["recordId"] for line in lines] == ["aaa", "bbb"]
    assert all(set(line) == {"recordId", "modelInput"} for line in lines)
    assert model_family("us.anthropic.claude-haiku-4-5-20251001-v1:0") == "anthropic"
    assert model_family("us.meta.llama3-3-70b-instruct-v1:0") == "llama"
    assert model_family("us.amazon.nova-pro-v1:0") == "nova"
    with pytest.raises(ProtocolError, match="no batch modelInput"):
        model_family("openai.gpt-oss-120b-1:0")


def test_batch_output_parsers_and_missing_usage():
    anthropic = parsed_output_rows("anthropic", [{
        "recordId": "a",
        "modelOutput": {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "stop_reason": "end_turn",
        },
    }])
    assert anthropic["a"]["text"] == "hello"
    assert anthropic["a"]["usage_observed"] is True
    llama = parsed_output_rows("llama", [{
        "recordId": "l",
        "modelOutput": {
            "generation": "yo",
            "prompt_token_count": 3,
            "generation_token_count": 1,
            "stop_reason": "stop",
        },
    }])
    assert llama["l"]["input_tokens"] == 3 and llama["l"]["output_tokens"] == 1
    nova = parsed_output_rows("nova", [{
        "recordId": "n",
        "modelOutput": {
            "output": {"message": {"content": [{"text": "nova"}]}},
            "usage": {"inputTokens": 7, "outputTokens": 2},
            "stopReason": "end_turn",
        },
    }])
    assert nova["n"]["text"] == "nova" and nova["n"]["stop_reason"] == "end_turn"
    failed = parsed_output_rows("anthropic", [{
        "recordId": "e",
        "error": {"errorCode": "ValidationException", "errorMessage": "bad"},
    }])
    assert failed["e"]["usage_observed"] is False
    assert failed["e"]["input_tokens"] == 0 and failed["e"]["output_tokens"] == 0
    with pytest.raises(ProtocolError, match="no token counts"):
        parsed_output_rows("anthropic", [{
            "recordId": "z",
            "modelOutput": {"content": [{"text": "hi"}]},
        }])


def test_missing_batch_role_or_bucket_stops(monkeypatch):
    monkeypatch.delenv("BEDROCK_BATCH_ROLE_ARN", raising=False)
    monkeypatch.delenv("BEDROCK_BATCH_BUCKET", raising=False)
    with pytest.raises(ProtocolError, match="role ARN"):
        resolve_batch_location(load_config())
    monkeypatch.setenv("BEDROCK_BATCH_ROLE_ARN", _ROLE)
    monkeypatch.setenv("BEDROCK_BATCH_BUCKET", _BUCKET)
    location = resolve_batch_location(load_config())
    assert location["role_arn"] == _ROLE
    assert location["bucket"] == _BUCKET
    assert "239571291755" not in location["role_arn"]


def test_batch_below_min_records_does_not_submit(tmp_path):
    cfg = _batch_cfg()
    cfg["batch"]["min_records"] = 100

    class Bedrock:
        def create_model_invocation_job(self, **_kwargs):
            raise AssertionError("create_model_invocation_job must not be called")

    class S3:
        def put_object(self, **_kwargs):
            raise AssertionError("the job must not be uploaded")

    budget = Budget(tmp_path, "generate", max_usd=10, total_usd_cap=30)
    summary = generate_rows_batch(
        cfg, [_retrieval("q1", unanswerable=False, source_retrieved=True)],
        tmp_path / "g.jsonl", budget,
        model_id=cfg["generator_model_id"], s3=S3(), bedrock=Bedrock(), seed=0,
        sleep=lambda _s: None,
    )
    assert summary["stopped"] is True
    assert summary["written"] == 0
    assert "min_records is 100" in summary["stop_reason"]
    assert "--inference-mode on_demand" in summary["stop_reason"]


def test_batch_cap_blocks_submission(tmp_path):
    cfg = _batch_cfg()

    class Bedrock:
        def create_model_invocation_job(self, **_kwargs):
            raise AssertionError("create_model_invocation_job must not be called")

    class S3:
        def put_object(self, **_kwargs):
            raise AssertionError("the job must not be uploaded")

    budget = Budget(tmp_path, "generate", max_usd=1e-6, total_usd_cap=30)
    summary = generate_rows_batch(
        cfg, [_retrieval("q1", unanswerable=False, source_retrieved=True)],
        tmp_path / "g.jsonl", budget,
        model_id=cfg["generator_model_id"], s3=S3(), bedrock=Bedrock(), seed=0,
        sleep=lambda _s: None,
    )
    assert summary["stopped"] is True
    assert summary["written"] == 0
    assert "spend cap reached" in summary["stop_reason"]


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
        self.creates = 0
        self.listed = 0

    def create_model_invocation_job(self, **kwargs):
        self.creates += 1
        assert kwargs["modelInvocationType"] == "InvokeModel"
        assert kwargs["clientRequestToken"] == kwargs["jobName"]
        assert kwargs["inputDataConfig"]["s3InputDataConfig"]["s3InputFormat"] == "JSONL"
        assert kwargs["roleArn"] == _ROLE
        assert "239571291755" not in json.dumps(kwargs)
        return {"jobArn": _JOB_ARN}

    def get_model_invocation_job(self, jobIdentifier):
        assert jobIdentifier == _JOB_ARN
        self._write_output()
        return {"status": "Completed"}

    def _write_output(self):
        inputs = [key for (_bucket, key) in self.store.objects if key.endswith("/input.jsonl")]
        assert len(inputs) == 1
        raw = self.store.objects[(_BUCKET, inputs[0])].decode("utf-8")
        lines = []
        for line in raw.splitlines():
            row = json.loads(line)
            lines.append(json.dumps({
                "recordId": row["recordId"],
                "modelOutput": {
                    "content": [{
                        "type": "text",
                        "text": '{"answer": "Paris", "claim": "Paris is the capital.", "confidence": 0.4}',
                    }],
                    "usage": {"input_tokens": 20, "output_tokens": 8},
                    "stop_reason": "end_turn",
                },
            }))
        out_key = inputs[0].replace("/input.jsonl", "/out/output.jsonl.out")
        self.store.objects[(_BUCKET, out_key)] = ("\n".join(lines) + "\n").encode("utf-8")


def test_batch_job_resumes_from_the_stored_arn_without_creating_another(tmp_path):
    cfg = _batch_cfg()
    store = _Store()
    s3 = _S3(store)
    bedrock = _Bedrock(store)
    out = tmp_path / "g.jsonl"
    budget = Budget(tmp_path, "generate", max_usd=10, total_usd_cap=30)
    row = _retrieval("q1", unanswerable=False, source_retrieved=True)
    first = generate_rows_batch(
        cfg, [row], out, budget, model_id=cfg["generator_model_id"],
        s3=s3, bedrock=bedrock, seed=0, sleep=lambda _s: None,
    )
    assert first["written"] == 1 and first["stopped"] is False
    assert bedrock.creates == 1
    assert s3.puts == 1
    written = json.loads(out.read_text().splitlines()[0])
    assert written["claim"] == "Paris is the capital."
    assert written["schema_version"] == 2
    assert written["pricing"] == "batch"
    assert written["price_usd_per_million"] == {"input": 0.55, "output": 2.75}
    assert written["abstention_label"] is None
    expected = cost_usd(cfg, cfg["generator_model_id"], 20, 8, "batch")
    assert written["usd"] == pytest.approx(expected)
    out.unlink()
    second_budget = Budget(tmp_path, "generate", max_usd=10, total_usd_cap=30)
    second = generate_rows_batch(
        cfg, [row], out, second_budget, model_id=cfg["generator_model_id"],
        s3=s3, bedrock=bedrock, seed=0, sleep=lambda _s: None,
    )
    assert second["written"] == 1
    assert bedrock.creates == 1
    assert s3.puts == 1
    again = json.loads(out.read_text().splitlines()[0])
    assert again["claim"] == written["claim"]
    assert again["usd"] == pytest.approx(expected)


def test_conflict_exception_resumes_the_named_job():
    class Conflict(Exception):
        response = {"Error": {"Code": "ConflictException", "Message": "exists"}}

    class Bedrock:
        def __init__(self):
            self.listed = 0

        def create_model_invocation_job(self, **_kwargs):
            raise Conflict()

        def list_model_invocation_jobs(self, nameContains, maxResults):
            self.listed += 1
            assert nameContains == "cwjob"
            assert maxResults == 100
            return {"invocationJobSummaries": [{
                "jobName": "cwjob",
                "jobArn": _JOB_ARN,
            }, {
                "jobName": "cwjob-other",
                "jobArn": "arn:aws:bedrock:us-east-1:example:model-invocation-job/other",
            }]}

    bedrock = Bedrock()
    arn = create_or_find_job(
        bedrock, job_name="cwjob", model_id="us.amazon.nova-pro-v1:0",
        role_arn=_ROLE, input_uri=f"s3://{_BUCKET}/in", output_uri=f"s3://{_BUCKET}/out",
        timeout_hours=1,
    )
    assert arn == _JOB_ARN
    assert bedrock.listed == 1


def test_generation_stores_claim_and_abstention_labels(tmp_path):
    cfg = load_config()
    cases = [
        ("q-miss", True, False, '{"answer": "insufficient evidence", "claim": "The passages do not contain the answer.", "confidence": 0.2}',
         ABSTAIN_RETRIEVAL_MISS, False),
        ("q-ok", True, True, '{"answer": "insufficient evidence", "claim": "The passages do not contain the answer.", "confidence": 0.2}',
         ABSTAIN_CORRECT, True),
        ("q-bad", False, True, '{"answer": "insufficient evidence", "claim": "The passages do not contain the answer.", "confidence": 0.2}',
         ABSTAIN_INCORRECT, False),
        ("q-span", False, True, '{"answer": "Paris", "confidence": 0.4}', None, None),
    ]
    for qid, unanswerable, retrieved, text, label, counts in cases:
        budget = Budget(tmp_path / qid, "generate", max_usd=10, total_usd_cap=30)
        summary = generate_rows(
            cfg, [_retrieval(qid, unanswerable=unanswerable, source_retrieved=retrieved)],
            tmp_path / qid / "g.jsonl", budget,
            model_id=cfg["generator_model_id"], client=_Reply(text), sleep=lambda _s: None,
        )
        assert summary["written"] == 1
        row = json.loads((tmp_path / qid / "g.jsonl").read_text().splitlines()[0])
        assert row["schema_version"] == 2
        assert row["pricing"] == "on_demand"
        assert row["abstention_label"] == label
        assert row["abstention_counts_correct"] is counts
        if qid == "q-span":
            assert row["claim"] is None
            assert row["answer"] == "Paris"
        else:
            assert row["claim"] == "The passages do not contain the answer."

    failed_budget = Budget(tmp_path / "failed", "generate", max_usd=10, total_usd_cap=30)
    generate_rows(
        cfg, [_retrieval("q-fail", unanswerable=True, source_retrieved=True)],
        tmp_path / "failed" / "g.jsonl", failed_budget,
        model_id=cfg["generator_model_id"], client=_Validation(), sleep=lambda _s: None,
    )
    failed = json.loads((tmp_path / "failed" / "g.jsonl").read_text().splitlines()[0])
    assert failed["abstention_label"] is None
    assert failed["abstention_counts_correct"] is None
    assert failed["usage_observed"] is False


def test_generation_file_without_schema_version_2_is_refused(tmp_path):
    cfg = load_config()
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"qid": "old", "arm": "semantic_search", "answer": "x"}) + "\n")
    budget = Budget(tmp_path, "generate", max_usd=10, total_usd_cap=30)
    with pytest.raises(ProtocolError, match="schema_version 2"):
        generate_rows(
            cfg, [_retrieval("q1", unanswerable=False, source_retrieved=True)],
            path, budget, model_id=cfg["generator_model_id"], client=_Reply("{}"),
            sleep=lambda _s: None,
        )


def test_ledger_prices_batch_rows_from_the_batch_table():
    cfg = load_config()
    model = cfg["generator_model_id"]
    on_demand = cost_usd(cfg, model, 1_000_000, 1_000_000, "on_demand")
    batch = cost_usd(cfg, model, 1_000_000, 1_000_000, "batch")
    assert on_demand == pytest.approx(1.10 + 5.50)
    assert batch == pytest.approx(0.55 + 2.75)
    assert price_for(cfg, model, "batch")["input"] == pytest.approx(0.55)
    mod = _reprice()
    updated, old_total, new_total = mod.reprice_rows(cfg, [
        {
            "model_id": model, "pricing": "batch", "input_tokens": 1_000_000,
            "output_tokens": 1_000_000, "usd": 1.0,
        },
        {
            "model_id": model, "input_tokens": 1_000_000, "output_tokens": 1_000_000, "usd": 2.0,
        },
    ], "2026-10-08T00:00:00+00:00", "ledger")
    assert old_total == pytest.approx(3.0)
    assert new_total == pytest.approx(batch + on_demand)
    assert updated[0]["pricing"] == "batch"
    assert updated[0]["usd"] == pytest.approx(batch)
    assert updated[1]["pricing"] == "on_demand"
    assert updated[1]["usd"] == pytest.approx(on_demand)
