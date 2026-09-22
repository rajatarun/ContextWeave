"""The health store is isolated, authenticated, deletable and uncached.

Four properties, each answering a specific way the expertise corpus is the
wrong place for a medical record. They are asserted rather than described,
because every one of them is invisible from the outside once it is wrong: a
health route that lost its authorizer still answers, a health query that read
the expertise table still returns chunks, and an answer written to the cache
still looks identical to one that was not.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.health_api import ingest  # noqa: E402
from src.health_api import handler as health_handler  # noqa: E402
from src.shared import health_db  # noqa: E402


class CfnLoader(yaml.SafeLoader):
    pass


def _keep(loader, suffix, node):
    """Keep the tag's argument. A loader that discards it turns
    `!Not [!Equals [!Ref SiweAuthorizerFunctionArn, '']]` into `{"fn": "Not"}`,
    and every assertion about what a condition depends on passes vacuously."""
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"fn": suffix, "arg": value}


CfnLoader.add_multi_constructor("!", _keep)
TEMPLATE = yaml.load((REPO / "template.yaml").read_text(), Loader=CfnLoader)
RESOURCES = TEMPLATE["Resources"]


# ── 1. authenticated, on an API where nothing else is ───────────────────────

def health_routes():
    events = RESOURCES["HealthAPIFunction"]["Properties"]["Events"]
    return {name: cfg["Properties"] for name, cfg in events.items()}


def test_every_health_route_carries_an_authorizer():
    routes = health_routes()
    assert routes, "no health routes at all"
    for name, props in routes.items():
        authorizer = (props.get("Auth") or {}).get("Authorizer")
        assert authorizer == "SiweAuthorizer", (
            f"{name} ({props.get('Method')} {props.get('Path')}) is open. Every "
            f"expertise route on this API is open too, which is why this one "
            f"cannot be."
        )


def test_the_expertise_routes_are_left_open_on_purpose():
    """Not an oversight, and not something to 'fix' by adding a default.

    The A2A card must be readable without a token or discovery cannot work, and
    TeamWeave calls /query-expertise with none. A DefaultAuthorizer here would
    break both -- which is the mistake TeamWeave's own API made, serving 401 on
    the card that exists to tell clients how to authenticate.
    """
    api = RESOURCES["ExpertiseAPI"]["Properties"]
    assert "DefaultAuthorizer" not in (api.get("Auth") or {}), (
        "a default authorizer would cover the A2A card and the expertise query"
    )
    assert "SiweAuthorizer" in (api.get("Auth") or {}).get("Authorizers", {})


def test_the_health_surface_does_not_exist_without_an_authorizer():
    """The gate that makes the above true by construction: no authorizer
    configured, no health resources at all. A health endpoint deployed open by
    accident is the failure this prevents."""
    for name in ("HealthAPIFunction", "HealthIngestFunction", "HealthPostgresSecret"):
        assert RESOURCES[name].get("Condition") == "HealthEnabled", name
    condition = TEMPLATE["Conditions"]["HealthEnabled"]
    assert "SiweAuthorizerFunctionArn" in json.dumps(condition)


# ── 2. a different store, not a different prefix ────────────────────────────

def test_the_health_store_refuses_to_fall_back_to_the_expertise_database(monkeypatch):
    """The single most dangerous line that could be written here is a default
    connection. It would put medical records in the corpus the LinkedIn writer
    reads, and nothing downstream would notice."""
    monkeypatch.setattr(health_db, "_conn", None)
    monkeypatch.delenv("HEALTH_POSTGRES_SECRET_ARN", raising=False)
    with pytest.raises(RuntimeError) as raised:
        health_db.get_connection()
    assert "no fallback" in str(raised.value)


def test_the_health_store_never_touches_the_expertise_table():
    source = (REPO / "src" / "shared" / "health_db.py").read_text()
    statements = [line for line in source.splitlines()
                  if "FROM " in line or "INSERT INTO" in line or "DELETE FROM" in line]
    assert statements
    for line in statements:
        assert "chunks" not in line or health_db.HEALTH_CHUNKS_TABLE in line, (
            f"health SQL touches the expertise chunks table: {line.strip()}"
        )


def test_the_health_store_has_its_own_credential():
    """Its own database and role, so the isolation is a grant rather than a
    naming convention."""
    for name in ("HealthAPIFunction", "HealthIngestFunction"):
        env = RESOURCES[name]["Properties"]["Environment"]["Variables"]
        assert "HEALTH_POSTGRES_SECRET_ARN" in env
        assert "POSTGRES_SECRET_ARN" not in env, (
            f"{name} also has the expertise credential, so one mis-set variable "
            f"reaches the wrong database"
        )


def test_health_documents_never_reach_the_knowledge_graph():
    """Memgraph and Neptune are shared and have no per-document delete, so a
    graph write is the one thing the delete path could not take back."""
    for module in ("src/health_api/ingest.py", "src/health_api/handler.py",
                   "src/shared/health_db.py"):
        source = (REPO / module).read_text()
        for forbidden in ("run_graph_query", "neptune", "memgraph", "get_memgraph_driver"):
            assert forbidden not in source.lower().replace("memgraph and neptune", ""), (
                f"{module} reaches the shared graph: {forbidden}"
            )


# ── 3. deletable, and the delete is confirmable ─────────────────────────────

class FakeCursor:
    def __init__(self, store): self.store, self.rowcount, self._rows = store, 0, []
    def __enter__(self): return self
    def __exit__(self, *_e): return False
    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("DELETE"):
            doc = params[0]
            before = len(self.store)
            self.store[:] = [r for r in self.store if r["doc_id"] != doc]
            self.rowcount = before - len(self.store)
        elif "INSERT" in sql:
            self.store.append({"doc_id": params[0], "source_key": params[1]})
    def fetchall(self): return self._rows


class FakeConn:
    def __init__(self): self.store, self.commits = [], 0
    def cursor(self): return FakeCursor(self.store)
    def commit(self): self.commits += 1


def test_deleting_a_document_removes_it_and_says_how_much(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(health_db, "get_connection", lambda: conn)
    health_db.replace_document("doc-1", "raw/a.txt", [("t", [0.0])])
    assert health_db.delete_document("doc-1") == 1
    # Nothing left, and the count distinguishes "removed" from "was not there".
    assert health_db.delete_document("doc-1") == 0


def test_the_api_answers_404_when_there_was_nothing_to_delete(monkeypatch):
    """Deleting a record you cannot confirm is gone is not deleting it."""
    monkeypatch.setattr(health_db, "delete_document", lambda _d: 0)
    response = health_handler.lambda_handler({
        "requestContext": {"http": {"method": "DELETE"}},
        "rawPath": "/health/documents/abc",
    })
    assert response["statusCode"] == 404
    assert json.loads(response["body"])["deleted"] == 0


def test_re_ingesting_replaces_rather_than_accumulates(monkeypatch):
    """A corrected record must not leave the superseded one retrievable."""
    conn = FakeConn()
    monkeypatch.setattr(health_db, "get_connection", lambda: conn)
    health_db.replace_document("doc-1", "raw/a.txt", [("old", [0.0])])
    health_db.replace_document("doc-1", "raw/a.txt", [("new", [0.0]), ("new2", [0.0])])
    assert len(conn.store) == 2


def test_removing_the_file_withdraws_the_record(monkeypatch):
    """Without this, deleting the object leaves it searchable."""
    seen = {}
    def fake_delete(doc_id):
        seen["doc"] = doc_id
        return 3

    monkeypatch.setattr(health_db, "delete_document", fake_delete)
    result = ingest.lambda_handler({
        "source": "aws.s3", "detail-type": "Object Deleted",
        "detail": {"bucket": {"name": "b"}, "object": {"key": "raw/rec.txt"}},
    })
    assert result["deleted"] == 3
    assert seen["doc"] == ingest.doc_id_for("raw/rec.txt")


def test_the_event_rule_listens_for_removal_as_well_as_creation():
    pattern = RESOURCES["HealthIngestRule"]["Properties"]["EventPattern"]
    assert set(pattern["detail-type"]) == {"Object Created", "Object Deleted"}


# ── 4. uncached, and it does not practise medicine ──────────────────────────

def test_no_answer_is_ever_cached():
    """The expertise path writes every answer to query_cache for seven days.
    An answer drawn from a medical record would outlive a delete of the record
    it came from."""
    # Parsed, not grepped: the module's own docstring explains *why* it does
    # not cache, so a substring search finds the word and fails on prose.
    import ast

    tree = ast.parse((REPO / "src" / "health_api" / "handler.py").read_text())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            called.add(fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", ""))
    assert not {"write_cache", "check_cache", "get_cached"} & called, called


def test_an_empty_retrieval_is_not_answered_from_general_knowledge(monkeypatch):
    """Letting the model answer anyway would present its recollection as the
    person's own chart."""
    monkeypatch.setattr(health_db, "search", lambda *_a, **_k: [])
    response = health_handler.query({"question": "what was my last reading"}, embed=lambda _q: [0.0])
    body = json.loads(response["body"])
    assert body["found"] is False
    assert body["answer"] == ""
    assert body["sources"] == []


def test_every_answer_carries_the_disclaimer(monkeypatch):
    monkeypatch.setattr(health_db, "search", lambda *_a, **_k: [])
    body = json.loads(health_handler.query({"question": "q"}, embed=lambda _q: [0.0])["body"])
    assert "not medical advice" in body["disclaimer"]


def test_the_prompt_forbids_diagnosis_and_dosage():
    prompt = health_handler.SYSTEM_PROMPT.lower()
    assert "never diagnose" in prompt
    assert "dose" in prompt
    assert "only from the excerpts" in prompt


def test_a_failure_never_echoes_the_database_error(monkeypatch):
    """A psycopg2 error can carry a row, and a row here is a medical record."""
    def boom(*_a, **_k):
        raise RuntimeError("duplicate key: patient Jane Doe, HbA1c 9.2")
    monkeypatch.setattr(health_db, "list_documents", boom)
    response = health_handler.lambda_handler({
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/health/documents",
    })
    assert response["statusCode"] == 500
    assert "Jane Doe" not in response["body"] and "9.2" not in response["body"]


def test_the_listing_returns_no_content(monkeypatch):
    monkeypatch.setattr(health_db, "list_documents",
                        lambda: [{"docId": "d", "sourceKey": "k", "chunks": 2, "ingestedAt": ""}])
    body = json.loads(health_handler.lambda_handler({
        "requestContext": {"http": {"method": "GET"}}, "rawPath": "/health/documents"})["body"])
    assert "content" not in json.dumps(body)


# ── ingest ──────────────────────────────────────────────────────────────────

def test_an_unreadable_type_is_refused_rather_than_half_read():
    """A PDF read as bytes would store mojibake and report success -- a record
    the person believes is searchable and is not."""
    result = ingest.ingest_object("b", "raw/scan.pdf")
    assert result["ingested"] == 0 and "unsupported" in result["reason"]


def test_a_document_whose_embeddings_all_failed_raises(monkeypatch):
    class S3:
        @staticmethod
        def get_object(Bucket, Key): return {"Body": __import__("io").BytesIO(b"some record")}

    with pytest.raises(RuntimeError) as raised:
        ingest.ingest_object("b", "raw/a.txt", s3=S3(), embed=lambda t: [None] * len(t))
    assert "nothing was stored" in str(raised.value)


def test_chunking_always_advances():
    assert len(ingest.chunk("x" * 3000)) > 1
    with pytest.raises(ValueError):
        ingest.chunk("abc", size=10, overlap=10)


def test_the_doc_id_is_stable_so_a_correction_replaces():
    assert ingest.doc_id_for("raw/a.txt") == ingest.doc_id_for("raw/a.txt")
    assert ingest.doc_id_for("raw/a.txt") != ingest.doc_id_for("raw/b.txt")


def test_the_bucket_is_encrypted_and_private():
    props = RESOURCES["HealthDocsBucket"]["Properties"]
    assert props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
        "ServerSideEncryptionByDefault"]["SSEAlgorithm"] == "aws:kms"
    for flag in props["PublicAccessBlockConfiguration"].values():
        assert flag is True
    assert RESOURCES["HealthDocsBucket"]["DeletionPolicy"] == "Retain"


def test_ingest_cannot_write_to_the_health_bucket():
    """Read-only: a write grant is a way for a bug to alter a record."""
    statements = RESOURCES["HealthIngestFunction"]["Properties"]["Policies"][-1]["Statement"]
    s3_actions = [a for st in statements for a in
                  (st["Action"] if isinstance(st["Action"], list) else [st["Action"]])
                  if a.startswith("s3:")]
    assert s3_actions == ["s3:GetObject"], s3_actions
