"""Remote MCP server: Streamable HTTP, protocol 2025-06-18.

The query path under test is the real query handler. Retrieval is replaced
with a fixed passage so the tool call does not need Postgres or Bedrock.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

TOKEN = "test-token-value"
PASSAGE = "Retrieved passage about AWS Lambda and API Gateway."

_STUBS = (
    "boto3",
    "botocore",
    "botocore.exceptions",
    "botocore.config",
    "mcp_observatory",
    "mcp_observatory.instrument",
)


class CfnLoader(yaml.SafeLoader):
    pass


def _keep(loader, suffix, node):
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


def _install_stubs(monkeypatch):
    for name in _STUBS:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(
        sys.modules["botocore.exceptions"], "ClientError",
        type("ClientError", (Exception,), {}), raising=False,
    )
    monkeypatch.setattr(
        sys.modules["botocore.config"], "Config",
        type("Config", (), {"__init__": lambda self, **kw: None}), raising=False,
    )
    monkeypatch.setattr(sys.modules["boto3"], "client", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(sys.modules["boto3"], "resource", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(
        sys.modules["mcp_observatory.instrument"], "instrument_wrapper_api",
        lambda *a, **kw: None, raising=False,
    )


@pytest.fixture
def mcp(monkeypatch):
    before = set(sys.modules)
    _install_stubs(monkeypatch)
    import importlib
    import mcp_server.auth as auth
    import mcp_server.handler as handler
    importlib.reload(auth)
    importlib.reload(handler)
    monkeypatch.setattr(handler.auth, "read_configured_token", lambda: TOKEN)
    monkeypatch.delenv("HEALTH_POSTGRES_SECRET_ARN", raising=False)
    yield handler
    for name in list(sys.modules):
        if name not in before and not name.startswith("mcp_observatory") and name not in _STUBS:
            # Leave stubbed AWS modules for monkeypatch to restore.
            if name.split(".", 1)[0] in {"mcp_server", "query_api", "shared"} or name in {
                "agent_card", "graph_expander", "rag_router", "retriever", "synthesizer",
                "models", "cache", "feedback", "routing_decisions_api", "verified_reward",
            }:
                sys.modules.pop(name, None)


def _event(method: str, body: dict | None = None, token: str | None = TOKEN, headers: dict | None = None) -> dict:
    hdrs = {"accept": "application/json, text/event-stream"}
    if token is not None:
        hdrs["authorization"] = f"Bearer {token}"
    if headers:
        hdrs.update(headers)
    event = {
        "headers": hdrs,
        "requestContext": {
            "http": {"method": method},
            "domainName": "example.execute-api.us-east-1.amazonaws.com",
            "stage": "prod",
        },
        "rawPath": "/mcp",
        "isBase64Encoded": False,
    }
    if body is not None:
        event["body"] = json.dumps(body)
    return event


def _rpc(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    message = {"jsonrpc": "2.0", "method": method}
    if req_id is not None:
        message["id"] = req_id
    if params is not None:
        message["params"] = params
    return message


def test_initialize_advertises_the_streamable_http_protocol(mcp):
    response = mcp.lambda_handler(_event("POST", _rpc("initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    }), headers={"authorization": f"bearer {TOKEN}"}), None)

    assert response["statusCode"] == 200
    assert response["headers"]["MCP-Protocol-Version"] == "2025-06-18"
    body = json.loads(response["body"])
    assert body["id"] == 1
    result = body["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["capabilities"]["tools"]["listChanged"] is False
    assert result["serverInfo"]["name"] == "ContextWeave"
    assert result["serverInfo"]["version"]


def test_tools_list_schemas_and_delete_destructive_hint(mcp, monkeypatch):
    response = mcp.lambda_handler(_event("POST", _rpc("tools/list", {})), None)
    assert response["statusCode"] == 200
    tools = json.loads(response["body"])["result"]["tools"]
    by_name = {tool["name"]: tool for tool in tools}

    assert set(by_name) == {
        "query_expertise",
        "submit_feedback",
        "list_routing_decisions",
        "health",
        "get_agent_card",
    }
    for tool in tools:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"
        assert "properties" in tool["inputSchema"]
        assert tool["annotations"]["destructiveHint"] is False

    assert by_name["query_expertise"]["inputSchema"]["required"] == ["question"]
    assert "question" in by_name["query_expertise"]["inputSchema"]["properties"]

    monkeypatch.setenv("HEALTH_POSTGRES_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:1:secret:health")
    response = mcp.lambda_handler(_event("POST", _rpc("tools/list", {})), None)
    tools = json.loads(response["body"])["result"]["tools"]
    by_name = {tool["name"]: tool for tool in tools}
    assert "health_query" in by_name
    assert "list_health_documents" in by_name
    delete = by_name["delete_health_document"]
    assert delete["annotations"]["destructiveHint"] is True
    assert delete["inputSchema"]["required"] == ["docId"]
    assert delete["inputSchema"]["properties"]["docId"]["type"] == "string"


def test_tools_call_query_expertise_uses_mocked_retrieval(mcp, monkeypatch):
    import importlib
    from types import SimpleNamespace

    query_handler = importlib.import_module("query_api.handler")
    from query_api.models import QueryResponse, RetrievedChunk

    seen = {}

    def retrieve(**kwargs):
        seen["retrieve"] = kwargs
        return [RetrievedChunk(content=PASSAGE, score=0.91, source_uri="s3://bucket/architecture.md")]

    def synthesize(**kwargs):
        seen["chunks"] = kwargs["chunks"]
        return QueryResponse(
            answer="From the passage: " + kwargs["chunks"][0].content,
            sources=[{"uri": "s3://bucket/architecture.md"}],
            inferred_skills=["AWS Lambda"],
            repeated_patterns=[],
            confidence=0.9,
            confidence_reported=True,
            question_type="architecture",
            graph_entities_used=[],
            retrieval_count=1,
            model_id="test-model",
        )

    monkeypatch.setattr(query_handler, "NEPTUNE_GRAPH_ID", "")
    monkeypatch.setattr(query_handler._cache, "is_time_sensitive", lambda question: True)
    monkeypatch.setattr(query_handler, "classify_question", lambda question: "architecture")
    monkeypatch.setattr(query_handler, "select_strategy", lambda **kwargs: SimpleNamespace(
        strategy="semantic_search",
        include_graph=False,
        boost_keywords=False,
        use_neptune_chunks=False,
        strategy_confidence=0.7,
        selection_propensity=0.5,
    ))
    monkeypatch.setattr(query_handler, "retrieve_with_strategy", retrieve)
    monkeypatch.setattr(query_handler, "synthesize_answer", synthesize)
    monkeypatch.setattr(query_handler._feedback, "record_decision", lambda *a, **kw: None)
    monkeypatch.delenv("ROUTER_JUDGE_MODEL_ID", raising=False)

    response = mcp.lambda_handler(_event("POST", _rpc("tools/call", {
        "name": "query_expertise",
        "arguments": {"question": "What has this developer shipped on Lambda?"},
    })), None)

    assert response["statusCode"] == 200
    result = json.loads(response["body"])["result"]
    assert result["isError"] is False
    assert seen["retrieve"]["question"] == "What has this developer shipped on Lambda?"
    assert seen["chunks"][0].content == PASSAGE
    assert PASSAGE in result["content"][0]["text"]
    assert result["structuredContent"]["answer"].startswith("From the passage:")
    assert result["structuredContent"]["inferredSkills"] == ["AWS Lambda"]


def test_auth_rejection_is_401_and_does_not_distinguish_the_failure(mcp):
    missing = mcp.lambda_handler(_event("POST", _rpc("initialize", {}), token=None), None)
    wrong = mcp.lambda_handler(_event("POST", _rpc("initialize", {}), token="not-the-token"), None)
    basic = mcp.lambda_handler(_event(
        "POST", _rpc("initialize", {}), token=None,
        headers={"authorization": "Basic dXNlcjpwYXNz"},
    ), None)
    get_missing = mcp.lambda_handler(_event("GET", token=None), None)

    for response in (missing, wrong, basic, get_missing):
        assert response["statusCode"] == 401
        assert response["headers"]["WWW-Authenticate"].startswith("Bearer ")
        body = json.loads(response["body"])
        assert body["error"]["message"] == "Unauthorized"
        assert TOKEN not in response["body"]
        assert "not-the-token" not in response["body"]
    assert missing["body"] == wrong["body"] == basic["body"]


def test_get_returns_405_when_the_token_is_valid(mcp):
    response = mcp.lambda_handler(_event("GET"), None)
    assert response["statusCode"] == 405
    assert response["headers"]["Allow"] == "POST"
    body = json.loads(response["body"])
    assert body["error"]["code"] == -32601


def test_token_compare_accepts_unequal_lengths_and_secret_shapes():
    from mcp_server import auth

    assert auth.tokens_equal("same", "same")
    assert not auth.tokens_equal("short", "a-much-longer-token")
    assert auth.token_from_secret_string("  raw-token  ") == "raw-token"
    assert auth.token_from_secret_string('{"token": "from-json"}') == "from-json"
    assert auth.token_from_secret_string('{"password": "nope"}') is None

    auth.clear_cache()
    os.environ["MCP_AUTH_TOKEN_SECRET_NAME"] = "/contextweave/prod/mcp-bearer-token"

    class FakeSsm:
        def __init__(self):
            self.calls = []

        def get_parameter(self, Name, WithDecryption):
            self.calls.append((Name, WithDecryption))
            return {"Parameter": {"Value": "ssm-token"}}

    ssm = FakeSsm()
    try:
        assert auth.read_configured_token(ssm_client=ssm, now=1.0) == "ssm-token"
        assert ssm.calls == [("/contextweave/prod/mcp-bearer-token", True)]
        # Served from the short-lived cache; the client is not called again.
        assert auth.read_configured_token(ssm_client=FakeSsm(), now=1.0) == "ssm-token"
    finally:
        os.environ.pop("MCP_AUTH_TOKEN_SECRET_NAME", None)
        auth.clear_cache()

    class FakeSecrets:
        def get_secret_value(self, SecretId):
            assert SecretId == "contextweave/prod/mcp-bearer-token"
            return {"SecretString": '{"bearerToken": "sm-token"}'}

    os.environ["MCP_AUTH_TOKEN_SECRET_NAME"] = "contextweave/prod/mcp-bearer-token"
    try:
        assert auth.read_configured_token(secrets_client=FakeSecrets(), now=10.0) == "sm-token"
    finally:
        os.environ.pop("MCP_AUTH_TOKEN_SECRET_NAME", None)
        auth.clear_cache()


def test_template_mcp_routes_leave_the_health_authorizer_in_place():
    """POST /mcp and GET /mcp are on the MCP function and do not take the health authorizer.

    There is no DefaultAuthorizer, so `Authorizer: NONE` is not a legal opt-out
    (the transform rejects it). Omitting Auth leaves the route open at the
    gateway; the function checks the bearer token. The custom authorizer stays
    on every /health/* route.
    """
    events = RESOURCES["McpServerFunction"]["Properties"]["Events"]
    assert {events["McpPost"]["Properties"]["Method"], events["McpGet"]["Properties"]["Method"]} == {"POST", "GET"}
    for props in (events["McpPost"]["Properties"], events["McpGet"]["Properties"]):
        assert props["Path"] == "/mcp"
        assert props["ApiId"]["fn"] == "Ref" and props["ApiId"]["arg"] == "ExpertiseAPI"
        assert "Auth" not in props

    api_auth = RESOURCES["ExpertiseAPI"]["Properties"].get("Auth") or {}
    assert "DefaultAuthorizer" not in api_auth
    assert "SiweAuthorizer" in api_auth.get("Authorizers", {})

    health_events = RESOURCES["HealthAPIFunction"]["Properties"]["Events"]
    assert health_events
    for props in (cfg["Properties"] for cfg in health_events.values()):
        assert props["Path"].startswith("/health/")
        assert props["Auth"]["Authorizer"] == "SiweAuthorizer"

    env = RESOURCES["McpServerFunction"]["Properties"]["Environment"]["Variables"]
    secret = env["HEALTH_POSTGRES_SECRET_ARN"]
    assert secret["fn"] == "If" and secret["arg"][0] == "HealthEnabled"
    assert "HealthPostgresSecret" in json.dumps(secret["arg"][1])

    assert TEMPLATE["Parameters"]["McpAuthTokenSecretName"]["Default"] == ""
    assert "VpcConfig" in RESOURCES["McpServerFunction"]["Properties"]
    assert TEMPLATE["Outputs"]["McpEndpoint"]["Value"]

    role_text = json.dumps(RESOURCES["McpServerRole"])
    assert "HealthPostgresSecret" in role_text
    assert "HealthEnabled" in role_text
    # Secrets Manager appends a hyphen and six random characters to the name.
    # The grant has to be the wildcard form or GetSecretValue never matches.
    assert "secretsmanager:GetSecretValue" in role_text
    assert "secret:${McpAuthTokenSecretName}-*" in role_text
    query_role = json.dumps(RESOURCES["QueryAPIRole"])
    shared_secrets = json.dumps(RESOURCES["SecretsReadPolicy"])
    assert "HealthPostgresSecret" not in query_role
    assert "HealthPostgresSecret" not in shared_secrets
