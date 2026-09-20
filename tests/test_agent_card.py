"""ContextWeave's A2A Agent Card.

TeamWeave reached this service by assuming `{CONTEXTWEAVE_URL}/query-expertise`
-- a path constant held in *TeamWeave's* repository. Move a route here and the
break surfaces there as a 404 its RAG layer degrades past in silence: the run
loses its grounding and nobody is told.

A card replaces that with a document the caller reads. What these pin is that
the document is true: the v1.0 shape rather than 0.3's, only capabilities
this service has, and a skill for every route a caller is expected to use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "query_api"))

import agent_card  # noqa: E402

EVENT = {"requestContext": {"domainName": "cw.example.com", "stage": "prod"}}


@pytest.fixture
def card(monkeypatch):
    monkeypatch.delenv("A2A_BASE_URL", raising=False)
    monkeypatch.delenv("A2A_AGENT_VERSION", raising=False)
    return agent_card.build_agent_card(EVENT)


def test_the_card_carries_the_required_fields(card):
    for field in ("name", "description", "version", "supportedInterfaces", "skills"):
        assert card.get(field), f"{field} is required on an AgentCard"


def test_the_v1_interface_shape_is_used_not_the_0_3_one(card):
    # v1.0 replaced the top-level url/preferredTransport with
    # supportedInterfaces. A 0.3-shaped card parses fine and tells a 1.0
    # client nothing.
    assert "url" not in card
    assert "preferredTransport" not in card
    interface = card["supportedInterfaces"][0]
    assert interface["protocolBinding"] == "HTTP+JSON"
    assert interface["protocolVersion"] == "1.0"


def test_the_url_is_derived_from_the_request(card):
    # Configured, it could advertise a URL this deployment does not serve --
    # the failure the card exists to prevent.
    assert card["supportedInterfaces"][0]["url"] == "https://cw.example.com/prod"


def test_an_explicit_base_url_wins(monkeypatch):
    monkeypatch.setenv("A2A_BASE_URL", "https://custom.example.com/")
    built = agent_card.build_agent_card(EVENT)
    assert built["supportedInterfaces"][0]["url"] == "https://custom.example.com"


def test_a_request_with_no_domain_yields_no_url(monkeypatch):
    monkeypatch.delenv("A2A_BASE_URL", raising=False)
    assert agent_card.build_agent_card({})["supportedInterfaces"][0]["url"] == ""


def test_capabilities_are_not_overclaimed(card):
    # This service implements neither A2A streaming nor push notifications.
    assert card["capabilities"]["streaming"] is False
    assert card["capabilities"]["pushNotifications"] is False


def test_every_route_a_caller_uses_is_a_skill(card):
    # A caller that cannot find query-expertise in the card is back to
    # guessing the path, which is the whole problem.
    assert {s["id"] for s in card["skills"]} >= {"query-expertise", "feedback", "routing-decisions"}


def test_each_skill_carries_the_four_required_fields(card):
    for skill in card["skills"]:
        for field in ("id", "name", "description", "tags"):
            assert skill.get(field), f"{skill.get('id')} is missing {field}"


def test_the_skill_description_names_the_route(card):
    # The description is where a caller learns the path, so it has to be in it.
    query = next(s for s in card["skills"] if s["id"] == "query-expertise")
    assert "/query-expertise" in query["description"]


def test_the_card_is_stable_for_the_same_request(card):
    import json
    assert json.dumps(card) == json.dumps(agent_card.build_agent_card(EVENT))


def test_mutating_a_returned_skill_does_not_poison_the_next_card(card):
    # SKILLS is module state; handing out the live dicts would let one
    # request's mutation leak into every later card from this container.
    card["skills"][0]["name"] = "clobbered"
    assert agent_card.build_agent_card(EVENT)["skills"][0]["name"] != "clobbered"


# ── the handler serves it ──────────────────────────────────────────────────

def test_the_handler_actually_serves_the_card():
    """Drive the real handler, not a grep over its source.

    Asserting the path constant appears somewhere in handler.py passed
    happily when the branch that serves it was disabled -- which is the only
    thing that matters.
    """
    import json

    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "src" / "query_api"))
    handler = pytest.importorskip("handler", reason="query_api handler needs its runtime deps")

    response = handler.lambda_handler(
        {"requestContext": {"http": {"method": "GET"},
                            "domainName": "cw.example.com", "stage": "prod"},
         "rawPath": "/.well-known/agent-card.json"},
        None,
    )
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["name"] == "ContextWeave"
    assert {s["id"] for s in body["skills"]} >= {"query-expertise"}


def test_the_card_route_is_declared_before_the_others():
    # It is how a caller learns the other routes exist; ordering it after a
    # catch-all would make it unreachable.
    source = (REPO / "src" / "query_api" / "handler.py").read_text()
    assert "import agent_card" in source
    assert source.index("is_agent_card_request") < source.index("is_health_request")


def test_the_template_forwards_the_well_known_uri():
    template = (REPO / "template.yaml").read_text()
    assert "/.well-known/agent-card.json" in template, \
        "the route exists in the handler but API Gateway never forwards it"
