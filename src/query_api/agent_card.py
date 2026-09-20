"""ContextWeave's A2A Agent Card.

TeamWeave reached this service by assuming `{CONTEXTWEAVE_URL}/query-expertise`
— a hardcoded path in a sibling repository, which is the coupling the weave
platform exists to avoid. [A2A](https://github.com/a2aproject/A2A) 1.0.0
(Linux Foundation, January 2026) replaces that with a document: a client
fetches `/.well-known/agent-card.json`, reads the interface URL and the skills,
and calls what it finds.

Written against the published v1.0 shape, which differs from 0.3.x in the part
that matters: `supportedInterfaces` replaced the top-level `url` and
`preferredTransport`, each entry carrying its own `protocolBinding` and
`protocolVersion`. A 0.3-shaped card parses fine and tells a 1.0 client
nothing.

The card describes only what this service serves. ContextWeave answers
questions and takes ratings over its own HTTP+JSON API; it does not implement
A2A's `message:send`, so it declares no A2A transport binding and no
streaming. A client reads the skills to learn what is here and the interface
to learn where — advertising a protocol that is not implemented would be a lie
a machine acts on.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

# The card's own version, not the protocol's.
CARD_VERSION = "1.0.0"
AGENT_CARD_PATH = "/.well-known/agent-card.json"

DEFAULT_MODES = ["application/json"]

# Each skill names the route that serves it, so a caller never has to guess a
# path again -- that guess is exactly what this replaces.
SKILLS: List[Dict[str, Any]] = [
    {
        "id": "query-expertise",
        "name": "Query expertise",
        "description": (
            "Answer a question about this developer's professional expertise, "
            "grounded in a GraphRAG corpus. Returns an answer with sources, "
            "inferred skills, a confidence score and a queryId. The router "
            "selects a retrieval strategy per question type and learns from "
            "feedback. POST to /query-expertise with {\"question\": \"...\"}."
        ),
        "tags": ["contextweave", "rag", "graphrag", "expertise", "retrieval"],
        "examples": [
            "What AWS services has this developer deployed in production?",
            "{\"question\": \"Describe their experience with vector databases\", \"top_k\": 6}",
        ],
        "inputModes": list(DEFAULT_MODES),
        "outputModes": list(DEFAULT_MODES),
    },
    {
        "id": "feedback",
        "name": "Rate an answer",
        "description": (
            "Fold an independent rating into the router's posterior for the "
            "strategy that produced an answer. Self-confidence alone rewards a "
            "confidently wrong answer, so this is the second reward source. "
            "POST to /feedback with {\"queryId\": \"...\", \"rating\": \"up\"}."
        ),
        "tags": ["contextweave", "feedback", "routing", "learning"],
        "inputModes": list(DEFAULT_MODES),
        "outputModes": list(DEFAULT_MODES),
    },
    {
        "id": "routing-decisions",
        "name": "Read the routing decision log",
        "description": (
            "Per-decision rows or a grouped summary with meanAbsDiff, the "
            "calibration check on whether self-confidence predicts ratings. "
            "Read-only and free of PII. GET /routing-decisions?mode=summary."
        ),
        "tags": ["contextweave", "observability", "routing", "calibration"],
        "inputModes": list(DEFAULT_MODES),
        "outputModes": list(DEFAULT_MODES),
    },
]


def base_url(event: Dict[str, Any]) -> str:
    """Where this deployment actually answers.

    Derived from the request rather than configured, so the card cannot
    advertise a URL this deployment does not serve -- which is the failure it
    exists to prevent.
    """
    configured = (os.environ.get("A2A_BASE_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    context = event.get("requestContext") or {}
    domain = context.get("domainName") or ""
    if not domain:
        return ""
    stage = context.get("stage") or ""
    suffix = f"/{stage}" if stage and stage != "$default" else ""
    return f"https://{domain}{suffix}"


def build_agent_card(event: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url(event)
    return {
        "name": "ContextWeave",
        "description": (
            "GraphRAG + CAG knowledge layer. Answers evidence-backed questions "
            "about a developer's expertise, with an adaptive router that learns "
            "which retrieval strategy suits each question type."
        ),
        "version": os.environ.get("A2A_AGENT_VERSION", CARD_VERSION),
        "supportedInterfaces": [
            {
                "url": url,
                # ContextWeave serves its own HTTP+JSON API. It does not
                # implement A2A's message:send, so this names the binding a
                # caller actually gets, and the skills name the routes.
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "provider": {"organization": os.environ.get("A2A_PROVIDER", "ContextWeave")},
        "defaultInputModes": list(DEFAULT_MODES),
        "defaultOutputModes": list(DEFAULT_MODES),
        "skills": [dict(skill) for skill in SKILLS],
    }
