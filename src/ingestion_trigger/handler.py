"""
ExpertiseRAG – DB Initializer Lambda Handler

Replaces the previous Bedrock Knowledge Base ingestion trigger.
Serves two roles:

1. CloudFormation Custom Resource (during deployment)
   CloudFormation sends Create/Update/Delete events.
   On Create/Update → initialise pgvector schema + seed Memgraph routing graph.
   On Delete → no-op.

2. Direct invocation (manual re-seeding)
   Accepts { "action": "init_db" | "seed_routing" } for targeted operations.

Both modes use the shared db_clients module to reach Memgraph and PostgreSQL.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from typing import Any

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def _get_shared_module(name: str):
    shared_dir = os.path.join(os.path.dirname(__file__), "..", "shared")
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    import importlib
    return importlib.import_module(name)


# ─────────────────────────────────────────────────────────────────────────────
# DB initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db_schema() -> dict:
    """
    Idempotently create the pgvector extension and chunks table in PostgreSQL.

    Returns {"status": "success", "message": "..."}.
    """
    try:
        db_clients = _get_shared_module("db_clients")
        db_clients.init_pgvector_schema()
        return {"status": "success", "message": "pgvector schema initialised"}
    except Exception as exc:
        logger.error("pgvector schema init failed: %s", exc, exc_info=True)
        return {"status": "failed", "reason": str(exc)}


def seed_routing_graph_memgraph() -> dict:
    """
    Seed the adaptive routing graph in Memgraph with initial prior weights.
    Idempotent – safe to call multiple times.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "query_api"))
        from rag_router import seed_routing_graph
        result = seed_routing_graph()
        logger.info("Routing graph seeded: %s", result)
        return {"status": "success", **result}
    except Exception as exc:
        logger.error("Routing graph seed failed: %s", exc, exc_info=True)
        return {"status": "failed", "reason": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# CloudFormation custom resource helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfn_send(
    event: dict,
    context: Any,
    status: str,
    physical_resource_id: str,
    data: dict | None = None,
    reason: str = "",
) -> None:
    """
    Send a response to the CloudFormation custom resource pre-signed S3 URL.
    This is required; omitting it will cause the stack to hang until timeout.
    """
    response_url = event.get("ResponseURL", "")
    if not response_url:
        logger.warning("No ResponseURL in CFN event – skipping response send")
        return

    body = {
        "Status": status,
        "Reason": reason or f"See logs in CloudWatch: {context.log_stream_name}",
        "PhysicalResourceId": physical_resource_id,
        "StackId": event.get("StackId", ""),
        "RequestId": event.get("RequestId", ""),
        "LogicalResourceId": event.get("LogicalResourceId", ""),
        "NoEcho": False,
        "Data": data or {},
    }
    json_body = json.dumps(body).encode("utf-8")

    logger.info("Sending CFN response: status=%s | URL=%s", status, response_url[:60])
    req = urllib.request.Request(
        response_url,
        data=json_body,
        headers={
            "Content-Type": "",
            "Content-Length": str(len(json_body)),
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            logger.info("CFN response sent: HTTP %d", resp.status)
    except Exception as exc:
        logger.error("Failed to send CFN response: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Lambda handler
# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    """
    Handles three event types:

    A) CloudFormation custom resource:
       event["RequestType"] in ("Create", "Update", "Delete")

    B) Direct invocation – init DB:
       event["action"] == "init_db"

    C) Direct invocation – seed routing graph:
       event["action"] == "seed_routing"
    """
    logger.info("Event: %s", json.dumps(event, default=str))

    # ── CloudFormation Custom Resource ────────────────────────────────────────
    if "RequestType" in event:
        request_type = event["RequestType"]
        physical_id = event.get("PhysicalResourceId", "expertise-rag-db-init")

        if request_type == "Delete":
            logger.info("Delete event – no DB cleanup required")
            _cfn_send(event, context, "SUCCESS", physical_id,
                      data={"Message": "No cleanup required"})
            return {"status": "deleted"}

        # Create or Update: init pgvector schema + seed routing graph
        errors: list[str] = []

        pg_result = init_db_schema()
        if pg_result["status"] != "success":
            errors.append(f"pgvector init: {pg_result.get('reason', 'unknown')}")

        seed_result = seed_routing_graph_memgraph()
        if seed_result["status"] != "success":
            # Non-fatal: Memgraph may not be ready yet on first deploy
            logger.warning("Routing graph seed failed (non-fatal): %s", seed_result)

        if errors:
            reason = "; ".join(errors)
            _cfn_send(event, context, "FAILED", physical_id, reason=reason)
            return {"status": "failed", "reason": reason}

        _cfn_send(
            event, context, "SUCCESS", physical_id,
            data={
                "Message": "DB schema initialised and routing graph seeded",
                "PgvectorStatus": pg_result["status"],
                "RoutingGraphNodes": str(seed_result.get("nodes", 0)),
                "RoutingGraphEdges": str(seed_result.get("edges", 0)),
            },
        )
        return {
            "status": "success",
            "pgvector": pg_result,
            "routing_graph": seed_result,
        }

    # ── Direct invocation: init DB schema ─────────────────────────────────────
    if event.get("action") == "init_db":
        result = init_db_schema()
        return {"action": "init_db", **result}

    # ── Direct invocation: seed routing graph ─────────────────────────────────
    if event.get("action") == "seed_routing":
        result = seed_routing_graph_memgraph()
        return {"action": "seed_routing", **result}

    # ── Default: run both init steps ──────────────────────────────────────────
    logger.info("Default invocation: running full DB initialisation")
    pg_result = init_db_schema()
    seed_result = seed_routing_graph_memgraph()
    return {
        "status": "success" if pg_result["status"] == "success" else "partial",
        "pgvector": pg_result,
        "routing_graph": seed_result,
    }
