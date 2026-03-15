"""
ExpertiseRAG – Ingestion Trigger Lambda Handler

Serves two roles:

1. CloudFormation Custom Resource (during deployment)
   CloudFormation sends Create/Update/Delete events.
   On Create/Update → start a Bedrock ingestion job and wait (up to timeout).
   On Delete → no-op (ingestion jobs cannot be cancelled mid-flight).

2. Step Functions task / direct invocation (after new uploads)
   Accepts { "action": "start" | "status", ... } and responds accordingly.
   The Step Functions state machine polls via "status" until COMPLETE | FAILED.

Both modes use the same underlying boto3 calls:
  - bedrock-agent: start_ingestion_job
  - bedrock-agent: get_ingestion_job

See also: scripts/start_ingestion.py for a standalone boto3 example.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Any

import boto3
import urllib.request
import urllib.parse
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
DATA_SOURCE_ID = os.environ.get("DATA_SOURCE_ID", "")
NEPTUNE_GRAPH_ID = os.environ.get("NEPTUNE_GRAPH_ID", "")

_BEDROCK_AGENT: Any = None


def _get_client() -> Any:
    global _BEDROCK_AGENT
    if _BEDROCK_AGENT is None:
        _BEDROCK_AGENT = boto3.client(
            "bedrock-agent",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _BEDROCK_AGENT


# ─────────────────────────────────────────────────────────────────────────────
# Core ingestion operations
# ─────────────────────────────────────────────────────────────────────────────

def start_ingestion_job(knowledge_base_id: str, data_source_id: str) -> dict:
    """
    Start a Bedrock Knowledge Base ingestion job.

    Returns the ingestion job dict from the Bedrock API response.
    """
    client = _get_client()
    client_token = str(uuid.uuid4())

    logger.info(
        "Starting ingestion job: KB=%s DataSource=%s",
        knowledge_base_id,
        data_source_id,
    )
    try:
        response = client.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            clientToken=client_token,
            description="ExpertiseRAG automated ingestion",
        )
        job = response.get("ingestionJob", {})
        logger.info(
            "Ingestion job started: %s | status=%s",
            job.get("ingestionJobId"),
            job.get("status"),
        )
        return job
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ConflictException":
            logger.warning(
                "Ingestion job already in progress – retrieving latest job"
            )
            return get_latest_ingestion_job(knowledge_base_id, data_source_id)
        raise


def get_ingestion_job(
    knowledge_base_id: str, data_source_id: str, ingestion_job_id: str
) -> dict:
    """Fetch the current status of a specific ingestion job."""
    client = _get_client()
    response = client.get_ingestion_job(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=data_source_id,
        ingestionJobId=ingestion_job_id,
    )
    return response.get("ingestionJob", {})


def get_latest_ingestion_job(knowledge_base_id: str, data_source_id: str) -> dict:
    """Return the most recently started ingestion job (any status)."""
    client = _get_client()
    try:
        response = client.list_ingestion_jobs(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            sortBy={"attribute": "STARTED_AT", "order": "DESCENDING"},
            maxResults=1,
        )
        jobs = response.get("ingestionJobSummaries", [])
        if jobs:
            job_id = jobs[0]["ingestionJobId"]
            return get_ingestion_job(knowledge_base_id, data_source_id, job_id)
    except ClientError as exc:
        logger.warning("Could not list ingestion jobs: %s", exc)
    return {}


def wait_for_ingestion(
    knowledge_base_id: str,
    data_source_id: str,
    ingestion_job_id: str,
    poll_interval: int = 15,
    max_wait_seconds: int = 840,  # 14 minutes (Lambda timeout = 15 min)
) -> dict:
    """
    Poll ingestion job status until COMPLETE or FAILED.
    Returns the final job dict.
    """
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        job = get_ingestion_job(knowledge_base_id, data_source_id, ingestion_job_id)
        status = job.get("status", "")
        logger.info("Ingestion job %s status: %s", ingestion_job_id, status)

        if status in ("COMPLETE", "FAILED", "STOPPED"):
            if status == "FAILED":
                failure_reasons = job.get("failureReasons", [])
                logger.error("Ingestion failed: %s", failure_reasons)
            return job

        time.sleep(poll_interval)

    logger.warning(
        "Ingestion job %s did not complete within %ds – returning last known status",
        ingestion_job_id,
        max_wait_seconds,
    )
    return get_ingestion_job(knowledge_base_id, data_source_id, ingestion_job_id)


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

    B) Step Functions task – start:
       event["action"] == "start"

    C) Step Functions task – poll status:
       event["action"] == "status"
       event["ingestion_job_id"] == "<job id>"
    """
    logger.info("Event: %s", json.dumps(event, default=str))

    kb_id = event.get("KnowledgeBaseId") or event.get("knowledge_base_id") or KNOWLEDGE_BASE_ID
    ds_id = event.get("DataSourceId") or event.get("data_source_id") or DATA_SOURCE_ID

    # ── CloudFormation Custom Resource ────────────────────────────────────────
    if "RequestType" in event:
        request_type = event["RequestType"]
        physical_id = event.get("PhysicalResourceId", f"ingestion-{kb_id}")

        if request_type == "Delete":
            logger.info("Delete event – nothing to clean up for ingestion jobs")
            _cfn_send(event, context, "SUCCESS", physical_id,
                      data={"Message": "No cleanup required"})
            return {"status": "deleted"}

        # Create or Update: seed routing graph, then start ingestion
        if not kb_id or not ds_id:
            msg = "KnowledgeBaseId and DataSourceId are required"
            logger.error(msg)
            _cfn_send(event, context, "FAILED", physical_id, reason=msg)
            return {"status": "failed", "reason": msg}

        # Seed the adaptive routing graph (idempotent)
        graph_id = event.get("NeptuneGraphId") or NEPTUNE_GRAPH_ID
        if graph_id:
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "query_api"))
                from rag_router import seed_routing_graph
                seed_result = seed_routing_graph(graph_id)
                logger.info("Routing graph seeded: %s", seed_result)
            except Exception as exc:
                logger.warning("Routing graph seed failed (non-fatal): %s", exc)

        try:
            job = start_ingestion_job(kb_id, ds_id)
            job_id = job.get("ingestionJobId", "")

            if job_id:
                final_job = wait_for_ingestion(kb_id, ds_id, job_id)
            else:
                final_job = job

            status = final_job.get("status", "UNKNOWN")
            if status == "COMPLETE":
                _cfn_send(
                    event, context, "SUCCESS", job_id or physical_id,
                    data={
                        "IngestionJobId": job_id,
                        "Status": status,
                        "Statistics": json.dumps(final_job.get("statistics", {})),
                    },
                )
                return {"status": "success", "ingestion_job_id": job_id}
            else:
                reason = f"Ingestion ended with status: {status}. Reasons: {final_job.get('failureReasons', [])}"
                _cfn_send(event, context, "FAILED", job_id or physical_id, reason=reason)
                return {"status": "failed", "reason": reason}

        except Exception as exc:
            reason = f"Unexpected error: {exc}"
            logger.error(reason, exc_info=True)
            _cfn_send(event, context, "FAILED", physical_id, reason=reason)
            return {"status": "failed", "reason": reason}

    # ── Seed routing graph (standalone invocation) ────────────────────────────
    if event.get("action") == "seed_routing":
        graph_id = event.get("neptune_graph_id") or NEPTUNE_GRAPH_ID
        if not graph_id:
            raise ValueError("neptune_graph_id or NEPTUNE_GRAPH_ID env var required")
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "query_api"))
            from rag_router import seed_routing_graph
            result = seed_routing_graph(graph_id)
            return {"action": "seed_routing", "status": "success", **result}
        except Exception as exc:
            logger.error("Routing graph seed failed: %s", exc, exc_info=True)
            return {"action": "seed_routing", "status": "failed", "reason": str(exc)}

    # ── Step Functions: start ingestion ───────────────────────────────────────
    if event.get("action") == "start":
        if not kb_id or not ds_id:
            raise ValueError("knowledge_base_id and data_source_id are required")
        job = start_ingestion_job(kb_id, ds_id)
        return {
            "action": "started",
            "ingestion_job_id": job.get("ingestionJobId", ""),
            "status": job.get("status", "STARTING"),
        }

    # ── Step Functions: poll status ───────────────────────────────────────────
    if event.get("action") == "status":
        job_id = event.get("ingestion_job_id", "")
        if not job_id:
            raise ValueError("ingestion_job_id is required for action=status")
        job = get_ingestion_job(kb_id, ds_id, job_id)
        return {
            "action": "status",
            "ingestion_job_id": job_id,
            "status": job.get("status", "UNKNOWN"),
            "statistics": job.get("statistics", {}),
            "failure_reasons": job.get("failureReasons", []),
        }

    # ── Direct invocation with no action: start + wait ────────────────────────
    logger.info("Direct invocation: start ingestion and wait for completion")
    if not kb_id or not ds_id:
        raise ValueError("KNOWLEDGE_BASE_ID and DATA_SOURCE_ID environment variables must be set")

    job = start_ingestion_job(kb_id, ds_id)
    job_id = job.get("ingestionJobId", "")
    if job_id:
        final_job = wait_for_ingestion(kb_id, ds_id, job_id)
    else:
        final_job = job

    return {
        "status": final_job.get("status", "UNKNOWN"),
        "ingestion_job_id": job_id,
        "statistics": final_job.get("statistics", {}),
        "failure_reasons": final_job.get("failureReasons", []),
    }
