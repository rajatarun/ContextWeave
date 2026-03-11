"""
ExpertiseRAG – Query API Lambda Handler

Exposes:
  POST /query-expertise  – main reasoning endpoint
  GET  /health           – health check

Request body (POST /query-expertise):
  {
    "question": "What AWS services has this developer built production systems with?",
    "topK": 10,
    "includeGraphExpansion": true,
    "minConfidence": 0.3
  }

Response body:
  {
    "answer": "...",
    "sources": [...],
    "inferredSkills": [...],
    "repeatedPatterns": [...],
    "confidence": 0.92,
    "questionType": "architecture",
    "graphEntitiesUsed": [...],
    "retrievalCount": 8,
    "modelId": "anthropic.claude-3-5-sonnet-..."
  }
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from typing import Any

import boto3

# Allow imports from sibling packages in flat Lambda zip layout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_api.graph_expander import expand_graph_context
from query_api.retriever import deduplicate_chunks, retrieve_chunks
from query_api.synthesizer import classify_question, synthesize_answer
from shared.models import QueryRequest

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
NEPTUNE_GRAPH_ID = os.environ.get("NEPTUNE_GRAPH_ID", "")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _response(status: int, body: Any) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "X-Content-Type-Options": "nosniff",
        },
        "body": json.dumps(body, default=str),
    }


def _error(status: int, message: str, details: str = "") -> dict:
    payload: dict[str, Any] = {"error": message}
    if details:
        payload["details"] = details
    return _response(status, payload)


# ─────────────────────────────────────────────────────────────────────────────
# Request validation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_request(event: dict) -> QueryRequest | None:
    """Parse and validate the incoming HTTP API event body."""
    body_raw = event.get("body", "") or ""
    if event.get("isBase64Encoded"):
        import base64
        body_raw = base64.b64decode(body_raw).decode("utf-8")

    try:
        body = json.loads(body_raw) if body_raw.strip() else {}
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON body: %s", exc)
        return None

    question = body.get("question", "").strip()
    if not question:
        return None

    return QueryRequest.from_dict(body)


# ─────────────────────────────────────────────────────────────────────────────
# Core reasoning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_query_pipeline(req: QueryRequest) -> dict:
    """
    Full reasoning pipeline:
      1. Classify the question
      2. Retrieve from Bedrock Knowledge Base
      3. Deduplicate chunks
      4. Expand graph context via Neptune Analytics (optional)
      5. Synthesize answer
      6. Return structured response
    """
    start_time = time.monotonic()

    # Step 1 – Classify
    question_type = req.question_type or classify_question(req.question)
    logger.info("Question type: %s | Question: %s", question_type, req.question[:100])

    # Step 2 – Retrieve from Bedrock KB
    if not KNOWLEDGE_BASE_ID:
        raise ValueError("KNOWLEDGE_BASE_ID environment variable is not set")

    chunks = retrieve_chunks(
        question=req.question,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        top_k=req.top_k,
        min_score=req.min_confidence,
    )

    # Step 3 – Dedup
    chunks = deduplicate_chunks(chunks)

    logger.info("Retrieved %d unique chunks", len(chunks))

    # Step 4 – Graph expansion
    graph_context: dict[str, Any] = {
        "person_summary": {},
        "skill_neighbourhood": [],
        "pattern_evidence": [],
        "aws_context": [],
        "inferred_skills": [],
        "repeated_patterns": [],
        "graph_entities_used": [],
    }

    if req.include_graph_expansion and NEPTUNE_GRAPH_ID:
        try:
            snippet_texts = [c.content for c in chunks]
            graph_context = expand_graph_context(
                retrieved_text_snippets=snippet_texts,
                graph_id=NEPTUNE_GRAPH_ID,
            )
            logger.info(
                "Graph expansion: %d inferred skills, %d repeated patterns",
                len(graph_context.get("inferred_skills", [])),
                len(graph_context.get("repeated_patterns", [])),
            )
        except Exception as exc:
            logger.warning("Graph expansion failed (non-fatal): %s", exc)

    # Step 5 – Synthesize
    query_response = synthesize_answer(
        question=req.question,
        chunks=chunks,
        graph_context=graph_context,
        question_type=question_type,
    )

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "Pipeline complete in %dms | confidence=%.2f | model=%s",
        elapsed_ms,
        query_response.confidence,
        query_response.model_id,
    )

    result = query_response.to_dict()
    result["latencyMs"] = elapsed_ms
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Lambda handler
# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    """
    HTTP API Gateway v2 Lambda proxy integration handler.
    """
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    path = event.get("rawPath", "")

    logger.info("Request: %s %s", method, path)

    # ── Health check ──────────────────────────────────────────────────────────
    if method == "GET" and "/health" in path:
        return _response(200, {
            "status": "healthy",
            "knowledgeBaseId": KNOWLEDGE_BASE_ID,
            "neptuneGraphId": NEPTUNE_GRAPH_ID,
            "environment": os.environ.get("ENVIRONMENT", "unknown"),
        })

    # ── POST /query-expertise ─────────────────────────────────────────────────
    if method == "POST" and "/query-expertise" in path:
        req = _parse_request(event)
        if req is None:
            return _error(
                400,
                "Invalid request",
                "Body must be JSON with a non-empty 'question' field. "
                "Optional: topK (int), includeGraphExpansion (bool), minConfidence (float).",
            )

        if len(req.question) > 2000:
            return _error(400, "Question too long", "Maximum question length is 2000 characters.")

        if req.top_k < 1 or req.top_k > 50:
            return _error(400, "Invalid topK", "topK must be between 1 and 50.")

        try:
            result = _run_query_pipeline(req)
            return _response(200, result)
        except ValueError as exc:
            logger.error("Configuration error: %s", exc)
            return _error(500, "Service misconfigured", str(exc))
        except Exception as exc:
            logger.error("Unhandled error: %s\n%s", exc, traceback.format_exc())
            return _error(500, "Internal server error", str(exc))

    # ── 404 ───────────────────────────────────────────────────────────────────
    return _error(404, "Not found", f"No handler for {method} {path}")
