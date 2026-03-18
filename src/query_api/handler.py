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
import time
import traceback
from typing import Any

import boto3

# With CodeUri: src/, Lambda adds /var/task/ to sys.path but the sibling
# modules live in /var/task/query_api/. Insert this directory so that
# bare module names (graph_expander, rag_router, etc.) resolve correctly.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from graph_expander import expand_graph_context
from rag_router import select_strategy, update_feedback
from retriever import deduplicate_chunks, retrieve_chunks, retrieve_with_strategy
from synthesizer import classify_question, synthesize_answer
from models import QueryRequest, RAGStrategyLabel

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Retained for backward compatibility with existing callers that pass these
# in the request; the retriever and router now use pgvector/Memgraph directly.
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "unused")
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
    Adaptive RAG reasoning pipeline:
      1. Classify the question
      2. Route: consult Neptune routing graph → select best RAG strategy
      3. Retrieve using chosen strategy (Bedrock KB ± Neptune vectors ± keyword boost)
      4. Deduplicate chunks
      5. Expand graph context via Neptune Analytics (controlled by routing config)
      6. Synthesize answer
      7. Update routing feedback in Neptune (learning loop)
      8. Return structured response with routingDecision field
    """
    start_time = time.monotonic()

    # Step 1 – Classify
    question_type = req.question_type or classify_question(req.question)
    logger.info("Question type: %s | Question: %s", question_type, req.question[:100])

    # Step 2 – Route: pick best retrieval strategy from Neptune routing graph
    retrieval_config = select_strategy(
        question_type=question_type,
        graph_id=NEPTUNE_GRAPH_ID or None,
    )
    logger.info(
        "Routing decision: strategy=%s graph=%s keywords=%s neptune_vecs=%s confidence=%.2f",
        retrieval_config.strategy,
        retrieval_config.include_graph,
        retrieval_config.boost_keywords,
        retrieval_config.use_neptune_chunks,
        retrieval_config.strategy_confidence,
    )

    # Step 3 – Retrieve with chosen strategy
    chunks = retrieve_with_strategy(
        question=req.question,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        config=retrieval_config,
        top_k=req.top_k,
        min_score=req.min_confidence,
        neptune_graph_id=NEPTUNE_GRAPH_ID,
    )

    # Step 4 – Dedup
    chunks = deduplicate_chunks(chunks)
    logger.info("Retrieved %d unique chunks via strategy=%s", len(chunks), retrieval_config.strategy)

    # Step 5 – Graph expansion
    # graph_first and hybrid strategies force graph expansion; others respect the request flag
    force_graph = retrieval_config.include_graph
    graph_context: dict[str, Any] = {
        "person_summary": {},
        "skill_neighbourhood": [],
        "pattern_evidence": [],
        "aws_context": [],
        "inferred_skills": [],
        "repeated_patterns": [],
        "graph_entities_used": [],
    }

    if (req.include_graph_expansion or force_graph) and NEPTUNE_GRAPH_ID:
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

    # Step 6 – Synthesize
    query_response = synthesize_answer(
        question=req.question,
        chunks=chunks,
        graph_context=graph_context,
        question_type=question_type,
    )

    # Step 7 – Routing feedback (learning loop)
    if NEPTUNE_GRAPH_ID:
        try:
            update_feedback(
                strategy=retrieval_config.strategy,
                question_type=question_type,
                confidence=query_response.confidence,
                graph_id=NEPTUNE_GRAPH_ID,
            )
        except Exception as exc:
            logger.warning("Routing feedback update failed (non-fatal): %s", exc)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "Pipeline complete in %dms | confidence=%.2f | strategy=%s | model=%s",
        elapsed_ms,
        query_response.confidence,
        retrieval_config.strategy,
        query_response.model_id,
    )

    # Attach routing metadata to response
    query_response.routing_decision = {
        "strategy": retrieval_config.strategy,
        "questionType": question_type,
        "strategyConfidence": retrieval_config.strategy_confidence,
        "graphExpansionForced": force_graph,
        "keywordBoostApplied": retrieval_config.boost_keywords,
        "neptuneVectorsUsed": retrieval_config.use_neptune_chunks,
    }

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
        from graph_expander import get_document_type_distribution
        doc_type_dist = []
        if NEPTUNE_GRAPH_ID:
            try:
                doc_type_dist = get_document_type_distribution(NEPTUNE_GRAPH_ID)
            except Exception:
                pass
        return _response(200, {
            "status": "healthy",
            "knowledgeBaseId": KNOWLEDGE_BASE_ID,
            "neptuneGraphId": NEPTUNE_GRAPH_ID,
            "environment": os.environ.get("ENVIRONMENT", "unknown"),
            "routingGraph": {
                "documentTypeDistribution": doc_type_dist,
            },
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
