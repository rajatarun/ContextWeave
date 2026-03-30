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
import cache as _cache
from shared.demo_logging import demo_for, demo_if, demo_step, demo_strategy_choice, resolve_log_level

logger = logging.getLogger()
logger.setLevel(resolve_log_level(os.environ.get("LOG_LEVEL", "INFO")))

# Retained for backward compatibility with existing callers that pass these
# in the request; the retriever and router now use pgvector/Memgraph directly.
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "unused")
NEPTUNE_GRAPH_ID = os.environ.get("NEPTUNE_GRAPH_ID", "")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "POST,GET,OPTIONS",
}


def _response(status: int, body: Any) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "X-Content-Type-Options": "nosniff",
            **_CORS_HEADERS,
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
      0. Embed question; check semantic cache (skip steps 1-6 on hit)
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
    demo_step(logger, "Starting adaptive query pipeline execution")

    # Step 0 – Semantic cache check (CAG)
    # Embed once here; retriever will embed again internally on a cache miss
    # (double embed cost ~$0.00002 — negligible vs synthesis savings).
    cache_allowed = not _cache.is_time_sensitive(req.question)
    demo_if(logger, "question is not time-sensitive (cache allowed)", cache_allowed)
    if cache_allowed:
        try:
            import sys as _sys, os as _os
            shared_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "shared")
            if shared_dir not in _sys.path:
                _sys.path.insert(0, shared_dir)
            import importlib
            _embedder = importlib.import_module("embedder")
            _db = importlib.import_module("db_clients")

            question_embedding = _embedder.embed_text(req.question)
            has_embedding = question_embedding is not None
            demo_if(logger, "question embedding was generated", has_embedding)
            if has_embedding:
                cached = _cache.check_cache(question_embedding, _db.get_pg_connection())
                cache_hit = cached is not None
                demo_if(logger, "semantic cache returned a result", cache_hit)
                if cache_hit:
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)
                    logger.info("Cache HIT for question (%.0fms): %s", elapsed_ms, req.question[:80])
                    cached["cacheHit"] = True
                    cached["latencyMs"] = elapsed_ms
                    return cached
        except Exception as exc:
            logger.warning("Cache check failed (non-fatal): %s", exc)
            question_embedding = None
    else:
        logger.info("Bypassing cache: time-sensitive question")
        question_embedding = None

    # Step 1 – Classify
    demo_step(logger, "Classifying incoming question for routing type")
    question_type = req.question_type or classify_question(req.question)
    logger.info("Question type: %s | Question: %s", question_type, req.question[:100])

    # Step 2 – Route: pick best retrieval strategy from Neptune routing graph
    demo_step(logger, "Selecting retrieval strategy from routing graph")
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
    demo_strategy_choice(
        logger,
        str(retrieval_config.strategy),
        float(retrieval_config.strategy_confidence),
    )

    # Step 3 – Retrieve with chosen strategy
    demo_step(logger, "Retrieving chunks using selected strategy configuration")
    chunks = retrieve_with_strategy(
        question=req.question,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        config=retrieval_config,
        top_k=req.top_k,
        min_score=req.min_confidence,
        neptune_graph_id=NEPTUNE_GRAPH_ID,
    )

    # Step 4 – Dedup
    demo_step(logger, "Deduplicating retrieved chunks for concise evidence set")
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

    should_expand_graph = (req.include_graph_expansion or force_graph) and bool(NEPTUNE_GRAPH_ID)
    demo_if(
        logger,
        "graph expansion requested/forced and NEPTUNE_GRAPH_ID is configured",
        should_expand_graph,
    )
    if should_expand_graph:
        try:
            snippet_texts: list[str] = []
            for index, chunk in enumerate(chunks, start=1):
                demo_for(logger, "retrieved chunks for graph context snippets", index, len(chunks))
                snippet_texts.append(chunk.content)
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
    demo_step(logger, "Synthesizing final answer from chunks and graph context")
    query_response = synthesize_answer(
        question=req.question,
        chunks=chunks,
        graph_context=graph_context,
        question_type=question_type,
    )

    # Step 7 – Write to semantic cache (non-fatal; skip low-confidence answers)
    should_cache_write = query_response.confidence >= 0.5
    demo_if(logger, "response confidence >= 0.5 (eligible for cache write)", should_cache_write)
    if should_cache_write:
        try:
            has_embedding = question_embedding is not None
            demo_if(logger, "question embedding available for cache write", has_embedding)
            if has_embedding:
                import importlib, sys as _sys, os as _os
                shared_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "shared")
                if shared_dir not in _sys.path:
                    _sys.path.insert(0, shared_dir)
                _db = importlib.import_module("db_clients")
                _cache.write_cache(
                    question_embedding,
                    query_response.to_dict(),
                    question_type,
                    _db.get_pg_connection(),
                )
        except Exception as exc:
            logger.warning("Cache write failed (non-fatal): %s", exc)
    else:
        logger.info(
            "Skipping cache write: confidence %.2f < 0.5", query_response.confidence
        )

    # Step 8 – Routing feedback (learning loop)
    demo_step(logger, "Updating routing feedback weights after answer confidence assessment")
    has_neptune_graph = bool(NEPTUNE_GRAPH_ID)
    demo_if(logger, "NEPTUNE_GRAPH_ID configured for routing feedback update", has_neptune_graph)
    if has_neptune_graph:
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
    result["cacheHit"] = False
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
    is_health_request = method == "GET" and "/health" in path
    demo_if(logger, "request targets GET /health", is_health_request)
    if is_health_request:
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
    is_query_request = method == "POST" and "/query-expertise" in path
    demo_if(logger, "request targets POST /query-expertise", is_query_request)
    if is_query_request:
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

    # ── CORS preflight ────────────────────────────────────────────────────────
    is_options = method == "OPTIONS"
    demo_if(logger, "request uses OPTIONS (CORS preflight)", is_options)
    if is_options:
        return {"statusCode": 200, "headers": _CORS_HEADERS, "body": ""}

    # ── 404 ───────────────────────────────────────────────────────────────────
    return _error(404, "Not found", f"No handler for {method} {path}")
