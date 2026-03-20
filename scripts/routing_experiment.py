#!/usr/bin/env python3
"""
ContextWeave / ExpertiseRAG – Adaptive Routing Experiment
==========================================================
Fires 100+ questions at the deployed /query-expertise endpoint to stress-test
and refine the Neptune routing-weight graph.

Each question is sent sequentially so that the feedback loop (weight updates
after each answered query) accumulates continuously.

Usage
-----
  python scripts/routing_experiment.py \
      --endpoint https://wnbdgd0z4g.execute-api.us-east-1.amazonaws.com/prod

  # Resume from a checkpoint if interrupted:
  python scripts/routing_experiment.py \
      --endpoint https://wnbdgd0z4g.execute-api.us-east-1.amazonaws.com/prod \
      --resume

  # Run only a subset of question types:
  python scripts/routing_experiment.py \
      --endpoint https://wnbdgd0z4g.execute-api.us-east-1.amazonaws.com/prod \
      --types skill_depth architecture

  # Increase concurrency (2-4 recommended; higher risks race-conditions on weights):
  python scripts/routing_experiment.py \
      --endpoint https://wnbdgd0z4g.execute-api.us-east-1.amazonaws.com/prod \
      --concurrency 2

Output
------
  - Live console table (one row per completed question)
  - JSON results file: experiment_results_<timestamp>.json
  - Summary report printed at the end
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─────────────────────────────────────────────────────────────────────────────
# Question bank – 120 questions across 6 types
# Repetition is intentional: identical/near-identical questions in succession
# let the weight updates accumulate and expose convergence behaviour.
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS: list[dict[str, str]] = [
    # ── skill_depth (20 questions) ────────────────────────────────────────────
    {"type": "skill_depth", "q": "How deep is Rajat's expertise with AWS Lambda?"},
    {"type": "skill_depth", "q": "How proficient is Rajat with Amazon Neptune?"},
    {"type": "skill_depth", "q": "What is Rajat's level of experience with Python?"},
    {"type": "skill_depth", "q": "How skilled is Rajat with AWS SAM and CloudFormation?"},
    {"type": "skill_depth", "q": "How strong is Rajat's knowledge of Amazon Bedrock?"},
    {"type": "skill_depth", "q": "How many years has Rajat worked with serverless architectures?"},
    {"type": "skill_depth", "q": "What is Rajat's depth of expertise with GraphRAG systems?"},
    {"type": "skill_depth", "q": "How expert is Rajat in building RAG pipelines?"},
    {"type": "skill_depth", "q": "How well does Rajat know AWS Step Functions?"},
    {"type": "skill_depth", "q": "What is Rajat's proficiency with vector databases?"},
    {"type": "skill_depth", "q": "How deep is Rajat's knowledge of CI/CD pipelines with GitHub Actions?"},
    {"type": "skill_depth", "q": "How experienced is Rajat with API Gateway design?"},
    {"type": "skill_depth", "q": "What is Rajat's expertise level with AWS KMS and encryption?"},
    {"type": "skill_depth", "q": "How strong are Rajat's skills in building event-driven systems?"},
    {"type": "skill_depth", "q": "How proficient is Rajat with OpenCypher graph queries?"},
    # Repeats to drive weight refinement
    {"type": "skill_depth", "q": "How skilled is Rajat at working with Bedrock Knowledge Bases?"},
    {"type": "skill_depth", "q": "What depth of expertise does Rajat have in AWS Lambda Python runtimes?"},
    {"type": "skill_depth", "q": "How well does Rajat understand Neptune Analytics graph traversal?"},
    {"type": "skill_depth", "q": "How experienced is Rajat with Amazon Titan Text Embeddings?"},
    {"type": "skill_depth", "q": "Does Rajat have deep knowledge of hierarchical RAG chunking?"},

    # ── architecture (20 questions) ───────────────────────────────────────────
    {"type": "architecture", "q": "Describe the overall architecture of the ContextWeave system."},
    {"type": "architecture", "q": "How does the GraphRAG pipeline work in ExpertiseRAG?"},
    {"type": "architecture", "q": "What is the event-driven design pattern used in this project?"},
    {"type": "architecture", "q": "How does the adaptive routing graph in Neptune work?"},
    {"type": "architecture", "q": "Explain the hierarchical chunking strategy and why it was chosen."},
    {"type": "architecture", "q": "How does the system orchestrate document ingestion with Step Functions?"},
    {"type": "architecture", "q": "What AWS services form the backbone of the ExpertiseRAG platform?"},
    {"type": "architecture", "q": "How is encryption and security implemented at rest and in transit?"},
    {"type": "architecture", "q": "Describe the multi-strategy retrieval approach (graph_first, hybrid, keyword_boosted, semantic_search)."},
    {"type": "architecture", "q": "How does the OIDC-based deployment pipeline work?"},
    {"type": "architecture", "q": "What is the role of Neptune Analytics in the retrieval pipeline?"},
    {"type": "architecture", "q": "How does the preprocessing Lambda classify document types?"},
    {"type": "architecture", "q": "Explain the feedback loop that refines routing weights after each query."},
    {"type": "architecture", "q": "How does the system achieve serverless scalability without polling in Lambda?"},
    {"type": "architecture", "q": "What design decisions were made to avoid anti-patterns like long-lived credentials?"},
    # Repeats
    {"type": "architecture", "q": "What is the C4 container model for ContextWeave?"},
    {"type": "architecture", "q": "How are parent and child chunks used in the Bedrock Knowledge Base?"},
    {"type": "architecture", "q": "How does graph expansion augment semantic retrieval in the query pipeline?"},
    {"type": "architecture", "q": "What CloudFormation resources are provisioned for this system?"},
    {"type": "architecture", "q": "How does ExpertiseRAG handle document classification and chunking strategy assignment?"},

    # ── project (20 questions) ────────────────────────────────────────────────
    {"type": "project", "q": "What is the ContextWeave project and what problem does it solve?"},
    {"type": "project", "q": "What did Rajat build in ExpertiseRAG?"},
    {"type": "project", "q": "What AWS services did Rajat deploy production systems with?"},
    {"type": "project", "q": "What projects has Rajat delivered using serverless architecture?"},
    {"type": "project", "q": "What role did Rajat play in building this GraphRAG platform?"},
    {"type": "project", "q": "What was Rajat's responsibility in the ExpertiseRAG system design?"},
    {"type": "project", "q": "What applications has Rajat shipped using Amazon Bedrock?"},
    {"type": "project", "q": "What experience does Rajat have with Neptune Analytics projects?"},
    {"type": "project", "q": "What production systems has Rajat built on AWS?"},
    {"type": "project", "q": "What CI/CD pipelines has Rajat implemented in GitHub Actions?"},
    {"type": "project", "q": "What document ingestion workflows has Rajat built?"},
    {"type": "project", "q": "What RAG systems has Rajat delivered in a production environment?"},
    {"type": "project", "q": "What Step Functions workflows has Rajat designed and deployed?"},
    {"type": "project", "q": "What infrastructure as code projects has Rajat completed using AWS SAM?"},
    {"type": "project", "q": "What knowledge graph products has Rajat created?"},
    # Repeats
    {"type": "project", "q": "What did Rajat build and deploy at AWS account teamweave?"},
    {"type": "project", "q": "What services did Rajat implement to enable adaptive RAG routing?"},
    {"type": "project", "q": "What systems did Rajat create for document ingestion and preprocessing?"},
    {"type": "project", "q": "Describe the projects Rajat has worked on involving vector search."},
    {"type": "project", "q": "What end-to-end machine learning or AI projects has Rajat completed?"},

    # ── comparison (20 questions) ─────────────────────────────────────────────
    {"type": "comparison", "q": "How does Neptune Analytics compare to pgvector for GraphRAG use cases?"},
    {"type": "comparison", "q": "What is the difference between hierarchical and fixed-size chunking in RAG?"},
    {"type": "comparison", "q": "How does graph_first retrieval differ from semantic_search?"},
    {"type": "comparison", "q": "Compare the hybrid retrieval strategy versus keyword_boosted."},
    {"type": "comparison", "q": "Why was Neptune Analytics chosen over a standalone vector database?"},
    {"type": "comparison", "q": "How does AWS SAM compare to raw CloudFormation for serverless deployments?"},
    {"type": "comparison", "q": "What are the trade-offs between polling in Lambda vs using Step Functions?"},
    {"type": "comparison", "q": "How does OIDC-based deployment compare to long-lived AWS credentials in CI/CD?"},
    {"type": "comparison", "q": "Compare sentence chunking vs fixed-256 chunking for structured documents."},
    {"type": "comparison", "q": "What is the difference between Bedrock Knowledge Base retrieval and direct Neptune vector search?"},
    {"type": "comparison", "q": "How does Claude Haiku compare to larger models for document parsing during ingestion?"},
    {"type": "comparison", "q": "Compare a monolithic Lambda versus three single-responsibility functions as used here."},
    {"type": "comparison", "q": "What's the trade-off between confidence-weighted feedback (±0.05) vs larger update steps?"},
    {"type": "comparison", "q": "How does hierarchical RAG differ from naive flat RAG in terms of retrieval quality?"},
    {"type": "comparison", "q": "Compare Titan Text Embeddings V2 (1024-dim) versus other embedding models for expertise retrieval."},
    # Repeats
    {"type": "comparison", "q": "What is better for architecture questions: graph_first or hybrid strategy?"},
    {"type": "comparison", "q": "How does Neptune Analytics vector search compare to Bedrock Knowledge Base search?"},
    {"type": "comparison", "q": "What are the pros and cons of event-driven architecture vs request-response for document ingestion?"},
    {"type": "comparison", "q": "Compare SSE-KMS versus SSE-S3 for data at rest in this system."},
    {"type": "comparison", "q": "How does the adaptive routing approach compare to a static rule-based router?"},

    # ── credential (10 questions) ─────────────────────────────────────────────
    {"type": "credential", "q": "What AWS certifications does Rajat hold?"},
    {"type": "credential", "q": "What training or courses has Rajat completed related to cloud architecture?"},
    {"type": "credential", "q": "Does Rajat have any formal qualifications in machine learning or AI?"},
    {"type": "credential", "q": "What educational background does Rajat have?"},
    {"type": "credential", "q": "Has Rajat received any awards or recognition for his technical work?"},
    {"type": "credential", "q": "What certifications prove Rajat's AWS expertise?"},
    {"type": "credential", "q": "What degree or academic qualifications does Rajat have?"},
    {"type": "credential", "q": "Is Rajat certified in any cloud or DevOps disciplines?"},
    {"type": "credential", "q": "What formal training has Rajat done on serverless architecture?"},
    {"type": "credential", "q": "What professional qualifications back Rajat's AI and ML experience?"},

    # ── general (15 questions) ────────────────────────────────────────────────
    {"type": "general", "q": "Tell me about Rajat Arun's professional background."},
    {"type": "general", "q": "What is Rajat's primary area of expertise?"},
    {"type": "general", "q": "What does Rajat do professionally?"},
    {"type": "general", "q": "Give me a summary of Rajat's technical skills."},
    {"type": "general", "q": "What technologies is Rajat most familiar with?"},
    {"type": "general", "q": "What makes Rajat a strong candidate for a senior cloud architect role?"},
    {"type": "general", "q": "What is ContextWeave and who built it?"},
    {"type": "general", "q": "What kind of developer is Rajat?"},
    {"type": "general", "q": "What AWS managed services has Rajat used in his projects?"},
    {"type": "general", "q": "Can you summarize Rajat's experience with AI and machine learning?"},
    {"type": "general", "q": "What is the most impressive technical project Rajat has built?"},
    {"type": "general", "q": "What development practices does Rajat follow?"},
    {"type": "general", "q": "What is Rajat's experience with infrastructure as code?"},
    {"type": "general", "q": "What are Rajat's strongest technical competencies?"},
    {"type": "general", "q": "How would you describe Rajat's software engineering philosophy?"},

    # ── Tarun Raja – skill_depth (10 questions) ───────────────────────────────
    {"type": "skill_depth", "q": "How deep is Tarun's expertise with LangChain and LLM orchestration?"},
    {"type": "skill_depth", "q": "How proficient is Tarun with LangGraph for building agentic workflows?"},
    {"type": "skill_depth", "q": "What is Tarun's level of expertise with Kubernetes and container orchestration?"},
    {"type": "skill_depth", "q": "How skilled is Tarun with CI/CD pipeline design and implementation?"},
    {"type": "skill_depth", "q": "How deep is Tarun's knowledge of OpenAI APIs and AI integration patterns?"},
    {"type": "skill_depth", "q": "How experienced is Tarun with Java Spring Framework for full-stack development?"},
    {"type": "skill_depth", "q": "How strong is Tarun's Site Reliability Engineering (SRE) background?"},
    {"type": "skill_depth", "q": "How proficient is Tarun with Terraform for infrastructure as code?"},
    {"type": "skill_depth", "q": "How skilled is Tarun at engineering leadership and cross-functional team management?"},
    {"type": "skill_depth", "q": "How experienced is Tarun with banking regulatory compliance and governance for payment systems?"},

    # ── Tarun Raja – architecture (6 questions) ───────────────────────────────
    {"type": "architecture", "q": "How did Tarun architect the TaskWeave agentic AI framework using LangGraph and LangChain?"},
    {"type": "architecture", "q": "Describe the architecture of the Jules library Tarun built for shift-left testing in Kubernetes."},
    {"type": "architecture", "q": "What cloud-native migration architecture did Tarun lead at JP Morgan Chase during the modernization effort?"},
    {"type": "architecture", "q": "How does TaskWeave use JSON configuration to define atomic tasks across LLM prompts, API calls, and data analysis?"},
    {"type": "architecture", "q": "How did Tarun design the Commercial Banking Portal for middle-market clients at JP Morgan Chase?"},
    {"type": "architecture", "q": "What three workstreams did Tarun lead at JP Morgan Chase and how were they structured?"},

    # ── Tarun Raja – project (6 questions) ────────────────────────────────────
    {"type": "project", "q": "What is the TaskWeave project and what problem does it solve?"},
    {"type": "project", "q": "What is the Jules library that Tarun built at JP Morgan Chase and why was it created?"},
    {"type": "project", "q": "What agentic AI projects has Tarun shipped using OpenAI APIs and LangGraph?"},
    {"type": "project", "q": "What banking portal did Tarun deliver for middle-market clients at JP Morgan Chase?"},
    {"type": "project", "q": "What AI Enablement initiatives did Tarun lead at JP Morgan Chase?"},
    {"type": "project", "q": "What full-stack applications did Tarun build using JavaScript and Java Spring Framework?"},

    # ── Tarun Raja – comparison (4 questions) ─────────────────────────────────
    {"type": "comparison", "q": "How does Tarun's LangGraph experience compare to his LangChain usage in TaskWeave?"},
    {"type": "comparison", "q": "How does Tarun's VP Software Engineering Lead role differ from his Senior Lead Software Engineer role?"},
    {"type": "comparison", "q": "Compare Tarun's front-end development background with his current AI and agentic systems work."},
    {"type": "comparison", "q": "How does Tarun's approach to multi-agent workflows compare to traditional microservices orchestration?"},

    # ── Tarun Raja – credential (5 questions) ─────────────────────────────────
    {"type": "credential", "q": "What AWS certifications does Tarun hold?"},
    {"type": "credential", "q": "Is Tarun a Certified Kubernetes Application Developer (CKAD)?"},
    {"type": "credential", "q": "What is Tarun's educational background and graduate degree?"},
    {"type": "credential", "q": "Does Tarun have an MBA and from which institution?"},
    {"type": "credential", "q": "What is Tarun's GPA from his Master of Science in Computer Engineering?"},

    # ── Tarun Raja – general (4 questions) ────────────────────────────────────
    {"type": "general", "q": "Give me a summary of Tarun's professional background and career at JP Morgan Chase."},
    {"type": "general", "q": "How long has Tarun worked at JP Morgan Chase and what roles has he held there?"},
    {"type": "general", "q": "What makes Tarun a strong candidate for a senior AI engineering or leadership role?"},
    {"type": "general", "q": "What is Tarun's most impressive technical project and what was its impact?"},
]

assert len(QUESTIONS) >= 100, f"Need at least 100 questions, got {len(QUESTIONS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Weight mechanics – mirror rag_router.py constants so this script can
# simulate/predict the exact same updates that happen server-side.
# ─────────────────────────────────────────────────────────────────────────────
_REINFORCE_THRESHOLD = 0.70   # conf ≥ this → weight += REINFORCE_DELTA
_PENALISE_THRESHOLD  = 0.40   # conf <  this → weight += PENALISE_DELTA
_REINFORCE_DELTA     = +0.05
_PENALISE_DELTA      = -0.02
_WEIGHT_FLOOR        = 0.10
_WEIGHT_CEIL         = 1.00
# To break even: need REINFORCE_DELTA / abs(PENALISE_DELTA) = 2.5 reinforces
# for every 1 penalise just to stay at the same weight level.
_BREAK_EVEN_RATIO    = _REINFORCE_DELTA / abs(_PENALISE_DELTA)  # 2.5


def _feedback_action(confidence: float | None) -> str:
    """Map a confidence score to the weight-update action string."""
    if confidence is None:
        return "unknown"
    if confidence >= _REINFORCE_THRESHOLD:
        return "reinforce"
    if confidence < _PENALISE_THRESHOLD:
        return "penalise"
    return "neutral"


def _feedback_delta(confidence: float | None) -> float:
    """Return the numeric weight delta that the server will apply."""
    action = _feedback_action(confidence)
    if action == "reinforce":
        return _REINFORCE_DELTA
    if action == "penalise":
        return _PENALISE_DELTA
    return 0.0


@dataclass
class QuestionResult:
    index: int
    declared_type: str
    question: str
    classified_type: str | None = None
    routing_decision: str | None = None
    confidence: float | None = None
    # Feedback action inferred from confidence (mirrors rag_router.update_feedback)
    feedback_action: str | None = None   # "reinforce" | "penalise" | "neutral"
    feedback_delta: float = 0.0          # +0.05 / -0.02 / 0.00
    retrieval_count: int | None = None
    graph_entities_used: list[str] = field(default_factory=list)
    inferred_skills: list[str] = field(default_factory=list)
    repeated_patterns: list[str] = field(default_factory=list)
    latency_ms: int | None = None
    status: str = "pending"   # pending | ok | error
    error: str | None = None
    answer_preview: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

ENDPOINT_SUFFIX = "/query-expertise"

def _post(endpoint: str, question: str, top_k: int, timeout: int) -> dict[str, Any]:
    url = endpoint.rstrip("/") + ENDPOINT_SUFFIX
    payload = json.dumps(
        {"question": question, "topK": top_k, "includeGraphExpansion": True, "minConfidence": 0.3}
    ).encode()
    headers = {"Content-Type": "application/json"}
    t0 = time.perf_counter()
    if HAS_REQUESTS:
        resp = _requests.post(url, data=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    else:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    data["_elapsed_ms"] = elapsed_ms
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def run_question(
    index: int,
    entry: dict[str, str],
    endpoint: str,
    top_k: int,
    timeout: int,
    retries: int = 3,
) -> QuestionResult:
    result = QuestionResult(
        index=index,
        declared_type=entry["type"],
        question=entry["q"],
    )
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            data = _post(endpoint, entry["q"], top_k, timeout)
            result.classified_type  = data.get("questionType")
            result.routing_decision = data.get("routingDecision")
            result.confidence       = data.get("confidence")
            result.retrieval_count  = data.get("retrievalCount")
            result.graph_entities_used = data.get("graphEntitiesUsed", [])
            result.inferred_skills     = data.get("inferredSkills", [])
            result.repeated_patterns   = data.get("repeatedPatterns", [])
            result.latency_ms          = data.get("latencyMs") or data.get("_elapsed_ms")
            answer                     = data.get("answer", "")
            result.answer_preview      = answer[:120].replace("\n", " ")
            result.feedback_action     = _feedback_action(result.confidence)
            result.feedback_delta      = _feedback_delta(result.confidence)
            result.status              = "ok"
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = 2 ** attempt
                time.sleep(wait)
    result.status = "error"
    result.error  = str(last_exc)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

_FB_SYMBOL = {"reinforce": "↑+.05", "penalise": "↓-.02", "neutral": "→   ", "unknown": "?    "}

_W_IDX    = 4
_W_DECL   = 12
_W_CLS    = 12
_W_ROUTE  = 16
_W_CONF   = 6
_W_FB     = 6
_W_LAT    = 7
_W_STATUS = 6
COL_W = {"idx": _W_IDX, "decl": _W_DECL, "cls": _W_CLS, "route": _W_ROUTE,
         "conf": _W_CONF, "fb": _W_FB, "lat": _W_LAT, "status": _W_STATUS}

# Use string methods (.ljust/.rjust) rather than f-string format specs so
# this works correctly on all Python versions and regardless of field type.
HEADER = (
    "#".rjust(_W_IDX) + "  " +
    "Declared".ljust(_W_DECL) + "  " +
    "Classified".ljust(_W_CLS) + "  " +
    "Route".ljust(_W_ROUTE) + "  " +
    "Conf".rjust(_W_CONF) + "  " +
    "Wt".ljust(_W_FB) + "  " +
    "Lat(ms)".rjust(_W_LAT) + "  " +
    "St".ljust(_W_STATUS) + "  " +
    "Answer preview"
)
SEP = "-" * min(len(HEADER) + 20, 145)


def _row(r: QuestionResult) -> str:
    conf_str = f"{r.confidence:.2f}" if isinstance(r.confidence, (int, float)) else "  -- "
    lat_str  = str(r.latency_ms) if r.latency_ms is not None else "   -- "
    status   = "OK" if r.status == "ok" else "ERR"
    route    = str(r.routing_decision) if r.routing_decision else "--"
    cls      = str(r.classified_type)  if r.classified_type  else "--"
    fb_key   = str(r.feedback_action)  if r.feedback_action  else "unknown"
    fb       = _FB_SYMBOL.get(fb_key, "?    ")
    preview  = str(r.error)[:80] if r.status == "error" else str(r.answer_preview)
    return (
        str(r.index).rjust(_W_IDX) + "  " +
        str(r.declared_type).ljust(_W_DECL) + "  " +
        cls.ljust(_W_CLS) + "  " +
        route.ljust(_W_ROUTE) + "  " +
        conf_str.rjust(_W_CONF) + "  " +
        fb.ljust(_W_FB) + "  " +
        lat_str.rjust(_W_LAT) + "  " +
        status.ljust(_W_STATUS) + "  " +
        preview
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100*n/total:.1f}%" if total else "  0%"


def _weight_bar(delta: float, width: int = 20) -> str:
    """ASCII bar showing net weight movement magnitude and direction."""
    norm = max(-1.0, min(1.0, delta))   # clamp to [-1, 1]
    half = width // 2
    pos = int(abs(norm) * half)
    if delta >= 0:
        bar = " " * half + "+" * pos + " " * (half - pos)
    else:
        bar = " " * (half - pos) + "-" * pos + " " * half
    return f"[{bar}]"


def print_summary(results: list[QuestionResult]) -> None:
    ok     = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status == "error"]
    total  = len(results)

    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)
    print(f"  Total questions : {total}")
    print(f"  Succeeded       : {len(ok)}  ({_pct(len(ok), total)})")
    print(f"  Failed          : {len(errors)}")

    if not ok:
        return

    # ── Feedback action breakdown ─────────────────────────────────────────────
    n_reinforce = sum(1 for r in ok if r.feedback_action == "reinforce")
    n_penalise  = sum(1 for r in ok if r.feedback_action == "penalise")
    n_neutral   = sum(1 for r in ok if r.feedback_action == "neutral")
    n_ok        = len(ok)

    total_delta = sum(r.feedback_delta for r in ok)

    print(f"\n  ── Weight feedback actions (adversarial pressure analysis) ──")
    print(f"    ↑ Reinforce (conf ≥ {_REINFORCE_THRESHOLD:.2f})  : "
          f"{n_reinforce:>3}  ({_pct(n_reinforce, n_ok)})  "
          f"net +{n_reinforce * _REINFORCE_DELTA:.2f}")
    print(f"    ↓ Penalise  (conf <  {_PENALISE_THRESHOLD:.2f})  : "
          f"{n_penalise:>3}  ({_pct(n_penalise, n_ok)})  "
          f"net {n_penalise * _PENALISE_DELTA:.2f}")
    print(f"    → Neutral                   : "
          f"{n_neutral:>3}  ({_pct(n_neutral, n_ok)})  no change")
    print(f"    ──────────────────────────────────────────────")
    sign = "+" if total_delta >= 0 else ""
    print(f"    Net estimated weight delta  : {sign}{total_delta:.3f}  "
          f"{_weight_bar(total_delta / max(n_ok, 1) * 10)}")

    # Break-even warning
    if n_penalise > 0:
        actual_ratio = n_reinforce / n_penalise if n_penalise else float("inf")
        status_str = "OK" if actual_ratio >= _BREAK_EVEN_RATIO else "BELOW BREAK-EVEN"
        print(f"\n    Break-even ratio required   : {_BREAK_EVEN_RATIO:.1f}× reinforces per penalise")
        print(f"    Actual ratio                : {actual_ratio:.1f}×  [{status_str}]")
        if actual_ratio < _BREAK_EVEN_RATIO:
            print(f"    WARNING  Adversarial controls are eroding weights faster than they")
            print(f"             are being built. The system likely has insufficient indexed")
            print(f"             data to synthesise high-confidence answers. Run /ingest first.")

    # ── Per (strategy, question_type) net delta ───────────────────────────────
    print(f"\n  ── Estimated net weight delta per (strategy, question_type) ──")
    print(f"    {'Strategy':<20}  {'Q-Type':<14}  {'↑':>3}  {'↓':>3}  {'→':>3}  {'Net Δ':>7}  Status")
    print(f"    {'─'*20}  {'─'*14}  {'─'*3}  {'─'*3}  {'─'*3}  {'─'*7}  ──────")

    pair_stats: dict[tuple[str,str], dict] = defaultdict(lambda: {"r": 0, "p": 0, "n": 0, "delta": 0.0})
    for r in ok:
        key = (r.routing_decision or "unknown", r.classified_type or r.declared_type)
        ps = pair_stats[key]
        if r.feedback_action == "reinforce":
            ps["r"] += 1
        elif r.feedback_action == "penalise":
            ps["p"] += 1
        else:
            ps["n"] += 1
        ps["delta"] += r.feedback_delta

    erosion_pairs: list[tuple[str,str]] = []
    for (strategy, qtype), ps in sorted(pair_stats.items()):
        delta = ps["delta"]
        sign  = "+" if delta >= 0 else ""
        ratio = ps["r"] / ps["p"] if ps["p"] else float("inf")
        if delta < 0:
            flag = "ERODING"
            erosion_pairs.append((strategy, qtype))
        elif ps["p"] > 0 and ratio < _BREAK_EVEN_RATIO:
            flag = "at-risk"
        else:
            flag = ""
        print(f"    {strategy:<20}  {qtype:<14}  "
              f"{ps['r']:>3}  {ps['p']:>3}  {ps['n']:>3}  "
              f"{sign}{delta:>6.3f}  {flag}")

    # ── Confidence stats ──────────────────────────────────────────────────────
    confs = [r.confidence for r in ok if r.confidence is not None]
    if confs:
        avg_conf = sum(confs) / len(confs)
        below40  = sum(1 for c in confs if c < _PENALISE_THRESHOLD)
        above70  = sum(1 for c in confs if c >= _REINFORCE_THRESHOLD)
        print(f"\n  ── Confidence distribution ──")
        print(f"    avg={avg_conf:.3f}  min={min(confs):.3f}  max={max(confs):.3f}")
        print(f"    < {_PENALISE_THRESHOLD:.2f} (penalise zone)  : "
              f"{below40:>3}  ({_pct(below40, len(confs))})")
        print(f"    ≥ {_REINFORCE_THRESHOLD:.2f} (reinforce zone) : "
              f"{above70:>3}  ({_pct(above70, len(confs))})")

    # ── Latency stats ─────────────────────────────────────────────────────────
    lats = [r.latency_ms for r in ok if r.latency_ms is not None]
    if lats:
        avg_lat = sum(lats) / len(lats)
        p95_lat = sorted(lats)[int(len(lats) * 0.95)]
        print(f"\n  ── Latency (ms) ──")
        print(f"    avg={avg_lat:.0f}  p95={p95_lat}  min={min(lats)}  max={max(lats)}")

    # ── Classifier accuracy ───────────────────────────────────────────────────
    print("\n  ── Classifier accuracy by declared type ──")
    by_decl: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in ok:
        by_decl[r.declared_type].append(r)
    for dtype in sorted(by_decl):
        group   = by_decl[dtype]
        correct = sum(1 for r in group if r.classified_type == dtype)
        print(f"    {dtype:<14}  {correct:>2}/{len(group):>2}  ({_pct(correct, len(group))})")

    # ── Routing distribution ──────────────────────────────────────────────────
    print("\n  ── Routing strategy distribution ──")
    route_counts: dict[str, int] = defaultdict(int)
    for r in ok:
        route_counts[r.routing_decision or "unknown"] += 1
    for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
        print(f"    {route:<20}  {count:>3}  ({_pct(count, len(ok))})")

    # ── Avg confidence by routing strategy ───────────────────────────────────
    print("\n  ── Avg confidence + feedback pressure by routing strategy ──")
    route_confs: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        if r.confidence is not None:
            route_confs[r.routing_decision or "unknown"].append(r.confidence)
    for route, confs in sorted(route_confs.items()):
        avg = sum(confs) / len(confs)
        n_r = sum(1 for c in confs if c >= _REINFORCE_THRESHOLD)
        n_p = sum(1 for c in confs if c < _PENALISE_THRESHOLD)
        ratio_str = f"{n_r/n_p:.1f}×" if n_p else "∞"
        flag = "" if n_p == 0 or (n_r/n_p) >= _BREAK_EVEN_RATIO else "  ← adversarial pressure"
        print(f"    {route:<20}  avg={avg:.3f}  ↑{n_r} ↓{n_p}  ratio={ratio_str}{flag}")

    # ── Type → route cross-tab ────────────────────────────────────────────────
    print("\n  ── Question type → routing strategy cross-tab ──")
    type_route: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in ok:
        type_route[r.declared_type][r.routing_decision or "unknown"] += 1
    for dtype in sorted(type_route):
        routes = ", ".join(
            f"{rte}×{cnt}" for rte, cnt in sorted(type_route[dtype].items(), key=lambda x: -x[1])
        )
        print(f"    {dtype:<14}  →  {routes}")

    # ── Graph expansion ───────────────────────────────────────────────────────
    graph_users = [r for r in ok if r.graph_entities_used]
    if graph_users:
        avg_entities = sum(len(r.graph_entities_used) for r in graph_users) / len(graph_users)
        print(f"\n  Graph expansion used in {len(graph_users)}/{len(ok)} queries "
              f"(avg {avg_entities:.1f} entities/query)")

    # ── Erosion summary ───────────────────────────────────────────────────────
    if erosion_pairs:
        print(f"\n  !! WEIGHT EROSION WARNING  ({len(erosion_pairs)} strategy/type pair(s)) !!")
        for strategy, qtype in erosion_pairs:
            ps = pair_stats[(strategy, qtype)]
            print(f"    {strategy} × {qtype}  "
                  f"net={ps['delta']:+.3f}  "
                  f"↑{ps['r']} ↓{ps['p']}  "
                  f"need {int(ps['p'] * _BREAK_EVEN_RATIO + 1)} more reinforces to recover")
        print(f"\n    Recommendation: ensure documents are fully ingested before running")
        print(f"    the experiment, or run /ingest and re-seed the routing graph, then")
        print(f"    run this experiment again with --resume to re-test eroded pairs.")

    # ── Top repeated patterns ─────────────────────────────────────────────────
    all_patterns: list[str] = []
    for r in ok:
        all_patterns.extend(r.repeated_patterns)
    if all_patterns:
        pattern_counts: dict[str, int] = defaultdict(int)
        for p in all_patterns:
            pattern_counts[p] += 1
        top5 = sorted(pattern_counts.items(), key=lambda x: -x[1])[:5]
        print("\n  ── Top repeated patterns observed ──")
        for pat, cnt in top5:
            print(f"    {pat:<40}  {cnt}×")

    # ── Errors ────────────────────────────────────────────────────────────────
    if errors:
        print(f"\n  ── Errors ({len(errors)}) ──")
        for r in errors:
            print(f"    [{r.index}] {r.question[:60]}…  →  {r.error}")

    print("=" * 75)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_FILE = Path("experiment_checkpoint.json")


def _load_checkpoint() -> dict[int, QuestionResult]:
    if not CHECKPOINT_FILE.exists():
        return {}
    with CHECKPOINT_FILE.open() as f:
        raw: list[dict] = json.load(f)
    out: dict[int, QuestionResult] = {}
    for d in raw:
        r = QuestionResult(**{k: v for k, v in d.items() if k in QuestionResult.__dataclass_fields__})
        out[r.index] = r
    return out


def _save_checkpoint(done: dict[int, QuestionResult]) -> None:
    with CHECKPOINT_FILE.open("w") as f:
        json.dump([asdict(r) for r in done.values()], f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExpertiseRAG adaptive routing experiment – sends 100+ questions to the API"
    )
    parser.add_argument(
        "--endpoint", "-e",
        default="https://wnbdgd0z4g.execute-api.us-east-1.amazonaws.com/prod",
        help="API Gateway base URL (default: prod endpoint)",
    )
    parser.add_argument("--top-k", "-n", type=int, default=8, help="topK per request (default: 8)")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout per request in seconds (default: 90)")
    parser.add_argument("--concurrency", "-c", type=int, default=1,
                        help="Number of parallel workers (default: 1 – sequential for clean weight accumulation)")
    parser.add_argument("--types", nargs="*",
                        choices=["skill_depth", "architecture", "project", "comparison", "credential", "general"],
                        help="Only run questions of these types")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already in experiment_checkpoint.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print questions without sending any requests")
    parser.add_argument("--output", "-o", default=None,
                        help="JSON output file (default: experiment_results_<timestamp>.json)")
    args = parser.parse_args()

    # Filter questions
    questions = QUESTIONS
    if args.types:
        questions = [q for q in questions if q["type"] in args.types]
    if not questions:
        print("No questions match the requested types.")
        sys.exit(0)

    # Load checkpoint
    done: dict[int, QuestionResult] = {}
    if args.resume:
        done = _load_checkpoint()
        skipped = len([i for i in range(len(questions)) if i in done])
        if skipped:
            print(f"Resuming: {skipped} questions already completed, {len(questions)-skipped} remaining.\n")

    if args.dry_run:
        print(f"DRY RUN – {len(questions)} questions:\n")
        for i, q in enumerate(questions):
            print(f"  [{i+1:>3}] [{q['type']:<14}] {q['q']}")
        sys.exit(0)

    # Output file
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output or f"experiment_results_{ts}.json"

    print(f"\n{'='*70}")
    print(f"  ExpertiseRAG Routing Experiment")
    print(f"  Endpoint    : {args.endpoint}")
    print(f"  Questions   : {len(questions)}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Top-K       : {args.top_k}")
    print(f"  Output      : {output_file}")
    print(f"{'='*70}\n")
    print(HEADER)
    print(SEP)

    results_order: list[int] = []  # preserve insertion order for output

    # --------------------------------------------------------------------------
    # Sequential path (default, cleanest for weight accumulation)
    # --------------------------------------------------------------------------
    if args.concurrency == 1:
        for i, entry in enumerate(questions):
            if i in done:
                r = done[i]
                print(_row(r))
                results_order.append(i)
                continue
            r = run_question(i, entry, args.endpoint, args.top_k, args.timeout)
            done[i] = r
            results_order.append(i)
            print(_row(r), flush=True)
            _save_checkpoint(done)
            # Running adversarial-pressure check every 10 questions
            completed = [v for v in done.values() if v.status == "ok"]
            if len(completed) % 10 == 0 and len(completed) >= 10:
                n_r = sum(1 for x in completed if x.feedback_action == "reinforce")
                n_p = sum(1 for x in completed if x.feedback_action == "penalise")
                net  = sum(x.feedback_delta for x in completed)
                sign = "+" if net >= 0 else ""
                ratio = f"{n_r/n_p:.1f}x" if n_p else "∞"
                flag  = "  << BELOW BREAK-EVEN" if n_p > 0 and (n_r/n_p) < _BREAK_EVEN_RATIO else ""
                print(f"  [weight-check @{len(completed)}]  "
                      f"↑{n_r} ↓{n_p} →{len(completed)-n_r-n_p}  "
                      f"net={sign}{net:.2f}  ratio={ratio}{flag}", flush=True)

    # --------------------------------------------------------------------------
    # Parallel path (faster but weight updates may race)
    # --------------------------------------------------------------------------
    else:
        pending = [(i, entry) for i, entry in enumerate(questions) if i not in done]
        completed_so_far = {i: done[i] for i in done}

        # Print already-done rows first
        for i in sorted(done):
            print(_row(done[i]))
            results_order.append(i)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_question, i, entry, args.endpoint, args.top_k, args.timeout): i
                for i, entry in pending
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                r = fut.result()
                done[idx] = r
                results_order.append(idx)
                print(_row(r), flush=True)
                _save_checkpoint(done)

    # Build ordered results list
    all_results = [done[i] for i in sorted(done.keys())]

    # Print summary
    print(SEP)
    print_summary(all_results)

    # Save JSON output
    output = {
        "experiment": {
            "timestamp": ts,
            "endpoint": args.endpoint,
            "top_k": args.top_k,
            "total_questions": len(questions),
            "concurrency": args.concurrency,
        },
        "results": [asdict(r) for r in all_results],
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {output_file}")

    # Clean up checkpoint on success
    if CHECKPOINT_FILE.exists() and all(r.status == "ok" for r in all_results):
        CHECKPOINT_FILE.unlink()
        print("Checkpoint file removed (all questions completed successfully).")


if __name__ == "__main__":
    main()
