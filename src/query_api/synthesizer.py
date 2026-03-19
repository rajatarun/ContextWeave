"""
Answer synthesizer for ExpertiseRAG.

Constructs a grounded prompt from:
  - Retrieved Bedrock KB chunks (sorted by effective score)
  - Neptune Analytics graph expansion context
  - Question classification
Then calls a Bedrock foundation model to generate a structured answer.

TODO: Select and configure the generation model below.
      Recommended options (as of 2025):
        - anthropic.claude-3-5-sonnet-20241022-v2:0   (best quality)
        - anthropic.claude-3-haiku-20240307-v1:0      (low latency / cost)
        - amazon.nova-pro-v1:0                        (AWS-native)
      Replace TODO_GENERATION_MODEL_ID with your chosen model ARN or ID.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import boto3
from botocore.exceptions import ClientError

from models import QueryResponse, RetrievedChunk

logger = logging.getLogger(__name__)

# TODO: Replace with your chosen generation model ID
#   e.g. "anthropic.claude-3-5-sonnet-20241022-v2:0"
#   or set GENERATION_MODEL_ID env var at deploy time
GENERATION_MODEL_ID = os.environ.get(
    "GENERATION_MODEL_ID",
    "TODO_GENERATION_MODEL_ID",
)

_BEDROCK_RUNTIME: Any = None


def _get_bedrock_runtime() -> Any:
    global _BEDROCK_RUNTIME
    if _BEDROCK_RUNTIME is None:
        _BEDROCK_RUNTIME = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _BEDROCK_RUNTIME


# ─────────────────────────────────────────────────────────────────────────────
# Question classification
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_PATTERNS = {
    "skill_depth": re.compile(
        # "experience with/in/using/of X" is a skill-depth question;
        # bare "experience" (e.g. "experience at JP Morgan") is NOT – it falls
        # through to the project classifier below.
        r"\b(expert|proficient|skill|know|familiar|deep|"
        r"strong|level|years?|how well|how long|speciali[sz]e)\b"
        r"|experience\s+(?:with|in|using|of|building)\b",
        re.IGNORECASE,
    ),
    "architecture": re.compile(
        r"\b(architect|design|pattern|system|infrastructure|cloud|aws|"
        r"microservice|serverless|event.driven|scalab|distribut)\b",
        re.IGNORECASE,
    ),
    "project": re.compile(
        # "experience" without a technology qualifier → employment/project context
        r"\b(project|built|created|deployed|implemented|work(?:ed)?|repo|product|"
        r"deliver|ship|application|service|platform|experience|role|responsibilit|"
        r"company|employe|position|tenure|joined)\b",
        re.IGNORECASE,
    ),
    "comparison": re.compile(
        r"\b(vs|versus|compare|difference|prefer|better|trade.?off|"
        r"choose|over|instead|rather)\b",
        re.IGNORECASE,
    ),
    "credential": re.compile(
        r"\b(certif|course|training|education|degree|qualified|background|"
        r"award|recognition)\b",
        re.IGNORECASE,
    ),
}


def classify_question(question: str) -> str:
    """
    Classify the question into one of:
      skill_depth | architecture | project | comparison | credential | general
    Returns the highest-priority matching type, or 'general'.
    """
    for qtype, pattern in _QUESTION_PATTERNS.items():
        if pattern.search(question):
            return qtype
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI system answering detailed questions about a developer's professional expertise.

Your answers must be:
1. GROUNDED: Only use information explicitly present in the provided evidence. Do not fabricate.
2. INFERENCE-AWARE: You may draw reasonable inferences from repeated patterns and co-occurring signals, but must label inferences as such.
3. EVIDENCE-BACKED: Cite the source document/file for each significant claim.
4. STRUCTURED: Return your answer as valid JSON matching the schema below.
5. HONEST: If the evidence is insufficient, say so clearly.

Weight evidence sources in this order (highest first):
  - architecture.md, CLAUDE.md (authoritative architecture decisions)
  - PlantUML-derived architecture summaries
  - code, README files
  - articles and blog posts
  - resume (supporting only, lower authority)

Prefer repeated implementation patterns over single mentions.

Response schema (return ONLY valid JSON, no markdown wrapper):
{
  "answer": "<comprehensive prose answer>",
  "sources": [{"file": "<filename>", "excerpt": "<relevant quote>", "weight": <0-1>}],
  "inferred_skills": ["<skill1>", ...],
  "repeated_patterns": ["<pattern1>", ...],
  "confidence": <0.0-1.0>,
  "reasoning_notes": "<brief explanation of evidence quality and any gaps>"
}"""


def _build_evidence_block(
    chunks: list[RetrievedChunk],
    graph_context: dict[str, Any],
    max_chars: int = 8000,
) -> str:
    """Assemble evidence from chunks + graph into a numbered evidence block."""
    lines = ["=== RETRIEVED EVIDENCE ===\n"]
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        if total_chars >= max_chars:
            break
        source = chunk.source_uri.split("/")[-1] if chunk.source_uri else "unknown"
        weight_label = (
            "HIGH" if chunk.source_weight >= 0.8
            else "MEDIUM" if chunk.source_weight >= 0.5
            else "LOW"
        )
        block = (
            f"[{i}] Source: {source} (authority={weight_label}, "
            f"score={chunk.effective_score:.3f})\n"
            f"{chunk.content.strip()}\n"
        )
        lines.append(block)
        total_chars += len(block)

    # Add graph context summary
    if graph_context.get("person_summary"):
        ps = graph_context["person_summary"]
        lines.append("\n=== GRAPH CONTEXT ===")
        if ps.get("skills"):
            lines.append(f"Demonstrated skills: {', '.join(ps['skills'][:15])}")
        if ps.get("patterns"):
            lines.append(f"Architecture patterns: {', '.join(ps['patterns'][:10])}")
        if ps.get("aws_services"):
            lines.append(f"AWS services used: {', '.join(ps['aws_services'][:15])}")

    if graph_context.get("repeated_patterns"):
        lines.append(
            f"Repeated patterns (cross-repo evidence): "
            f"{', '.join(graph_context['repeated_patterns'])}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_answer(
    question: str,
    chunks: list[RetrievedChunk],
    graph_context: dict[str, Any],
    question_type: str = "general",
    model_id: str | None = None,
) -> QueryResponse:
    """
    Synthesize a grounded, evidence-backed answer using Bedrock.

    If GENERATION_MODEL_ID is not configured, returns a mock response
    with the retrieved evidence for debugging.
    """
    effective_model = model_id or GENERATION_MODEL_ID

    if effective_model == "TODO_GENERATION_MODEL_ID":
        logger.warning(
            "GENERATION_MODEL_ID not set – returning raw evidence (no LLM synthesis). "
            "Set the GENERATION_MODEL_ID environment variable or Lambda parameter."
        )
        return _build_mock_response(question, chunks, graph_context, question_type)

    evidence_block = _build_evidence_block(chunks, graph_context)
    user_message = (
        f"Question type: {question_type}\n\n"
        f"{evidence_block}\n\n"
        f"=== QUESTION ===\n{question}"
    )

    # Build Bedrock Converse API request (works for all Claude + Nova models)
    request_body = {
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "inferenceConfig": {
            "maxTokens": 2048,
            "temperature": 0.1,      # Low temperature for factual grounding
            "topP": 0.9,
        },
    }

    client = _get_bedrock_runtime()
    try:
        response = client.converse(
            modelId=effective_model,
            **request_body,
        )
    except ClientError as exc:
        logger.error("Bedrock Converse failed: %s", exc)
        raise

    # Extract text response
    output_message = response.get("output", {}).get("message", {})
    raw_text = ""
    for content_block in output_message.get("content", []):
        if content_block.get("type") == "text" or "text" in content_block:
            raw_text = content_block.get("text", "")
            break

    # Parse the JSON response from the model
    parsed = _parse_model_response(raw_text)

    # Merge graph-derived signals into the parsed response
    inferred_skills = list(set(
        parsed.get("inferred_skills", [])
        + graph_context.get("inferred_skills", [])
    ))[:15]
    repeated_patterns = list(set(
        parsed.get("repeated_patterns", [])
        + graph_context.get("repeated_patterns", [])
    ))[:10]

    return QueryResponse(
        answer=parsed.get("answer", raw_text),
        sources=parsed.get("sources", [c.to_dict() for c in chunks[:5]]),
        inferred_skills=inferred_skills,
        repeated_patterns=repeated_patterns,
        confidence=float(parsed.get("confidence", 0.7)),
        question_type=question_type,
        graph_entities_used=graph_context.get("graph_entities_used", []),
        retrieval_count=len(chunks),
        model_id=effective_model,
    )


def _parse_model_response(raw_text: str) -> dict:
    """Extract JSON from model output, handling markdown code fences."""
    text = raw_text.strip()
    # Strip markdown code fence if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse model JSON response; returning raw text")
        return {"answer": raw_text, "sources": [], "confidence": 0.5}


def _build_mock_response(
    question: str,
    chunks: list[RetrievedChunk],
    graph_context: dict[str, Any],
    question_type: str,
) -> QueryResponse:
    """
    Fallback when no generation model is configured.
    Returns the top retrieved evidence as the 'answer' for debugging.
    """
    top_chunks = chunks[:3]
    answer_parts = [
        f"[DEBUG MODE – set GENERATION_MODEL_ID for LLM synthesis]\n",
        f"Question: {question}\n",
        "Top retrieved evidence:\n",
    ]
    for i, c in enumerate(top_chunks, 1):
        src = c.source_uri.split("/")[-1]
        answer_parts.append(f"{i}. [{src}] {c.content[:300]}...")

    return QueryResponse(
        answer="\n".join(answer_parts),
        sources=[c.to_dict() for c in top_chunks],
        inferred_skills=graph_context.get("inferred_skills", []),
        repeated_patterns=graph_context.get("repeated_patterns", []),
        confidence=0.0,
        question_type=question_type,
        graph_entities_used=graph_context.get("graph_entities_used", []),
        retrieval_count=len(chunks),
        model_id="none",
    )
