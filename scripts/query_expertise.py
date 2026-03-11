#!/usr/bin/env python3
"""
ExpertiseRAG – Query Expertise API (standalone boto3 + HTTP script)

Two modes:
  1. Direct Bedrock Retrieve (bypasses API Gateway, useful for debugging)
  2. HTTP POST to the deployed API Gateway endpoint

Usage:
  # Direct Bedrock Retrieve
  python scripts/query_expertise.py bedrock \\
      --knowledge-base-id <KB_ID> \\
      --question "What AWS services has this developer built production systems with?" \\
      --top-k 10

  # HTTP API
  python scripts/query_expertise.py api \\
      --endpoint https://<id>.execute-api.us-east-1.amazonaws.com/dev \\
      --question "What patterns does this developer use?" \\
      --top-k 8

Prerequisites:
    pip install boto3 requests
    aws configure
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error


# ─────────────────────────────────────────────────────────────────────────────
# Direct Bedrock Retrieve
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_WEIGHTS = {
    "architecture.md": 1.0,
    "CLAUDE.md": 1.0,
    "plantuml_derived": 0.8,
    "c4_diagram": 0.8,
    "README.md": 0.6,
    "readme.md": 0.6,
    "repo-signals.yaml": 0.7,
    "article": 0.5,
    "resume": 0.3,
}


def _get_source_weight(uri: str) -> float:
    lower = uri.lower()
    for key, w in SOURCE_WEIGHTS.items():
        if key.lower() in lower:
            return w
    return 0.4


def bedrock_retrieve(
    knowledge_base_id: str,
    question: str,
    top_k: int = 10,
    region: str = "us-east-1",
    profile: str | None = None,
) -> dict[str, Any]:
    """
    Call Bedrock Retrieve directly via boto3.

    This is the same API called inside QueryAPIFunction's retriever.py.
    Useful for debugging retrieval quality without the full API stack.
    """
    session = boto3.Session(region_name=region, profile_name=profile)
    client = session.client("bedrock-agent-runtime")

    print(f"Querying Bedrock Knowledge Base: {knowledge_base_id}")
    print(f"Question: {question}")
    print(f"Top-K: {top_k}\n")

    try:
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "HYBRID",
                }
            },
        )
    except ClientError as exc:
        return {"error": str(exc)}

    results = response.get("retrievalResults", [])
    chunks = []

    for i, result in enumerate(results, 1):
        content = result.get("content", {}).get("text", "")
        score = float(result.get("score", 0.0))
        location = result.get("location", {})
        source_uri = (
            location.get("s3Location", {}).get("uri", "")
            or location.get("uri", "")
        )
        source_file = source_uri.split("/")[-1] if source_uri else "unknown"
        source_weight = _get_source_weight(source_file)
        effective_score = score * source_weight

        chunks.append({
            "rank": i,
            "source_file": source_file,
            "source_uri": source_uri,
            "score": round(score, 4),
            "source_weight": source_weight,
            "effective_score": round(effective_score, 4),
            "content_preview": content[:300] + ("..." if len(content) > 300 else ""),
        })

    chunks.sort(key=lambda c: c["effective_score"], reverse=True)

    # Re-rank by effective score
    for i, c in enumerate(chunks, 1):
        c["rank"] = i

    return {
        "question": question,
        "knowledge_base_id": knowledge_base_id,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "note": (
            "To synthesize an answer, set GENERATION_MODEL_ID in your Lambda "
            "and POST to /query-expertise, or extend this script to call "
            "bedrock-runtime converse() with the retrieved chunks."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTTP API call
# ─────────────────────────────────────────────────────────────────────────────

def api_query(
    endpoint: str,
    question: str,
    top_k: int = 10,
    include_graph_expansion: bool = True,
    min_confidence: float = 0.3,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    POST to the deployed /query-expertise endpoint.
    """
    url = endpoint.rstrip("/") + "/query-expertise"
    payload = json.dumps({
        "question": question,
        "topK": top_k,
        "includeGraphExpansion": include_graph_expansion,
        "minConfidence": min_confidence,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    print(f"POST {url}")
    print(f"Question: {question}\n")

    if HAS_REQUESTS:
        resp = _requests.post(url, data=payload, headers=headers, timeout=60)
        return resp.json()
    else:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return {"error": str(exc), "body": exc.read().decode()}


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_retrieve_result(result: dict) -> None:
    if "error" in result:
        print(f"✗ Error: {result['error']}")
        return
    print(f"Retrieved {result['chunk_count']} chunks:\n")
    for chunk in result["chunks"]:
        print(
            f"  [{chunk['rank']}] {chunk['source_file']}"
            f" | score={chunk['score']:.4f} × weight={chunk['source_weight']}"
            f" = effective={chunk['effective_score']:.4f}"
        )
        print(f"       {chunk['content_preview'][:120]}")
        print()
    print(result.get("note", ""))


def print_api_result(result: dict) -> None:
    if "error" in result:
        print(f"✗ Error: {result['error']}")
        return

    print(f"Answer (confidence={result.get('confidence', 0):.2f}, "
          f"type={result.get('questionType', '?')}, "
          f"latency={result.get('latencyMs', '?')}ms):\n")
    print(result.get("answer", "No answer returned"))

    sources = result.get("sources", [])
    if sources:
        print(f"\nSources ({len(sources)}):")
        for s in sources:
            file_ = s.get("file") or s.get("sourceUri", "").split("/")[-1]
            score = s.get("effectiveScore") or s.get("score", 0)
            print(f"  - {file_} (score={score:.3f})")

    skills = result.get("inferredSkills", [])
    if skills:
        print(f"\nInferred skills: {', '.join(skills)}")

    patterns = result.get("repeatedPatterns", [])
    if patterns:
        print(f"Repeated patterns: {', '.join(patterns)}")

    entities = result.get("graphEntitiesUsed", [])
    if entities:
        print(f"Graph entities used: {', '.join(entities)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the ExpertiseRAG system (Bedrock Retrieve or HTTP API)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # bedrock subcommand
    bedrock_p = subparsers.add_parser("bedrock", help="Direct Bedrock Retrieve (no synthesis)")
    bedrock_p.add_argument("--knowledge-base-id", "-k", required=True)
    bedrock_p.add_argument("--question", "-q", required=True)
    bedrock_p.add_argument("--top-k", "-n", type=int, default=10)
    bedrock_p.add_argument("--region", "-r", default="us-east-1")
    bedrock_p.add_argument("--profile", "-p", default=None)
    bedrock_p.add_argument("--json", "-j", action="store_true", help="Output raw JSON")

    # api subcommand
    api_p = subparsers.add_parser("api", help="POST to deployed API endpoint")
    api_p.add_argument("--endpoint", "-e", required=True,
                       help="API endpoint URL (e.g. https://xxx.execute-api.us-east-1.amazonaws.com/dev)")
    api_p.add_argument("--question", "-q", required=True)
    api_p.add_argument("--top-k", "-n", type=int, default=10)
    api_p.add_argument("--no-graph", action="store_true", help="Disable graph expansion")
    api_p.add_argument("--api-key", default=None, help="Optional API key")
    api_p.add_argument("--min-confidence", type=float, default=0.3)
    api_p.add_argument("--json", "-j", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if args.mode == "bedrock":
        result = bedrock_retrieve(
            knowledge_base_id=args.knowledge_base_id,
            question=args.question,
            top_k=args.top_k,
            region=args.region,
            profile=args.profile,
        )
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            print_retrieve_result(result)

    elif args.mode == "api":
        result = api_query(
            endpoint=args.endpoint,
            question=args.question,
            top_k=args.top_k,
            include_graph_expansion=not args.no_graph,
            min_confidence=args.min_confidence,
            api_key=args.api_key,
        )
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            print_api_result(result)


if __name__ == "__main__":
    main()
