"""
Neptune Analytics graph expansion for ExpertiseRAG.

When Bedrock Retrieve returns evidence chunks, this module uses the
Neptune Analytics openCypher query API to expand the graph neighbourhood
around referenced entities, surfacing:
  - Co-occurring skills / technologies
  - Repeated implementation patterns
  - AWS services used in the same repositories
  - Supporting claims and evidence nodes
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_NEPTUNE_CLIENT: Any = None


def _get_neptune_client() -> Any:
    global _NEPTUNE_CLIENT
    if _NEPTUNE_CLIENT is None:
        _NEPTUNE_CLIENT = boto3.client(
            "neptune-graph",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _NEPTUNE_CLIENT


NEPTUNE_GRAPH_ID = os.environ.get("NEPTUNE_GRAPH_ID", "")


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_query(graph_id: str, query: str, parameters: dict | None = None) -> list[dict]:
    """Execute an openCypher query against Neptune Analytics and return rows."""
    client = _get_neptune_client()
    kwargs: dict[str, Any] = {
        "graphIdentifier": graph_id,
        "queryString": query,
        "language": "OPEN_CYPHER",
    }
    if parameters:
        kwargs["parameters"] = json.dumps(parameters)

    try:
        response = client.execute_query(**kwargs)
        payload = response.get("payload")
        if hasattr(payload, "read"):
            raw = payload.read()
        else:
            raw = payload or b"{}"
        data = json.loads(raw)
        return data.get("results", [])
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("ResourceNotFoundException", "ThrottlingException"):
            logger.warning("Neptune query skipped (%s): %s", code, exc)
            return []
        logger.error("Neptune query failed: %s\nQuery: %s", exc, query)
        return []
    except Exception as exc:
        logger.error("Neptune query error: %s", exc, exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction from retrieved chunks
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _ENTITY_SLUG.sub("_", text.lower()).strip("_")


def extract_entity_ids_from_chunks(chunks_text: list[str]) -> list[str]:
    """
    Heuristic: extract capitalised multi-word phrases and technology names
    that might correspond to graph node IDs.
    This is a lightweight approach; a full NER pass would improve recall.
    """
    combined = " ".join(chunks_text)

    # Find technology / service names using same patterns as extractor
    from models import SOURCE_WEIGHTS  # reuse signal weights

    # Import extractors for pattern matching
    try:
        from extractors import _find_aws_services, _find_technologies, _find_patterns
        entities = (
            [_slug("aws_service_" + s) for s in _find_aws_services(combined)]
            + [_slug("technology_" + t) for t in _find_technologies(combined)]
            + [_slug("pattern_" + p) for p in _find_patterns(combined)]
        )
    except ImportError:
        entities = []

    return list(set(entities))[:20]  # cap to avoid huge queries


# ─────────────────────────────────────────────────────────────────────────────
# Graph expansion queries
# ─────────────────────────────────────────────────────────────────────────────

def get_skill_neighbourhood(graph_id: str, skill_labels: list[str]) -> list[dict]:
    """
    Expand a list of skill labels to their co-occurring skills, patterns,
    and repositories via DEMONSTRATES_SKILL / STRENGTHENS / BUILT edges.
    """
    if not skill_labels or not graph_id:
        return []

    query = """
    MATCH (p:Person)-[:DEMONSTRATES_SKILL]->(s:Skill)
    WHERE toLower(s.label) IN $skill_labels
    OPTIONAL MATCH (s)-[:STRENGTHENS]->(related)
    OPTIONAL MATCH (repo:Repository)-[:USES_TECH|USES_AWS_SERVICE]->(tech)
    WHERE (repo)<-[:BUILT]-(p)
    RETURN
        s.label AS skill,
        collect(DISTINCT related.label)[..5] AS related_entities,
        collect(DISTINCT tech.label)[..5] AS co_technologies
    LIMIT 20
    """
    params = {"skill_labels": [l.lower() for l in skill_labels[:10]]}
    return _run_query(graph_id, query, params)


def get_pattern_evidence(graph_id: str, pattern_labels: list[str]) -> list[dict]:
    """
    Return repositories + documents that demonstrate requested patterns,
    with evidence weights.
    """
    if not pattern_labels or not graph_id:
        return []

    query = """
    MATCH (n)-[:DEMONSTRATES_PATTERN]->(pat:Pattern)
    WHERE toLower(pat.label) IN $pattern_labels
    OPTIONAL MATCH (n)-[:CONTAINS|INDICATES_SKILL]->(related)
    RETURN
        pat.label AS pattern,
        labels(n)[0] AS entity_type,
        n.label AS entity_label,
        n.source_file AS source_file,
        collect(DISTINCT related.label)[..5] AS related
    ORDER BY pat.label
    LIMIT 30
    """
    params = {"pattern_labels": [l.lower() for l in pattern_labels[:10]]}
    return _run_query(graph_id, query, params)


def get_aws_service_context(graph_id: str, aws_labels: list[str]) -> list[dict]:
    """
    Find repositories and architecture patterns associated with given AWS services.
    """
    if not aws_labels or not graph_id:
        return []

    query = """
    MATCH (repo:Repository)-[:USES_AWS_SERVICE]->(svc:AWSService)
    WHERE toLower(svc.label) IN $aws_labels
    OPTIONAL MATCH (repo)-[:USES_AWS_SERVICE]->(other_svc:AWSService)
    WHERE other_svc.label <> svc.label
    OPTIONAL MATCH (repo)-[:USES_TECH]->(tech:Technology)
    RETURN
        svc.label AS aws_service,
        repo.label AS repository,
        collect(DISTINCT other_svc.label)[..5] AS co_services,
        collect(DISTINCT tech.label)[..5] AS co_technologies
    ORDER BY svc.label
    LIMIT 30
    """
    params = {"aws_labels": [l.lower() for l in aws_labels[:10]]}
    return _run_query(graph_id, query, params)


def get_person_summary(graph_id: str) -> dict:
    """
    High-level summary of the Person node: top skills, patterns, AWS services.
    Used to enrich every response with baseline expertise context.
    """
    if not graph_id:
        return {}

    query = """
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:DEMONSTRATES_SKILL]->(s:Skill)
    OPTIONAL MATCH (p)-[:DEMONSTRATES_PATTERN]->(pat:Pattern)
    OPTIONAL MATCH (p)-[:BUILT]->(repo:Repository)-[:USES_AWS_SERVICE]->(svc:AWSService)
    RETURN
        p.label AS person,
        collect(DISTINCT s.label)[..20] AS skills,
        collect(DISTINCT pat.label)[..15] AS patterns,
        collect(DISTINCT svc.label)[..20] AS aws_services,
        count(DISTINCT repo) AS repo_count
    LIMIT 1
    """
    rows = _run_query(graph_id, query)
    return rows[0] if rows else {}


# ─────────────────────────────────────────────────────────────────────────────
# High-level expansion entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_routing_strategy(
    question_type: str,
    graph_id: str | None = None,
) -> list[dict]:
    """
    Query the routing graph for EFFECTIVE_FOR edge weights associated with
    the given question_type. Used by the RAGRouter as a data source.

    Returns rows of {strategy, weight, feedback_count}.
    """
    gid = graph_id or NEPTUNE_GRAPH_ID
    if not gid:
        return []

    query = """
    MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
    WHERE d.question_type = $question_type
    RETURN r.label AS strategy,
           e.weight AS weight,
           coalesce(e.feedback_count, 0) AS feedback_count
    ORDER BY e.weight DESC
    """
    return _run_query(gid, query, {"question_type": question_type})


def get_document_type_distribution(graph_id: str | None = None) -> list[dict]:
    """
    Summarise how many documents of each type have been ingested.
    Used to surface routing graph health in the /health endpoint.
    """
    gid = graph_id or NEPTUNE_GRAPH_ID
    if not gid:
        return []

    query = """
    MATCH (doc:Document)-[:HAS_TYPE]->(dt:DocumentType)
    RETURN dt.label AS doc_type, count(doc) AS doc_count
    ORDER BY doc_count DESC
    """
    return _run_query(gid, query)


def expand_graph_context(
    retrieved_text_snippets: list[str],
    graph_id: str | None = None,
) -> dict[str, Any]:
    """
    Given a list of retrieved text snippets, run graph expansion queries
    to surface corroborating evidence.

    Returns:
        {
          "person_summary": {...},
          "skill_neighbourhood": [...],
          "pattern_evidence": [...],
          "aws_context": [...],
          "inferred_skills": [...],
          "repeated_patterns": [...],
          "graph_entities_used": [...],
        }
    """
    gid = graph_id or NEPTUNE_GRAPH_ID
    if not gid:
        logger.warning("NEPTUNE_GRAPH_ID not set – skipping graph expansion")
        return {
            "person_summary": {},
            "skill_neighbourhood": [],
            "pattern_evidence": [],
            "aws_context": [],
            "inferred_skills": [],
            "repeated_patterns": [],
            "graph_entities_used": [],
        }

    # Extract potential entity mentions from retrieved text
    try:
        from preprocessor.extractors import _find_aws_services, _find_technologies, _find_patterns
        combined = " ".join(retrieved_text_snippets)
        skill_labels = _find_technologies(combined)
        pattern_labels = _find_patterns(combined)
        aws_labels = _find_aws_services(combined)
    except ImportError:
        skill_labels, pattern_labels, aws_labels = [], [], []

    # Run graph queries in sequence (Lambda network calls)
    person_summary = get_person_summary(gid)
    skill_neighbourhood = get_skill_neighbourhood(gid, skill_labels[:10])
    pattern_evidence = get_pattern_evidence(gid, pattern_labels[:10])
    aws_context = get_aws_service_context(gid, aws_labels[:10])

    # Aggregate inferred skills from graph results
    inferred_skills: list[str] = list(set(
        [row.get("skill", "") for row in skill_neighbourhood if row.get("skill")]
        + person_summary.get("skills", [])
    ))[:15]

    # Identify repeated patterns (appear in multiple graph rows = multiple repos/docs)
    pattern_counts: dict[str, int] = {}
    for row in pattern_evidence:
        pat = row.get("pattern", "")
        if pat:
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
    repeated_patterns = [p for p, c in pattern_counts.items() if c >= 2]

    # Collect all entity labels touched
    graph_entities_used: list[str] = list(set(
        [row.get("skill", "") for row in skill_neighbourhood]
        + [row.get("pattern", "") for row in pattern_evidence]
        + [row.get("aws_service", "") for row in aws_context]
    ))

    return {
        "person_summary": person_summary,
        "skill_neighbourhood": skill_neighbourhood,
        "pattern_evidence": pattern_evidence,
        "aws_context": aws_context,
        "inferred_skills": [s for s in inferred_skills if s],
        "repeated_patterns": [p for p in repeated_patterns if p],
        "graph_entities_used": [e for e in graph_entities_used if e],
    }
