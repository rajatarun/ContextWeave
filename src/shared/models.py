"""
Shared domain models for ExpertiseRAG.

These dataclasses define the canonical shapes of graph entities, edges,
expertise signals, and API payloads used across the preprocessor and
query API Lambdas.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Graph node / edge type enumerations
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    PERSON = "Person"
    REPOSITORY = "Repository"
    DOCUMENT = "Document"
    SKILL = "Skill"
    PATTERN = "Pattern"
    TECHNOLOGY = "Technology"
    AWS_SERVICE = "AWSService"
    ARCHITECTURE_STYLE = "ArchitectureStyle"
    EVIDENCE = "Evidence"
    CLAIM = "Claim"


class EdgeType(str, Enum):
    BUILT = "BUILT"
    CONTAINS = "CONTAINS"
    USES_TECH = "USES_TECH"
    USES_AWS_SERVICE = "USES_AWS_SERVICE"
    DEMONSTRATES_PATTERN = "DEMONSTRATES_PATTERN"
    SUPPORTS_CLAIM = "SUPPORTS_CLAIM"
    INDICATES_SKILL = "INDICATES_SKILL"
    DEMONSTRATES_SKILL = "DEMONSTRATES_SKILL"
    STRENGTHENS = "STRENGTHENS"


# ─────────────────────────────────────────────────────────────────────────────
# Evidence weighting constants
# ─────────────────────────────────────────────────────────────────────────────

# Source weight table – higher = more authoritative
SOURCE_WEIGHTS: dict[str, float] = {
    "architecture.md": 1.0,
    "CLAUDE.md": 1.0,
    "plantuml_derived": 0.8,
    "c4_diagram": 0.8,
    "aws_architecture": 0.8,
    "code": 0.6,
    "README.md": 0.6,
    "readme": 0.6,
    "article": 0.5,
    "blog": 0.5,
    "resume": 0.3,
    "repo-signals.yaml": 0.7,
    "default": 0.4,
}

def get_source_weight(source_key: str) -> float:
    """Return evidence weight for a given source identifier."""
    for k, w in SOURCE_WEIGHTS.items():
        if k.lower() in source_key.lower():
            return w
    return SOURCE_WEIGHTS["default"]


# ─────────────────────────────────────────────────────────────────────────────
# Graph entity models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    node_id: str                        # Stable, deterministic ID (e.g. slug)
    node_type: str                      # NodeType value
    label: str                          # Human-readable name
    properties: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""               # Originating file within repo
    confidence: float = 1.0            # 0–1, extraction confidence

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class GraphEdge:
    edge_id: str                        # Stable ID: {from_id}__{rel_type}__{to_id}
    from_id: str                        # Source node ID
    to_id: str                          # Target node ID
    relationship: str                   # EdgeType value
    weight: float = 1.0                # Evidence weight
    properties: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExpertiseSignal:
    signal_type: str                    # e.g. "skill", "pattern", "technology"
    value: str                          # e.g. "AWS Lambda", "event-driven architecture"
    context: str = ""                   # Surrounding text snippet
    source_file: str = ""
    weight: float = 1.0
    frequency: int = 1                  # Occurrences across the corpus
    co_occurring: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Derived artifact envelope written to S3 under derived/
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DerivedArtifact:
    """
    Top-level envelope written as derived/<repo>/<base>.derived.json.
    Contains all preprocessor outputs for a single source file.
    """
    schema_version: str = "1.0"
    source_bucket: str = ""
    source_key: str = ""
    repo_prefix: str = ""
    file_type: str = ""                 # markdown | yaml | plantuml | text
    extracted_text: str = ""            # Clean text for Bedrock ingestion
    summary: str = ""                   # Short prose summary (used in derived/ for KB)
    expertise_signals: list[dict] = field(default_factory=list)
    graph_nodes: list[dict] = field(default_factory=list)
    graph_edges: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# API request / response models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryRequest:
    question: str
    top_k: int = 10
    include_graph_expansion: bool = True
    min_confidence: float = 0.3
    question_type: Optional[str] = None  # auto-classified if None

    @classmethod
    def from_dict(cls, d: dict) -> "QueryRequest":
        return cls(
            question=d.get("question", ""),
            top_k=int(d.get("topK", d.get("top_k", 10))),
            include_graph_expansion=bool(d.get("includeGraphExpansion", True)),
            min_confidence=float(d.get("minConfidence", 0.3)),
            question_type=d.get("questionType"),
        )


@dataclass
class RetrievedChunk:
    content: str
    score: float
    source_uri: str
    source_weight: float = 0.4
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_score(self) -> float:
        """Score adjusted by source authority weight."""
        return self.score * self.source_weight

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "score": self.score,
            "effectiveScore": self.effective_score,
            "sourceUri": self.source_uri,
            "sourceWeight": self.source_weight,
            "metadata": self.metadata,
        }


@dataclass
class QueryResponse:
    answer: str
    sources: list[dict]
    inferred_skills: list[str]
    repeated_patterns: list[str]
    confidence: float
    question_type: str
    graph_entities_used: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    model_id: str = ""                  # TODO: populated once generation model selected

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "inferredSkills": self.inferred_skills,
            "repeatedPatterns": self.repeated_patterns,
            "confidence": self.confidence,
            "questionType": self.question_type,
            "graphEntitiesUsed": self.graph_entities_used,
            "retrievalCount": self.retrieval_count,
            "modelId": self.model_id,
        }
