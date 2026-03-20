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
    # Adaptive routing nodes
    DOCUMENT_TYPE = "DocumentType"
    CHUNKING_STRATEGY = "ChunkingStrategy"
    RAG_STRATEGY = "RAGStrategy"


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
    # Adaptive routing edges
    CHUNKED_WITH = "CHUNKED_WITH"       # Document → ChunkingStrategy
    HAS_TYPE = "HAS_TYPE"               # Document → DocumentType
    EFFECTIVE_FOR = "EFFECTIVE_FOR"     # RAGStrategy → DocumentType (learned weight)


# ─────────────────────────────────────────────────────────────────────────────
# Routing constants
# ─────────────────────────────────────────────────────────────────────────────

class DocumentTypeLabel(str, Enum):
    TECHNICAL_SPEC = "technical_spec"
    NARRATIVE = "narrative"
    STRUCTURED_DATA = "structured_data"
    CODE = "code"
    DIAGRAM_DERIVED = "diagram_derived"


class ChunkingStrategyLabel(str, Enum):
    HIERARCHICAL = "hierarchical"   # 1500/300 tokens – default Bedrock KB
    SENTENCE = "sentence"           # sentence-boundary splits
    FIXED_512 = "fixed_512"         # fixed 512-token windows
    FIXED_256 = "fixed_256"         # fixed 256-token windows


class RAGStrategyLabel(str, Enum):
    SEMANTIC = "semantic_search"        # pure vector similarity (default)
    GRAPH_FIRST = "graph_first"         # graph traversal → vector retrieval
    KEYWORD_BOOSTED = "keyword_boosted" # semantic + keyword overlap reranking
    HYBRID = "hybrid"                   # semantic + Neptune vector + graph + keyword


# Initial EFFECTIVE_FOR weights: (RAGStrategy, QuestionType) → weight
# Represents the agent's prior before any query feedback.
ROUTING_PRIORS: dict[tuple[str, str], float] = {
    (RAGStrategyLabel.GRAPH_FIRST,     "skill_depth"):   0.80,
    (RAGStrategyLabel.SEMANTIC,        "skill_depth"):   0.60,
    (RAGStrategyLabel.HYBRID,          "skill_depth"):   0.65,
    (RAGStrategyLabel.KEYWORD_BOOSTED, "skill_depth"):   0.50,

    (RAGStrategyLabel.GRAPH_FIRST,     "architecture"):  0.80,
    (RAGStrategyLabel.HYBRID,          "architecture"):  0.70,
    (RAGStrategyLabel.SEMANTIC,        "architecture"):  0.60,
    (RAGStrategyLabel.KEYWORD_BOOSTED, "architecture"):  0.55,

    (RAGStrategyLabel.KEYWORD_BOOSTED, "project"):       0.70,
    (RAGStrategyLabel.SEMANTIC,        "project"):       0.65,
    (RAGStrategyLabel.GRAPH_FIRST,     "project"):       0.60,
    (RAGStrategyLabel.HYBRID,          "project"):       0.60,

    (RAGStrategyLabel.HYBRID,          "comparison"):    0.70,
    (RAGStrategyLabel.SEMANTIC,        "comparison"):    0.65,
    (RAGStrategyLabel.GRAPH_FIRST,     "comparison"):    0.60,
    (RAGStrategyLabel.KEYWORD_BOOSTED, "comparison"):    0.55,

    (RAGStrategyLabel.KEYWORD_BOOSTED, "credential"):    0.80,
    (RAGStrategyLabel.SEMANTIC,        "credential"):    0.60,
    (RAGStrategyLabel.GRAPH_FIRST,     "credential"):    0.50,
    (RAGStrategyLabel.HYBRID,          "credential"):    0.55,

    (RAGStrategyLabel.SEMANTIC,        "general"):       0.70,
    (RAGStrategyLabel.HYBRID,          "general"):       0.60,
    (RAGStrategyLabel.KEYWORD_BOOSTED, "general"):       0.55,
    (RAGStrategyLabel.GRAPH_FIRST,     "general"):       0.55,
}


# ─────────────────────────────────────────────────────────────────────────────
# Evidence weighting constants
# ─────────────────────────────────────────────────────────────────────────────

# Source weight table – higher = more authoritative (used as default / fallback)
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

# Strategy-specific weight overrides applied at query time.
# Different retrieval strategies treat document types differently:
#   keyword_boosted  → PDF/resume/doc is primary evidence for project work
#   graph_first      → authoritative MD/diagram files dominate
#   hybrid           → balanced between the two
#   semantic_search  → falls back to SOURCE_WEIGHTS (no override)
STRATEGY_SOURCE_WEIGHTS: dict[str, dict[str, float]] = {
    # project / credential questions: PDF/resume is first-hand evidence;
    # authoritative design docs are less relevant than project artefacts.
    "keyword_boosted": {
        ".pdf": 0.9,
        ".docx": 0.85,
        ".doc": 0.85,
        "resume": 0.9,
        "repo-signals": 0.8,
        "readme": 0.7,
        "README": 0.7,
        "architecture.md": 0.6,
        "CLAUDE.md": 0.5,
        "default": 0.65,
    },
    # skill_depth / architecture questions: design docs dominate;
    # PDFs/resumes are low-signal.
    "graph_first": {
        "architecture.md": 1.0,
        "CLAUDE.md": 1.0,
        "plantuml_derived": 0.9,
        "c4_diagram": 0.9,
        "aws_architecture": 0.9,
        "repo-signals": 0.8,
        "code": 0.6,
        "readme": 0.5,
        "README": 0.5,
        ".pdf": 0.2,
        ".docx": 0.2,
        ".doc": 0.2,
        "resume": 0.2,
        "default": 0.45,
    },
    # comparison questions: balanced, slight boost to authoritative docs.
    "hybrid": {
        "architecture.md": 0.9,
        "CLAUDE.md": 0.9,
        "plantuml_derived": 0.8,
        "repo-signals": 0.75,
        ".pdf": 0.65,
        ".docx": 0.6,
        ".doc": 0.6,
        "resume": 0.6,
        "code": 0.6,
        "default": 0.55,
    },
}


def get_source_weight(source_key: str, strategy: str = "") -> float:
    """Return evidence weight for a given source identifier.

    When strategy is provided, uses the strategy-specific weight table so
    PDF/resume chunks score higher for project/credential queries and
    authoritative MD files score higher for skill_depth/architecture queries.
    Falls back to SOURCE_WEIGHTS when no strategy override exists.
    """
    weights = STRATEGY_SOURCE_WEIGHTS.get(strategy, SOURCE_WEIGHTS)
    for k, w in weights.items():
        if k == "default":
            continue
        if k.lower() in source_key.lower():
            return w
    return weights.get("default", SOURCE_WEIGHTS["default"])


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
class DocumentTypeAnalysis:
    """Result of routing_analyzer.analyze_document()."""
    doc_type: str                           # DocumentTypeLabel value
    chunking_strategy: str                  # ChunkingStrategyLabel value
    text_stats: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class RetrievalConfig:
    """Routing decision produced by RAGRouter."""
    strategy: str                           # RAGStrategyLabel value
    include_graph: bool = True
    boost_keywords: bool = False
    use_neptune_chunks: bool = False        # also search Neptune vector store
    strategy_confidence: float = 0.5       # weight of the winning EFFECTIVE_FOR edge


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
    model_id: str = ""
    routing_decision: dict[str, Any] = field(default_factory=dict)

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
            "routingDecision": self.routing_decision,
        }
