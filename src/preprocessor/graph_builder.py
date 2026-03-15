"""
Graph entity and edge builder for ExpertiseRAG.

Takes extraction results (expertise signals, metadata) and constructs
typed graph nodes and edges conforming to the ExpertiseRAG graph schema.

Node types : Person, Repository, Document, Skill, Pattern, Technology,
             AWSService, ArchitectureStyle, Evidence, Claim
Edge types : BUILT, CONTAINS, USES_TECH, USES_AWS_SERVICE,
             DEMONSTRATES_PATTERN, SUPPORTS_CLAIM, INDICATES_SKILL,
             DEMONSTRATES_SKILL, STRENGTHENS
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from models import EdgeType, GraphEdge, GraphNode, NodeType


# ─────────────────────────────────────────────────────────────────────────────
# ID helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stable_id(node_type: str, label: str) -> str:
    """Deterministic node ID from type + normalised label."""
    normalised = re.sub(r"[^a-z0-9]", "_", label.lower()).strip("_")
    return f"{node_type.lower()}_{normalised}"


def _edge_id(from_id: str, rel: str, to_id: str) -> str:
    return f"{from_id}__{rel.lower()}__{to_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Stateful accumulator: call add_extraction() for each source file,
    then call build() to get the final deduplicated nodes + edges.
    """

    def __init__(self, person_name: str = "Developer", repo_id: str = ""):
        self._person_name = person_name
        self._repo_id = repo_id or _stable_id("repository", "default_repo")

        # Use dict keyed by stable ID to deduplicate
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}

        # Seed person node
        person_node_id = _stable_id(NodeType.PERSON, person_name)
        self._nodes[person_node_id] = GraphNode(
            node_id=person_node_id,
            node_type=NodeType.PERSON,
            label=person_name,
            properties={"name": person_name},
        )
        self._person_id = person_node_id

        # Seed repository node
        self._nodes[self._repo_id] = GraphNode(
            node_id=self._repo_id,
            node_type=NodeType.REPOSITORY,
            label=self._repo_id,
            properties={"repo_id": self._repo_id},
        )

        # Person BUILT repository
        self._add_edge(self._person_id, EdgeType.BUILT, self._repo_id)

    # ── Node helpers ──────────────────────────────────────────────────────────

    def _upsert_node(
        self,
        node_type: NodeType,
        label: str,
        source_file: str = "",
        properties: dict | None = None,
        confidence: float = 1.0,
    ) -> str:
        nid = _stable_id(node_type, label)
        if nid in self._nodes:
            existing = self._nodes[nid]
            # Boost confidence if seen again
            existing.confidence = min(1.0, existing.confidence + 0.05)
            # Merge properties
            if properties:
                existing.properties.update(properties)
        else:
            self._nodes[nid] = GraphNode(
                node_id=nid,
                node_type=node_type,
                label=label,
                source_file=source_file,
                properties=properties or {},
                confidence=confidence,
            )
        return nid

    def _add_edge(
        self,
        from_id: str,
        rel: EdgeType,
        to_id: str,
        source_file: str = "",
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> None:
        eid = _edge_id(from_id, rel.value, to_id)
        if eid in self._edges:
            # Strengthen existing edge
            self._edges[eid].weight = min(1.0, self._edges[eid].weight + 0.1)
        else:
            self._edges[eid] = GraphEdge(
                edge_id=eid,
                from_id=from_id,
                to_id=to_id,
                relationship=rel.value,
                weight=weight,
                source_file=source_file,
                properties=properties or {},
            )

    # ── Main ingestion ─────────────────────────────────────────────────────────

    def add_extraction(
        self,
        source_file: str,
        file_type: str,
        expertise_signals: list[dict[str, Any]],
        extracted_text: str = "",
        weight: float = 0.5,
    ) -> None:
        """
        Process a single file's extraction output, creating nodes and edges.
        """
        # Document node
        doc_id = _stable_id(NodeType.DOCUMENT, source_file)
        doc_props: dict[str, Any] = {
            "source_file": source_file,
            "file_type": file_type,
            "weight": weight,
        }
        if extracted_text:
            # Short fingerprint for dedup
            doc_props["text_hash"] = hashlib.md5(
                extracted_text[:500].encode()
            ).hexdigest()

        self._upsert_node(
            NodeType.DOCUMENT,
            source_file,
            source_file=source_file,
            properties=doc_props,
        )
        # Repository CONTAINS document
        self._add_edge(self._repo_id, EdgeType.CONTAINS, doc_id, source_file, weight)

        # Process each expertise signal
        skill_ids: list[str] = []
        tech_ids: list[str] = []
        aws_ids: list[str] = []
        pattern_ids: list[str] = []

        for sig in expertise_signals:
            sig_type = sig.get("signal_type", "")
            value = sig.get("value", "").strip()
            if not value:
                continue
            freq = int(sig.get("frequency", 1))
            sig_weight = float(sig.get("weight", weight))
            # Boost weight for repeated occurrences
            effective_weight = min(1.0, sig_weight * (1.0 + 0.1 * min(freq, 5)))

            if sig_type == "skill":
                nid = self._upsert_node(
                    NodeType.SKILL, value, source_file,
                    {"category": "skill", "frequency": freq},
                    confidence=effective_weight,
                )
                skill_ids.append(nid)
                self._add_edge(doc_id, EdgeType.INDICATES_SKILL, nid, source_file, effective_weight)
                self._add_edge(self._person_id, EdgeType.DEMONSTRATES_SKILL, nid, source_file, effective_weight)

            elif sig_type == "technology":
                nid = self._upsert_node(
                    NodeType.TECHNOLOGY, value, source_file,
                    {"frequency": freq},
                    confidence=effective_weight,
                )
                tech_ids.append(nid)
                self._add_edge(doc_id, EdgeType.USES_TECH, nid, source_file, effective_weight)
                self._add_edge(self._repo_id, EdgeType.USES_TECH, nid, source_file, effective_weight)

            elif sig_type == "aws_service":
                nid = self._upsert_node(
                    NodeType.AWS_SERVICE, value, source_file,
                    {"service_name": value, "frequency": freq},
                    confidence=effective_weight,
                )
                aws_ids.append(nid)
                self._add_edge(doc_id, EdgeType.USES_AWS_SERVICE, nid, source_file, effective_weight)
                self._add_edge(self._repo_id, EdgeType.USES_AWS_SERVICE, nid, source_file, effective_weight)

            elif sig_type == "pattern":
                nid = self._upsert_node(
                    NodeType.PATTERN, value, source_file,
                    {"pattern_name": value, "frequency": freq},
                    confidence=effective_weight,
                )
                pattern_ids.append(nid)
                self._add_edge(doc_id, EdgeType.DEMONSTRATES_PATTERN, nid, source_file, effective_weight)
                self._add_edge(self._person_id, EdgeType.DEMONSTRATES_PATTERN, nid, source_file, effective_weight)

        # Cross-entity STRENGTHENS edges (skills ↔ patterns, technologies ↔ AWS services)
        for sid in skill_ids:
            for pid in pattern_ids:
                self._add_edge(sid, EdgeType.STRENGTHENS, pid, source_file, 0.5)
        for tid in tech_ids:
            for aid in aws_ids:
                self._add_edge(tid, EdgeType.STRENGTHENS, aid, source_file, 0.5)

    def build(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (nodes_list, edges_list) as plain dicts."""
        nodes = [n.to_dict() for n in self._nodes.values()]
        edges = [e.to_dict() for e in self._edges.values()]
        return nodes, edges


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_from_extractions(
    extractions: list[dict[str, Any]],
    person_name: str = "Developer",
    repo_prefix: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    Build a complete graph from multiple extraction results.

    Args:
        extractions: List of dicts, each with keys:
                     source_file, file_type, expertise_signals, extracted_text, weight
        person_name: Owner of the repository/content
        repo_prefix: S3 key prefix / repo identifier

    Returns:
        (nodes, edges) as lists of plain dicts
    """
    repo_id = _stable_id("repository", repo_prefix or "default_repo")
    builder = GraphBuilder(person_name=person_name, repo_id=repo_id)

    for ext in extractions:
        builder.add_extraction(
            source_file=ext.get("source_file", ""),
            file_type=ext.get("file_type", "text"),
            expertise_signals=ext.get("expertise_signals", []),
            extracted_text=ext.get("extracted_text", ""),
            weight=float(ext.get("weight", 0.5)),
        )

    return builder.build()
