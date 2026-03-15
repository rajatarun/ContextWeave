"""
Adaptive RAG Routing – Document Analysis Module

Classifies each ingested document into a DocumentType and recommends the
best ChunkingStrategy based on textual characteristics (structure density,
sentence length, code ratio, heading count, etc.).

Results are stored in the Neptune routing graph via GraphBuilder so that the
system can improve its routing decisions as more documents are ingested.
"""
from __future__ import annotations

import re
from typing import Any

from models import (
    ChunkingStrategyLabel,
    DocumentTypeLabel,
    DocumentTypeAnalysis,
)


# ─────────────────────────────────────────────────────────────────────────────
# Text statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_text_stats(text: str) -> dict[str, Any]:
    """
    Derive lightweight structural metrics from raw document text.

    Returns a dict with:
      heading_count       – number of Markdown headings (# / ##)
      code_block_count    – number of fenced code blocks
      code_char_ratio     – fraction of characters inside code blocks
      avg_sentence_len    – mean word-count per sentence
      avg_line_len        – mean character length per non-empty line
      word_count          – total word count
      has_yaml_keys       – True if the text contains "key: value" patterns
      has_plantuml        – True if the text starts with @startuml / @startc4
    """
    if not text:
        return {
            "heading_count": 0, "code_block_count": 0, "code_char_ratio": 0.0,
            "avg_sentence_len": 0.0, "avg_line_len": 0.0, "word_count": 0,
            "has_yaml_keys": False, "has_plantuml": False,
        }

    lines = text.splitlines()
    heading_count = sum(1 for l in lines if re.match(r"^#{1,6}\s", l))

    # Extract fenced code blocks
    code_blocks = re.findall(r"```[\s\S]*?```", text)
    code_block_count = len(code_blocks)
    code_chars = sum(len(b) for b in code_blocks)
    code_char_ratio = code_chars / max(len(text), 1)

    # Strip code blocks for prose metrics
    prose = re.sub(r"```[\s\S]*?```", " ", text)
    prose = re.sub(r"`[^`]+`", " ", prose)

    sentences = re.split(r"[.!?]+", prose)
    sentences = [s.strip() for s in sentences if s.strip()]
    word_counts = [len(s.split()) for s in sentences]
    avg_sentence_len = sum(word_counts) / max(len(word_counts), 1)

    non_empty_lines = [l for l in lines if l.strip()]
    avg_line_len = sum(len(l) for l in non_empty_lines) / max(len(non_empty_lines), 1)

    word_count = len(text.split())
    has_yaml_keys = bool(re.search(r"^\s*\w[\w_-]+:\s+\S", text, re.MULTILINE))
    has_plantuml = bool(re.match(r"\s*@start(uml|c4|mindmap|wbs)", text, re.IGNORECASE))

    return {
        "heading_count": heading_count,
        "code_block_count": code_block_count,
        "code_char_ratio": round(code_char_ratio, 3),
        "avg_sentence_len": round(avg_sentence_len, 1),
        "avg_line_len": round(avg_line_len, 1),
        "word_count": word_count,
        "has_yaml_keys": has_yaml_keys,
        "has_plantuml": has_plantuml,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document type classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_doc_type(
    filename: str,
    stats: dict[str, Any],
    signals: list[dict],
) -> str:
    """
    Rule-based document type classifier.

    Priority (highest wins):
      1. diagram_derived  – PlantUML or derived JSON
      2. code             – source code files / high code ratio
      3. structured_data  – YAML / JSON with key:value density
      4. technical_spec   – Markdown with ≥3 headings and AWS signal density
      5. narrative        – everything else (prose / README / articles)
    """
    fname = filename.lower()

    # 1. Diagram / derived artifacts
    if (
        fname.endswith((".puml", ".plantuml", ".pu", ".wsd"))
        or fname.endswith(".derived.json")
        or stats.get("has_plantuml")
    ):
        return DocumentTypeLabel.DIAGRAM_DERIVED

    # 2. Source code files or predominantly code content
    code_extensions = (
        ".py", ".ts", ".js", ".tsx", ".jsx", ".java", ".go", ".rs",
        ".cpp", ".c", ".cs", ".rb", ".php", ".swift", ".kt",
    )
    if fname.endswith(code_extensions) or stats.get("code_char_ratio", 0) > 0.35:
        return DocumentTypeLabel.CODE

    # 3. Structured data (YAML / JSON with key:value density)
    if fname.endswith((".yaml", ".yml", ".json")) or stats.get("has_yaml_keys"):
        return DocumentTypeLabel.STRUCTURED_DATA

    # 4. Technical spec: well-headed Markdown with AWS / tech signals
    aws_signal_count = sum(
        1 for s in signals if s.get("signal_type") == "aws_service"
    )
    if stats.get("heading_count", 0) >= 3 and aws_signal_count >= 2:
        return DocumentTypeLabel.TECHNICAL_SPEC

    # 5. Default: narrative prose
    return DocumentTypeLabel.NARRATIVE


# ─────────────────────────────────────────────────────────────────────────────
# Chunking strategy recommendation
# ─────────────────────────────────────────────────────────────────────────────

_CHUNKING_MAP: dict[str, str] = {
    DocumentTypeLabel.TECHNICAL_SPEC:  ChunkingStrategyLabel.HIERARCHICAL,
    DocumentTypeLabel.NARRATIVE:       ChunkingStrategyLabel.SENTENCE,
    DocumentTypeLabel.STRUCTURED_DATA: ChunkingStrategyLabel.FIXED_256,
    DocumentTypeLabel.CODE:            ChunkingStrategyLabel.FIXED_512,
    DocumentTypeLabel.DIAGRAM_DERIVED: ChunkingStrategyLabel.FIXED_256,
}


def recommend_chunking_strategy(doc_type: str, stats: dict[str, Any]) -> str:
    """
    Return the best ChunkingStrategyLabel for a given document type.
    Falls back to HIERARCHICAL if type is unknown.
    """
    return _CHUNKING_MAP.get(doc_type, ChunkingStrategyLabel.HIERARCHICAL)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_document(
    text: str,
    filename: str,
    signals: list[dict] | None = None,
) -> DocumentTypeAnalysis:
    """
    Analyse a document and return its type + recommended chunking strategy.

    Args:
        text:      Extracted text content (post-extraction, not raw)
        filename:  Original source filename (used for extension heuristics)
        signals:   Expertise signals already extracted by the extractors

    Returns:
        DocumentTypeAnalysis with doc_type, chunking_strategy, text_stats
    """
    signals = signals or []
    stats = compute_text_stats(text)
    doc_type = classify_doc_type(filename, stats, signals)
    strategy = recommend_chunking_strategy(doc_type, stats)

    return DocumentTypeAnalysis(
        doc_type=doc_type,
        chunking_strategy=strategy,
        text_stats=stats,
        confidence=1.0,
    )
