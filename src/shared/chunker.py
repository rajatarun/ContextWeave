"""
Text chunking strategies for ExpertiseRAG.

Replaces the chunking previously performed by Amazon Bedrock Knowledge Base.
Supports four strategies that mirror the Bedrock KB configuration:

  hierarchical  – parent (1500 tokens) + child (300 tokens) chunks with
                  60-token overlap; Bedrock KB HIERARCHICAL equivalent
  sentence      – sentence-boundary splits grouped into ~512-char windows
  fixed_512     – sliding window of ~512 tokens with 50-token overlap
  fixed_256     – sliding window of ~256 tokens with 25-token overlap

Token counts are approximated as word counts (1 token ≈ 1 word for
English technical prose). No external tokeniser is required.
"""
from __future__ import annotations

import re
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Chunk data structure
# ─────────────────────────────────────────────────────────────────────────────

class Chunk(NamedTuple):
    content: str
    parent_content: str | None  # non-None for child chunks in hierarchical mode
    is_child: bool
    chunk_index: int             # position within the document


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _approx_token_count(text: str) -> int:
    """Approximate token count as word count (safe for English technical text)."""
    return len(text.split())


def _words_to_char_offset(words: list[str], word_index: int) -> int:
    """Sum up character lengths (including spaces) up to word_index."""
    return sum(len(w) + 1 for w in words[:word_index])


def _split_into_windows(
    text: str,
    window_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Split text into overlapping windows of approximately window_tokens words.
    Returns a list of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, window_tokens - overlap_tokens)
    i = 0

    while i < len(words):
        window = words[i : i + window_tokens]
        chunks.append(" ".join(window))
        if i + window_tokens >= len(words):
            break
        i += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_hierarchical(
    text: str,
    parent_tokens: int = 1500,
    child_tokens: int = 300,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """
    Hierarchical chunking: split into parent windows, then sub-chunk each
    parent into child windows. Mirrors Bedrock KB HIERARCHICAL strategy.

    Both parent and child chunks are returned; children carry the parent text
    so the retriever can surface full-context answers when needed.
    """
    parent_windows = _split_into_windows(text, parent_tokens, overlap_tokens)
    chunks: list[Chunk] = []
    chunk_index = 0

    for parent_text in parent_windows:
        # Emit the parent chunk itself (is_child=False)
        chunks.append(Chunk(
            content=parent_text,
            parent_content=None,
            is_child=False,
            chunk_index=chunk_index,
        ))
        chunk_index += 1

        # Emit child chunks derived from this parent
        child_windows = _split_into_windows(parent_text, child_tokens, overlap_tokens)
        for child_text in child_windows:
            # Skip child if it's essentially the same as the parent
            if child_text.strip() == parent_text.strip():
                continue
            chunks.append(Chunk(
                content=child_text,
                parent_content=parent_text,
                is_child=True,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    return chunks


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_SENTENCE_FALLBACK_LEN = 200  # chars, for sentences without terminal punctuation


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, falling back to line breaks."""
    # Try regex split on sentence-ending punctuation
    sentences = _SENTENCE_END.split(text)
    # Further split any very long "sentences" at newlines
    result: list[str] = []
    for sent in sentences:
        if len(sent) > _SENTENCE_FALLBACK_LEN * 3:
            result.extend(line.strip() for line in sent.splitlines() if line.strip())
        else:
            stripped = sent.strip()
            if stripped:
                result.append(stripped)
    return result


def _chunk_sentence(text: str, window_chars: int = 512) -> list[Chunk]:
    """
    Sentence-boundary chunking: group sentences into windows of ~window_chars
    characters with one-sentence overlap between consecutive windows.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(sentences):
        window: list[str] = []
        length = 0
        j = i

        while j < len(sentences) and length < window_chars:
            window.append(sentences[j])
            length += len(sentences[j]) + 1
            j += 1

        content = " ".join(window).strip()
        if content:
            chunks.append(Chunk(
                content=content,
                parent_content=None,
                is_child=False,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        # Advance by all but the last sentence (one-sentence overlap)
        advance = max(1, j - i - 1)
        i += advance

    return chunks


def _chunk_fixed(text: str, window_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Fixed-size token window chunking."""
    windows = _split_into_windows(text, window_tokens, overlap_tokens)
    return [
        Chunk(
            content=w,
            parent_content=None,
            is_child=False,
            chunk_index=idx,
        )
        for idx, w in enumerate(windows)
        if w.strip()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, strategy: str) -> list[Chunk]:
    """
    Chunk text using the named strategy.

    Args:
        text:     Extracted document text to chunk.
        strategy: One of 'hierarchical', 'sentence', 'fixed_512', 'fixed_256'.
                  Unknown values fall back to 'fixed_512'.

    Returns:
        List of Chunk namedtuples, ordered by position in the document.
    """
    if not text or not text.strip():
        return []

    if strategy == "hierarchical":
        return _chunk_hierarchical(text)
    elif strategy == "sentence":
        return _chunk_sentence(text)
    elif strategy == "fixed_256":
        return _chunk_fixed(text, window_tokens=256, overlap_tokens=25)
    else:
        # fixed_512 and any unknown strategy
        return _chunk_fixed(text, window_tokens=512, overlap_tokens=50)
