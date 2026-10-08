"""Retrieval hit and abstention labels for one (question, arm).

``source_passage_ids`` is the set of paragraphs the question is asked against:

* SQuAD 2.0: the single context paragraph.
* HotpotQA: every supporting-fact paragraph.
* Natural Questions: every window of the source context that contains a gold
  answer string.

``gold_in_top_k`` is the retrieval label. It is true when at least one source
passage id is among the top-k passages given to the generator.

``source_retrieved`` is stricter. It is true when every source passage id is
in that top-k. SQuAD has one source paragraph, so the two flags agree. Hotpot
and a multi-window NQ context require the full source set.

An abstention is a calibrated refusal only when that full source set was in
the generator's top-k and the question is unanswerable from it. The outcome
is then ``abstain_correct`` and it counts as correct.

When any source passage is missing from the top-k, the abstention is
``abstain_retrieval_miss``. It does not count as correct: the model never saw
the paragraph the question was asked against, so the refusal is not evidence
the model knew the context was empty.

When the full source set was retrieved and the question is answerable, the
abstention is ``abstain_incorrect``. The answer was in the passages the
generator saw. It does not count as correct.

A non-abstention has a null abstention label. Token F1, and the later
adjudication of F1 in [0.2, 0.8], decide those rows.
"""
from __future__ import annotations

from typing import Any, Sequence

from experiments.common import ProtocolError

ABSTAIN_CORRECT = "abstain_correct"
ABSTAIN_RETRIEVAL_MISS = "abstain_retrieval_miss"
ABSTAIN_INCORRECT = "abstain_incorrect"

ABSTENTION_LABELS = (ABSTAIN_CORRECT, ABSTAIN_RETRIEVAL_MISS, ABSTAIN_INCORRECT)


def _source_ids(source_passage_ids: Sequence[str]) -> list[str]:
    if isinstance(source_passage_ids, str) or not isinstance(source_passage_ids, Sequence):
        raise ProtocolError("source_passage_ids must be a non-empty list")
    ids = [str(item) for item in source_passage_ids]
    if not ids or any(not item for item in ids):
        raise ProtocolError("source_passage_ids must be a non-empty list of passage ids")
    return ids


def gold_in_top_k(source_passage_ids: Sequence[str], retrieved_ids: Sequence[str]) -> bool:
    """True when at least one source passage was given to the generator."""
    source = set(_source_ids(source_passage_ids))
    return bool(source & set(retrieved_ids))


def source_was_retrieved(source_passage_ids: Sequence[str], retrieved_ids: Sequence[str]) -> bool:
    """True when every source passage was given to the generator."""
    source = set(_source_ids(source_passage_ids))
    return source <= set(retrieved_ids)


def retrieval_label(source_passage_ids: Sequence[str], retrieved_ids: Sequence[str]) -> dict[str, Any]:
    source = _source_ids(source_passage_ids)
    got = set(retrieved_ids)
    n_in = len(set(source) & got)
    return {
        "n_source_passages": len(set(source)),
        "n_source_in_top_k": n_in,
        "gold_in_top_k": n_in >= 1,
        "source_retrieved": set(source) <= got,
    }


def abstention_outcome(
    *,
    abstained: bool,
    unanswerable: bool,
    source_retrieved: bool,
) -> dict[str, Any]:
    """Label one abstention. A non-abstention returns nulls.

    ``source_retrieved`` is :func:`source_was_retrieved`: every source passage
    was in the top-k. ``unanswerable`` is the dataset flag (SQuAD 2.0 empty
    gold list). Hotpot and the MRQA NQ file are answerable.
    """
    if not abstained:
        return {"abstention_label": None, "abstention_counts_correct": None}
    if source_retrieved and unanswerable:
        return {
            "abstention_label": ABSTAIN_CORRECT,
            "abstention_counts_correct": True,
        }
    if not source_retrieved:
        return {
            "abstention_label": ABSTAIN_RETRIEVAL_MISS,
            "abstention_counts_correct": False,
        }
    return {
        "abstention_label": ABSTAIN_INCORRECT,
        "abstention_counts_correct": False,
    }
