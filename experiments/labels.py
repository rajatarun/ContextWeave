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

A non-abstention has a null abstention label. Joined correctness then uses
token F1 outside the adjudication band and the adjudicator's label inside it.
``source_retrieved`` is the flag that abstention correctness reads.
``gold_in_top_k`` is stored on the same row. The two flags have to agree:
every source passage retrieved implies at least one source passage retrieved.
"""
from __future__ import annotations

from typing import Any, Sequence

from experiments.common import ProtocolError
from experiments.metrics import score_answer

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


def f1_band(f1: float, low: float, high: float) -> bool:
    """True when token F1 is inside the closed adjudication interval."""
    if low < 0 or high > 1 or low > high:
        raise ProtocolError(f"adjudication band [{low}, {high}] is not inside [0, 1]")
    return float(low) <= float(f1) <= float(high)


def decide_correctness(
    *,
    answer: str,
    gold_answers: Sequence[str],
    unanswerable: bool,
    self_status: str,
    source_retrieved: bool,
    gold_in_top_k: bool,
    yes_no: bool,
    f1_low: float,
    f1_high: float,
    stored_label: str | None = None,
    stored_counts: bool | None = None,
    check_stored: bool = False,
    adjudication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Binary ``correct`` for one joined row.

    An ``ok`` or ``omitted`` abstention uses the abstention label.
    ``abstain_correct`` is the only abstention that counts as correct, and it
    requires ``source_retrieved``. A failed or truncated call keeps a null
    abstention label, and token F1 decides it.

    A non-abstention whose token F1 sits in ``[f1_low, f1_high]`` takes the
    adjudicator's 0/1 label. Outside that band, token F1 of at least 0.5 is
    correct. ``token_f1_correct`` always keeps that 0.5-threshold bit.
    """
    if self_status not in ("ok", "omitted", "unparseable", "failed", "truncated"):
        raise ProtocolError(f"unknown self_status {self_status!r}")
    if not isinstance(source_retrieved, bool) or not isinstance(gold_in_top_k, bool):
        raise ProtocolError("gold_in_top_k and source_retrieved must be booleans")
    if not isinstance(yes_no, bool):
        raise ProtocolError("yes_no must be a boolean")
    if source_retrieved and not gold_in_top_k:
        raise ProtocolError(
            "source_retrieved is true and gold_in_top_k is false. "
            "Every source passage retrieved includes at least one."
        )
    scored = score_answer(answer, gold_answers, unanswerable)
    labeled = self_status in ("ok", "omitted") and bool(scored["abstained"])
    outcome = abstention_outcome(
        abstained=labeled,
        unanswerable=unanswerable,
        source_retrieved=source_retrieved,
    )
    if check_stored and (
        stored_label != outcome["abstention_label"]
        or stored_counts != outcome["abstention_counts_correct"]
    ):
        raise ProtocolError(
            "generation abstention label disagrees with source_retrieved and the answer. "
            f"Stored {stored_label!r} / {stored_counts!r}, "
            f"recomputed {outcome['abstention_label']!r} / {outcome['abstention_counts_correct']!r}."
        )
    in_band = outcome["abstention_label"] is None and f1_band(scored["f1"], f1_low, f1_high)
    correct: int | None
    source: str | None
    if outcome["abstention_label"] is not None:
        correct = 1 if outcome["abstention_counts_correct"] else 0
        source = "abstention"
    elif in_band:
        if adjudication is None:
            correct = None
            source = None
        else:
            correct = _adjudicated_bit(adjudication)
            source = "adjudication"
    else:
        correct = int(scored["correct"])
        source = "token_f1"
    return {
        "f1": scored["f1"],
        "token_f1_correct": int(scored["correct"]),
        "gold_contained": scored["gold_contained"],
        "abstained": bool(scored["abstained"]),
        "abstention_label": outcome["abstention_label"],
        "abstention_counts_correct": outcome["abstention_counts_correct"],
        "gold_in_top_k": gold_in_top_k,
        "source_retrieved": source_retrieved,
        "yes_no": yes_no,
        "in_f1_band": in_band,
        "correct": correct,
        "correct_source": source,
    }


def _adjudicated_bit(row: dict[str, Any]) -> int:
    if row.get("reason") is not None or row.get("value") not in (0, 1):
        where = f"qid={row.get('qid')} arm={row.get('arm')}"
        raise ProtocolError(
            f"adjudication for {where} has value {row.get('value')!r} "
            f"and reason {row.get('reason')!r}. Refusing to fall back to token F1."
        )
    return int(row["value"])
