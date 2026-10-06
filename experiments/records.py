"""Join on-disk stage outputs into per-(qid, arm) records and reward values.

The verified reward is ``verified_reward.combine``: a weighted mean of the
signals that were observed, and missing when none were. Judge weight and
grounding weight come from the config. Self-confidence enters only when its
parse status is ``ok``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))

import verified_reward as V  # noqa: E402

from experiments.common import ProtocolError, read_jsonl, validate_rows
from experiments.confidence import FAILED, OMITTED, UNPARSEABLE
from experiments.metrics import score_answer

_FALLBACK = {"omitted": OMITTED, "unparseable": UNPARSEABLE, "failed": FAILED}


def _index(rows: Sequence[dict[str, Any]], path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for i, row in enumerate(rows, 1):
        key = (row["qid"], row["arm"])
        if key in out:
            raise ProtocolError(f"{path}:{i}: duplicate (qid, arm) {key}")
        out[key] = row
    return out


def combine_reward(parts: dict[str, tuple[float | None, float]]) -> float | None:
    signals = [V.RewardSignal(name, value, weight) for name, (value, weight) in parts.items()]
    value, _used = V.combine(signals)
    return value


def build_rewards(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float | None]:
    status = row["self_status"]
    if status not in ("ok", "omitted", "unparseable", "failed"):
        raise ProtocolError(f"qid={row['qid']} arm={row['arm']}: unknown self_status {status!r}")
    self_ok = float(row["self_confidence"]) if status == "ok" else None
    if status == "ok":
        fallback = float(row["self_confidence"])
    else:
        fallback = float(cfg[f"fallback_{status}"])
        if abs(float(row["self_confidence"]) - _FALLBACK[status]) > 1e-9:
            raise ProtocolError(
                f"qid={row['qid']} arm={row['arm']}: status {status} stored self_confidence "
                f"{row['self_confidence']} but the deployed fallback is {_FALLBACK[status]}"
            )
    g_w = float(cfg["grounding_weight"])
    j_w = float(cfg["judge_weight"])
    s_w = float(cfg["self_weight"])
    grounding = row.get("lexical_grounding")
    judge = row.get("judge")
    verified = combine_reward({
        "grounding": (grounding, g_w),
        "judge": (judge, j_w),
    })
    verified_self = combine_reward({
        "grounding": (grounding, g_w),
        "judge": (judge, j_w),
        "self": (self_ok, s_w),
    })
    return {
        "self": self_ok,
        "self_with_fallbacks": fallback,
        "lexical_grounding": None if grounding is None else float(grounding),
        "verified": verified,
        "verified_plus_self": verified_self,
        "oracle": float(row["correct"]),
    }


def load_joined(results: Path, cfg: dict[str, Any], datasets: Sequence[str] | None = None,
                attach_signals: bool = True) -> list[dict[str, Any]]:
    """Require sample, retrieval, and generation. Signals are joined when present."""
    sample_dir = results / "samples"
    if not sample_dir.is_dir():
        raise ProtocolError(f"sample directory not found: {sample_dir}")
    names = list(datasets) if datasets else ["squad", "hotpot", "nq"]
    questions = []
    for name in names:
        path = sample_dir / f"{name}.jsonl"
        rows = read_jsonl(path)
        validate_rows(rows, ("qid", "dataset", "question", "question_type", "gold_answers", "unanswerable"), path)
        questions.extend(rows)
    retrieval_rows = []
    generation_rows = []
    for name in names:
        rpath = results / "retrieval" / f"{name}.jsonl"
        gpath = results / "generation" / f"{name}.jsonl"
        retrieval_rows.extend(read_jsonl(rpath))
        generation_rows.extend(read_jsonl(gpath))
    validate_rows(retrieval_rows, ("qid", "arm", "passages"), results / "retrieval")
    validate_rows(generation_rows, ("qid", "arm", "answer", "self_confidence", "self_status", "self_reported"), results / "generation")
    retrieval = _index(retrieval_rows, results / "retrieval")
    generation = _index(generation_rows, results / "generation")
    signals = {}
    for kind, fname in (
        ("lexical_grounding", "lexical.jsonl"),
        ("nli_grounding", "nli.jsonl"),
        ("judge", "judge.jsonl"),
    ):
        path = results / "signals" / fname
        if not attach_signals:
            signals[kind] = None
        elif path.is_file():
            rows = read_jsonl(path)
            validate_rows(rows, ("qid", "arm", "value", "reason"), path)
            signals[kind] = _index(rows, path)
        else:
            signals[kind] = None
    joined = []
    for q in questions:
        for arm in ("semantic_search", "graph_first", "keyword_boosted", "hybrid"):
            key = (q["qid"], arm)
            if key not in retrieval or key not in generation:
                raise ProtocolError(f"missing retrieval or generation for {key}")
            gen = generation[key]
            scored = score_answer(gen["answer"], q["gold_answers"], bool(q["unanswerable"]))
            rec = {
                "qid": q["qid"],
                "dataset": q["dataset"],
                "arm": arm,
                "question_type": q["question_type"],
                "question": q["question"],
                "answer": gen["answer"],
                "self_confidence": gen["self_confidence"],
                "self_reported": gen["self_reported"],
                "self_status": gen["self_status"],
                "f1": scored["f1"],
                "correct": scored["correct"],
                "passages": [p["text"] for p in retrieval[key]["passages"]],
            }
            for kind, table in signals.items():
                if table is None:
                    rec[kind] = None
                    rec[f"{kind}_reason"] = "signal_file_missing"
                else:
                    if key not in table:
                        raise ProtocolError(f"signal {kind} missing row for {key}")
                    rec[kind] = table[key]["value"]
                    rec[f"{kind}_reason"] = table[key]["reason"]
            rec["lexical_grounding"] = rec["lexical_grounding"]
            rec["judge"] = rec["judge"]
            rec["rewards"] = build_rewards(rec, cfg)
            joined.append(rec)
    return joined
