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

from experiments.common import ProtocolError, read_json, read_jsonl, validate_rows
from experiments.confidence import FAILED, OMITTED, UNPARSEABLE
from experiments.judge_access import JUDGE_ACCESS_REASON, access_reason
from experiments.ledger import SPEND_CAP_REASON
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


def _pending_keys(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    data = read_json(path)
    reason = str(data.get("reason") or SPEND_CAP_REASON)
    out = {}
    for row in data.get("rows") or []:
        out[(row["qid"], row["arm"])] = reason
    return out


def _require_status(status: str, where: str) -> None:
    if status not in ("ok", "omitted", "unparseable", "failed", "truncated"):
        raise ProtocolError(f"{where}: unknown status {status!r}")


def _fallback_from_status(status: str, value: Any, cfg: dict[str, Any], where: str) -> float | None:
    """Deployed fallback row. ``ok`` keeps the reported number. Truncation stays missing."""
    _require_status(status, where)
    if status == "ok":
        if value is None:
            raise ProtocolError(f"{where}: status ok stored no confidence")
        return float(value)
    if status == "truncated":
        if value is not None:
            raise ProtocolError(f"{where}: truncated rows store no confidence")
        return None
    expected = float(cfg[f"fallback_{status}"])
    if value is None or abs(float(value) - _FALLBACK[status]) > 1e-9:
        raise ProtocolError(
            f"{where}: status {status} stored {value} but the deployed fallback is {_FALLBACK[status]}"
        )
    return expected


def build_rewards(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float | None]:
    status = row["self_status"]
    where = f"qid={row['qid']} arm={row['arm']}"
    _require_status(status, where)
    if status == "ok":
        if row["self_confidence"] is None:
            raise ProtocolError(f"{where}: status ok stored no confidence")
        self_ok = float(row["self_confidence"])
    else:
        # Robust non-ok is not an observation. Older rows stored the fallback
        # constant on self_confidence; newer rows store null. Neither is a reward.
        self_ok = None
        if status == "truncated" and row["self_confidence"] is not None:
            raise ProtocolError(f"{where}: truncated rows store no confidence")
        if status != "truncated" and row["self_confidence"] is not None:
            if abs(float(row["self_confidence"]) - _FALLBACK[status]) > 1e-9:
                raise ProtocolError(
                    f"{where}: status {status} stored self_confidence "
                    f"{row['self_confidence']} but the deployed fallback is {_FALLBACK[status]}"
                )
    if "deployed_self_status" in row:
        fallback = _fallback_from_status(
            row["deployed_self_status"], row.get("deployed_self_confidence"), cfg,
            f"{where} deployed",
        )
    else:
        fallback = _fallback_from_status(status, row["self_confidence"], cfg, where)
    g_w = float(cfg["grounding_weight"])
    j_w = float(cfg["judge_weight"])
    s_w = float(cfg["self_weight"])
    grounding = row.get("lexical_grounding")
    judge = row.get("judge")
    judge_reason = row.get("judge_reason")
    # Access denied, or a row the spend cap stopped before judging: do not
    # quietly score verified from grounding alone.
    withhold_verified = judge_reason in (JUDGE_ACCESS_REASON, SPEND_CAP_REASON) or (
        isinstance(judge_reason, str) and judge_reason.startswith(SPEND_CAP_REASON)
    )
    if withhold_verified:
        verified = None
        verified_self = None
    else:
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
    gen_pending = _pending_keys(results / "generation" / "pending.json")
    judge_pending = _pending_keys(results / "signals" / "judge_pending.json")
    judge_blocked = access_reason(results)
    retrieval_rows = []
    generation_rows = []
    for name in names:
        rpath = results / "retrieval" / f"{name}.jsonl"
        gpath = results / "generation" / f"{name}.jsonl"
        retrieval_rows.extend(read_jsonl(rpath))
        try:
            generation_rows.extend(read_jsonl(gpath))
        except ProtocolError:
            if not gen_pending:
                raise
    if generation_rows:
        validate_rows(generation_rows, ("qid", "arm", "answer", "self_confidence", "self_status", "self_reported"), results / "generation")
    validate_rows(retrieval_rows, ("qid", "arm", "passages"), results / "retrieval")
    retrieval = _index(retrieval_rows, results / "retrieval")
    generation = _index(generation_rows, results / "generation")
    signals: dict[str, Any] = {}
    for kind, fname in (
        ("lexical_grounding", "lexical.jsonl"),
        ("nli_grounding", "nli.jsonl"),
        ("judge", "judge.jsonl"),
    ):
        path = results / "signals" / fname
        if not attach_signals or (kind == "judge" and judge_blocked):
            signals[kind] = None
            continue
        try:
            rows = read_jsonl(path)
        except ProtocolError:
            signals[kind] = None
            continue
        validate_rows(rows, ("qid", "arm", "value", "reason"), path)
        signals[kind] = _index(rows, path)
    joined = []
    arms = ("semantic_search", "graph_first", "keyword_boosted", "hybrid")
    for q in questions:
        keys = [(q["qid"], arm) for arm in arms]
        if any(key not in retrieval for key in keys):
            missing = [key for key in keys if key not in retrieval]
            raise ProtocolError(f"missing retrieval for {missing[0]}")
        if any(key not in generation for key in keys):
            absent = [key for key in keys if key not in generation]
            if all(key in gen_pending for key in absent):
                continue
            raise ProtocolError(f"missing generation for {absent[0]}")
        for arm in arms:
            key = (q["qid"], arm)
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
                **(
                    {"trailing_truncated": bool(gen["trailing_truncated"])}
                    if "trailing_truncated" in gen else {}
                ),
                **(
                    {
                        "deployed_self_confidence": gen.get("deployed_self_confidence"),
                        "deployed_self_status": gen["deployed_self_status"],
                    }
                    if "deployed_self_status" in gen else {}
                ),
                "f1": scored["f1"],
                "correct": scored["correct"],
                "gold_contained": scored["gold_contained"],
                "passages": [p["text"] for p in retrieval[key]["passages"]],
            }
            for kind, table in signals.items():
                if kind == "judge" and judge_blocked:
                    rec[kind] = None
                    rec[f"{kind}_reason"] = JUDGE_ACCESS_REASON
                elif table is None:
                    rec[kind] = None
                    rec[f"{kind}_reason"] = "signal_file_missing"
                elif key not in table:
                    if kind == "judge" and key in judge_pending:
                        rec[kind] = None
                        rec[f"{kind}_reason"] = judge_pending[key]
                    else:
                        raise ProtocolError(f"signal {kind} missing row for {key}")
                else:
                    rec[kind] = table[key]["value"]
                    rec[f"{kind}_reason"] = table[key]["reason"]
            rec["rewards"] = build_rewards(rec, cfg)
            joined.append(rec)
    return joined
