"""Join on-disk stage outputs into per-(qid, arm) records and reward values.

The verified reward is ``verified_reward.combine``: a weighted mean of the
signals that were observed, and missing when none were. Judge weight and
grounding weight come from the config. Self-confidence enters only when its
parse status is ``ok``.

Joined rows are the question ids in ``results/samples/``. Retrieval,
generation, and signal rows for any other question id are ignored.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))

import verified_reward as V  # noqa: E402

from experiments.common import ProtocolError, in_sample, read_json, read_jsonl, validate_rows
from experiments.confidence import FAILED, OMITTED, UNPARSEABLE
from experiments.judge_access import JUDGE_ACCESS_REASON, access_reason
from experiments.labels import decide_correctness
from experiments.ledger import SPEND_CAP_REASON

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
    allowed: dict[str, set[str]] = {}
    for name in names:
        path = sample_dir / f"{name}.jsonl"
        rows = read_jsonl(path)
        validate_rows(rows, ("qid", "dataset", "question", "question_type", "gold_answers", "unanswerable"), path)
        questions.extend(rows)
        allowed[name] = {row["qid"] for row in rows}
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
    retrieval_rows = [row for row in retrieval_rows if in_sample(row, allowed)]
    generation_rows = [row for row in generation_rows if in_sample(row, allowed)]
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
        rows = [row for row in rows if in_sample(row, allowed)]
        validate_rows(rows, ("qid", "arm", "value", "reason"), path)
        signals[kind] = _index(rows, path)
    for kind, fname in (
        ("self_percentile", "self_percentile.jsonl"),
        ("logistic", "logistic.jsonl"),
        ("oracle_correct", "oracle_correct.jsonl"),
        ("oracle_retrieval", "oracle_retrieval.jsonl"),
    ):
        path = results / "signals" / fname
        if not attach_signals:
            continue
        try:
            rows = read_jsonl(path)
        except ProtocolError:
            continue
        rows = [row for row in rows if in_sample(row, allowed)]
        validate_rows(rows, ("qid", "arm", "value", "reason"), path)
        signals[kind] = _index(rows, path)
    low = float(cfg["v2"]["adjudication_f1_low"])
    high = float(cfg["v2"]["adjudication_f1_high"])
    adjudication: dict[tuple[str, str], dict[str, Any]] | None = None
    adj_pending: dict[tuple[str, str], str] | None = None

    def _adjudication() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
        nonlocal adjudication, adj_pending
        if adjudication is None:
            adj_pending = _pending_keys(results / "adjudication" / "pending.json")
            path = results / "adjudication" / "adjudication.jsonl"
            try:
                rows = read_jsonl(path)
            except ProtocolError:
                rows = []
            adjudication = _index(rows, path) if rows else {}
        return adjudication, adj_pending or {}

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
        if "yes_no" not in q:
            raise ProtocolError(f"sample row {q['qid']} has no yes_no")
        decisions = []
        skip_question = False
        for arm in arms:
            key = (q["qid"], arm)
            gen = generation[key]
            ret = retrieval[key]
            for flag in ("gold_in_top_k", "source_retrieved"):
                if flag not in ret:
                    raise ProtocolError(f"retrieval row {key} has no {flag}")
            decision = decide_correctness(
                answer=gen["answer"],
                gold_answers=q["gold_answers"],
                unanswerable=bool(q["unanswerable"]),
                self_status=gen["self_status"],
                source_retrieved=bool(ret["source_retrieved"]),
                gold_in_top_k=bool(ret["gold_in_top_k"]),
                yes_no=bool(q["yes_no"]),
                f1_low=low,
                f1_high=high,
                stored_label=gen.get("abstention_label"),
                stored_counts=gen.get("abstention_counts_correct"),
                check_stored="abstention_label" in gen,
            )
            if decision["in_f1_band"]:
                table, pending = _adjudication()
                if key in pending:
                    skip_question = True
                    break
                if key not in table:
                    raise ProtocolError(
                        f"token F1 for {key} is {decision['f1']} inside [{low}, {high}]. "
                        "Run scripts/experiments/adjudicate.py. Refusing to use the 0.5 threshold."
                    )
                stored = table[key]
                if "f1" not in stored:
                    raise ProtocolError(f"adjudication for {key} has no f1")
                if abs(float(stored["f1"]) - float(decision["f1"])) > 1e-9:
                    raise ProtocolError(
                        f"adjudication for {key} stored f1 {stored.get('f1')} "
                        f"and the answer now scores {decision['f1']}."
                    )
                decision = decide_correctness(
                    answer=gen["answer"],
                    gold_answers=q["gold_answers"],
                    unanswerable=bool(q["unanswerable"]),
                    self_status=gen["self_status"],
                    source_retrieved=bool(ret["source_retrieved"]),
                    gold_in_top_k=bool(ret["gold_in_top_k"]),
                    yes_no=bool(q["yes_no"]),
                    f1_low=low,
                    f1_high=high,
                    stored_label=gen.get("abstention_label"),
                    stored_counts=gen.get("abstention_counts_correct"),
                    check_stored="abstention_label" in gen,
                    adjudication=stored,
                )
            decisions.append((arm, gen, ret, decision))
        if skip_question:
            continue
        for arm, gen, ret, decision in decisions:
            key = (q["qid"], arm)
            if decision["correct"] is None:
                raise ProtocolError(f"joined correctness for {key} is missing")
            rec = {
                "qid": q["qid"],
                "dataset": q["dataset"],
                "arm": arm,
                "question_type": q["question_type"],
                "question": q["question"],
                "answer": gen["answer"],
                "claim": gen.get("claim"),
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
                "f1": decision["f1"],
                "token_f1_correct": decision["token_f1_correct"],
                "correct": decision["correct"],
                "correct_source": decision["correct_source"],
                "gold_contained": decision["gold_contained"],
                "abstention_label": decision["abstention_label"],
                "abstention_counts_correct": decision["abstention_counts_correct"],
                "gold_in_top_k": decision["gold_in_top_k"],
                "source_retrieved": decision["source_retrieved"],
                "yes_no": decision["yes_no"],
                "in_f1_band": decision["in_f1_band"],
                "passages": [p["text"] for p in ret["passages"]],
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
