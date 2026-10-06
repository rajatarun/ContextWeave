"""Verdicts for the claims the offline protocol can decide.

Each verdict is supported, contradicted, or open. Open means the artifact
that would decide it is missing or the datasets disagree. The rules are
fixed here, in advance of the model run. They are stated in terms of the
artifacts, not copied from the paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.common import read_json
from experiments.judge_access import JUDGE_ACCESS_REASON, access_reason

DATASETS = ("squad", "hotpot", "nq")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


def _estimate(block: Any) -> float | None:
    if isinstance(block, dict) and isinstance(block.get("estimate"), (int, float)):
        return float(block["estimate"])
    return None


def _open(name: str, because: str, artifacts: list[str]) -> dict[str, Any]:
    return {"id": name, "verdict": "open", "because": because, "artifacts": artifacts}


def _done(name: str, verdict: str, because: str, artifacts: list[str]) -> dict[str, Any]:
    return {"id": name, "verdict": verdict, "because": because, "artifacts": artifacts}


def assumption_1(analyses: dict[str, Any] | None) -> dict[str, Any]:
    """Self-confidence given correctness does not depend on the arm.

    Contradicted when two arms have non-overlapping bootstrap intervals for
    E[R | Y, arm] on any dataset. Supported when every dataset has estimable
    intervals and no such pair.
    """
    cite = ["results/analyses/analyses.json"]
    if analyses is None:
        return _open("Assumption 1", "analyses artifact is missing", cite)
    datasets = analyses.get("datasets") or {}
    contradicted = False
    for name in DATASETS:
        block = (datasets.get(name) or {}).get("self") or {}
        assumption = block.get("assumption") or {}
        flags = assumption.get("nonoverlapping_pairs")
        by_arm = assumption.get("by_arm") or {}
        if not isinstance(flags, list) or len(by_arm) < 2:
            return _open("Assumption 1", f"{name} has no arm-conditional intervals for self-confidence", cite)
        estimable = 0
        for arm_block in by_arm.values():
            for y in ("y0", "y1"):
                if _estimate((arm_block or {}).get(y)) is not None:
                    estimable += 1
                    break
        if estimable < 2:
            return _open("Assumption 1", f"{name} does not have two arms with a finite conditional interval", cite)
        if flags:
            contradicted = True
    if contradicted:
        return _done(
            "Assumption 1", "contradicted",
            "at least one dataset has non-overlapping E[self-confidence | Y, arm] intervals",
            cite,
        )
    return _done(
        "Assumption 1", "supported",
        "on every dataset the arm-conditional intervals for self-confidence overlap",
        cite,
    )


def proposition_1(calibration: dict[str, Any] | None) -> dict[str, Any]:
    """Arm offsets can make the self-confidence ranking differ from correctness.

    Supported when any dataset's self-confidence strategy ranking disagrees
    with mean correctness. Contradicted when every dataset's rankings agree.
    """
    cite = ["results/calibration/calibration.json"]
    if calibration is None:
        return _open("Proposition 1", "calibration artifact is missing", cite)
    disagreements = []
    for name in DATASETS:
        signal = (((calibration.get("datasets") or {}).get(name) or {}).get("signals") or {}).get("self")
        if not isinstance(signal, dict) or not isinstance(signal.get("rank_agrees"), bool):
            return _open("Proposition 1", f"{name} has no self-confidence ranking check", cite)
        if signal["rank_agrees"] is False:
            disagreements.append(name)
    if disagreements:
        return _done(
            "Proposition 1", "supported",
            "self-confidence ranks strategies differently from mean correctness on " + ", ".join(disagreements),
            cite,
        )
    return _done(
        "Proposition 1", "contradicted",
        "self-confidence ranks strategies in the same order as mean correctness on every dataset",
        cite,
    )


def proposition_2(analyses: dict[str, Any] | None) -> dict[str, Any]:
    """Filling unobserved confidences with the deployed constants can reorder arms.

    Compared with skipping those rounds. Supported when the two orderings differ
    on a dataset where every arm has both means. Contradicted when every
    dataset is comparable and the orderings match.
    """
    cite = ["results/analyses/analyses.json"]
    if analyses is None:
        return _open("Proposition 2", "analyses artifact is missing", cite)
    reversed_on = []
    for name in DATASETS:
        block = (analyses.get("datasets") or {}).get(name) or {}
        fallback = block.get("fallback")
        if not isinstance(fallback, dict) or "order_reversed" not in fallback:
            return _open("Proposition 2", f"{name} has no fallback-ordering block", cite)
        if not fallback.get("order_skip") or not fallback.get("order_fallback"):
            return _open("Proposition 2", f"{name} cannot compare skip-unobserved and fallback means", cite)
        if fallback["order_reversed"]:
            reversed_on.append(name)
    if reversed_on:
        return _done(
            "Proposition 2", "supported",
            "fallback constants reorder arms relative to skip-unobserved on " + ", ".join(reversed_on),
            cite,
        )
    return _done(
        "Proposition 2", "contradicted",
        "fallback constants and skip-unobserved rank the arms the same way on every dataset",
        cite,
    )


def theorem_1_scale(analyses: dict[str, Any] | None, replay: dict[str, Any] | None, fraction: float) -> dict[str, Any]:
    """The scale objection: per-arm normalization of self-confidence removes the regret gap.

    The gap is fractional pseudo-regret of self minus lexical grounding.
    Supported when normalization closes at least ``fraction`` of that gap on
    every dataset and self's slope is smaller than grounding's. Contradicted
    when the closed fraction is below that on every dataset. Open when the
    datasets disagree or a number is missing.
    """
    cite = ["results/analyses/analyses.json", "results/replay/replay_summary.json"]
    if analyses is None or replay is None:
        return _open("Theorem 1 scale objection", "analyses or replay artifact is missing", cite)
    closed = []
    for name in DATASETS:
        signals = (analyses.get("datasets") or {}).get(name) or {}
        s_self = _estimate(((signals.get("self") or {}).get("slope") or {}).get("s"))
        s_ground = _estimate(((signals.get("lexical_grounding") or {}).get("slope") or {}).get("s"))
        rewards = (replay.get("datasets") or {}).get(name) or {}
        def regret(reward: str) -> float | None:
            block = ((rewards.get(reward) or {}).get("fractional") or {})
            value = block.get("pseudo_regret_mean")
            return float(value) if isinstance(value, (int, float)) else None
        r_self, r_norm, r_ground = regret("self"), regret("normalized_self"), regret("lexical_grounding")
        if None in (s_self, s_ground, r_self, r_norm, r_ground):
            return _open("Theorem 1 scale objection", f"{name} is missing a slope or a fractional pseudo-regret", cite)
        assert s_self is not None and s_ground is not None
        assert r_self is not None and r_norm is not None and r_ground is not None
        if not (s_self < s_ground and r_self > r_ground):
            return _open(
                "Theorem 1 scale objection",
                f"{name} does not have a smaller self slope and a higher self regret than lexical grounding",
                cite,
            )
        gap = r_self - r_ground
        fraction_closed = (r_self - r_norm) / gap if gap else None
        if fraction_closed is None:
            return _open("Theorem 1 scale objection", f"{name} has a zero regret gap", cite)
        closed.append(fraction_closed >= fraction)
    if all(closed):
        return _done(
            "Theorem 1 scale objection", "supported",
            f"per-arm normalization closes at least {fraction:.2f} of the self-versus-grounding regret gap on every dataset",
            cite,
        )
    if not any(closed):
        return _done(
            "Theorem 1 scale objection", "contradicted",
            f"per-arm normalization closes less than {fraction:.2f} of that gap on every dataset",
            cite,
        )
    return _open("Theorem 1 scale objection", "datasets disagree on whether normalization closes the regret gap", cite)


def proposition_3(analyses: dict[str, Any] | None, judge_reason: str | None) -> dict[str, Any]:
    """A positive-slope combination of positive-slope signals stays positive.

    Measured on rows where lexical grounding and the judge were both observed.
    """
    cite = ["results/analyses/analyses.json", "results/signals/judge_unavailable.json"]
    if judge_reason:
        return _open("Proposition 3", judge_reason, cite)
    if analyses is None:
        return _open("Proposition 3", "analyses artifact is missing", cite)
    positive = []
    for name in DATASETS:
        combo = ((analyses.get("datasets") or {}).get(name) or {}).get("combination")
        if not isinstance(combo, dict):
            return _open("Proposition 3", f"{name} has no rows where grounding and the judge were both observed", cite)
        slopes = {
            part: _estimate(((combo.get(part) or {}).get("s")))
            for part in ("grounding", "judge", "verified")
        }
        if any(value is None for value in slopes.values()):
            return _open("Proposition 3", f"{name} combination slope is undefined", cite)
        assert slopes["grounding"] is not None and slopes["judge"] is not None and slopes["verified"] is not None
        if slopes["grounding"] > 0 and slopes["judge"] > 0 and slopes["verified"] <= 0:
            return _done(
                "Proposition 3", "contradicted",
                f"{name} has positive component slopes and a non-positive combined slope",
                cite,
            )
        positive.append(slopes["grounding"] > 0 and slopes["judge"] > 0 and slopes["verified"] > 0)
    if all(positive):
        return _done(
            "Proposition 3", "supported",
            "on every dataset the combined reward has a positive slope when both components do",
            cite,
        )
    return _open("Proposition 3", "a component slope is not positive, so the combination claim is not triggered", cite)


def section_6(calibration: dict[str, Any] | None) -> dict[str, Any]:
    """High-coverage low-AUROC self-confidence, and grounding that ranks strategies."""
    cite = ["results/calibration/calibration.json"]
    if calibration is None:
        return _open("Section 6 prediction", "calibration artifact is missing", cite)
    verdict = (calibration.get("prediction") or {}).get("verdict")
    if verdict == "supported":
        return _done("Section 6 prediction", "supported", "every dataset met the pre-registered prediction", cite)
    if verdict == "contradicted":
        return _done("Section 6 prediction", "contradicted", "a decided dataset failed the pre-registered prediction", cite)
    return _open("Section 6 prediction", "the prediction is not decided for every dataset", cite)


def claim_verdicts(results: Path, gap_fraction: float) -> list[dict[str, Any]]:
    analyses = _load(results / "analyses" / "analyses.json")
    calibration = _load(results / "calibration" / "calibration.json")
    replay = _load(results / "replay" / "replay_summary.json")
    judge_reason = access_reason(results)
    if judge_reason and judge_reason != JUDGE_ACCESS_REASON:
        judge_reason = JUDGE_ACCESS_REASON
    return [
        assumption_1(analyses),
        proposition_1(calibration),
        proposition_2(analyses),
        theorem_1_scale(analyses, replay, gap_fraction),
        proposition_3(analyses, judge_reason),
        section_6(calibration),
    ]
