"""Operational reading of the protocol's falsifiable prediction.

Fixed before any model run (thresholds live in the config):

* self-confidence has coverage at least ``high_coverage`` (default 0.8)
* self-confidence AUROC is at most ``low_auroc_max`` (default 0.6)
* self-confidence's strategy ranking disagrees with mean correctness
  on every dataset (this is how "frequent ranking failures" is decided)
* lexical grounding has lower coverage than self-confidence
* lexical grounding's strategy ranking matches mean correctness

A dataset is ``supported`` when every clause holds, ``contradicted`` when
every required number is present and a clause fails, and ``pending`` when
a required number is missing. The overall verdict is ``pending`` until all
three datasets are decided, ``supported`` when each is supported, and
``contradicted`` when each is decided and at least one is contradicted.
"""
from __future__ import annotations

from typing import Any

REQUIRED_DATASETS = ("squad", "hotpot", "nq")


def dataset_verdict(self_m: dict[str, Any] | None, ground_m: dict[str, Any] | None,
                    high_coverage: float, low_auroc_max: float) -> dict[str, Any]:
    if not self_m or not ground_m:
        return {"verdict": "pending", "reasons": ["calibration rows for self or lexical grounding are missing"]}
    needed = {
        "self.coverage": self_m.get("coverage"),
        "self.auroc": self_m.get("auroc"),
        "self.rank_agrees": self_m.get("rank_agrees"),
        "grounding.coverage": ground_m.get("coverage"),
        "grounding.rank_agrees": ground_m.get("rank_agrees"),
    }
    missing = [name for name, val in needed.items() if val is None]
    if missing:
        return {"verdict": "pending", "reasons": [f"undefined: {', '.join(missing)}"]}
    checks = {
        "self_coverage_high": self_m["coverage"] >= high_coverage,
        "self_auroc_low": self_m["auroc"] <= low_auroc_max,
        "self_ranking_fails": self_m["rank_agrees"] is False,
        "grounding_coverage_lower": ground_m["coverage"] < self_m["coverage"],
        "grounding_ranking_agrees": ground_m["rank_agrees"] is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        return {"verdict": "contradicted", "reasons": failed, "checks": checks}
    return {"verdict": "supported", "reasons": [], "checks": checks}


def overall_verdict(per_dataset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [d for d in REQUIRED_DATASETS if d not in per_dataset]
    if missing:
        return {"verdict": "pending", "reasons": [f"datasets missing: {', '.join(missing)}"]}
    states = [per_dataset[d]["verdict"] for d in REQUIRED_DATASETS]
    if any(s == "pending" for s in states):
        return {"verdict": "pending", "reasons": ["at least one dataset is still pending"]}
    if all(s == "supported" for s in states):
        return {"verdict": "supported", "reasons": []}
    return {"verdict": "contradicted", "reasons": ["at least one dataset contradicted the prediction"]}
