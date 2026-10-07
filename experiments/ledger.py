"""One spend ledger for generation and the judge.

Both stages append a row per paid call to ``results/cost_ledger.jsonl`` (or
``.jsonl.gz``). Before a call, a stage refuses to proceed when its own
``--max-usd`` or the global ``--total-usd-cap`` (default 30) would be crossed.
The cap check uses the estimate; the row that was already paid for is kept.

The cap sums each row's ``usd``. Generation and the judge write that field
from the current config price and the call's token counts.
``scripts/experiments/reprice_ledger.py`` rewrites ``usd`` the same way after
a price change, so the cap tracks the bill.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.common import ProtocolError, append_jsonl, locate_jsonl, read_jsonl, stage_jsonl

SPEND_CAP_REASON = "spend cap reached"


def ledger_file(results: Path) -> Path:
    return stage_jsonl(results, "cost_ledger")


def read_ledger(results: Path) -> list[dict[str, Any]]:
    try:
        return read_jsonl(locate_jsonl(results / "cost_ledger.jsonl"))
    except ProtocolError:
        return []


def spent_usd(results: Path, stage: str | None = None) -> float:
    total = 0.0
    for row in read_ledger(results):
        if stage is not None and row.get("stage") != stage:
            continue
        total += float(row.get("usd") or 0.0)
    return total


class Budget:
    """Stage cap and global cap, both measured from the shared ledger."""

    def __init__(self, results: Path, stage: str, max_usd: float, total_usd_cap: float):
        if max_usd <= 0:
            raise ProtocolError("--max-usd must be a positive number")
        if total_usd_cap <= 0:
            raise ProtocolError("--total-usd-cap must be a positive number")
        self.results = results
        self.stage = stage
        self.max_usd = float(max_usd)
        self.total_usd_cap = float(total_usd_cap)
        self.global_spent = spent_usd(results)
        self.stage_spent = spent_usd(results, stage)
        self.path = ledger_file(results)

    def blocking_reason(self, next_usd: float) -> str | None:
        if self.stage_spent + next_usd > self.max_usd + 1e-12:
            return (
                f"{SPEND_CAP_REASON}: {self.stage} spent ${self.stage_spent:.6f}, "
                f"next call estimated ${next_usd:.6f}, stage cap ${self.max_usd:.6f}"
            )
        if self.global_spent + next_usd > self.total_usd_cap + 1e-12:
            return (
                f"{SPEND_CAP_REASON}: ledger total ${self.global_spent:.6f}, "
                f"next call estimated ${next_usd:.6f}, total cap ${self.total_usd_cap:.6f}"
            )
        return None

    def record(self, row: dict[str, Any]) -> None:
        usd = float(row["usd"])
        self.stage_spent += usd
        self.global_spent += usd
        body = {"stage": self.stage, "spent_after": self.global_spent, "stage_spent_after": self.stage_spent}
        body.update(row)
        append_jsonl(self.path, body)
