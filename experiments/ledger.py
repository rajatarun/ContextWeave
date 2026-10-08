"""One spend ledger for generation, the judge, and adjudication.

Generation, the judge, and adjudication append a row per paid call to
``results/cost_ledger.jsonl`` (or ``.jsonl.gz``). Before a call, a stage
refuses to proceed when its own ``--max-usd`` or the global
``--total-usd-cap`` (default 30) would be crossed. The cap check uses the
estimate; the row that was already paid for is kept.

The cap sums each row's ``usd`` and adds ``already_spent_usd`` from the
config. That value is spend already on the AWS bill from prior ledgers that
are not in this results directory. The total-cap check refuses a job when
that sum plus the projected job cost times ``spend_safety_factor`` exceeds
``total_usd_cap``. The stage ``--max-usd`` check does not apply the factor.
The projected output length is ``expected_output_tokens`` until this results
ledger has a numeric output mean for that model. ``scripts/experiments/reprice_ledger.py``
rewrites ``usd`` from the current price after a price change, so the ledger
half of the cap tracks the bill.
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


def _output_sample(row: dict[str, Any]) -> tuple[str, float] | None:
    """Numeric observed output tokens, or None when the row cannot enter a mean."""
    if row.get("usage_observed") is False:
        return None
    model = row.get("model_id")
    value = row.get("output_tokens")
    if not isinstance(model, str) or not model:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return model, float(value)


def observed_output_mean(results: Path, model_id: str) -> tuple[float, int] | None:
    """Mean numeric ``output_tokens`` for ``model_id`` in this results ledger.

    Rows with ``usage_observed`` false, a bool, or a non-numeric count are
    skipped. No such rows returns None.
    """
    total = 0.0
    n = 0
    for row in read_ledger(results):
        sample = _output_sample(row)
        if sample is None or sample[0] != model_id:
            continue
        total += sample[1]
        n += 1
    if n < 1:
        return None
    return total / n, n


def expected_output_tokens(cfg: dict[str, Any], model_id: str) -> float:
    table = cfg.get("expected_output_tokens")
    if not isinstance(table, dict) or model_id not in table:
        known = ", ".join(sorted(table)) if isinstance(table, dict) else "(missing)"
        raise ProtocolError(
            f"expected_output_tokens has no entry for {model_id!r}. Known: {known}. "
            "Refusing to price the request maximum."
        )
    value = table[model_id]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ProtocolError(
            f"expected_output_tokens for {model_id} is {value!r}. It must be a number > 0."
        )
    return float(value)


def planned_output_tokens(
    cfg: dict[str, Any],
    model_id: str,
    *,
    budget: "Budget | None" = None,
    results: Path | None = None,
) -> tuple[float, str]:
    """Output length to price: this run's ledger mean, else the config figure.

    A mean is the numeric ``output_tokens`` already recorded for ``model_id``.
    The request ``maxTokens`` cap is not a fallback.
    """
    if budget is not None:
        observed = budget.observed_output_mean(model_id)
        if observed is not None:
            return observed[0], "ledger_mean"
    elif results is not None:
        observed = observed_output_mean(results, model_id)
        if observed is not None:
            return observed[0], "ledger_mean"
    return expected_output_tokens(cfg, model_id), "expected_output_tokens"


def spent_usd(results: Path, stage: str | None = None) -> float:
    total = 0.0
    for row in read_ledger(results):
        if stage is not None and row.get("stage") != stage:
            continue
        total += float(row.get("usd") or 0.0)
    return total


class Budget:
    """Stage cap and global cap.

    ``global_spent`` is this results ledger plus ``already_spent_usd``. The
    total-cap check is that sum plus the next estimate times
    ``spend_safety_factor``. The stage cap uses only this stage's rows in
    the ledger and does not apply the factor. Output means loaded here are
    what later estimates in this process price, and ``record`` keeps them
    current so the next job sees rows written by the previous one.
    """

    def __init__(
        self,
        results: Path,
        stage: str,
        max_usd: float,
        total_usd_cap: float,
        already_spent_usd: float = 0.0,
        spend_safety_factor: float = 1.0,
    ):
        if max_usd <= 0:
            raise ProtocolError("--max-usd must be a positive number")
        if total_usd_cap <= 0:
            raise ProtocolError("--total-usd-cap must be a positive number")
        if isinstance(already_spent_usd, bool) or not isinstance(already_spent_usd, (int, float)):
            raise ProtocolError(f"already_spent_usd is {already_spent_usd!r}")
        if already_spent_usd < 0:
            raise ProtocolError(f"already_spent_usd is {already_spent_usd}. It must be >= 0.")
        if isinstance(spend_safety_factor, bool) or not isinstance(spend_safety_factor, (int, float)):
            raise ProtocolError(f"spend_safety_factor is {spend_safety_factor!r}")
        if spend_safety_factor <= 0:
            raise ProtocolError(
                f"spend_safety_factor is {spend_safety_factor}. It must be > 0."
            )
        self.results = results
        self.stage = stage
        self.max_usd = float(max_usd)
        self.total_usd_cap = float(total_usd_cap)
        self.already_spent_usd = float(already_spent_usd)
        self.spend_safety_factor = float(spend_safety_factor)
        self.ledger_spent = spent_usd(results)
        self.global_spent = self.ledger_spent + self.already_spent_usd
        self.stage_spent = spent_usd(results, stage)
        self.path = ledger_file(results)
        self._output_sum: dict[str, float] = {}
        self._output_n: dict[str, int] = {}
        for row in read_ledger(results):
            self._note_output(row)

    def cap_label(self) -> str:
        return (
            f"ledger ${self.ledger_spent:.6f} plus already billed ${self.already_spent_usd:.6f} "
            f"(${self.global_spent:.6f}) against total cap ${self.total_usd_cap:.6f}"
        )

    def _note_output(self, row: dict[str, Any]) -> None:
        sample = _output_sample(row)
        if sample is None:
            return
        model, value = sample
        self._output_sum[model] = self._output_sum.get(model, 0.0) + value
        self._output_n[model] = self._output_n.get(model, 0) + 1

    def observed_output_mean(self, model_id: str) -> tuple[float, int] | None:
        n = self._output_n.get(model_id, 0)
        if n < 1:
            return None
        return self._output_sum[model_id] / n, n

    def blocking_reason(self, next_usd: float) -> str | None:
        if self.stage_spent + next_usd > self.max_usd + 1e-12:
            return (
                f"{SPEND_CAP_REASON}: {self.stage} spent ${self.stage_spent:.6f}, "
                f"next call estimated ${next_usd:.6f}, stage cap ${self.max_usd:.6f}"
            )
        guarded = next_usd * self.spend_safety_factor
        if self.global_spent + guarded > self.total_usd_cap + 1e-12:
            return (
                f"{SPEND_CAP_REASON}: {self.cap_label()}, "
                f"next call estimated ${next_usd:.6f} times safety {self.spend_safety_factor:.2f} "
                f"(${guarded:.6f})"
            )
        return None

    def record(self, row: dict[str, Any]) -> None:
        usd = float(row["usd"])
        self.stage_spent += usd
        self.ledger_spent += usd
        self.global_spent += usd
        body = {"stage": self.stage, "spent_after": self.global_spent, "stage_spent_after": self.stage_spent}
        body.update(row)
        append_jsonl(self.path, body)
        self._note_output(body)


def budget_for(
    results: Path,
    stage: str,
    max_usd: float,
    cfg: dict[str, Any],
    total_usd_cap: float | None = None,
) -> Budget:
    """Budget whose total cap includes billed spend and the safety factor."""
    cap = float(cfg["total_usd_cap"]) if total_usd_cap is None else float(total_usd_cap)
    return Budget(
        results, stage, max_usd, cap,
        already_spent_usd=float(cfg["already_spent_usd"]),
        spend_safety_factor=float(cfg["spend_safety_factor"]),
    )
