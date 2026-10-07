#!/usr/bin/env python3
"""Rewrite a cost ledger so each row's usd matches the current config price.

The default ledger is ``results/cost_ledger.jsonl.gz``. Pass ``--ledger`` to
choose another file. Each row's ``usd`` is recomputed from its
``input_tokens`` and ``output_tokens`` at the config price for that row's
``model_id``. The previous ``usd`` is stored as ``usd_at_logged_price`` and
left in place when the row was already repriced. The row also stores the
config price and one UTC timestamp for the run. The file is replaced by
writing a temporary file in the same directory and renaming it. A row
without token counts stops the script and leaves the ledger unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (  # noqa: E402
    RESULTS, ProtocolError, cost_usd, load_config, now_iso, price_for,
)


DEFAULT_LEDGER = RESULTS / "cost_ledger.jsonl.gz"


def _token_count(row: dict[str, Any], field: str, where: str) -> int:
    if field not in row or row[field] is None:
        raise ProtocolError(f"{where}: missing {field}. Refusing to guess a cost.")
    value = row[field]
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ProtocolError(f"{where}: {field} is {value!r}. Refusing to guess a cost.")
    if not isinstance(value, (int, float)):
        raise ProtocolError(f"{where}: {field} is {value!r}. Refusing to guess a cost.")
    count = int(value)
    if count < 0:
        raise ProtocolError(f"{where}: {field} is negative. Refusing to guess a cost.")
    return count


def _logged_usd(row: dict[str, Any], where: str) -> float:
    if "usd" not in row or row["usd"] is None or isinstance(row["usd"], bool):
        raise ProtocolError(f"{where}: missing usd. Refusing to guess the logged total.")
    if not isinstance(row["usd"], (int, float)):
        raise ProtocolError(f"{where}: usd is {row['usd']!r}. Refusing to guess the logged total.")
    return float(row["usd"])


def reprice_rows(cfg: dict[str, Any], rows: list[dict[str, Any]], when: str,
                 where: str) -> tuple[list[dict[str, Any]], float, float]:
    """Return repriced copies plus the old and new totals.

    ``usd_at_logged_price`` is the usd stored before the first reprice. A
    later run keeps that field and still recomputes ``usd``.
    """
    old_total = 0.0
    new_total = 0.0
    updated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        place = f"{where}:{index}"
        if not isinstance(row, dict):
            raise ProtocolError(f"{place}: expected a JSON object")
        model_id = row.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ProtocolError(f"{place}: missing model_id. Refusing to guess a price.")
        price = price_for(cfg, model_id)
        in_tok = _token_count(row, "input_tokens", place)
        out_tok = _token_count(row, "output_tokens", place)
        old_usd = _logged_usd(row, place)
        new_usd = cost_usd(cfg, model_id, in_tok, out_tok)
        body = dict(row)
        if body.get("usd_at_logged_price") is None:
            body["usd_at_logged_price"] = old_usd
        body["usd"] = new_usd
        body["price_usd_per_million"] = {
            "input": float(price["input"]),
            "output": float(price["output"]),
        }
        body["repriced_at"] = when
        updated.append(body)
        old_total += old_usd
        new_total += new_usd
    return updated, old_total, new_total


def read_ledger_file(path: Path) -> list[dict[str, Any]]:
    """Read ``path`` itself. A sibling ``.jsonl`` / ``.jsonl.gz`` is not a substitute."""
    if not path.is_file():
        raise ProtocolError(f"ledger not found: {path}")
    opener = gzip.open if path.name.endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for index, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{path}:{index}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ProtocolError(f"{path}:{index}: expected a JSON object")
            rows.append(row)
    return rows


def write_ledger_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to a temporary file in ``path``'s directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(tmp, "wt", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def reprice_ledger(cfg: dict[str, Any], path: Path) -> tuple[float, float, int]:
    rows = read_ledger_file(path)
    updated, old_total, new_total = reprice_rows(cfg, rows, now_iso(), str(path))
    write_ledger_atomic(path, updated)
    return old_total, new_total, len(updated)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                    help="cost ledger to rewrite (default: results/cost_ledger.jsonl.gz)")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        old_total, new_total, n = reprice_ledger(cfg, args.ledger)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"rows: {n}")
    print(f"old total USD: {old_total:.6f}")
    print(f"new total USD: {new_total:.6f}")
    print(f"wrote {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
