"""Marker written when the judge model refuses the call.

Downstream stages keep running. Judge cells and verified-reward cells stay
pending with this reason instead of being scored as if the judge had run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.common import ProtocolError, read_json, write_json

JUDGE_ACCESS_REASON = "judge model access not yet granted"
MARKER_NAME = "judge_unavailable.json"


def marker_path(results: Path) -> Path:
    return results / "signals" / MARKER_NAME


def write_marker(results: Path, detail: str) -> None:
    write_json(marker_path(results), {"reason": JUDGE_ACCESS_REASON, "detail": detail})


def access_reason(results: Path) -> str | None:
    path = marker_path(results)
    if not path.is_file():
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        raise ProtocolError(f"{path} is not an object")
    reason = data.get("reason") or JUDGE_ACCESS_REASON
    return str(reason)


def clear_marker(results: Path) -> None:
    path = marker_path(results)
    if path.is_file():
        path.unlink()
