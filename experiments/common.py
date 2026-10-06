"""Shared config, artifact metadata, JSONL resume, and cost accounting."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "config.yaml"
RESULTS = ROOT / "results"

ARMS = ("semantic_search", "graph_first", "keyword_boosted", "hybrid")
DATASETS = ("squad", "hotpot", "nq")

SAMPLE_FIELDS = (
    "qid", "dataset", "question", "question_type", "gold_answers",
    "unanswerable", "pool",
)
RETRIEVAL_FIELDS = ("qid", "dataset", "arm", "question_type", "passages")
GENERATION_FIELDS = (
    "qid", "dataset", "arm", "question_type", "answer", "raw_response",
    "self_confidence", "self_reported", "self_status", "input_tokens",
    "output_tokens", "model_id", "error",
)
SIGNAL_FIELDS = ("qid", "dataset", "arm", "value", "reason")


class ProtocolError(SystemExit):
    """A missing input, schema violation, or refused fallback. Message is the error."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_PATH
    if not path.is_file():
        raise ProtocolError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ProtocolError(f"config is not a mapping: {path}")
    required = (
        "seed", "n_per_dataset", "top_k", "embedding_model", "nli_model",
        "grounding_threshold", "judge_sample_rate", "grounding_weight",
        "judge_weight", "self_weight", "generator_model_id", "judge_model_id",
        "region", "temperature", "generator_max_output_tokens",
        "judge_max_output_tokens", "prices_usd_per_million_tokens",
        "fallback_omitted", "fallback_unparseable", "fallback_failed",
        "keyword_weight", "bm25_k1", "bm25_b", "bootstrap_samples",
        "replay_seeds", "high_coverage", "low_auroc_max",
        "nq_window_chars", "nq_overlap_chars",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ProtocolError(f"config {path} is missing keys: {missing}")
    prices = data["prices_usd_per_million_tokens"]
    if not isinstance(prices, dict) or not prices:
        raise ProtocolError("prices_usd_per_million_tokens is empty")
    for model_id, row in prices.items():
        if not isinstance(row, dict) or "input" not in row or "output" not in row:
            raise ProtocolError(f"price row for {model_id} needs input and output")
        if row["input"] is None or row["output"] is None:
            raise ProtocolError(f"price row for {model_id} has a null price")
    return data


def price_for(cfg: dict[str, Any], model_id: str) -> dict[str, Any]:
    table = cfg["prices_usd_per_million_tokens"]
    if model_id not in table:
        known = ", ".join(sorted(table))
        raise ProtocolError(
            f"no price for model {model_id!r}. The price table has: {known}. "
            "Refusing to guess a price."
        )
    return table[model_id]


_SNAPSHOT_RE = re.compile(r"/snapshots/([0-9a-f]{40})(?:/|$)")


def snapshot_revision(*values: Any) -> str | None:
    """Revision of a Hugging Face snapshot.

    A 40-hex commit is returned as itself. A local cache path of the form
    ``.../snapshots/<commit>/...`` yields that commit. The library often
    leaves ``config._commit_hash`` empty after a cache load, while the
    tokenizer file path still names the snapshot.
    """
    for value in values:
        if value is None:
            continue
        text = str(value)
        if re.fullmatch(r"[0-9a-f]{40}", text):
            return text
        match = _SNAPSHOT_RE.search(text)
        if match:
            return match.group(1)
    return None


def estimate_tokens(text: str) -> int:
    """Approximate token count: ceil(utf-8 bytes / 4). Not a provider tokenizer."""
    if not isinstance(text, str):
        raise ProtocolError("token estimate expected a string")
    n = len(text.encode("utf-8"))
    if n == 0:
        return 0
    return (n + 3) // 4


def cost_usd(cfg: dict[str, Any], model_id: str, input_tokens: int, output_tokens: int) -> float:
    row = price_for(cfg, model_id)
    return (input_tokens * float(row["input"]) + output_tokens * float(row["output"])) / 1_000_000


def git_meta() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=ROOT,
        ) != 0
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ProtocolError(f"git metadata is required and could not be read: {exc}") from exc
    return {"commit": sha, "dirty": dirty}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_meta(cfg: dict[str, Any], seed: int | Sequence[int], **extra: Any) -> dict[str, Any]:
    meta = {
        "created_at": now_iso(),
        "seed": seed,
        "git": git_meta(),
        "config": cfg,
    }
    meta.update(extra)
    return meta


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise ProtocolError(f"missing JSON file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"{path}: invalid JSON: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{path}:{i}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ProtocolError(f"{path}:{i}: expected a JSON object")
            rows.append(row)
    return rows


def validate_rows(rows: Sequence[dict[str, Any]], required: Sequence[str], path: Path) -> None:
    for i, row in enumerate(rows, 1):
        missing = [k for k in required if k not in row]
        if missing:
            raise ProtocolError(f"{path}:{i}: missing keys {missing}")


def done_keys(path: Path, fields: Sequence[str]) -> set[tuple]:
    if not path.is_file():
        return set()
    keys = set()
    for row in read_jsonl(path):
        validate_rows([row], fields, path)
        keys.add(tuple(row[f] for f in fields))
    return keys


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def parse_seed_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ProtocolError("seed list is empty")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ProtocolError(f"seed list {text!r} is not integers") from exc


def require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise ProtocolError(f"{what} not found: {path}")


def row_key(row: dict[str, Any], fields: Iterable[str] = ("qid", "arm")) -> tuple:
    return tuple(row[f] for f in fields)
