"""Shared config, artifact metadata, JSONL resume, and cost accounting."""
from __future__ import annotations

import gzip
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

SAMPLE_SCHEMA_VERSION = 2
GENERATION_SCHEMA_VERSION = 2

SAMPLE_FIELDS = (
    "qid", "dataset", "question", "question_type", "gold_answers",
    "unanswerable", "source_passage_ids", "yes_no", "schema_version",
)
RETRIEVAL_FIELDS = ("qid", "dataset", "arm", "question_type", "passages")
GENERATION_FIELDS = (
    "qid", "dataset", "arm", "question_type", "answer", "claim", "raw_response",
    "self_confidence", "self_reported", "self_status", "input_tokens",
    "output_tokens", "model_id", "error", "schema_version", "pricing",
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
        "subset_generator_model_id", "adjudicator_model_id",
        "region", "temperature", "generator_max_output_tokens",
        "judge_max_output_tokens", "prices_usd_per_million_tokens",
        "fallback_omitted", "fallback_unparseable", "fallback_failed",
        "keyword_weight", "bm25_k1", "bm25_b", "bootstrap_samples",
        "replay_seeds", "high_coverage", "low_auroc_max",
        "nq_window_chars", "nq_overlap_chars", "nq_pool_size",
        "nli_max_tokens", "nli_special_tokens",
        "normalized_self_gap_fraction", "total_usd_cap",
        "batch_prices_usd_per_million_tokens", "inference_mode", "batch",
        "spacy_model", "graph_damping", "graph_max_iter", "graph_tol",
        "schema_version", "v2",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ProtocolError(f"config {path} is missing keys: {missing}")
    if data["schema_version"] != SAMPLE_SCHEMA_VERSION:
        raise ProtocolError(
            f"config schema_version must be {SAMPLE_SCHEMA_VERSION}, got {data['schema_version']!r}"
        )
    if data["inference_mode"] not in ("on_demand", "batch"):
        raise ProtocolError(
            f"inference_mode must be on_demand or batch, got {data['inference_mode']!r}"
        )
    batch = data["batch"]
    if not isinstance(batch, dict):
        raise ProtocolError("config batch must be a mapping")
    for key in ("role_arn_env", "bucket_env", "prefix", "min_records", "poll_seconds", "timeout_hours"):
        if key not in batch:
            raise ProtocolError(f"config batch is missing {key}")
    if int(batch["min_records"]) < 1:
        raise ProtocolError("batch.min_records must be at least 1")
    v2 = data["v2"]
    if not isinstance(v2, dict):
        raise ProtocolError("config v2 must be a mapping")
    for key in (
        "judge_sample_rate", "held_out_fraction", "subset_questions",
        "subset_haiku_samples", "subset_temperature", "replay_seeds",
        "replay_rounds", "judge_coverage", "drift_discounts",
        "adjudication_f1_low", "adjudication_f1_high",
    ):
        if key not in v2:
            raise ProtocolError(f"config v2 is missing {key}")
    _validate_price_table(data["prices_usd_per_million_tokens"], "prices_usd_per_million_tokens")
    _validate_price_table(
        data["batch_prices_usd_per_million_tokens"], "batch_prices_usd_per_million_tokens",
    )
    return data


def _validate_price_table(prices: Any, name: str) -> None:
    if not isinstance(prices, dict) or not prices:
        raise ProtocolError(f"{name} is empty")
    for model_id, row in prices.items():
        if not isinstance(row, dict) or "input" not in row or "output" not in row:
            raise ProtocolError(f"price row for {model_id} in {name} needs input and output")
        if row["input"] is None or row["output"] is None:
            raise ProtocolError(f"price row for {model_id} in {name} has a null price")


def price_table(cfg: dict[str, Any], pricing: str = "on_demand") -> dict[str, Any]:
    if pricing == "on_demand":
        return cfg["prices_usd_per_million_tokens"]
    if pricing == "batch":
        table = cfg.get("batch_prices_usd_per_million_tokens")
        if not isinstance(table, dict) or not table:
            raise ProtocolError(
                "batch_prices_usd_per_million_tokens is missing. Refusing to guess a batch price."
            )
        return table
    raise ProtocolError(f"unknown pricing mode {pricing!r}. Expected on_demand or batch.")


def price_for(cfg: dict[str, Any], model_id: str, pricing: str = "on_demand") -> dict[str, Any]:
    table = price_table(cfg, pricing)
    if model_id not in table:
        known = ", ".join(sorted(table))
        raise ProtocolError(
            f"no {pricing} price for model {model_id!r}. The price table has: {known}. "
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


def cost_usd(
    cfg: dict[str, Any], model_id: str, input_tokens: int, output_tokens: int,
    pricing: str = "on_demand",
) -> float:
    row = price_for(cfg, model_id, pricing)
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


def locate_jsonl(path: Path) -> Path:
    """Find ``path`` or its ``.jsonl`` / ``.jsonl.gz`` sibling.

    Readers accept either spelling so a stage can resume a plain file written
    earlier and still open a gzipped file written later.
    """
    if path.is_file():
        return path
    text = str(path)
    alt = Path(text[:-3]) if text.endswith(".gz") else Path(text + ".gz")
    if alt.is_file():
        return alt
    raise ProtocolError(f"missing JSONL file: {path}")


def stage_jsonl(directory: Path, stem: str) -> Path:
    """Where a raw stage file is read or appended.

    An existing ``.jsonl`` or ``.jsonl.gz`` is kept. A new file is gzipped.
    Both present at once is an error: the reader would have to guess which
    copy is current.
    """
    plain = directory / f"{stem}.jsonl"
    gz = directory / f"{stem}.jsonl.gz"
    if plain.is_file() and gz.is_file():
        raise ProtocolError(f"both {plain} and {gz} exist; refusing to guess which is current")
    if gz.is_file():
        return gz
    if plain.is_file():
        return plain
    return gz


def _open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = locate_jsonl(path)
    rows: list[dict[str, Any]] = []
    with _open_jsonl(path) as fh:
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


def sample_qids_by_dataset(results: Path, datasets: Sequence[str]) -> dict[str, set[str]]:
    """Question ids in ``results/samples/<dataset>.jsonl`` for each dataset."""
    out: dict[str, set[str]] = {}
    for name in datasets:
        path = results / "samples" / f"{name}.jsonl"
        rows = read_jsonl(path)
        qids: list[Any] = []
        for i, row in enumerate(rows, 1):
            qid = row.get("qid")
            if qid is None or qid == "":
                raise ProtocolError(f"{path}:{i}: sample row has no qid")
            qids.append(qid)
        if len(qids) != len(set(qids)):
            raise ProtocolError(f"{path}: duplicate question ids in the sample")
        out[name] = set(qids)
    return out


def in_sample(row: dict[str, Any], qids_by_dataset: dict[str, set[str]]) -> bool:
    """True when ``row`` is a question id written under ``results/samples/``."""
    dataset = row.get("dataset")
    return dataset in qids_by_dataset and row.get("qid") in qids_by_dataset[dataset]


def validate_rows(rows: Sequence[dict[str, Any]], required: Sequence[str], path: Path) -> None:
    for i, row in enumerate(rows, 1):
        missing = [k for k in required if k not in row]
        if missing:
            raise ProtocolError(f"{path}:{i}: missing keys {missing}")


def done_keys(path: Path, fields: Sequence[str]) -> set[tuple]:
    try:
        actual = locate_jsonl(path)
    except ProtocolError:
        return set()
    keys = set()
    for row in read_jsonl(actual):
        validate_rows([row], fields, actual)
        keys.add(tuple(row[f] for f in fields))
    return keys


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    if str(path).endswith(".gz"):
        with gzip.open(path, "at", encoding="utf-8") as fh:
            fh.write(line)
    else:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


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
