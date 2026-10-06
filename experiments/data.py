"""Dataset download, candidate pools, and the seeded question sample.

Pools are built from each dataset's own context. This is closed-pool retrieval:
the gold passage is inside the pool. It is not retrieval over Wikipedia at large.

* SQuAD 2.0 validation. The pool is every distinct context paragraph in the
  validation split that shares the question's article title. Paragraphs that
  are never a question's context are absent from the split, so they are absent
  here too.
* HotpotQA distractor validation. The pool is the ten paragraphs shipped with
  the question. ``bridge`` and ``comparison`` are kept as question types.
* Natural Questions, tractable form: the MRQA 2019 in-domain dev file
  ``NaturalQuestionsShort.jsonl.gz`` (gold short answers, Wikipedia context
  truncated to the first 800 tokens, kept when the short answer is inside that
  window). The context is split into overlapping character windows so the pool
  has more than one passage. The full Wikipedia page is not in this file.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.common import DATASETS, ProtocolError, artifact_meta, sha256_file, write_json

NQ_URL_DEFAULT = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz"
NQ_MD5_DEFAULT = "c0347eebbca02d10d1b07b9a64efe61d"


def _pid(dataset: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{dataset}:{digest}"


def chunk_windows(text: str, window_chars: int, overlap_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if window_chars <= overlap_chars:
        raise ProtocolError("nq window must be longer than the overlap")
    if len(text) <= window_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + window_chars)
        if end < len(text):
            split = text.rfind(" ", start + window_chars // 2, end)
            if split != -1:
                end = split
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        nxt = end - overlap_chars
        if nxt <= start:
            nxt = start + 1
        start = nxt
    return chunks


def _record(dataset: str, qid: str, question: str, question_type: str,
            gold: list[str], unanswerable: bool, pool: list[dict[str, str]]) -> dict[str, Any]:
    if not qid:
        raise ProtocolError(f"{dataset}: a question has an empty id")
    if not pool:
        raise ProtocolError(f"{dataset}:{qid}: candidate pool is empty")
    return {
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "question_type": question_type,
        "gold_answers": gold,
        "unanswerable": unanswerable,
        "pool": pool,
    }


def load_squad(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ProtocolError("datasets is not installed; pip install -r experiments/requirements.txt") from exc
    spec = cfg["datasets"]["squad"]
    ds = load_dataset(spec["huggingface"], split=spec["split"])
    by_title: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, set[str]] = defaultdict(set)
    rows = []
    for row in ds:
        title = row["title"]
        context = row["context"]
        if context not in seen[title]:
            seen[title].add(context)
            by_title.setdefault(title, []).append({
                "id": _pid("squad", title + "\n" + context),
                "title": title,
                "text": context,
            })
        answers = list(row["answers"]["text"])
        unanswerable = len(answers) == 0
        rows.append(_record(
            "squad", row["id"], row["question"],
            "unanswerable" if unanswerable else "answerable",
            [] if unanswerable else answers,
            unanswerable, list(by_title[title]),
        ))
    # Pools were snapshotted as the file was scanned, so earlier questions in an
    # article saw a shorter pool. Rebuild each pool from the finished article.
    for rec in rows:
        title = rec["pool"][0]["title"]
        rec["pool"] = list(by_title[title])
    info = {
        "name": spec["huggingface"],
        "split": spec["split"],
        "n_raw": len(rows),
        "fingerprint": getattr(ds, "_fingerprint", None),
    }
    return rows, info


def load_hotpot(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ProtocolError("datasets is not installed; pip install -r experiments/requirements.txt") from exc
    spec = cfg["datasets"]["hotpot"]
    ds = load_dataset(spec["huggingface"], spec["config"], split=spec["split"])
    rows = []
    for row in ds:
        qtype = row["type"]
        if qtype not in ("bridge", "comparison"):
            raise ProtocolError(f"hotpot question {row['id']} has unexpected type {qtype!r}")
        ctx = row["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]
        if len(titles) != len(sentences):
            raise ProtocolError(f"hotpot question {row['id']} has mismatched context columns")
        pool = []
        for title, sents in zip(titles, sentences):
            text = title + "\n" + " ".join(sents)
            pool.append({
                "id": _pid("hotpot", row["id"] + "\n" + title + "\n" + text),
                "title": title,
                "text": text,
            })
        answer = row["answer"]
        gold = [] if answer is None or answer == "" else [answer]
        rows.append(_record("hotpot", row["id"], row["question"], qtype, gold, not gold, pool))
    info = {
        "name": spec["huggingface"],
        "config": spec["config"],
        "split": spec["split"],
        "n_raw": len(rows),
        "fingerprint": getattr(ds, "_fingerprint", None),
    }
    return rows, info


def _download(url: str, dest: Path, md5_hex: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        digest = hashlib.md5(dest.read_bytes()).hexdigest()
        if digest == md5_hex:
            return
        dest.unlink()
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise ProtocolError(f"failed to download {url}: {exc}") from exc
    digest = hashlib.md5(dest.read_bytes()).hexdigest()
    if digest != md5_hex:
        raise ProtocolError(
            f"checksum mismatch for {url}: expected md5 {md5_hex}, got {digest}. "
            "Refusing to parse a file that is not the published MRQA dev set."
        )


def load_nq(cfg: dict[str, Any], cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = cfg["datasets"]["nq"]
    url = spec["url"]
    md5_hex = spec["md5"]
    if not url or not md5_hex:
        raise ProtocolError("nq url and md5 are required in the config")
    dest = cache_dir / "NaturalQuestionsShort.jsonl.gz"
    _download(url, dest, md5_hex)
    window = int(cfg["nq_window_chars"])
    overlap = int(cfg["nq_overlap_chars"])
    rows = []
    n_lines = 0
    with gzip.open(dest, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "qas" not in obj:
                # The MRQA files start with a header object.
                continue
            n_lines += 1
            context = obj.get("context") or ""
            pieces = chunk_windows(context, window, overlap)
            pool = [{
                "id": _pid("nq", context + f"\n{i}"),
                "title": "",
                "text": piece,
            } for i, piece in enumerate(pieces)]
            for qa in obj["qas"]:
                gold = [a for a in qa.get("answers") or [] if isinstance(a, str) and a.strip()]
                rows.append(_record(
                    "nq", str(qa["qid"]), qa["question"], "nq", gold, not gold, pool,
                ))
    info = {
        "name": "mrqa-natural-questions-short-dev",
        "url": url,
        "md5": md5_hex,
        "n_contexts": n_lines,
        "n_raw": len(rows),
        "window_chars": window,
        "overlap_chars": overlap,
        "limitation": (
            "Wikipedia context is the MRQA truncation (first 800 tokens, answer "
            "inside the window), then split into overlapping character windows. "
            "The full Wikipedia page is not in this file."
        ),
    }
    return rows, info


LOADERS = {"squad": load_squad, "hotpot": load_hotpot}


def sample_rows(rows: list[dict[str, Any]], n: int, seed: int, dataset: str) -> list[dict[str, Any]]:
    qids = [r["qid"] for r in rows]
    if len(qids) != len(set(qids)):
        raise ProtocolError(f"{dataset}: duplicate question ids; refusing to sample")
    if len(rows) < n:
        raise ProtocolError(
            f"{dataset} has {len(rows)} questions and --n-per-dataset is {n}. "
            "Refusing to pad the sample."
        )
    ordered = sorted(rows, key=lambda r: r["qid"])
    rng = random.Random(seed)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    picked = [ordered[i] for i in idx[:n]]
    picked.sort(key=lambda r: r["qid"])
    return picked


def write_sample(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sample(cfg: dict[str, Any], results: Path, n: int, seed: int,
                 datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    cache = results / "cache"
    per_info = {}
    counts = {}
    out_dir = results / "samples"
    all_qids = []
    for name in datasets:
        if name not in DATASETS:
            raise ProtocolError(f"unknown dataset {name!r}")
        if name == "nq":
            rows, info = load_nq(cfg, cache)
        else:
            rows, info = LOADERS[name](cfg)
        picked = sample_rows(rows, n, seed, name)
        path = out_dir / f"{name}.jsonl"
        write_sample(picked, path)
        info["sha256"] = sha256_file(path)
        info["n_sampled"] = len(picked)
        info["path"] = str(path.relative_to(results.parent)) if path.is_absolute() else str(path)
        per_info[name] = info
        counts[name] = len(picked)
        all_qids.append({"dataset": name, "qids": [r["qid"] for r in picked]})
    manifest = artifact_meta(
        cfg, seed,
        n_per_dataset=n,
        datasets=per_info,
        counts=counts,
        stage="sample",
    )
    # qids are part of the manifest so the sample can be checked without
    # re-reading every pool.
    manifest["qids"] = all_qids
    write_json(out_dir / "sample_manifest.json", manifest)
    return manifest
