"""Dataset download, the shared passage collection, and the seeded question sample.

Each dataset has one passage collection: every context paragraph in the dev
split. Every question, and every retrieval arm, ranks that same collection.
The gold passage is inside it because the collection is the split the
question came from. This is not retrieval over Wikipedia at large.

* SQuAD 2.0 validation. One passage per distinct ``(title, context)``. The
  question's source passage is that context paragraph. ``question_type`` is
  ``squad``. An empty gold list is unanswerable.
* HotpotQA distractor validation. One passage per distinct paragraph text.
  The source passages are the supporting-fact titles. ``question_type`` is
  ``hotpot``. ``bridge`` and ``comparison`` stay on ``hotpot_type``. A yes/no
  answer is kept and ``yes_no`` is true.
* Natural Questions, tractable form: the MRQA 2019 in-domain dev file
  ``NaturalQuestionsShort.jsonl.gz``. Each context is split into overlapping
  character windows, and every window in the file is in the collection. The
  source passages are the windows that contain a gold answer string.
  ``question_type`` is ``nq``.

Question type is the dataset name. The sample is a seeded prefix: sort by
qid, ``random.Random(seed)`` shuffles, and ``n`` is the first ``n`` of that
order. A smaller ``n`` with the same seed is that prefix. A manifest that is
not schema version 2 is refused. v1 per-question pools are not reused.

``expand_nq_pools`` remains for the older per-question hard-negative helper.
The v2 sample does not call it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from experiments.common import (
    DATASETS, ProtocolError, artifact_meta, read_json, read_jsonl, sha256_file, write_json,
)
from experiments.retrieve_stage import BM25Index, tokenize

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
    # A context shared by several questions must not share one list: later
    # pool edits (NQ hard negatives) would otherwise land on every question.
    pool = [dict(p) for p in pool]
    return {
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "question_type": question_type,
        "gold_answers": gold,
        "unanswerable": unanswerable,
        "pool": pool,
    }


def _question(
    dataset: str, qid: str, question: str, gold: list[str], unanswerable: bool,
    source_ids: Sequence[str], *, yes_no: bool, hotpot_type: str | None,
) -> dict[str, Any]:
    if not qid:
        raise ProtocolError(f"{dataset}: a question has an empty id")
    if not source_ids:
        raise ProtocolError(f"{dataset}:{qid}: source_passage_ids is empty")
    if unanswerable and gold:
        raise ProtocolError(f"{dataset}:{qid}: unanswerable question carried gold answers")
    return {
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "question_type": dataset,
        "gold_answers": list(gold),
        "unanswerable": bool(unanswerable),
        "source_passage_ids": list(source_ids),
        "yes_no": bool(yes_no),
        "hotpot_type": hotpot_type,
        "schema_version": 2,
    }


def squad_collection(examples: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """One passage per distinct SQuAD ``(title, context)``. Source is that paragraph."""
    passages_by_key: dict[str, dict[str, str]] = {}
    questions = []
    for row in examples:
        title = row["title"]
        context = row["context"]
        key = title + "\n" + context
        passage = passages_by_key.get(key)
        if passage is None:
            passage = {"id": _pid("squad", key), "title": title, "text": context}
            passages_by_key[key] = passage
        answers = list(row["answers"]["text"])
        unanswerable = len(answers) == 0
        questions.append(_question(
            "squad", row["id"], row["question"],
            [] if unanswerable else answers,
            unanswerable, [passage["id"]], yes_no=False, hotpot_type=None,
        ))
    passages = sorted(passages_by_key.values(), key=lambda p: p["id"])
    return questions, passages


def hotpot_collection(examples: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """One passage per distinct Hotpot paragraph. Source ids are the supporting facts.

    Yes/no answers are kept. ``yes_no`` is true when the gold answer is yes or no.
    ``question_type`` is ``hotpot``; ``bridge`` / ``comparison`` stay on ``hotpot_type``.
    """
    passages_by_key: dict[str, dict[str, str]] = {}
    questions = []
    for row in examples:
        qtype = row["type"]
        if qtype not in ("bridge", "comparison"):
            raise ProtocolError(f"hotpot question {row['id']} has unexpected type {qtype!r}")
        ctx = row["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]
        if len(titles) != len(sentences):
            raise ProtocolError(f"hotpot question {row['id']} has mismatched context columns")
        by_title: dict[str, str] = {}
        for title, sents in zip(titles, sentences):
            text = title + "\n" + " ".join(sents)
            key = title + "\n" + text
            passage = passages_by_key.get(key)
            if passage is None:
                passage = {"id": _pid("hotpot", key), "title": title, "text": text}
                passages_by_key[key] = passage
            if title in by_title and by_title[title] != passage["id"]:
                raise ProtocolError(f"hotpot question {row['id']} repeats title {title!r} with two texts")
            by_title[title] = passage["id"]
        support = list((row.get("supporting_facts") or {}).get("title") or [])
        source: list[str] = []
        for title in support:
            if title not in by_title:
                raise ProtocolError(
                    f"hotpot question {row['id']}: supporting title {title!r} is not in the context"
                )
            pid = by_title[title]
            if pid not in source:
                source.append(pid)
        if not source:
            raise ProtocolError(f"hotpot question {row['id']}: no supporting-fact paragraphs")
        answer = row["answer"]
        gold = [] if answer is None or answer == "" else [answer]
        yes_no = isinstance(answer, str) and answer.strip().casefold() in {"yes", "no"}
        questions.append(_question(
            "hotpot", row["id"], row["question"], gold, not gold, source,
            yes_no=yes_no, hotpot_type=qtype,
        ))
    passages = sorted(passages_by_key.values(), key=lambda p: p["id"])
    return questions, passages


def nq_collection(
    objects: Sequence[dict[str, Any]], window: int, overlap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    """Every MRQA window is in the collection. Source windows contain a gold string."""
    passages_by_id: dict[str, dict[str, str]] = {}
    questions = []
    n_contexts = 0
    for obj in objects:
        context = obj.get("context") or ""
        pieces = chunk_windows(context, window, overlap)
        if not pieces:
            raise ProtocolError("nq context produced no windows")
        n_contexts += 1
        window_ids = []
        for i, piece in enumerate(pieces):
            pid = _pid("nq", context + f"\n{i}")
            previous = passages_by_id.get(pid)
            if previous is None:
                passages_by_id[pid] = {"id": pid, "title": "", "text": piece}
            elif previous["text"] != piece:
                raise ProtocolError(f"nq passage {pid} has two different texts")
            window_ids.append(pid)
        for qa in obj["qas"]:
            qid = str(qa["qid"])
            gold = [a for a in qa.get("answers") or [] if isinstance(a, str) and a.strip()]
            if not gold:
                raise ProtocolError(
                    f"nq {qid}: no gold answer. Refusing to guess which window is the source."
                )
            source = [
                pid for pid in window_ids
                if any(answer in passages_by_id[pid]["text"] for answer in gold)
            ]
            if not source:
                raise ProtocolError(
                    f"nq {qid}: no window contains a gold answer. "
                    "Refusing to mark a different window as the source."
                )
            questions.append(_question(
                "nq", qid, qa["question"], gold, False, source,
                yes_no=False, hotpot_type=None,
            ))
    passages = sorted(passages_by_id.values(), key=lambda p: p["id"])
    return questions, passages, n_contexts


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ProtocolError(
            "datasets is not installed; pip install -r experiments/requirements.txt"
        ) from exc
    return load_dataset


def load_squad(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    load_dataset = _require_datasets()
    spec = cfg["datasets"]["squad"]
    ds = load_dataset(spec["huggingface"], split=spec["split"])
    questions, passages = squad_collection(ds)
    info = {
        "name": spec["huggingface"],
        "split": spec["split"],
        "n_raw": len(questions),
        "n_passages": len(passages),
        "collection": "shared_dev_split",
        "fingerprint": getattr(ds, "_fingerprint", None),
    }
    return questions, passages, info


def load_hotpot(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    load_dataset = _require_datasets()
    spec = cfg["datasets"]["hotpot"]
    ds = load_dataset(spec["huggingface"], spec["config"], split=spec["split"])
    questions, passages = hotpot_collection(ds)
    info = {
        "name": spec["huggingface"],
        "config": spec["config"],
        "split": spec["split"],
        "n_raw": len(questions),
        "n_passages": len(passages),
        "collection": "shared_dev_split",
        "fingerprint": getattr(ds, "_fingerprint", None),
        "n_yes_no": sum(1 for q in questions if q["yes_no"]),
    }
    return questions, passages, info


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


def load_nq(
    cfg: dict[str, Any], cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    spec = cfg["datasets"]["nq"]
    url = spec["url"]
    md5_hex = spec["md5"]
    if not url or not md5_hex:
        raise ProtocolError("nq url and md5 are required in the config")
    dest = cache_dir / "NaturalQuestionsShort.jsonl.gz"
    _download(url, dest, md5_hex)
    window = int(cfg["nq_window_chars"])
    overlap = int(cfg["nq_overlap_chars"])
    objects = []
    with gzip.open(dest, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "qas" not in obj:
                # The MRQA files start with a header object.
                continue
            objects.append(obj)
    questions, passages, n_contexts = nq_collection(objects, window, overlap)
    info = {
        "name": "mrqa-natural-questions-short-dev",
        "url": url,
        "md5": md5_hex,
        "n_contexts": n_contexts,
        "n_raw": len(questions),
        "n_passages": len(passages),
        "collection": "shared_dev_split",
        "window_chars": window,
        "overlap_chars": overlap,
        "limitation": (
            "Wikipedia context is the MRQA truncation (first 800 tokens, answer "
            "inside the window), then split into overlapping character windows. "
            "The full Wikipedia page is not in this file. The shared collection "
            "is every window in the dev file. A question's source passages are "
            "the windows that contain a gold answer string."
        ),
    }
    return questions, passages, info


def expand_nq_pools(rows: list[dict[str, Any]], target: int, k1: float, b: float) -> dict[str, Any]:
    """Fill each NQ pool to ``target`` passages. Gold windows are kept.

    Hard negatives are windows from other sampled documents, ranked by BM25
    against the question. The ranking is a pure function of the sampled
    corpus: score descending, passage id ascending. The sample seed is what
    makes the corpus, and therefore the pools, reproducible.
    """
    if target < 2:
        raise ProtocolError("nq_pool_size must be at least 2 so a negative can be added")
    corpus: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("dataset") != "nq":
            raise ProtocolError("expand_nq_pools received a non-NQ row")
        if not row["pool"]:
            raise ProtocolError(f"nq {row['qid']}: gold pool is empty")
        for passage in row["pool"]:
            if passage.get("role") not in (None, "gold"):
                raise ProtocolError(f"nq {row['qid']}: expand expects gold windows only")
            source = passage.get("source")
            if not source:
                raise ProtocolError(f"nq {row['qid']}: gold window has no source id")
            passage["role"] = "gold"
            previous = corpus.get(passage["id"])
            if previous is None:
                corpus[passage["id"]] = passage
            elif previous["text"] != passage["text"] or previous.get("source") != source:
                raise ProtocolError(f"nq passage {passage['id']} has two different texts")
    tokens = {pid: tokenize(p["text"]) for pid, p in corpus.items()}
    # One index per source document. Every question from that document has the
    # same gold windows, so the candidate set (and its document frequencies)
    # does not change from question to question.
    indexes: dict[str, tuple[list[dict[str, str]], BM25Index]] = {}
    added = 0
    sizes = []
    gold_sizes = []
    for row in rows:
        gold = list(row["pool"])
        gold_ids = {p["id"] for p in gold}
        gold_text = {p["text"] for p in gold}
        source = gold[0]["source"]
        if any(p.get("source") != source for p in gold):
            raise ProtocolError(f"nq {row['qid']}: gold windows come from more than one document")
        gold_sizes.append(len(gold))
        need = target - len(gold)
        if need > 0:
            prepared = indexes.get(source)
            if prepared is None:
                candidates = [
                    p for p in corpus.values()
                    if p.get("source") != source and p["id"] not in gold_ids and p["text"] not in gold_text
                ]
                candidates.sort(key=lambda p: p["id"])
                prepared = (candidates, BM25Index([tokens[p["id"]] for p in candidates], k1, b))
                indexes[source] = prepared
            candidates, index = prepared
            scores = index.scores(tokenize(row["question"]))
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]["id"]))
            for i in order[:need]:
                picked = dict(candidates[i])
                picked["role"] = "hard_negative"
                row["pool"].append(picked)
                added += 1
        if len(row["pool"]) < target:
            raise ProtocolError(
                f"nq {row['qid']}: pool has {len(row['pool'])} passages and nq_pool_size is {target}. "
                "The sampled documents did not contain enough hard-negative windows."
            )
        sizes.append(len(row["pool"]))
    return {
        "nq_pool_size": target,
        "negative_selection": (
            "BM25 over windows from other documents in the seeded sample; "
            "ties break by passage id. Every gold window is kept, so a "
            "document that already has more windows than nq_pool_size stays larger."
        ),
        "pool_size_min": min(sizes),
        "pool_size_max": max(sizes),
        "pool_size_mean": sum(sizes) / len(sizes),
        "gold_windows_mean": sum(gold_sizes) / len(gold_sizes),
        "hard_negatives_added": added,
    }


LOADERS = {"squad": load_squad, "hotpot": load_hotpot}

# Logged on the sample manifest as sampling.order. The seed is stored beside it.
SAMPLE_ORDER = (
    "Questions are sorted by qid, then random.Random(seed) shuffles those "
    "positions. The sample is the first n_per_dataset of that order and is "
    "written sorted by qid. A smaller n with the same seed is that prefix. "
    "The passage collection is the full dev split and does not shrink with n. "
    "When a larger schema-version-2 sample with this seed is already on disk, "
    "the prefix keeps the previously written question rows. A manifest that "
    "is not schema version 2 is refused: v1 per-question pools are not reused. "
    "The seed is stored on this manifest."
)


def seeded_order(rows: list[dict[str, Any]], seed: int, dataset: str) -> list[dict[str, Any]]:
    """Shuffle order for ``seed``.

    Rows are sorted by qid first, so the order does not depend on the loader.
    ``random.Random(seed)`` then shuffles those positions. A sample of size n
    is the first n rows of this list.
    """
    qids = [r["qid"] for r in rows]
    if len(qids) != len(set(qids)):
        raise ProtocolError(f"{dataset}: duplicate question ids; refusing to sample")
    ordered = sorted(rows, key=lambda r: r["qid"])
    rng = random.Random(seed)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    return [ordered[i] for i in idx]


def sample_rows(rows: list[dict[str, Any]], n: int, seed: int, dataset: str) -> list[dict[str, Any]]:
    """First ``n`` rows of :func:`seeded_order`, returned sorted by qid.

    The same rows and seed make a smaller n the question ids of that prefix.
    The returned list is sorted by qid, so the written file is not in shuffle order.
    """
    order = seeded_order(rows, seed, dataset)
    if len(order) < n:
        raise ProtocolError(
            f"{dataset} has {len(order)} questions and --n-per-dataset is {n}. "
            "Refusing to pad the sample."
        )
    picked = list(order[:n])
    picked.sort(key=lambda r: r["qid"])
    return picked


def summarize_nq_pools(rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
    """Pool-size stats for rows whose hard negatives were already chosen."""
    if not rows:
        raise ProtocolError("nq sample is empty")
    sizes = []
    gold_sizes = []
    added = 0
    for row in rows:
        pool = row.get("pool") or []
        if not pool:
            raise ProtocolError(f"nq {row.get('qid')}: pool is empty")
        sizes.append(len(pool))
        gold_sizes.append(sum(1 for passage in pool if passage.get("role") == "gold"))
        added += sum(1 for passage in pool if passage.get("role") == "hard_negative")
    return {
        "nq_pool_size": target,
        "negative_selection": (
            "BM25 over windows from other documents in the seeded sample; "
            "ties break by passage id. Every gold window is kept, so a "
            "document that already has more windows than nq_pool_size stays larger."
        ),
        "pool_size_min": min(sizes),
        "pool_size_max": max(sizes),
        "pool_size_mean": sum(sizes) / len(sizes),
        "gold_windows_mean": sum(gold_sizes) / len(gold_sizes),
        "hard_negatives_added": added,
    }


def _rows_for_qids(path: Path, qids: list[str]) -> list[dict[str, Any]] | None:
    """Rows from an existing sample file, in ``qids`` order."""
    try:
        rows = read_jsonl(path)
    except ProtocolError:
        return None
    by_qid: dict[Any, dict[str, Any]] = {}
    for row in rows:
        qid = row.get("qid")
        if qid in by_qid:
            return None
        by_qid[qid] = row
    if any(qid not in by_qid for qid in qids):
        return None
    return [by_qid[qid] for qid in qids]


def write_sample(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def passages_path(samples_dir: Path, dataset: str) -> Path:
    return samples_dir / f"{dataset}.passages.jsonl"


def _passage_rows(passages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(passages, key=lambda p: p["id"])
    seen: set[str] = set()
    for passage in ordered:
        pid = passage.get("id")
        if not pid or pid in seen:
            raise ProtocolError(f"shared collection passage id {pid!r} is missing or duplicated")
        if "text" not in passage:
            raise ProtocolError(f"shared collection passage {pid} has no text")
        seen.add(pid)
    return [{"id": p["id"], "title": p.get("title") or "", "text": p["text"]} for p in ordered]


def _same_passages(left: Sequence[dict[str, str]], right: Sequence[dict[str, str]]) -> bool:
    def key(row: dict[str, str]) -> tuple[str, str, str]:
        return (row["id"], row.get("title") or "", row["text"])
    return sorted(map(key, left)) == sorted(map(key, right))


def _qid_list(manifest: dict[str, Any], dataset: str) -> list[str] | None:
    for block in manifest.get("qids") or []:
        if block.get("dataset") == dataset:
            qids = block.get("qids")
            return list(qids) if isinstance(qids, list) else None
    return None


def _check_question_rows(rows: Sequence[dict[str, Any]], passages: Sequence[dict[str, str]], dataset: str) -> None:
    known = {p["id"] for p in passages}
    for row in rows:
        if row.get("schema_version") != 2:
            raise ProtocolError(
                f"{dataset}:{row.get('qid')}: question row is not schema_version 2. "
                "v1 rows are not reused."
            )
        if row.get("question_type") != dataset:
            raise ProtocolError(
                f"{dataset}:{row.get('qid')}: question_type is {row.get('question_type')!r}. "
                "v2 question_type is the dataset name."
            )
        missing = [pid for pid in row.get("source_passage_ids") or [] if pid not in known]
        if not row.get("source_passage_ids") or missing:
            raise ProtocolError(
                f"{dataset}:{row.get('qid')}: source passage is not in the shared collection"
            )


def build_sample(cfg: dict[str, Any], results: Path, n: int, seed: int,
                 datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    cache = results / "cache"
    out_dir = results / "samples"
    manifest_path = out_dir / "sample_manifest.json"
    previous = read_json(manifest_path) if manifest_path.is_file() else None
    if previous is not None and previous.get("schema_version") != 2:
        raise ProtocolError(
            "results/samples/sample_manifest.json is not schema_version 2. "
            "v1 per-question pools and v1 generation rows are not reused. "
            "Move that samples directory aside and rerun."
        )
    per_info = {}
    counts = {}
    all_qids = []
    requested = []
    for name in datasets:
        if name not in DATASETS:
            raise ProtocolError(f"unknown dataset {name!r}")
        if name in requested:
            raise ProtocolError(f"dataset {name} was requested twice")
        requested.append(name)
    # Resolve every dataset before writing, so a refused id change leaves the
    # committed files untouched.
    prepared = []
    shrunk_from: dict[str, int] = {}
    for name in requested:
        if name == "nq":
            rows, passages, info = load_nq(cfg, cache)
        else:
            rows, passages, info = LOADERS[name](cfg)
        passages = _passage_rows(passages)
        picked = sample_rows(rows, n, seed, name)
        preserved = False
        if previous is not None and previous.get("seed") == seed:
            old_qids = _qid_list(previous, name)
            new_qids = [r["qid"] for r in picked]
            if old_qids is not None and list(old_qids) != new_qids:
                old_n = len(old_qids)
                expected_old = None
                if 0 < n < old_n and len(rows) >= old_n:
                    expected_old = [r["qid"] for r in sample_rows(rows, old_n, seed, name)]
                if expected_old != list(old_qids):
                    raise ProtocolError(
                        f"{name}: question ids changed under seed {seed}. "
                        "Refusing to replace a committed sample."
                    )
                path = out_dir / f"{name}.jsonl"
                kept = _rows_for_qids(path, new_qids)
                if kept is None:
                    raise ProtocolError(
                        f"{name}: the first {n} questions of the seed-{seed} shuffle "
                        f"are not all in {path.name}. Refusing to rebuild a committed sample."
                    )
                on_disk = passages_path(out_dir, name)
                if not on_disk.is_file():
                    raise ProtocolError(
                        f"{name}: question rows were kept and {on_disk.name} is missing. "
                        "Refusing to rebuild the shared collection under a committed sample."
                    )
                disk_passages = read_jsonl(on_disk)
                if not _same_passages(disk_passages, passages):
                    raise ProtocolError(
                        f"{name}: the shared passage collection changed under seed {seed}. "
                        "Refusing to replace a committed sample."
                    )
                passages = _passage_rows(disk_passages)
                picked = kept
                preserved = True
                shrunk_from[name] = old_n
        _check_question_rows(picked, passages, name)
        info["n_yes_no"] = sum(1 for row in picked if row.get("yes_no"))
        info["n_passages"] = len(passages)
        prepared.append((name, picked, passages, info, preserved))
    for name, picked, passages, info, preserved in prepared:
        path = out_dir / f"{name}.jsonl"
        write_sample(picked, path)
        ppath = passages_path(out_dir, name)
        write_sample(passages, ppath)
        info["sha256"] = sha256_file(path)
        info["passages_sha256"] = sha256_file(ppath)
        info["n_sampled"] = len(picked)
        info["path"] = str(path.relative_to(results.parent)) if path.is_absolute() else str(path)
        info["passages_path"] = str(ppath.relative_to(results.parent)) if ppath.is_absolute() else str(ppath)
        info["collection"] = "shared_dev_split"
        if preserved:
            info["rows_kept_from_previous_sample"] = True
        per_info[name] = info
        counts[name] = len(picked)
        all_qids.append({"dataset": name, "qids": [r["qid"] for r in picked]})
    if previous is not None:
        for name in DATASETS:
            if name in requested:
                continue
            old_info = (previous.get("datasets") or {}).get(name)
            old_count = (previous.get("counts") or {}).get(name)
            old_qids = _qid_list(previous, name)
            if not old_info or old_count is None or old_qids is None:
                raise ProtocolError(
                    f"sample rebuild asked only for {requested}, and the existing "
                    f"manifest has no complete entry for {name}."
                )
            per_info[name] = old_info
            counts[name] = old_count
            all_qids.append({"dataset": name, "qids": old_qids})
    all_qids.sort(key=lambda block: DATASETS.index(block["dataset"]))
    manifest = artifact_meta(
        cfg, seed,
        schema_version=2,
        n_per_dataset=n,
        datasets=per_info,
        counts=counts,
        stage="sample",
        collection="shared_dev_split",
        question_type="dataset",
        sampling={
            "seed": seed,
            "n_per_dataset": n,
            "order": SAMPLE_ORDER,
            "prefix_of_n": shrunk_from or None,
        },
    )
    manifest["qids"] = all_qids
    write_json(out_dir / "sample_manifest.json", manifest)
    return manifest
