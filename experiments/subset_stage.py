"""Seeded subset and agreement across temperature-1 samples.

``subset_questions`` is a total, stratified across the datasets in the order
they are given. Largest remainder: ``base, rem = divmod(total, n_datasets)``,
and the first ``rem`` datasets receive ``base + 1``. Each dataset then sorts
its question ids, shuffles with
``random.Random(stream_seed(seed, "subset:{dataset}"))``, and keeps the first
quota of that order. A dataset with fewer questions than its quota stops the
run. Nothing is padded. A smaller total with the same seed and the same
dataset order is a prefix inside each dataset.

Agreement is the modal fraction of SQuAD-normalised answers across the
configured samples, with pairwise agreement stored beside it. A (question,
arm) that is missing any sample is not scored. Temperature 1 is a model
random step. The run seed and the sample index are logged on the sample
meta. The Converse request has no seed parameter.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import verified_reward_bench as B  # noqa: E402

from experiments.common import ARMS, ProtocolError
from experiments.signal_score import stream_seed

SUBSET_STREAM = "subset"
HAIKU_TEMPERATURE = 1.0
NOVA_MODEL_ID = "us.amazon.nova-pro-v1:0"
NOVA_TEMPERATURE = 0.0


def subset_stream_name(dataset: str) -> str:
    return f"{SUBSET_STREAM}:{dataset}"


def stratum_quotas(datasets: Sequence[str], total: int) -> dict[str, int]:
    """Largest-remainder split of ``total`` across ``datasets``, in that order.

    ``squad, hotpot, nq`` and 250 is 84, 83, 83. A quota of 0 is a dataset the
    total did not reach. The caller does not draw from it.
    """
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ProtocolError(f"subset total must be a positive int, got {total!r}")
    names = [str(name) for name in datasets]
    if not names:
        raise ProtocolError("subset allocation needs at least one dataset")
    if len(names) != len(set(names)):
        raise ProtocolError(f"duplicate dataset in subset allocation: {names}")
    base, rem = divmod(total, len(names))
    return {name: base + (1 if index < rem else 0) for index, name in enumerate(names)}


def choose_ids(qids: Sequence[str], n: int, seed: int, dataset: str) -> dict[str, Any]:
    """First ``n`` of the seeded shuffle. Fewer than ``n`` ids is an error."""
    if n < 1:
        raise ProtocolError(f"subset size must be at least 1, got {n}")
    ordered = sorted(set(qids))
    if len(ordered) != len(list(qids)):
        raise ProtocolError(f"{dataset}: duplicate question ids in the sample")
    if len(ordered) < n:
        raise ProtocolError(
            f"{dataset} has {len(ordered)} questions and the subset asks for {n}. "
            "Refusing to pad."
        )
    name = subset_stream_name(dataset)
    stream_value = stream_seed(seed, name)
    rng = random.Random(stream_value)
    rng.shuffle(ordered)
    chosen = ordered[:n]
    return {
        "dataset": dataset,
        "n": n,
        "n_available": len(ordered),
        "stream": name,
        "stream_seed": stream_value,
        "qids": chosen,
        "rule": (
            "sorted question ids, random.Random(stream_seed).shuffle, "
            "first n. A smaller n with the same seed is a prefix."
        ),
    }


def choose_subset(
    qids_by_dataset: dict[str, Sequence[str]], n: int, seed: int,
) -> dict[str, Any]:
    """Allocate ``n`` questions in total across the datasets, in dict order."""
    names = list(qids_by_dataset)
    quotas = stratum_quotas(names, n)
    datasets = {}
    for name in names:
        quota = quotas[name]
        if quota < 1:
            continue
        datasets[name] = choose_ids(qids_by_dataset[name], quota, seed, name)
    return {
        "subset_scope": "stratified_total",
        "subset_questions": n,
        "quotas": quotas,
        "seed": seed,
        "datasets": datasets,
        "allocation": (
            "Largest remainder over the datasets in the order given. "
            "base, remainder = divmod(total, n_datasets). "
            "The first remainder datasets receive base+1 and the rest receive base. "
            "Each dataset shuffles its own sorted question ids with "
            "random.Random(stream_seed(seed, 'subset:{dataset}')) and keeps the first quota. "
            "A smaller total with the same seed and the same dataset order is a prefix "
            "inside each dataset. A dataset with fewer questions than its quota stops "
            "the run. Nothing is padded."
        ),
    }


def agreement_scores(answers: Sequence[str]) -> dict[str, Any]:
    """Modal fraction and pairwise agreement of SQuAD-normalised answers."""
    if len(answers) < 2:
        raise ProtocolError("agreement needs at least two answers")
    norms = [B.normalize_answer(answer) for answer in answers]
    counts = Counter(norms)
    mode_count = max(counts.values())
    n = len(norms)
    pairs = n * (n - 1) // 2
    agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] == norms[j]:
                agree += 1
    return {
        "n_samples": n,
        "modal_fraction": mode_count / n,
        "pairwise_agreement": agree / pairs,
        "n_distinct_normalised": len(counts),
    }


def require_haiku_settings(cfg: dict[str, Any]) -> tuple[int, float]:
    n = cfg["v2"]["subset_samples"]
    temperature = cfg["v2"]["subset_temperature"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ProtocolError(
            f"v2.subset_samples is {n!r}. Agreement needs at least two temperature-1 samples."
        )
    if temperature != HAIKU_TEMPERATURE:
        raise ProtocolError(
            f"v2.subset_temperature is {temperature!r}. "
            f"The Haiku agreement samples run at temperature {HAIKU_TEMPERATURE}."
        )
    return int(n), float(temperature)


def require_nova_settings(cfg: dict[str, Any]) -> tuple[str, float]:
    model_id = cfg["subset_generator_model_id"]
    temperature = cfg["temperature"]
    if model_id != NOVA_MODEL_ID:
        raise ProtocolError(
            f"subset_generator_model_id is {model_id!r}. This study calls {NOVA_MODEL_ID}."
        )
    if temperature != NOVA_TEMPERATURE:
        raise ProtocolError(
            f"config temperature is {temperature!r}. Nova subset generation runs at temperature 0."
        )
    return model_id, float(temperature)


def expected_keys(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(dataset, qid, arm) in manifest order, then ``ARMS`` order."""
    keys = []
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ProtocolError("subset manifest has no datasets")
    for dataset, block in datasets.items():
        qids = block.get("qids") if isinstance(block, dict) else None
        if not qids:
            raise ProtocolError(f"subset manifest has no qids for {dataset}")
        for qid in qids:
            for arm in ARMS:
                keys.append((str(dataset), str(qid), arm))
    return keys


def score_agreement(
    keys: Sequence[tuple[str, str, str]],
    by_sample: Sequence[dict[tuple[str, str], dict[str, Any]]],
) -> dict[str, Any]:
    """Score every key that has all samples. Any gap leaves the means null.

    A partial set is not averaged. The incomplete keys are listed with a reason.
    """
    n_samples = len(by_sample)
    if n_samples < 2:
        raise ProtocolError("agreement needs the sample files")
    incomplete = []
    scored: list[dict[str, Any]] = []
    for dataset, qid, arm in keys:
        answers = []
        missing = []
        for index, table in enumerate(by_sample):
            row = table.get((qid, arm))
            if row is None or "answer" not in row:
                missing.append(index)
                continue
            answers.append(row["answer"])
        if missing:
            incomplete.append({
                "dataset": dataset,
                "qid": qid,
                "arm": arm,
                "missing_samples": missing,
                "reason": "a sample file has no row for this question and arm",
            })
            continue
        scored.append({
            "dataset": dataset,
            "qid": qid,
            "arm": arm,
            **agreement_scores(answers),
        })
    by_dataset: dict[str, Any] = {}
    datasets = sorted({dataset for dataset, _qid, _arm in keys})
    partial = bool(incomplete)
    for dataset in datasets:
        rows = [row for row in scored if row["dataset"] == dataset]
        if partial or not rows:
            by_dataset[dataset] = {
                "n_scored": 0,
                "mean_modal_fraction": None,
                "mean_pairwise_agreement": None,
                "reason": (
                    "at least one question and arm is missing a sample, so no mean is published"
                    if partial else "no rows"
                ),
            }
            continue
        by_dataset[dataset] = {
            "n_scored": len(rows),
            "mean_modal_fraction": sum(row["modal_fraction"] for row in rows) / len(rows),
            "mean_pairwise_agreement": sum(row["pairwise_agreement"] for row in rows) / len(rows),
            "reason": None,
        }
    return {
        "n_samples": n_samples,
        "partial": partial,
        "n_incomplete": len(incomplete),
        "incomplete": incomplete,
        "rows": [] if partial else scored,
        "datasets": by_dataset,
        "definition": (
            "Modal fraction is the share of SQuAD-normalised answers equal to the most "
            "common normalised answer. Pairwise agreement is the fraction of unordered "
            "pairs whose normalised answers are equal. A missing sample leaves every "
            "mean null."
        ),
    }
