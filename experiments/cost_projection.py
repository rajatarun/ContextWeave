"""Projected Bedrock spend from the config, before any model call.

Call counts are exact for a sample of ``n`` questions in each dataset. The
subset total in ``v2.subset_questions`` is stratified across those datasets.

When sample and retrieval files are on disk, input tokens are
``ceil(utf-8 bytes / 4)`` summed over the prompts those stages send: the
generator system prompt plus the user message built from the question and the
retrieved passages, and the judge prompt with those passage texts. The
projection does not call a model. When those files are absent, input tokens
fall back to the prompt template and the artifact says so.

Output tokens are the mean ``output_tokens`` of a prior ledger for that
``model_id`` when the ledger is on disk and has at least one numeric count.
Otherwise they are ``expected_output_tokens`` for that model, and the artifact
says which. The request maxTokens cap is recorded and is not priced. Printed
USD is the projected cost and is not multiplied by ``spend_safety_factor``.

Adjudication calls are ``take_count(judged rows, band fraction)``. The default
fraction is ``v2.adjudication_band_fraction`` (0.15 of judged rows). That
count is an assumption. It is not a count of answers inside the token-F1 band.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterator

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))

import verified_reward as V  # noqa: E402

from experiments.adjudicate_stage import ADJUDICATOR_SYSTEM, user_prompt
from experiments.common import ARMS, DATASETS, ProtocolError, cost_usd, estimate_tokens, locate_jsonl
from experiments.ledger import expected_output_tokens
from experiments.data import seeded_order
from experiments.generate_stage import build_user_message, prompt_token_estimate, system_prompt
from experiments.signal_score import seeded_ids, stream_seed, take_count
from experiments.signals_stage import build_judge_prompt
from experiments.subset_stage import NOVA_MODEL_ID, choose_subset, stratum_quotas

TEMPLATE_SCOPE = (
    "Input tokens are the prompt template. The generator template is the system "
    "prompt plus a user message whose question is '?' and whose evidence is "
    "'(no passages)'. The judge template is the judge prompt with the same "
    "placeholders. Passage text and the real question strings are excluded. "
    "This is the fallback used when sample and retrieval files are not both present."
)

MEASURED_SCOPE = (
    "Input tokens are ceil(utf-8 bytes / 4) summed over the prompts the stages "
    "send. A generator prompt is the system prompt plus the user message built "
    "from the question and the retrieved passages, including passage text and "
    "titles. A judge prompt is the judge user message with the question, the "
    "passage texts (cut at 12000 characters, the same cap the judge stage uses), "
    "and an answer stand-in. An adjudicator prompt is the system prompt plus the "
    "user message with the question, the gold answers, and the same stand-in. "
    "No model was called."
)

STAND_IN_GENERATION = "generation_answer"
STAND_IN_GOLD = "first_gold_answer"
STAND_IN_ABSTAIN = "insufficient_evidence"
ABSTAIN_TEXT = "insufficient evidence"


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} is {value!r}. Refusing to guess a fraction.")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} is {value!r}. Refusing to guess a fraction.") from exc
    if rate < 0.0 or rate > 1.0:
        raise ProtocolError(f"{name} must be in [0, 1], got {rate}")
    return rate


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    located = locate_jsonl(path)
    fh = gzip.open(located, "rt", encoding="utf-8") if str(located).endswith(".gz") else located.open(encoding="utf-8")
    with fh:
        for index, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{located}:{index}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ProtocolError(f"{located}:{index}: expected a JSON object")
            yield row


def _file_present(path: Path) -> bool:
    try:
        locate_jsonl(path)
    except ProtocolError:
        return False
    return True


def _dataset_pair(results: Path, name: str) -> tuple[bool, bool]:
    sample = _file_present(results / "samples" / f"{name}.jsonl")
    retrieval = _file_present(results / "retrieval" / f"{name}.jsonl")
    return sample, retrieval


def ledger_output_means(path: Path | None) -> dict[str, Any]:
    """Mean numeric ``output_tokens`` by ``model_id``.

    A missing path, or a path that is not on disk, leaves the means empty and
    records that output tokens stay at ``expected_output_tokens``. A bool, a
    non-numeric ``output_tokens``, or ``usage_observed`` false is skipped and
    counted. The mean is not guessed.
    """
    if path is None:
        return {
            "path": None,
            "present": False,
            "means": {},
            "n_rows": 0,
            "skipped_rows": 0,
            "label": "No ledger was provided. Output tokens are expected_output_tokens from the config.",
        }
    if not _file_present(path):
        return {
            "path": str(path),
            "present": False,
            "means": {},
            "n_rows": 0,
            "skipped_rows": 0,
            "label": (
                f"Ledger {path} is not on disk. Output tokens are expected_output_tokens from the config."
            ),
        }
    buckets: dict[str, list[float]] = {}
    skipped = 0
    n_rows = 0
    for row in _iter_jsonl(path):
        n_rows += 1
        model = row.get("model_id")
        value = row.get("output_tokens")
        if row.get("usage_observed") is False:
            skipped += 1
            continue
        if not isinstance(model, str) or not model:
            skipped += 1
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            skipped += 1
            continue
        buckets.setdefault(model, []).append(float(value))
    means = {
        model: {"mean": sum(values) / len(values), "n": len(values)}
        for model, values in buckets.items()
    }
    return {
        "path": str(path),
        "present": True,
        "means": means,
        "n_rows": n_rows,
        "skipped_rows": skipped,
        "label": (
            f"Empirical mean output tokens by model_id from {path} "
            f"({n_rows} rows, {skipped} skipped for a missing or non-numeric output_tokens)."
        ),
    }


def _output_plan(
    ledger: dict[str, Any], model_id: str, cap: int, calls: int, expected: float,
) -> dict[str, Any]:
    stats = (ledger.get("means") or {}).get(model_id)
    if isinstance(stats, dict) and int(stats.get("n") or 0) >= 1:
        mean = float(stats["mean"])
        return {
            "output_token_source": "ledger_mean",
            "output_tokens_per_call": mean,
            "output_tokens_total": mean * calls,
            "output_tokens_n": int(stats["n"]),
            "output_tokens_per_call_cap": cap,
            "output_token_label": (
                f"Output tokens are the ledger mean {mean:.4f} over {int(stats['n'])} "
                f"rows for {model_id} in {ledger['path']}."
            ),
        }
    known = ", ".join(sorted((ledger.get("means") or {}))) or "(none)"
    if ledger.get("present"):
        detail = (
            f"The ledger has no numeric output_tokens for {model_id}. "
            f"Models with a mean: {known}."
        )
    else:
        detail = str(ledger.get("label") or "")
    return {
        "output_token_source": "expected_output_tokens",
        "output_tokens_per_call": float(expected),
        "output_tokens_total": float(expected) * calls,
        "output_tokens_n": None,
        "output_tokens_per_call_cap": cap,
        "output_token_label": (
            f"Output tokens are expected_output_tokens {expected:g} per call for {model_id}. {detail}"
        ),
    }


def _templates() -> dict[str, int]:
    system = system_prompt()
    judge = V.JUDGE_PROMPT.format(question="?", evidence="(no passages)", answer="?")
    adj_user = user_prompt("?", "?", ["?"])
    return {
        "generate": prompt_token_estimate(system, build_user_message("?", [])),
        "judge": estimate_tokens(judge),
        "adjudicate": estimate_tokens(ADJUDICATOR_SYSTEM) + estimate_tokens(adj_user),
    }


def _stand_in(sample: dict[str, Any], generated: str | None) -> tuple[str, str]:
    if isinstance(generated, str) and generated.strip():
        return generated, STAND_IN_GENERATION
    gold = sample.get("gold_answers") or []
    if sample.get("unanswerable") or not gold:
        return ABSTAIN_TEXT, STAND_IN_ABSTAIN
    first = gold[0]
    if not isinstance(first, str) or not first.strip():
        return ABSTAIN_TEXT, STAND_IN_ABSTAIN
    return first, STAND_IN_GOLD


def _load_samples(path: Path, dataset: str) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for raw in _iter_jsonl(path):
        qid = raw.get("qid")
        if not isinstance(qid, str) or not qid:
            raise ProtocolError(f"{path}: a sample row has no qid")
        if qid in seen:
            raise ProtocolError(f"{dataset}: duplicate question id {qid}")
        seen.add(qid)
        if "question" not in raw or "gold_answers" not in raw or "unanswerable" not in raw:
            raise ProtocolError(
                f"{dataset} {qid}: sample row needs question, gold_answers, and unanswerable"
            )
        gold = raw["gold_answers"]
        if not isinstance(gold, list):
            raise ProtocolError(f"{dataset} {qid}: gold_answers is not a list")
        rows.append({
            "qid": qid,
            "dataset": dataset,
            "question": str(raw["question"]),
            "gold_answers": gold,
            "unanswerable": bool(raw["unanswerable"]),
        })
    if not rows:
        raise ProtocolError(f"{path}: sample file is empty")
    return rows


def _load_retrieval(path: Path, wanted: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _iter_jsonl(path):
        qid = raw.get("qid")
        arm = raw.get("arm")
        if not isinstance(qid, str) or qid not in wanted:
            continue
        if arm not in ARMS:
            raise ProtocolError(f"{path}: arm {arm!r} is not one of {list(ARMS)}")
        key = (qid, arm)
        if key in found:
            raise ProtocolError(f"{path}: duplicate retrieval row {key}")
        passages = raw.get("passages")
        if not isinstance(passages, list) or not passages:
            raise ProtocolError(f"{path}: {key} has no passages")
        cleaned = []
        for passage in passages:
            if not isinstance(passage, dict) or not isinstance(passage.get("text"), str):
                raise ProtocolError(f"{path}: {key} has a passage without text")
            if not passage["text"].strip():
                raise ProtocolError(f"{path}: {key} has an empty passage")
            title = passage.get("title") or ""
            if not isinstance(title, str):
                title = str(title)
            cleaned.append({"title": title, "text": passage["text"]})
        question = raw.get("question")
        if not isinstance(question, str) or not question:
            raise ProtocolError(f"{path}: {key} has no question")
        found[key] = {"qid": qid, "arm": arm, "question": question, "passages": cleaned}
    return found


def _load_generation(results: Path, dataset: str, wanted: set[str]) -> dict[tuple[str, str], str] | None:
    path = results / "generation" / f"{dataset}.jsonl"
    if not _file_present(path):
        return None
    answers: dict[tuple[str, str], str] = {}
    for raw in _iter_jsonl(path):
        qid = raw.get("qid")
        arm = raw.get("arm")
        if not isinstance(qid, str) or qid not in wanted or arm not in ARMS:
            continue
        answer = raw.get("answer")
        if isinstance(answer, str) and answer.strip():
            answers[(qid, arm)] = answer
    return answers


def _select_sample(
    rows: list[dict[str, Any]], n: int, seed: int, dataset: str,
) -> tuple[list[dict[str, Any]], str]:
    if len(rows) < n:
        raise ProtocolError(
            f"{dataset} sample has {len(rows)} questions and the projection asks for {n}. "
            "Refusing to pad."
        )
    if len(rows) == n:
        return list(rows), (
            f"{dataset}: the on-disk sample has {n} questions, so the projection uses every "
            "question in that file."
        )
    ordered = seeded_order(rows, seed, dataset)
    return list(ordered[:n]), (
        f"{dataset}: the on-disk sample has {len(rows)} questions and does not store the "
        f"shuffle order (the file is written sorted by qid). The projection takes the first "
        f"{n} of seeded_order with seed {seed}: sort by qid, then random.Random(seed).shuffle. "
        f"That draw is a seeded subset of the file. It is not the protocol prefix of a "
        f"full-dev shuffle, because that order is not in the file."
    )


def load_prompt_rows(
    results: Path, n: int, seed: int, datasets: tuple[str, ...] = DATASETS,
) -> dict[str, Any]:
    """Questions, arms, and passages for the projection. No model call."""
    present = {name: _dataset_pair(results, name) for name in datasets}
    missing = [
        name for name, (sample, retrieval) in present.items()
        if not sample or not retrieval
    ]
    found = [name for name, (sample, retrieval) in present.items() if sample or retrieval]
    if missing and not found:
        return {"loaded": False, "missing": missing}
    if missing:
        detail = ", ".join(
            f"{name} sample={present[name][0]} retrieval={present[name][1]}" for name in missing
        )
        raise ProtocolError(
            f"sample and retrieval are incomplete under {results}: {detail}. "
            "Refusing to price a mix of real prompts and the template."
        )
    selected: dict[str, list[dict[str, Any]]] = {}
    notes = []
    retrieval: dict[tuple[str, str], dict[str, Any]] = {}
    generation: dict[tuple[str, str], str] = {}
    generation_missing = []
    for name in datasets:
        print(f"projection: reading {name} sample", file=sys.stderr)
        sample_rows = _load_samples(results / "samples" / f"{name}.jsonl", name)
        chosen, note = _select_sample(sample_rows, n, seed, name)
        notes.append(note)
        selected[name] = chosen
        wanted = {row["qid"] for row in chosen}
        print(f"projection: reading {name} retrieval ({len(wanted)} questions)", file=sys.stderr)
        arm_rows = _load_retrieval(results / "retrieval" / f"{name}.jsonl", wanted)
        for qid in wanted:
            for arm in ARMS:
                key = (qid, arm)
                if key not in arm_rows:
                    raise ProtocolError(
                        f"retrieval for {name} arm {arm} has no row for {qid}. "
                        "Refusing to invent passages."
                    )
        retrieval.update(arm_rows)
        answers = _load_generation(results, name, wanted)
        if answers is None:
            generation_missing.append(name)
        else:
            generation.update(answers)
    qids = [row["qid"] for rows in selected.values() for row in rows]
    if len(qids) != len(set(qids)):
        raise ProtocolError("question ids collide across datasets. Refusing to build one judge sample.")
    by_qid = {row["qid"]: row for rows in selected.values() for row in rows}
    return {
        "loaded": True,
        "results": str(results),
        "selected": selected,
        "by_qid": by_qid,
        "retrieval": retrieval,
        "generation": generation,
        "generation_missing": generation_missing,
        "qids": qids,
        "selection_notes": notes,
    }


def _generator_tokens(system_tokens: int, question: str, passages: list[dict[str, Any]]) -> int:
    return system_tokens + estimate_tokens(build_user_message(question, passages))


def _judge_tokens(question: str, answer: str, passages: list[dict[str, Any]]) -> int:
    built = build_judge_prompt(question, answer, [passage["text"] for passage in passages])
    prompt = built.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ProtocolError(
            f"judge prompt for {question!r} has no evidence. Refusing to price an empty judge prompt."
        )
    return estimate_tokens(prompt)


def _adjudicator_tokens(question: str, answer: str, gold: list[Any]) -> int:
    gold_text = [str(item) for item in gold]
    return estimate_tokens(ADJUDICATOR_SYSTEM) + estimate_tokens(user_prompt(question, answer, gold_text))


def _measure(corpus: dict[str, Any], judge_qids: set[str], subset_qids: set[str]) -> dict[str, Any]:
    system_tokens = estimate_tokens(system_prompt())
    generate_input = 0
    subset_input = 0
    judge_input = 0
    adjudicator_input = 0
    n_judge = 0
    stand_counts = {STAND_IN_GENERATION: 0, STAND_IN_GOLD: 0, STAND_IN_ABSTAIN: 0}
    for qid, sample in corpus["by_qid"].items():
        for arm in ARMS:
            row = corpus["retrieval"][(qid, arm)]
            generate_input += _generator_tokens(system_tokens, row["question"], row["passages"])
            if qid in subset_qids:
                subset_input += _generator_tokens(system_tokens, row["question"], row["passages"])
            if qid not in judge_qids:
                continue
            answer, source = _stand_in(sample, corpus["generation"].get((qid, arm)))
            stand_counts[source] += 1
            n_judge += 1
            judge_input += _judge_tokens(sample["question"], answer, row["passages"])
            adjudicator_input += _adjudicator_tokens(sample["question"], answer, sample["gold_answers"])
    if corpus["generation_missing"]:
        stand_label = (
            "Generation files are absent for "
            + ", ".join(corpus["generation_missing"])
            + ". Judge and adjudicator prompts use an answer stand-in: the first gold "
            "answer, or 'insufficient evidence' when the question is unanswerable or the "
            "gold list is empty. The stand-in is not a generated answer."
        )
    else:
        stand_label = (
            "Judge and adjudicator prompts use the generated answer when that file has one, "
            "otherwise the first gold answer, otherwise 'insufficient evidence'."
        )
    return {
        "generate_input": generate_input,
        "subset_input": subset_input,
        "judge_input": judge_input,
        "adjudicator_input_sum": adjudicator_input,
        "n_judge_rows": n_judge,
        "stand_in_counts": stand_counts,
        "stand_in_label": stand_label,
    }


def _stage(
    name: str,
    model_id: str | None,
    calls: int | None,
    *,
    on_demand: float | None,
    batch: float | None,
    note: str,
    calls_reason: str | None = None,
    usd_reason: str | None = None,
    input_token_source: str | None = None,
    input_tokens_total: int | None = None,
    input_token_label: str | None = None,
    output_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = output_plan or {}
    return {
        "stage": name,
        "model_id": model_id,
        "calls": calls,
        "on_demand_usd": on_demand,
        "batch_usd": batch,
        "note": note,
        "calls_reason": calls_reason,
        "usd_reason": usd_reason,
        "input_token_source": input_token_source,
        "input_tokens_total": input_tokens_total,
        "input_token_label": input_token_label,
        "output_token_source": plan.get("output_token_source"),
        "output_tokens_per_call": plan.get("output_tokens_per_call"),
        "output_tokens_total": plan.get("output_tokens_total"),
        "output_tokens_n": plan.get("output_tokens_n"),
        "output_tokens_per_call_cap": plan.get("output_tokens_per_call_cap"),
        "output_token_label": plan.get("output_token_label"),
    }


def _local(name: str, note: str) -> dict[str, Any]:
    return _stage(name, None, 0, on_demand=0.0, batch=0.0, note=note)


def _price(cfg: dict[str, Any], model_id: str, input_tokens: float, output_tokens: float) -> tuple[float, float]:
    return (
        cost_usd(cfg, model_id, input_tokens, output_tokens, "on_demand"),
        cost_usd(cfg, model_id, input_tokens, output_tokens, "batch"),
    )


def project(
    cfg: dict[str, Any],
    n: int,
    judge_rate: float,
    *,
    results: Path | None = None,
    ledger_path: Path | None = None,
    band_fraction: float | None = None,
    seed: int = 0,
    require_results: bool = False,
) -> dict[str, Any]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ProtocolError(f"n must be a positive int, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProtocolError(f"seed must be an int, got {seed!r}")
    rate = _fraction(judge_rate, "judge rate")
    if band_fraction is None:
        fraction = _fraction(cfg["v2"]["adjudication_band_fraction"], "v2.adjudication_band_fraction")
        fraction_source = "v2.adjudication_band_fraction"
    else:
        fraction = _fraction(band_fraction, "band fraction")
        fraction_source = "--band-fraction"
    n_datasets = len(DATASETS)
    n_arms = len(ARMS)
    subset_n = int(cfg["v2"]["subset_questions"])
    quotas = stratum_quotas(DATASETS, subset_n)
    haiku_samples = cfg["v2"]["subset_samples"]
    if isinstance(haiku_samples, bool) or not isinstance(haiku_samples, int) or haiku_samples < 2:
        raise ProtocolError(
            f"v2.subset_samples is {haiku_samples!r}. Agreement needs at least two samples."
        )
    if cfg["subset_generator_model_id"] != NOVA_MODEL_ID:
        raise ProtocolError(
            f"subset_generator_model_id is {cfg['subset_generator_model_id']!r}. "
            f"The projection prices {NOVA_MODEL_ID}."
        )

    generator = cfg["generator_model_id"]
    judge = cfg["judge_model_id"]
    nova = cfg["subset_generator_model_id"]
    adjudicator = cfg["adjudicator_model_id"]
    gen_out = int(cfg["generator_max_output_tokens"])
    judge_out = int(cfg["judge_max_output_tokens"])
    adj_out = int(cfg["adjudicator_max_output_tokens"])
    ledger = ledger_output_means(ledger_path)

    corpus = None
    if results is not None:
        corpus = load_prompt_rows(Path(results), n, seed)
        if not corpus["loaded"]:
            corpus = None
            if require_results:
                raise ProtocolError(
                    f"retrieve stage output is missing under {results}: {', '.join(DATASETS)}. "
                    "Refusing to price the template as if it were retrieved passages."
                )
    measured = corpus is not None
    templates = None if measured else _templates()

    generate_calls = n_datasets * n * n_arms
    if measured:
        judge_qids = seeded_ids(corpus["qids"], rate, stream_seed(seed, "judge"))
        if len(judge_qids) != take_count(len(corpus["qids"]), rate):
            raise ProtocolError("judge sample size does not match take_count")
        judge_calls = len(judge_qids) * n_arms
    else:
        judge_qids = []
        judge_calls = take_count(n_datasets * n, rate) * n_arms

    subset_reason = None
    for name, quota in quotas.items():
        if quota > n:
            subset_reason = (
                f"{name} quota {quota} exceeds the {n} questions per dataset in this projection. "
                "The subset stage refuses to pad, so these calls are not counted."
            )
            break
    subset_manifest = None
    subset_qids: set[str] = set()
    if measured and subset_reason is None:
        pools = {name: [row["qid"] for row in corpus["selected"][name]] for name in DATASETS}
        subset_manifest = choose_subset(pools, subset_n, seed)
        subset_qids = {
            qid
            for block in subset_manifest["datasets"].values()
            for qid in block["qids"]
        }
        if len(subset_qids) != subset_n:
            raise ProtocolError(
                f"stratified subset has {len(subset_qids)} questions and the total is {subset_n}"
            )

    measured_tokens = None
    if measured:
        print("projection: summing prompt tokens", file=sys.stderr)
        measured_tokens = _measure(corpus, set(judge_qids), subset_qids)

    def _input(kind: str, calls: int, measured_total: int | None, repeats: int = 1) -> tuple[str, int, str]:
        if measured_tokens is not None and measured_total is not None:
            total = measured_total * repeats
            return (
                "measured_prompts",
                total,
                (
                    f"Input tokens are the summed prompts ({total} tokens, source measured_prompts)."
                ),
            )
        each = templates[kind]
        total = each * calls
        return (
            "template",
            total,
            (
                f"Input tokens are the template, {each} per call, {total} in total "
                "(source template). Passage text is excluded."
            ),
        )

    quota_text = ", ".join(f"{name} {quota}" for name, quota in quotas.items())
    stages: list[dict[str, Any]] = []

    gen_source, gen_in, gen_label = _input(
        "generate", generate_calls, None if measured_tokens is None else measured_tokens["generate_input"],
    )
    gen_plan = _output_plan(ledger, generator, gen_out, generate_calls, expected_output_tokens(cfg, generator))
    gen_on, gen_batch = _price(cfg, generator, gen_in, gen_plan["output_tokens_total"])
    stages.append(_stage(
        "generate", generator, generate_calls,
        on_demand=gen_on, batch=gen_batch,
        note=(
            "Haiku answers for every question and arm. Temperature is the config temperature. "
            + gen_label + " " + gen_plan["output_token_label"]
        ),
        input_token_source=gen_source,
        input_tokens_total=gen_in,
        input_token_label=gen_label,
        output_plan=gen_plan,
    ))

    judge_source, judge_in, judge_label = _input(
        "judge", judge_calls, None if measured_tokens is None else measured_tokens["judge_input"],
    )
    judge_plan = _output_plan(ledger, judge, judge_out, judge_calls, expected_output_tokens(cfg, judge))
    judge_on, judge_batch = _price(cfg, judge, judge_in, judge_plan["output_tokens_total"])
    stages.append(_stage(
        "judge", judge, judge_calls,
        on_demand=judge_on, batch=judge_batch,
        note=(
            "Llama judge. Question count is one seeded sample named 'judge' over the questions, "
            "take_count with half-up rounding, then multiplied by the arm count. "
            + judge_label + " " + judge_plan["output_token_label"]
        ),
        input_token_source=judge_source,
        input_tokens_total=judge_in,
        input_token_label=judge_label,
        output_plan=judge_plan,
    ))

    if subset_reason:
        for name, model, base in (
            ("subset_haiku", generator, f"{haiku_samples} temperature-1 Haiku samples on the subset."),
            ("subset_nova", nova, "One Nova Pro answer per subset question and arm."),
        ):
            stages.append(_stage(
                name, model, None,
                on_demand=None, batch=None,
                note=base,
                calls_reason=subset_reason,
                usd_reason=subset_reason,
            ))
    else:
        haiku_calls = subset_n * n_arms * haiku_samples
        nova_calls = subset_n * n_arms
        haiku_source, haiku_in, haiku_label = _input(
            "generate", haiku_calls,
            None if measured_tokens is None else measured_tokens["subset_input"],
            haiku_samples,
        )
        haiku_plan = _output_plan(
            ledger, generator, gen_out, haiku_calls, expected_output_tokens(cfg, generator),
        )
        haiku_on, haiku_batch = _price(cfg, generator, haiku_in, haiku_plan["output_tokens_total"])
        stages.append(_stage(
            "subset_haiku", generator, haiku_calls,
            on_demand=haiku_on, batch=haiku_batch,
            note=(
                f"{haiku_samples} Haiku samples at temperature {cfg['v2']['subset_temperature']} "
                f"on {subset_n} questions in total, stratified ({quota_text}). "
                + haiku_label + " " + haiku_plan["output_token_label"]
            ),
            input_token_source=haiku_source,
            input_tokens_total=haiku_in,
            input_token_label=haiku_label,
            output_plan=haiku_plan,
        ))
        nova_source, nova_in, nova_label = _input(
            "generate", nova_calls,
            None if measured_tokens is None else measured_tokens["subset_input"],
        )
        nova_plan = _output_plan(ledger, nova, gen_out, nova_calls, expected_output_tokens(cfg, nova))
        nova_on, nova_batch = _price(cfg, nova, nova_in, nova_plan["output_tokens_total"])
        stages.append(_stage(
            "subset_nova", nova, nova_calls,
            on_demand=nova_on, batch=nova_batch,
            note=(
                f"Nova Pro at temperature {cfg['temperature']} on {subset_n} questions in total, "
                f"stratified ({quota_text}). "
                + nova_label + " " + nova_plan["output_token_label"]
            ),
            input_token_source=nova_source,
            input_tokens_total=nova_in,
            input_token_label=nova_label,
            output_plan=nova_plan,
        ))

    adj_calls = take_count(judge_calls, fraction)
    assumption = (
        f"ASSUMPTION: adjudication calls are {fraction} of judged rows "
        f"({adj_calls} of {judge_calls}), from {fraction_source}. "
        "This is not a count of answers whose token F1 is inside "
        f"[{cfg['v2']['adjudication_f1_low']}, {cfg['v2']['adjudication_f1_high']}]. "
        "That count is known only after generation. The fraction is configurable."
    )
    if measured_tokens is not None and measured_tokens["n_judge_rows"]:
        if measured_tokens["n_judge_rows"] != judge_calls:
            raise ProtocolError(
                f"measured {measured_tokens['n_judge_rows']} judge prompts and the call count is {judge_calls}"
            )
        mean_adj = measured_tokens["adjudicator_input_sum"] / measured_tokens["n_judge_rows"]
        adj_in = mean_adj * adj_calls
        adj_source = "measured_prompts"
        adj_input_label = (
            f"Input tokens are the mean adjudicator prompt over the {measured_tokens['n_judge_rows']} "
            f"judged rows ({mean_adj:.4f} tokens) times the assumed call count. "
            "The call count is an assumption. The prompts are measured."
        )
    else:
        adj_source, adj_in, adj_input_label = _input("adjudicate", adj_calls, None)
        adj_input_label = adj_input_label + " The call count is an assumption."
    adj_plan = _output_plan(
        ledger, adjudicator, adj_out, adj_calls, expected_output_tokens(cfg, adjudicator),
    )
    adj_on, adj_batch = _price(cfg, adjudicator, adj_in, adj_plan["output_tokens_total"])
    stages.append(_stage(
        "adjudicate", adjudicator, adj_calls,
        on_demand=adj_on, batch=adj_batch,
        note=(
            "gpt-oss-120b. The stage calls on demand. The batch column is the batch price of the "
            "same token estimate. " + assumption + " " + adj_input_label + " " + adj_plan["output_token_label"]
        ),
        calls_reason=assumption,
        usd_reason=assumption,
        input_token_source=adj_source,
        input_tokens_total=adj_in,
        input_token_label=adj_input_label,
        output_plan=adj_plan,
    ))

    stages.extend([
        _local("sample", "No Bedrock call."),
        _local("retrieve", "No Bedrock call."),
        _local("lexical", "No Bedrock call."),
        _local("nli", "No Bedrock call. The NLI model is local."),
        _local("self_percentile", "No Bedrock call."),
        _local("logistic", "No Bedrock call."),
        _local("oracle", "No Bedrock call."),
        _local("calibrate", "No Bedrock call."),
        _local("replay", "No Bedrock call."),
        _local("analyses", "No Bedrock call."),
        _local("study", "No Bedrock call."),
        _local("agreement", "No Bedrock call."),
        _local("write_results", "No Bedrock call."),
    ])

    determined = [
        stage for stage in stages
        if stage["calls"] is not None and stage["on_demand_usd"] is not None
    ]
    if measured:
        selection_note = " ".join(corpus["selection_notes"])
        prompt_source = "retrieval"
        input_scope = MEASURED_SCOPE
        stand_in = {
            "counts": measured_tokens["stand_in_counts"],
            "label": measured_tokens["stand_in_label"],
        }
        from_results = corpus["results"]
    else:
        selection_note = (
            "No sample and retrieval pair was loaded. Question and passage text are not in the "
            "input estimate."
        )
        prompt_source = "template"
        input_scope = TEMPLATE_SCOPE
        stand_in = None
        from_results = None

    return {
        "kind": "projection",
        "called_model": False,
        "seed": seed,
        "n_per_dataset": n,
        "judge_sample_rate": rate,
        "n_datasets": n_datasets,
        "n_arms": n_arms,
        "subset_questions": subset_n,
        "subset_scope": "stratified_total",
        "subset_quotas": quotas,
        "datasets": list(DATASETS),
        "prompt_source": prompt_source,
        "from_results": from_results,
        "selection_note": selection_note,
        "input_scope": input_scope,
        "output_scope": ledger["label"],
        "ledger": {
            "path": ledger["path"],
            "present": ledger["present"],
            "n_rows": ledger["n_rows"],
            "skipped_rows": ledger["skipped_rows"],
            "models": {
                model: {"mean": stats["mean"], "n": stats["n"]}
                for model, stats in sorted((ledger.get("means") or {}).items())
            },
        },
        "already_spent_usd": float(cfg["already_spent_usd"]),
        "total_usd_cap": float(cfg["total_usd_cap"]),
        "remaining_headroom_usd": float(cfg["total_usd_cap"]) - float(cfg["already_spent_usd"]),
        "spend_safety_factor": float(cfg["spend_safety_factor"]),
        "cap_rule": (
            "Refuse a job when ledger usd + already_spent_usd + projected job cost "
            "x spend_safety_factor exceeds total_usd_cap. Printed USD is the projected "
            "cost and is not multiplied by the safety factor."
        ),
        "estimator": "ceil(utf-8 bytes / 4)",
        "adjudication_band_fraction": fraction,
        "adjudication_band_fraction_source": fraction_source,
        "adjudication_assumption": assumption,
        "answer_stand_in": stand_in,
        "stages": stages,
        "sum_on_demand_usd": sum(stage["on_demand_usd"] for stage in determined),
        "sum_batch_usd": sum(stage["batch_usd"] for stage in determined),
        "sum_note": (
            "The sums add every stage whose call count is determined, including adjudication. "
            + assumption
            + (" Subset calls are outside the sums." if subset_reason else "")
        ),
    }


def _usd_cell(value: float | None) -> str:
    return "pending" if value is None else f"{value:.6f}"


def format_projection(body: dict[str, Any]) -> str:
    quotas = body.get("subset_quotas") or {}
    quota_text = ", ".join(f"{name}={quota}" for name, quota in quotas.items())
    lines = [
        "cost projection",
        "kind: projection",
        "called_model: false",
        f"seed: {body.get('seed')}",
        f"n_per_dataset: {body['n_per_dataset']}",
        f"judge_sample_rate: {body['judge_sample_rate']}",
        f"datasets: {body['n_datasets']}",
        f"arms: {body['n_arms']}",
        f"subset_questions_total: {body['subset_questions']}",
        f"subset_scope: {body.get('subset_scope')}",
        f"subset_quotas: {quota_text}",
        f"prompt_source: {body.get('prompt_source')}",
        f"from_results: {body.get('from_results')}",
        f"estimator: {body['estimator']}",
        f"output_scope: {body.get('output_scope')}",
        f"adjudication_band_fraction: {body.get('adjudication_band_fraction')}",
        f"adjudication_band_fraction_source: {body.get('adjudication_band_fraction_source')}",
        str(body.get("adjudication_assumption") or ""),
        str(body.get("selection_note") or ""),
        body["input_scope"],
    ]
    stand_in = body.get("answer_stand_in")
    if isinstance(stand_in, dict):
        lines.append(str(stand_in.get("label") or ""))
        lines.append(f"answer_stand_in_counts: {stand_in.get('counts')}")
    lines.append("")
    lines.append("stage\tcalls\ton_demand_usd\tbatch_usd\tnote")
    for stage in body["stages"]:
        calls = "pending" if stage["calls"] is None else str(stage["calls"])
        lines.append(
            f"{stage['stage']}\tcalls={calls}\t"
            f"on_demand_usd={_usd_cell(stage['on_demand_usd'])}\t"
            f"batch_usd={_usd_cell(stage['batch_usd'])}\t{stage['note']}"
        )
    lines.append("")
    lines.append("stage\tinput_token_source\tinput_tokens_total\toutput_token_source\toutput_tokens_per_call")
    for stage in body["stages"]:
        if stage["model_id"] is None:
            continue
        per_call = stage.get("output_tokens_per_call")
        per_text = "pending" if per_call is None else f"{per_call:.4f}"
        lines.append(
            f"{stage['stage']}\t{stage.get('input_token_source')}\t"
            f"{stage.get('input_tokens_total')}\t{stage.get('output_token_source')}\t{per_text}"
        )
    lines.append("")
    lines.append(f"sum_on_demand_usd: {body['sum_on_demand_usd']:.6f}")
    lines.append(f"sum_batch_usd: {body['sum_batch_usd']:.6f}")
    lines.append(body["sum_note"])
    lines.append(f"already_spent_usd: {float(body['already_spent_usd']):.2f}")
    lines.append(f"total_usd_cap: {float(body['total_usd_cap']):.2f}")
    lines.append(f"remaining_before_this_ledger_usd: {float(body['remaining_headroom_usd']):.2f}")
    lines.append(f"spend_safety_factor: {float(body['spend_safety_factor']):.2f}")
    lines.append(str(body.get("cap_rule") or ""))
    lines.append("")
    return "\n".join(lines)
