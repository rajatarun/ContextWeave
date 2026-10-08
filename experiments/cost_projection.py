"""Projected Bedrock spend from the config, before any model call.

Call counts are exact for a sample of ``n`` questions in each dataset. Input
tokens are the real prompt templates with a placeholder question and no
passage text. Output tokens are the configured maximum on every call. Both
are priced from the on-demand table and the batch table. Adjudication calls
depend on how many answers land in the token-F1 band, so that stage stays
pending until those answers exist.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))

import verified_reward as V  # noqa: E402

from experiments.adjudicate_stage import ADJUDICATOR_SYSTEM, user_prompt
from experiments.common import ARMS, DATASETS, ProtocolError, cost_usd, estimate_tokens
from experiments.generate_stage import build_user_message, prompt_token_estimate, system_prompt
from experiments.signal_score import take_count
from experiments.subset_stage import HAIKU_SAMPLES, NOVA_MODEL_ID

INPUT_SCOPE = (
    "Input tokens are the prompt template only. The generator template is the "
    "system prompt plus a user message whose question is '?' and whose evidence "
    "is '(no passages)'. The judge template is the judge prompt with the same "
    "placeholders. Passage text and the real question strings are excluded. "
    "Output tokens are the configured maximum on every call. "
    "The USD figure is that combination, priced from the config. It is a projection."
)


def _usd(cfg: dict[str, Any], model_id: str, calls: int, input_each: int, output_each: int, pricing: str) -> float:
    return cost_usd(cfg, model_id, calls * input_each, calls * output_each, pricing)


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
    template_input_tokens_per_call: int | None = None,
    output_tokens_per_call_cap: int | None = None,
) -> dict[str, Any]:
    return {
        "stage": name,
        "model_id": model_id,
        "calls": calls,
        "on_demand_usd": on_demand,
        "batch_usd": batch,
        "note": note,
        "calls_reason": calls_reason,
        "usd_reason": usd_reason,
        "template_input_tokens_per_call": template_input_tokens_per_call,
        "output_tokens_per_call_cap": output_tokens_per_call_cap,
    }


def _local(name: str, note: str) -> dict[str, Any]:
    return _stage(name, None, 0, on_demand=0.0, batch=0.0, note=note)


def project(cfg: dict[str, Any], n: int, judge_rate: float) -> dict[str, Any]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ProtocolError(f"n must be a positive int, got {n!r}")
    try:
        rate = float(judge_rate)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"judge rate {judge_rate!r} is not a number") from exc
    if rate < 0.0 or rate > 1.0:
        raise ProtocolError(f"judge rate must be in [0, 1], got {rate}")
    n_datasets = len(DATASETS)
    n_arms = len(ARMS)
    subset_n = int(cfg["v2"]["subset_questions"])
    haiku_samples = int(cfg["v2"]["subset_haiku_samples"])
    if haiku_samples != HAIKU_SAMPLES:
        raise ProtocolError(
            f"v2.subset_haiku_samples is {haiku_samples}. The projection uses {HAIKU_SAMPLES}."
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

    gen_in = prompt_token_estimate(system_prompt(), build_user_message("?", []))
    judge_in = estimate_tokens(
        V.JUDGE_PROMPT.format(question="?", evidence="(no passages)", answer="?")
    )
    adj_user = user_prompt("?", "?", ["?"])
    adj_in = estimate_tokens(ADJUDICATOR_SYSTEM) + estimate_tokens(adj_user)

    generate_calls = n_datasets * n * n_arms
    judge_questions = take_count(n_datasets * n, rate)
    judge_calls = judge_questions * n_arms

    stages = [
        _stage(
            "generate", generator, generate_calls,
            on_demand=_usd(cfg, generator, generate_calls, gen_in, gen_out, "on_demand"),
            batch=_usd(cfg, generator, generate_calls, gen_in, gen_out, "batch"),
            note="Haiku answers for every question and arm. Temperature is the config temperature.",
            template_input_tokens_per_call=gen_in,
            output_tokens_per_call_cap=gen_out,
        ),
        _stage(
            "judge", judge, judge_calls,
            on_demand=_usd(cfg, judge, judge_calls, judge_in, judge_out, "on_demand"),
            batch=_usd(cfg, judge, judge_calls, judge_in, judge_out, "batch"),
            note=(
                "Llama judge. Question count is take_count(datasets * n, rate) over one "
                "sample, then multiplied by the arm count. That matches the judge script "
                "when every dataset has n questions."
            ),
            template_input_tokens_per_call=judge_in,
            output_tokens_per_call_cap=judge_out,
        ),
    ]

    subset_reason = None
    if n < subset_n:
        subset_reason = (
            f"n={n} is below subset_questions={subset_n}. "
            "The subset stage refuses to pad, so these calls are not counted."
        )
    if subset_reason:
        for name, model, note in (
            ("subset_haiku", generator, "Five temperature-1 Haiku samples on the subset."),
            ("subset_nova", nova, "One Nova Pro answer per subset question and arm."),
        ):
            stages.append(_stage(
                name, model, None,
                on_demand=None, batch=None, note=note,
                calls_reason=subset_reason, usd_reason=subset_reason,
                template_input_tokens_per_call=gen_in,
                output_tokens_per_call_cap=gen_out,
            ))
    else:
        haiku_calls = n_datasets * subset_n * n_arms * haiku_samples
        nova_calls = n_datasets * subset_n * n_arms
        stages.append(_stage(
            "subset_haiku", generator, haiku_calls,
            on_demand=_usd(cfg, generator, haiku_calls, gen_in, gen_out, "on_demand"),
            batch=_usd(cfg, generator, haiku_calls, gen_in, gen_out, "batch"),
            note=(
                f"{haiku_samples} Haiku samples at temperature "
                f"{cfg['v2']['subset_temperature']} on {subset_n} questions per dataset."
            ),
            template_input_tokens_per_call=gen_in,
            output_tokens_per_call_cap=gen_out,
        ))
        stages.append(_stage(
            "subset_nova", nova, nova_calls,
            on_demand=_usd(cfg, nova, nova_calls, gen_in, gen_out, "on_demand"),
            batch=_usd(cfg, nova, nova_calls, gen_in, gen_out, "batch"),
            note=f"Nova Pro at temperature {cfg['temperature']} on {subset_n} questions per dataset.",
            template_input_tokens_per_call=gen_in,
            output_tokens_per_call_cap=gen_out,
        ))

    adj_reason = (
        "Call count is the number of answers whose token F1 lands in "
        f"[{cfg['v2']['adjudication_f1_low']}, {cfg['v2']['adjudication_f1_high']}]. "
        "That count is known only after generation is on disk."
    )
    stages.append(_stage(
        "adjudicate", adjudicator, None,
        on_demand=None, batch=None,
        note="gpt-oss-120b on demand. Batch is priced in the table and this stage does not submit a batch job.",
        calls_reason=adj_reason, usd_reason=adj_reason,
        template_input_tokens_per_call=adj_in,
        output_tokens_per_call_cap=adj_out,
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

    determined = [stage for stage in stages if stage["calls"] is not None and stage["on_demand_usd"] is not None]
    return {
        "kind": "projection",
        "called_model": False,
        "n_per_dataset": n,
        "judge_sample_rate": rate,
        "n_datasets": n_datasets,
        "n_arms": n_arms,
        "subset_questions": subset_n,
        "datasets": list(DATASETS),
        "input_scope": INPUT_SCOPE,
        "estimator": "ceil(utf-8 bytes / 4)",
        "stages": stages,
        "sum_on_demand_usd": sum(stage["on_demand_usd"] for stage in determined),
        "sum_batch_usd": sum(stage["batch_usd"] for stage in determined),
        "sum_note": (
            "The sums add stages whose call counts are determined. "
            "Adjudication is outside the sums."
        ),
    }


def format_projection(body: dict[str, Any]) -> str:
    lines = [
        "cost projection",
        "kind: projection",
        "called_model: false",
        f"n_per_dataset: {body['n_per_dataset']}",
        f"judge_sample_rate: {body['judge_sample_rate']}",
        f"datasets: {body['n_datasets']}",
        f"arms: {body['n_arms']}",
        f"subset_questions: {body['subset_questions']}",
        f"estimator: {body['estimator']}",
        body["input_scope"],
        "",
        "stage\tcalls\ton_demand_usd\tbatch_usd\tnote",
    ]
    for stage in body["stages"]:
        calls = "pending" if stage["calls"] is None else str(stage["calls"])
        on_demand = "pending" if stage["on_demand_usd"] is None else f"{stage['on_demand_usd']:.6f}"
        batch = "pending" if stage["batch_usd"] is None else f"{stage['batch_usd']:.6f}"
        lines.append(
            f"{stage['stage']}\tcalls={calls}\ton_demand_usd={on_demand}\tbatch_usd={batch}\t{stage['note']}"
        )
    lines.append("")
    lines.append(f"sum_on_demand_usd: {body['sum_on_demand_usd']:.6f}")
    lines.append(f"sum_batch_usd: {body['sum_batch_usd']:.6f}")
    lines.append(body["sum_note"])
    lines.append("")
    return "\n".join(lines)
