"""Information ratios and Thompson-sampling streams on a logged reward.

``c0`` and ``c1`` are the means of an observed reward ``R`` on joined-correct
``Y`` of 0 and 1. ``s = c1 - c0``. ``m`` is the mean of ``Y`` on the rows where
``R`` was observed. That is a different quantity from the missingness rate
``m_a`` in the analyses artifact.

``Var(R)`` is the unbiased sample variance. ``s^2 / Var(R)`` is null when the
variance is missing or zero. ``s^2 / (m(1-m))`` is null when ``m`` is 0 or 1.
Information per round is ``s^2`` divided by the residual variance
``Var(R) - s^2 m (1-m)``, and it is null when that residual is missing or not
positive.

The long streams draw questions with replacement from the log. Beta Thompson
sampling is the deployed fractional update, prior mean taken from the
``general`` routing prior and scaled by ``ROUTER_PRIOR_STRENGTH``. Gaussian
Thompson sampling uses that same prior mean and variance. Drift multiplies the
observation counts by the discount and leaves the prior in place. The reward
on these streams is joined correct.

Judge coverage keeps stored judge values on a seeded question sample and
recomputes the verified reward. It does not rerun the long streams. When no
judge value is stored, the coverage ratios stay null.
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))
sys.path.insert(0, str(_ROOT / "src" / "shared"))

import rag_router as RAG  # noqa: E402

from experiments.analyses_stage import _arm_conditional, _signal_rows
from experiments.common import ARMS, ProtocolError
from experiments.metrics import score_answer
from experiments.records import combine_reward
from experiments.subset_stage import NOVA_MODEL_ID, NOVA_TEMPERATURE
from experiments.replay_stage import general_priors
from experiments.signal_score import seeded_ids, stream_seed

DEFINITIONS = {
    "c0": "Mean of observed R on rows whose joined correct Y is 0.",
    "c1": "Mean of observed R on rows whose joined correct Y is 1.",
    "s": "c1 - c0.",
    "m": (
        "Mean of joined correct Y among rows where R is observed. "
        "This is not the missingness rate m_a."
    ),
    "var_r": "Unbiased sample variance of observed R, divisor n-1. Null when fewer than two rewards were observed.",
    "s2_over_var_r": "s^2 / Var(R). Null when s is null or Var(R) is null or 0.",
    "s2_over_m_1m": "s^2 / (m(1-m)). Null when s is null or m is 0 or 1.",
    "residual": "Var(R) - s^2 * m * (1-m). Null when an input is null.",
    "information_per_round": "s^2 / residual. Null when the residual is null or not positive.",
    "beta_thompson": (
        "One Beta draw per arm. Posterior alpha is prior_alpha plus the discounted "
        "sum of rewards, and beta is prior_beta plus the discounted sum of 1-R. "
        "The prior is the general routing weight times ROUTER_PRIOR_STRENGTH. "
        "Ties keep the earlier arm in the deployed priority order."
    ),
    "gaussian_thompson": (
        "The Gaussian prior matches the Beta prior mean and variance. The pseudo-count "
        "n0 is ROUTER_PRIOR_STRENGTH. Observation count, observation mean, and the sum "
        "of squared deviations decay by the discount. The prior does not decay. "
        "The posterior mean blends n0 * prior_mean with n_obs * observation_mean. "
        "Until two observations have been counted the draw variance is the prior variance. "
        "After that it is the unbiased observation variance. The draw is "
        "Normal(posterior_mean, sqrt(variance / (n0 + n_obs))) from random.gauss."
    ),
    "streams": (
        "Each stream draws a question with replacement for the configured number of "
        "rounds. The reward is joined correct. Seeds are seed, seed+1, ... ."
    ),
    "drift": (
        "On each observation the stored observation statistics are multiplied by the "
        "discount and the new reward is added. The prior is left as it started."
    ),
    "judge_coverage": (
        "Question ids are shuffled once with stream judge_coverage. Each rate keeps "
        "the first take_count(n, rate) of that shuffle, so 1% is a prefix of 5%, "
        "20%, and 100%. Judge values outside the prefix are treated as unobserved, "
        "and the verified reward is recomputed from lexical grounding and the "
        "remaining judge values. The long streams are not rerun."
    ),
}

RATIO_KEYS = (
    "c0", "c1", "s", "m", "var_r", "s2_over_var_r", "s2_over_m_1m",
    "residual", "information_per_round",
)

REWARD_SOURCES: tuple[tuple[str, str], ...] = (
    ("self", "rewards.self"),
    ("lexical_grounding", "lexical_grounding"),
    ("nli_grounding", "nli_grounding"),
    ("judge", "judge"),
    ("verified", "rewards.verified"),
    ("oracle", "correct"),
)


def _reward_value(row: dict[str, Any], name: str) -> float | None:
    if name == "self":
        value = row["rewards"]["self"]
    elif name == "verified":
        value = row["rewards"]["verified"]
    elif name == "oracle":
        value = row["correct"]
    else:
        value = row.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"reward {name} is {value!r}")
    return float(value)


def _y(row: dict[str, Any]) -> int:
    value = row["correct"]
    if isinstance(value, bool):
        bit = int(value)
    elif isinstance(value, int) and value in (0, 1):
        bit = value
    elif isinstance(value, float) and value in (0.0, 1.0):
        bit = int(value)
    else:
        raise ProtocolError(f"joined correct must be 0 or 1, got {value!r}")
    return bit


def information_ratios(pairs: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Ratios for observed ``(Y, R)`` pairs. Nulls carry a reason."""
    out: dict[str, Any] = {"n": len(pairs)}
    reasons: dict[str, str] = {}
    for key in RATIO_KEYS:
        out[key] = None
    if not pairs:
        reasons["all"] = "no observed rewards"
        out["null_reasons"] = reasons
        return out
    ys = [y for y, _r in pairs]
    rs = [r for _y, r in pairs]
    n = len(pairs)
    m = sum(ys) / n
    out["m"] = m
    y0 = [r for y, r in pairs if y == 0]
    y1 = [r for y, r in pairs if y == 1]
    c0 = sum(y0) / len(y0) if y0 else None
    c1 = sum(y1) / len(y1) if y1 else None
    out["c0"] = c0
    out["c1"] = c1
    if c0 is None or c1 is None:
        reasons["s"] = "one of the two correctness classes has no observed reward"
        s = None
    else:
        s = c1 - c0
    out["s"] = s
    if n < 2:
        reasons["var_r"] = "fewer than two observed rewards"
        var_r = None
    else:
        mean_r = sum(rs) / n
        var_r = sum((r - mean_r) ** 2 for r in rs) / (n - 1)
    out["var_r"] = var_r
    if s is None:
        reasons["s2_over_var_r"] = reasons["s"]
        reasons["s2_over_m_1m"] = reasons["s"]
        reasons["residual"] = reasons["s"]
        reasons["information_per_round"] = reasons["s"]
    else:
        s2 = s * s
        if var_r is None:
            reasons["s2_over_var_r"] = reasons["var_r"]
        elif var_r == 0.0:
            reasons["s2_over_var_r"] = "Var(R) is 0"
        else:
            out["s2_over_var_r"] = s2 / var_r
        if m == 0.0 or m == 1.0:
            reasons["s2_over_m_1m"] = "m is 0 or 1"
        else:
            out["s2_over_m_1m"] = s2 / (m * (1.0 - m))
        if var_r is None or m == 0.0 or m == 1.0:
            reasons["residual"] = "residual inputs are missing or m is 0 or 1"
            reasons["information_per_round"] = reasons["residual"]
        else:
            residual = var_r - s2 * m * (1.0 - m)
            if residual <= 1e-12:
                reasons["residual"] = "residual is not positive"
                reasons["information_per_round"] = "residual is not positive"
            else:
                out["residual"] = residual
                out["information_per_round"] = s2 / residual
    out["null_reasons"] = reasons
    return out


def ratios_by_dataset(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    grouped["all"] = list(rows)
    out: dict[str, Any] = {}
    for dataset, group in grouped.items():
        out[dataset] = {}
        for name, _source in REWARD_SOURCES:
            pairs = []
            for row in group:
                reward = _reward_value(row, name)
                if reward is None or isinstance(reward, float) and math.isnan(reward):
                    continue
                pairs.append((_y(row), reward))
            out[dataset][name] = information_ratios(pairs)
    return out


def _pool(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_qid[str(row["qid"])][str(row["arm"])] = row
    questions = []
    excluded = 0
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for qid, arms in by_qid.items():
        if any(arm not in arms for arm in ARMS):
            excluded += 1
            continue
        first = arms[ARMS[0]]
        ys = {arm: _y(arms[arm]) for arm in ARMS}
        qt = str(first["question_type"])
        questions.append({
            "qid": qid,
            "question_type": qt,
            "y": ys,
            "best_y": max(ys.values()),
        })
        for arm, bit in ys.items():
            buckets[qt][arm].append(bit)
    mu: dict[str, Any] = {}
    for qt, arms in buckets.items():
        means = {arm: sum(vals) / len(vals) for arm, vals in arms.items()}
        best = max(means.values())
        mu[qt] = {
            "mean": means,
            "best": best,
            "best_arms": {arm for arm, value in means.items() if value == best},
        }
    return questions, mu, excluded


def gaussian_prior(weight: float, strength: float) -> dict[str, float]:
    alpha = weight * strength
    beta = (1.0 - weight) * strength
    total = alpha + beta
    mean = alpha / total
    var = alpha * beta / (total * total * (total + 1.0))
    return {"mean": mean, "variance": var, "n0": strength, "alpha": alpha, "beta": beta}


def beta_update(obs_a: float, obs_b: float, reward: float, discount: float) -> tuple[float, float]:
    """Decay the observation counts, then add the reward. The prior is not in here."""
    return discount * obs_a + reward, discount * obs_b + (1.0 - reward)


def gaussian_update(state: dict[str, float], reward: float, discount: float) -> None:
    """Decay the observation statistics, then fold in ``reward``. The prior stays put."""
    n_obs = state["n_obs"]
    if n_obs == 0.0:
        state["n_obs"] = 1.0
        state["mean_obs"] = reward
        state["m2"] = 0.0
        return
    n_new = discount * n_obs + 1.0
    delta = reward - state["mean_obs"]
    mean = state["mean_obs"] + delta / n_new
    state["m2"] = discount * state["m2"] + delta * (reward - mean)
    state["mean_obs"] = mean
    state["n_obs"] = n_new


def gaussian_draw_params(prior: dict[str, float], state: dict[str, float]) -> tuple[float, float]:
    n_obs = state["n_obs"]
    n0 = prior["n0"]
    if n_obs == 0.0:
        mean = prior["mean"]
    else:
        mean = (n0 * prior["mean"] + n_obs * state["mean_obs"]) / (n0 + n_obs)
    if n_obs < 2.0:
        var = prior["variance"]
    else:
        var = state["m2"] / (n_obs - 1.0)
        if var <= 0.0:
            var = prior["variance"]
    sd = math.sqrt(var / (n0 + n_obs))
    return mean, sd


def discount_label(discount: float | None) -> str:
    if discount is None:
        return "none"
    return format(float(discount), ".6g")


def _select_beta(
    rng: random.Random,
    qt: str,
    obs: dict[str, dict[str, list[float]]],
    prior_a: dict[str, float],
    prior_b: dict[str, float],
    priority: Sequence[str],
) -> str:
    best_arm = priority[0]
    best_v = -1.0
    for arm in priority:
        alpha = prior_a[arm] + obs[qt][arm][0]
        beta = prior_b[arm] + obs[qt][arm][1]
        draw = rng.betavariate(alpha, beta)
        if draw > best_v:
            best_v = draw
            best_arm = arm
    return best_arm


def _select_gaussian(
    rng: random.Random,
    qt: str,
    state: dict[str, dict[str, dict[str, float]]],
    priors: dict[str, dict[str, float]],
    priority: Sequence[str],
) -> str:
    best_arm = priority[0]
    best_v = -math.inf
    for arm in priority:
        mean, sd = gaussian_draw_params(priors[arm], state[qt][arm])
        draw = rng.gauss(mean, sd)
        if draw > best_v:
            best_v = draw
            best_arm = arm
    return best_arm


def _one_stream(
    questions: Sequence[dict[str, Any]],
    mu: dict[str, Any],
    rng: random.Random,
    policy: str,
    discount: float,
    prior_a: dict[str, float],
    prior_b: dict[str, float],
    gauss_priors: dict[str, dict[str, float]],
    priority: Sequence[str],
    rounds: int,
) -> dict[str, float]:
    qtypes = sorted({q["question_type"] for q in questions})
    n_q = len(questions)
    pseudo = 0.0
    realized = 0.0
    hits = 0
    if policy == "beta":
        obs = {
            qt: {arm: [0.0, 0.0] for arm in ARMS}
            for qt in qtypes
        }
    elif policy == "gaussian":
        gauss = {
            qt: {arm: {"n_obs": 0.0, "mean_obs": 0.0, "m2": 0.0} for arm in ARMS}
            for qt in qtypes
        }
    else:
        raise ProtocolError(f"unknown study policy {policy!r}")
    for _ in range(rounds):
        question = questions[rng.randrange(n_q)]
        qt = question["question_type"]
        if policy == "beta":
            arm = _select_beta(rng, qt, obs, prior_a, prior_b, priority)
        else:
            arm = _select_gaussian(rng, qt, gauss, gauss_priors, priority)
        y = question["y"][arm]
        reward = float(y)
        if policy == "beta":
            oa, ob = obs[qt][arm]
            obs[qt][arm] = list(beta_update(oa, ob, reward, discount))
        else:
            gaussian_update(gauss[qt][arm], reward, discount)
        mu_qt = mu[qt]
        pseudo += mu_qt["best"] - mu_qt["mean"][arm]
        realized += question["best_y"] - y
        if arm in mu_qt["best_arms"]:
            hits += 1
    return {
        "pseudo_regret": pseudo,
        "realized_regret": realized,
        "best_arm_share": hits / rounds,
    }


def run_streams(
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    rounds: int,
    discounts: Sequence[float],
) -> dict[str, Any]:
    if rounds < 1:
        raise ProtocolError(f"replay rounds must be at least 1, got {rounds}")
    if not seeds:
        raise ProtocolError("the study needs at least one seed")
    for discount in discounts:
        if not isinstance(discount, (int, float)) or isinstance(discount, bool):
            raise ProtocolError(f"drift discount {discount!r} is not a number")
        if float(discount) <= 0.0 or float(discount) > 1.0:
            raise ProtocolError(f"drift discount {discount} is outside (0, 1]")
    questions, mu, excluded = _pool(rows)
    strength = float(RAG._PRIOR_STRENGTH)
    weights = general_priors()
    prior_a = {arm: weights[arm] * strength for arm in ARMS}
    prior_b = {arm: (1.0 - weights[arm]) * strength for arm in ARMS}
    gauss_priors = {arm: gaussian_prior(weights[arm], strength) for arm in ARMS}
    priority = list(RAG._STRATEGY_PRIORITY)
    body: dict[str, Any] = {
        "reward": "oracle",
        "reward_definition": "Joined correct, as a 0/1 reward.",
        "prior": "general routing prior, scaled by ROUTER_PRIOR_STRENGTH, shared by every question type.",
        "prior_strength": strength,
        "prior_weights": weights,
        "gaussian_priors": gauss_priors,
        "n_questions": len(questions),
        "n_excluded": excluded,
        "n_rounds": rounds,
        "seeds": list(seeds),
        "definitions": {
            "beta_thompson": DEFINITIONS["beta_thompson"],
            "gaussian_thompson": DEFINITIONS["gaussian_thompson"],
            "streams": DEFINITIONS["streams"],
            "drift": DEFINITIONS["drift"],
        },
        "streams": [],
    }
    if not questions:
        body["reason"] = "no question has all four arms"
        return body
    factors: list[tuple[str, float]] = [("none", 1.0)]
    for discount in discounts:
        factors.append((discount_label(float(discount)), float(discount)))
    for policy in ("beta", "gaussian"):
        for label, factor in factors:
            pseudo = []
            realized = []
            share = []
            for seed_i in seeds:
                rng = random.Random(stream_seed(int(seed_i), f"study:{policy}:{label}"))
                one = _one_stream(
                    questions, mu, rng, policy, factor, prior_a, prior_b,
                    gauss_priors, priority, rounds,
                )
                pseudo.append(one["pseudo_regret"])
                realized.append(one["realized_regret"])
                share.append(one["best_arm_share"])
            n = len(seeds)
            body["streams"].append({
                "policy": policy,
                "discount_label": label,
                "discount": None if label == "none" else factor,
                "n_seeds": n,
                "n_rounds": rounds,
                "pseudo_regret_mean": sum(pseudo) / n,
                "realized_regret_mean": sum(realized) / n,
                "best_arm_share_mean": sum(share) / n,
                "pseudo_regret_by_seed": pseudo,
                "realized_regret_by_seed": realized,
                "best_arm_share_by_seed": share,
            })
    return body


def judge_coverage(
    rows: Sequence[dict[str, Any]],
    rates: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    qids = sorted({str(row["qid"]) for row in rows})
    judge_values = [row.get("judge") for row in rows]
    present = any(value is not None for value in judge_values)
    reason = None
    if not present:
        reasons = sorted({str(row.get("judge_reason")) for row in rows})
        reason = "judge values are absent (" + ", ".join(reasons) + ")"
    blocks = []
    for rate in rates:
        rate_f = float(rate)
        if rate_f < 0.0 or rate_f > 1.0:
            raise ProtocolError(f"judge coverage rate {rate} is outside [0, 1]")
        label = format(rate_f, ".6g")
        stream = "judge_coverage"
        stream_value = stream_seed(seed, stream)
        chosen = seeded_ids(qids, rate_f, stream_value)
        block: dict[str, Any] = {
            "rate": rate_f,
            "rate_label": label,
            "stream": stream,
            "stream_seed": stream_value,
            "n_questions": len(qids),
            "n_sampled": len(chosen),
            "sampled_qids": chosen,
            "rule": "sorted question ids, Random(stream_seed).shuffle, first take_count(n, rate)",
        }
        block["reason"] = reason
        block["verified"] = None
        block["judge"] = None
        blocks.append(block)
    return {"reason": reason, "rates": blocks, "judge_observed": present}


def judge_coverage_weighted(
    rows: Sequence[dict[str, Any]],
    rates: Sequence[float],
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Same sample as ``judge_coverage``, with config weights on the verified reward."""
    body = judge_coverage(rows, rates, seed)
    if not body["judge_observed"]:
        return body
    g_w = float(cfg["grounding_weight"])
    j_w = float(cfg["judge_weight"])
    qids = sorted({str(row["qid"]) for row in rows})
    for block in body["rates"]:
        kept = set(block["sampled_qids"])
        pairs = []
        judge_pairs = []
        for row in rows:
            judge_value = row.get("judge") if str(row["qid"]) in kept else None
            verified = combine_reward({
                "grounding": (row.get("lexical_grounding"), g_w),
                "judge": (judge_value, j_w),
            })
            y = _y(row)
            if verified is not None:
                pairs.append((y, float(verified)))
            if judge_value is not None and not isinstance(judge_value, bool):
                judge_pairs.append((y, float(judge_value)))
        block["verified"] = information_ratios(pairs)
        block["judge"] = information_ratios(judge_pairs)
        block["weights"] = {"grounding": g_w, "judge": j_w}
    body["n_questions_in_log"] = len(qids)
    return body


def assumption_checks(
    rows: Sequence[dict[str, Any]], seed: int, n_boot: int,
) -> dict[str, Any]:
    if n_boot < 1:
        raise ProtocolError(f"bootstrap samples must be at least 1, got {n_boot}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    datasets: dict[str, Any] = {}
    for dataset, group in grouped.items():
        datasets[dataset] = {}
        for signal in ("self", "lexical_grounding", "nli_grounding", "judge"):
            prepared = _signal_rows(group, signal)
            rng = random.Random(stream_seed(seed, f"assumption:{dataset}:{signal}"))
            block = _arm_conditional(prepared, rng, n_boot)
            datasets[dataset][signal] = {
                "n_nonoverlapping_pairs": len(block["nonoverlapping_pairs"]),
                "nonoverlapping_pairs": block["nonoverlapping_pairs"],
                "by_arm": block["by_arm"],
            }
    return {
        "definition": (
            "Cluster-bootstrap intervals of E[R | Y, arm, R observed]. "
            "A non-overlapping pair is two arms whose intervals at the same Y do not overlap."
        ),
        "bootstrap_samples": n_boot,
        "datasets": datasets,
    }


def score_second_generator(
    nova_rows: Sequence[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    keys: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Token F1 of the Nova answers on the subset. A missing row leaves the means null."""
    by_key = {(str(row["qid"]), str(row["arm"])): row for row in nova_rows}
    incomplete = []
    scored = []
    for dataset, qid, arm in keys:
        row = by_key.get((qid, arm))
        if row is None or "answer" not in row:
            incomplete.append({
                "dataset": dataset, "qid": qid, "arm": arm,
                "reason": "nova generation has no row for this question and arm",
            })
            continue
        if row.get("model_id") != NOVA_MODEL_ID or row.get("temperature") != NOVA_TEMPERATURE:
            incomplete.append({
                "dataset": dataset, "qid": qid, "arm": arm,
                "reason": (
                    f"model_id {row.get('model_id')!r} temperature {row.get('temperature')!r} "
                    f"is not {NOVA_MODEL_ID} at temperature {NOVA_TEMPERATURE}"
                ),
            })
            continue
        question = questions.get(qid)
        if question is None:
            raise ProtocolError(f"nova row {qid} is not in the sample")
        scored_row = score_answer(row["answer"], question["gold_answers"], bool(question["unanswerable"]))
        scored.append({
            "dataset": dataset,
            "qid": qid,
            "arm": arm,
            "f1": scored_row["f1"],
            "token_f1_correct": scored_row["correct"],
        })
    partial = bool(incomplete)
    datasets: dict[str, Any] = {}
    names = sorted({dataset for dataset, _qid, _arm in keys})
    for dataset in names:
        group = [row for row in scored if row["dataset"] == dataset]
        if partial or not group:
            datasets[dataset] = {
                "n": 0,
                "mean_f1": None,
                "token_f1_correct_rate": None,
                "reason": "nova rows are missing, so no mean is published" if partial else "no rows",
            }
            continue
        datasets[dataset] = {
            "n": len(group),
            "mean_f1": sum(row["f1"] for row in group) / len(group),
            "token_f1_correct_rate": sum(row["token_f1_correct"] for row in group) / len(group),
            "reason": None,
        }
    return {
        "model_id": NOVA_MODEL_ID,
        "temperature": NOVA_TEMPERATURE,
        "partial": partial,
        "n_incomplete": len(incomplete),
        "incomplete": incomplete,
        "rows": [] if partial else scored,
        "datasets": datasets,
        "definition": "Token F1 of the Nova answer against the sample gold, on subset questions only.",
    }


def configured_study(cfg: dict[str, Any]) -> dict[str, Any]:
    """The v2 study sizes. The short replay seed list is a different key."""
    n_seeds = cfg["v2"]["replay_seeds"]
    rounds = cfg["v2"]["replay_rounds"]
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, int):
        raise ProtocolError(
            f"v2.replay_seeds is {n_seeds!r}. It is the stream count. "
            "The short replay list stays on the top-level replay_seeds key."
        )
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ProtocolError(f"v2.replay_rounds is {rounds!r}")
    coverage = cfg["v2"]["judge_coverage"]
    discounts = cfg["v2"]["drift_discounts"]
    if not isinstance(coverage, list) or not coverage:
        raise ProtocolError("v2.judge_coverage must be a non-empty list")
    if not isinstance(discounts, list) or not discounts:
        raise ProtocolError("v2.drift_discounts must be a non-empty list")
    return {
        "replay_seeds": n_seeds,
        "replay_rounds": rounds,
        "judge_coverage": [float(rate) for rate in coverage],
        "drift_discounts": [float(rate) for rate in discounts],
    }


def seed_list(start: int, count: int) -> list[int]:
    if count < 1:
        raise ProtocolError(f"seed count must be at least 1, got {count}")
    return [start + i for i in range(count)]
