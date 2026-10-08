"""Build markdown tables from ``results/`` only.

A missing file, a missing key, or a null estimate becomes the cell text
``pending``. ``PENDING.md`` lists every such cell and the reason. Numbers are
formatted from the stored JSON; this module does not compute a metric that
the artifact does not already contain, except the simulation-assumption table,
which is copied out of ``results/simulation_assumptions.json``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.assumed import assumed_slopes
from experiments.claims import claim_verdicts
from experiments.common import artifact_meta, load_config, read_json, write_json
from experiments.judge_access import access_reason

DATASETS = ("squad", "hotpot", "nq")
SIGNALS = ("self", "lexical_grounding", "nli_grounding", "judge")
HARNESS = ("coverage", "brier", "ece", "auroc", "spearman", "kendall_tau", "rank_agrees")
REWARDS = (
    "self", "self_with_fallbacks", "lexical_grounding", "verified",
    "verified_plus_self", "normalized_self", "oracle",
)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _ci(block: dict[str, Any] | None) -> tuple[str, str | None]:
    if not isinstance(block, dict) or block.get("estimate") is None:
        return "pending", "estimate is null or the block is missing"
    text = _fmt(block["estimate"])
    if block.get("lo") is not None and block.get("hi") is not None:
        text += f" [{_fmt(block['lo'])}, {_fmt(block['hi'])}]"
    return text, None


class _Doc:
    def __init__(self) -> None:
        self.lines: list[str] = ["# Experiment findings", ""]
        self.lines.append(
            "Every number is copied from a file under `results/`. "
            "A cell whose artifact is missing or whose estimate is null is `pending`."
        )
        self.lines.append("")
        self.pending: list[dict[str, str]] = []

    def h(self, text: str) -> None:
        self.lines += [f"## {text}", ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def table(self, title: str, headers: list[str], rows: list[list[str]]) -> None:
        self.lines.append(f"### {title}")
        self.lines.append("")
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(row) + " |")
        self.lines.append("")

    def cell(self, table: str, cell: str, value: Any, reason: str) -> str:
        if value is None:
            self.pending.append({"table": table, "cell": cell, "reason": reason})
            return "pending"
        return _fmt(value)

    def ci_cell(self, table: str, cell: str, block: Any, missing_reason: str) -> str:
        if not isinstance(block, dict):
            self.pending.append({"table": table, "cell": cell, "reason": missing_reason})
            return "pending"
        text, why = _ci(block)
        if why:
            self.pending.append({"table": table, "cell": cell, "reason": f"{missing_reason}: {why}"})
            return "pending"
        return text


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    return data


def write_assumptions(path: Path, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    body = artifact_meta(cfg, seed, stage="simulation_assumptions", assumptions=assumed_slopes())
    write_json(path, body)
    return body


def _ratio_cell(doc: _Doc, table: str, cell: str, block: Any, key: str) -> str:
    if not isinstance(block, dict):
        return doc.cell(table, cell, None, "ratio block missing")
    value = block.get(key)
    if value is None:
        reasons = block.get("null_reasons") or {}
        reason = reasons.get(key) or reasons.get("all") or f"{key} null"
        return doc.cell(table, cell, None, str(reason))
    return doc.cell(table, cell, value, f"{key} null")


def _pending_file(doc: _Doc, results: Path, relative: str, table: str) -> None:
    path = results / relative
    if not path.is_file():
        return
    data = read_json(path)
    rows = data.get("rows") or []
    reason = str(data.get("detail") or data.get("reason") or "pending")
    doc.p(f"`results/{relative}` lists {len(rows)} rows not yet written. {reason}")
    doc.pending.append({"table": table, "cell": f"{len(rows)} rows", "reason": reason})


def render(results: Path) -> tuple[str, str]:
    doc = _Doc()
    fraction = float(load_config()["normalized_self_gap_fraction"])
    marker = access_reason(results)
    doc.h("Claims")
    for claim in claim_verdicts(results, fraction):
        doc.p(f"**{claim['id']}**: {claim['verdict']}.")
        doc.p(claim["because"])
        doc.p("Artifacts: " + ", ".join(f"`{path}`" for path in claim["artifacts"]) + ".")
        if claim["verdict"] == "open":
            doc.pending.append({"table": "claims", "cell": claim["id"], "reason": claim["because"]})
    _pending_file(doc, results, "generation/pending.json", "generation")
    _pending_file(doc, results, "signals/judge_pending.json", "judge")
    _pending_file(doc, results, "adjudication/pending.json", "adjudication")

    sample = _load(results / "samples" / "sample_manifest.json")
    doc.h("Sample")
    if sample is None:
        doc.p("Sample manifest is missing.")
        for name in DATASETS:
            doc.pending.append({"table": "sample", "cell": name, "reason": "results/samples/sample_manifest.json is missing"})
    else:
        counts = sample.get("counts") or {}
        rows = []
        for name in DATASETS:
            n = counts.get(name)
            rows.append([name, doc.cell("sample", f"{name} n", n, f"no count for {name}")])
        doc.table("Sampled questions", ["dataset", "n"], rows)
        doc.p(f"Seed stored in the manifest: `{sample.get('seed')}`.")
        order = (sample.get("sampling") or {}).get("order")
        if isinstance(order, str) and order:
            doc.p(order)

    doc.h("Retrieval")
    stats = _load(results / "retrieval" / "retrieval_stats.json")
    headers = ["dataset", "n", "pool min", "pool mean", "pool max", "top-1 differs", "mean Jaccard"]
    rows = []
    for name in DATASETS:
        block = (stats or {}).get("datasets", {}).get(name) if stats else None
        if block is None:
            reason = "results/retrieval/retrieval_stats.json is missing" if stats is None else f"no retrieval stats for {name}"
            rows.append([name] + ["pending"] * 6)
            for col in headers[1:]:
                doc.pending.append({"table": "retrieval", "cell": f"{name} {col}", "reason": reason})
            continue
        rows.append([
            name,
            doc.cell("retrieval", f"{name} n", block.get("n_questions"), "n_questions null"),
            doc.cell("retrieval", f"{name} pool min", block.get("pool_size_min"), "pool_size_min null"),
            doc.cell("retrieval", f"{name} pool mean", block.get("pool_size_mean"), "pool_size_mean null"),
            doc.cell("retrieval", f"{name} pool max", block.get("pool_size_max"), "pool_size_max null"),
            doc.cell("retrieval", f"{name} top-1 differs", block.get("fraction_top1_differs_across_arms"), "fraction null"),
            doc.cell("retrieval", f"{name} Jaccard", block.get("mean_pairwise_topk_jaccard"), "jaccard null"),
        ])
    doc.table("Retrieval over the per-question pools", headers, rows)

    doc.h("Generation dry-run")
    dry = _load(results / "generation" / "dry_run_cost.json")
    est = (dry or {}).get("estimate") if dry else None
    if est is None:
        doc.p("Dry-run cost artifact is missing.")
        doc.pending.append({
            "table": "dry-run", "cell": "usd_upper_bound",
            "reason": "results/generation/dry_run_cost.json is missing",
        })
    else:
        doc.table("Generator cost upper bound", ["field", "value"], [
            ["model", _fmt(est.get("model_id"))],
            ["calls", doc.cell("dry-run", "calls", est.get("n_calls"), "n_calls null")],
            ["input tokens (estimate)", doc.cell("dry-run", "input tokens", est.get("input_tokens_estimate"), "input tokens null")],
            ["output tokens (upper bound)", doc.cell("dry-run", "output tokens", est.get("output_tokens_upper_bound"), "output tokens null")],
            ["USD upper bound", doc.cell("dry-run", "usd", est.get("usd_upper_bound"), "usd null")],
        ])
        doc.p(str(est.get("estimator", "")))
        doc.p(str(est.get("output_policy", "")))

    doc.h("Calibration")
    doc.p(
        "The judge grades how well the retrieved passages support the answer "
        "and whether the answer addresses the question. Its prompt scores 1.0 "
        "when every claim is supported and the question is answered, and it "
        "scores 1.0 for an answer that says the evidence is insufficient when "
        "the evidence is insufficient. The score is that grounding judgement. "
        "It is not token-F1 correctness. The table below pairs each stored "
        "signal, including the judge, with token F1 as recorded. A judge score "
        "of 1.0 on an abstention stays 1.0 when correctness is 0 because a "
        "gold answer existed."
    )
    cal = _load(results / "calibration" / "calibration.json")
    cal_rows = []
    for name in DATASETS:
        dblock = ((cal or {}).get("datasets") or {}).get(name) if cal else None
        for signal in SIGNALS:
            sblock = ((dblock or {}).get("signals") or {}).get(signal) if dblock else None
            if cal is None:
                reason = marker if (signal == "judge" and marker) else "results/calibration/calibration.json is missing"
            elif signal == "judge" and ((cal or {}).get("unavailable_signals") or {}).get("judge") and sblock is None:
                reason = (cal or {})["unavailable_signals"]["judge"]
            elif dblock is None:
                reason = f"no calibration block for {name}"
            elif sblock is None:
                reason = f"no {signal} block for {name}"
            else:
                reason = ""
            cells = [name, signal]
            for metric in HARNESS:
                cell_name = f"{name} {signal} {metric}"
                if sblock is None:
                    doc.pending.append({"table": "calibration", "cell": cell_name, "reason": reason})
                    cells.append("pending")
                    continue
                if metric == "rank_agrees":
                    cells.append(doc.cell("calibration", cell_name, sblock.get("rank_agrees"), f"{cell_name} null"))
                    continue
                ci = (sblock.get("ci") or {}).get(metric)
                if ci is None and sblock.get(metric) is None:
                    cells.append(doc.ci_cell("calibration", cell_name, None, f"{cell_name} missing"))
                elif isinstance(ci, dict):
                    cells.append(doc.ci_cell("calibration", cell_name, ci, cell_name))
                else:
                    cells.append(doc.cell("calibration", cell_name, sblock.get(metric), f"{cell_name} null"))
            # rankings are point lists, not intervals
            if sblock is None:
                cells.append("pending")
                cells.append("pending")
                doc.pending.append({"table": "calibration", "cell": f"{name} {signal} rankings", "reason": reason})
            else:
                cells.append(
                    doc.cell(
                        "calibration", f"{name} {signal} correctness ranking",
                        " > ".join(sblock.get("correctness_ranking") or []) or None,
                        "correctness ranking empty",
                    )
                )
                cells.append(
                    doc.cell(
                        "calibration", f"{name} {signal} signal ranking",
                        " > ".join(sblock.get("signal_ranking") or []) or None,
                        "signal ranking empty",
                    )
                )
            cal_rows.append(cells)
    doc.table(
        "Harness metrics",
        ["dataset", "signal", *HARNESS, "correctness order", "signal order"],
        cal_rows,
    )

    doc.h("Prediction")
    pred = (cal or {}).get("prediction") if cal else None
    if not isinstance(pred, dict):
        doc.p("Verdict: pending.")
        doc.pending.append({"table": "prediction", "cell": "overall", "reason": "calibration artifact has no prediction block"})
    else:
        doc.p(f"Overall verdict: **{pred.get('verdict', 'pending')}**.")
        per = ((cal or {}).get("datasets") or {})
        for name in DATASETS:
            v = ((per.get(name) or {}).get("prediction") or {}).get("verdict")
            doc.p(f"{name}: {doc.cell('prediction', name, v, f'no per-dataset verdict for {name}')}.")

    doc.h("Slopes")
    analyses = _load(results / "analyses" / "analyses.json")
    assumptions = _load(results / "simulation_assumptions.json")
    assumed = (assumptions or {}).get("assumptions") if assumptions else None
    slope_rows = []
    for name in DATASETS:
        for signal in SIGNALS:
            block = (((analyses or {}).get("datasets") or {}).get(name) or {}).get(signal) if analyses else None
            if isinstance(block, dict) and block.get("unavailable"):
                slope = None
                reason = str(block["unavailable"])
            else:
                slope = (block or {}).get("slope") if block else None
                if analyses is None:
                    reason = marker if (signal == "judge" and marker) else "results/analyses/analyses.json is missing"
                elif block is None:
                    reason = f"no analyses block for {name} {signal}"
                else:
                    reason = f"slope block missing for {name} {signal}"
            slope_rows.append([
                name, signal,
                doc.ci_cell("slopes", f"{name} {signal} c0", (slope or {}).get("c0") if slope else None, reason),
                doc.ci_cell("slopes", f"{name} {signal} c1", (slope or {}).get("c1") if slope else None, reason),
                doc.ci_cell("slopes", f"{name} {signal} s", (slope or {}).get("s") if slope else None, reason),
            ])
    doc.table("c0, c1, and slope", ["dataset", "signal", "c0", "c1", "s"], slope_rows)
    if assumed is None:
        doc.p("Simulation assumptions file is missing.")
        doc.pending.append({
            "table": "assumptions", "cell": "s_self",
            "reason": "results/simulation_assumptions.json is missing",
        })
    else:
        self_a = assumed["self"]
        ground_a = assumed["grounding"]
        doc.table("Assumed slopes from the verified-reward simulation", ["signal", "c0", "c1", "s"], [
            ["self", _fmt(self_a["c0"]), _fmt(self_a["c1"]), _fmt(self_a["s"])],
            ["grounding", _fmt(ground_a["c0"]), _fmt(ground_a["c1"]), _fmt(ground_a["s"])],
        ])
        doc.p(f"Source: {assumed.get('source')}. {assumed.get('clip')}.")
        regret = assumed.get("routing_regret_sim") or {}
        doc.p(
            f"`routing_regret_sim.py` draws {regret.get('reward')} with noise_k={regret.get('noise_k')}. "
            f"{regret.get('note')}"
        )

    doc.h("Replay")
    replay = _load(results / "replay" / "replay_summary.json")
    replay_rows = []
    for name in DATASETS:
        for reward in REWARDS:
            for update in ("fractional", "bernoulli"):
                block = ((((replay or {}).get("datasets") or {}).get(name) or {}).get(reward) or {}).get(update) if replay else None
                cell = f"{name} {reward} {update}"
                if replay is None:
                    if marker and reward in ("verified", "verified_plus_self"):
                        reason = marker
                    else:
                        reason = "results/replay/replay_summary.json is missing"
                elif reward in ((replay or {}).get("unavailable_rewards") or {}) and block is None:
                    reason = (replay or {})["unavailable_rewards"][reward]
                elif block is None:
                    reason = f"no replay block for {cell}"
                else:
                    reason = f"{cell} null"
                replay_rows.append([
                    name, reward, update,
                    doc.cell("replay", f"{cell} pseudo", (block or {}).get("pseudo_regret_mean") if block else None, reason),
                    doc.cell("replay", f"{cell} realized", (block or {}).get("realized_regret_mean") if block else None, reason),
                    doc.cell("replay", f"{cell} best share", (block or {}).get("best_arm_share_mean") if block else None, reason),
                ])
    doc.table(
        "True-correctness regret (mean over seeds)",
        ["dataset", "reward", "update", "pseudo-regret", "realized regret", "best-arm share"],
        replay_rows,
    )
    doc.p("Curve CSV and PNG paths are listed in the replay artifact when that artifact exists. They are pending while it does not.")
    if replay is None:
        doc.pending.append({"table": "replay", "cell": "curves", "reason": "results/replay/replay_summary.json is missing"})

    doc.h("Correctness sensitivity")
    rule = None
    if analyses:
        for name in DATASETS:
            block = ((analyses.get("datasets") or {}).get(name) or {}).get("correctness_sensitivity")
            if isinstance(block, dict) and block.get("primary_rule"):
                rule = str(block["primary_rule"])
                break
    if rule:
        doc.p(rule)
    else:
        doc.p("The correctness rule is pending until analyses.json stores primary_rule.")
        doc.pending.append({
            "table": "correctness",
            "cell": "primary rule",
            "reason": "results/analyses/analyses.json has no primary_rule",
        })
    sens_rows = []
    for name in DATASETS:
        block = (((analyses or {}).get("datasets") or {}).get(name) or {}).get("correctness_sensitivity") if analyses else None
        if analyses is None:
            reason = "results/analyses/analyses.json is missing"
        elif not isinstance(block, dict):
            reason = f"no correctness sensitivity block for {name}"
        else:
            reason = ""
        sens_rows.append([
            name,
            doc.cell("correctness", f"{name} primary", (block or {}).get("primary_rate") if block else None, reason or "primary rate null"),
            doc.cell("correctness", f"{name} gold contained", (block or {}).get("secondary_rate") if block else None, reason or "secondary rate null"),
        ])
    doc.table(
        "Joined correctness and gold contained in the answer",
        ["dataset", "primary rate (joined correct)", "secondary rate (gold contained)"],
        sens_rows,
    )
    doc.h("HotpotQA yes/no")
    hotpot = ((analyses or {}).get("datasets") or {}).get("hotpot") if analyses else None
    yes_no = hotpot.get("yes_no") if isinstance(hotpot, dict) else None
    if analyses is None:
        yes_reason = "results/analyses/analyses.json is missing"
    elif not isinstance(hotpot, dict):
        yes_reason = "hotpot was not in this analyses run"
    elif not isinstance(yes_no, dict):
        yes_reason = "analyses.json has no hotpot yes_no block"
    else:
        yes_reason = ""
    yes_rows = []
    for label, title in (("yes_no", "yes/no"), ("other", "other HotpotQA")):
        block = yes_no.get(label) if isinstance(yes_no, dict) else None
        if not isinstance(block, dict):
            reason = yes_reason or f"no {label} slice"
            block = {}
        elif block.get("n") == 0:
            reason = f"no HotpotQA rows in the {title} slice"
        else:
            reason = ""
        yes_rows.append([
            title,
            doc.cell("hotpot yes/no", f"{label} n", block.get("n"), reason or "n null"),
            doc.cell("hotpot yes/no", f"{label} correct", block.get("correct_rate"), reason or "correct rate null"),
            doc.cell("hotpot yes/no", f"{label} token f1", block.get("token_f1_rate"), reason or "token F1 rate null"),
            doc.cell("hotpot yes/no", f"{label} gold in top k", block.get("gold_in_top_k_rate"), reason or "gold_in_top_k rate null"),
            doc.cell("hotpot yes/no", f"{label} source retrieved", block.get("source_retrieved_rate"), reason or "source_retrieved rate null"),
            doc.cell("hotpot yes/no", f"{label} abstain correct", block.get("abstain_correct"), reason or "abstain_correct null"),
            doc.cell("hotpot yes/no", f"{label} abstain retrieval miss", block.get("abstain_retrieval_miss"), reason or "abstain_retrieval_miss null"),
            doc.cell("hotpot yes/no", f"{label} abstain incorrect", block.get("abstain_incorrect"), reason or "abstain_incorrect null"),
        ])
    doc.table(
        "HotpotQA yes/no questions, reported apart from the other HotpotQA questions",
        [
            "slice", "n", "joined correct", "token F1", "gold_in_top_k", "source_retrieved",
            "abstain_correct", "abstain_retrieval_miss", "abstain_incorrect",
        ],
        yes_rows,
    )

    doc.h("Assumption check and missingness")
    doc.p(
        "For self-confidence, a null value keeps its status as the missingness "
        "reason. `omitted` is a reply that did not report a confidence, "
        "including a bare abstention whose answer is `insufficient evidence`. "
        "`truncated` is a JSON object cut off before it could be parsed. "
        "An `ok` row with `trailing_truncated` set is an observation: the "
        "object was complete and prose after it hit the output cap."
    )
    if analyses is None:
        doc.p("Analyses artifact is missing.")
        doc.pending.append({"table": "assumption", "cell": "all", "reason": "results/analyses/analyses.json is missing"})
        doc.pending.append({"table": "missingness", "cell": "all", "reason": "results/analyses/analyses.json is missing"})
    else:
        a_rows = []
        m_rows = []
        for name in DATASETS:
            for signal in ("self", "lexical_grounding"):
                block = ((analyses.get("datasets") or {}).get(name) or {}).get(signal)
                if block is None:
                    reason = f"no analyses block for {name} {signal}"
                    a_rows.append([name, signal, "pending", "pending"])
                    doc.pending.append({"table": "assumption", "cell": f"{name} {signal}", "reason": reason})
                    m_rows.append([name, signal, "pending", "pending", "pending"])
                    doc.pending.append({"table": "missingness", "cell": f"{name} {signal}", "reason": reason})
                    continue
                flags = (block.get("assumption") or {}).get("nonoverlapping_pairs")
                any_flag = None if not isinstance(flags, list) else bool(flags)
                a_rows.append([
                    name, signal,
                    doc.cell("assumption", f"{name} {signal} pairs", len(flags) if isinstance(flags, list) else None, "nonoverlapping_pairs missing"),
                    doc.cell("assumption", f"{name} {signal} any", any_flag, "nonoverlapping_pairs missing"),
                ])
                miss = block.get("missingness") or {}
                # one summary row: mean m across arms, and whether accuracies differ enough to report
                ms = [v.get("m") for v in miss.values() if isinstance(v, dict) and v.get("m") is not None]
                acc_m = [v.get("accuracy_when_missing") for v in miss.values() if isinstance(v, dict)]
                acc_o = [v.get("accuracy_when_observed") for v in miss.values() if isinstance(v, dict)]
                m_rows.append([
                    name, signal,
                    doc.cell("missingness", f"{name} {signal} m", (sum(ms) / len(ms)) if ms else None, "no per-arm missingness"),
                    ", ".join(f"{arm}={_fmt(v.get('m'))}" for arm, v in sorted(miss.items())),
                    ", ".join(
                        f"{arm}: miss={_fmt(v.get('accuracy_when_missing'))} obs={_fmt(v.get('accuracy_when_observed'))}"
                        for arm, v in sorted(miss.items())
                    ) if acc_m or acc_o else "pending",
                ])
        doc.table("Non-overlapping E[R | Y, arm] intervals", ["dataset", "signal", "n pairs", "any"], a_rows)
        doc.table("Missingness m_a", ["dataset", "signal", "mean m", "m by arm", "accuracy missing vs observed"], m_rows)
        doc.p(str(analyses.get("normalized_self_definition") or ""))

    doc.h("Cost projection")
    cfg = load_config()
    doc.p(
        f"The Bedrock ceiling is total_usd_cap {cfg['total_usd_cap']}. "
        f"already_spent_usd {cfg['already_spent_usd']} is spend already on the AWS bill "
        "from prior ledgers. The cap check adds it to this results ledger before applying the ceiling. "
        f"Bulk stages default to `{cfg['inference_mode']}`."
    )
    doc.p(
        "Projected spend is copied from `results/cost_projection.json` when that file is present. "
        "That artifact records whether input tokens were measured from the retrieved passages "
        "or taken from the prompt template, whether output tokens are a ledger mean or the "
        "configured maximum, and the adjudication call count, which is an assumption."
    )
    projection = _load(results / "cost_projection.json")
    if projection is None or projection.get("kind") != "projection":
        doc.p("Cost projection artifact is missing.")
        doc.pending.append({
            "table": "cost projection",
            "cell": "all",
            "reason": "results/cost_projection.json is missing",
        })
    else:
        proj_rows = []
        for stage in projection.get("stages") or []:
            name = str(stage.get("stage"))
            calls_reason = stage.get("calls_reason") or "calls null"
            usd_reason = stage.get("usd_reason") or "usd null"
            proj_rows.append([
                name,
                doc.cell("cost projection", f"{name} calls", stage.get("calls"), calls_reason),
                doc.cell("cost projection", f"{name} on-demand", stage.get("on_demand_usd"), usd_reason),
                doc.cell("cost projection", f"{name} batch", stage.get("batch_usd"), usd_reason),
            ])
        doc.table("Projected spend by stage", ["stage", "calls", "on-demand USD", "batch USD"], proj_rows)
        doc.p(str(projection.get("input_scope") or ""))
        doc.p(str(projection.get("sum_note") or ""))
        doc.p(
            "Sum of determined stages, on-demand "
            + doc.cell("cost projection", "sum on-demand", projection.get("sum_on_demand_usd"), "sum null")
            + ", batch "
            + doc.cell("cost projection", "sum batch", projection.get("sum_batch_usd"), "sum null")
            + "."
        )

    doc.h("Subset")
    subset = _load(results / "subset" / "subset.json")
    if subset is None:
        doc.p("Subset manifest is missing.")
        doc.pending.append({
            "table": "subset", "cell": "all",
            "reason": "results/subset/subset.json is missing",
        })
    else:
        scope = subset.get("subset_scope")
        doc.p(f"Scope stored on the manifest: `{scope}`. Seed `{subset.get('seed')}`.")
        sub_rows = []
        for name in DATASETS:
            block = (subset.get("datasets") or {}).get(name)
            reason = f"no subset block for {name}"
            sub_rows.append([
                name,
                doc.cell("subset", f"{name} n", (block or {}).get("n") if isinstance(block, dict) else None, reason),
                doc.cell("subset", f"{name} stream", (block or {}).get("stream") if isinstance(block, dict) else None, reason),
            ])
        doc.table("Seeded subset", ["dataset", "n", "stream"], sub_rows)

    doc.h("Haiku agreement")
    cfg = load_config()
    doc.p(
        f"`v2.subset_samples` is {cfg['v2']['subset_samples']}. "
        f"The subset study draws that many Haiku answers at temperature {cfg['v2']['subset_temperature']}."
    )
    agreement = _load(results / "subset" / "agreement.json")
    if agreement is None:
        doc.p("Agreement artifact is missing.")
        doc.pending.append({
            "table": "agreement", "cell": "all",
            "reason": "results/subset/agreement.json is missing",
        })
    else:
        if agreement.get("definition"):
            doc.p(str(agreement["definition"]))
        doc.p(
            "Samples recorded on the artifact: "
            + doc.cell("agreement", "n_samples", agreement.get("n_samples"), "n_samples missing")
            + "."
        )
        agr_rows = []
        for name in DATASETS:
            block = (agreement.get("datasets") or {}).get(name)
            if not isinstance(block, dict):
                reason = f"no agreement block for {name}"
                block = {}
            else:
                reason = str(block.get("reason") or "agreement value null")
            agr_rows.append([
                name,
                doc.cell("agreement", f"{name} n", block.get("n_scored"), reason),
                doc.cell("agreement", f"{name} modal", block.get("mean_modal_fraction"), reason),
                doc.cell("agreement", f"{name} pairwise", block.get("mean_pairwise_agreement"), reason),
            ])
        doc.table(
            "Temperature-1 Haiku samples",
            ["dataset", "n", "mean modal fraction", "mean pairwise agreement"],
            agr_rows,
        )

    study = _load(results / "study" / "study.json")
    doc.h("Signal information")
    if study is None:
        doc.p("Study artifact is missing.")
        doc.pending.append({
            "table": "information", "cell": "all",
            "reason": "results/study/study.json is missing",
        })
    else:
        definitions = study.get("definitions") or {}
        for key in ("m", "s2_over_var_r", "s2_over_m_1m", "information_per_round"):
            if definitions.get(key):
                doc.p(str(definitions[key]))
        info = study.get("information") or {}
        info_rows = []
        for name in (*DATASETS, "all"):
            for signal in ("self", "lexical_grounding", "nli_grounding", "judge", "verified", "oracle"):
                block = (info.get(name) or {}).get(signal)
                info_rows.append([
                    name, signal,
                    _ratio_cell(doc, "information", f"{name} {signal} n", block, "n"),
                    _ratio_cell(doc, "information", f"{name} {signal} m", block, "m"),
                    _ratio_cell(doc, "information", f"{name} {signal} s2/var", block, "s2_over_var_r"),
                    _ratio_cell(doc, "information", f"{name} {signal} s2/m", block, "s2_over_m_1m"),
                    _ratio_cell(doc, "information", f"{name} {signal} info", block, "information_per_round"),
                ])
        doc.table(
            "s^2 / Var(R), s^2 / (m(1-m)), and information per round",
            ["dataset", "signal", "n", "m", "s^2/Var(R)", "s^2/(m(1-m))", "information per round"],
            info_rows,
        )

    doc.h("Thompson sampling streams")
    if study is None:
        doc.p("Study artifact is missing.")
        doc.pending.append({
            "table": "streams", "cell": "all",
            "reason": "results/study/study.json is missing",
        })
    else:
        thompson = study.get("thompson") or {}
        for key in ("beta_thompson", "gaussian_thompson", "streams", "drift"):
            text = ((thompson.get("definitions") or {}).get(key)) or ((study.get("definitions") or {}).get(key))
            if text:
                doc.p(str(text))
        stream_rows = []
        blocks = thompson.get("streams") or []
        if not blocks:
            reason = str(thompson.get("reason") or "no stream blocks")
            stream_rows.append(["pending", "pending", "pending", "pending", "pending", "pending", "pending"])
            doc.pending.append({"table": "streams", "cell": "all", "reason": reason})
        for block in blocks:
            label = f"{block.get('policy')} {block.get('discount_label')}"
            stream_rows.append([
                doc.cell("streams", f"{label} policy", block.get("policy"), "policy null"),
                doc.cell("streams", f"{label} discount", block.get("discount_label"), "discount null"),
                doc.cell("streams", f"{label} seeds", block.get("n_seeds"), "n_seeds null"),
                doc.cell("streams", f"{label} rounds", block.get("n_rounds"), "n_rounds null"),
                doc.cell("streams", f"{label} pseudo", block.get("pseudo_regret_mean"), "pseudo regret null"),
                doc.cell("streams", f"{label} realized", block.get("realized_regret_mean"), "realized regret null"),
                doc.cell("streams", f"{label} share", block.get("best_arm_share_mean"), "best-arm share null"),
            ])
        doc.table(
            "Beta and Gaussian Thompson sampling on joined correct",
            ["policy", "discount", "seeds", "rounds", "pseudo-regret", "realized regret", "best-arm share"],
            stream_rows,
        )

    doc.h("Judge coverage")
    if study is None:
        doc.p("Study artifact is missing.")
        doc.pending.append({
            "table": "judge coverage", "cell": "all",
            "reason": "results/study/study.json is missing",
        })
    else:
        coverage = study.get("judge_coverage") or {}
        definition = (study.get("definitions") or {}).get("judge_coverage")
        if definition:
            doc.p(str(definition))
        if coverage.get("reason"):
            doc.p(str(coverage["reason"]))
        cov_rows = []
        rate_blocks = coverage.get("rates") or []
        if not rate_blocks:
            doc.pending.append({
                "table": "judge coverage", "cell": "rates",
                "reason": str(coverage.get("reason") or "no coverage rates"),
            })
        for block in rate_blocks:
            rate = block.get("rate")
            for signal in ("verified", "judge"):
                ratios = block.get(signal)
                label = f"{rate} {signal}"
                if ratios is None:
                    reason = str(block.get("reason") or coverage.get("reason") or "coverage ratios null")
                    cov_rows.append([str(rate), signal, "pending", "pending", "pending"])
                    for metric in ("s2/var", "s2/m", "info"):
                        doc.pending.append({"table": "judge coverage", "cell": f"{label} {metric}", "reason": reason})
                    continue
                cov_rows.append([
                    _fmt(rate) if rate is not None else "pending",
                    signal,
                    _ratio_cell(doc, "judge coverage", f"{label} s2/var", ratios, "s2_over_var_r"),
                    _ratio_cell(doc, "judge coverage", f"{label} s2/m", ratios, "s2_over_m_1m"),
                    _ratio_cell(doc, "judge coverage", f"{label} info", ratios, "information_per_round"),
                ])
        if cov_rows:
            doc.table(
                "Judge coverage on the pooled log",
                ["rate", "signal", "s^2/Var(R)", "s^2/(m(1-m))", "information per round"],
                cov_rows,
            )

    doc.h("Study assumption check")
    if study is None:
        doc.p("Study artifact is missing.")
        doc.pending.append({
            "table": "study assumption", "cell": "all",
            "reason": "results/study/study.json is missing",
        })
    else:
        assumption = study.get("assumption") or {}
        if assumption.get("definition"):
            doc.p(str(assumption["definition"]))
        a_rows = []
        for name in DATASETS:
            for signal in SIGNALS:
                block = ((assumption.get("datasets") or {}).get(name) or {}).get(signal)
                reason = f"no study assumption block for {name} {signal}"
                a_rows.append([
                    name, signal,
                    doc.cell(
                        "study assumption", f"{name} {signal} pairs",
                        block.get("n_nonoverlapping_pairs") if isinstance(block, dict) else None,
                        reason,
                    ),
                ])
        doc.table("Non-overlapping pairs from the study artifact", ["dataset", "signal", "n pairs"], a_rows)

    doc.h("Second generator")
    if study is None:
        doc.p("Study artifact is missing.")
        doc.pending.append({
            "table": "nova", "cell": "all",
            "reason": "results/study/study.json is missing",
        })
    else:
        nova = study.get("nova") or {}
        if nova.get("definition"):
            doc.p(str(nova["definition"]))
        if nova.get("reason"):
            doc.p(str(nova["reason"]))
        nova_rows = []
        for name in DATASETS:
            block = (nova.get("datasets") or {}).get(name) if isinstance(nova.get("datasets"), dict) else None
            if not isinstance(block, dict):
                reason = str(nova.get("reason") or f"no nova block for {name}")
                block = {}
            else:
                reason = str(block.get("reason") or "nova value null")
            nova_rows.append([
                name,
                doc.cell("nova", f"{name} n", block.get("n"), reason),
                doc.cell("nova", f"{name} f1", block.get("mean_f1"), reason),
                doc.cell("nova", f"{name} token f1", block.get("token_f1_correct_rate"), reason),
            ])
        doc.table(
            "Nova Pro token F1 on the subset",
            ["dataset", "n", "mean token F1", "token-F1 correct rate"],
            nova_rows,
        )

    findings = "\n".join(doc.lines).rstrip() + "\n"
    pend_lines = ["# Pending cells", ""]
    if not doc.pending:
        pend_lines.append("No pending cells.")
    else:
        pend_lines.append("| table | cell | why |")
        pend_lines.append("| --- | --- | --- |")
        for item in doc.pending:
            why = item["reason"].replace("|", "/")
            pend_lines.append(f"| {item['table']} | {item['cell']} | {why} |")
    pend_lines.append("")
    return findings, "\n".join(pend_lines)
