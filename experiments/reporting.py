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

    doc.h("Assumption check and missingness")
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
