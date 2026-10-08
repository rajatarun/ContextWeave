"""Order of the end-to-end experiment runner.

The projection is printed by the caller before any of these commands. Model
stages receive ``--max-usd`` on a real run and ``--dry-run`` on a dry run.
A real run without ``--max-usd`` does not start them. Bulk stages (generate,
judge, subset Haiku, subset Nova) receive ``inference_mode``. The config
default is batch. Adjudication does not: that stage calls on demand.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "experiments"


def _py(*parts: str) -> list[str]:
    return [sys.executable, str(SCRIPTS.joinpath(*parts))]


def stage_commands(
    *,
    results: Path,
    seed: int | None,
    max_usd: float | None,
    total_usd_cap: float | None,
    inference_mode: str | None,
    dry_run: bool,
    datasets: str | None,
) -> list[list[str]]:
    def common(script: str) -> list[str]:
        cmd = _py(script)
        cmd += ["--results", str(results)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if datasets:
            cmd += ["--datasets", datasets]
        return cmd

    def paid(script: str, *, mode: bool) -> list[str]:
        cmd = common(script)
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd += ["--max-usd", str(max_usd)]
            if total_usd_cap is not None:
                cmd += ["--total-usd-cap", str(total_usd_cap)]
        if mode and inference_mode:
            cmd += ["--inference-mode", inference_mode]
        return cmd

    commands = [
        common("sample_datasets.py"),
        common("retrieve.py"),
        paid("generate.py", mode=True),
        paid("adjudicate.py", mode=False),
    ]
    for signal in ("lexical", "nli", "judge", "self_percentile", "oracle", "logistic"):
        cmd = common("signals.py")
        # signals.py takes the signal name as a positional before the options
        # that common() appended. Rebuild so the positional stays first.
        cmd = _py("signals.py") + [signal, "--results", str(results)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if datasets:
            cmd += ["--datasets", datasets]
        if signal == "judge":
            if dry_run:
                cmd.append("--dry-run")
            else:
                cmd += ["--max-usd", str(max_usd)]
                if total_usd_cap is not None:
                    cmd += ["--total-usd-cap", str(total_usd_cap)]
            if inference_mode:
                cmd += ["--inference-mode", inference_mode]
        commands.append(cmd)

    select = _py("subset.py") + ["select", "--results", str(results)]
    if seed is not None:
        select += ["--seed", str(seed)]
    if datasets:
        select += ["--datasets", datasets]
    commands.append(select)
    for action in ("haiku", "nova"):
        cmd = _py("subset.py") + [action, "--results", str(results)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if datasets:
            cmd += ["--datasets", datasets]
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd += ["--max-usd", str(max_usd)]
            if total_usd_cap is not None:
                cmd += ["--total-usd-cap", str(total_usd_cap)]
        if inference_mode:
            cmd += ["--inference-mode", inference_mode]
        commands.append(cmd)
    agreement = _py("subset.py") + ["agreement", "--results", str(results)]
    if seed is not None:
        agreement += ["--seed", str(seed)]
    commands.append(agreement)

    commands.append(common("calibrate.py"))
    replay = _py("replay.py") + ["--results", str(results)]
    if datasets:
        replay += ["--datasets", datasets]
    # The short replay keeps its own seed list. The 500-seed streams are study.py.
    commands.append(replay)
    commands.append(common("analyses.py"))
    commands.append(common("study.py"))
    write = _py("write_results.py") + ["--results", str(results)]
    if seed is not None:
        write += ["--seed", str(seed)]
    commands.append(write)
    return commands
