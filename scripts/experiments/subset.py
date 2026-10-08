#!/usr/bin/env python3
"""Seeded subset, temperature-1 Haiku samples, Nova Pro, and agreement.

The sample count is ``v2.subset_samples``.

``select`` writes the question ids. ``haiku`` and ``nova`` read that file and
write generations. ``agreement`` reads the Haiku files. Each command stops
when its input is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (
    ARMS, DATASETS, RESULTS, ProtocolError, artifact_meta, load_config, read_json,
    read_jsonl, stage_jsonl, write_json,
)
from experiments.generate_stage import (
    dry_run, generate_rows, generate_rows_batch, load_retrieval, make_batch_clients,
    make_client, system_prompt,
)
from experiments.ledger import SPEND_CAP_REASON, budget_for
from experiments.subset_stage import (
    choose_subset, expected_keys, require_haiku_settings, require_nova_settings,
    score_agreement,
)


def _names(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ProtocolError(f"unknown datasets: {unknown}")
    return names


def _sample_qids(results: Path, names: list[str]) -> dict[str, list[str]]:
    out = {}
    for name in names:
        path = results / "samples" / f"{name}.jsonl"
        rows = read_jsonl(path)
        qids = []
        for row in rows:
            qid = row.get("qid")
            if not qid:
                raise ProtocolError(f"{path}: a sample row has no qid")
            qids.append(str(qid))
        out[name] = qids
    return out


def _manifest(results: Path) -> dict:
    path = results / "subset" / "subset.json"
    data = read_json(path)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ProtocolError(f"{path} has no datasets")
    return data


def _subset_rows(results: Path, names: list[str], manifest: dict) -> list[dict]:
    rows = []
    for name in names:
        block = (manifest.get("datasets") or {}).get(name)
        if not isinstance(block, dict) or not block.get("qids"):
            raise ProtocolError(f"subset manifest has no qids for {name}")
        wanted = set(block["qids"])
        loaded = load_retrieval(results / "retrieval" / f"{name}.jsonl")
        for arm in ARMS:
            found = {row["qid"] for row in loaded if row["arm"] == arm and row["qid"] in wanted}
            missing = wanted - found
            if missing:
                example = sorted(missing)[0]
                raise ProtocolError(
                    f"retrieval for {name} arm {arm} has no row for subset qid {example}"
                )
        rows.extend(row for row in loaded if row["qid"] in wanted)
    return rows


def _select(args: argparse.Namespace, cfg: dict, seed: int, names: list[str]) -> int:
    n = int(cfg["v2"]["subset_questions"])
    chosen = choose_subset(_sample_qids(args.results, names), n, seed)
    body = artifact_meta(cfg, seed, stage="subset", **chosen)
    path = args.results / "subset" / "subset.json"
    write_json(path, body)
    quotas = body.get("quotas") or {}
    print(
        f"subset total {body['subset_questions']} ({body['subset_scope']}); "
        + ", ".join(f"{name} {quotas.get(name, 0)}" for name in names)
    )
    for name in names:
        block = body["datasets"].get(name)
        if block is None:
            print(f"{name}: 0 questions")
            continue
        print(f"{name}: {block['n']} questions, stream {block['stream']}")
    print(f"wrote {path}")
    return 0


def _run_generation(
    args: argparse.Namespace,
    cfg: dict,
    seed: int,
    names: list[str],
    *,
    model_id: str,
    temperature: float,
    stage: str,
    out_dir: Path,
    note: str,
    repeats: int,
) -> int:
    manifest = _manifest(args.results)
    rows = _subset_rows(args.results, names, manifest)
    run_cfg = {**cfg, "temperature": temperature, "inference_mode": args.inference_mode or cfg["inference_mode"]}
    if args.dry_run:
        estimate = dry_run(run_cfg, rows, model_id)
        estimate["n_repeats"] = repeats
        estimate["n_calls_one_pass"] = estimate["n_calls"]
        for key in (
            "n_calls", "input_tokens_estimate", "output_tokens_upper_bound",
            "usd_upper_bound", "on_demand_usd_upper_bound", "batch_usd_upper_bound",
        ):
            estimate[key] = estimate[key] * repeats
        estimate["repeat_note"] = (
            "Each repeat sends the same prompts. Totals multiply one pass by the repeat count."
        )
        path = out_dir / "dry_run_cost.json"
        write_json(path, artifact_meta(
            run_cfg, seed, stage=f"{stage}_dry_run", estimate=estimate, called_model=False, note=note,
        ))
        print(f"calls: {estimate['n_calls']}")
        print(f"on-demand USD upper bound: {estimate['on_demand_usd_upper_bound']:.6f}")
        print(f"batch USD upper bound: {estimate['batch_usd_upper_bound']:.6f}")
        print(f"wrote {path}")
        return 0
    if args.max_usd is None:
        raise ProtocolError("--max-usd is required. Use --dry-run to estimate first.")
    total_cap = float(cfg["total_usd_cap"]) if args.total_usd_cap is None else args.total_usd_cap
    budget = budget_for(args.results, stage, args.max_usd, cfg, args.total_usd_cap)
    mode = run_cfg["inference_mode"]
    client = None
    s3 = bedrock = None
    if mode == "batch":
        s3, bedrock = make_batch_clients(cfg["region"])
    else:
        client = make_client(cfg["region"])
    pending: list[dict] = []
    stop_reason = None
    for index in range(repeats):
        sample_dir = out_dir / f"sample_{index}" if repeats > 1 else out_dir
        out = stage_jsonl(sample_dir, "generations")
        if mode == "batch":
            summary = generate_rows_batch(
                run_cfg, rows, out, budget, model_id=model_id, s3=s3, bedrock=bedrock, seed=seed,
            )
        else:
            summary = generate_rows(
                run_cfg, rows, out, budget, model_id=model_id, client=client,
            )
        print(
            f"{stage} sample {index}: wrote {summary['written']} skipped {summary['skipped']} "
            f"cap total ${summary['spent_usd']:.6f}"
        )
        meta = artifact_meta(
            run_cfg, seed, stage=stage, model_id=model_id, sample_index=index,
            temperature=temperature, inference_mode=mode, called_model=True,
            bedrock_seed_parameter=False, note=note,
            prompts={"generator_system": system_prompt()},
            stopped=bool(summary["stopped"]), stop_reason=summary["stop_reason"],
        )
        write_json(sample_dir / "meta.json", meta)
        if summary["stopped"]:
            stop_reason = summary["stop_reason"]
            pending.extend(summary["pending"])
            for later in range(index + 1, repeats):
                pending.append({"sample_index": later, "reason": stop_reason})
            break
    pending_path = out_dir / "pending.json"
    if pending:
        write_json(pending_path, {"reason": SPEND_CAP_REASON, "detail": stop_reason, "rows": pending})
        print(stop_reason)
    elif pending_path.is_file():
        pending_path.unlink()
    return 0


def _index_generation(path: Path) -> dict[tuple[str, str], dict]:
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        key = (row["qid"], row["arm"])
        if key in out:
            raise ProtocolError(f"{path}: duplicate (qid, arm) {key}")
        out[key] = row
    return out


def _agreement(args: argparse.Namespace, cfg: dict, seed: int) -> int:
    n_samples, temperature = require_haiku_settings(cfg)
    manifest = _manifest(args.results)
    tables = []
    for index in range(n_samples):
        path = args.results / "subset" / "haiku" / f"sample_{index}" / "generations.jsonl"
        try:
            table = _index_generation(path)
        except ProtocolError as exc:
            raise ProtocolError(
                f"Haiku sample {index} is missing ({path}). Run subset.py haiku first. {exc}"
            ) from exc
        for row in table.values():
            if row.get("temperature") != temperature:
                raise ProtocolError(
                    f"sample {index} qid={row.get('qid')} stored temperature {row.get('temperature')!r}. "
                    f"Agreement expects {temperature}."
                )
        tables.append(table)
    body = score_agreement(expected_keys(manifest), tables)
    path = args.results / "subset" / "agreement.json"
    write_json(path, artifact_meta(
        cfg, seed, stage="agreement", temperature=temperature, **body,
    ))
    print(f"incomplete: {body['n_incomplete']}")
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None)
    common.add_argument("--results", type=Path, default=RESULTS)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--datasets", default=",".join(DATASETS))
    sub.add_parser("select", parents=[common])
    paid = argparse.ArgumentParser(add_help=False)
    paid.add_argument("--max-usd", type=float, default=None)
    paid.add_argument("--total-usd-cap", type=float, default=None)
    paid.add_argument("--inference-mode", choices=("on_demand", "batch"), default=None)
    paid.add_argument("--dry-run", action="store_true")
    sub.add_parser("haiku", parents=[common, paid])
    sub.add_parser("nova", parents=[common, paid])
    sub.add_parser("agreement", parents=[common])
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        seed = cfg["seed"] if args.seed is None else args.seed
        names = _names(args.datasets)
        if args.cmd == "select":
            return _select(args, cfg, seed, names)
        if args.cmd == "haiku":
            n_samples, temperature = require_haiku_settings(cfg)
            return _run_generation(
                args, cfg, seed, names,
                model_id=cfg["generator_model_id"], temperature=temperature,
                stage="subset_haiku", out_dir=args.results / "subset" / "haiku",
                note=(
                    "Temperature 1 is a model random step. The run seed and sample_index "
                    "are logged on meta.json. The Converse request has no seed parameter."
                ),
                repeats=n_samples,
            )
        if args.cmd == "nova":
            model_id, temperature = require_nova_settings(cfg)
            return _run_generation(
                args, cfg, seed, names,
                model_id=model_id, temperature=temperature,
                stage="subset_nova", out_dir=args.results / "subset" / "nova",
                note="One Nova Pro answer per subset question and arm, at temperature 0.",
                repeats=1,
            )
        if args.cmd == "agreement":
            return _agreement(args, cfg, seed)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    raise ProtocolError(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
