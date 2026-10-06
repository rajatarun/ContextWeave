#!/usr/bin/env python3
"""Replay Thompson sampling on logged rewards. Writes a summary, per-seed curves, and PNGs."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (
    DATASETS, RESULTS, ProtocolError, artifact_meta, load_config, parse_seed_list, write_json,
)
from experiments.records import load_joined
from experiments.replay_stage import REWARDS, UPDATES, empirical_mu, general_priors, run_one


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _plot(path: Path, curves: dict[str, list[list[float]]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ProtocolError("matplotlib is not installed; pip install -r experiments/requirements.txt") from exc
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, series in curves.items():
        length = min(len(s) for s in series)
        xs = list(range(1, length + 1))
        ys = [_mean([s[i] for s in series]) for i in range(length)]
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("questions seen")
    ax.set_ylabel("cumulative pseudo-regret")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, default from config")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seeds = parse_seed_list(args.seeds) if args.seeds else [int(s) for s in cfg["replay_seeds"]]
    names = [p.strip() for p in args.datasets.split(",") if p.strip()]
    try:
        for name in names:
            for fname in ("lexical.jsonl", "judge.jsonl"):
                path = args.results / "signals" / fname
                if not path.is_file():
                    raise ProtocolError(f"replay needs {path}. Run the signals stage first.")
            if not (args.results / "generation" / f"{name}.jsonl").is_file():
                raise ProtocolError(f"generation output missing for {name}")
        joined = load_joined(args.results, cfg, names)
        import rag_router as R
        priors = general_priors()
        strength = float(R._PRIOR_STRENGTH)
        out_dir = args.results / "replay"
        summary: dict = {"datasets": {}}
        for name in names:
            group = [r for r in joined if r["dataset"] == name]
            questions = []
            seen = set()
            by_qid: dict[str, dict[str, dict]] = defaultdict(dict)
            for row in group:
                by_qid[row["qid"]][row["arm"]] = row
                if row["qid"] not in seen:
                    seen.add(row["qid"])
                    questions.append({"qid": row["qid"], "question_type": row["question_type"]})
            mu = empirical_mu(group)
            summary["datasets"][name] = {}
            for update in UPDATES:
                curves: dict[str, list[list[float]]] = defaultdict(list)
                csv_path = out_dir / f"{name}_{update}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=[
                        "dataset", "reward", "update", "seed", "t",
                        "cumulative_pseudo_regret", "cumulative_realized_regret",
                        "best_arm_share", "selected_arm", "applied_reward",
                    ])
                    writer.writeheader()
                    for reward in REWARDS:
                        pseudos, realizeds, shares = [], [], []
                        for seed in seeds:
                            run = run_one(questions, by_qid, reward, update, seed, mu, priors, strength)
                            pseudos.append(run["pseudo_regret"])
                            realizeds.append(run["realized_regret"])
                            shares.append(run["best_arm_share"])
                            series = []
                            for point in run["curve"]:
                                writer.writerow({
                                    "dataset": name, "reward": reward, "update": update, "seed": seed,
                                    **point,
                                })
                                series.append(point["cumulative_pseudo_regret"])
                            curves[reward].append(series)
                        summary["datasets"][name].setdefault(reward, {})[update] = {
                            "pseudo_regret_mean": _mean(pseudos),
                            "realized_regret_mean": _mean(realizeds),
                            "best_arm_share_mean": _mean(shares),
                            "seeds": seeds,
                        }
                png = out_dir / f"{name}_{update}.png"
                _plot(png, curves)
                print(f"wrote {csv_path} and {png}", flush=True)
        body = artifact_meta(cfg, seeds, stage="replay", priors=priors, prior_strength=strength, **summary)
        write_json(out_dir / "replay_summary.json", body)
        print(f"wrote {out_dir / 'replay_summary.json'}")
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
