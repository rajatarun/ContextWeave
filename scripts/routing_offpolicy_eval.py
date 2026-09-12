#!/usr/bin/env python3
"""
Off-policy evaluation of a candidate routing policy from the decision log.

The router only ever observes the strategy it selected, so the value of a
strategy it did *not* select on a given query is never seen directly. Because
each decision now records the probability the sampler had of making it
(``routingDecision.selectionPropensity``), the log can be reweighted to
estimate what a different policy would have scored, without deploying it.

Estimators (Horvitz-Thompson / Swaminathan & Joachims):

  IPW    1/n * sum_i  w_i r_i          w_i = pi(a_i | x_i) / p_i
  SNIPS  sum_i w_i r_i / sum_i w_i     self-normalised; biased but far lower variance
  DM     sum_x P(x) sum_a pi(a|x) mu(a,x)   direct method from per-(type, arm) mean reward

where x_i is the question type, a_i the strategy that ran, p_i its logged
propensity and r_i the synthesis confidence. 95% intervals are bootstrap
percentiles. The effective sample size (sum w)^2 / sum w^2 says how many
log rows actually inform the estimate; a small ESS means the candidate mostly
does things the logging policy rarely did, and the interval will say so.

Two things this cannot do. A log written under ``ROUTER_EXPLORATION=greedy``
has every propensity at 1.0 and is refused: there is no counterfactual
information in it. And the reward is still the synthesiser's self-confidence;
this estimates what a policy would have scored on *that*, not on relevance.

Inputs
------
  experiment_results_*.json     written by scripts/routing_experiment.py
  *.jsonl                       one record per line with keys
                                questionType, strategy, selectionPropensity, confidence

Candidate policies (--policy, repeatable)
------
  fixed:<strategy>                     always that strategy
  map:<type>=<strategy>,...[,default=<strategy>]
  uniform                              1/K over the four strategies
  logging                              value of the logging policy itself (sanity)

Usage
-----
  python scripts/routing_offpolicy_eval.py experiment_results_*.json \
      --policy fixed:graph_first --policy map:architecture=graph_first,default=semantic_search

  # Validate the estimator against known truth using the deployed router in-process:
  python scripts/routing_offpolicy_eval.py --simulate
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

STRATEGIES = ["graph_first", "hybrid", "keyword_boosted", "semantic_search"]

Policy = Callable[[str, str], float]  # (question_type, strategy) -> pi(strategy | type)


@dataclass
class Record:
    question_type: str
    action: str
    propensity: float
    reward: float


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _record(qt, action, prop, reward) -> Record | None:
    if not qt or not action or reward is None or prop is None:
        return None
    return Record(str(qt), str(action), float(prop), min(1.0, max(0.0, float(reward))))


def load_records(paths: list[str]) -> tuple[list[Record], int]:
    """Returns (records, rows_skipped_for_missing_fields)."""
    records: list[Record] = []
    skipped = 0
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path) as f:
                if path.endswith(".jsonl"):
                    rows = [json.loads(line) for line in f if line.strip()]
                    for r in rows:
                        rd = r.get("routingDecision") or {}
                        rec = _record(r.get("questionType"),
                                      r.get("strategy") or rd.get("strategy"),
                                      r.get("selectionPropensity") or rd.get("selectionPropensity"),
                                      r.get("confidence"))
                        records.append(rec) if rec else None
                        skipped += rec is None
                else:
                    data = json.load(f)
                    for r in data.get("results", []):
                        if r.get("status") != "ok":
                            continue
                        rec = _record(r.get("classified_type") or r.get("declared_type"),
                                      r.get("routing_decision"),
                                      r.get("selection_propensity"),
                                      r.get("confidence"))
                        records.append(rec) if rec else None
                        skipped += rec is None
    return records, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Policies
# ─────────────────────────────────────────────────────────────────────────────

def parse_policy(spec: str) -> tuple[str, Policy]:
    if spec == "uniform":
        return spec, lambda qt, a: 1.0 / len(STRATEGIES)
    if spec == "logging":
        return spec, None  # type: ignore[return-value]
    kind, _, body = spec.partition(":")
    if kind == "fixed":
        if body not in STRATEGIES:
            raise SystemExit(f"unknown strategy {body!r}; choose from {STRATEGIES}")
        return spec, lambda qt, a, b=body: 1.0 if a == b else 0.0
    if kind == "map":
        table: dict[str, str] = {}
        for pair in body.split(","):
            k, _, v = pair.partition("=")
            if v not in STRATEGIES:
                raise SystemExit(f"unknown strategy {v!r} in {spec!r}")
            table[k.strip()] = v.strip()
        default = table.pop("default", None)

        def pi(qt, a, table=table, default=default):
            target = table.get(qt, default)
            if target is None:
                return float("nan")  # policy undefined for this type
            return 1.0 if a == target else 0.0
        return spec, pi
    raise SystemExit(f"unrecognised policy spec {spec!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Estimators
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Estimate:
    n: int
    ipw: float
    ipw_ci: tuple[float, float]
    snips: float
    snips_ci: tuple[float, float]
    dm: float | None
    ess: float
    max_weight: float
    matched: int          # rows where pi(a_i|x_i) > 0


def _weights(records: list[Record], pi: Policy, clip: float | None) -> list[float]:
    w = []
    for r in records:
        p = pi(r.question_type, r.action)
        if math.isnan(p):
            w.append(0.0)
            continue
        wi = p / r.propensity
        if clip is not None:
            wi = min(wi, clip)
        w.append(wi)
    return w


def _ipw(w, r):
    return sum(wi * ri for wi, ri in zip(w, r)) / len(w)


def _snips(w, r):
    denom = sum(w)
    return sum(wi * ri for wi, ri in zip(w, r)) / denom if denom > 0 else float("nan")


def _bootstrap(fn, w, r, rng: random.Random, reps: int) -> tuple[float, float]:
    n = len(w)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(fn([w[i] for i in idx], [r[i] for i in idx]))
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return (float("nan"), float("nan"))
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return (lo, hi)


def direct_method(records: list[Record], pi: Policy) -> float | None:
    """sum_x P(x) sum_a pi(a|x) mu(a,x); None if the policy needs an unseen (type, arm)."""
    by_type: dict[str, dict[str, list[float]]] = {}
    for r in records:
        by_type.setdefault(r.question_type, {}).setdefault(r.action, []).append(r.reward)
    total = 0.0
    n = len(records)
    for qt, arms in by_type.items():
        px = sum(len(v) for v in arms.values()) / n
        for a in STRATEGIES:
            p = pi(qt, a)
            if math.isnan(p) or p == 0.0:
                continue
            if a not in arms:
                return None  # no observation of this arm for this type
            total += px * p * statistics.mean(arms[a])
    return total


def evaluate(records: list[Record], pi: Policy | None, clip: float | None = None,
             reps: int = 1000, seed: int = 0) -> Estimate:
    r = [x.reward for x in records]
    if pi is None:  # logging policy: every weight is 1
        w = [1.0] * len(records)
    else:
        w = _weights(records, pi, clip)
    rng = random.Random(seed)
    sw = sum(w)
    ess = (sw * sw / sum(wi * wi for wi in w)) if sw > 0 else 0.0
    return Estimate(
        n=len(records),
        ipw=_ipw(w, r), ipw_ci=_bootstrap(_ipw, w, r, rng, reps),
        snips=_snips(w, r), snips_ci=_bootstrap(_snips, w, r, rng, reps),
        dm=(statistics.mean(r) if pi is None else direct_method(records, pi)),
        ess=ess, max_weight=max(w) if w else 0.0,
        matched=sum(1 for wi in w if wi > 0),
    )


def check_evaluable(records: list[Record]) -> None:
    if not records:
        raise SystemExit("no usable records (need questionType, strategy, selectionPropensity, confidence)")
    if all(r.propensity >= 1.0 for r in records):
        raise SystemExit(
            "every logged propensity is 1.0: this log was written under greedy selection "
            "and carries no counterfactual information. Nothing can be estimated from it."
        )
    bad = [r for r in records if not (0.0 < r.propensity <= 1.0)]
    if bad:
        raise SystemExit(f"{len(bad)} records have propensity outside (0, 1]")


# ─────────────────────────────────────────────────────────────────────────────
# Validation against known truth, using the deployed router
# ─────────────────────────────────────────────────────────────────────────────

def simulate_log(true_means: dict[str, float], n: int, seed: int,
                 question_type: str = "architecture", noise_k: float = 20.0) -> list[Record]:
    """Run the real Thompson router for n queries and return its decision log."""
    here = os.path.dirname(os.path.abspath(__file__))
    for sub in ("query_api", "shared"):
        p = os.path.join(here, "..", "src", sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import rag_router as R  # noqa: E402

    rng = random.Random(seed)
    random.seed(seed)
    prior = R._PRIOR_STRENGTH
    store = {s: [0.5 * prior, 0.5 * prior] for s in STRATEGIES}

    def fake_query(query, params=None):
        if "RETURN r.label AS strategy" in query:
            return [{"strategy": s, "weight": None, "alpha": a, "beta": b}
                    for s, (a, b) in store.items()]
        return []

    original = R._run_query
    R._run_query = fake_query  # type: ignore[assignment]
    try:
        log: list[Record] = []
        for _ in range(n):
            cfg = R.select_strategy(question_type, explore=True)
            mu = true_means[cfg.strategy]
            c = mu if noise_k <= 0 else rng.betavariate(mu * noise_k, (1 - mu) * noise_k)
            a, b = store[cfg.strategy]
            store[cfg.strategy] = [a + c, b + (1 - c)]
            log.append(Record(question_type, cfg.strategy, cfg.selection_propensity, c))
    finally:
        R._run_query = original  # type: ignore[assignment]
    return log


def run_simulation(n: int, seed: int) -> None:
    truth = {"graph_first": 0.90, "hybrid": 0.85, "semantic_search": 0.55, "keyword_boosted": 0.50}
    print(f"Simulated log: n={n}, Thompson router in-process, true means={truth}")
    log = simulate_log(truth, n, seed)
    check_evaluable(log)
    picks = {s: sum(1 for r in log if r.action == s) for s in STRATEGIES}
    print(f"logging policy picks: {picks}   mean reward={statistics.mean(r.reward for r in log):.3f}")
    print(f"\n{'candidate':<24}{'truth':>7}{'IPW':>8}{'95% CI':>17}{'SNIPS':>8}{'95% CI':>17}{'DM':>7}{'ESS':>8}{'rows':>6}")
    worst = 0.0
    for s in STRATEGIES:
        _, pi = parse_policy(f"fixed:{s}")
        e = evaluate(log, pi, reps=400, seed=seed)
        dm = f"{e.dm:7.3f}" if e.dm is not None else "    n/a"
        print(f"fixed:{s:<18}{truth[s]:7.3f}{e.ipw:8.3f}  [{e.ipw_ci[0]:.3f}, {e.ipw_ci[1]:.3f}]"
              f"{e.snips:8.3f}  [{e.snips_ci[0]:.3f}, {e.snips_ci[1]:.3f}]{dm}{e.ess:8.1f}{e.matched:6d}")
        worst = max(worst, abs(e.snips - truth[s]))
    print(f"\nlargest |SNIPS - truth| across arms: {worst:.3f}")
    print("Rarely-selected arms have small ESS and wide intervals; that is the estimator being "
          "honest about how little the log says about them, not a defect.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="*", help="experiment_results_*.json and/or *.jsonl")
    ap.add_argument("--policy", action="append", default=[],
                    help="candidate policy spec (repeatable); see module docstring")
    ap.add_argument("--clip", type=float, default=None, help="cap importance weights at this value")
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--simulate", action="store_true",
                    help="validate the estimator against known truth with the deployed router")
    ap.add_argument("--n", type=int, default=3000, help="rows for --simulate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.simulate:
        run_simulation(args.n, args.seed)
        return
    if not args.logs:
        ap.error("give one or more log files, or --simulate")

    records, skipped = load_records(args.logs)
    if skipped:
        print(f"skipped {skipped} rows lacking a propensity or reward "
              "(collected before propensity logging was added?)")
    check_evaluable(records)

    by_type: dict[str, int] = {}
    for r in records:
        by_type[r.question_type] = by_type.get(r.question_type, 0) + 1
    print(f"{len(records)} decisions; question types: {by_type}")
    print(f"mean logged propensity {statistics.mean(r.propensity for r in records):.3f}, "
          f"min {min(r.propensity for r in records):.3f}")

    specs = args.policy or ["logging"] + [f"fixed:{s}" for s in STRATEGIES]
    print(f"\n{'policy':<44}{'IPW':>8}{'95% CI':>17}{'SNIPS':>8}{'95% CI':>17}{'DM':>7}{'ESS':>8}{'rows':>6}{'max w':>7}")
    for spec in specs:
        name, pi = parse_policy(spec)
        e = evaluate(records, pi, clip=args.clip, reps=args.reps, seed=args.seed)
        dm = f"{e.dm:7.3f}" if e.dm is not None else "    n/a"
        print(f"{name:<44}{e.ipw:8.3f}  [{e.ipw_ci[0]:.3f}, {e.ipw_ci[1]:.3f}]"
              f"{e.snips:8.3f}  [{e.snips_ci[0]:.3f}, {e.snips_ci[1]:.3f}]{dm}{e.ess:8.1f}{e.matched:6d}{e.max_weight:7.1f}")
    print("\nRead SNIPS first; IPW is unbiased but its interval widens fast when the candidate "
          "prefers arms the logger rarely chose. DM is n/a when the candidate needs a (type, arm) "
          "pair the log never observed.")


if __name__ == "__main__":
    main()
