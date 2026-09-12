#!/usr/bin/env python3
"""
Offline regret simulation for the adaptive RAG router.

No network, no Memgraph. The deployed selection and update logic in
``src/query_api/rag_router.py`` is exercised in-process against an in-memory
edge store, alongside the rule it replaced and two textbook baselines, so the
question "did the Thompson sampling change actually fix the lock-in, and what
did it cost?" is answered by measurement rather than by argument.

Policies
--------
  legacy               argmax over a scalar weight; +0.05 if confidence >= 0.70,
                       -0.02 if < 0.40, no change otherwise; clipped to [0.1, 1.0].
                       This is the rule that shipped and silently stopped learning.
  greedy_mean          argmax over the Beta posterior *mean* with the new
                       fractional update. Isolates the update-rule fix from the
                       exploration fix: it shows the dead-zone repair alone is
                       not enough.
  thompson             what is deployed: one draw per arm from Beta(alpha, beta),
                       alpha += c, beta += 1 - c.  (rag_router.select_strategy)
  thompson_bernoulli   Agrawal & Goyal (2012): reward r ~ Bernoulli(c), alpha += r,
                       beta += 1 - r. The published regret bound is proved for
                       this variant; included to show the fractional update
                       behaves the same.
  ucb1                 Auer et al. (2002): mean + sqrt(2 ln t / n).

Scenarios are true mean confidences per strategy; each observed confidence is
drawn from Beta(mu*k, (1-mu)*k) with k = 20 so the synthesiser's score varies
from query to query the way it does in production. Priors are the ones seeded
for the "architecture" question type, rescaled to a weak Beta as the router does.

Regret is pseudo-regret: sum over t of (mu_best - mu_{a_t}). The Lai-Robbins
constant  sum_{i != *} Delta_i / KL(mu_i, mu_*)  times ln T is printed as the
asymptotic lower bound any policy must pay; Thompson sampling is known to match
it, so a result near that line is as good as it gets.

Usage
-----
  python scripts/routing_regret_sim.py                # 40 seeds, T = 2000
  python scripts/routing_regret_sim.py --seeds 10 --horizon 500
  python scripts/routing_regret_sim.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass

# Propensity estimation is a per-request cost that has no place in a 2000-step
# loop run across dozens of seeds. Must be set before rag_router is imported.
os.environ.setdefault("ROUTER_PROPENSITY_SAMPLES", "0")
os.environ.setdefault("ROUTER_EXPLORATION", "thompson")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "query_api"))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "shared"))

import rag_router as R  # noqa: E402

ARMS = ["semantic_search", "graph_first", "hybrid", "keyword_boosted"]

# Seeded scalar priors used for the "architecture" question type.
PRIORS = {"semantic_search": 0.60, "graph_first": 0.55,
          "hybrid": 0.50, "keyword_boosted": 0.45}

SCENARIOS: dict[str, dict[str, float]] = {
    # Incumbent answers inside the old [0.40, 0.70) dead zone; never penalised.
    "dead_zone_incumbent": {"semantic_search": 0.55, "graph_first": 0.90,
                            "hybrid": 0.85, "keyword_boosted": 0.50},
    # Incumbent is plainly bad; the old rule *could* demote this one.
    "bad_incumbent":       {"semantic_search": 0.30, "graph_first": 0.90,
                            "hybrid": 0.60, "keyword_boosted": 0.50},
    # Prior favourite is genuinely best; measures the price of exploring.
    "good_incumbent":      {"semantic_search": 0.90, "graph_first": 0.60,
                            "hybrid": 0.60, "keyword_boosted": 0.55},
    # Hard instance: small gaps, regret accrues slowly and steadily.
    "small_gaps":          {"semantic_search": 0.58, "graph_first": 0.62,
                            "hybrid": 0.60, "keyword_boosted": 0.55},
    # Nothing to learn; any exploration is free.
    "all_equal":           {"semantic_search": 0.60, "graph_first": 0.60,
                            "hybrid": 0.60, "keyword_boosted": 0.60},
}

NOISE_K = 20.0  # concentration of observed confidence around its mean; <= 0 → exact


def observe(rng: random.Random, mu: float, noise_k: float = NOISE_K) -> float:
    if noise_k <= 0 or mu <= 0.0 or mu >= 1.0:
        return min(1.0, max(0.0, mu))
    return rng.betavariate(mu * noise_k, (1.0 - mu) * noise_k)


def kl_bernoulli(p: float, q: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    q = min(max(q, 1e-9), 1 - 1e-9)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def _digamma(x: float) -> float:
    # Recurrence up to x >= 6, then the asymptotic series; accurate to ~1e-10.
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1/12 - f * (1/120 - f * (1/252 - f * (1/240 - f / 132))))


def _lbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def kl_beta(a1: float, b1: float, a2: float, b2: float) -> float:
    """KL( Beta(a1,b1) || Beta(a2,b2) )."""
    return (_lbeta(a2, b2) - _lbeta(a1, b1)
            + (a1 - a2) * _digamma(a1) + (b1 - b2) * _digamma(b1)
            + (a2 - a1 + b2 - b1) * _digamma(a1 + b1))


def reward_kl(m1: float, m2: float, noise_k: float) -> float:
    """KL between the reward distributions of an arm with mean m1 and one with mean m2.

    With noise_k <= 0 rewards are deterministic and the KL is infinite for any
    m1 != m2, which makes the lower bound zero: a single pull identifies an arm.
    """
    if noise_k <= 0:
        return math.inf
    return kl_beta(m1 * noise_k, (1 - m1) * noise_k, m2 * noise_k, (1 - m2) * noise_k)


def lai_robbins(mu: dict[str, float], horizon: int, noise_k: float) -> float:
    """sum_{i != *} Delta_i / KL(R_i || R_*) * ln T, for the reward model actually simulated.

    Computed with the KL of the *simulated* reward distributions, not the
    Bernoulli KL. Beta(mu*k, (1-mu)*k) rewards have variance mu(1-mu)/(k+1),
    far below a Bernoulli's mu(1-mu), so arms are easier to tell apart and the
    Bernoulli constant would overstate the floor by roughly a factor of k.
    """
    best = max(mu.values())
    total = 0.0
    for m in mu.values():
        if m < best:
            kl = reward_kl(m, best, noise_k)
            total += 0.0 if math.isinf(kl) else (best - m) / kl
    return total * math.log(horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Policies. Each exposes select() -> arm and update(arm, confidence).
# ─────────────────────────────────────────────────────────────────────────────

class Legacy:
    """The rule that shipped: scalar weight, argmax, fixed steps, dead band."""
    name = "legacy"

    def __init__(self, rng: random.Random):
        self.w = dict(PRIORS)

    def select(self) -> str:
        return max(R._STRATEGY_PRIORITY, key=lambda s: self.w.get(s, 0.0))

    def update(self, arm: str, c: float) -> None:
        if c >= 0.70:
            d = 0.05
        elif c < 0.40:
            d = -0.02
        else:
            return
        self.w[arm] = min(1.0, max(0.1, self.w[arm] + d))


class RouterBacked:
    """Drives the real rag_router against an in-memory edge store."""

    def __init__(self, rng: random.Random, explore: bool):
        self.rng = rng
        self.explore = explore
        p = R._PRIOR_STRENGTH
        self.store = {s: [w * p, (1.0 - w) * p] for s, w in PRIORS.items()}
        R._run_query = self._fake_query  # type: ignore[assignment]
        random.seed(rng.random())

    def _fake_query(self, query, params=None):
        if "RETURN r.label AS strategy" in query:
            return [{"strategy": s, "weight": None, "alpha": a, "beta": b}
                    for s, (a, b) in self.store.items()]
        return []

    def select(self) -> str:
        return R.select_strategy("architecture", explore=self.explore).strategy

    def update(self, arm: str, c: float) -> None:
        # Mirrors the Cypher in update_feedback(): alpha += c, beta += 1 - c.
        a, b = self.store[arm]
        self.store[arm] = [a + c, b + (1.0 - c)]


class Thompson(RouterBacked):
    name = "thompson"

    def __init__(self, rng):
        super().__init__(rng, explore=True)


class GreedyMean(RouterBacked):
    name = "greedy_mean"

    def __init__(self, rng):
        super().__init__(rng, explore=False)


class ThompsonBernoulli(RouterBacked):
    """Agrawal-Goyal Bernoulli-ised reward; the variant the regret bound covers."""
    name = "thompson_bernoulli"

    def __init__(self, rng):
        super().__init__(rng, explore=True)

    def update(self, arm: str, c: float) -> None:
        r = 1.0 if self.rng.random() < c else 0.0
        a, b = self.store[arm]
        self.store[arm] = [a + r, b + (1.0 - r)]


class UCB1:
    name = "ucb1"

    def __init__(self, rng: random.Random):
        self.n = {s: 0 for s in ARMS}
        self.sum = {s: 0.0 for s in ARMS}
        self.t = 0

    def select(self) -> str:
        self.t += 1
        for s in ARMS:
            if self.n[s] == 0:
                return s
        return max(ARMS, key=lambda s: self.sum[s] / self.n[s]
                   + math.sqrt(2.0 * math.log(self.t) / self.n[s]))

    def update(self, arm: str, c: float) -> None:
        self.n[arm] += 1
        self.sum[arm] += c


POLICIES = [Legacy, GreedyMean, Thompson, ThompsonBernoulli, UCB1]


@dataclass
class Run:
    regret: float
    best_frac: float
    picks: dict[str, int]


def simulate(policy_cls, mu: dict[str, float], horizon: int, seed: int,
             noise_k: float = NOISE_K) -> Run:
    rng = random.Random(seed)
    pol = policy_cls(rng)
    best = max(mu.values())
    regret = 0.0
    picks = {s: 0 for s in ARMS}
    for _ in range(horizon):
        arm = pol.select()
        picks[arm] += 1
        regret += best - mu[arm]
        pol.update(arm, observe(rng, mu[arm], noise_k))
    best_arms = {s for s, m in mu.items() if m == best}
    return Run(regret, sum(picks[s] for s in best_arms) / horizon, picks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--noise-k", type=float, default=NOISE_K,
                    help="Beta concentration of observed confidence; 0 = exact "
                         "(reproduces the deterministic figures in the paper)")
    ap.add_argument("--json", default=None, help="write full results here")
    args = ap.parse_args()

    noise = "exact confidences" if args.noise_k <= 0 else f"confidence ~ Beta(mu*k,(1-mu)*k), k={args.noise_k:g}"
    print(f"seeds={args.seeds}  horizon={args.horizon}  rewards: {noise}")
    out: dict[str, dict] = {"config": {"seeds": args.seeds, "horizon": args.horizon,
                                       "noise_k": args.noise_k}}
    for sname, mu in SCENARIOS.items():
        lr = lai_robbins(mu, args.horizon, args.noise_k)
        print(f"\n=== {sname}   true means={mu}")
        print(f"    Lai-Robbins constant x ln T for this reward model: {lr:8.1f}"
              "   (asymptotic floor; finite-T regret can sit below it)")
        print(f"    {'policy':<20} {'regret mean':>12} {'± sd':>8} {'best-arm %':>11}   picks (mean)")
        out[sname] = {"true_means": mu, "lai_robbins": lr, "policies": {}}
        for cls in POLICIES:
            runs = [simulate(cls, mu, args.horizon, seed, args.noise_k)
                    for seed in range(args.seeds)]
            regrets = [r.regret for r in runs]
            fracs = [r.best_frac for r in runs]
            picks = {s: statistics.mean(r.picks[s] for r in runs) for s in ARMS}
            m = statistics.mean(regrets)
            sd = statistics.stdev(regrets) if len(regrets) > 1 else 0.0
            out[sname]["policies"][cls.name] = {
                "regret_mean": m, "regret_sd": sd,
                "best_arm_frac": statistics.mean(fracs), "picks_mean": picks,
            }
            pk = " ".join(f"{s[:8]}={picks[s]:.0f}" for s in ARMS)
            print(f"    {cls.name:<20} {m:12.1f} {sd:8.1f} {100*statistics.mean(fracs):10.1f}%   {pk}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
