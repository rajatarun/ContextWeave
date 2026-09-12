#!/usr/bin/env python3
"""
Per-request cost of Thompson-sampling selection, in isolation.

Exploration's cost in *answer quality* is measured by routing_regret_sim.py.
This measures its cost in *latency*: the four Beta draws, the argmax, and
the Monte Carlo propensity estimate that select_strategy() now performs on
every request, against the greedy posterior-mean lookup it replaced. The
graph round-trip is stubbed out (it is identical for both policies and is
the dominant cost either way); what is left is the pure selection overhead.

Usage
-----
  python scripts/bench_router.py                   # 20000 iterations
  python scripts/bench_router.py --iters 5000 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "query_api"))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "shared"))

import rag_router as R  # noqa: E402

STORE = {"semantic_search": [2.4, 1.6], "graph_first": [2.2, 1.8],
         "hybrid": [2.0, 2.0], "keyword_boosted": [1.8, 2.2]}


def _fake_query(query, params=None):
    if "RETURN r.label AS strategy" in query:
        return [{"strategy": s, "weight": None, "alpha": a, "beta": b} for s, (a, b) in STORE.items()]
    return []


def _time(fn, iters):
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    xs.sort()
    return {"median_ms": statistics.median(xs),
            "p99_ms": xs[min(len(xs) - 1, int(0.99 * len(xs)))],
            "max_ms": xs[-1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    R._run_query = _fake_query  # type: ignore[assignment]
    out = {}

    def row(name, fn, iters=args.iters):
        out[name] = _time(fn, iters)
        r = out[name]
        print(f"{name:<52}{r['median_ms']:9.4f}{r['p99_ms']:9.4f}{r['max_ms']:9.4f}")

    print(f"iters={args.iters}\n{'selection variant':<52}{'median':>9}{'p99':>9}{'max':>9}   (ms)")
    saved = R._PROPENSITY_SAMPLES

    R._PROPENSITY_SAMPLES = 0
    row("greedy (posterior mean argmax)", lambda: R.select_strategy("architecture", explore=False))
    row("thompson, no propensity estimate", lambda: R.select_strategy("architecture", explore=True))
    for n in (100, 500, 2000):
        R._PROPENSITY_SAMPLES = n
        row(f"thompson + propensity ({n} MC draws)",
            lambda: R.select_strategy("architecture", explore=True),
            iters=max(2000, args.iters // (n // 100)))
    R._PROPENSITY_SAMPLES = saved

    post = R._query_strategy_posteriors("architecture")
    row("routing_health(), one question type, 2000 draws",
        lambda: R.routing_health(["architecture"], samples=2000), iters=200)

    print("\nFor scale: one Bedrock synthesis call is O(10^3) ms, one pgvector query O(10) ms, "
          "one Memgraph round-trip O(1-10) ms.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
