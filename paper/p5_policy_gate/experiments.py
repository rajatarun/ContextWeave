"""Regenerate every number in paper 5 (offline gate). No AWS.

For each (log size, seed): run the deployed Thompson router against known truth,
gate every fixed-arm candidate, and for each PROMOTE run the candidate "live"
and verify. Also: gate decisions from separate (unpaired) intervals, to measure
what pairing buys.
"""
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import routing_offpolicy_eval as O  # noqa: E402
import routing_policy_gate as G  # noqa: E402

TRUTHS = {
    "separated": {"graph_first": 0.90, "hybrid": 0.85, "semantic_search": 0.55, "keyword_boosted": 0.50},
    "close":     {"graph_first": 0.78, "hybrid": 0.75, "semantic_search": 0.72, "keyword_boosted": 0.70},
}


def unpaired_promote(log, cand, reps=400, seed=0):
    _, pi = O.parse_policy(cand)
    e = O.evaluate(log, pi, reps=reps, seed=seed)
    inc = O.evaluate(log, None, reps=reps, seed=seed)
    return e.snips_ci[0] > inc.snips_ci[1]


out = {}
for tname, truth in TRUTHS.items():
    for n in (500, 1500, 3000):
        rows = []
        for seed in range(12):
            log = O.simulate_log(truth, n, seed)
            inc_true = statistics.fmean(truth[r.action] for r in log)
            for arm in O.STRATEGIES:
                cand = f"fixed:{arm}"
                v = G.gate(log, cand, reps=400, seed=seed, reward_source="simulated")
                should = truth[arm] > inc_true
                row = {"seed": seed, "arm": arm, "should": should, "decision": v.decision,
                       "lift": v.lift, "true_lift": round(truth[arm] - inc_true, 4),
                       "lift_ci": v.lift_ci, "ess": v.ess,
                       "unpaired_promote": unpaired_promote(log, cand, seed=seed)}
                if v.decision == "PROMOTE":
                    row["verify"] = G.verify(asdict(v), G._live_run(truth, cand, 400, seed + 101))["status"]
                row["covers"] = v.lift_ci[0] <= row["true_lift"] <= v.lift_ci[1]
                rows.append(row)
        promoted = [r for r in rows if r["decision"] == "PROMOTE"]
        out[f"{tname}/n={n}"] = {
            "candidates": len(rows),
            "should_promote": sum(r["should"] for r in rows),
            "promoted": len(promoted),
            "false_promotions": sum(1 for r in promoted if not r["should"]),
            "missed": sum(1 for r in rows if r["should"] and r["decision"] != "PROMOTE"),
            "unpaired_promoted": sum(r["unpaired_promote"] for r in rows),
            "unpaired_false": sum(1 for r in rows if r["unpaired_promote"] and not r["should"]),
            "verify": {s: sum(1 for r in promoted if r.get("verify") == s)
                       for s in ("CONFIRMED", "INCONCLUSIVE", "MISCALIBRATED")},
            "lift_ci_coverage": round(sum(r["covers"] for r in rows) / len(rows), 3),
        }
print(json.dumps(out, indent=1))
