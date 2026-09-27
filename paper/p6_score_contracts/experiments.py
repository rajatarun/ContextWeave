"""Regenerate every number in paper 6 (score contracts).

  python paper/p6_score_contracts/experiments.py [--repos ../DeviceWeave ../CipherWeave ../mcp-observatory]

1. Census: every declared score in each repository's contracts/scores.json, by
   kind, source and calibration status, and every pair can_combine would allow.
2. Rank reversal: averaging a similarity with a heuristic score is not invariant
   under a monotone rescaling of either -- the ranking of two candidates flips
   while each score's own ordering is unchanged (Proposition 1, instantiated).
3. Calibration: an overconfident score fed through calibration_report and the
   isotonic fit, before and after.
"""
import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from contracts import score_contract as SC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repos", nargs="*", default=[str(ROOT.parent / r) for r in ("DeviceWeave", "CipherWeave", "mcp-observatory")])
args = ap.parse_args()

entries = []
for repo in [str(ROOT)] + args.repos:
    p = Path(repo) / "contracts" / "scores.json"
    if p.exists():
        for e in SC.load_registry(p):
            entries.append({**e, "_repo": Path(repo).name})
out = {"census": {
    "scores": len(entries),
    "by_repo": dict(Counter(e["_repo"] for e in entries)),
    "by_kind": dict(Counter(e["kind"] for e in entries)),
    "by_source": dict(Counter(e["source"] for e in entries)),
    "calibrated": [e["name"] for e in entries if e.get("calibrated")],
    "registry_problems": SC.check_registry(entries, resolve_producers=False),
}}
envs = {e["name"]: SC.envelope(0.5, e, evidence=10) for e in entries}
allowed = {op: [] for op in ("average", "multiply", "max")}
for a, b in itertools.combinations(envs, 2):
    for op in allowed:
        if SC.can_combine(envs[a], envs[b], op)[0]:
            allowed[op].append([a, b])
out["census"]["cross_pairs"] = len(list(itertools.combinations(envs, 2)))
out["census"]["allowed_pairs"] = {op: v for op, v in allowed.items()}

# 2. Rank reversal under a monotone rescaling of the similarity.
A = {"cos": 0.90, "beh": 0.10}
B = {"cos": 0.60, "beh": 0.50}
alpha = 0.5
f = lambda c: c ** 3  # monotone on [0,1]: same ordering of every cosine  # noqa: E731
avg = lambda s, g=lambda x: x: alpha * g(s["cos"]) + (1 - alpha) * s["beh"]  # noqa: E731
out["rank_reversal"] = {
    "A": A, "B": B, "alpha": alpha,
    "raw": {"A": round(avg(A), 4), "B": round(avg(B), 4)},
    "cubed_cosine": {"A": round(avg(A, f), 4), "B": round(avg(B, f), 4)},
}

# 3. Calibration of an overconfident score.
rng = random.Random(0)
pairs = []
for _ in range(5000):
    correct = rng.random() < 0.6
    s = min(1.0, max(0.0, rng.gauss(0.9 if correct else 0.8, 0.05)))
    pairs.append((s, 1.0 if correct else 0.0))
train, test = pairs[:2500], pairs[2500:]
fit = SC.isotonic_fit(train)
before = SC.calibration_report(test)
after = SC.calibration_report([(SC.apply_isotonic(fit, s), o) for s, o in test])
out["calibration"] = {"before": {k: before[k] for k in ("n", "brier", "ece")},
                      "after_isotonic": {k: after[k] for k in ("n", "brier", "ece")},
                      "fit_breakpoints": len(fit)}
print(json.dumps(out, indent=1))
