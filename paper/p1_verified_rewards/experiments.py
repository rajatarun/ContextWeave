"""Regenerate every number in paper 1 (verified rewards). No AWS, no model.

  python paper/p1_verified_rewards/experiments.py > paper/p1_verified_rewards/results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verified_reward_bench as B  # noqa: E402

BASE = dict(self_bias=0.25, grounding_noise=0.2, judge_rate=0.05, judge_noise=0.1)

out = {"main": B.simulate(40, 2000, overconfidence=0.7, **BASE), "sweep": {}, "noise_sweep": {}}
for oc in (0.0, 0.3, 0.5, 0.7, 0.9):
    out["sweep"][str(oc)] = B.simulate(20, 2000, overconfidence=oc, **BASE)
for gn in (0.1, 0.2, 0.3, 0.4):
    kw = dict(BASE, grounding_noise=gn)
    out["noise_sweep"][str(gn)] = B.simulate(20, 2000, overconfidence=0.7, **kw)
print(json.dumps(out, indent=1))
