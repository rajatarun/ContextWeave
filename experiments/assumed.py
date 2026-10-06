"""Slopes assumed by the verified-reward simulation, derived from its code.

``scripts/verified_reward_bench.py`` draws self-confidence from
``N(0.88, 0.06)`` on a correct answer and, with probability ``--overconfidence``
(default 0.7), on a wrong answer too; otherwise from ``N(0.55 + self_bias, 0.12)``
with ``--self-bias`` default 0.25, so the low mean is 0.80. Grounding is
``N(0.8, sigma)`` when correct and ``N(0.3, sigma)`` otherwise. Values are then
clipped to [0, 1]. The slopes below are the pre-clip means. Clipping moves the
realised means by a small amount that this file does not estimate.

``scripts/routing_regret_sim.py`` draws a reward from ``Beta(mu * 20, (1 - mu) * 20)``
around each arm's true mean. It does not define an overconfidence slope.
"""
from __future__ import annotations

from typing import Any

# Locked to verified_reward_bench.draw_rewards and its CLI defaults.
RHO = 0.7
SELF_C1 = 0.88
SELF_LOW_MEAN = 0.55 + 0.25
GROUND_C1 = 0.8
GROUND_C0 = 0.3
REGRET_SIM_NOISE_K = 20.0


def assumed_slopes() -> dict[str, Any]:
    c0_self = RHO * SELF_C1 + (1.0 - RHO) * SELF_LOW_MEAN
    s_self = SELF_C1 - c0_self
    return {
        "source": "scripts/verified_reward_bench.py draw_rewards, CLI defaults",
        "clip": "means are before the [0, 1] clip applied by draw_rewards",
        "rho": RHO,
        "self": {
            "c1": SELF_C1,
            "c0_prime": SELF_LOW_MEAN,
            "c0": c0_self,
            "s": s_self,
        },
        "grounding": {
            "c1": GROUND_C1,
            "c0": GROUND_C0,
            "s": GROUND_C1 - GROUND_C0,
        },
        "routing_regret_sim": {
            "source": "scripts/routing_regret_sim.py",
            "reward": "Beta(mu * noise_k, (1 - mu) * noise_k)",
            "noise_k": REGRET_SIM_NOISE_K,
            "s_self": None,
            "s_ground": None,
            "note": "That simulation has no overconfidence slope. The slopes above are the verified-reward simulation.",
        },
    }
