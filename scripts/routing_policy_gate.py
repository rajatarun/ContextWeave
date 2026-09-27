#!/usr/bin/env python3
"""
Safe rollout gate for routing policy changes.

``routing_offpolicy_eval.py`` estimates what a candidate policy would have
scored from the decision log. An estimate is not a decision: a point estimate
above the incumbent's says nothing about whether the difference is larger than
the noise, and a candidate that mostly does what the logger rarely did can post
a spectacular SNIPS on eleven effective rows. This script turns the estimate
into a decision and then checks the decision.

  gate    PROMOTE only if the *lower* bound of a paired-bootstrap interval on
          (candidate - incumbent) clears --min-lift, and the candidate's
          effective sample size clears --min-ess. Otherwise HOLD, with the
          reason. The verdict is written as JSON: the predicted value and
          interval travel with it, because they are what the live check tests.

  verify  After the candidate has run live, compare what it actually scored
          against the verdict: did the live mean land inside the predicted
          interval, and did it beat the incumbent's logged value? A gate whose
          predictions miss is miscalibrated and must not be trusted with the
          next change, however good its last promotion looked.

Why paired: the candidate and the incumbent are evaluated on the *same* log
rows, so their estimates are strongly correlated. Bootstrapping each one alone
and comparing intervals ignores that and is far too conservative; resampling
rows once and computing both values on each resample gives the interval of the
difference directly.

The reward is whichever the log carries: the verified ``reward`` column when
present (see ``src/query_api/verified_reward.py``), else the self-confidence.
A gate over self-confidence promotes policies the model likes; the verdict
records which reward it was computed on so the two cannot be confused later.

Usage
-----
  python scripts/routing_policy_gate.py gate decisions.jsonl \\
      --candidate map:architecture=graph_first,default=semantic_search --out verdict.json
  python scripts/routing_policy_gate.py verify verdict.json live.jsonl
  python scripts/routing_policy_gate.py simulate      # end-to-end against known truth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import routing_offpolicy_eval as O  # noqa: E402


@dataclass
class Verdict:
    decision: str                      # PROMOTE | HOLD
    reasons: list[str]
    candidate: str
    reward_source: str
    n: int
    ess: float
    incumbent_value: float
    candidate_value: float
    candidate_ci: tuple[float, float]
    lift: float
    lift_ci: tuple[float, float]
    min_lift: float
    min_ess: float
    confidence: float
    extra: dict[str, Any] = field(default_factory=dict)


def _paired_bootstrap(w: list[float], r: list[float], reps: int, seed: int,
                      level: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Intervals for SNIPS(candidate) and SNIPS(candidate) - mean(logged reward)."""
    rng = random.Random(seed)
    n = len(r)
    vals, diffs = [], []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        ws = [w[i] for i in idx]
        rs = [r[i] for i in idx]
        v = O._snips(ws, rs)
        if math.isnan(v):
            continue
        vals.append(v)
        diffs.append(v - statistics.fmean(rs))
    if not vals:
        nan = (float("nan"), float("nan"))
        return nan, nan
    tail = (1 - level) / 2

    def pct(xs):
        xs = sorted(xs)
        return (xs[int(tail * (len(xs) - 1))], xs[int((1 - tail) * (len(xs) - 1))])

    return pct(vals), pct(diffs)


def gate(records: list[O.Record], candidate: str, *, min_lift: float = 0.0, min_ess: float = 50.0,
         confidence: float = 0.95, clip: float | None = None, reps: int = 2000, seed: int = 0,
         reward_source: str = "unknown") -> Verdict:
    O.check_evaluable(records)
    _, pi = O.parse_policy(candidate)
    if pi is None:
        raise SystemExit("the logging policy cannot be a candidate against itself")
    w = O._weights(records, pi, clip)
    r = [x.reward for x in records]
    sw = sum(w)
    ess = (sw * sw / sum(wi * wi for wi in w)) if sw > 0 else 0.0
    incumbent = statistics.fmean(r)
    value = O._snips(w, r)
    cand_ci, lift_ci = _paired_bootstrap(w, r, reps, seed, confidence)

    reasons = []
    if ess < min_ess:
        reasons.append(f"effective sample size {ess:.1f} < {min_ess}: the log says too little "
                       "about what this candidate would do")
    if math.isnan(lift_ci[0]):
        reasons.append("the candidate never agrees with a logged decision; nothing to estimate")
    elif lift_ci[0] <= min_lift:
        reasons.append(f"lower bound of the lift interval {lift_ci[0]:+.4f} does not clear "
                       f"{min_lift:+.4f} at {confidence:.0%}")
    return Verdict(
        decision="HOLD" if reasons else "PROMOTE",
        reasons=reasons or [f"lift {value - incumbent:+.4f}, {confidence:.0%} interval "
                            f"[{lift_ci[0]:+.4f}, {lift_ci[1]:+.4f}] clears {min_lift:+.4f}; ESS {ess:.1f}"],
        candidate=candidate, reward_source=reward_source, n=len(records), ess=round(ess, 2),
        incumbent_value=round(incumbent, 4), candidate_value=round(value, 4),
        candidate_ci=(round(cand_ci[0], 4), round(cand_ci[1], 4)),
        lift=round(value - incumbent, 4), lift_ci=(round(lift_ci[0], 4), round(lift_ci[1], 4)),
        min_lift=min_lift, min_ess=min_ess, confidence=confidence,
    )


def verify(verdict: dict[str, Any], live: list[O.Record]) -> dict[str, Any]:
    """Did the prediction hold once the candidate ran for real?"""
    if not live:
        raise SystemExit("no live records to verify against")
    rewards = [x.reward for x in live]
    mean = statistics.fmean(rewards)
    se = statistics.pstdev(rewards) / math.sqrt(len(rewards)) if len(rewards) > 1 else float("inf")
    lo, hi = verdict["candidate_ci"]
    # The prediction held if the live mean's own interval overlaps the predicted
    # one; a bare "inside the interval" test would fail a correct prediction on
    # a short live window through the live window's noise alone.
    live_ci = (mean - 1.96 * se, mean + 1.96 * se)
    overlaps = live_ci[0] <= hi and live_ci[1] >= lo
    beat = live_ci[0] > verdict["incumbent_value"]
    status = ("CONFIRMED" if overlaps and beat else
              "MISCALIBRATED" if not overlaps else
              "INCONCLUSIVE")
    return {
        "status": status,
        "liveN": len(live),
        "liveMean": round(mean, 4),
        "liveCi": (round(live_ci[0], 4), round(live_ci[1], 4)),
        "predictedCi": (lo, hi),
        "predictionHeld": overlaps,
        "beatIncumbent": beat,
        "incumbentValue": verdict["incumbent_value"],
        "note": {
            "CONFIRMED": "the live result is inside the predicted range and above the incumbent",
            "MISCALIBRATED": "the live result is outside the predicted range: do not trust this "
                             "gate's next promotion until the cause (reward drift, stale propensities, "
                             "a changed question mix) is found",
            "INCONCLUSIVE": "the prediction held but the live window cannot yet show the lift; "
                            "collect more live traffic before declaring the change a win",
        }[status],
    }


def load_with_reward(paths: list[str]) -> tuple[list[O.Record], str, int]:
    """Load decision rows, preferring the verified ``reward`` column over ``confidence``."""
    records, skipped = [], 0
    sources = set()
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rd = row.get("routingDecision") or {}
                if row.get("reward") is not None:
                    reward, src = row["reward"], "verified"
                elif row.get("confidence") is not None:
                    reward, src = row["confidence"], "self"
                else:
                    skipped += 1
                    continue
                rec = O._record(row.get("questionType"), row.get("strategy") or rd.get("strategy"),
                                row.get("propensity", row.get("selectionPropensity") or rd.get("selectionPropensity")),
                                reward)
                if rec is None:
                    skipped += 1
                    continue
                records.append(rec)
                sources.add(src)
    source = sources.pop() if len(sources) == 1 else ("mixed" if sources else "none")
    return records, source, skipped


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end simulation against known truth
# ─────────────────────────────────────────────────────────────────────────────

def _live_run(truth: dict[str, float], candidate: str, n: int, seed: int,
              question_type: str = "architecture") -> list[O.Record]:
    _, pi = O.parse_policy(candidate)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        weights = [pi(question_type, s) for s in O.STRATEGIES]
        arm = rng.choices(O.STRATEGIES, weights=weights)[0]
        mu = truth[arm]
        out.append(O.Record(question_type, arm, 1.0, rng.betavariate(mu * 20, (1 - mu) * 20)))
    return out


def simulate(n_log: int = 2500, n_live: int = 400, seed: int = 0) -> dict[str, Any]:
    truth = {"graph_first": 0.90, "hybrid": 0.85, "semantic_search": 0.55, "keyword_boosted": 0.50}
    log = O.simulate_log(truth, n_log, seed)
    incumbent_true = statistics.fmean(truth[r.action] for r in log)
    results = {}
    for s in O.STRATEGIES:
        cand = f"fixed:{s}"
        v = gate(log, cand, min_lift=0.0, reward_source="simulated")
        should = truth[s] > incumbent_true
        row = {"truth": truth[s], "incumbentTrue": round(incumbent_true, 4), "verdict": v.decision,
               "correctDecision": (v.decision == "PROMOTE") == should,
               "lift": v.lift, "liftCi": v.lift_ci, "ess": v.ess}
        if v.decision == "PROMOTE":
            row["verify"] = verify(asdict(v), _live_run(truth, cand, n_live, seed + 1))["status"]
        results[cand] = row
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("logs", nargs="+")
    g.add_argument("--candidate", required=True)
    g.add_argument("--min-lift", type=float, default=0.0)
    g.add_argument("--min-ess", type=float, default=50.0)
    g.add_argument("--confidence", type=float, default=0.95)
    g.add_argument("--clip", type=float, default=None)
    g.add_argument("--reps", type=int, default=2000)
    g.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("verdict")
    v.add_argument("live", nargs="+")
    sub.add_parser("simulate")
    args = ap.parse_args(argv)

    if args.cmd == "gate":
        records, source, skipped = load_with_reward(args.logs)
        if skipped:
            print(f"skipped {skipped} rows with no propensity or reward")
        if source == "mixed":
            print("WARNING: the log mixes verified rewards and self-confidences; "
                  "the estimate averages two different quantities")
        verdict = gate(records, args.candidate, min_lift=args.min_lift, min_ess=args.min_ess,
                       confidence=args.confidence, clip=args.clip, reps=args.reps, reward_source=source)
        print(json.dumps(asdict(verdict), indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(asdict(verdict), indent=2))
        return 0 if verdict.decision == "PROMOTE" else 3
    if args.cmd == "verify":
        verdict = json.loads(Path(args.verdict).read_text())
        live, _, _ = load_with_reward(args.live)
        result = verify(verdict, live)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "CONFIRMED" else 3
    print(json.dumps(simulate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
