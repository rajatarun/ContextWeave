#!/usr/bin/env python3
"""
Can the router learn without trusting the model's opinion of itself?

Offline, no AWS. Two subcommands answer the two halves of that question.

calibrate  -- how far does each reward signal sit from real correctness?
-----------------------------------------------------------------------
Input is JSONL, one answered query per line, from any public QA benchmark
where the gold answer is known (SQuAD v2, Natural Questions, HotpotQA, ...):

  {"id": "...", "question": "...", "strategy": "semantic_search",
   "passages": ["...", "..."], "answer": "...",
   "self_confidence": 0.9,            # null when the model did not report one
   "gold_answers": ["...", "..."],    # [] means unanswerable (SQuAD v2 style)
   "judge": 0.8}                      # optional: a judge model's score

Correctness is token F1 against the best gold answer (the SQuAD metric); an
unanswerable question is correct iff the answer abstains. For every signal --
``self``, ``grounding`` (verified_reward.grounding_signal, with the lexical
verifier or ``--verifier nli``), ``judge`` if present -- it reports:

  coverage   fraction of queries where the signal was observed at all
  brier      mean (signal - correct)^2 over observed queries
  ece        expected calibration error, 10 equal-width bins
  auroc      P(signal ranks a correct answer above an incorrect one)
  spearman   rank correlation with F1
  rank_agree whether ranking strategies by the signal's mean reproduces the
             ranking by mean correctness -- the only property a router needs

The last one is the point. A signal can be badly calibrated and still be a fine
reward if it orders strategies the way correctness does; a well-calibrated
signal that cannot tell strategies apart teaches the router nothing.

simulate  -- what does a miscalibrated reward cost the router?
--------------------------------------------------------------
Thompson sampling with the deployed fractional update (alpha += r,
beta += 1 - r), four strategies with true correctness rates, and three
rewards drawn per query:

  self       overconfident: high on correct answers and, with probability
             --overconfidence, high on wrong ones too (the failure mode)
  grounding  a noisy but unbiased view of correctness (--grounding-noise)
  verified   grounding, plus a judge on --judge-rate of queries (weight 2)

Regret is measured against *correctness*, not against the reward the router
saw, so a router that learns the wrong thing confidently shows up as regret.
With ``--self-bias`` a strategy that retrieves off-topic passages still gets
a confident synthesiser -- the concrete way self-confidence misleads a router.

Usage
-----
  python scripts/verified_reward_bench.py calibrate --input runs.jsonl
  python scripts/verified_reward_bench.py calibrate --input runs.jsonl --verifier nli
  python scripts/verified_reward_bench.py simulate --seeds 40 --horizon 2000
  python scripts/verified_reward_bench.py simulate --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src" / "query_api"))

import verified_reward as V  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Correctness (SQuAD-style)
# ─────────────────────────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(_ARTICLES.sub(" ", s).split())


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    return bool(V._ABSTAIN_RE.search(answer or "")) or not normalize_answer(answer)


def correctness(answer: str, gold_answers: list[str]) -> float:
    """Best token F1 against any gold answer; abstention scored on unanswerables."""
    if not gold_answers:
        return 1.0 if is_abstention(answer) else 0.0
    if is_abstention(answer):
        return 0.0
    return max(token_f1(answer, g) for g in gold_answers)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def brier(pairs: list[tuple[float, float]]) -> float | None:
    return statistics.fmean((s - c) ** 2 for s, c in pairs) if pairs else None


def ece(pairs: list[tuple[float, float]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for s, c in pairs:
        buckets[min(bins - 1, int(s * bins))].append((s, c))
    n = len(pairs)
    return sum(len(b) / n * abs(statistics.fmean(x for x, _ in b) - statistics.fmean(y for _, y in b))
               for b in buckets.values())


def auroc(pairs: list[tuple[float, float]], threshold: float = 0.5) -> float | None:
    """Mann-Whitney AUROC of the signal for binarised correctness (F1 >= threshold)."""
    pos = [s for s, c in pairs if c >= threshold]
    neg = [s for s, c in pairs if c < threshold]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    a, b = _ranks([s for s, _ in pairs]), _ranks([c for _, c in pairs])
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else None


# ─────────────────────────────────────────────────────────────────────────────
# calibrate
# ─────────────────────────────────────────────────────────────────────────────

def nli_verifier(model_name: str = "cross-encoder/nli-deberta-v3-small") -> V.Verifier:
    """P(entailment) from a local NLI cross-encoder. Needs sentence-transformers."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("--verifier nli needs `pip install sentence-transformers`") from exc
    model = CrossEncoder(model_name)
    labels = [l.lower() for l in getattr(model.config, "id2label", {}).values()] or [
        "contradiction", "entailment", "neutral"]
    ent = labels.index("entailment") if "entailment" in labels else 1

    def verify(claim: str, passage: str) -> float:
        logits = model.predict([(passage, claim)], apply_softmax=True)[0]
        return float(logits[ent])

    return verify


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def calibrate(records: Iterable[dict[str, Any]], verifier: V.Verifier = V.lexical_support,
              threshold: float | None = None) -> dict[str, Any]:
    rows = []
    for r in records:
        g = V.grounding_signal(r.get("answer", ""), r.get("passages", []), verifier=verifier,
                               threshold=threshold)
        s = V.self_signal(r.get("self_confidence"), r.get("self_confidence") is not None)
        rows.append({
            "strategy": r.get("strategy", "unknown"),
            "correct": correctness(r.get("answer", ""), r.get("gold_answers", [])),
            "self": s.value,
            "grounding": g.value,
            "judge": r.get("judge"),
        })

    out: dict[str, Any] = {"n": len(rows), "meanCorrect": statistics.fmean(r["correct"] for r in rows) if rows else None,
                           "signals": {}}
    by_strategy_correct: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_strategy_correct[r["strategy"]].append(r["correct"])
    truth_order = sorted(by_strategy_correct, key=lambda k: -statistics.fmean(by_strategy_correct[k]))
    out["strategyCorrect"] = {k: round(statistics.fmean(v), 4) for k, v in by_strategy_correct.items()}

    for sig in ("self", "grounding", "judge"):
        pairs = [(float(r[sig]), r["correct"]) for r in rows if r[sig] is not None]
        per_strategy: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r[sig] is not None:
                per_strategy[r["strategy"]].append(float(r[sig]))
        sig_order = sorted(per_strategy, key=lambda k: -statistics.fmean(per_strategy[k]))
        comparable = [k for k in truth_order if k in per_strategy]
        out["signals"][sig] = {
            "coverage": round(len(pairs) / len(rows), 4) if rows else 0.0,
            "brier": _r(brier(pairs)),
            "ece": _r(ece(pairs)),
            "auroc": _r(auroc(pairs)),
            "spearman": _r(spearman(pairs)),
            "strategyMean": {k: round(statistics.fmean(v), 4) for k, v in per_strategy.items()},
            "bestStrategyAgrees": bool(comparable) and bool(sig_order) and sig_order[0] == comparable[0],
            "rankAgrees": sig_order == comparable if comparable else None,
        }
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


# ─────────────────────────────────────────────────────────────────────────────
# simulate
# ─────────────────────────────────────────────────────────────────────────────

ARMS = ["semantic_search", "graph_first", "hybrid", "keyword_boosted"]

SCENARIOS: dict[str, dict[str, float]] = {
    # The prior favourite is not the best, and every strategy gets a confident
    # synthesiser -- self-confidence cannot separate them.
    "flat_self":        {"semantic_search": 0.55, "graph_first": 0.80, "hybrid": 0.70, "keyword_boosted": 0.45},
    # Close call: the best strategy is only slightly better.
    "close":            {"semantic_search": 0.70, "graph_first": 0.75, "hybrid": 0.65, "keyword_boosted": 0.60},
    # Incumbent is best; measures what verification costs when self is fine.
    "good_incumbent":   {"semantic_search": 0.85, "graph_first": 0.60, "hybrid": 0.55, "keyword_boosted": 0.50},
}

PRIORS = {"semantic_search": 0.60, "graph_first": 0.55, "hybrid": 0.50, "keyword_boosted": 0.45}
PRIOR_STRENGTH = 2.0


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def draw_rewards(rng: random.Random, correct: bool, *, overconfidence: float, self_bias: float,
                 grounding_noise: float, judge_rate: float, judge_noise: float) -> dict[str, float | None]:
    """One query's worth of signals, given whether the answer was actually correct."""
    if correct or rng.random() < overconfidence:
        self_c = _clip(rng.gauss(0.88, 0.06))
    else:
        self_c = _clip(rng.gauss(0.55 + self_bias, 0.12))
    ground = _clip(rng.gauss(0.8 if correct else 0.3, grounding_noise))
    judge = _clip(rng.gauss(1.0 if correct else 0.1, judge_noise)) if rng.random() < judge_rate else None
    verified = (ground + 2 * judge) / 3 if judge is not None else ground
    return {"self": self_c, "grounding": ground, "verified": verified}


def run_policy(rng: random.Random, mus: dict[str, float], reward_key: str, horizon: int,
               **noise: float) -> dict[str, Any]:
    ab = {a: [PRIORS[a] * PRIOR_STRENGTH, (1 - PRIORS[a]) * PRIOR_STRENGTH] for a in ARMS}
    best = max(mus.values())
    regret = 0.0
    picks = Counter()
    for _ in range(horizon):
        arm = max(ARMS, key=lambda a: rng.betavariate(ab[a][0], ab[a][1]))
        picks[arm] += 1
        correct = rng.random() < mus[arm]
        r = draw_rewards(rng, correct, **noise)[reward_key]
        ab[arm][0] += r
        ab[arm][1] += 1 - r
        regret += best - mus[arm]
    best_arm = max(mus, key=mus.get)
    return {"regret": regret, "bestShare": picks[best_arm] / horizon}


def simulate(seeds: int, horizon: int, **noise: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mus in SCENARIOS.items():
        out[name] = {}
        for offset, key in enumerate(("self", "grounding", "verified")):
            # Seeded per (seed, reward) so a run is reproducible; not hash(key),
            # which Python randomises per process.
            runs = [run_policy(random.Random(1000 * s + offset), mus, key, horizon, **noise)
                    for s in range(seeds)]
            regrets = [r["regret"] for r in runs]
            out[name][key] = {
                "regretMean": round(statistics.fmean(regrets), 2),
                "regretSd": round(statistics.pstdev(regrets), 2),
                "bestShare": round(statistics.fmean(r["bestShare"] for r in runs), 3),
            }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="measure each reward signal against gold correctness")
    c.add_argument("--input", required=True)
    c.add_argument("--verifier", choices=["lexical", "nli"], default="lexical")
    c.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    c.add_argument("--threshold", type=float, default=None,
                   help="claim support threshold (default ROUTER_GROUNDING_THRESHOLD or 0.6)")
    c.add_argument("--json", default=None)

    s = sub.add_parser("simulate", help="router regret under self vs verified reward")
    s.add_argument("--seeds", type=int, default=40)
    s.add_argument("--horizon", type=int, default=2000)
    s.add_argument("--overconfidence", type=float, default=0.7,
                   help="P(self-confidence is high on a wrong answer)")
    s.add_argument("--self-bias", type=float, default=0.25)
    s.add_argument("--grounding-noise", type=float, default=0.2)
    s.add_argument("--judge-rate", type=float, default=0.05)
    s.add_argument("--judge-noise", type=float, default=0.1)
    s.add_argument("--json", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "calibrate":
        verifier = nli_verifier(args.nli_model) if args.verifier == "nli" else V.lexical_support
        result = calibrate(load_records(args.input), verifier=verifier, threshold=args.threshold)
        print(f"n={result['n']}  mean correctness={result['meanCorrect']}")
        print(f"strategy correctness: {result['strategyCorrect']}")
        print(f"{'signal':<10} {'cover':>6} {'brier':>7} {'ece':>7} {'auroc':>7} {'rho':>7}  best?  rank?")
        for name, m in result["signals"].items():
            print(f"{name:<10} {m['coverage']:>6} {_fmt(m['brier'])} {_fmt(m['ece'])} "
                  f"{_fmt(m['auroc'])} {_fmt(m['spearman'])}  {str(m['bestStrategyAgrees']):<5}  {m['rankAgrees']}")
    else:
        result = simulate(args.seeds, args.horizon, overconfidence=args.overconfidence,
                          self_bias=args.self_bias, grounding_noise=args.grounding_noise,
                          judge_rate=args.judge_rate, judge_noise=args.judge_noise)
        print(f"{'scenario':<16} {'reward':<10} {'regret':>9} {'sd':>7} {'best-arm share':>15}")
        for scen, by in result.items():
            for key, m in by.items():
                print(f"{scen:<16} {key:<10} {m['regretMean']:>9} {m['regretSd']:>7} {m['bestShare']:>15}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


def _fmt(x: float | None) -> str:
    return f"{'-':>7}" if x is None else f"{x:>7.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
