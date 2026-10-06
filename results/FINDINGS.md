# Experiment findings

Every number is copied from a file under `results/`. A cell whose artifact is missing or whose estimate is null is `pending`.

## Sample

### Sampled questions

| dataset | n |
| --- | --- |
| squad | 1500 |
| hotpot | 1500 |
| nq | 1500 |

Seed stored in the manifest: `0`.

## Retrieval

### Retrieval over the per-question pools

| dataset | n | pool min | pool mean | pool max | top-1 differs | mean Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| squad | 1500 | 21 | 36.7473 | 49 | 0.8360 | 0.4349 |
| hotpot | 1500 | 2 | 9.9560 | 10 | 0.9187 | 0.5717 |
| nq | 1500 | 1 | 1.6800 | 9 | 0.1207 | 0.9895 |

## Generation dry-run

### Generator cost upper bound

| field | value |
| --- | --- |
| model | us.anthropic.claude-haiku-4-5-20251001-v1:0 |
| calls | 18000 |
| input tokens (estimate) | 15386077 |
| output tokens (upper bound) | 9216000 |
| USD upper bound | 61.4661 |

ceil(utf-8 bytes / 4) summed over the system prompt and the user message

upper bound charges generator_max_output_tokens on every call

## Calibration

### Harness metrics

| dataset | signal | coverage | brier | ece | auroc | spearman | kendall_tau | rank_agrees | correctness order | signal order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| squad | self | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| squad | lexical_grounding | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| squad | nli_grounding | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| squad | judge | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hotpot | self | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hotpot | lexical_grounding | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hotpot | nli_grounding | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hotpot | judge | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nq | self | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nq | lexical_grounding | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nq | nli_grounding | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nq | judge | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Prediction

Verdict: pending.

## Slopes

### c0, c1, and slope

| dataset | signal | c0 | c1 | s |
| --- | --- | --- | --- | --- |
| squad | self | pending | pending | pending |
| squad | lexical_grounding | pending | pending | pending |
| squad | nli_grounding | pending | pending | pending |
| squad | judge | pending | pending | pending |
| hotpot | self | pending | pending | pending |
| hotpot | lexical_grounding | pending | pending | pending |
| hotpot | nli_grounding | pending | pending | pending |
| hotpot | judge | pending | pending | pending |
| nq | self | pending | pending | pending |
| nq | lexical_grounding | pending | pending | pending |
| nq | nli_grounding | pending | pending | pending |
| nq | judge | pending | pending | pending |

### Assumed slopes from the verified-reward simulation

| signal | c0 | c1 | s |
| --- | --- | --- | --- |
| self | 0.8560 | 0.8800 | 0.0240 |
| grounding | 0.3000 | 0.8000 | 0.5000 |

Source: scripts/verified_reward_bench.py draw_rewards, CLI defaults. means are before the [0, 1] clip applied by draw_rewards.

`routing_regret_sim.py` draws Beta(mu * noise_k, (1 - mu) * noise_k) with noise_k=20.0. That simulation has no overconfidence slope. The slopes above are the verified-reward simulation.

## Replay

### True-correctness regret (mean over seeds)

| dataset | reward | update | pseudo-regret | realized regret | best-arm share |
| --- | --- | --- | --- | --- | --- |
| squad | self | fractional | pending | pending | pending |
| squad | self | bernoulli | pending | pending | pending |
| squad | self_with_fallbacks | fractional | pending | pending | pending |
| squad | self_with_fallbacks | bernoulli | pending | pending | pending |
| squad | lexical_grounding | fractional | pending | pending | pending |
| squad | lexical_grounding | bernoulli | pending | pending | pending |
| squad | verified | fractional | pending | pending | pending |
| squad | verified | bernoulli | pending | pending | pending |
| squad | verified_plus_self | fractional | pending | pending | pending |
| squad | verified_plus_self | bernoulli | pending | pending | pending |
| squad | normalized_self | fractional | pending | pending | pending |
| squad | normalized_self | bernoulli | pending | pending | pending |
| squad | oracle | fractional | pending | pending | pending |
| squad | oracle | bernoulli | pending | pending | pending |
| hotpot | self | fractional | pending | pending | pending |
| hotpot | self | bernoulli | pending | pending | pending |
| hotpot | self_with_fallbacks | fractional | pending | pending | pending |
| hotpot | self_with_fallbacks | bernoulli | pending | pending | pending |
| hotpot | lexical_grounding | fractional | pending | pending | pending |
| hotpot | lexical_grounding | bernoulli | pending | pending | pending |
| hotpot | verified | fractional | pending | pending | pending |
| hotpot | verified | bernoulli | pending | pending | pending |
| hotpot | verified_plus_self | fractional | pending | pending | pending |
| hotpot | verified_plus_self | bernoulli | pending | pending | pending |
| hotpot | normalized_self | fractional | pending | pending | pending |
| hotpot | normalized_self | bernoulli | pending | pending | pending |
| hotpot | oracle | fractional | pending | pending | pending |
| hotpot | oracle | bernoulli | pending | pending | pending |
| nq | self | fractional | pending | pending | pending |
| nq | self | bernoulli | pending | pending | pending |
| nq | self_with_fallbacks | fractional | pending | pending | pending |
| nq | self_with_fallbacks | bernoulli | pending | pending | pending |
| nq | lexical_grounding | fractional | pending | pending | pending |
| nq | lexical_grounding | bernoulli | pending | pending | pending |
| nq | verified | fractional | pending | pending | pending |
| nq | verified | bernoulli | pending | pending | pending |
| nq | verified_plus_self | fractional | pending | pending | pending |
| nq | verified_plus_self | bernoulli | pending | pending | pending |
| nq | normalized_self | fractional | pending | pending | pending |
| nq | normalized_self | bernoulli | pending | pending | pending |
| nq | oracle | fractional | pending | pending | pending |
| nq | oracle | bernoulli | pending | pending | pending |

Curve CSV and PNG paths are listed in the replay artifact when that artifact exists. They are pending while it does not.

## Assumption check and missingness

Analyses artifact is missing.
