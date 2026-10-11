# Experiment findings

Every number is copied from a file under `results/`. A cell whose artifact is missing or whose estimate is null is `pending`.

## Claims

**Assumption 1**: contradicted.

at least one dataset has non-overlapping E[self-confidence | Y, arm] intervals

Artifacts: `results/analyses/analyses.json`.

**Proposition 1**: supported.

self-confidence ranks strategies differently from mean correctness on squad, hotpot, nq

Artifacts: `results/calibration/calibration.json`.

**Proposition 2**: supported.

fallback constants reorder arms relative to skip-unobserved on squad, hotpot, nq

Artifacts: `results/analyses/analyses.json`.

**Theorem 1 scale objection**: open.

datasets disagree on whether normalization closes the regret gap

Artifacts: `results/analyses/analyses.json`, `results/replay/replay_summary.json`.

**Proposition 3**: open.

a component slope is not positive, so the combination claim is not triggered

Artifacts: `results/analyses/analyses.json`, `results/signals/judge_unavailable.json`.

**Section 6 prediction**: contradicted.

a decided dataset failed the pre-registered prediction

Artifacts: `results/calibration/calibration.json`.

## Sample

### Sampled questions

| dataset | n |
| --- | --- |
| squad | 1100 |
| hotpot | 1100 |
| nq | 1100 |

Seed stored in the manifest: `0`.

Questions are sorted by qid, then random.Random(seed) shuffles those positions. The sample is the first n_per_dataset of that order and is written sorted by qid. A smaller n with the same seed is that prefix. The passage collection is the full dev split and does not shrink with n. When a larger schema-version-2 sample with this seed is already on disk, the prefix keeps the previously written question rows. A manifest that is not schema version 2 is refused: v1 per-question pools are not reused. The seed is stored on this manifest.

## Retrieval

### Retrieval over the per-question pools

| dataset | n | pool min | pool mean | pool max | top-1 differs | mean Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| squad | 1100 | 1204 | 1204.0000 | 1204 | 0.6509 | 0.4717 |
| hotpot | 1100 | 66635 | 66635.0000 | 66635 | 0.8355 | 0.3137 |
| nq | 1100 | 9152 | 9152.0000 | 9152 | 0.7964 | 0.3511 |

## Generation dry-run

### Generator cost estimate

| field | value |
| --- | --- |
| model | us.anthropic.claude-haiku-4-5-20251001-v1:0 |
| calls | 18000 |
| input tokens (estimate) | 20111359 |
| output tokens (upper bound) | 4608000 |
| USD upper bound | 43.1514 |

ceil(utf-8 bytes / 4) summed over the system prompt and the user message

upper bound charges generator_max_output_tokens on every call

## Calibration

The judge grades how well the retrieved passages support the answer and whether the answer addresses the question. Its prompt scores 1.0 when every claim is supported and the question is answered, and it scores 1.0 for an answer that says the evidence is insufficient when the evidence is insufficient. The score is that grounding judgement. It is not token-F1 correctness. The table below pairs each stored signal, including the judge, with token F1 as recorded. A judge score of 1.0 on an abstention stays 1.0 when correctness is 0 because a gold answer existed.

### Harness metrics

| dataset | signal | coverage | brier | ece | auroc | spearman | kendall_tau | rank_agrees | correctness order | signal order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| squad | self | 1.0000 [1.0000, 1.0000] | 0.1952 [0.1795, 0.2124] | 0.1892 [0.1697, 0.2082] | 0.5850 [0.5595, 0.6107] | 0.1744 [0.1270, 0.2206] | -0.6667 [-1.0000, 0.0000] | no | keyword_boosted > semantic_search > hybrid > graph_first | graph_first > semantic_search > hybrid > keyword_boosted |
| squad | lexical_grounding | 0.5132 [0.4875, 0.5386] | 0.2629 [0.2310, 0.2967] | 0.2960 [0.2648, 0.3290] | 0.5088 [0.4995, 0.5213] | 0.0566 [-0.0168, 0.1226] | -1.0000 [-1.0000, 0.6667] | no | keyword_boosted > semantic_search > hybrid > graph_first | graph_first > hybrid > semantic_search > keyword_boosted |
| squad | nli_grounding | 0.5132 [0.4884, 0.5400] | 0.4778 [0.4428, 0.5111] | 0.5114 [0.4785, 0.5476] | 0.4972 [0.4564, 0.5348] | -0.0062 [-0.0769, 0.0625] | 0.6667 [-0.6667, 1.0000] | no | keyword_boosted > semantic_search > hybrid > graph_first | keyword_boosted > hybrid > semantic_search > graph_first |
| squad | judge | 0.5927 [0.5639, 0.6236] | 0.2098 [0.1864, 0.2343] | 0.2257 [0.2008, 0.2499] | 0.5468 [0.5274, 0.5687] | 0.2107 [0.1447, 0.2753] | -0.6667 [-1.0000, 0.0000] | no | keyword_boosted > semantic_search > hybrid > graph_first | graph_first > semantic_search > hybrid > keyword_boosted |
| hotpot | self | 0.9995 [0.9989, 1.0000] | 0.5702 [0.5521, 0.5886] | 0.6005 [0.5819, 0.6210] | 0.4899 [0.4680, 0.5103] | -0.0102 [-0.0570, 0.0343] | -1.0000 [-1.0000, -0.3333] | no | hybrid > keyword_boosted > semantic_search > graph_first | graph_first > semantic_search > keyword_boosted > hybrid |
| hotpot | lexical_grounding | 0.4430 [0.4216, 0.4664] | 0.1920 [0.1654, 0.2178] | 0.2448 [0.2197, 0.2728] | 0.5155 [0.5028, 0.5312] | 0.0788 [0.0121, 0.1371] | 0.6667 [-0.6667, 1.0000] | no | hybrid > keyword_boosted > semantic_search > graph_first | hybrid > keyword_boosted > graph_first > semantic_search |
| hotpot | nli_grounding | 0.4430 [0.4220, 0.4643] | 0.4075 [0.3782, 0.4372] | 0.4618 [0.4309, 0.4918] | 0.4863 [0.4500, 0.5265] | -0.0688 [-0.1310, -0.0032] | 0.6667 [0.0000, 1.0000] | no | hybrid > keyword_boosted > semantic_search > graph_first | hybrid > semantic_search > keyword_boosted > graph_first |
| hotpot | judge | 0.6125 [0.5832, 0.6407] | 0.6230 [0.5950, 0.6492] | 0.6442 [0.6183, 0.6683] | 0.4963 [0.4840, 0.5083] | -0.0194 [-0.0818, 0.0370] | -0.6667 [-1.0000, 0.0000] | no | hybrid > keyword_boosted > semantic_search > graph_first | graph_first > semantic_search > hybrid > keyword_boosted |
| nq | self | 1.0000 [1.0000, 1.0000] | 0.4489 [0.4302, 0.4681] | 0.4916 [0.4704, 0.5141] | 0.4471 [0.4250, 0.4689] | -0.1411 [-0.1844, -0.0988] | -0.3333 [-1.0000, 0.0000] | no | keyword_boosted > semantic_search > hybrid > graph_first | graph_first > semantic_search > keyword_boosted > hybrid |
| nq | lexical_grounding | 0.6439 [0.6248, 0.6650] | 0.2205 [0.1979, 0.2428] | 0.2946 [0.2725, 0.3185] | 0.4947 [0.4904, 0.4986] | -0.0537 [-0.0851, -0.0121] | 0.3333 [-0.6667, 1.0000] | no | keyword_boosted > semantic_search > hybrid > graph_first | semantic_search > keyword_boosted > graph_first > hybrid |
| nq | nli_grounding | 0.6439 [0.6245, 0.6645] | 0.2891 [0.2666, 0.3128] | 0.3637 [0.3379, 0.3885] | 0.5001 [0.4756, 0.5269] | 0.0478 [-0.0042, 0.1038] | 0.6667 [-0.6667, 1.0000] | no | keyword_boosted > semantic_search > hybrid > graph_first | keyword_boosted > hybrid > semantic_search > graph_first |
| nq | judge | 0.5920 [0.5623, 0.6202] | 0.4740 [0.4470, 0.5011] | 0.5183 [0.4936, 0.5470] | 0.4991 [0.4856, 0.5129] | -0.0068 [-0.0605, 0.0473] | -1.0000 [-1.0000, -0.3333] | no | keyword_boosted > semantic_search > hybrid > graph_first | graph_first > hybrid > semantic_search > keyword_boosted |

## Prediction

Overall verdict: **contradicted**.

squad: contradicted.

hotpot: contradicted.

nq: contradicted.

## Slopes

### c0, c1, and slope

| dataset | signal | c0 | c1 | s |
| --- | --- | --- | --- | --- |
| squad | self | 0.9403 [0.9344, 0.9453] | 0.9528 [0.9510, 0.9543] | 0.0124 [0.0072, 0.0180] |
| squad | lexical_grounding | 0.9725 [0.9504, 0.9900] | 0.9914 [0.9840, 0.9973] | 0.0189 [-0.0019, 0.0418] |
| squad | nli_grounding | 0.4699 [0.4041, 0.5362] | 0.4836 [0.4406, 0.5286] | 0.0137 [-0.0653, 0.0881] |
| squad | judge | 0.9359 [0.9134, 0.9575] | 0.9800 [0.9701, 0.9887] | 0.0441 [0.0209, 0.0713] |
| hotpot | self | 0.9345 [0.9285, 0.9405] | 0.9349 [0.9282, 0.9409] | 0.0003 [-0.0082, 0.0089] |
| hotpot | lexical_grounding | 0.9550 [0.9261, 0.9794] | 0.9874 [0.9810, 0.9930] | 0.0323 [0.0061, 0.0620] |
| hotpot | nli_grounding | 0.5981 [0.5231, 0.6690] | 0.6025 [0.5629, 0.6414] | 0.0044 [-0.0720, 0.0785] |
| hotpot | judge | 0.9570 [0.9435, 0.9692] | 0.9575 [0.9386, 0.9750] | 0.0006 [-0.0233, 0.0224] |
| nq | self | 0.9546 [0.9511, 0.9579] | 0.9478 [0.9448, 0.9505] | -0.0068 [-0.0112, -0.0027] |
| nq | lexical_grounding | 0.9930 [0.9860, 0.9983] | 0.9887 [0.9817, 0.9946] | -0.0042 [-0.0132, 0.0054] |
| nq | nli_grounding | 0.7847 [0.7361, 0.8301] | 0.7968 [0.7702, 0.8226] | 0.0121 [-0.0448, 0.0712] |
| nq | judge | 0.9653 [0.9528, 0.9774] | 0.9719 [0.9614, 0.9811] | 0.0066 [-0.0081, 0.0235] |

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
| squad | self | fractional | 113.4558 | 171.8000 | 0.1327 |
| squad | self | bernoulli | 125.1953 | 181.4000 | 0.0818 |
| squad | self_with_fallbacks | fractional | 166.0978 | 224.0000 | 0.1300 |
| squad | self_with_fallbacks | bernoulli | 158.5155 | 211.0000 | 0.1165 |
| squad | lexical_grounding | fractional | 60.8815 | 110.4000 | 0.0422 |
| squad | lexical_grounding | bernoulli | 60.1624 | 110.2000 | 0.0625 |
| squad | verified | fractional | 91.1122 | 143.2000 | 0.0362 |
| squad | verified | bernoulli | 94.0947 | 151.0000 | 0.0469 |
| squad | verified_plus_self | fractional | 113.2984 | 160.2000 | 0.0864 |
| squad | verified_plus_self | bernoulli | 117.6360 | 171.4000 | 0.0862 |
| squad | normalized_self | fractional | 93.9820 | 145.8000 | 0.2045 |
| squad | normalized_self | bernoulli | 128.1596 | 180.0000 | 0.1969 |
| squad | oracle | fractional | 16.5678 | 71.0000 | 0.6076 |
| squad | oracle | bernoulli | 16.5678 | 71.0000 | 0.6076 |
| hotpot | self | fractional | 190.6425 | 310.0000 | 0.1227 |
| hotpot | self | bernoulli | 135.3111 | 254.8000 | 0.1662 |
| hotpot | self_with_fallbacks | fractional | 83.3585 | 195.8000 | 0.3698 |
| hotpot | self_with_fallbacks | bernoulli | 115.1495 | 225.8000 | 0.1815 |
| hotpot | lexical_grounding | fractional | 46.5093 | 159.0000 | 0.1793 |
| hotpot | lexical_grounding | bernoulli | 36.5980 | 147.4000 | 0.2418 |
| hotpot | verified | fractional | 143.6569 | 259.2000 | 0.0227 |
| hotpot | verified | bernoulli | 131.6335 | 243.8000 | 0.0547 |
| hotpot | verified_plus_self | fractional | 194.9658 | 312.4000 | 0.0465 |
| hotpot | verified_plus_self | bernoulli | 136.5564 | 248.2000 | 0.2016 |
| hotpot | normalized_self | fractional | 110.2011 | 229.8000 | 0.2175 |
| hotpot | normalized_self | bernoulli | 157.9415 | 278.0000 | 0.0947 |
| hotpot | oracle | fractional | 13.6744 | 130.8000 | 0.3927 |
| hotpot | oracle | bernoulli | 13.6744 | 130.8000 | 0.3927 |
| nq | self | fractional | 265.8442 | 333.2000 | 0.0705 |
| nq | self | bernoulli | 188.3422 | 258.6000 | 0.0456 |
| nq | self_with_fallbacks | fractional | 213.2285 | 285.8000 | 0.1287 |
| nq | self_with_fallbacks | bernoulli | 197.2571 | 264.0000 | 0.1587 |
| nq | lexical_grounding | fractional | 17.4300 | 84.8000 | 0.1295 |
| nq | lexical_grounding | bernoulli | 17.4038 | 84.6000 | 0.1295 |
| nq | verified | fractional | 178.9898 | 245.6000 | 0.0425 |
| nq | verified | bernoulli | 60.2209 | 123.6000 | 0.1667 |
| nq | verified_plus_self | fractional | 184.8862 | 246.8000 | 0.0718 |
| nq | verified_plus_self | bernoulli | 221.7840 | 286.0000 | 0.0233 |
| nq | normalized_self | fractional | 135.4347 | 202.6000 | 0.2284 |
| nq | normalized_self | bernoulli | 96.1511 | 165.2000 | 0.2545 |
| nq | oracle | fractional | 16.5622 | 86.0000 | 0.4269 |
| nq | oracle | bernoulli | 16.5622 | 86.0000 | 0.4269 |

Curve CSV and PNG paths are listed in the replay artifact when that artifact exists. They are pending while it does not.

## Correctness sensitivity

Joined correct uses the abstention label when the answer abstains: it is correct only when every source passage was retrieved and the question is unanswerable. Otherwise token F1 decides outside [0.2, 0.8], and the adjudicator decides inside that band.

### Joined correctness and gold contained in the answer

| dataset | primary rate (joined correct) | secondary rate (gold contained) |
| --- | --- | --- |
| squad | 0.7059 | 0.7845 |
| hotpot | 0.3618 | 0.3105 |
| nq | 0.5145 | 0.4432 |

## HotpotQA yes/no

### HotpotQA yes/no questions, reported apart from the other HotpotQA questions

| slice | n | joined correct | token F1 | gold_in_top_k | source_retrieved | abstain_correct | abstain_retrieval_miss | abstain_incorrect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yes/no | 264 | 0.5152 | 0.5152 | 0.8674 | 0.5341 | 0 | 104 | 18 |
| other HotpotQA | 4136 | 0.3520 | 0.3433 | 0.8443 | 0.3431 | 0 | 2127 | 177 |

## Assumption check and missingness

For self-confidence, a null value keeps its status as the missingness reason. `omitted` is a reply that did not report a confidence, including a bare abstention whose answer is `insufficient evidence`. `truncated` is a JSON object cut off before it could be parsed. An `ok` row with `trailing_truncated` set is an observation: the object was complete and prose after it hit the output cap.

### Non-overlapping E[R | Y, arm] intervals

| dataset | signal | n pairs | any |
| --- | --- | --- | --- |
| squad | self | 4 | yes |
| squad | lexical_grounding | 0 | no |
| hotpot | self | 3 | yes |
| hotpot | lexical_grounding | 0 | no |
| nq | self | 3 | yes |
| nq | lexical_grounding | 0 | no |

### Missingness m_a

| dataset | signal | mean m | m by arm | accuracy missing vs observed |
| --- | --- | --- | --- | --- |
| squad | self | 0.0000 | graph_first=0.0000, hybrid=0.0000, keyword_boosted=0.0000, semantic_search=0.0000 | graph_first: miss=None obs=0.5400, hybrid: miss=None obs=0.7682, keyword_boosted: miss=None obs=0.7891, semantic_search: miss=None obs=0.7264 |
| squad | lexical_grounding | 0.4868 | graph_first=0.6182, hybrid=0.4336, keyword_boosted=0.4227, semantic_search=0.4727 | graph_first: miss=0.4059 obs=0.7571, hybrid: miss=0.8176 obs=0.7303, keyword_boosted: miss=0.8624 obs=0.7354, semantic_search: miss=0.6962 obs=0.7534 |
| hotpot | self | 0.0005 | graph_first=0.0000, hybrid=0.0009, keyword_boosted=0.0000, semantic_search=0.0009 | graph_first: miss=None obs=0.1700, hybrid: miss=0.0000 obs=0.4586, keyword_boosted: miss=None obs=0.4591, semantic_search: miss=0.0000 obs=0.3603 |
| hotpot | lexical_grounding | 0.5570 | graph_first=0.7918, hybrid=0.4409, keyword_boosted=0.4409, semantic_search=0.5545 | graph_first: miss=0.0034 obs=0.8035, hybrid: miss=0.0082 obs=0.8130, keyword_boosted: miss=0.0062 obs=0.8163, semantic_search: miss=0.0000 obs=0.8082 |
| nq | self | 0.0000 | graph_first=0.0000, hybrid=0.0000, keyword_boosted=0.0000, semantic_search=0.0000 | graph_first: miss=None obs=0.2445, hybrid: miss=None obs=0.5855, keyword_boosted: miss=None obs=0.6209, semantic_search: miss=None obs=0.6073 |
| nq | lexical_grounding | 0.3561 | graph_first=0.6964, hybrid=0.2627, keyword_boosted=0.2264, semantic_search=0.2391 | graph_first: miss=0.0000 obs=0.8054, hybrid: miss=0.0000 obs=0.7941, keyword_boosted: miss=0.0000 obs=0.8026, semantic_search: miss=0.0000 obs=0.7981 |

Causal per-arm rank of verbalized self-confidence. History H is the self-confidences seen on earlier selections of this (question type, arm) in this replay only. Reward = (|{h in H: h < c}| + 0.5 |{h in H: h = c}|) / |H|. Empty H yields a missing reward and no posterior update. Logged confidences from arms that were not selected are not used.

## Cost projection

The Bedrock ceiling is total_usd_cap 30. already_spent_usd 12.02 is spend already on the AWS bill from prior ledgers. The cap check adds it to this results ledger. A job is refused when that sum plus the projected job cost times spend_safety_factor 1.1 exceeds the ceiling. Bulk stages default to `batch`.

Projected spend is copied from `results/cost_projection.json` when that file is present. That artifact records whether input tokens were measured from the retrieved passages or taken from the prompt template, whether output tokens are a ledger mean or expected_output_tokens, and the adjudication call count, which is an assumption. The printed USD is the projected cost. The safety factor is applied only in the cap check.

### Projected spend by stage

| stage | calls | on-demand USD | batch USD |
| --- | --- | --- | --- |
| generate | 13200 | 21.1480 | 10.5740 |
| judge | 7920 | 5.7695 | 2.8848 |
| subset_haiku | 4000 | 6.4591 | 3.2295 |
| subset_nova | 1000 | 1.1184 | 0.5592 |
| adjudicate | 1188 | 0.0631 | 0.0316 |
| sample | 0 | 0.0000 | 0.0000 |
| retrieve | 0 | 0.0000 | 0.0000 |
| lexical | 0 | 0.0000 | 0.0000 |
| nli | 0 | 0.0000 | 0.0000 |
| self_percentile | 0 | 0.0000 | 0.0000 |
| logistic | 0 | 0.0000 | 0.0000 |
| oracle | 0 | 0.0000 | 0.0000 |
| calibrate | 0 | 0.0000 | 0.0000 |
| replay | 0 | 0.0000 | 0.0000 |
| analyses | 0 | 0.0000 | 0.0000 |
| study | 0 | 0.0000 | 0.0000 |
| agreement | 0 | 0.0000 | 0.0000 |
| write_results | 0 | 0.0000 | 0.0000 |

Input tokens are ceil(utf-8 bytes / 4) summed over the prompts the stages send. A generator prompt is the system prompt plus the user message built from the question and the retrieved passages, including passage text and titles. A judge prompt is the judge user message with the question, the passage texts (cut at 12000 characters, the same cap the judge stage uses), and an answer stand-in. An adjudicator prompt is the system prompt plus the user message with the question, the gold answers, and the same stand-in. No model was called.

The sums add every stage whose call count is determined, including adjudication. ASSUMPTION: adjudication calls are 0.15 of judged rows (1188 of 7920), from v2.adjudication_band_fraction. This is not a count of answers whose token F1 is inside [0.2, 0.8]. That count is known only after generation. The fraction is configurable.

Sum of determined stages, on-demand 34.5580, batch 17.2790.

## Subset

Scope stored on the manifest: `stratified_total`. Seed `0`.

### Seeded subset

| dataset | n | stream |
| --- | --- | --- |
| squad | 84 | subset:squad |
| hotpot | 83 | subset:hotpot |
| nq | 83 | subset:nq |

## Haiku agreement

`v2.subset_samples` is 4. The subset study draws that many Haiku answers at temperature 1.

Modal fraction is the share of SQuAD-normalised answers equal to the most common normalised answer. Pairwise agreement is the fraction of unordered pairs whose normalised answers are equal. A missing sample leaves every mean null.

Samples recorded on the artifact: 4.

### Temperature-1 Haiku samples

| dataset | n | mean modal fraction | mean pairwise agreement |
| --- | --- | --- | --- |
| squad | 336 | 0.9554 | 0.9241 |
| hotpot | 332 | 0.9736 | 0.9573 |
| nq | 332 | 0.9578 | 0.9232 |

## Signal information

Mean of joined correct Y among rows where R is observed. This is not the missingness rate m_a.

s^2 / Var(R). Null when s is null or Var(R) is null or 0.

s^2 / (m(1-m)). Null when s is null or m is 0 or 1.

s^2 / residual. Null when the residual is null or not positive.

### s^2 / Var(R), s^2 / (m(1-m)), and information per round

| dataset | signal | n | m | s^2/Var(R) | s^2/(m(1-m)) | information per round |
| --- | --- | --- | --- | --- | --- | --- |
| squad | self | 4400 | 0.7059 | 0.0701 | 0.0007 | 0.0712 |
| squad | lexical_grounding | 2258 | 0.7427 | 0.0270 | 0.0019 | 0.0271 |
| squad | nli_grounding | 2258 | 0.7427 | 0.0008 | 0.0010 | 0.0008 |
| squad | judge | 2608 | 0.7170 | 0.0748 | 0.0096 | 0.0760 |
| squad | verified | 3519 | 0.7170 | 0.0527 | 0.0036 | 0.0532 |
| squad | oracle | 4400 | 0.7059 | 4.8158 | 4.8169 | 21189.6404 |
| hotpot | self | 4398 | 0.3620 | 0.0000 | 0.0000 | 0.0000 |
| hotpot | lexical_grounding | 1949 | 0.8117 | 0.0589 | 0.0068 | 0.0594 |
| hotpot | nli_grounding | 1949 | 0.8117 | 0.0001 | 0.0001 | 0.0001 |
| hotpot | judge | 2695 | 0.3670 | 0.0000 | 0.0000 | 0.0000 |
| hotpot | verified | 3446 | 0.4608 | 0.0054 | 0.0005 | 0.0054 |
| hotpot | oracle | 4400 | 0.3618 | 4.3298 | 4.3308 | 19051.0601 |
| nq | self | 4400 | 0.5145 | 0.0173 | 0.0002 | 0.0173 |
| nq | lexical_grounding | 2833 | 0.7992 | 0.0018 | 0.0001 | 0.0018 |
| nq | nli_grounding | 2833 | 0.7992 | 0.0009 | 0.0009 | 0.0009 |
| nq | judge | 2605 | 0.5259 | 0.0021 | 0.0002 | 0.0021 |
| nq | verified | 3732 | 0.6066 | 0.0010 | 0.0000 | 0.0010 |
| nq | oracle | 4400 | 0.5145 | 4.0025 | 4.0034 | 17610.9038 |
| all | self | 13198 | 0.5275 | 0.0033 | 0.0001 | 0.0033 |
| all | lexical_grounding | 7040 | 0.7845 | 0.0132 | 0.0010 | 0.0132 |
| all | nli_grounding | 7040 | 0.7845 | 0.0032 | 0.0043 | 0.0032 |
| all | judge | 7908 | 0.5348 | 0.0095 | 0.0011 | 0.0095 |
| all | verified | 10697 | 0.5960 | 0.0124 | 0.0008 | 0.0125 |
| all | oracle | 13200 | 0.5274 | 4.0118 | 4.0121 | 52955.3084 |

## Measured slopes against the simulation's assumptions

s = c1 - c0 is the mean reward on correct answers minus the mean on wrong ones. The assumed slopes are the verified-reward simulation's pre-clip values from `results/simulation_assumptions.json`. The judge and the verified reward have none. The interval on s is the cluster-bootstrap interval from `results/analyses/analyses.json`; that stage has none for the verified reward or the pooled row.

### Measured slopes by dataset

| dataset | signal | n | m | c0 | c1 | s | s interval | excludes 0 | Var(R) | s^2/Var(R) | assumed s | measured / assumed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| squad | self | 4400 | 0.7059 | 0.9403 | 0.9528 | 0.0124 | [0.0072, 0.0180] | yes | 0.0022 | 0.0701 | 0.0240 | 0.5185 |
| squad | lexical_grounding | 2258 | 0.7427 | 0.9725 | 0.9914 | 0.0189 | [-0.0019, 0.0418] | no | 0.0132 | 0.0270 | 0.5000 | 0.0378 |
| squad | nli_grounding | 2258 | 0.7427 | 0.4699 | 0.4836 | 0.0137 | [-0.0653, 0.0881] | no | 0.2490 | 0.0008 | 0.5000 | 0.0274 |
| squad | judge | 2608 | 0.7170 | 0.9359 | 0.9800 | 0.0441 | [0.0209, 0.0713] | yes | 0.0260 | 0.0748 | none | none |
| squad | verified | 3519 | 0.7170 | 0.9583 | 0.9854 | 0.0271 | none | none | 0.0140 | 0.0527 | none | none |
| hotpot | self | 4398 | 0.3620 | 0.9345 | 0.9349 | 0.0003 | [-0.0082, 0.0089] | no | 0.0129 | 0.0000 | 0.0240 | 0.0143 |
| hotpot | lexical_grounding | 1949 | 0.8117 | 0.9550 | 0.9874 | 0.0323 | [0.0061, 0.0620] | yes | 0.0177 | 0.0589 | 0.5000 | 0.0646 |
| hotpot | nli_grounding | 1949 | 0.8117 | 0.5981 | 0.6025 | 0.0044 | [-0.0720, 0.0785] | no | 0.2358 | 0.0001 | 0.5000 | 0.0088 |
| hotpot | judge | 2695 | 0.3670 | 0.9570 | 0.9575 | 0.0006 | [-0.0233, 0.0224] | no | 0.0394 | 0.0000 | none | none |
| hotpot | verified | 3446 | 0.4608 | 0.9636 | 0.9748 | 0.0112 | none | none | 0.0230 | 0.0054 | none | none |
| nq | self | 4400 | 0.5145 | 0.9546 | 0.9478 | -0.0068 | [-0.0112, -0.0027] | yes | 0.0027 | 0.0173 | 0.0240 | -0.2819 |
| nq | lexical_grounding | 2833 | 0.7992 | 0.9930 | 0.9887 | -0.0042 | [-0.0132, 0.0054] | no | 0.0100 | 0.0018 | 0.5000 | -0.0085 |
| nq | nli_grounding | 2833 | 0.7992 | 0.7847 | 0.7968 | 0.0121 | [-0.0448, 0.0712] | no | 0.1620 | 0.0009 | 0.5000 | 0.0242 |
| nq | judge | 2605 | 0.5259 | 0.9653 | 0.9719 | 0.0066 | [-0.0081, 0.0235] | no | 0.0209 | 0.0021 | none | none |
| nq | verified | 3732 | 0.6066 | 0.9783 | 0.9815 | 0.0032 | none | none | 0.0108 | 0.0010 | none | none |

### Measured slopes pooled over the datasets (summary)

| dataset | signal | n | m | c0 | c1 | s | s interval | excludes 0 | Var(R) | s^2/Var(R) | assumed s | measured / assumed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | self | 13198 | 0.5275 | 0.9426 | 0.9471 | 0.0045 | none | none | 0.0060 | 0.0033 | 0.0240 | 0.1862 |
| all | lexical_grounding | 7040 | 0.7845 | 0.9759 | 0.9891 | 0.0132 | none | none | 0.0132 | 0.0132 | 0.5000 | 0.0264 |
| all | nli_grounding | 7040 | 0.7845 | 0.6190 | 0.6461 | 0.0271 | none | none | 0.2285 | 0.0032 | 0.5000 | 0.0541 |
| all | judge | 7908 | 0.5348 | 0.9556 | 0.9721 | 0.0166 | none | none | 0.0289 | 0.0095 | none | none |
| all | verified | 10697 | 0.5960 | 0.9674 | 0.9814 | 0.0140 | none | none | 0.0158 | 0.0124 | none | none |

The pooled row mixes datasets with different base accuracy (oracle m: squad 0.7059, hotpot 0.3618, nq 0.5145). Read the per-dataset rows first.

### Self-confidence against the verified reward

| dataset | self s | verified s | self sign | verified sign | verified s / self s | self s^2/Var(R) | verified s^2/Var(R) | verified / self s^2/Var(R) | larger s^2/Var(R) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| squad | 0.0124 | 0.0271 | positive | positive | 2.1799 | 0.0701 | 0.0527 | 0.7507 | self |
| hotpot | 0.0003 | 0.0112 | positive | positive | 32.6439 | 0.0000 | 0.0054 | 596.1410 | verified |
| nq | -0.0068 | 0.0032 | negative | positive | pending | 0.0173 | 0.0010 | 0.0562 | self |
| all | 0.0045 | 0.0140 | positive | positive | 3.1378 | 0.0033 | 0.0124 | 3.7202 | verified |

On squad, self-confidence has the larger s^2/Var(R) (0.0701 against 0.0527) with slope 0.0124. Its Var(R) is 0.0022 against 0.0140 for the verified reward, so the ratio is large because the variance is small, not because the slope is.

On nq, self-confidence has the larger s^2/Var(R) (0.0173 against 0.0010) with slope -0.0068. Its Var(R) is 0.0027 against 0.0108 for the verified reward, so the ratio is large because the variance is small, not because the slope is.

On nq, the self slope is negative (-0.0068).

On nq, the lexical_grounding slope is negative (-0.0042).

On squad, the assumed grounding slope 0.5000 is 26.4658 times off the measured lexical_grounding slope 0.0189.

On squad, the assumed grounding slope 0.5000 is 36.4374 times off the measured nli_grounding slope 0.0137.

On hotpot, the assumed self slope 0.0240 is 69.9870 times off the measured self slope 0.0003.

On hotpot, the assumed grounding slope 0.5000 is 15.4718 times off the measured lexical_grounding slope 0.0323.

On hotpot, the assumed grounding slope 0.5000 is 113.2572 times off the measured nli_grounding slope 0.0044.

On nq, the assumed self slope 0.0240 has the wrong sign: self measures -0.0068.

On nq, the assumed grounding slope 0.5000 has the wrong sign: lexical_grounding measures -0.0042.

On nq, the assumed grounding slope 0.5000 is 41.2890 times off the measured nli_grounding slope 0.0121.

On all, the assumed grounding slope 0.5000 is 37.8874 times off the measured lexical_grounding slope 0.0132.

On all, the assumed grounding slope 0.5000 is 18.4699 times off the measured nli_grounding slope 0.0271.

## Thompson sampling streams

One Beta draw per arm. Posterior alpha is prior_alpha plus the discounted sum of rewards, and beta is prior_beta plus the discounted sum of 1-R. The prior is the general routing weight times ROUTER_PRIOR_STRENGTH. Ties keep the earlier arm in the deployed priority order.

The Gaussian prior matches the Beta prior mean and variance. The pseudo-count n0 is ROUTER_PRIOR_STRENGTH. Observation count, observation mean, and the sum of squared deviations decay by the discount. The prior does not decay. The posterior mean blends n0 * prior_mean with n_obs * observation_mean. Until two observations have been counted the draw variance is the prior variance. After that it is the unbiased observation variance. The draw is Normal(posterior_mean, sqrt(variance / (n0 + n_obs))) from random.gauss.

Each stream draws a question with replacement for the configured number of rounds. The reward is joined correct. Seeds are seed, seed+1, ... .

On each observation the stored observation statistics are multiplied by the discount and the new reward is added. The prior is left as it started.

### Beta and Gaussian Thompson sampling on joined correct

| policy | discount | seeds | rounds | pseudo-regret | realized regret | best-arm share |
| --- | --- | --- | --- | --- | --- | --- |
| beta | none | 500 | 10000 | 103.1127 | 823.6140 | 0.5480 |
| beta | 0.99 | 500 | 10000 | 140.8791 | 861.7940 | 0.4701 |
| beta | 0.995 | 500 | 10000 | 122.1386 | 843.3960 | 0.5029 |
| beta | 0.999 | 500 | 10000 | 106.0125 | 828.2340 | 0.5403 |
| gaussian | none | 500 | 10000 | 101.4349 | 821.2400 | 0.5346 |
| gaussian | 0.99 | 500 | 10000 | 136.5730 | 857.0680 | 0.4662 |
| gaussian | 0.995 | 500 | 10000 | 121.1586 | 843.1460 | 0.4881 |
| gaussian | 0.999 | 500 | 10000 | 105.1104 | 827.7800 | 0.5163 |

## Judge coverage

Question ids are shuffled once with stream judge_coverage. Each rate keeps the first take_count(n, rate) of that shuffle, so 1% is a prefix of 5%, 20%, and 100%. Judge values outside the prefix are treated as unobserved, and the verified reward is recomputed from lexical grounding and the remaining judge values. The long streams are not rerun.

### Judge coverage on the pooled log

| rate | signal | s^2/Var(R) | s^2/(m(1-m)) | information per round |
| --- | --- | --- | --- | --- |
| 0.0100 | verified | 0.0173 | 0.0014 | 0.0174 |
| 0.0100 | judge | 0.2149 | 0.0609 | 0.2266 |
| 0.0500 | verified | 0.0151 | 0.0011 | 0.0151 |
| 0.0500 | judge | 0.0190 | 0.0019 | 0.0191 |
| 0.2000 | verified | 0.0160 | 0.0011 | 0.0161 |
| 0.2000 | judge | 0.0065 | 0.0007 | 0.0065 |
| 1.0000 | verified | 0.0124 | 0.0008 | 0.0125 |
| 1.0000 | judge | 0.0095 | 0.0011 | 0.0095 |

## Study assumption check

Cluster-bootstrap intervals of E[R | Y, arm, R observed]. A non-overlapping pair is two arms whose intervals at the same Y do not overlap.

### Non-overlapping pairs from the study artifact

| dataset | signal | n pairs |
| --- | --- | --- |
| squad | self | 3 |
| squad | lexical_grounding | 0 |
| squad | nli_grounding | 0 |
| squad | judge | 2 |
| hotpot | self | 3 |
| hotpot | lexical_grounding | 0 |
| hotpot | nli_grounding | 0 |
| hotpot | judge | 3 |
| nq | self | 3 |
| nq | lexical_grounding | 0 |
| nq | nli_grounding | 0 |
| nq | judge | 3 |

## Second generator

Token F1 of the Nova answer against the sample gold, on subset questions only.

### Nova Pro token F1 on the subset

| dataset | n | mean token F1 | token-F1 correct rate |
| --- | --- | --- | --- |
| squad | 336 | 0.6035 | 0.6071 |
| hotpot | 332 | 0.4300 | 0.4488 |
| nq | 332 | 0.5294 | 0.5783 |
