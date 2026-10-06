# Real-data routing protocol

This directory is the offline protocol for the paper on verified rewards for an
adaptive retrieval router. The router, the fractional Beta update, lexical
grounding, and the judge prompt are the ones already in the repository
(`src/query_api/rag_router.py`, `src/query_api/verified_reward.py`). Model runs
that need a hosted LLM are pending. Stages that do not call a model write
artifacts under `results/`. `results/FINDINGS.md` is generated from those
artifacts; a missing artifact is an empty cell marked pending.

## What each stage writes

| Stage | Script | Needs a hosted LLM |
| --- | --- | --- |
| Sample | `scripts/experiments/sample_datasets.py` | no |
| Retrieval | `scripts/experiments/retrieve.py` | no |
| Generation | `scripts/experiments/generate.py` | yes, except `--dry-run` |
| Signals | `scripts/experiments/signals.py` | judge only |
| Calibration | `scripts/experiments/calibrate.py` | no |
| Replay | `scripts/experiments/replay.py` | no |
| Analyses | `scripts/experiments/analyses.py` | no |
| Tables | `scripts/experiments/write_results.py` | no |

Commands, in order, are in `RUNBOOK.md`.

## Datasets and the candidate pool

Each question is retrieved from its own candidate pool. The gold passage is
inside that pool. This is closed-pool retrieval, not retrieval over Wikipedia
at large.

* **SQuAD 2.0** validation (`rajpurkar/squad_v2`). The pool is every distinct
  context paragraph in the validation split that shares the question's article
  title. Paragraphs that are never a question context are not in the split.
  Question type is `answerable` or `unanswerable`. An unanswerable question
  has an empty gold-answer list.
* **HotpotQA** distractor validation (`hotpotqa/hotpot_qa`, config
  `distractor`). The pool is the ten paragraphs shipped with the question.
  `bridge` and `comparison` are kept as question types.
* **Natural Questions**, tractable form: the MRQA 2019 in-domain dev file
  `NaturalQuestionsShort.jsonl.gz` (md5 `c0347eebbca02d10d1b07b9a64efe61d`).
  Each row has gold short-answer strings and a Wikipedia context that MRQA
  truncated to the first 800 tokens, kept only when the short answer occurs
  in that window. The context is split into overlapping character windows
  (`nq_window_chars` / `nq_overlap_chars` in `config.yaml`) so the pool has
  more than one passage. The full Wikipedia page is not in this file.
  Question type is `nq`.

The sample is a seeded shuffle of question ids, `--n-per-dataset` (default
1500), written to `results/samples/<dataset>.jsonl` and listed in
`results/samples/sample_manifest.json`. Fewer questions than requested is an
error. The script does not pad.

## Retrieval arms

All four arms rank the same pool. `top_k` (default 5) passages are stored with
id, text, and score.

* `semantic_search`: cosine similarity with
  `sentence-transformers/all-MiniLM-L6-v2` on CPU. The artifact records the
  snapshot revision: `config._commit_hash` when the library sets it, otherwise
  the commit id in the local tokenizer path (`snapshots/<commit>/`).
* `graph_first`: an entity/co-occurrence graph built on that question's pool.
  Entities are maximal capitalised phrases that are not a single stopword,
  plus numeric tokens. A query term matches an entity when it casefolds equal
  to a word in the entity (these questions are mostly lowercase). An undirected
  edge joins entities that share a passage. A passage scores the count of
  query entities it contains, plus half the count of their graph neighbours it
  contains. Ties break by passage id. A pool with no entities ranks passages
  by id.
* `keyword_boosted`: Okapi BM25 (`k1=1.5`, `b=0.75`), then the deployed blend
  from `retriever._keyword_boost_rerank`: keyword weight 0.25 and vector
  weight 0.75, after min-max normalising both scores inside the pool.
* `hybrid`: the mean of the min-max normalised vector, graph, and BM25 scores.
  This is the local stand-in for the deployed hybrid arm, which mixes vector
  search, the graph, and a keyword boost.

## Correctness

Token F1 uses the SQuAD normalisation in `scripts/verified_reward_bench.py`.
An answerable question is correct when that F1 is at least 0.5. An
unanswerable question is correct exactly when the answer abstains. Abstention
is `verified_reward_bench.is_abstention`: the normalised answer is empty, or
it matches `verified_reward._ABSTAIN_RE` (phrases such as "insufficient
evidence" and "cannot answer" that the claim splitter already drops).

## Self-confidence

The generator returns JSON `{"answer", "confidence"}`. The prompt is
`experiments/prompts/generator_system.txt` and is copied into the generation
artifact. Parsing follows `synthesizer._confidence_from`:

| Status | When | Stored number | Used as a reward |
| --- | --- | --- | --- |
| `ok` | a number in [0, 1] | that number | yes |
| `omitted` | JSON without a usable confidence | 0.7 | only in the fallback replay row |
| `unparseable` | output is not a JSON object | 0.5 | only in the fallback replay row |
| `failed` | the call failed | 0.0 | only in the fallback replay row |

The 0.7 / 0.5 / 0.0 values are the deployed fallbacks. The skip-unobserved
reward leaves every status other than `ok` as missing.

## Grounding, NLI, judge

Lexical grounding is `verified_reward.grounding_signal` with
`lexical_support` and tau 0.6. A claim is a sentence with at least three
content tokens, and abstentions are dropped. A claim is supported when some
retrieved passage reaches tau. Any number in the claim must appear verbatim
in the passage. The signal is the fraction of claims supported. It is missing
when there are no passages or no checkable claims; the row records that reason.

NLI replaces the lexical verifier with `cross-encoder/nli-deberta-v3-small`.
The score is the entailment-class probability. The same tau and the same
missingness rules apply. The artifact records the model revision the library
reports.

The judge prompt is `verified_reward.JUDGE_PROMPT`, copied verbatim into the
judge artifact. The judge model defaults to
`us.meta.llama3-3-70b-instruct-v1:0`. The run stops if that id equals the
generator id unless `--allow-same-judge` is passed, and the flag is stored
either way. Sampling is `verified_reward.should_judge`: SHA-256 of the
question id, first 8 hex characters as an integer, divided by 2^32, included
when the value is below `judge_sample_rate` (0.05). The draw does not use
`--seed`. Every arm of a sampled question is judged. Other questions are
written with reason `not_sampled`.

The verified reward is `verified_reward.combine`: grounding weight 1, judge
weight 2, and missing when neither was observed. `verified+self` adds
self-confidence at weight 1 when its status is `ok`.

## Replay

`rag_router.select_strategy` draws one Beta sample per arm. The fractional
update is `alpha += R`, `beta += 1 - R`. A missing reward does not update.
The Bernoulli-trick check draws `B ~ Bernoulli(R)` and updates with `B`.

These question types have no seeded routing edge. Each posterior starts from
the deployed `general` prior in `models.ROUTING_PRIORS`, scaled by
`ROUTER_PRIOR_STRENGTH`, which is how a scalar weight is turned into a Beta
prior on read.

Question order is a seeded permutation (`replay_seeds` in the config).
Pseudo-regret sums `mu*(type) - mu(selected, type)` with `mu` the arm's mean
binary correctness on that question type over the whole log. Realized regret
sums `max_arm Y - Y_selected` on that question. Best-arm share counts a pull
when the selected arm is tied for the highest `mu` in its question type.

`normalized_self` is computed inside the replay, not from the finished log.
For the selected `(question type, arm)`, history is the self-confidences seen
on earlier selections of that pair in this replay. The reward is the fraction
of that history below the current value, with ties counting half. An empty
history is a missing reward. Confidences from arms that were not selected are
not used. The definition is stored on the analyses artifact.

`self_with_fallbacks` substitutes 0.7 / 0.5 / 0.0 for omitted / unparseable /
failed. `oracle` uses binary correctness.

## Calibration and the prediction

Per dataset and per signal the harness reports coverage, Brier, 10-bin ECE,
AUROC at F1 >= 0.5, Spearman with F1, both strategy rankings, and Kendall tau.
Brier, ECE, AUROC, and Spearman are the functions in
`scripts/verified_reward_bench.py`. Brier and ECE use that file's continuous
F1 target. Intervals are cluster-bootstrap percentile intervals over question
ids (`bootstrap_samples`, default 1000) with the run seed.

The prediction, with thresholds fixed in `config.yaml` before any model run:
self-confidence coverage is at least 0.8, its AUROC is at most 0.6, and its
strategy ranking disagrees with mean correctness on every dataset; lexical
grounding has lower coverage and its ranking agrees. A dataset is supported
when every clause holds, contradicted when every required number is present
and a clause fails, and pending when a number is missing. The overall verdict
stays pending until squad, hotpot, and nq are all decided.

Slopes are `c0 = E[R | Y=0, R observed]`, `c1 = E[R | Y=1, R observed]`,
`s = c1 - c0`. The verified-reward simulation's pre-clip assumptions
(`s_self = 0.024` at rho 0.7, `s_ground = 0.5`) are written beside them from
`scripts/verified_reward_bench.py`. `scripts/routing_regret_sim.py` draws
`Beta(mu * 20, (1-mu) * 20)` and does not define those slopes.

## Cost

Prices are on-demand USD per million tokens in `config.yaml`, with the price
list version they were read from. A model id that is not in the table stops
the run. `--dry-run` estimates input tokens as `ceil(utf-8 bytes / 4)` of the
system prompt plus the user message, and charges `generator_max_output_tokens`
on every call as an upper bound. It does not call the model. A real run
requires `--max-usd` and stops before a call whose estimate would cross the
cap. Throttling is retried with backoff. `ValidationException` is stored on
that row. `AccessDenied` stops the process. Five identical validation messages
in a row also stop the process.

## Randomness and metadata

Every script takes `--seed` (the config default is 0). The judge sample does
not use it. Each artifact stores the seed, the full config, the git commit,
model ids, and a UTC timestamp.
