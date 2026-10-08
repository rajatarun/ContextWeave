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
| Adjudication | `scripts/experiments/adjudicate.py` | yes, except `--dry-run` |
| Signals | `scripts/experiments/signals.py` | judge only |
| Calibration | `scripts/experiments/calibrate.py` | no |
| Replay | `scripts/experiments/replay.py` | no |
| Analyses | `scripts/experiments/analyses.py` | no |
| Subset | `scripts/experiments/subset.py` | Haiku and Nova only |
| Study | `scripts/experiments/study.py` | no |
| Projection | `scripts/experiments/project_cost.py` | no |
| Runner | `scripts/experiments/run_all.py` | the model stages |
| Tables | `scripts/experiments/write_results.py` | no |

Commands, in order, are in `RUNBOOK.md`.

## Datasets and the shared passage collection

Each dataset has one passage collection: every context paragraph in the dev
split. Every question and every retrieval arm ranks that same collection.
The collection is the full split. It does not shrink when the question sample
does. Sampling writes `results/samples/{dataset}.jsonl` (questions) and
`results/samples/{dataset}.passages.jsonl` (`id`, `title`, `text`). A
manifest that is not schema version 2 is refused. v1 per-question pools and
v1 generation rows are not reused.

Question type is the dataset name (`squad`, `hotpot`, `nq`). That removes the
answerable/unanswerable split from the routing type.

* **SQuAD 2.0** validation (`rajpurkar/squad_v2`). One passage per distinct
  `(title, context)`. The question's source passage is that context paragraph.
  An empty gold-answer list is unanswerable.
* **HotpotQA** distractor validation (`hotpotqa/hotpot_qa`, config
  `distractor`). One passage per distinct paragraph text. The source passages
  are the supporting-fact paragraphs. `bridge` and `comparison` stay on
  `hotpot_type`. A gold answer of yes or no sets `yes_no`. The sample
  manifest reports `n_yes_no` for the drawn questions. Yes/no questions stay
  in the sample.
* **Natural Questions**, tractable form: the MRQA 2019 in-domain dev file
  `NaturalQuestionsShort.jsonl.gz` (md5 `c0347eebbca02d10d1b07b9a64efe61d`).
  Each row has gold short-answer strings and a Wikipedia context that MRQA
  truncated to the first 800 tokens, kept only when the short answer occurs
  in that window. The context is split into overlapping character windows
  (`nq_window_chars` / `nq_overlap_chars`), and every window in the file is
  in the collection. The source passages are the windows that contain a gold
  answer string (case-sensitive containment). An empty gold list, or a gold
  string that appears in no window, stops the sample. `nq_pool_size` is
  unused by this collection.

The sample is a seeded shuffle. Questions are sorted by qid, then
`random.Random(seed)` shuffles those positions, and the sample is the first
`--n-per-dataset` of that order (config default 1100). The files are written
sorted by qid, so file order is not shuffle order. A smaller n with the same
seed is that prefix: 1100 is the first 1100 of the 1200 draw, and 1200 is
the first 1200 of the 1500 draw. Re-running
`sample_datasets.py` at that smaller n, over a schema-version-2 sample
already on disk, keeps the previously written question rows for the prefix
and keeps the passage file when it matches the collection just built.
`retrieve.py` skips `(qid, arm)` rows it has already written. The seed is
stored on `results/samples/sample_manifest.json` (`seed`, and again under
`sampling`). Fewer questions than requested is an error. The script does not
pad. A same-seed run replaces a committed sample only when the question ids
are unchanged or are this prefix, and only when the shared passage file is
present and unchanged. Generation, adjudication, signals, calibration, replay,
analyses, and retrieval stats read only question ids in `results/samples/`.

## Retrieval arms

All four arms rank the shared collection. `top_k` (default 5) passages are
stored with id, text, `score`, and `raw_score`.

`score` is a per-query min-max of that arm's raw scores over the collection,
so each arm's stored scores sit on [0, 1]. Equal raw scores store 0.5. Rank
order follows the raw score, then passage id. Min-max does not reorder an
arm. The row records `score_normalization: per_query_minmax`.

* `semantic_search`: cosine similarity with
  `sentence-transformers/all-MiniLM-L6-v2` on CPU. The artifact records the
  snapshot revision: `config._commit_hash` when the library sets it, otherwise
  the commit id in the local tokenizer path (`snapshots/<commit>/`).
* `graph_first`: one bipartite graph on the shared collection. Passage
  entities come from spaCy NER (`spacy_model`, default `en_core_web_sm`). A
  missing install or a missing model stops retrieval. An entity key is the
  surface form, casefolded, with whitespace collapsed. Document frequency is
  the number of passages that contain the key. An undirected edge joins a
  passage node and an entity node with weight `1/df`. A query seed is a spaCy
  entity in the question whose key is in the graph, or a content token of the
  question that equals an entity key. Seeds split the personalization mass
  evenly. The passage score is personalized PageRank (power iteration,
  damping `graph_damping`, iteration cap `graph_max_iter`, tolerance
  `graph_tol`). The iteration is deterministic and the retrieval artifact
  records `seed: null` for it. When the question matches no entity, the graph
  has no edges, or every passage mass is zero, the arm's raw scores are the
  vector cosines and the row records `graph_score_source: vector_fallback`.
  Otherwise it records `pagerank`.
* `keyword_boosted`: Okapi BM25 (`k1=1.5`, `b=0.75`), then the deployed blend
  from `retriever._keyword_boost_rerank`: keyword weight 0.25 and vector
  weight 0.75, after min-max normalising both component scores. The stored
  score is a second min-max of that blend.
* `hybrid`: the mean of the min-max normalised vector, graph, and BM25
  scores, then a second min-max for storage. This is the local stand-in for
  the deployed hybrid arm, which mixes vector search, the graph, and a
  keyword boost.

Each arm row also stores the retrieval labels in `experiments/labels.py`.
`gold_in_top_k` is true when at least one source passage id is in the top-k
given to the generator. `source_retrieved` is true when every source passage
id is in that top-k. SQuAD has one source paragraph, so the two flags agree.
Hotpot supporting-fact paragraphs and a multi-window NQ context can differ.

## Correctness

Token F1 uses the SQuAD normalisation in `scripts/verified_reward_bench.py`.
`token_f1_correct` on the joined row is 1 when that F1 is at least 0.5.
An unanswerable question scores token F1 1 exactly when the answer abstains.
Abstention is `verified_reward_bench.is_abstention`: the normalised answer is
empty, or it matches `verified_reward._ABSTAIN_RE` (phrases such as
"insufficient evidence", "do not contain", "no information", "not mentioned",
and "cannot answer" that the claim splitter already drops).

The joined `correct` field is the label replay, calibration, and analyses
use. An `ok` or `omitted` abstention follows the abstention label below.
`token_f1_correct` stays on the row beside it. A failed or truncated call
keeps a null abstention label, and `correct` follows token F1.

A second rate is stored beside the token-F1 label: whether the normalised
gold string is contained in the normalised answer. It is for answers that
quote the span and then keep writing. It is not a correctness label. The
prompt asks for a short extractive span (`yes` or `no` on a yes/no question)
and, when the passages do not contain the answer, for the object
`{"answer": "insufficient evidence", "claim": "The passages do not contain the answer.", "confidence": <0-1>}`.

### Abstention labels

`experiments/labels.py` labels an abstention on the generation row
(`abstention_label`, `abstention_counts_correct`). Only a call whose parse
status is `ok` or `omitted`, and whose answer is an abstention, is labeled.
A failed call and a truncated call leave both fields null.

An abstention counts as correct only when two things are both true: every
source passage was in the top-k given to the generator (`source_retrieved`),
and the question is unanswerable from those passages. The label is
`abstain_correct` and `abstention_counts_correct` is true. For SQuAD the
source is the one context paragraph. For Hotpot it is every supporting-fact
paragraph. For Natural Questions it is every window that contains a gold
answer string.

When any source passage is absent from that top-k, the label is
`abstain_retrieval_miss` and `abstention_counts_correct` is false. The
generator did not see the paragraph the question was asked against, so the
refusal is a retrieval miss. Analysis reports this label on its own. It is
not credited as a correct abstention.

When every source passage is in the top-k and the question is answerable, the
label is `abstain_incorrect` and `abstention_counts_correct` is false. The
answer was in the passages the generator saw.

A non-abstention has a null abstention label. When its token F1 is outside
`[v2.adjudication_f1_low, v2.adjudication_f1_high]` (0.2 and 0.8, both ends
included), `correct` is `token_f1_correct`. When token F1 is inside that
band, `correct` is the adjudicator's 0/1 label. The joined row also stores
`gold_in_top_k`, `source_retrieved`, `yes_no`, and `correct_source`
(`abstention`, `token_f1`, or `adjudication`). `source_retrieved` true with
`gold_in_top_k` false stops the join. A band row with no adjudication row
stops the join. A question with an arm still listed in
`results/adjudication/pending.json` is left out of the join until that arm
is written.

`scripts/experiments/adjudicate.py` calls `openai.gpt-oss-120b-1:0` through
Bedrock Converse, on demand, at temperature 0. The run seed is stored on the
artifact and on every row. Each row stores the system prompt, the user
prompt, and the raw reply. The spend cap is checked before the call, on the
shared ledger, under stage `adjudicate`. `--dry-run` prices an upper bound
with `ceil(utf-8 bytes / 4)` and `adjudicator_max_output_tokens` and does not
call the model. A reply that is not `{"correct": true}` or
`{"correct": false}` is stored as `unparseable` and the script exits
non-zero. The join then stops on that row. Access denied writes
`results/adjudication/adjudicator_unavailable.json` and exits non-zero
without inventing a label.

HotpotQA yes/no questions stay in the hotpot sample. Analyses writes
`datasets.hotpot.yes_no` with a `yes_no` slice and an `other` slice. Each
slice has the joined-correct rate, the token-F1 rate, `gold_in_top_k`,
`source_retrieved`, and the three abstention counts. `write_results.py`
prints that block under its own heading.

## Self-confidence

The generator is asked for JSON `{"answer", "claim", "confidence"}`. The
prompt is `experiments/prompts/generator_system.txt` and is copied into the
generation artifact. The saved answer is the `answer` field of the first JSON
object in the reply. It is never the raw reply. `claim` is the `claim` field
of that object: one sentence the model asserts. Lexical grounding and NLI
score that stored claim. A missing `claim` field is null. An empty string is
an empty claim. The claim stays that field. The confidence
sentence asks for the model's own probability and leaves the number
unassigned, so the logged self signal stays unshaped, as the deployed
synthesizer does when it asks for a confidence in [0, 1].

The robust reading (the `self` signal) is that first JSON object. Prose
before or after it is ignored. A number in [0, 1] is an observation. Anything
else stores a null robust value. A reply whose first line is
`insufficient evidence` and which contains no JSON object is that abstention:
the saved answer is the phrase, the robust value is null, and the status is
`omitted`. That is the model leaving the confidence out, so the missingness
reason is `omitted`.

The deployed reading is the strict whole-reply parse in
`synthesizer._confidence_from`, stored as `deployed_self_status` and
`deployed_self_confidence`. The fallback replay row uses that reading:

| Deployed status | When | Stored number | Fallback row |
| --- | --- | --- | --- |
| `ok` | the whole reply is a JSON object with a number in [0, 1] | that number | that number |
| `omitted` | JSON without a usable confidence | 0.7 | 0.7 |
| `unparseable` | the whole reply is not one JSON object | 0.5 | 0.5 |
| `failed` | the call failed | 0.0 | 0.0 |

A reply that is prose around a valid object is robust `ok` and deployed
`unparseable`. The `self` signal uses the parsed confidence. The fallback
row uses 0.5, which is what the deployed parser would have substituted.
When the output stops on `maxTokens` after a complete JSON object,
`trailing_truncated` is true and the robust status stays `ok` (or `omitted`
if that object has no usable confidence). The deployed strict parse is still
stored beside it. Only a JSON object cut off mid-token sets both statuses to
`truncated` and both values to null. That is not one of the three fallback
constants.

## Grounding, NLI, judge

Lexical grounding scores the stored claim with `verified_reward.lexical_support`
and tau 0.6. `split_claims` keeps sentences with at least three content
tokens and drops abstentions, so an abstention claim is `no_claims` and a
null claim is `no_claim`. A sentence is supported when some single retrieved
passage reaches tau, or when some pair of retrieved passages, joined by a
newline, reaches tau. Any number in the claim must appear verbatim in that
text. The signal is the fraction of sentences supported. The row records how
many were supported by a single passage and how many by a pair. It is missing
when there are no passages or no checkable sentences.

NLI uses the same claim, the same tau, and the same single-or-pair support
rule. The score of a text is the entailment-class probability from
`cross-encoder/nli-deberta-v3-small`, after the text is split into sentence
windows. A window plus the claim plus `nli_special_tokens` (3, for CLS, SEP,
and SEP) must fit in `nli_max_tokens` (512), or in the model's own maximum
when that maximum is smaller. A sentence that does not fit is cut to the
budget with the model's tokenizer, and the cut is counted. The row stores
`n_truncated_windows`. The NLI artifact sums those counts
(`n_truncated_windows`, `n_rows_with_truncation`) and records the limit that
was used. The model revision the library reports is stored beside them.

The judge prompt is `verified_reward.JUDGE_PROMPT`, copied verbatim into the
judge artifact. It grades the answer, which is the field the deployed prompt
names. A score of 1.0 means every claim is supported and the question is
answered. The same prompt scores 1.0 for an answer that says the evidence is
insufficient when the evidence is insufficient. The score is that grounding
judgement. It is not token-F1 correctness. Calibration pairs the stored score
with token F1 as recorded, with no adjustment when the judge scores an
abstention 1.0 and correctness is 0 because a gold answer existed. The judge
model defaults to `us.meta.llama3-3-70b-instruct-v1:0`. The run stops if that
id equals the generator id unless `--allow-same-judge` is passed, and the
flag is stored either way.

The experiment sample is a seeded prefix at `v2.judge_sample_rate` (0.60).
Question ids are sorted, `random.Random(stream_seed)` shuffles them, and the
sample is the first `take_count(n, rate)` of that order. `take_count` rounds
`n * rate` half up. The stream seed is SHA-256 of `{seed}:judge`, so the
holdout draw below is a different shuffle. The artifact stores the seed, the
stream name, the stream seed, and the sampled ids. A higher rate with the
same seed keeps the smaller sample as a prefix. Every arm of a sampled
question is judged. Other questions are written with reason `not_sampled`.
`judge_sampled` remains the deployed hash sample (`verified_reward.should_judge`)
and does not use the run seed. The deployed router rate stays
`judge_sample_rate` 0.05.

`--inference-mode on_demand` calls Converse per sampled row.
`--inference-mode batch` writes InvokeModel JSONL and calls
`create_model_invocation_job`, reusing `experiments/bedrock_batch.py`. The
projected batch cost is checked before the job is created. A job under
`batch.min_records` is not submitted and is not padded. Evidence longer than
12,000 characters is cut, and the row records `evidence_truncated`.

## Self percentile, logistic, oracles

`self_percentile` is the fraction of other ok confidences in the same dataset
that fall below this row, with ties counting half. A non-ok status is stored
with that status as the reason. A dataset with no other ok confidence is
`no_peers`. The value is a function of the whole log, so a file that does not
already match the current rows is refused.

`logistic` fits one model per dataset on the questions outside a seeded 20%
holdout (`v2.held_out_fraction`, stream `{seed}:holdout`). The training label
is token-F1 correctness. The features are `self` (ok confidence), `lexical`,
and `nli`, in that order, the three signals present on every complete row.
Holdout rows get the
predicted probability. Fit-fold rows are stored with reason `fit_fold` and a
null value. A row missing a feature is `feature_missing` and is left out of
the fit. One class, fewer rows than coefficients, or collinear features stops
the stage. The artifact stores the seed, the holdout ids, and the
coefficients. There is no random step inside the fit.

`oracle` writes two files and does not call a model. `oracle_correct` is the
token-F1 label. Joined `correct` applies the abstention rule and, inside the
F1 band, the adjudicator, on top of that bit.
`oracle_retrieval` is 1 when `source_retrieved` is true. A retrieval row
without that field stops the stage. The row also keeps `gold_in_top_k` so a
partial source hit can be counted later.

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

`self_with_fallbacks` substitutes 0.7 / 0.5 / 0.0 from the deployed strict
status (omitted / unparseable / failed). `oracle` uses the joined `correct`
bit: the abstention label, the adjudicator inside the F1 band, and token F1
outside it. The gold-contained rate stays a sensitivity check. The file
`signals/oracle_correct.jsonl` remains the token-F1 oracle written before
adjudication.

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

Prices are USD per million tokens in `config.yaml`, with the source they were
read from. On-demand rates live in `prices_usd_per_million_tokens`. Batch
rates live in `batch_prices_usd_per_million_tokens`. A model id that is not
in the table for the row's pricing stops the run. Generation and the judge
write each new ledger row's `usd` from that price and the call's token
counts, and the row records `pricing` (`on_demand` or `batch`) and
`price_usd_per_million`. A row with no `pricing` field is on-demand.
`scripts/experiments/reprice_ledger.py` rewrites an existing ledger the same
way (default `results/cost_ledger.jsonl.gz`, or `--ledger`). It keeps the
previous `usd` as `usd_at_logged_price`, records the config price and a
timestamp, and replaces the file by rename. A row without token counts is an
error. The spend cap sums `usd`, so it tracks the repriced bill.
`--dry-run` estimates input tokens as `ceil(utf-8 bytes / 4)` of the system
prompt plus the user message, and charges `generator_max_output_tokens` on
every call as an upper bound. It does not call the model. `usd_upper_bound`
follows `inference_mode`. The artifact also stores
`on_demand_usd_upper_bound` and `batch_usd_upper_bound`. A real run requires
`--max-usd` and stops before a call, or before a batch job, whose estimate
would cross the cap.

`--inference-mode on_demand` (the config default) calls Bedrock Converse once
per row. `--inference-mode batch` writes Bedrock batch JSONL (`recordId` and
`modelInput` in that model's InvokeModel body), uploads it, and calls
`create_model_invocation_job`. The role ARN and bucket come from
`batch.role_arn` / `batch.bucket` or from the environment variables named in
the config. A missing one stops the run. The projected batch cost is checked
against the cap before the job is created. A job with fewer than
`batch.min_records` rows is not submitted and is not padded, and the run does
not switch to on-demand on its own. The state file under
`results/generation/batch/` is written before upload. A later run with the
same input polls the stored job ARN. A successful output line with no token
counts stops the run. An error line with no usage stores zero tokens and
`usage_observed: false`.
Throttling is retried with backoff. `ValidationException` is stored on that
row. `AccessDenied` stops the process. Five identical validation messages in
a row also stop the process.

## Subset, second generator, and the study

`subset.py select` draws `v2.subset_questions` (250) question ids per dataset.
The shuffle stream is `subset:{dataset}`. A dataset with fewer questions than
that stops the run.

`subset.py haiku` draws `v2.subset_haiku_samples` (5) answers at
`v2.subset_temperature` (1) with the generator model. Each sample has its own
file under `results/subset/haiku/sample_{i}/`. The run seed and the sample
index are stored on `meta.json`. The request itself has no seed parameter.
`subset.py agreement` writes `results/subset/agreement.json`. The modal
fraction uses SQuAD normalisation. A question that is missing any sample
leaves the means null.

`subset.py nova` calls `subset_generator_model_id`
(`us.amazon.nova-pro-v1:0`) once per subset question and arm, at temperature
0. `--inference-mode batch` and `--inference-mode on_demand` both go through
the generation stage. A batch smaller than `batch.min_records` is not
submitted and is not padded.

`study.py` reads the joined log and writes `results/study/study.json`. It
records `s^2 / Var(R)`, `s^2 / (m(1-m))`, and information per round
(`s^2` over the residual variance). `m` is the mean of joined correct on rows
where the reward was observed. Beta and Gaussian Thompson sampling run for
`v2.replay_seeds` seeds and `v2.replay_rounds` rounds, with drift discounts
`v2.drift_discounts`. Judge coverage uses `v2.judge_coverage` and recomputes
the verified reward. The assumption check is the same cluster-bootstrap
comparison the analyses stage writes. A missing subset, agreement, judge, or
Nova file becomes a null block with a reason. The short replay in
`replay.py` keeps the top-level `replay_seeds` list.

`project_cost.py` prints call counts and on-demand and batch USD from the
config prices before any model call. Input tokens are the prompt template
with a placeholder question and no passage text. Output tokens are the
configured maximum. Adjudication calls stay pending until the token-F1 band
has been counted. `--write` stores `results/cost_projection.json`.

`run_all.py` prints that projection, then runs the stages in order.
`--project-only` stops after the print. A real run requires `--max-usd`.

`write_results.py` copies these artifacts into `results/FINDINGS.md`. A
missing file or a null estimate is the cell text `pending`.

## Randomness and metadata

Every script takes `--seed` (the config default is 0). The judge sample does
not use it. Each artifact stores the seed, the full config, the git commit,
model ids, and a UTC timestamp.
