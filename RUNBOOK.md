# Runbook

Install once, from the repository root:

```bash
pip install -r experiments/requirements.txt
```

Every command is run from the repository root. `--seed` defaults to 0 and
`--n-per-dataset` defaults to 1100. A missing dataset file, price, or
credential stops the script with an error. Do not fill a missing stage with
stand-in model output.

## 1. Sample

```bash
python3 scripts/experiments/sample_datasets.py --seed 0 --n-per-dataset 1100
```

Writes `results/samples/{squad,hotpot,nq}.jsonl`,
`results/samples/{squad,hotpot,nq}.passages.jsonl`, and
`results/samples/sample_manifest.json`. The question files are a seeded
prefix. The passage files are the full dev-split collection, shared by every
question and every arm, and they do not shrink with `n`. The draw sorts
question ids, shuffles those positions with `random.Random(seed)`, and keeps
the first `--n-per-dataset`. The question files are written sorted by qid.
1100 with seed 0 is the first 1100 of the 1200 draw, and that 1200 is the
first 1200 of the 1500 draw, all with seed 0. Question type is the dataset
name. The manifest is schema version 2. A manifest with any other version
is refused, and v1 per-question pools are not reused. Re-running at that
smaller n, with the same seed, over a schema-version-2 sample already on
disk keeps the previously written question rows for the prefix and keeps the
passage file, so retrieval for those questions stays valid. The seed is
stored on the manifest (`seed`, and again under `sampling`). Generation,
signals, calibration, replay, and analyses read only question ids in
`results/samples/`.

## 2. Retrieval

```bash
python3 scripts/experiments/retrieve.py --seed 0 --top-k 5
```

Writes one retrieval file per dataset, plus
`results/retrieval/retrieval_stats.json`. This does not call a hosted model.
Each arm ranks that dataset's shared passage file. Install the spaCy model
named by `spacy_model` first (`python -m spacy download en_core_web_sm`). A
missing model stops retrieval. The graph arm is spaCy entities, `1/df`
edges, and personalized PageRank, with the vector scores when the question
matches no entity. Stored scores are a per-query min-max;
`raw_score` keeps the pre-normalization value. Each row records
`gold_in_top_k` (any source passage in the top-k) and `source_retrieved`
(every source passage in the top-k). Re-running skips `(qid, arm)` rows
already in the file. Rows for question ids outside the current sample stay
in the file and are left out of the stats. A new file is `.jsonl.gz`; an
existing `.jsonl` is left as it is. Passing `--datasets nq` keeps the other
datasets' stat blocks.

## 3. Generation dry-run, then generation

```bash
python3 scripts/experiments/generate.py --dry-run --seed 0
```

Writes `results/generation/dry_run_cost.json`. The upper bound charges
`generator_max_output_tokens` on every call, so it can sit above the stage
budget below. The run still uses those caps and stops cleanly when the ledger
reaches one.

```bash
python3 scripts/experiments/generate.py --seed 0 --max-usd 27 --total-usd-cap 29.80
python3 scripts/experiments/generate.py --seed 0 --inference-mode batch --max-usd 27 --total-usd-cap 29.80
```

The default `--inference-mode` is `on_demand`. That path needs AWS
credentials that can call Bedrock Converse in `us-east-1`. Batch mode needs
credentials that can call `bedrock:CreateModelInvocationJob` and write the
bucket named by `BEDROCK_BATCH_BUCKET`, with the role ARN in
`BEDROCK_BATCH_ROLE_ARN` (or `batch.role_arn` / `batch.bucket` in the
config). A missing role or bucket stops the run. The projected batch cost is
checked before each job is created. A job smaller than `batch.min_records`
(100) is not submitted and is not padded. Pass `--inference-mode on_demand`
for a remainder below that minimum. A state file under
`results/generation/batch/` is written before upload, so a rerun polls the
stored job instead of creating another one.

The generator id is `us.anthropic.claude-haiku-4-5-20251001-v1:0` (override
with `--model-id`). Temperature is 0. `maxTokens` is 256: the reply is one
JSON object and nothing else. The answer is a few words, `yes` or `no` on a
yes/no question, or
`{"answer": "insufficient evidence", "claim": "The passages do not contain the answer.", "confidence": <0-1>}`
when the passages do not contain the answer. The `claim` field is stored on
the generation row. A missing claim is null. A complete JSON object whose
trailing prose hits the cap is robust status `ok` with `trailing_truncated`
set, and the confidence is kept. Only an object cut off mid-token is
`self_status` `truncated`, with a null confidence. A first line of
`insufficient evidence` and no JSON object is stored as that answer, robust
status `omitted`, and a null confidence. The deployed strict whole-reply
parse stays in `deployed_self_status` and is what the fallback row uses.
Rows already written are skipped. A generation file whose rows are not
schema version 2 is refused.

An abstention is `abstain_correct`, and counts as correct on
`abstention_counts_correct`, only when `source_retrieved` is true and the
question is unanswerable. When a source passage is missing from the top-k,
the label is `abstain_retrieval_miss` and it does not count as correct. When
the source was retrieved and the question is answerable, the label is
`abstain_incorrect`. The joined token-F1 `correct` field is unchanged.
Ledger rows record `pricing` so a batch call is billed from the batch table.

Generation and the judge share `results/cost_ledger.jsonl` (gzipped when the
file is new). Each new row's `usd` is the call's input and output token
counts at the current price in `experiments/config.yaml`. `--total-usd-cap`
defaults to 30. The full run passes `--total-usd-cap 29.80` because a
10-question smoke on this budget already spent $0.19. `--max-usd 27` reserves
the generation stage. The ledger is what the judge reads, so a judge call is
refused when generation spend plus that call would cross $29.80. If
generation stops under $27, the unused part of the $29.80 can be given to the
judge by raising its `--max-usd` up to `29.80` minus the ledger total.

After a price change, recompute the ledger so the cap matches the bill:

```bash
python3 scripts/experiments/reprice_ledger.py
python3 scripts/experiments/reprice_ledger.py --ledger results/cost_ledger.jsonl.gz
```

The default file is `results/cost_ledger.jsonl.gz`. Each row's `usd` is
rewritten from its stored token counts. The previous `usd` is kept as
`usd_at_logged_price`, and the row records the config price and a timestamp.
The script writes a temporary file in the same directory and renames it over
the ledger. A row without input and output token counts stops the script and
leaves the file unchanged.

If a stage hits either cap it stops before the next call, keeps the rows it
already wrote, and lists the rest in `results/generation/pending.json` or
`results/signals/judge_pending.json`. That stop is not an error. A row named
there is pending, and later stages score only the completed rows.

## 4. Signals

Lexical grounding and NLI score the stored claim against each passage and
against each pair. NLI splits those texts into sentence windows that fit
512 tokens and records every window it had to cut. Then the judge, then the
self percentile, the oracles, and the holdout logistic. The judge is a
separate Bedrock model, `us.meta.llama3-3-70b-instruct-v1:0`.

```bash
python3 scripts/experiments/signals.py lexical --seed 0
python3 scripts/experiments/signals.py nli --seed 0
python3 scripts/experiments/signals.py judge --dry-run --seed 0
python3 scripts/experiments/signals.py judge --seed 0 --max-usd 3 --total-usd-cap 29.80
python3 scripts/experiments/signals.py judge --seed 0 --inference-mode batch --max-usd 3 --total-usd-cap 29.80
python3 scripts/experiments/signals.py self_percentile --seed 0
python3 scripts/experiments/signals.py oracle --seed 0
python3 scripts/experiments/signals.py logistic --seed 0
```

The judge sample is a seeded 60% prefix (`v2.judge_sample_rate`). The logistic
holdout is a seeded 20% prefix on a different stream (`v2.held_out_fraction`).
Both seeds are the `--seed` value. Batch mode needs `BEDROCK_BATCH_ROLE_ARN`
and `BEDROCK_BATCH_BUCKET`. The projected cost is checked before the job is
created. A job smaller than `batch.min_records` is not submitted. Percentile
and logistic read the whole log; if their output file does not match the
current rows, move it aside and rerun. Logistic needs the lexical and NLI
files.

Lexical grounding and calibration, replay, analyses, and `write_results` read
only the committed sample, the retrieval files, and the generation output.
The judge calls Bedrock. The NLI stage downloads its pinned cross-encoder
from Hugging Face. Nothing else in those stages uses the network.

If the judge model returns access denied, the stage exits non-zero and writes
`results/signals/judge_unavailable.json`. Calibration, replay, analyses, and
`write_results` still run. Judge cells and verified-reward cells stay pending
with the reason `judge model access not yet granted`.

New raw outputs (generation, signals, and a retrieval file that does not
already exist) are written as `.jsonl.gz`. Readers accept either suffix.

The judge dry-run counts sampled calls. It does not price them, because the
prompt contains the generated answer and the dry-run does not invent one.
Pass `--allow-same-judge` only if you intentionally set the judge id equal to
the generator id. The flag is recorded in `results/signals/judge_meta.json`.

## 5. Calibration, replay, analyses, tables

These read the files above. If a signal file is missing they stop, and
`write_results.py` leaves the corresponding cells pending.

```bash
python3 scripts/experiments/calibrate.py --seed 0 --bootstrap 1000
python3 scripts/experiments/replay.py --seeds 0,1,2,3,4
python3 scripts/experiments/analyses.py --seed 0 --bootstrap 1000
python3 scripts/experiments/write_results.py --seed 0
```

Replay writes `results/replay/replay_summary.json`, a CSV of cumulative
regret per seed, and a PNG per dataset and update rule.
`results/FINDINGS.md` and `results/PENDING.md` are overwritten from the
artifacts.

## Smoke run

Use a fresh directory so the smoke rows are not appended to the full sample.

```bash
SMOKE=results/smoke
python3 scripts/experiments/sample_datasets.py --seed 0 --n-per-dataset 10 --results "$SMOKE"
python3 scripts/experiments/retrieve.py --seed 0 --results "$SMOKE"
python3 scripts/experiments/generate.py --dry-run --seed 0 --results "$SMOKE"
python3 scripts/experiments/generate.py --seed 0 --max-usd 1 --total-usd-cap 2 --results "$SMOKE"
python3 scripts/experiments/signals.py lexical --seed 0 --results "$SMOKE"
python3 scripts/experiments/signals.py nli --seed 0 --results "$SMOKE"
python3 scripts/experiments/signals.py judge --seed 0 --max-usd 1 --total-usd-cap 2 --results "$SMOKE"
python3 scripts/experiments/signals.py self_percentile --seed 0 --results "$SMOKE"
python3 scripts/experiments/signals.py oracle --seed 0 --results "$SMOKE"
python3 scripts/experiments/signals.py logistic --seed 0 --results "$SMOKE"
python3 scripts/experiments/calibrate.py --seed 0 --bootstrap 200 --results "$SMOKE"
python3 scripts/experiments/replay.py --seeds 0,1 --results "$SMOKE"
python3 scripts/experiments/analyses.py --seed 0 --bootstrap 200 --results "$SMOKE"
python3 scripts/experiments/write_results.py --seed 0 --results "$SMOKE"
```

`results/smoke/` is a local run. Keep it out of the published `results/`
tables unless the run used the real models and the full sample.

## Tests

```bash
python3 -m pytest tests/test_experiment_pipeline.py tests/test_experiment_v2.py tests/test_experiment_signals.py tests -q
```
