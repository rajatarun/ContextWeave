# Runbook

Install once, from the repository root:

```bash
pip install -r experiments/requirements.txt
```

Every command is run from the repository root. `--seed` defaults to 0 and
`--n-per-dataset` defaults to 1500. A missing dataset file, price, or
credential stops the script with an error. Do not fill a missing stage with
stand-in model output.

## 1. Sample

```bash
python3 scripts/experiments/sample_datasets.py --seed 0 --n-per-dataset 1500
```

Writes `results/samples/{squad,hotpot,nq}.jsonl` and
`results/samples/sample_manifest.json`.

## 2. Retrieval

```bash
python3 scripts/experiments/retrieve.py --seed 0 --top-k 5
```

Writes one retrieval file per dataset, plus
`results/retrieval/retrieval_stats.json`. This does not call a hosted model.
Re-running skips `(qid, arm)` rows already in the file. A new file is
`.jsonl.gz`; an existing `.jsonl` is left as it is. Passing `--datasets nq`
keeps the other datasets' stat blocks.

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
```

Requires AWS credentials that can call Bedrock Converse in `us-east-1`.
The generator id is `us.anthropic.claude-haiku-4-5-20251001-v1:0` (override
with `--model-id`). Temperature is 0. `maxTokens` is 256: the reply is one
JSON object and nothing else. The answer is a few words, `yes` or `no` on a
yes/no question, or `{"answer": "insufficient evidence", "confidence": <0-1>}`
when the passages do not contain the answer. A complete JSON object whose
trailing prose hits the cap is robust status `ok` with `trailing_truncated`
set, and the confidence is kept. Only an object cut off mid-token is
`self_status` `truncated`, with a null confidence. A first line of
`insufficient evidence` and no JSON object is stored as that answer, robust
status `omitted`, and a null confidence. The deployed strict whole-reply
parse stays in `deployed_self_status` and is what the fallback row uses.
Rows already written are skipped.

Generation and the judge share `results/cost_ledger.jsonl` (gzipped when the
file is new). `--total-usd-cap` defaults to 30. The full run passes
`--total-usd-cap 29.80` because a 10-question smoke on this budget already
spent $0.19. `--max-usd 27` reserves the generation stage. The ledger is what
the judge reads, so a judge call is refused when generation spend plus that
call would cross $29.80. If generation stops under $27, the unused part of
the $29.80 can be given to the judge by raising its `--max-usd` up to
`29.80` minus the ledger total.

If a stage hits either cap it stops before the next call, keeps the rows it
already wrote, and lists the rest in `results/generation/pending.json` or
`results/signals/judge_pending.json`. That stop is not an error. A row named
there is pending, and later stages score only the completed rows.

## 4. Signals

Lexical grounding, then the local NLI model, then the judge. The judge is a
separate Bedrock model, `us.meta.llama3-3-70b-instruct-v1:0`.

```bash
python3 scripts/experiments/signals.py lexical --seed 0
python3 scripts/experiments/signals.py nli --seed 0
python3 scripts/experiments/signals.py judge --dry-run --seed 0
python3 scripts/experiments/signals.py judge --seed 0 --max-usd 3 --total-usd-cap 29.80
```

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
python3 scripts/experiments/calibrate.py --seed 0 --bootstrap 200 --results "$SMOKE"
python3 scripts/experiments/replay.py --seeds 0,1 --results "$SMOKE"
python3 scripts/experiments/analyses.py --seed 0 --bootstrap 200 --results "$SMOKE"
python3 scripts/experiments/write_results.py --seed 0 --results "$SMOKE"
```

`results/smoke/` is a local run. Keep it out of the published `results/`
tables unless the run used the real models and the full sample.

## Tests

```bash
python3 -m pytest tests/test_experiment_pipeline.py tests -q
```
