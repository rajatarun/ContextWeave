# What the numbers mean: confidence semantics across the weave systems

Three systems on this account each emit a number in `[0, 1]` and call it, or
treat it as, a confidence:

- **mcp-observatory** scores a proposed tool call with a composite risk `s`.
- **ContextWeave** routes retrieval on a Beta posterior and rewards it with the
  synthesiser's self-reported `confidence`.
- **DeviceWeave** fuses an embedding similarity with a behaviour score into a
  `final_score` and executes above a threshold.

These numbers are not the same kind of thing, they are not comparable, and
several of them are not probabilities. This document states, from the code,
what each one is; which are safe to threshold, average, or multiply; and what
would have to be true before any two of them could be combined. It ends with
a proposed common envelope and a list of concrete defects found while
writing it.

Evidence weight for the RAG index: this file is authoritative about the
semantics of the three systems' scores as of the commit that adds it.

---

## 1. Inventory

| System | Symbol / field | Range | What it actually is | Compared against | Updated by |
|---|---|---|---|---|---|
| mcp-observatory | `composite_risk_score` `s` | `[0,1]` or undefined | Weighted mean of *risk indicators* over the defined set `D` (`risk/scoring.py`) | `0.20` / `0.35` per criticality (`policy/engine.py`) | nothing — static weights |
| mcp-observatory | `signals_defined` `|D|` | `0..6` | How many indicators were computable | `min_signals_{high,medium}` | nothing |
| ContextWeave | `QueryResponse.confidence` | `[0,1]` | The **model's own** number, parsed from its JSON output (`synthesizer.py`) | `0.5` cache-write gate | nothing — it *is* the reward |
| ContextWeave | `EFFECTIVE_FOR` posterior `Beta(α,β)`, mean `weight` | `(0,1)` | Posterior over the *expected self-confidence* of a strategy for a question type (`rag_router.py`) | other arms, by sampling | `α += w·c`, `β += w·(1−c)` from self-confidence (w=1) and human rating (w=2) |
| ContextWeave | `selectionPropensity` | `(0,1]` | **A genuine probability**: P(this arm was chosen) under Thompson sampling at decision time | nothing (logged) | — |
| ContextWeave | `POST /feedback` reward | `{0, 0.5, 1}` or `[0,1]` | Someone else's judgement of the answer (`feedback.py`) | — | folded into the posterior |
| DeviceWeave | `cosine_score` | `[−1,1]` in principle, `[0,1]` in practice | Embedding cosine between the utterance's device phrase and a device name (`device_resolver.py`) | — | nothing |
| DeviceWeave | `behavior_score` `b` | `[0,1]` | Time-of-day / frequency / weather blend; `0.5` when no history (`behavior_engine.py`) | — | history accumulates |
| DeviceWeave | `alpha` | `{0.9, 0.5}` | **A two-level step** on history count (`< 10` events → 0.9), not an anneal | — | history count |
| DeviceWeave | `final_score` | `[0,1]` | `α·cosine + (1−α)·b` (`decision_engine.compute_score`) | `CONFIDENCE_THRESHOLD = 0.4` (`app.py`) | nothing |

---

## 2. What each number is, and is not

### 2.1 mcp-observatory: `s` is a risk *indicator*, not a probability of harm

`s = Σ_{i∈D} w_i r_i / Σ_{i∈D} w_i`. The `r_i` are lexical proxies — Jaccard
disagreement between two generations, relative spread of extracted numbers,
presence of hedging words, a hash-changed bit. Their weights are hand-set.
Nothing in the pipeline ever compares `s` to an outcome, so `s = 0.35` does
not mean "35% chance this call is harmful"; it means "the weighted average of
these six indicators is 0.35". Two consequences:

- **Thresholding is a design choice, not a calibration.** `0.20` and `0.35`
  are defaults nobody has fit to data. Moving them changes review load and
  nothing is known about what it does to the miss rate.
- **`s` is comparable only at equal `|D|`** (proved as P1 in
  `mcp-observatory/docs/gate-properties.md`). A `0.05` from one signal and a
  `0.05` from six are different claims. The policy now consults `|D|`; any
  consumer of `s` outside the policy must too.

What `s` *is* good for: ordering calls by how much the cheap indicators
disagree with each other, and refusing to clear a critical call when few of
them could be computed at all. That is a useful gate. It is not a probability.

### 2.2 ContextWeave: three different numbers share one word

**(a) `confidence` on the response** is whatever the model wrote in the
`"confidence"` field of its JSON. It is self-report. Known to be miscalibrated
in general, and in this pipeline never checked against anything until the
feedback endpoint added a place to check it. Three fallbacks matter:

| Situation | Value that reaches the router | What it should be |
|---|---|---|
| Model omits the field | `0.7` | *no observation* |
| Model output is not JSON | `0.5` | *no observation* |
| Bedrock call fails | `0.0` | *no observation* — a pipeline error is not evidence about the strategy |

Every one of these currently **trains the router**. Under the old fixed-step
rule the `0.7` default sat exactly on the reinforce threshold, so a model that
forgot the field reinforced whatever strategy ran. Under Thompson sampling it
pulls the posterior toward 0.7. In all three cases the router is learning
from a constant the code chose, not from anything the model or the user said.
See defect D1 below.

**(b) The routing posterior** `Beta(α, β)` for a (strategy, question type)
pair is a posterior over the *mean of (a)* for that pair — i.e. over "how
confident does the synthesiser tend to say it is when this strategy retrieves
for this kind of question". Its mean is a calibrated estimate of *that*. It
is not an estimate of answer correctness, except insofar as (a) tracks
correctness, which is exactly the thing not established. The human rating
path (`POST /feedback`) is what lets the posterior start meaning "how useful
are the answers" rather than "how confident does the model sound".

**(c) `selectionPropensity`** is the one number in the whole inventory that
is an actual probability with a precise event: P(Thompson sampling selects
this arm | posteriors at decision time). It is estimated by Monte Carlo
(default 200 draws, s.e. ≤ 0.035). It may be used as a probability — divided
by, multiplied, used in an importance weight — and that is its only purpose.

### 2.3 DeviceWeave: a similarity and a frequency, averaged

`final_score = α · cosine + (1 − α) · b`, thresholded at `0.4`.

- `cosine` is a similarity between two embeddings. Its scale is a property of
  the embedding model: a "clearly right" match might sit at 0.8 for one model
  and 0.6 for another, and "clearly wrong" is not 0 but wherever unrelated
  phrases land — often 0.2–0.4. It is not a probability that the resolved
  device is the intended one; it has no notion of the *other* candidates.
- `b` is a blend of an hour-of-day match ratio, an action frequency ratio and
  a weather prior, with `0.5` meaning "no history". It is closer to a
  probability-like quantity than `cosine` is, but its neutral point is 0.5,
  whereas `cosine`'s neutral point is wherever unrelated text lands.
- `α` switches from 0.9 to 0.5 at 10 events. The paper describing this
  system calls it annealing; it is a step. With `α = 0.9` the threshold of
  `0.4` on `final` is effectively a threshold of about `0.39` on cosine alone
  (since `0.1 · b ≤ 0.1`); with `α = 0.5` a device that matched at cosine 0.3
  passes if the behaviour score is 0.5 — i.e. a *neutral* behaviour score
  can rescue a weak name match once there is history. Whether that is
  intended is not recorded.

Averaging a similarity with a frequency ratio produces a number in `[0, 1]`,
but its meaning changes with `α`, with the embedding model, and with the
device's history depth, and the single threshold `0.4` is applied to all of
those regimes. It is a workable heuristic; it is not a confidence in any
sense that supports combination with anything else. See defects D2, D3.

---

## 3. Rules for combining

The systems do not compose today. When they do — a tool call gated by
mcp-observatory that routes through ContextWeave for context and lands on a
DeviceWeave action is the obvious chain — these are the rules.

**R1. Only probabilities compose by probability rules.** Independent
conjunctive requirements multiply; redundant checks combine by noisy-OR;
posteriors update by Bayes. Of the inventory, only `selectionPropensity`
qualifies today. `s`, `final_score` and the synthesiser's `confidence` do
not.

**R2. Calibrate before you combine.** A score becomes a probability by being
mapped, monotonically, onto observed outcome frequencies. Each system needs
an outcome to calibrate against and a table pairing score with outcome:

| System | Outcome that would calibrate it | Where the pairs would come from | Status |
|---|---|---|---|
| ContextWeave self-confidence | human rating | `routing_decisions(confidence, rating)` | **table exists as of this branch; no data yet** |
| DeviceWeave `final_score` | user correction / undo within N seconds | not logged | missing |
| mcp-observatory `s` | reviewer verdict on REVIEW decisions; incident on ALLOW | not logged | missing |

Once pairs exist, fit isotonic regression (monotone, non-parametric, the
right choice for a score that is only claimed to be *ordered* correctly) and
publish the map alongside the score. Until then, the honest label on every
number except the propensity is *uncalibrated*.

**R3. Never average scores of different kinds.** `α·cosine + (1−α)·b` is
this rule being broken inside one system. Across systems it would be worse.
If two uncalibrated scores must be combined, combine *decisions* (both gates
must pass) rather than *numbers*.

**R4. Carry evidence count with the score.** `|D|` for mcp-observatory,
`α+β−prior` for the routing posterior, `total_events` for DeviceWeave. A
consumer that sees a score without its evidence count cannot tell a
confident estimate from a default.

**R5. Absence is not zero.** A signal that could not be computed is
undefined, not 0.0 (the mcp-observatory fix); a confidence the model did not
report is not 0.7 (D1 below); a behaviour score with no history is 0.5 only
because DeviceWeave chose 0.5 as its neutral, and that choice should be
visible to any consumer.

---

## 4. Proposed common envelope

A system that emits a confidence should emit this, not a bare float:

```json
{
  "value": 0.62,
  "kind": "score | similarity | probability | posterior_mean",
  "event": "P(selected arm) under Thompson sampling at decision time",
  "evidence": 37,
  "calibrated": false,
  "calibration_ref": null,
  "neutral": 0.5,
  "source": "self | human | model | measurement"
}
```

- `kind` says which combination rules apply (R1).
- `event` is the sentence the number is a probability *of*; required when
  `kind = probability`, so the claim can be checked.
- `evidence` carries R4.
- `calibrated` / `calibration_ref` carry R2: false until a fitted map exists,
  then a pointer to it.
- `neutral` carries R5: the value that means "no information" for this score.
- `source` distinguishes the model grading itself from anyone else.

No code implements this envelope yet. It is a contract for the unified
console to adopt; the point of writing it down is that today each system
emits a float and the console would otherwise have to guess.

---

## 5. Defects found while writing this

**D1 — ContextWeave trains its router on default constants.** When the model
omits `confidence` the reward is `0.7`; when its output is not JSON, `0.5`;
when Bedrock fails, `0.0`. None is an observation. Fix: mark the response
`confidence_reported = false` in those cases and skip both the routing
update and the decision record. (Fixed alongside this document.)

**D2 — DeviceWeave's α is documented as annealing and implemented as a step.**
`alpha = 0.9 if total_events < 10 else 0.5`. Either the documentation or the
code should change; the behaviour at the boundary (event 9 → event 10 swings
the cosine weight from 0.9 to 0.5 in one step) is the thing to decide about.
Not changed here; recorded for the DeviceWeave owner.

**D3 — DeviceWeave thresholds a mixture whose meaning depends on α.** The
same `0.4` applies to a 90/10 and a 50/50 blend of two different kinds of
score. The cleanest repair is to threshold the two components separately
(a minimum cosine for "we resolved the right device" and a minimum behaviour
score for "this action is plausible now") and combine the *decisions*, per
R3. Not changed here.

**D4 — mcp-observatory's thresholds have never met an outcome.** `0.20` and
`0.35` are defaults. The policy engine records its decision but nothing
records what happened next. A `review_verdict` column on the span is the
minimum needed to start calibrating (R2).

**D5 — the propensity estimate was costing 2.7 ms per request.** Measured
in `scripts/bench_router.py`; the comment in the code said "well under a
millisecond". Default draws reduced from 500 to 200 (≈1.1 ms, s.e. ≤ 0.035).
Still the largest gate-side cost in the routing step, and still three orders
of magnitude below the synthesis call it precedes.

---

## 6. What this document does not do

It does not make the three scores comparable; it explains why they are not
and what data would be needed. It does not implement the envelope. It does
not fix D2–D4, which belong to their owners. It fixes D1 and D5 because both
are in ContextWeave and both are small.
