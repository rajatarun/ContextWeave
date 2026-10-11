# Proposed edits

For the author. Each block is a paraphrase of the current statement, then the statement proposed as a replacement. Nothing here is a patch against the paper source.

## Theorem 1

Current statement. Under Assumption 1 and the Bernoulli trick, with slope \(s>0\), the reward-optimal arm is the correctness-optimal arm, and every policy consistent on the Bernoulli family satisfies
\[
\liminf_{T\to\infty}\frac{\E[\Reg_Y(T)]}{\ln T}
\ge\sum_{a\neq a^\star}\frac{r_{a^\star}(1-r_{a^\star})}{s^2\Delta_a}.
\]
For self-confidence, \(s_{\mathrm{self}}=(1-\rho)(c_1-c_0')\). If \(s=0\), regret is linear up to relabeling.

This display is valid for that observation. The factor \(1/s^2\) still contains the encoding scale \(c_1-c_0'\). When \(r_{a^\star}(1-r_{a^\star})\) stays bounded away from \(0\) as \(\rho\to 1\), the bound is of order \(\ln T/(1-\rho)^2\). When the observation is the honest-wrong bit itself, the same display collapses to order \(\ln T/(1-\rho)\), because the Bernoulli variance shrinks like \(1-\rho\).

Proposed statement.

**Theorem (scale-free).** Let a wrong answer copy the correct-answer confidence law with probability \(\rho\in[0,1)\) and otherwise draw an honest-wrong bit \(W\). Let the learner observe any random element whose conditional law given \(W\) does not depend on the arm (any measurable per-arm or global re-encoding of the self-signal is included). For every strongly consistent policy,
\[
\liminf_{T\to\infty}\frac{\E[\Reg_Y(T)]}{\ln T}
\ge\sum_{a\neq a^\star}
\frac{\Delta_a}{\kl\big((1-\mu_a)(1-\rho),\,(1-\mu_{a^\star})(1-\rho)\big)}
\ge\sum_{a\neq a^\star}
\frac{\mu_{a^\star}(1-\mu_{a^\star})}{(1-\rho)\,\Delta_a}.
\]
The constant does not depend on \(c_1-c_0'\). Re-encoding cannot reduce it.

**Corollary (power 2, extra hypothesis).** If instead the learner observes only \(B\sim\mathrm{Bernoulli}(R)\) with mean gap at most \((1-\rho)(c_1-c_0')\Delta_a\) and with \(r_{a^\star}\in[\delta,1-\delta]\) for a \(\rho\)-independent \(\delta>0\), then the liminf is at least
\[
\sum_{a\neq a^\star}\frac{\delta(1-\delta)}{(1-\rho)^2(c_1-c_0')^2\Delta_a}.
\]
The original display is the special case \(\delta(1-\delta)=r_{a^\star}(1-r_{a^\star})\) and \(s=(1-\rho)(c_1-c_0')\). Do not state the square as a bound for every encoding of self-confidence.

Keep the \(s=0\) (equivalently \(\rho=1\)) linear-regret claim; the note proves it by relabeling.

Leave the matching upper bound for the fractional update \(\alpha\mathrel{+}=R\), \(\beta\mathrel{+}=1-R\) unproved when \(R\in(0,1)\). The Bernoulli trick is already covered by the published Thompson-sampling analysis. The note labels the fractional case a conjecture.

## Proposition 3

Current statement. If each signal satisfies Assumption 1 with slope \(s_i>0\), the normalized weighted average of the observed signals satisfies Assumption 1 with slope \(\sum_i w_i s_i/\sum_i w_i>0\).

The conditioning step in the proof is valid only given the mask, and only if that conditional mean does not depend on the arm. Independence of the mask and the arm given \(Y\) is an extra assumption. Positivity of the mask-averaged slope needs a further assumption: the mask law does not depend on \(Y\).

Proposed statement.

**Proposition.** Assume that for every signal and every mask \(s\) containing it, \(\E[V_i\mid Y=y,a,S=s]=c_{i,y}\) does not depend on the arm, with \(s_i=c_{i,1}-c_{i,0}>0\), and assume \(S\perp\mathrm{arm}\mid Y\).

1. Given \(S=s\neq\emptyset\), the combined reward has slope \(\sum_{i\in s}w_i s_i/\sum_{i\in s}w_i>0\) and does not depend on the arm.
2. The reward marginalized over the mask still satisfies Assumption 1.
3. If also \(S\perp Y\), the marginal slope equals the average of the mask-conditional slopes and is positive, so the reward-optimal arm is the correctness-optimal arm.

**Remark (counterexample, marginal slope).** \(S\perp\mathrm{arm}\mid Y\) does not force the marginal slope to be positive. Two signals with slopes \(0.1\) and \(0.05\), the first observed only on correct answers and the second only on incorrect answers, produce marginal slope \(-0.8\).

**Remark (counterexample, arm-dependent coverage).** Drop \(S\perp\mathrm{arm}\mid Y\). Let grounding have slope \(0.02\) and a second signal have slope \(1\), exactly one of them observed, independent of \(Y\) given the arm. If the better arm (\(\mu=0.9\)) shows grounding with probability \(0.99\) and the worse arm (\(\mu=0.7\)) shows the steep signal with probability \(0.99\), the combined means are approximately \(0.522\) and \(0.698\). The reward order is the reverse of the correctness order. Each signal alone would have preserved it.

**What survives.**

- One signal, skipped when missing, with \(\P(\mathrm{missing}\mid Y,a)=\P(\mathrm{missing}\mid a)\): the observed mean is still \(c_0+s\mu_a\). Arm-dependent coverage alone does not reverse a single skipped signal. This is the skipping half of Proposition 2.
- Replacing the missing value by a constant \(\kappa\) can reverse the order. This is the filling half of Proposition 2, and it is the case arm-dependent grounding coverage actually breaks if unobserved grounding is filled rather than skipped.
- If every signal has a known propensity \(\pi_a(i)=\P(i\in S\mid a)\ge\pi_{\min}>0\), and the mask is independent of \((Y,V)\) given the arm, the inverse-propensity reward
\[
\hat R=\frac{\sum_i w_i V_i\mathbf 1\{i\in S\}/\pi_a(i)}{\sum_i w_i}
\]
has mean \(c_0+\bar s\mu_a\) with \(\bar s=\sum_i w_i s_i/\sum_i w_i\). On the numerical example the corrected means are \(0.709\) and \(0.607\). These propensities are not the router's arm-selection propensities. If the mask depends on \(Y\), this weight is the wrong one, and the note does not claim a correction.

## Slope values used in the paper

Current statement. The numerical illustrations use the verified-reward simulation's assumed slopes, \(s_{\mathrm{self}}=0.024\) (\(\rho=0.7\), \(c_1=0.88\), \(c_0'=0.80\)) and \(s_{\mathrm{ground}}=0.5\). These are inputs to `scripts/verified_reward_bench.py`, not measurements, and the study contradicts them.

Proposed statement. Replace both with the measured slopes. Every number below is copied from `results/study/study.json` (\(s\), \(s^2/\mathrm{Var}(R)\)) and `results/analyses/analyses.json` (cluster-bootstrap 95% interval on \(s\), 1000 resamples over 1100 questions per dataset), and the same table is generated in `results/FINDINGS.md`, section "Measured slopes against the simulation's assumptions". \(s=c_1-c_0\) is the mean reward on correct answers minus the mean on wrong ones. The verified reward is lexical grounding plus the judge; the analyses stage stores no interval for it or for the pooled row.

Lead with the per-dataset table, and give the pooled row as a summary only: it mixes datasets whose base accuracy differs (0.71 SQuAD, 0.36 HotpotQA, 0.51 NQ). Reproduce the whole table. Quoting only the favourable cells would be cherry-picking.

| dataset | signal | \(s\) | 95% interval | \(s^2/\mathrm{Var}(R)\) |
|---|---|---|---|---|
| SQuAD | self-confidence | 0.0124 | [0.0072, 0.0180] | 0.0701 |
| SQuAD | lexical grounding | 0.0189 | [−0.0019, 0.0418] | 0.0270 |
| SQuAD | NLI grounding | 0.0137 | [−0.0653, 0.0881] | 0.0008 |
| SQuAD | judge | 0.0441 | [0.0209, 0.0713] | 0.0748 |
| SQuAD | verified | 0.0271 | — | 0.0527 |
| HotpotQA | self-confidence | 0.0003 | [−0.0082, 0.0089] | 0.000009 |
| HotpotQA | lexical grounding | 0.0323 | [0.0061, 0.0620] | 0.0589 |
| HotpotQA | NLI grounding | 0.0044 | [−0.0720, 0.0785] | 0.0001 |
| HotpotQA | judge | 0.0006 | [−0.0233, 0.0224] | 0.000008 |
| HotpotQA | verified | 0.0112 | — | 0.0054 |
| NQ | self-confidence | −0.0068 | [−0.0112, −0.0027] | 0.0173 |
| NQ | lexical grounding | −0.0042 | [−0.0132, 0.0054] | 0.0018 |
| NQ | NLI grounding | 0.0121 | [−0.0448, 0.0712] | 0.0009 |
| NQ | judge | 0.0066 | [−0.0081, 0.0235] | 0.0021 |
| NQ | verified | 0.0032 | — | 0.0010 |
| pooled | self-confidence | 0.0045 | — | 0.0033 |
| pooled | lexical grounding | 0.0132 | — | 0.0132 |
| pooled | NLI grounding | 0.0271 | — | 0.0032 |
| pooled | judge | 0.0166 | — | 0.0095 |
| pooled | verified | 0.0140 | — | 0.0124 |

**What supports the paper.**

- The verified slope is positive on every dataset and pooled. Self-confidence's is not: on NQ it is negative with an interval that excludes 0, so there higher confidence goes with worse answers.
- Pooled, the verified reward has 3.1 times the slope of self-confidence (0.0140 against 0.0045) and 3.7 times its \(s^2/\mathrm{Var}(R)\) (0.0124 against 0.0033).
- HotpotQA: self-confidence carries no signal (0.0003, interval spans 0). Lexical grounding's slope is 0.0323 with an interval that excludes 0. The verified slope is 0.0112.
- SQuAD: the verified slope is 2.2 times self-confidence's. The judge's slope is 0.0441 with an interval that excludes 0.

**What does not support the paper.**

- The assumed grounding slope 0.5 is off by more than an order of magnitude on every dataset (15 to 26 times lexical grounding where that slope is positive, about 38 times pooled). The assumed self slope 0.024 is also too large: it is 70 times the HotpotQA value, and on NQ the sign is wrong.
- On SQuAD and NQ self-confidence has the larger \(s^2/\mathrm{Var}(R)\) (0.070 against 0.053, and 0.0173 against 0.0010). Its reward variance is tiny (0.0022 and 0.0027 against 0.0140 and 0.0108), so the ratio is large while the slope is small (SQuAD) or negative (NQ).
- On SQuAD self-confidence's slope is positive with an interval that excludes 0. It is not uninformative there.
- The components of the verified reward are not reliably signed either. Lexical grounding is negative on NQ (−0.0042, interval spans 0). The judge is about 0 on HotpotQA (0.0006). NLI grounding's intervals span 0 on every dataset. Only two component cells have intervals that exclude 0: lexical grounding on HotpotQA and the judge on SQuAD.
- The pre-registered Section 6 prediction is contradicted on all three datasets (`results/calibration/calibration.json`). The self-confidence clauses hold everywhere: coverage at least 0.8, AUROC at most 0.6 (0.585, 0.490, 0.447), and a strategy ranking that disagrees with correctness. The grounding clause fails everywhere: lexical grounding's strategy ranking also disagrees with correctness, and its AUROC is about 0.5 (0.509, 0.516, 0.495). The paper should not say grounding ranks strategies correctly.
- Every slope is small in absolute terms. The largest is 0.044.

**Wording of the claim.** State that verified rewards carry a more reliable, correctly signed signal than self-confidence: the verified slope is positive on every dataset, while self-confidence's is about zero on HotpotQA and negative on NQ, and the verified slope is larger pooled. Do not state that they have large slopes, that they dominate self-confidence on \(s^2/\mathrm{Var}(R)\) on every dataset, or that grounding alone ranks strategies correctly.

**Effect on Theorem 1's illustration.** With the assumed values, the \(1/s^2\) factor makes self-confidence about \((0.5/0.024)^2\approx 434\) times more expensive than grounding. With the measured pooled values the slope-squared ratio of verified to self is \((0.0140/0.0045)^2\approx 9.8\) and the \(s^2/\mathrm{Var}(R)\) ratio is about 3.7. On SQuAD and NQ that ratio favours self-confidence. Any sentence that quotes the size of the gap should quote the measured numbers and give this per-dataset caveat.
