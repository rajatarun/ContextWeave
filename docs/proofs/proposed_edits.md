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
