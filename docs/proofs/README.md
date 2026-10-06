# Proof note

`tightened_claims.tex` is a standalone note. It does not replace the paper. The statements below use the paper's numbering (one counter per environment).

| Note | Paper | Status |
|---|---|---|
| Assumption 1, recalled | Assumption 1 | Unchanged. Mean of the reward given correctness does not depend on the arm, on rounds where the reward is observed. |
| Lemma 1, recalled | Lemma 1 | Same identity \(r_a = c_0 + s\mu_a\). The note adds the hypothesis that the probability of being observed does not depend on \(Y\) given the arm, which is automatic when the reward is almost surely observed and is the skipping clause of Proposition 2 otherwise. |
| Definition of \(\rho\), Assumption (arm enters only through \(W\)) | Paragraph defining \(\rho\) above Theorem 1 | \(\rho\) is the probability that a wrong answer draws its confidence from the correct-answer law. The note lifts that mixture from means to laws and allows any garbling of the honest-wrong bit, including a per-arm re-encoding. |
| Theorem (scale-free form of Theorem 1) and the \(\ln T/(1-\rho)\) corollary | Theorem 1 | Proved. Lower bound of Lai–Robbins type for every strongly consistent policy. The constant is \(\sum\Delta_a/\mathrm{kl}(\alpha_a,\alpha_{a^\star})\) with \(\alpha=(1-\mu)(1-\rho)\), and is at least \(\sum\mu_{a^\star}(1-\mu_{a^\star})/((1-\rho)\Delta_a)\). No dependence on the numeric gap \(c_1-c_0'\). |
| Proposition (where the power 2 holds) | Theorem 1, display with \(r(1-r)/(s^2\Delta)\) | Proved only when the observation is a Bernoulli (trick) reward whose mean stays in a subinterval of \((0,1)\) that does not shrink with \(\rho\), and whose mean gap is at most \((1-\rho)(c_1-c_0')\Delta\). This is the paper's bound. It is scale-dependent. |
| Remark on the power of \(1-\rho\) | Theorem 1 | The power 1 is what the honest-wrong channel forces. The power 2 is not true for every encoding: the bit \(W\) itself makes the paper's formula \(\Theta(1/(1-\rho))\). KL-UCB on that bit matches the power-1 constant. A universal \(\Omega(\ln T/(1-\rho)^2)\) claim is false, not left open. |
| Proposition (\(\rho=1\)) | Theorem 1, third claim | Proved, by the same relabeling argument. |
| Conjecture (fractional Beta update) | Remark under Theorem 1 comparing the fractional update to the Bernoulli trick | Not proved. Open only when the reward takes values in \((0,1)\). The missing step is a regret analysis of \(\alpha\mathrel{+}=R\), \(\beta\mathrel{+}=1-R\). The Bernoulli trick is covered by the existing Thompson-sampling bound. |
| Proposition (corrected Proposition 3), parts 1–2 | Proposition 3 | Proved under two added hypotheses: each signal's conditional mean given the mask does not depend on the arm, and the mask is independent of the arm given \(Y\). Given the mask, the slope is the renormalized weighted average and is positive. The marginalized reward still satisfies Assumption 1. |
| Proposition (corrected Proposition 3), part 3 | Proposition 3, slope formula | Proved only with the further assumption that the mask is independent of \(Y\). Without it the marginal slope can be negative; see the remark immediately after the proposition. |
| Remark (arm-dependent coverage) and Proposition (inverse propensity) | Proposition 3, interaction with Proposition 2 | Proved counterexample: arm-dependent masks reverse the ordering while every signal keeps a positive arm-free slope. Single-signal coverage that depends on the arm but not on \(Y\) does not reverse a skipped reward (Proposition 2, skipping). Filling with a constant can reverse it (Proposition 2). Known observation propensities, bounded away from zero, restore \(c_0+\bar s\mu_a\) by inverse-propensity weighting when the mask is independent of correctness given the arm. |

Assumptions added beyond the paper:

- The self-signal is a garbling of an honest-wrong bit with arm-independent conditional law (scale-free Theorem 1).
- Strong consistency: correctness regret \(o(T^b)\) for every \(b>0\), on every correctness vector in \((0,1)^K\).
- For the square-power corollary, a variance floor \(\delta\) independent of \(\rho\).
- For Proposition 3, mask-conditional means, \(S\perp\mathrm{arm}\mid Y\), and, for the marginal slope formula, \(S\perp Y\).
- For the inverse-propensity statement, propensities known and bounded away from zero, and missingness independent of \((Y,V)\) given the arm.

`proposed_edits.md` pairs each paper statement with the replacement proposed for the author.
