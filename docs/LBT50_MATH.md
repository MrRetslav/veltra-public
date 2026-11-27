# LBT-50: methods, formulas and what the measurements said

Each method was written down with the condition for dropping it **before** it was measured; the outcomes were added afterwards. "New" means new
to this project: no literature search was done, and related published work exists for parts of it (named from memory, not checked).

## 0. The equation behind 50 tok/s
Per accepted token the engine reads `B = (B_att + B_head + B_router) / A  +  k_eff * b_exp * (U / k) / A`, with `A` accepted tokens per pass and
`U` the union of experts per layer. At 28 GB/s, 50 tok/s needs `B <= 560 MB`. The starting point was `390 + 157 + 25 + 8 * 85.3 = 1,254 MB`
at `A = 1`. No single measure gets there; only a product of independent factors could: experts x bits per weight x output head x tokens per pass.

## 1. Two-tier ("shadow") experts
The router ranks the 8 experts. The first `R` run in full. Every further one runs a **shadow** of itself with only `s` of its 768 neurons and,
optionally, a recomputed output projection:

`f_e(x) = D_e h_e(x)`, `h_e(x) = silu(G_e x) * (U_e x)`

`D'_e = argmin_D  sum_t pi_e(x_t) ||D_e h_e(x_t) - D h_{e,S}(x_t)||^2 + lambda ||D - D_e[:,S]||^2`
`     = (Y H_S^T + lambda D_e[:,S]) (H_S H_S^T + lambda I)^-1`   (closed form, no gradient training)

The neuron set `S` comes from block-wise orthogonal matching pursuit. `moe-k` drops the tail experts completely; a shadow approximates them for
`s/768` of their bytes.
**Drop if** it is not better than the best `moe-k` at equal bytes. Related: neuron pruning of MoE experts, column subset selection.

**Outcome: not built.** Per layer it beat dropping experts, and the closed-form refit did not help over the original columns. Over all 48 layers
the best variant with at most 5.5 expert reads reached held-out KL 0.0383, uniform `--moe-k 6` (6.00 reads) 0.0350. At equal quality the saving is
about 3% of the bytes per token.

## 2. Learned codebooks that live in the cache (additive vector quantisation for Zen 2)
`w_{r,g} in R^4 ~ sigma_{r,b} * (C_1[i] + C_2[j])`, 16 vectors per codebook, shared by a block of 32 columns; 4 + 4 bits per 4 weights plus a
scale. The codebooks are learned in a space scaled by the 4x4 input covariance, so that the **output** error is minimised. On Zen 2 this is the
LUT kernel: one table of 16 inner products per codebook, looked up with `vpshufb`.
**Drop if** the output error at fewer bits is higher than that of the current weights. Related: AQLM, QuIP#, QTIP.

**Outcome: rejected** (held-out relative output error, layers 12 / 24 / 36):
| tensor | current weights | codebooks |
|---|---|---|
| gate | 2.625 bits: 0.040 / 0.035 / 0.035 | 2.25 bits: 0.067 / 0.059 / 0.059 |
| down | 3.56 bits: 0.019 / 0.021 / 0.021 | 3.25 bits: 0.036 / 0.039 / 0.039 |
This was one training recipe; a better optimiser could narrow the gap.

## 3. One-bit estimator with an error band for the output head
Rotate input and rows with `R = H diag(+-1)` (three rounds, nothing stored). Per row `o = R w / ||w||`, code `b = sign(o) / sqrt(D)`,
`c = <b, o>`. Estimate `<w, x> ~ ||w|| ||x|| <b, R x> / c` with an error band proportional to `sqrt((1 - c^2) / c^2) / sqrt(D - 1)`. Only rows
whose upper bound beats the best lower bound are re-scored exactly. First pass about 40 MB per token instead of 92 MB (2-bit screener).
**Drop if** the true argmax ever falls outside the survivors, or so many rows survive that the gain disappears. Related: RaBitQ.

**Outcome: not built** (500 held-out positions, 151,936 rows):
| band width | survivors, median / 99th percentile | true argmax missed |
|---|---|---|
| 1.9 | 105 / 8,743 | 1 of 500 |
| 3.0 | 2,778 / 70,809 | 0 |
| 4.0 | 25,069 / 135,633 | 0 |
The narrow band breaks the first condition. The safe-looking band has a heavy tail: on the worst 1% of tokens almost half of the head is re-scored.
Even in the median case the saving is only about 4% of the bytes per token.

## 4. Quick exclusion test for delta computation
Delta schemes read only the columns whose input changed: `y_t = y_{t-1} + W (h_t - h_{t-1})`. That needs a small
`||h_t - h_{t-1}|| / ||h_t||`.
**Outcome: rejected.** Median ratio 1.04 / 1.04 / 0.97 at layers 12 / 24 / 36.

## 5. Limits that follow from the maths
- Co-activation `P(e, f)` does not appear in the expected reconstruction error of a shared basis (the sum over chosen experts is linear).
- A separate `A B` factorisation per expert (768 x 2048) only saves bytes below rank 558.
- Skipping single zeros saves no bytes: with 33% zeros only 0.017% of the aligned 32-column blocks were entirely zero (measured).

## 6. The path to 50 that was proposed, and where it ended
Before the measurements the only path on paper was a product: 2-bit screened head, then shadow experts, codebooks and a one-bit head, then
lossless speculation on predictable text. The screened head exists and works. Shadow experts and codebooks did not pass their conditions, the
one-bit head is marginal, and a C++ EAGLE-3 drafter was dropped because a pass of 4 tokens costs 2.42 passes of 1 token. **On this machine no
measured route to 50 tok/s on ordinary text without quality loss remains.**
