# LBT-50 research log

Goal: 50+ tok/s for Qwen3-30B-A3B class models on the existing machine (Ryzen 9 3900, 8 threads) without noticeable loss of quality.
Rules: only measured numbers count; a change must lower the bytes read by the memory layout, be faster in an interleaved A/B on the same box, and
keep the quality (160-task public set plus output-distribution checks). Experiments live in `benchmarks/apex_proto/`, `benchmarks/lbt50/` and
`engine/budget_probe.cpp`; the default decoder stays bit-identical. The calibration and held-out text of these experiments came from files of
the development repository, some of which are not published, so these tables cannot be reproduced exactly from the public files.

## Starting point (measured)
- About 1.24 GB of weights per decoded token; about 21 tok/s greedy on prose. 50 tok/s at 28-29 GB/s allows at most about 580 MB per token.
- Verdict of a first paper review, before the experiments: No-Go for 50+ tok/s on ordinary text on this machine.

## Correction to the first review
The "hard upper bound" for a shared basis in that review holds only for the **unweighted** Frobenius error. What matters is the error
on real activations, `||(W - W')x||^2`, whose spectrum can be much more favourable. `benchmarks/lbt50/matrices.py` measures that version on
held-out activations; it is the deciding test, not the unweighted one.

## Experiments and decision rules (written before the results)
| id | what | keep only if |
|---|---|---|
| E1 `lbt_verifycost` | cost of a verify pass of T = 1..6 tokens on the current build | decides E6: with C(4) near 2.4 an EAGLE-3 drafter gains almost nothing, near 1.75 about +35% on code/JSON |
| E2 `lbt_budget` | per-layer expert budget from output sensitivity vs uniform k with the same number of expert reads | lower held-out KL at equal reads, by a margin worth at least 5% bytes at equal KL |
| E3 `lbt_matrices` A | activation-weighted shared basis for gate+up with 4-bit coefficients | held-out output error at or below 2-bit quantisation with clearly fewer bytes |
| E3 `lbt_matrices` B | activation-weighted low rank of attention q, k, v | error at or below 3-bit at a parameter ratio well under 1 |
| E3 `lbt_matrices` C | certified skipping of 32-neuron blocks in down_proj (two neuron orders) | a certificate at 5-10% error allows skipping at least 20% of the blocks |
| E4 `apex_overlap` | union of routed experts over 1-5 consecutive tokens | feeds the byte model for any multi-token scheme |
| E5 `apex_mem`, `apex_kernels`, `apex_head` | bandwidth, pinning, prefetch; three 2-bit kernels; head time | a measured speedup above noise |
| E6 | EAGLE-3 draft head (int4) in C++ | only after E1 says the verify cost allows a gain |

## Results

### E1 verify-pass cost, current build (02:55, hf30, 8 threads, ms per forward of T real tokens, median of 7; identical with screen-head 1 and 2)
| T | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ms | 47.9 | 73.7 | 95.3 | 115.7 | 133.9 | 161.6 |
| x T=1 | 1.00 | 1.54 | 1.99 | 2.42 | 2.80 | 3.37 |
The byte model alone would give 1.30x at T=2 (U2 = 12.43); the measured 1.54x means multi-token passes are partly compute-bound.

**E6 EAGLE-3: NO-GO.** With the acceptance measured for this head on this target (earlier probe: tokens per pass at K=3 code 2.73, JSON 3.01,
Dutch 1.25) and ignoring the draft cost entirely, the best case is 2.73/2.42 = 1.13x (code), 1.24x (JSON), 0.52x (Dutch). With about 10 ms of
draft work per pass: 1.04x, 1.14x, loss. K=1 and K=2 end at 1.05-1.10x. This is days of engine work for at most +15% on some traffic.

### E4 expert unions over m consecutive tokens (teacher-forced, 400 tokens per text type, all 48 layers)
| m | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| mean U_m | 8.00 | 12.43 | 16.01 | 19.04 | 21.69 |
| MB per pass (572 + 85.3 U) | 1254 | 1632 | 1938 | 2196 | 2422 |
Neighbouring tokens share only slightly more experts than random pairs (U3 15.1-16.8 vs 17.4-20.6 shuffled). Any m-token scheme therefore needs
well above 1.3 / 1.5 / 1.75 accepted tokens per pass on bytes alone (m = 2 / 3 / 4), and more because of compute.

### E5 memory, kernels, head (02:45-02:52)
- Stand-alone read bandwidth: 22.2 GB/s with 1 thread, 31.8 with 2, 32.6-33.0 with 4-8: saturated by 2-4 threads. Pinning (4+4 over the CCDs,
  2x4 over the CCXs, one CCD with SMT) 31.9-33.5 and software prefetch 2-16 lines 32.5-33.1: inside the run-to-run spread. **In the engine pinning
  did not help** (unpinned 21.06 tok/s, pinned 20.55 and 20.66). Pinning and prefetch rejected. At 1.254 GB per token, 21.06 tok/s is 26.4 GB/s,
  about 80% of the stand-alone 33 GB/s.
- 2-bit kernels (exact, no scales), cycles per 32 weights in L2: maddubs 3.05, bit-plane 4.91, **lut32 1.49** (6.9 instead of 11.2 instructions).
  From DRAM on 1 thread: 37.7 / 22.8 / **54.8** Gweights/s. On 8 threads all three end at 126-129 Gweights/s (32 GB/s, memory-bound). The LUT
  layout would give compute headroom (multi-token passes, weaker CPUs), not single-token speed on this box. Bit-plane rejected.
- LM head, decode: unscreened int8 9.62 ms of 51.3 ms (18.8%, 19.52 tok/s); 4-bit screen 5.24 ms (11.4%, 21.50 tok/s); 2-bit screen 3.38 ms
  (7.6%, **22.30 tok/s, +3.7%**). Whatever replaces the head screener can gain at most about 3.4 ms (about 8%) over the 2-bit screen.
- Shared basis, **unweighted** bound on the HF weights (mid layers): relative squared error 0.30-0.35 at r = 1152 against 0.108 for plain 2-bit and
  0.020 for 3-bit; the engine's own Q2_K is about 0.08. Unweighted the idea is far off; the activation-weighted test (E3) decides.

### E2 per-layer expert budget (03:06, 900 fit and 900 held-out positions, 48 layers) — REJECTED
| expert reads (of 384) | uniform k: held-out KL / top-1 | per-layer budget: held-out KL / top-1 |
|---|---|---|
| 336 (k=7) | 0.0132 / 95.67% | 0.0207 / 94.56% |
| 288 (k=6) | 0.0350 / 92.89% | 0.0597 / 90.11% |
| 240 (k=5) | 0.0849 / 88.89% | 0.0939 / 87.67% |
Held-out NLL of the full model 2.2131. The greedy budget is worse than uniform k already on the fit text: per-layer sensitivities do not add up
when many layers change at once. Uniform `--moe-k` stays.

### E3 activation-weighted matrices (held-out activations; relative squared output error; ratio = parameters / parameters of the current weights)
- A, shared basis for gate+up, 4-bit coefficients: much better than the unweighted bound, still a loss. Layer 12: r=768 0.225 at 0.83x bytes,
  r=1152 0.138 at 1.25x, plain 2-bit 0.062, 3-bit 0.011. Layer 24: r=1152 0.114 vs 2-bit 0.050. Best case, layer 2: 0.050 vs 0.038. **FT-SLBF
  REJECTED**: worse than 2-bit while reading more bytes.
- B, low-rank attention: in the middle layers q at r=512 (0.38x) has error 0.0155-0.021 against 0.004-0.006 for 3-bit. Only layer 45 competes
  (q r=512 0.0065 vs 3-bit 0.008; k r=128 0.0136 vs 0.0129; v r=128 0.026 vs 0.028; factors not yet quantised). **Rejected**, marginal.
- C, certified skipping in down_proj: a certificate at 10% error skips 0.2-4.5% of the 32-neuron blocks, at 20% error 4.3-9.7% (an oracle:
  10-28%). Gate was at least 20% at 5-10% error. **REJECTED**.

### M4 temporal coherence — REJECTED
Median ||h_t - h_(t-1)|| / ||h_t|| = 1.04 / 1.04 / 0.97 (layers 12 / 24 / 36): consecutive hidden states differ by as much as their own size,
so delta computation has nothing to reuse.

### M1 dual-resolution ("shadow") experts — passes its offline gate (single layers only)
The top-R routed experts run in full, every further one keeps s of its 768 neurons (orthogonal matching pursuit on calibration text). Relative
squared error of the layer's MoE output against the current model, held-out:
| variant | expert reads | layer 12 | layer 24 | layer 36 |
|---|---|---|---|---|
| moe-k 5 (drop tail) | 5.00 | 0.180 | 0.187 | 0.144 |
| moe-k 6 (drop tail) | 6.00 | 0.084 | 0.088 | 0.064 |
| R2, s=128 | 3.11 | 0.254 / 0.237 | | |
| R3, s=128 | 3.92 | 0.173 / 0.158 | 0.199 / 0.181 | 0.133 / 0.128 |
| R4, s=128 | 4.74 | 0.116 / 0.105 | 0.135 / 0.122 | 0.087 / 0.083 |
| R4, s=256 | 5.48 | 0.090 / 0.077 | 0.106 / 0.089 | 0.066 / 0.060 |
Cells are "refit down columns / original down columns". **The closed-form refit does not help**; keeping the original columns of the selected
neurons is as good or better (the refit result also goes through 4-bit requantisation). That is simpler: a shadow is a neuron subset.
Open at this point: errors compound over 48 layers. Next test, decision rule written before the run (`benchmarks/lbt50/pick_shadow.py`): emulate
the neuron subsets inside the engine on all layers (`--shadow-full R --shadow-masks`, same bytes read, quality only); of the variants with at
most 5.5 expert reads take the lowest held-out KL; only if it beats uniform `--moe-k 6` run the 160-task set.

### M1 end to end (05:51, all 48 layers, 900 held-out positions) — REJECTED by the gate
| variant | expert reads | held-out KL | top-1 |
|---|---|---|---|
| uniform k=6 | 6.00 | 0.0350 | 92.89% |
| uniform k=5 | 5.00 | 0.0849 | 88.89% |
| uniform k=4 | 4.00 | 0.1717 | 83.22% |
| R2, s=256 (OMP) | 4.21 | 0.1157 | 84.67% |
| R4, s=128 (OMP / energy) | 4.74 | 0.0574 / 0.0539 | 90.11 / 89.67% |
| R3, s=256 (OMP) | 4.85 | 0.0661 | 88.22% |
| R4, s=256 (OMP / energy) | 5.48 | 0.0424 / 0.0383 | 91.89 / 92.56% |
| R5, s=128 (OMP) | 5.56 | 0.0351 | 91.67% |
| R6, s=128 (OMP) | 6.37 | 0.0186 | 94.11% |
The best variant with at most 5.5 reads (0.0383) does not beat uniform k=6 (0.0350), so the 160-task run was skipped as decided beforehand.
Shadows do improve on dropping experts at equal reads, but at the quality of k=6 they save only about 0.44 expert reads (about 38 MB of about
1,233 MB per token, about 3%). That does not pay for a new packed layout. No speed was measured. 629 of the 6,144 experts also had too little
calibration data for their selection, so a larger calibration set could move these numbers somewhat.

### M2 cache-resident additive codebooks and M3 one-bit head estimator (06:46) — M2 REJECTED, M3 not built
- M2, held-out relative output error (layers 12 / 24 / 36): gate at 2.25 bits 0.067 / 0.059 / 0.059 against 0.040 / 0.035 / 0.035 for the
  current 2.625 bits; down at 3.25 bits 0.036 / 0.039 / 0.039 against 0.019 / 0.021 / 0.021 for the current 3.56 bits.
- M3, 500 positions: first pass 40.1 MB instead of 92 MB. Band 1.9: median 105 survivors, true argmax missed once. Band 3.0: no miss, median
  2,778 survivors, 99th percentile 70,809 of 151,936. At most about 4% fewer bytes per token in the median case, with a heavy tail. Details in
  `docs/LBT50_MATH.md`.

### CI on the public tree
Linux CMake build, all self-tests, synthetic dense + MoE models: 11/11 checks, eval set reproducible byte for byte, Windows cross-build
(final tree, 06:44: `veltra.exe` 6,725,496 bytes).
