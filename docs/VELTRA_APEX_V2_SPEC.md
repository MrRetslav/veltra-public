# Veltra Apex (`.vlt v2`): proposed format and what the measurements allow

Status: **specification only. No v2 file, writer or kernel exists.** Each part lists the measurement that supports or rejects it
(details: `docs/LBT50_LOG.md`). Reference: Qwen3-30B-A3B, Ryzen 9 3900, 8 threads, about 1.25 GB of weights read per token.

## Verdict
Apex as proposed (50-55+ tok/s on ordinary text on this machine without noticeable quality loss) is **not supported by the measurements**.
The memory delivers at most about 33 GB/s and the engine reaches about 80% of that, so 50 tok/s needs at most about 580 MB per token on
ordinary text (660 MB even at the full 33 GB/s), less than half of today's amount.

## Parts and their state
| part | idea | measurement | state |
|---|---|---|---|
| Shared basis for gate/up (FT-SLBF) | `W_e ~ C_e B` | activation-weighted error 0.138 at 1.25x the bytes against 0.062 for 2-bit (layer 12) | rejected |
| Low-rank attention | `q, k, v ~ A B` | worse than 3-bit in the middle layers, marginal in the last ones | rejected |
| Certified block skipping | skip down_proj blocks with a proven bound | 0.2-4.5% of blocks at 10% error | rejected |
| Per-layer expert budget | different k per layer | higher KL than uniform k at equal reads | rejected |
| Delta computation | reuse the previous token | relative hidden-state step about 1.0 | rejected |
| EAGLE-3 draft head in C++ | several tokens per pass | a pass of 4 tokens costs 2.42 passes of 1 token: best case +24% | rejected |
| Thread pinning, software prefetch | memory placement | inside run-to-run noise | rejected |
| LUT kernel for 2-bit weights | `vpshufb` tables | 1.49 instead of 3.05 cycles per 32 weights in L2, no gain from DRAM with 8 threads | useful only where compute is the limit; needs a new weight layout |
| Shadow experts | tail experts keep a subset of their neurons | at 5.48 reads held-out KL 0.0383, uniform k=6 (6.00 reads) 0.0350 | gate not passed |

## The best candidate, and why it was not built: shadow experts
**Rule.** Of the 8 routed experts of a token the first `R` (by router probability) run in full. Every further one runs with `s` of its 768
neurons: `f'_e(x) = D_e[:,S_e] h_e,S(x)`, where `S_e` is a fixed set per expert, chosen once from calibration text. No refit: the original
down-projection columns performed at least as well as refitted ones.

**Layout (proposal).**
- Extra tensors per layer and expert: `blk.i.s{e}.gate`, `blk.i.s{e}.up` (`s` rows, taken unchanged from the full tensors, so lossless) and
  `blk.i.s{e}.down` (2048 x `s`; the column subset has to be re-quantised because the K-quant superblocks span 256 columns).
- Meta: `"shadow": {"full_ranks": R, "neurons": s, "selector": "..."}`.
- A reader without shadow support ignores the extra tensors and runs the full model.
- Cost: file and memory grow by about `s/768` of the expert weights (about one third for `s = 256`).

**Bytes per token** (expert weights only, in full-expert equivalents): `R + (8 - R) * rel(s)`, `rel(128) = 0.185`, `rel(256) = 0.369`.
R=4, s=128: 4.74. R=4, s=256: 5.48. Uniform `--moe-k 6`: 6.00.

**What is measured.** Quality only, by emulation inside the engine (`--shadow-full`, `--shadow-masks`; it reads the same bytes as before),
on 900 held-out positions (full table in `docs/LBT50_LOG.md`). **No speed of a shadow layout has been measured.** At the quality of
`--moe-k 6` (R=5, s=128: KL 0.0351 at 5.56 reads) the saving is about 0.44 expert reads, about 3% of the bytes per token.

**Gate (set before the run): not passed.** With at most 5.5 expert reads no variant was as close to the full model as uniform `--moe-k 6`,
so the 160-task run was skipped and nothing was built. A larger calibration set (629 of 6,144 experts had too little data) is the one thing
that could change this; even then the measured margin points to a few percent, not to 50 tok/s.
