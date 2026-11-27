# `.qk` — llama.cpp K-quant experts inside `.lbt` (Q2_K / Q3_K / Q4_K)

Why: plain RTN INT2 g128 is a quality no-go for Qwen3-30B-A3B; the imatrix-calibrated `UD-Q2_K_XL` GGUF (11.8 GB) scores 20/20 on `eval_raptor`. The experts of that GGUF are
transcoded losslessly into our zero-copy, row-group-streamed layout and run by our own kernel (`engine/q2k_gemv.*`); no llama.cpp code at run time.

## What the GGUF contains (UD-Q2_K_XL, 48 layers x 128 experts)
| tensor class | types (per layer) |
|---|---|
| `ffn_gate_exps`, `ffn_up_exps` [128, 768, 2048] | Q2_K in all 48 layers (3.17 GB each class) |
| `ffn_down_exps` [128, 2048, 768] | Q3_K in 38 layers (3.29 GB), Q4_K in layers 1, 2, 3, 28, 41, 42, 43, 45, 46, 47 (1.13 GB) |
| `attn_q/k/v` | Q4_K (Q5_K in layers 1, 2, 46, 47); `attn_output` mostly Q4_K, Q6_K in 10 layers, Q5_K in 2 |
| `token_embd` Q4_K, `output` Q6_K, router `ffn_gate_inp` F32, norms F32 | |
Experts stack as `[E, rows, cols]` (rows = output features); one expert = the contiguous slice `data[e]`, row-major, each row = `cols/256` superblocks. Matches HF `gate/up_proj` [768, 2048], `down_proj` [2048, 768].

## Tensors
`blk.i.e{e}.{gate,up,down}.{q2k|q3k|q4k}` — U8, logical shape `[rows, cols/256 * SB]`, `nbytes = rows * cols/256 * SB`, 64-byte aligned. The suffix is the type; the engine's `load_linear`
picks it up (`LbLinear::qk`), so a file may mix `.n4` and `.qk` linears. Every other tensor of the model is exactly as in the n4 export (attention stays `.n4`+`.gs`, i8 head, ...). Meta gains `qk` (`tensor_types`, counts).

## Byte layout (`python/gguf_qk.py`, `engine/q2k_gemv.h`)
Stream per group of 4 rows: `[group g][superblock t][row 0..3][SB bytes]`; the 4 rows of a group are processed together, one sequential read stream per group.
| type | SB | per (row, superblock) |
|---|---|---|
| Q2_K | 84 | `sc[16]` (lo nibble scale, hi nibble min) · `qs[64]` · `d` f16 · `dmin` f16 (= ggml block) |
| Q3_K | 114 | `hmask[32]` · `qs[64]` · `sc[16]` int8 (ggml 6-bit scale minus 32, unpacked) · `d` f16 |
| Q4_K | 148 | `d` f16 · `dmin` f16 · `sc[8]` u8 · `mn[8]` u8 (unpacked from ggml's 12 B) · `qs[128]` |
Only the scale unpacking differs from ggml (+3.6 % bytes for Q3_K, +2.8 % for Q4_K); all values are identical, verified by `export_qwen3_moe_qk.py --verify` (dequantised tensors == gguf-py `dequantize`, `array_equal`).

## Kernel
`y[r] += d * sum_j sc_j * S_j - dmin * sum_j mn_j * B_j`, `S_j = sum q*a` over a 16- (Q4_K: 32-) weight sub-block, `B_j = sum a` (per-16 activation sums, `qk_sums16`, computed once per token slot).
Q3_K: `q = u - 4` is folded as `- 4 * sum_j sc_j B_j` into the integer accumulator. Per chunk of 32 weights: unpack (and/shift, Q3_K or hbit), `vpmaddubsw` (u8 x s8) -> `vpmaddwd` with the
per-16 scales (`vpshufb` from the broadcast scale vector) -> int32; one `cvtepi32_ps` + FMA per superblock for `d` and one for `dmin`. AVX2 only (Zen 2 has no VNNI; AVX-512 hosts use the same kernel).
Activations: int8 per token (+ optional second residual plane, same policy as the n4 path); results are independent of the number of slots per pass (T=1 == batched, bit-identical; tested in `lb_selftest`).
Execution: `linear_forward_jobs` (MoE experts, gate+up share activations) / `linear_forward` / `linear_forward_multi` route qk linears to `qk_forward_jobs` in `lbt_model.cpp`; tasks = runs of 64-row blocks.

## Bench (Ryzen 9 3900, 8 threads, 8 GB random-expert pool, load 15-17; the bench program is not published)
Q2_K gate/up + Q3_K down, 665 MB per token (48 layers x 8 experts): 26.2-28.0 ms/token = 23.8-25.4 GB/s, **93-97 %** of the pure-load ceiling for the same bytes/pattern (stream 27-28.6 GB/s).
With Q4_K down: 91 %. Batched slots (2-4 per pass) are compute-bound (46-55 % of ceiling) - prefill only.

# `.qk` v2 (Qwen3.6-35B-A3B export, `python/export_qwen36_stream.py`, `python/gguf_qk2.py`)

## FINDING: the 35B `UD-Q2_K_XL` experts are i-quants, NOT K-quants (verified on the file, 24 Nov)
`unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (12.29 GB file, 733 tensors, 40 layers x 256 experts, arch `qwen35moe`). Stacked experts `ffn_*_exps` are `[256, rows, cols]` exactly like the 30B:
| tensor | type | layers |
|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps` [256, 512, 2048] | **IQ2_XS** (74 B / 256 w = 2.3125 bpw) | 39 layers (all but 1) |
| same | **IQ3_XXS** (98 B = 3.0625 bpw) | layer 1 |
| `ffn_down_exps` [256, 2048, 512] | **IQ3_XXS** | 36 layers |
| same | **IQ4_XS** (136 B = 4.25 bpw) | layers 1, 34, 38, 39 |
No Q2_K / Q3_K / Q4_K expert exists in this file, so the engine's `.q2k/.q3k/.q4k` kernels cannot read them. Non-expert tensors in the GGUF (Q5_K/Q6_K attention, Q4_K output, Q5_K embeddings, F32 router/norms/ssm) are not used: the exporter takes them from HF bf16 (V-head reordering of the GGUF converter, contract 0.8).
Gate/up order of the fused HF `gate_up_proj [256,1024,2048]` (checked: dequantised GGUF `ffn_gate_exps[e]` vs HF rows, layers 0 and 2, experts 0/77/255): corr(gate, rows 0..511) = 0.951-0.956, corr(up, rows 512..1023) = 0.951-0.956, all cross terms |corr| < 0.03, interleaved (even/odd rows) ~0; `down_proj[e]` vs `ffn_down_exps[e]` corr 0.954-0.983. So gate = FIRST half, up = SECOND half, no transposition.

## New tensor types (suffix = type, same naming/placement rules as v1)
`blk.i.e{e}.{gate,up,down}.{iq2xs|iq3xxs|iq4xs}` — U8, logical shape `[rows, cols/256 * SB]`, `nbytes = rows * cols/256 * SB`, 64-byte aligned. For these types the transcode is a PURE BYTE PERMUTATION of the ggml blocks (no scale unpacking):
`[group of 4 rows][superblock][row 0..3][ggml block bytes]`. Block layouts = ggml-common.h `block_iq2_xs` (74 B: d f16, qs[32] u16 = 9-bit grid index + 7-bit sign index, scales[8], two 4-bit scales per byte), `block_iq3_xxs` (98 B: d f16, qs[96] = 64 grid indices (u8) then 8 x u32 = 4 x 7-bit sign index + 4-bit scale in bits 28-31), `block_iq4_xs` (136 B: d f16, scales_h u16, scales_l[4], qs[128], 4-bit indices into `kvalues_iq4nl`). Needed tables for a kernel: `iq2xs_grid[512]` (u64), `iq3xxs_grid[256]` (u32), `ksigns_iq2xs[128]`, `kmask_iq2xs[8]`, `kvalues_iq4nl[16]` (all in ggml-common.h / gguf-py `quants.py`). Reference: ggml `ggml_vec_dot_iq2_xs_q8_K`, `..._iq3_xxs_q8_K`, `..._iq4_xs_q8_K` (quants.c), reference dequant = gguf-py `dequantize`.
Meta `qk` = `{version: 2, source, tensor_types {"{layer}.{gate|up|down}": "IQ2_XS"...}, counts {suffix: n}, suffix {type: suffix}, sb_bytes {type: bytes}, expert_order, colnorm}`. The K-quant suffixes of v1 stay valid (the exporter accepts any GGUF whose expert types are in `gguf_qk2.SUFFIX`).

## Layout differences vs the 30B file
* Expert tensors are written EXPERT-MAJOR per layer (`e0.gate, e0.up, e0.down, e1.gate, ...`; ~1.0 MB contiguous per expert at IQ2_XS/IQ3_XXS) instead of matrix-major. Lookup is by name; only the on-disk locality differs.
* Optional group `--colnorm` (default OFF): `blk.i.e{e}.down.colnorm` F16 [512] = L2 norm over the 2048 output rows of each input column of the DEQUANTISED `down` matrix (ColumnSkip data). Absent unless requested; meta `qk.colnorm` = true.
* MTP block (`mtp.blk.0.*`, `mtp.fc`, `mtp.*norm`, optional `--no-mtp`): the GGUF has no MTP tensors, so its 256 experts per matrix are n4 g128 from HF bf16 (`mtp.blk.0.e{e}.{gate,up,down}` `.n4`+`.gs`), meta `mtp`.
* Other tensor names, norm `1 + w` baking, meta keys: (a research file that is not published). Extra meta written by the exporter: `partial_rotary_factor`, `mrope_section`, `mrope_interleaved`, `thinking = {default, template_variable "enable_thinking", generation_suffix_on "<think>\n", generation_suffix_off "<think>\n\n</think>\n\n", eval_harness_mode "off"}` (the Qwen3.6 chat template renders `<think>\n` after `<|im_start|>assistant\n` by default, i.e. thinking is ON; with `enable_thinking=false` it renders the empty think block, which the sanity/eval harness uses).
* `tokenizer_config.json` blob has `chat_template` injected (the HF repo ships it as `chat_template.jinja`).
