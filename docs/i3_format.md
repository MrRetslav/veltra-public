# `.i3` - INT3 g128 asymmetric linears (attention of Qwen3-30B-A3B)

Why: the attention linears of the n4 model are 0.47 GB of the ~1.3 GB streamed per decoded token (q 4096x2048, o 2048x4096, k/v 512x2048 per layer, 48 layers = 906 M weights; q/o = 93 %).
n4 streams 4.25 bit/weight in practice (4 bit + a 4-byte f32 group scale per 128 weights); `.i3` streams 3.25 bit/weight (3 bit + f16 scale + f16 min per 128) = -0.11 GB/token.

## Tensor
`blk.i.{q,k,v,o}.i3` - U8, logical shape `[m, n/256 * 104]` (m % 64 == 0, n % 256 == 0), `nbytes = m * n/256 * 104`, 64-byte aligned. `load_linear` prefers `.i3` over `.n4`, so a file may mix both
(per layer / per matrix); everything else of the model is untouched. Meta gains `i3 = {version, group, mats, layers, count, bpw}`.

## Byte layout (`engine/i3_gemv.h`, `python/export_i3_attn.py: pack_i3`)
Stream of chunks `[row block rb (64 rows)][tile pair p (256 columns)]`, 6656 bytes each (rows of a block are streamed together, one sequential read stream per row block):
| part | bytes | content |
|---|---|---|
| header | 512 | f16 `[tile 2][field 2][row 64]`: field 0 = scale, field 1 = min. `W_hat = q * scale + min`, q in 0..7, group = one 128-column tile |
| data | 6144 | `[row 64][96 B]`: bytes 0..31 = low 2 bits of tile 0, bytes 32..63 = low 2 bits of tile 1 (byte i: weights `i, i+32, i+64, i+96` of the tile in bit fields 0-1, 2-3, 4-5, 6-7); bytes 64..95 = high bit plane, bit `v` of byte `i` = bit 2 of weight `i + 32*(v%4)` of tile `v/4` |

## Kernel
Per row and tile: `S = sum q*a` (u8 x s8 `vpmaddubsw`, 4 vectors per tile summed in int16 (<= 7168), `vpmaddwd`), `A` = tile sum of the int8 activations (shared with the n4 path), contribution
`fma(float(S), scale, float(A) * min)`, `acc += contribution` (x the per-tile activation scale in prefill tile mode). Unpack = 5 uops per 32 weights (and/shift for the 2-bit field, shift+and for the high bit,
or). The scalar reference, the AVX2 kernel and every slot count give bit-identical results (`lb_selftest`: 40 checks; `scripts/bench_repr/bench_i3.cpp check`).
Activation preparation (int8 per token, optional second plane for outlier tokens, tile scales in prefill) is the nibble path's (`nibble_forward_multi`), so q/k/v/o may be any mix of n4 and i3.

## Producing a file
1. `LB_HESS=<dir> lb_gen ...` on calibration text dumps the f16 inputs of the attention linears of the n4 model (`python/calib_text_i3.py` makes the token sequences, own text only).
2. `python/quant_i3.py`: Hessians `H = X^T X / N`, column-sequential GPTQ (damp 0.01, plain column order, groups of 128, per-group per-row clip search for (scale, min), Hessian-diagonal weighted),
   writes `L<ll>_<mat>.npz` + per-matrix held-out error statistics (n4-RTN, i3-RTN, i3-GPTQ).
3. `python/export_i3_attn.py --src q30b_qk.lbt --q <dir> --out q30b_qk_i3.lbt [--mats ...] [--skip-layers ...]` (all other tensors copied bit-exact).
4. (a research file that is not published) (NLL / KL gate through `lb_gen --dump-logits --seq-logits`).
