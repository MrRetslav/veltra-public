# Bit-plane grid linears (INT2 / INT3, group 128) in `.lbt`

Status: contract v1, 23 Nov 2026. Engine side: `engine/lbt_model.*`.

## 1. Math (single source of truth)

For a linear with `rows = m` (out), `cols = n` (in, **n % 128 == 0**), `P` planes (P = 3 for INT3, P = 2 for INT2), group `G` (128, or 0 = one scale per row):

```
W_hat[r, c] = gs[r, c // G] * q[r, c]              (G = 0: gs[r, 0])
q[r, c]     = sum_{p=0..P-1} 2^(P-2-p) * (2*bit_p[r, c] - 1)          bit_p in {0,1}, 1 -> +1
```
INT3 plane weights `[2, 1, 0.5]` -> q in {+-0.5, +-1.5, +-2.5, +-3.5}; INT2 weights `[1, 0.5]` -> q in {+-0.5, +-1.5}.
P = 4 (INT4) is also allowed: plane weights `[4, 2, 1, 0.5]`, `q = j - 7.5` in {+-0.5 ... +-7.5}, same tensors `gp0_lut..gp3_lut` (added 23 Nov for the INT4 g128 run).
Level index `j = sum_p bit_p * 2^(P-1-p)`, `q = j - (2^P - 1)/2` (so `j = 0` is the most negative level). `gs >= 0`, stored f16.

Forward (what the engine must compute, per token, `x` float -> int8 per-token quantisation exactly as the existing LittleBit path does):
```
y[r] = sum_g gs[r, g] * sum_p w_p * S_p[r, g]          S_p[r, g] = sum_{c in group g} sign_p[r, c] * x[c]
```
With G = 128 a group is exactly one kernel tile (`kTileK = 128` columns = `kTileGroups` 32 LUT groups), so `S_p[r, g]` is the per-tile
int accumulator of the existing LUT kernel for plane p. **One activation quantisation + one LUT build serve all planes.**
G = 0 (row scale) is the fallback if the per-tile epilogue is late: then `y[r] = gs[r] * sum_p w_p * (full-row S_p[r])`.

## 2. Tensors (names, dtypes, shapes)

Per decoder linear `blk.{i}.{L}` with `L in {q,k,v,o,gate,up,down}` (dense Qwen3; MoE experts keep using the old LittleBit layout):

| tensor | dtype | shape | content |
|---|---|---|---|
| `blk.{i}.{L}.gp{p}_lut`, p = 0..P-1 | LUT8 | `[m, n]` | plane p, **tile-major** (below) |
| `blk.{i}.{L}.gs` | F16 | `[m, n/G]` (`[m, 1]` if G = 0) | group scales |

Tile-major layout: for tile `t = 0 .. n/128-1` take `bit_p[:, 128t : 128t+128]` as a bool matrix `[m, 128]`, pack it with
`lbt_format.pack_lut8` (rows padded to 64, K = 128 -> `pad_rows(m) * 16` bytes), and concatenate the tiles in column order.
`nbytes(gp{p}_lut) = (n/128) * pad_rows(m) * 16`. Each tile is therefore an ordinary LUT8 matrix of `cols = 128` that
`binary_gemv()` accepts as a `BinMatView{rows = m, cols = 128, lut_w = base + t * pad_rows(m) * 16}`; the matching activation LUT for tile t starts at
`luts + t * kTileGroups * 32` bytes (AVX2 layout) / `t * 1024` bytes (AVX-512 layout) inside the per-token LUT.
The number of planes of a linear = number of consecutive `gp{p}_lut` tensors present (mixed INT2/INT3 files are allowed).
A linear is a grid linear iff `blk.{i}.{L}.gp0_lut` exists; otherwise the old `.prim/.res` LittleBit tensors apply.
Reference implementation of writer/reader/dequant: (a research file that is not published) (`add_grid_linear`, `read_grid_linear`, `grid_dequant`, self-check: `python grid_lbt.py`).

Everything else is unchanged from `docs/INTERFACES.md` / `export_lbt`:
- `tok_embeddings` **I8** `[vocab, hidden]` + `tok_embeddings.scale` **F16** `[vocab]` (per-row), `meta.embed_dtype = "i8"`; tied head (no `output` tensor) for Qwen3-1.7B.
- `output_norm`, `blk.{i}.attn_norm|ffn_norm|q_norm|k_norm` F32, `tokenizer.json`, `tokenizer_config.json` blobs.
- **No `*_bits` (u64) tensors**, header flag `FLAG_HAS_U64` = 0.
- meta: all fields of `export_lbt` (arch qwen3, sizes, rope, eos ids, chat_template, `draft_layer_skip` list + header mask) plus
  `"grid": {"version": 1, "group": 128, "mode": "int3" | "int2" | "mixed", "bpw_linear_total": <float incl. scales>}`.
  Drop `ranks`/`bpw` per-linear LittleBit entries (or leave empty).
- Self-speculative draft = layer skipping only. Grid linears have no "primary-only" mode (`use_res` is ignored).

## 3. Size (Qwen3-1.7B, 28 layers, 1.41 B linear weights, vocab 151 936)
| mode | bits / weight | linears | embeddings i8 | file |
|---|---|---|---|---|
| INT3 g128 | 3.125 | ~551 MB | ~312 MB | ~0.87 GB |
| mixed (down_proj + layers 0 and 27 INT3, rest INT2) | ~2.4 | ~420 MB | ~312 MB | ~0.74 GB |
| INT2 g128 | 2.125 | ~375 MB | ~312 MB | ~0.69 GB |

## 4. Acceptance checks
1. `python python/grid_lbt.py` passes (writer/reader/dequant round trip).
2. Engine vs PyTorch, per linear: cosine(engine `lb_linear_forward`, `x @ W_hat.T`) >= 0.9999 on random/real activations, AVX2 and AVX-512, T = 1 and T = 4.
3. Whole model: cosine of last-token logits (`lb_gen --dump-logits`) vs the PyTorch forward of the dequantised student >= 0.999.
4. All existing tests still pass (`lb_selftest`, `tests/verify_*.py`, old `.lbt` files still load: tiny.lbt, tiny_moe.lbt, qwen3-1.7b-3.0bpw-blockwise.lbt).

## 5. Nibble-INT4 layout (`.n4`, contract v1, 23 Nov 2026)

Lossless repack of the P=4 (INT4) grid linears for hardware multiply-add kernels; INT2/INT3 linears keep the bit-plane layout (a file may mix both).
For a linear `blk.{i}.{L}` the 4 tensors `gp0..gp3_lut` are replaced by ONE tensor:

| tensor | dtype | shape | content |
|---|---|---|---|
| `blk.{i}.{L}.n4` | U8 | `[m, n]` (logical; nbytes = `pad_rows(m)/64 * (n/128) * 4096`) | nibble-packed level indices, row-block-major |
| `blk.{i}.{L}.gs` | F16 | `[m, n/128]` | unchanged (group 128 only) |

Level index `j = (bit0<<3)|(bit1<<2)|(bit2<<1)|bit3` in 0..15 (bit_p = plane p bit of the bit-plane format), value `q = j - 7.5`, `W_hat = gs[r, c//128] * q` (identical to the plane format, so the repack is exact).
Byte layout: `[row_block rb = r // 64][tile t = c // 128][row r % 64][byte i, 0..63]`, each row-tile = 64 bytes with
`byte[i] = j[r, 128 t + i] | (j[r, 128 t + 64 + i] << 4)` (low nibble = columns 0..63 of the tile, high nibble = columns 64..127).
So one row block is ONE contiguous stream of `(n/128) * 4096` bytes (64 rows x 64 B per tile); rows >= m are zero padded (their output is ignored).
Math for a kernel (activations int8 per token as in the plane path, `a`): per tile `S = sum_c j[r,c]*a[c]` (unsigned x signed, e.g. `vpmaddubsw`/`vpdpbusd`), correction `A = sum_c a[c]` once per tile and token,
`y[r] = act_scale * sum_g gs[r,g] * (S[r,g] - 7.5 * A[g])` (or use `2j-15` odd-integer weights and half the scale). Loader rule: a linear with `.n4` is a nibble linear (no `gp*` tensors). Kernels must read the mmap'd tensor directly (no extra RAM copy on any ISA).
Reference: `python/nibble_lbt.py` (`repack_file`, `read_n4`, `n4_dequant`; self-check `python python/nibble_lbt.py`), CLI `scripts/repack_int4_nibble.py in.lbt out.lbt`.
