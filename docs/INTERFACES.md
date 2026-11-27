# Veltra interfaces (source of truth for all modules)

Veltra is the product name of the engine formerly called LittleBit-Turbo (renamed 25 Nov 2026). The rename is
backwards compatible; C++ namespaces (`lb::`), source file names and internal identifiers keep the old prefix.

| What | Veltra name | Legacy name (still works) |
|---|---|---|
| HTTP server | `veltra-server` | `lb_server` (symlink on Linux builds) |
| CLI / bench tool | `veltra-gen` | `lb_gen` (symlink) |
| kernel self-test | `veltra-selftest` | `lb_selftest` (symlink) |
| model file | `.vlt`, magic `"VELTRA\0\0"` | `.lbt`, magic `"LITTLEBT"` (loaded unchanged) |
| env knobs | `VELTRA_<NAME>` (wins) | `LB_<NAME>` (fallback), via `lb::env_get("<NAME>")` in engine/lb_common.h |

CLI quick reference:
```
veltra-server --model m.vlt [--host 127.0.0.1] [--port 8080] [--threads N] [--screen-head 1] [--rest rest.idx]
              [--bench-json chat/benchmarks_live.json] [--web-dir chat] [--compare-upstream http://127.0.0.1:8098]
  GET /health, /v1/models, /benchmarks, /arena, /chat, /api/benchmarks   POST /v1/chat/completions, /v1/completions,
  /api/benchmarks/run (loopback or VELTRA_ADMIN_TOKEN), /compare/llama/v1/chat/completions (proxy to --compare-upstream,
  loopback upstream only, 300 s timeout; 404 when not configured)
veltra-gen --model m.vlt --ids 1,2,3 --n 32 [--spec 0|1] [--dump-logits out.bin] [--sweep 1,2,4] ...
veltra-selftest
```

Toolchain on Windows: zig-as-clang (`cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-windows.cmake`). The NEON kernels
have only been compile-checked (`zig c++ -target aarch64-linux-gnu`); all measurements are on x86.

## Math
`W_hat = W_prim + W_res`, each path `diag(h) · U_b · diag(l) · V_b^T · diag(g)`
(U_b: m x r, V_b: n x r, entries +-1). Forward for activations x (n): `y = h * (U_b @ (l * (V_b^T @ (g * x))))`.
Bit convention: bit = 1 -> +1, bit = 0 -> -1. LittleBit-2 checkpoint scales: `u1`=h, `v2`=g, `l = u2 * v1`.

## `.vlt` model file (formerly `.lbt`; python/lbt_format.py is the reference writer/reader)
Little endian. `[64B header][meta JSON][table: n*160B][tensor data, each 64B aligned]`.
Header: `magic "VELTRA\0\0" or legacy "LITTLEBT"(8) | u32 version=1 | u32 n_tensors | u64 meta_off | u64 meta_len | u64 table_off | u64 data_off | u64 draft_layer_mask | u32 flags | u32 reserved`.
`draft_layer_mask`: bit i set => layer i is SKIPPED in the draft phase (layers 0..63; full list also in meta `draft_layer_skip`). flags bit0 = file also contains `*_bits` uint64 tensors.
Writers emit `"VELTRA\0\0"` by default (`LbtWriter(legacy_magic=True)` / `LbtStreamWriter(..., legacy_magic=True)` or env
`VELTRA_LEGACY_MAGIC=1` write `"LITTLEBT"` for engines built before the rename); every reader accepts both magics, the
layout is identical. Check: `python python/lbt_format.py` (round trip) and `python tests/verify_magic.py` (engine).
Table entry (160B): `char name[96] | u32 dtype | u32 ndim | u64 shape[4] | u64 offset | u64 nbytes | 8B pad`.
dtype: 0 f32, 1 f16, 2 bf16, 3 u8, 4 u64 bits, 5 LUT8, 6 i8, 7 i32.
Binary matrices: `X_lut` (LUT8, shape [rows, cols]; layout [row_block(64)][pair(2 groups of 4 cols)][64 bytes],
byte i = nibble(group 2p, row) | nibble(group 2p+1, row) << 4, nibble bit j = col 4g+j, rows padded to 64, cols to 128)
and optionally `X_bits` (u64, row-major, bit k of row j = word k/64, bit k%64).

Tensor names (all f32/f16 unless noted):
- `tok_embeddings` [vocab, hidden] f16 (or i8 + `tok_embeddings.scale` f16 [vocab] when meta `embed_dtype`=="i8")
- `output_norm` f32 [hidden]; `output` [vocab, hidden] f16 (absent when tied -> reuse tok_embeddings)
- `tokenizer.json`, `tokenizer_config.json` (u8 blobs)
- per layer i: `blk.{i}.attn_norm`, `blk.{i}.ffn_norm` f32 [hidden]; Qwen3 only `blk.{i}.q_norm`,`blk.{i}.k_norm` f32 [head_dim];
  optional `blk.{i}.{q,k,v}.bias` f32
- linears `blk.{i}.{q,k,v,o,gate,up,down}` with, for P in {prim,res} (res optional):
  `blk.{i}.{L}.{P}.u_lut|u_bits` shape [m, r], `blk.{i}.{L}.{P}.vt_lut|vt_bits` shape [r, n] (this is V_b^T),
  `.{P}.h` [m], `.{P}.l` [r], `.{P}.g` [n] (f16 or f32).
Meta JSON keys: `format,version,arch("qwen3"|"qwen2"|"llama"),model_name,hidden_size,num_layers,num_heads,num_kv_heads,head_dim,
intermediate_size,vocab_size,rope_theta,rms_norm_eps,max_position_embeddings,tie_word_embeddings,qk_norm,attn_bias,rope_scaling,
bos_token_id,eos_token_ids[],embed_dtype,chat_template,draft_layer_skip[],draft_layer_cos[],ranks{"blk.0.q":[r_prim,r_res],..},bpw{...}`.

## Kernels (engine/binary_gemv.h)  — done and bit-exact (scalar, AVX2, AVX-512 VBMI, NEON compile-checked)

## C++ engine API (engine/engine.h)
See `engine/engine.h`.

## C API (engine/capi.h), for JNI / python ctypes
See `engine/capi.h`.
