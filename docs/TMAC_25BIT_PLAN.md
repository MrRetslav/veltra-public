# LUT ("T-MAC style") kernels for low-bit weights: what was measured and what a v0.2 format would need

Source: `benchmarks/apex_proto/q2kernels.cpp` on the reference server (Ryzen 9 3900, Zen 2, AVX2), exact integer kernels, no scales, counters from
`perf_event_open`. **Only 2-bit weights were built and measured; a 2.5-bit kernel was not built.** Nothing below is an engine speedup.

## Measured (2048 columns, int8 activations)
| kernel | cycles / 32 weights, in L2 | instructions / 32 w | Gweights/s, DRAM, 1 thread | Gweights/s, DRAM, 8 threads |
|---|---|---|---|---|
| maddubs (unpack + `vpmaddubsw`, like the engine's K-quant kernels) | 3.05 | 11.2 | 37.7 | 128.4 |
| bit-plane (two masks per 32 weights) | 4.91 | 21.2 | 22.8 | 126.3 |
| LUT32 (2 weights -> 4-bit index, lane-local `vpshufb` into per-activation tables) | **1.49** | 6.9 | **54.8** | 129.0 |

- Building the tables costs 3.3 µs per 2048-long activation vector; they are shared by every matrix that reads that vector (all routed experts'
  `gate`/`up` of a layer read the same one).
- With 8 threads every kernel is limited by memory (about 32 GB/s), so on this server a LUT layout does **not** make single-token decode faster.
  It matters where the code is compute-bound: multi-token passes (speculation, batching: measured 1.54x time for 2 tokens although the bytes only
  grow 1.30x) and CPUs with fewer cores than the memory can feed.

## Why the current files cannot use it
Q2_K stores a scale and a minimum per 16 weights **per row**. A LUT shares one table per activation pair across 32 rows, so it needs scales that
are shared by the rows of a tile (or none inside a tile). That is a new format: re-quantisation from the original weights, not a repack.

## What 2.5 bits would look like (not built)
- Codes of 5 bits for 2 weights (e.g. 6 x 5 levels, or a 32-entry joint codebook) do not fit a 16-entry `vpshufb` table; AVX2 would need two
  lookups and a blend per index, roughly doubling the shuffle work. A cleaner way to spend bits is **two additive 4-bit codebooks over 4 weights**
  (2 bits per weight plus scales): the same instruction mix as LUT32 with a learned codebook (`docs/LBT50_MATH.md`, method 2).
- Quality decides first: the format only makes sense if its output error on held-out activations is at or below the current Q2_K weights
  (measured per matrix by `benchmarks/lbt50/novel.py`).

## Order of work for v0.2 (only if method 2 passes its gate)
1. Quantiser for the tiled additive-codebook layout, verified against the numpy reference used in the experiment.
2. Scalar reference kernel, AVX2 LUT kernel, bit-exactness self-test.
3. One layer type at a time (`gate`/`up` first), with the 160-task set after each step.
