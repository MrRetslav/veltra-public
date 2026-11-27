"""Qwen3-30B-A3B Veltra `.vlt` with llama.cpp K-quant experts: expert tensors are transcoded bit-exactly from a GGUF (Q2_K / Q3_K / Q4_K),
every other tensor (attention, router, norms, i8 head / embeddings, tokenizer) is copied unchanged from an existing `.lbt` (the n4 export).

  python export_qwen3_moe_qk.py --gguf Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf --base q30b_n4.lbt --out q30b_qk.vlt [--gguf-py <llama.cpp/gguf-py>]
  python export_qwen3_moe_qk.py ... --verify 200      # afterwards: dequantised expert tensors == gguf-py dequantize (exact), N random samples

New tensors per expert: `blk.i.e{e}.{gate,up,down}.{q2k|q3k|q4k}` (U8 [rows, cols/256 * SB] logical shape, `nbytes = rows*cols/256*SB`),
layout in gguf_qk.py / docs/qk_format.md.  The engine detects the format from the tensor-name suffix.  Peak RAM ~ one layer (< 1 GB).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lbt_format as lbt
from lbt_stream import LbtStreamWriter
import gguf_qk as qk

PROJ = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_gguf(path, gguf_py=None):
    if gguf_py:
        sys.path.insert(0, gguf_py)
    import gguf
    r = gguf.GGUFReader(path)
    return gguf, {t.name: t for t in r.tensors}


def expert_raw(t, e):
    """ggml bytes of expert e of a stacked tensor [E, rows, bytes_per_row]."""
    d = t.data
    assert d.ndim == 3, d.shape
    return d[e]


def export(args):
    gguf, T = load_gguf(args.gguf, args.gguf_py)
    base = lbt.LbtReader(args.base)
    meta = dict(base.meta)
    E = int(meta["num_experts"]); L = int(meta["num_layers"])
    names = sorted(base.entries, key=lambda n: base.entries[n].offset)
    def expert_name(n):
        p = n.split(".")
        return len(p) >= 5 and p[0] == "blk" and p[2].startswith("e") and p[2][1:].isdigit() and p[3] in PROJ
    keep = [n for n in names if not expert_name(n)]
    layers_sparse = sorted({int(n.split(".")[1]) for n in names if expert_name(n)})
    max_tensors = len(keep) + len(layers_sparse) * E * 3 + 8
    out_tmp = args.out + ".part"
    w = LbtStreamWriter(out_tmp, max_tensors)
    types = {}
    counts = {}
    emitted = set()
    t0 = time.time()

    def emit_layer(i):
        for proj, g in PROJ.items():
            t = T[f"blk.{i}.{g}.weight"]
            qt = t.tensor_type.name
            assert qt in qk.SUFFIX, (t.name, qt)
            d = t.data
            rows, bpr = d.shape[1], d.shape[2]
            cols = bpr // qk.GGML_BYTES[qt] * qk.QK_K
            types[f"{i}.{proj}"] = qt
            for e in range(E):
                buf = qk.pack_qk(d[e], qt)
                w.add_tensor(f"blk.{i}.e{e}.{proj}.{qk.SUFFIX[qt]}", buf, lbt.U8, (rows, cols // qk.QK_K * qk.QK_BYTES[qt]))
                counts[qt] = counts.get(qt, 0) + 1
        log(f"layer {i} experts done ({time.time() - t0:.0f}s, file {w.pos / 2**30:.2f} GB)")

    for n in keep:
        # experts of layer i go in front of the first tensor of layer i+1 / the first top-level tensor after layer i's attention
        p = n.split(".")
        if p[0] == "blk":
            li = int(p[1])
            for lj in layers_sparse:
                if lj < li and lj not in emitted:
                    emit_layer(lj); emitted.add(lj)
        ent = base.entries[n]
        if ent.dtype == lbt.BF16:
            raise SystemExit("bf16 tensors not supported in copy")
        w.add_tensor(n, base.raw(n).view(lbt._NP[ent.dtype]), ent.dtype, tuple(ent.shape))   # raw bytes (logical shape != nbytes for packed tensors)
    for lj in layers_sparse:
        if lj not in emitted:
            emit_layer(lj); emitted.add(lj)
    meta["qk"] = {"version": 1, "source": os.path.basename(args.gguf), "tensor_types": types, "counts": counts, "suffix": qk.SUFFIX,
                  "sb_bytes": qk.QK_BYTES}
    w.set_meta(**meta)
    w.draft_layer_mask = base.draft_layer_mask
    info = w.close()
    os.replace(out_tmp, args.out)
    log(f"DONE {args.out}: {info['bytes'] / 2**30:.2f} GB, {info['tensors']} tensors, {(time.time() - t0) / 60:.1f} min, counts {counts}")
    if args.verify:
        verify(args.out, gguf, T, args.verify)


def verify(path, gguf, T, nsamples, seed=1):
    r = lbt.LbtReader(path)
    E = int(r.meta["num_experts"])
    layers = sorted({int(n.split(".")[1]) for n in r.entries if ".e0.gate." in n})
    rnd = random.Random(seed)
    bad = 0
    samples = [(i, rnd.randrange(E), p) for i in layers for p in PROJ]          # every (layer, matrix) once, random expert
    samples += [(rnd.choice(layers), rnd.randrange(E), rnd.choice(list(PROJ))) for _ in range(max(0, nsamples - len(samples)))]
    samples = samples[:max(nsamples, len(layers) * 3)]
    for (i, e, proj) in samples:
        t = T[f"blk.{i}.{PROJ[proj]}.weight"]
        qt = t.tensor_type.name
        name = f"blk.{i}.e{e}.{proj}.{qk.SUFFIX[qt]}"
        rows, bpr = t.data.shape[1], t.data.shape[2]
        n = bpr // qk.GGML_BYTES[qt] * qk.QK_K
        ref = gguf.quants.dequantize(np.ascontiguousarray(t.data[e]), t.tensor_type)
        got = qk.unpack_qk(r.raw(name), qt, rows, n)
        ok = ref.shape == got.shape and np.array_equal(ref, got)
        if not ok:
            bad += 1
            print("MISMATCH", name, qt, float(np.abs(ref - got).max()) if ref.shape == got.shape else (ref.shape, got.shape))
    log(f"verify: {len(samples)} expert tensors vs gguf-py dequantize, mismatches {bad}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--base", required=True, help="existing .lbt (n4 export) supplying every non-expert tensor + meta")
    ap.add_argument("--out", required=True, help="output model file (.vlt; a legacy .lbt name works too; env VELTRA_LEGACY_MAGIC=1 writes the old magic)")
    ap.add_argument("--gguf-py", default="", help="path to llama.cpp/gguf-py if the gguf package is not installed")
    ap.add_argument("--verify", type=int, default=0, help="after export: compare N expert tensors with gguf-py dequantize (exact)")
    export(ap.parse_args())


if __name__ == "__main__":
    main()
