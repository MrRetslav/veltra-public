#!/usr/bin/env python3
"""Tiny random Veltra models for CI (numpy only): `ci_dense.vlt` and `ci_moe.vlt`. Weights are noise: the files exist to exercise the loader,
the Q4_K kernels, forward vs forward_batch, expert routing and speculative decoding; their text output means nothing.

  python3 tests/make_ci_model.py --out DIR        # needs numpy and a C++20 compiler (for the shared Q4_K quantiser)
"""
import argparse
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
import lbt_format as lbt                  # noqa: E402
from lbt_stream import LbtStreamWriter    # noqa: E402

H, I, HEADS, KVH, HD, V, LAYERS = 256, 256, 2, 1, 128, 1024, 3
EXPERTS, TOPK = 8, 3


def quantiser(tmp):
    so = os.path.join(tmp, "libq4k.so")
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-O2", "-shared", "-fPIC", "-I", os.path.join(ROOT, "engine"),
                    os.path.join(ROOT, "engine", "q4k_quant.cpp"), "-o", so], check=True)
    lib = ctypes.CDLL(so)
    lib.veltra_q4k_quantize.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
    return lib


def q4k(lib, w):
    m, n = w.shape
    nsb = n // 256
    out = np.zeros((m // 4, nsb, 4, 148), dtype=np.uint8)
    for r in range(m):
        row = np.ascontiguousarray(w[r], dtype=np.float32)
        blk = np.empty((nsb, 148), dtype=np.uint8)
        lib.veltra_q4k_quantize(row.ctypes.data, nsb, blk.ctypes.data)
        out[r // 4, :, r % 4, :] = blk
    return out.reshape(m, nsb * 148)


def build(lib, path, moe, seed):
    rng = np.random.default_rng(seed)
    w = LbtStreamWriter(path, max_tensors=16 + LAYERS * (10 + 3 * EXPERTS), meta_reserve=1 << 14)
    meta = dict(format="lbt", version=1, arch="qwen3", model_name=os.path.splitext(os.path.basename(path))[0], hidden_size=H, num_layers=LAYERS,
                num_heads=HEADS, num_kv_heads=KVH, head_dim=HD, intermediate_size=I, vocab_size=V, max_position_embeddings=2048,
                rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=False, qk_norm=False, attn_bias=False, embed_dtype="i8",
                bos_token_id=-1, eos_token_ids=[V - 1], num_experts=EXPERTS if moe else 0)
    if moe:
        meta.update(num_experts_per_tok=TOPK, moe_intermediate_size=I, norm_topk_prob=True)
    w.set_meta(**meta)

    def linear(name, m, n, std):
        w.add_tensor(name + ".q4k", q4k(lib, rng.normal(0, std, (m, n)).astype(np.float32)), lbt.U8)

    def rows_i8(name, std):
        x = rng.normal(0, std, (V, H)).astype(np.float32)
        s = (np.abs(x).max(axis=1) / 127.0).astype(np.float16)
        w.add_tensor(name, np.clip(np.rint(x / s.astype(np.float32)[:, None]), -127, 127).astype(np.int8), lbt.I8)
        w.add_tensor(name + ".scale", s, lbt.F16)

    rows_i8("tok_embeddings", 0.4)
    rows_i8("output", 0.3)
    w.add_tensor("output_norm", np.ones(H, np.float32), lbt.F32)
    for l in range(LAYERS):
        b = "blk.%d." % l
        for nm in ("attn_norm", "ffn_norm"):
            w.add_tensor(b + nm, (1.0 + 0.1 * rng.standard_normal(H)).astype(np.float32), lbt.F32)
        linear(b + "q", HEADS * HD, H, 0.06)
        linear(b + "k", KVH * HD, H, 0.06)
        linear(b + "v", KVH * HD, H, 0.06)
        linear(b + "o", H, HEADS * HD, 0.05)
        if moe:
            w.add_tensor(b + "moe_gate", rng.normal(0, 0.3, (EXPERTS, H)).astype(np.float16), lbt.F16)
            for e in range(EXPERTS):
                linear(b + "e%d.gate" % e, I, H, 0.06)
                linear(b + "e%d.up" % e, I, H, 0.06)
                linear(b + "e%d.down" % e, H, I, 0.05)
        else:
            linear(b + "gate", I, H, 0.06)
            linear(b + "up", I, H, 0.06)
            linear(b + "down", H, I, 0.05)
    print("wrote %s: %d bytes" % (path, w.close()["bytes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        lib = quantiser(tmp)
        build(lib, os.path.join(a.out, "ci_dense.vlt"), False, 1)
        build(lib, os.path.join(a.out, "ci_moe.vlt"), True, 2)


if __name__ == "__main__":
    main()
