#!/usr/bin/env python3
"""Apex FT-SLBF go/no-go: the Eckart-Young UPPER BOUND of any shared basis, on the original (bf16) expert weights of Qwen3-30B-A3B.

For a layer, stack all experts' gate_proj (and up_proj) rows: the best possible shared INPUT basis of rank r captures exactly the top-r eigenvalue
mass of  G = sum_e W_e^T W_e  (2048 x 2048). For down_proj the shared OUTPUT basis uses sum_e W_e W_e^T. No structured basis (SLBF included) of
the same rank can do better. Compared against the relative squared error of a simple 2-bit / 3-bit / 4-bit quantisation of the same weights
(asymmetric uniform levels per group of 16, the scale of Q-K sub-blocks; llama.cpp's imatrix quantiser is somewhat better still).

  <venv with torch+safetensors>/python basis_bound.py --hf /path/Qwen3-30B-A3B --out basis_bound.json [--layers 0,12,24,36,47]
"""
import argparse
import json
import os
import time

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="0,12,24,36,47")
a = ap.parse_args()
torch.set_num_threads(8)
index = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
handles = {}


def tensor(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.hf, f), framework="pt")
    return handles[f].get_tensor(name).float()


def quant_err(w, bits, group=16):
    """relative squared error of per-group asymmetric uniform quantisation (groups along the input dimension)."""
    g = w.reshape(w.shape[0], -1, group)
    lo = g.amin(-1, keepdim=True)
    step = (g.amax(-1, keepdim=True) - lo).clamp_min(1e-12) / (2 ** bits - 1)
    q = ((g - lo) / step).round() * step + lo
    return float((q - g).pow(2).sum() / g.pow(2).sum())


RANKS = [192, 384, 576, 768, 1152, 1536]
out = {"ranks": RANKS, "layers": {}}
for l in [int(x) for x in a.layers.split(",")]:
    t0 = time.time()
    res = {}
    for kind in ("gate", "up", "gate+up", "down"):
        grams = torch.zeros(2048, 2048, dtype=torch.float64)
        own = {r: 0.0 for r in RANKS if r < 768}
        total, n_exp = 0.0, 128
        qe = {2: 0.0, 3: 0.0, 4: 0.0}
        for e in range(n_exp):
            names = ["gate_proj", "up_proj"] if kind == "gate+up" else [kind + "_proj"]
            for nm in names:
                w = tensor("model.layers.%d.mlp.experts.%d.%s.weight" % (l, e, nm))
                if kind == "down":
                    w = w.t().contiguous()          # 768 x 2048: rows span the output space
                grams += (w.t() @ w).double()
                total += float(w.pow(2).sum())
                if e % 16 == 0:                      # reference errors on a sample of experts
                    for b in qe:
                        qe[b] += quant_err(w.t() if kind == "down" else w, b)   # groups along the matrix's own input dimension
                    sv = torch.linalg.svdvals(w).double() ** 2
                    for r in own:
                        own[r] += float(sv[:r].sum() / sv.sum())
        ev = torch.linalg.eigvalsh(grams).flip(0).clamp_min(0)
        cum = torch.cumsum(ev, 0) / ev.sum()
        samples = len(range(0, n_exp, 16)) * (2 if kind == "gate+up" else 1)
        res[kind] = {"shared_basis_rel_sq_error": {str(r): round(float(1 - cum[r - 1]), 4) for r in RANKS},
                     "own_rank_r_rel_sq_error": {str(r): round(1 - own[r] / samples, 4) for r in own},
                     "uniform_quant_rel_sq_error": {str(b): round(qe[b] / samples, 4) for b in qe}}
        print("layer %2d %-8s shared-basis rel^2 error r=576 %.3f r=1152 %.3f r=1536 %.3f | 2-bit %.3f 3-bit %.3f 4-bit %.4f | own rank-576 %.3f"
              % (l, kind, res[kind]["shared_basis_rel_sq_error"]["576"], res[kind]["shared_basis_rel_sq_error"]["1152"],
                 res[kind]["shared_basis_rel_sq_error"]["1536"], res[kind]["uniform_quant_rel_sq_error"]["2"], res[kind]["uniform_quant_rel_sq_error"]["3"],
                 res[kind]["uniform_quant_rel_sq_error"]["4"], res[kind]["own_rank_r_rel_sq_error"]["576"]), flush=True)
    out["layers"][str(l)] = res
    print("layer %d done in %.0f s" % (l, time.time() - t0), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
