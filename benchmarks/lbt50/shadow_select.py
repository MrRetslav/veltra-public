#!/usr/bin/env python3
"""Neuron masks for the shadow-expert quality emulation (veltra-gen / veltra-server --shadow-full R --shadow-masks FILE).

For every layer and expert it picks s of the expert's neurons from calibration activations of the running engine (repository text; the files of
the held-out probe sets are excluded, no evaluation material is read). Two selectors:
  omp    batched orthogonal matching pursuit on the expert's weighted output (the selector measured as method M1 in novel.py)
  energy the s neurons with the largest E[h_i^2] * ||D[:, i]||^2 (one pass, no solve)
Writes <out>/<selector>_s<s>.bin: [layers][experts][neurons] bytes, 0 or 1.

  <venv: torch, safetensors>/python shadow_select.py --gen bin/veltra-gen --model M.vlt --hf DIR --work DIR --out DIR
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
from vlt_weights import Vlt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sizes", default="128,256")
ap.add_argument("--fit-segments", type=int, default=24)
ap.add_argument("--segment-tokens", type=int, default=500)
a = ap.parse_args()
torch.set_num_threads(8)
SIZES = sorted(int(x) for x in a.sizes.split(","))
H, I, NL, E, TOPK, STEP = 2048, 768, 48, 128, 8, 32
os.makedirs(a.work, exist_ok=True)
os.makedirs(a.out, exist_ok=True)
index = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
handles = {}
vlt = Vlt(a.model)
t0 = time.time()


def hf(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.hf, f), framework="pt")
    return handles[f].get_tensor(name).float()


def eng(layer, e, which):
    return torch.from_numpy(vlt.linear("blk.%d.e%d.%s" % (layer, e, which))[0].copy())


def encode(text):
    req = os.path.join(a.work, "enc.jsonl")
    with open(req, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": text}) + "\n")
    p = subprocess.run([a.gen, "--model", a.model, "--encode-file", req], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


# every file a held-out probe set is cut from (novel.py and prep_ids.py)
HELD_FILES = ["docs/bitplane_spec.md", "engine/platform.cpp", "docs/INTERFACES.md", "docs/UNPUBLISHED_NOTES.md"]
FIT_FILES = sorted(f for f in glob.glob(os.path.join(ROOT, "engine", "*.cpp")) + glob.glob(os.path.join(ROOT, "docs", "*.md")) +
                   glob.glob(os.path.join(ROOT, "python", "*.py")) if os.path.relpath(f, ROOT).replace(os.sep, "/") not in HELD_FILES
                   and "unicode" not in f and os.path.getsize(f) > 6000)


def segments(files, count):
    segs, k = [], 0
    while len(segs) < count and k < 20 * count:
        path = files[k % len(files)]
        off = (k // len(files)) * a.segment_tokens * 8
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()[off: off + a.segment_tokens * 8]
        k += 1
        if len(text) < a.segment_tokens * 2:
            continue
        ids = encode(text)[: a.segment_tokens]
        if len(ids) >= a.segment_tokens // 2:
            segs.append(ids)
    return segs


def collect(segs):
    norms = [hf("model.layers.%d.post_attention_layernorm.weight" % l) for l in range(NL)]
    acc = [[] for _ in range(NL)]
    prefix = os.path.join(a.work, "dump")
    logits = os.path.join(a.work, "logits.bin")
    for ids in segs:
        T = len(ids)
        for s in (".x", ".xa", ".e"):
            if os.path.exists(prefix + s):
                os.remove(prefix + s)
        subprocess.run(["nice", "-n", "10", a.gen, "--model", a.model, "--threads", "8", "--ctx", "1024", "--ids", ",".join(map(str, ids)),
                        "--dump-logits", logits], check=True, capture_output=True, env=dict(os.environ, VELTRA_DUMP_LAYERS=prefix))
        xa = np.memmap(prefix + ".xa", dtype=np.float32, mode="r").reshape(NL, T, H)
        for l in range(NL):
            h = torch.from_numpy(np.array(xa[l]))
            acc[l].append(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * norms[l])
        del xa
        for s in (".x", ".xa", ".e"):
            os.remove(prefix + s)
        os.remove(logits)
    return [torch.cat(v) for v in acc]


def routed(layer, x):
    p = torch.softmax(x @ hf("model.layers.%d.mlp.gate.weight" % layer).t(), -1)
    top = p.topk(TOPK, -1)
    return top.indices, top.values / top.values.sum(-1, keepdim=True)


def omp_sets(Hf, D, Yf):
    """-> {s: neuron indices} for every s in SIZES from one greedy run (the set at size s is a prefix of the larger ones)."""
    chosen = torch.zeros(I, dtype=torch.bool)
    resid = Yf
    col = Hf.pow(2).sum(0).clamp_min(1e-12)
    got = {}
    while len(got) < len(SIZES):
        nxt = SIZES[len(got)]
        score = (resid.t() @ Hf).pow(2).sum(0) / col
        score += 1e-9 * col * D.pow(2).sum(0)
        score[chosen] = -1.0
        chosen[score.topk(min(STEP, nxt - int(chosen.sum()))).indices] = True
        idx = chosen.nonzero()[:, 0]
        if len(idx) >= nxt:
            got[nxt] = idx
        if len(got) < len(SIZES):
            Z = Hf[:, idx]
            G = Z.t() @ Z
            lam = 1e-2 * G.diagonal().mean() + 1e-8
            resid = Yf - Z @ torch.linalg.solve(G + lam * torch.eye(len(idx)), Z.t() @ Yf)
    return got


segs = segments(FIT_FILES, a.fit_segments)
fit = collect(segs)
print("activations: %d tokens from %d segments (%.0f s)" % (len(fit[0]), len(segs), time.time() - t0), flush=True)
masks = {(sel, s): np.zeros((NL, E, I), dtype=np.uint8) for sel in ("omp", "energy") for s in SIZES}
unfitted = 0
for l in range(NL):
    idx, w = routed(l, fit[l])
    for e in range(E):
        rows = (idx == e).nonzero()
        g, u, D = eng(l, e, "gate"), eng(l, e, "up"), eng(l, e, "down")
        dn = D.pow(2).sum(0)
        if len(rows) < 8:                                  # (almost) never routed in calibration: keep the largest down columns
            unfitted += 1
            order = dn.argsort(descending=True)
            for s in SIZES:
                masks[("omp", s)][l, e, order[:s].numpy()] = 1
                masks[("energy", s)][l, e, order[:s].numpy()] = 1
            continue
        t, rank = rows[:, 0], rows[:, 1]
        x = fit[l][t]
        Hf = torch.nn.functional.silu(x @ g.t()) * (x @ u.t()) * w[t, rank][:, None].sqrt()
        order = (Hf.pow(2).sum(0) * dn).argsort(descending=True)
        got = omp_sets(Hf, D, Hf @ D.t())
        for s in SIZES:
            masks[("energy", s)][l, e, order[:s].numpy()] = 1
            masks[("omp", s)][l, e, got[s].numpy()] = 1
    print("layer %2d done (%.0f s)" % (l, time.time() - t0), flush=True)
for (sel, s), m in masks.items():
    assert (m.sum(axis=2) == s).all(), (sel, s)
    m.tofile(os.path.join(a.out, "%s_s%d.bin" % (sel, s)))
json.dump({"fit_tokens": int(len(fit[0])), "experts_without_calibration": unfitted, "experts": NL * E, "seconds": round(time.time() - t0)},
          open(os.path.join(a.out, "select_info.json"), "w"), indent=2)
print("experts with too little calibration data: %d of %d; wrote %s (%.0f s)" % (unfitted, NL * E, a.out, time.time() - t0))
