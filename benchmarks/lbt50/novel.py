#!/usr/bin/env python3
"""LBT-50 new-method experiments on hf30, all errors on HELD-OUT activations of the running engine.

 M1 dual-resolution ("shadow") experts: the top-R routed experts run in full; every further routed expert runs a small shadow of itself that keeps s of
    its 768 neurons (selected by batched orthogonal matching pursuit) with a down projection REFIT in closed form on calibration activations:
        D' = (Y H_S^T + lam D[:,S]) (H_S H_S^T + lam I)^-1
    gate/up rows of the shadow are the model's own quantised rows; D' is quantised with the same Q4_K quantiser. Reference = the CURRENT model's MoE
    output. Baselines: dropping the tail (moe-k R) and the uniform moe-k with the same bytes.
 M2 learned additive codebooks (2 x 4-bit codes per 4 weights, codebooks shared per 32 columns, input-covariance weighted) vs the engine's own
    Q2_K / Q3_K weights, both measured against the original bf16 weights.
 M3 one-bit rotated estimator with an error certificate for the output head (RaBitQ family): bytes read, survivors, misses of the true argmax.
 M4 temporal coherence: ||h_t - h_(t-1)|| / ||h_t||, the quantity any delta-computation scheme depends on.

  <venv: torch, safetensors>/python novel.py --gen bin/veltra-gen --model M.vlt --hf DIR --work DIR --out out.json
"""
import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
from vlt_weights import Vlt, decode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="12,24,36")
ap.add_argument("--fit-segments", type=int, default=24)
ap.add_argument("--segment-tokens", type=int, default=500)
ap.add_argument("--parts", default="m1,m2,m3", help="M4 always runs")
a = ap.parse_args()
torch.set_num_threads(8)
LAYERS = [int(x) for x in a.layers.split(",")]
H, I, NL, E, TOPK = 2048, 768, 48, 128, 8
os.makedirs(a.work, exist_ok=True)
index = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
handles = {}
vlt = Vlt(a.model)
out = {}
t0 = time.time()


def hf(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.hf, f), framework="pt")
    return handles[f].get_tensor(name).float()


def eng(layer, e, which):
    return torch.from_numpy(vlt.linear("blk.%d.e%d.%s" % (layer, e, which))[0].copy())


def save():
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


# ------------------------------------------------------------ quantiser shared with the exporter
tmp = tempfile.mkdtemp()
so = os.path.join(tmp, "libq4k.so")
subprocess.run(["g++", "-std=c++20", "-O3", "-fno-math-errno", "-shared", "-fPIC", "-I", os.path.join(ROOT, "engine"),
                os.path.join(ROOT, "engine", "q4k_quant.cpp"), "-o", so], check=True)
qlib = ctypes.CDLL(so)
qlib.veltra_q4k_quantize.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]


def q4k_roundtrip(w):
    x = np.ascontiguousarray(w.numpy(), dtype=np.float32)
    m, n = x.shape
    blk = np.empty((m * (n // 256), 148), dtype=np.uint8)
    qlib.veltra_q4k_quantize(x.ctypes.data, blk.shape[0], blk.ctypes.data)
    return torch.from_numpy(decode("q4k", blk).reshape(m, n))


# ------------------------------------------------------------ activations (repository text only; held-out files differ from fit files)
def encode(text):
    req = os.path.join(a.work, "enc.jsonl")
    with open(req, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": text}) + "\n")
    p = subprocess.run([a.gen, "--model", a.model, "--encode-file", req], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


HELD_FILES = ["docs/bitplane_spec.md", "engine/platform.cpp", "docs/INTERFACES.md"]
FIT_FILES = sorted(f for f in glob.glob(os.path.join(ROOT, "engine", "*.cpp")) + glob.glob(os.path.join(ROOT, "docs", "*.md")) +
                   glob.glob(os.path.join(ROOT, "python", "*.py")) if os.path.relpath(f, ROOT).replace(os.sep, "/") not in HELD_FILES
                   and "unicode" not in f and os.path.getsize(f) > 6000)


def segments(files, count, per_file_offset=0):
    segs, k = [], 0
    while len(segs) < count and k < 20 * count:
        path = files[k % len(files)]
        off = per_file_offset + (k // len(files)) * a.segment_tokens * 8
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
    moe = {l: [] for l in LAYERS}
    coh = {l: [] for l in LAYERS}
    last = []
    norms = {l: hf("model.layers.%d.post_attention_layernorm.weight" % l) for l in LAYERS}
    for ids in segs:
        T = len(ids)
        prefix = os.path.join(a.work, "dump")
        for s in (".x", ".xa", ".e"):
            if os.path.exists(prefix + s):
                os.remove(prefix + s)
        subprocess.run(["nice", "-n", "10", a.gen, "--model", a.model, "--threads", "8", "--ctx", "1024", "--ids", ",".join(map(str, ids)), "--dump-logits",
                        os.path.join(a.work, "logits.bin")], check=True, capture_output=True, env=dict(os.environ, VELTRA_DUMP_LAYERS=prefix))
        x = np.memmap(prefix + ".x", dtype=np.float32, mode="r").reshape(NL, T, H)
        xa = np.memmap(prefix + ".xa", dtype=np.float32, mode="r").reshape(NL, T, H)
        for l in LAYERS:
            h = torch.from_numpy(np.array(xa[l]))
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * norms[l]
            moe[l].append(h)
            coh[l].append(((h[1:] - h[:-1]).norm(dim=1) / h[1:].norm(dim=1)))
        last.append(torch.from_numpy(np.array(x[NL - 1])))
        del x, xa
        for s in (".x", ".xa", ".e"):
            os.remove(prefix + s)
        os.remove(os.path.join(a.work, "logits.bin"))
    return {l: torch.cat(v) for l, v in moe.items()}, {l: torch.cat(v) for l, v in coh.items()}, torch.cat(last)


fit_segs = segments(FIT_FILES, a.fit_segments)
held_segs = segments([os.path.join(ROOT, f) for f in HELD_FILES], 3)
fit, _, _ = collect(fit_segs)
held, coherence, held_last = collect(held_segs)
out["fit_tokens"], out["held_tokens"] = int(len(fit[LAYERS[0]])), int(len(held[LAYERS[0]]))
print("activations: %d fit and %d held-out tokens (%.0f s)" % (out["fit_tokens"], out["held_tokens"], time.time() - t0), flush=True)

# ------------------------------------------------------------ M4
out["M4_temporal_coherence"] = {str(l): {"median_rel_step": round(float(c.median()), 3), "p10": round(float(c.quantile(0.1)), 3)} for l, c in coherence.items()}
print("M4 relative step ||h_t - h_t-1|| / ||h_t|| per layer:", json.dumps(out["M4_temporal_coherence"]), flush=True)
save()


# ------------------------------------------------------------ M1
def routed(layer, x):
    logits = x @ hf("model.layers.%d.mlp.gate.weight" % layer).t()
    p = torch.softmax(logits, -1)
    top = p.topk(TOPK, -1)
    return top.indices, top.values / top.values.sum(-1, keepdim=True)


def neurons(g, u, x):
    return torch.nn.functional.silu(x @ g.t()) * (x @ u.t())      # [T, I]


def omp_select(Hf, D, Yf, s, step=32):
    """batched orthogonal matching pursuit over neurons: repeatedly add the `step` neurons that explain most of the current residual."""
    chosen = torch.zeros(I, dtype=torch.bool)
    resid = Yf.clone()
    col = Hf.pow(2).sum(0).clamp_min(1e-12)
    idx = None
    while int(chosen.sum()) < s:
        score = (resid.t() @ Hf).pow(2).sum(0) / col                # gain of projecting the residual on neuron i
        score += 1e-9 * col * D.pow(2).sum(0)                       # tie-break when the residual is already tiny
        score[chosen] = -1.0
        chosen[score.topk(min(step, s - int(chosen.sum()))).indices] = True
        idx = chosen.nonzero()[:, 0]
        Z = Hf[:, idx]
        G = Z.t() @ Z
        lam = 1e-2 * G.diagonal().mean()
        W = torch.linalg.solve(G + lam * torch.eye(len(idx)), Z.t() @ Yf)
        resid = Yf - Z @ W
    return idx


SIZES = [128, 256]
FULL_RANKS = [1, 2, 3, 4]
out["M1_shadow_experts"] = {}
for l in (LAYERS if "m1" in a.parts else []):
    idx_f, w_f = routed(l, fit[l])
    idx_h, w_h = routed(l, held[l])
    Th = len(held[l])
    y_full = torch.zeros(Th, H)
    contrib = torch.zeros(Th, TOPK, H)                              # per held token and rank: w * f_e(x) of the current model
    shadow = {s: torch.zeros(Th, TOPK, H) for s in SIZES}
    shadow_naive = {s: torch.zeros(Th, TOPK, H) for s in SIZES}     # same neurons, original down columns (no refit): shows what the refit adds
    used = {s: 0 for s in SIZES}
    for e in range(E):
        rows_h = (idx_h == e).nonzero()
        if not len(rows_h):
            continue
        g, u, D = eng(l, e, "gate"), eng(l, e, "up"), eng(l, e, "down")      # D: [H, I]
        t, rank = rows_h[:, 0], rows_h[:, 1]
        hh = neurons(g, u, held[l][t])
        contrib[t, rank] = (hh @ D.t()) * w_h[t, rank][:, None]
        rows_f = (idx_f == e).nonzero()
        tf, rf = rows_f[:, 0], rows_f[:, 1]
        if len(tf) >= 8:
            Hf = neurons(g, u, fit[l][tf]) * w_f[tf, rf][:, None].sqrt()
            Yf = Hf @ D.t()
        for s in SIZES:
            if len(tf) < 8:                                          # never routed in calibration: keep the largest down columns
                sel = D.pow(2).sum(0).topk(s).indices
                D2 = D[:, sel]
            else:
                sel = omp_select(Hf, D, Yf, s)
                Z = Hf[:, sel]
                G = Z.t() @ Z
                lam = 0.05 * G.diagonal().mean() + 1e-8
                D2 = torch.linalg.solve(G + lam * torch.eye(s), Z.t() @ Yf + lam * D[:, sel].t()).t()    # [H, s], pulled toward the original columns
                used[s] += 1
            if s % 256 == 0:
                D2q = q4k_roundtrip(D2)
            else:                                                    # 128 columns: the engine's 4-bit nibble format; emulate with per-128-group 4 bit
                lo = D2.amin(1, keepdim=True)
                step = (D2.amax(1, keepdim=True) - lo).clamp_min(1e-12) / 15
                D2q = ((D2 - lo) / step).round() * step + lo
            shadow[s][t, rank] = (hh[:, sel] @ D2q.t()) * w_h[t, rank][:, None]
            shadow_naive[s][t, rank] = (hh[:, sel] @ D[:, sel].t()) * w_h[t, rank][:, None]
    y_full = contrib.sum(1)
    den = float(y_full.pow(2).sum())
    err = lambda y: round(float((y - y_full).pow(2).sum()) / den, 5)
    res = {"shadow_experts_fitted": used}

    def drop_tail(k):   # moe-k k: keep the first k ranks, renormalise their weights
        keep = contrib[:, :k].sum(1)
        scale = (1.0 / w_h[:, :k].sum(1))[:, None]
        return keep * scale

    res["drop_tail"] = {str(k): err(drop_tail(k)) for k in range(1, 8)}
    res["shadow"] = {}
    for s in SIZES:
        # bytes of a shadow expert relative to a full one: s of 768 gate/up rows (same format) plus a Q4_K / 4-bit down of s columns
        rel = (2 * s * 2.625 + s * 4.625) / (2 * 768 * 2.625 + 768 * 3.66)
        for R in FULL_RANKS:
            y = contrib[:, :R].sum(1) + shadow[s][:, R:].sum(1)
            yn = contrib[:, :R].sum(1) + shadow_naive[s][:, R:].sum(1)
            eq = R + (TOPK - R) * rel                                # expert reads in units of one full expert
            res["shadow"]["R%d_s%d" % (R, s)] = {"err": err(y), "err_without_refit": err(yn), "expert_reads_equivalent": round(eq, 2),
                                                 "err_drop_tail_same_R": res["drop_tail"][str(R)]}
    out["M1_shadow_experts"][str(l)] = res
    print("M1 layer %2d drop-tail err: %s" % (l, " ".join("k%s=%.3f" % kv for kv in res["drop_tail"].items())), flush=True)
    for k, v in res["shadow"].items():
        print("M1 layer %2d shadow %-8s reads=%.2f experts  err %.4f (no refit %.4f) | drop tail at same R %.4f" % (
            l, k, v["expert_reads_equivalent"], v["err"], v["err_without_refit"], v["err_drop_tail_same_R"]), flush=True)
    save()
    del contrib, shadow, shadow_naive


# ------------------------------------------------------------ M2
def additive_codebooks(W, X, books, iters=12):
    """W [m, n] original; X [N, n] calibration inputs. Groups of 4 weights; `books` codebooks of 16 vectors shared by each 32-column block; one
    8-bit-quantised scale per (row, 32 columns). Fitted in the space whitened by the pooled 4 x 4 input covariance of the block."""
    m, n = W.shape
    What = torch.empty_like(W)
    for c0 in range(0, n, 32):
        Xb = X[:, c0:c0 + 32].reshape(len(X), 8, 4)
        cov = torch.einsum("tgi,tgj->ij", Xb, Xb) / (len(X) * 8)
        ev, V = torch.linalg.eigh(cov + 1e-6 * torch.eye(4) * cov.diagonal().mean())
        sq, isq = (V * ev.clamp_min(1e-12).sqrt()) @ V.t(), (V * ev.clamp_min(1e-12).rsqrt()) @ V.t()
        v = W[:, c0:c0 + 32].reshape(m, 8, 4) @ sq                    # whitened weights
        sc = v.reshape(m, 32).pow(2).mean(1).sqrt().clamp_min(1e-12)
        lmax = sc.max()
        sc = ((sc / lmax).log2().clamp(-12, 0) * (255 / 12)).round() * (12 / 255)
        sc = lmax * 2.0 ** sc                                         # 8-bit log scale
        pts = (v / sc[:, None, None]).reshape(m * 8, 4)
        weight = sc.repeat_interleave(8).pow(2)
        g = torch.Generator().manual_seed(c0)
        cbs, codes, resid = [], [], pts.clone()
        for b in range(books):                                        # residual k-means initialisation
            C = resid[torch.randperm(len(resid), generator=g)[:16]].clone()
            for _ in range(iters):
                a_ = torch.cdist(resid, C).argmin(1)
                num = torch.zeros(16, 4).index_add_(0, a_, resid * weight[:, None])
                cnt = torch.zeros(16).index_add_(0, a_, weight)
                C = torch.where(cnt[:, None] > 0, num / cnt[:, None].clamp_min(1e-12), C)
            a_ = torch.cdist(resid, C).argmin(1)
            cbs.append(C)
            codes.append(a_)
            resid = resid - C[a_]
        for _ in range(3):                                            # joint refinement: least-squares codebooks, then greedy re-assignment
            A = torch.zeros(len(pts), 16 * books)
            for b in range(books):
                A[torch.arange(len(pts)), 16 * b + codes[b]] = 1.0
            Aw = A * weight[:, None]
            # rank deficient by construction (a constant can move between two books): float64 and a ridge relative to the normal matrix
            Ad, Awd = A.double(), Aw.double()
            M = Ad.t() @ Awd
            M += (1e-6 * M.diagonal().mean() + 1e-20) * torch.eye(16 * books, dtype=torch.float64)
            sol = torch.linalg.solve(M, Awd.t() @ pts.double()).float()
            cbs = [sol[16 * b:16 * (b + 1)] for b in range(books)]
            for b in range(books):
                rest = pts - sum(cbs[o][codes[o]] for o in range(books) if o != b) if books > 1 else pts
                codes[b] = torch.cdist(rest, cbs[b]).argmin(1)
        rec = sum(cbs[b][codes[b]] for b in range(books)).reshape(m, 8, 4) * sc[:, None, None]
        What[:, c0:c0 + 32] = (rec @ isq).reshape(m, 32)
    return What


out["M2_additive_codebooks"] = {}
for l in (LAYERS if "m2" in a.parts else []):
    idx_f, _ = routed(l, fit[l])
    idx_h, _ = routed(l, held[l])
    top = [int(i) for i in torch.bincount(idx_h.reshape(-1), minlength=E).argsort(descending=True)[:6]]
    acc = {k: [0.0, 0.0] for k in ("gate_engine", "gate_avq2", "down_engine", "down_avq2", "down_avq3")}
    bpw = {}
    for e in top:
        for which in ("gate", "down"):
            ref = hf("model.layers.%d.mlp.experts.%d.%s_proj.weight" % (l, e, which))
            cur, nbytes = vlt.linear("blk.%d.e%d.%s" % (l, e, which))
            bpw[which] = round(8 * nbytes / cur.size, 3)
            tf = (idx_f == e).any(1).nonzero()[:, 0]
            th = (idx_h == e).any(1).nonzero()[:, 0]
            if len(tf) < 40 or len(th) < 4:
                continue
            if which == "gate":
                Xf, Xh = fit[l][tf], held[l][th]
            else:
                g, u = eng(l, e, "gate"), eng(l, e, "up")
                Xf, Xh = neurons(g, u, fit[l][tf]), neurons(g, u, held[l][th])
            target = Xh @ ref.t()
            for key, What in [(which + "_engine", torch.from_numpy(cur.copy()))] + [(which + "_avq%d" % b, additive_codebooks(ref, Xf, b)) for b in ((2,) if which == "gate" else (2, 3))]:
                acc[key][0] += float((Xh @ What.t() - target).pow(2).sum())
                acc[key][1] += float(target.pow(2).sum())
    res = {k: round(v[0] / v[1], 5) for k, v in acc.items() if v[1] > 0}
    res["engine_bits_per_weight"] = bpw
    res["avq_bits_per_weight"] = {"avq2": 2.25, "avq3": 3.25}
    out["M2_additive_codebooks"][str(l)] = res
    print("M2 layer %2d output error vs bf16: %s" % (l, json.dumps(res)), flush=True)
    save()

# ------------------------------------------------------------ M3
W = hf("lm_head.weight")
hn = held_last * torch.rsqrt(held_last.pow(2).mean(-1, keepdim=True) + 1e-6) * hf("model.norm.weight")
hn = hn[::max(1, len(hn) // 400)]
V_, D = W.shape
gen = torch.Generator().manual_seed(1)
signs = (torch.randint(0, 2, (3, D), generator=gen).float() * 2 - 1)


def rotate(x):   # three rounds of random signs + orthonormal Walsh-Hadamard: O(D log D), no stored matrix
    for r in range(3):
        x = x * signs[r]
        h = 1
        while h < D:
            y = x.reshape(-1, D // (2 * h), 2, h)
            x = torch.cat([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], dim=2).reshape(-1, D)
            h *= 2
        x = x / (D ** 0.5)
    return x


norms = W.norm(dim=1).clamp_min(1e-12)
R = torch.cat([rotate(W[i:i + 8192] / norms[i:i + 8192, None]) for i in range(0, V_, 8192)])
B = torch.where(R >= 0, 1.0, -1.0) / (D ** 0.5)
c = (B * R).sum(1)                                                    # <code, rotated unit row>, stored per row
truth = (hn @ W.t()).argmax(1)
Q = rotate(hn / hn.norm(dim=1, keepdim=True))
qmax = Q.abs().amax(1, keepdim=True)
Q4 = (Q / qmax * 7).round().clamp(-7, 7) / 7 * qmax                   # 4-bit query, as a bit-plane kernel would use it
res = {"head_bytes_mb_1bit": round(V_ * D / 8 / 1e6 + V_ * 8 / 1e6, 1), "head_bytes_mb_screen2": 92, "head_bytes_mb_screen4": 157, "samples": len(hn), "eps0": {}}
est = (Q4 @ B.t()) / c[None, :] * norms[None, :] * hn.norm(dim=1, keepdim=True)
base_err = torch.sqrt((1 - c ** 2).clamp_min(0) / c ** 2 / (D - 1)) * norms
for eps0 in (1.9, 3.0, 4.0):
    half = eps0 * base_err[None, :] * hn.norm(dim=1, keepdim=True)
    lower = (est - half).amax(1, keepdim=True)
    surv = (est + half) >= lower
    n = surv.sum(1).float()
    missed = int((~surv[torch.arange(len(hn)), truth]).sum())
    res["eps0"][str(eps0)] = {"survivors_median": int(n.median()), "survivors_p99": int(n.quantile(0.99)), "argmax_missed": missed}
order = est.argsort(dim=1, descending=True)
for N in (64, 256, 1024):
    res["recall_top%d" % N] = round(float((order[:, :N] == truth[:, None]).any(1).float().mean()), 4)
out["M3_one_bit_head"] = res
print("M3 one-bit head estimator:", json.dumps(res), flush=True)
save()
print("done in %.0f s, wrote %s" % (time.time() - t0, a.out))
