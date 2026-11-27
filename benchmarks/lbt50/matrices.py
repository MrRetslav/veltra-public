#!/usr/bin/env python3
"""LBT-50 offline go/no-go tests on hf30 with REAL activations (collected from the running engine) and the ORIGINAL bf16 weights.
All errors are output errors on HELD-OUT activations:  sum ||(W - W') x||^2 / sum ||W x||^2,  weighted with the router weight of the expert.

 A. shared basis for gate+up (FT-SLBF family): activation-weighted shared basis fitted on the fit activations, per-expert coefficients by
    ridge least squares, coefficients optionally quantised to 4 bits; compared with 2/3/4-bit quantisation of the same matrices.
 B. activation-weighted low rank for the attention matrices q, k, v:  W ~ U (U^T W).
 C. certified block skipping on down_proj: skip 32-neuron blocks while  sum_S sigma_max(W[:,S]) * ||h_S||  <= eps * ||W h||  (a guaranteed
    bound), with the natural neuron order and with an offline order by mean energy; also the share that could be skipped if the true error
    were known (oracle, greedy), to show how loose the guarantee is.

  <venv: torch, safetensors>/python matrices.py --gen bin/veltra-gen --model M.vlt --hf DIR --ids DIR --work DIR --out out.json
"""
import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="2,12,24,36,45")
a = ap.parse_args()
torch.set_num_threads(8)
LAYERS = [int(x) for x in a.layers.split(",")]
H, NL, E, TOPK = 2048, 48, 128, 8
os.makedirs(a.work, exist_ok=True)
index = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
handles = {}


def tensor(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.hf, f), framework="pt")
    return handles[f].get_tensor(name).float()


def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


# ---------------------------------------------------------------- activations from the engine
def collect(kind):
    attn = {l: [] for l in LAYERS}
    moe = {l: [] for l in LAYERS}
    for path in sorted(glob.glob(os.path.join(a.ids, kind + "_*.ids"))):
        ids = open(path).read()
        T = len(ids.split(","))
        prefix = os.path.join(a.work, "dump")
        for s in (".x", ".xa", ".e"):
            if os.path.exists(prefix + s):
                os.remove(prefix + s)
        subprocess.run(["nice", "-n", "10", a.gen, "--model", a.model, "--threads", "8", "--ctx", "2048", "--ids", ids, "--dump-logits",
                        os.path.join(a.work, "logits.bin")], check=True, capture_output=True, env=dict(os.environ, VELTRA_DUMP_LAYERS=prefix))
        x = np.memmap(prefix + ".x", dtype=np.float32, mode="r").reshape(NL, T, H)     # after each layer
        xa = np.memmap(prefix + ".xa", dtype=np.float32, mode="r").reshape(NL, T, H)   # after each attention residual
        for l in LAYERS:
            moe[l].append(torch.from_numpy(np.array(xa[l])))
            attn[l].append(torch.from_numpy(np.array(x[l - 1])))
        del x, xa
        for s in (".x", ".xa", ".e"):
            os.remove(prefix + s)
        os.remove(os.path.join(a.work, "logits.bin"))
    for l in LAYERS:
        moe[l] = rmsnorm(torch.cat(moe[l]), tensor("model.layers.%d.post_attention_layernorm.weight" % l))
        attn[l] = rmsnorm(torch.cat(attn[l]), tensor("model.layers.%d.input_layernorm.weight" % l))
    return attn, moe


def quant(w, bits, group=16):
    g = w.reshape(w.shape[0], -1, group)
    lo = g.amin(-1, keepdim=True)
    step = (g.amax(-1, keepdim=True) - lo).clamp_min(1e-12) / (2 ** bits - 1)
    return (((g - lo) / step).round() * step + lo).reshape(w.shape)


def routing(layer, x):
    p = torch.softmax(x @ tensor("model.layers.%d.mlp.gate.weight" % layer).t(), -1)
    top = p.topk(TOPK, -1)
    w = top.values / top.values.sum(-1, keepdim=True)
    full = torch.zeros_like(p)
    full.scatter_(1, top.indices, w)
    return full      # [T, E] router weight of the routed experts, 0 elsewhere


def down_input(layer, e, x):
    g = tensor("model.layers.%d.mlp.experts.%d.gate_proj.weight" % (layer, e))
    u = tensor("model.layers.%d.mlp.experts.%d.up_proj.weight" % (layer, e))
    return torch.nn.functional.silu(x @ g.t()) * (x @ u.t())


t0 = time.time()
fit_attn, fit_moe = collect("fit")
held_attn, held_moe = collect("held")
print("activations: fit %d, held-out %d tokens (%.0f s)" % (len(fit_moe[LAYERS[0]]), len(held_moe[LAYERS[0]]), time.time() - t0), flush=True)
out = {"layers": {}, "fit_tokens": len(fit_moe[LAYERS[0]]), "held_tokens": len(held_moe[LAYERS[0]])}
RANKS = [192, 384, 576, 768, 1152]
EPS = [0.01, 0.05, 0.1, 0.2]

for l in LAYERS:
    res = {}
    xf, xh = fit_moe[l].double(), held_moe[l].double()
    pf, ph = routing(l, fit_moe[l]).double(), routing(l, held_moe[l]).double()
    # ---- A: shared input basis for gate+up
    cov = xf.t() @ xf / len(xf)
    ev, V = torch.linalg.eigh(cov)
    ev = ev.clamp_min(ev.max() * 1e-6)
    sqrtc, isqrtc = (V * ev.sqrt()) @ V.t(), (V * ev.rsqrt()) @ V.t()
    M = torch.zeros(H, H, dtype=torch.float64)
    mats = []
    for e in range(E):
        w = torch.cat([tensor("model.layers.%d.mlp.experts.%d.%s_proj.weight" % (l, e, n)) for n in ("gate", "up")]).double()
        M += float(pf[:, e].sum()) * (w.t() @ w)
        mats.append(w.float())
    evals, evecs = torch.linalg.eigh(sqrtc @ M @ sqrtc)
    order = evals.argsort(descending=True)
    quant_err = {b: [0.0, 0.0] for b in (2, 3, 4)}
    for b in (2, 3, 4):
        for e in range(E):
            sel = ph[:, e] > 0
            if not sel.any():
                continue
            w = mats[e].double()
            diff = (quant(mats[e], b).double() - w) @ xh[sel].t()
            quant_err[b][0] += float((diff.pow(2).sum(0) * ph[sel, e]).sum())
            quant_err[b][1] += float(((w @ xh[sel].t()).pow(2).sum(0) * ph[sel, e]).sum())
    res["quant_output_err"] = {str(b): round(v[0] / v[1], 5) for b, v in quant_err.items()}
    bases = {}
    for r in RANKS:
        B = evecs[:, order[:r]].t() @ isqrtc                      # r x H
        bases[r] = (B, xf @ B.t(), xh @ B.t(), B @ cov, {0: 0.0, 4: 0.0})
    den = 0.0
    for e in range(E):                                            # experts outside: the large products are shared by all ranks
        sel_h = ph[:, e] > 0
        if not sel_h.any():
            continue
        w = mats[e].double()
        sel_f = pf[:, e] > 0
        root = pf[sel_f, e].sqrt()[:, None]
        tgt = (xf[sel_f] @ w.t()) * root
        y = xh[sel_h] @ w.t()
        den += float((y.pow(2).sum(1) * ph[sel_h, e]).sum())
        covw = cov @ w.t()
        for r, (B, zf, zh, Bcov, num) in bases.items():
            # ridge least squares on this expert's own routed fit tokens, shrunk toward the solution under the pooled covariance
            z = zf[sel_f] * root
            lam = 4.0
            A = torch.linalg.solve(z.t() @ z + lam * (Bcov @ B.t()) + 1e-9 * torch.eye(r, dtype=torch.float64), z.t() @ tgt + lam * (B @ covw)).t()
            for bits in (0, 4):
                Aq = A
                if bits:
                    pad = (16 - r % 16) % 16
                    Aq = quant(torch.nn.functional.pad(A.float(), (0, pad)), bits)[:, :r].double()
                d = zh[sel_h] @ Aq.t() - y
                num[bits] += float((d.pow(2).sum(1) * ph[sel_h, e]).sum())
    res["shared_basis"] = {}
    for r, (_, _, _, _, num) in bases.items():
        # bytes of gate+up per token and layer: 8 routed experts of 1536 x r coefficients at 4.5 bits, plus the shared r x 2048 INT8 basis
        orig_mb = 8 * 1536 * 2048 * 2.625 / 8 / 1e6
        new_mb = (8 * 1536 * r * 4.5 / 8 + r * 2048) / 1e6
        res["shared_basis"][str(r)] = {"err_f32_coeffs": round(num[0] / den, 5), "err_4bit_coeffs": round(num[4] / den, 5),
                                       "bytes_ratio_vs_q2k": round(new_mb / orig_mb, 3)}
    del bases
    print("layer %2d gate+up: quant err 2b %.4f 3b %.4f 4b %.4f | shared basis (4-bit coeffs) %s" % (
        l, res["quant_output_err"]["2"], res["quant_output_err"]["3"], res["quant_output_err"]["4"],
        " ".join("r%s=%.3f(x%.2f)" % (k, v["err_4bit_coeffs"], v["bytes_ratio_vs_q2k"]) for k, v in res["shared_basis"].items())), flush=True)
    del mats
    # ---- B: attention matrices, activation-weighted low rank
    af, ah = fit_attn[l].double(), held_attn[l].double()
    ca = af.t() @ af / len(af)
    ev2, V2 = torch.linalg.eigh(ca)
    sq = (V2 * ev2.clamp_min(0).sqrt()) @ V2.t()
    res["attention_lowrank"] = {}
    for name in ("q", "k", "v"):
        w = tensor("model.layers.%d.self_attn.%s_proj.weight" % (l, name))
        y = ah @ w.t().double()
        den = float(y.pow(2).sum())
        U, _, _ = torch.linalg.svd((w.double() @ sq), full_matrices=False)
        m, n = w.shape
        row = {"quant3_err": round(float((ah @ (quant(w, 3) - w).t().double()).pow(2).sum()) / den, 5),
               "quant4_err": round(float((ah @ (quant(w, 4) - w).t().double()).pow(2).sum()) / den, 5), "rank": {}}
        for r in [x for x in (64, 128, 256, 384, 512) if x < min(m, n)]:
            Ur = U[:, :r]
            d = y @ Ur @ Ur.t() - y
            row["rank"][str(r)] = {"err": round(float(d.pow(2).sum()) / den, 5), "params_ratio": round(r * (m + n) / (m * n), 3)}
        res["attention_lowrank"][name] = row
        print("layer %2d %s %dx%d: 3-bit err %.4f 4-bit %.4f | rank %s" % (l, name, m, n, row["quant3_err"], row["quant4_err"],
              " ".join("r%s=%.4f(x%.2f)" % (k, v["err"], v["params_ratio"]) for k, v in row["rank"].items())), flush=True)
    # ---- C: certified block skipping on down_proj
    stats = {o: {str(eps): [0.0, 0.0, 0.0, 0] for eps in EPS} for o in ("natural", "energy_order")}   # certified share, true err, oracle share, n
    rh = routing(l, held_moe[l])
    rf = routing(l, fit_moe[l])
    experts = [int(i) for i in (rf > 0).sum(0).argsort(descending=True)[:24]]          # the 24 most routed experts of the fit set
    for e in experts:
        sel_h, sel_f = rh[:, e] > 0, rf[:, e] > 0
        if sel_h.sum() < 2 or sel_f.sum() < 2:
            continue
        W = tensor("model.layers.%d.mlp.experts.%d.down_proj.weight" % (l, e))         # 2048 x 768
        hh = down_input(l, e, held_moe[l][sel_h])
        hf = down_input(l, e, fit_moe[l][sel_f])
        yy = hh @ W.t()
        ynorm = yy.norm(dim=1).clamp_min(1e-9)
        for order_name in ("natural", "energy_order"):
            perm = torch.arange(768) if order_name == "natural" else (hf.pow(2).mean(0) * W.pow(2).sum(0)).argsort()
            Wp, hp = W[:, perm], hh[:, perm]
            blocks = Wp.reshape(H, 24, 32).permute(1, 0, 2)                             # 24 x 2048 x 32
            sigma = torch.linalg.matrix_norm(blocks, ord=2)                             # 24
            contrib = torch.stack([hp[:, 32 * b:32 * b + 32] @ blocks[b].t() for b in range(24)], 1)   # T x 24 x 2048
            bound = sigma[None, :] * hp.reshape(len(hp), 24, 32).norm(dim=2)            # T x 24, guaranteed ||contrib|| <= bound
            srt = bound.argsort(dim=1)
            cum_bound = bound.gather(1, srt).cumsum(1)
            ordered = contrib.gather(1, srt[:, :, None].expand(-1, -1, H))
            true_err = ordered.cumsum(1).norm(dim=2) / ynorm[:, None]                   # T x 24: error after skipping the first i+1 blocks
            for eps in EPS:
                k_cert = (cum_bound <= eps * ynorm[:, None]).sum(1)
                ok = true_err <= eps
                k_orc = torch.where(ok, torch.arange(1, 25)[None, :], torch.zeros(1, 24, dtype=torch.long)).amax(1)
                err_cert = torch.where(k_cert > 0, true_err.gather(1, (k_cert - 1).clamp_min(0)[:, None])[:, 0], torch.zeros(len(hp)))
                assert float((err_cert - eps).max()) <= 1e-3, "certificate violated"
                st = stats[order_name][str(eps)]
                st[0] += float(k_cert.sum()) / 24
                st[1] += float(err_cert.sum())
                st[2] += float(k_orc.sum()) / 24
                st[3] += len(hp)
    res["certified_skip"] = {o: {eps: {"certified_share": round(v[0] / v[3], 4), "true_err_mean": round(v[1] / v[3], 5), "oracle_share": round(v[2] / v[3], 4),
                                       "samples": v[3]} for eps, v in d.items() if v[3]} for o, d in stats.items()}
    for o, d in res["certified_skip"].items():
        print("layer %2d skip %-12s %s" % (l, o, " ".join("eps %s: certified %.1f%% oracle %.1f%%" % (e, 100 * v["certified_share"], 100 * v["oracle_share"]) for e, v in d.items())), flush=True)
    out["layers"][str(l)] = res
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("layer %d done (%.0f s since start)" % (l, time.time() - t0), flush=True)
