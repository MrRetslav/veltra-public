#!/usr/bin/env python3
"""INT3 g128 asymmetric attention quantiser (GPTQ with real-model Hessians) for Qwen3-30B-A3B.

Input : HF bf16 weights (pread, like export_qwen3_moe_stream.HFSource) + activation dumps written by the engine hook
        (LB_HESS=dir: <dir>/L<ll>_h.f16 = post attn_norm [N][2048] input of q/k/v, <dir>/L<ll>_a.f16 = attention output [N][4096] input of o).
Output: <out>/L<ll>_<q|k|v|o>.npz  with q (uint8 [m, n], 0..7), sm (f16 [m, n/128, 2] = (scale, min); W_hat = q * scale + min)
        and <out>/stats.jsonl : per matrix held-out relative output error for n4-RTN (the current model), i3-RTN, i3-GPTQ (and variants).
Plain column order (no act-order), group 128, damp 0.01, per-group per-row clip search for (scale, min).

  python quant_i3.py --model <dir>/hf30 --cal <dir>/hess_cal --held <dir>/hess_held --out <dir>/i3q [--layers 0:48] [--threads 8]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_qwen3_moe_stream import HFSource, quant_n4, DEFAULT_CLIPS_N4  # noqa: E402

G = 128
QMAX = 7
SYM = False        # n4 control grid: W_hat = (q - 7.5) * scale, q in 0..15 (set by --fmt n4)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------- group parameter search
CLIPS_I3 = tuple(np.round(np.arange(0.55, 1.0001, 0.025), 4))


def find_params(w: torch.Tensor, hw: torch.Tensor | None = None, clips=None, f16: bool = True):
    """w [m, g] -> (scale [m,1], lo [m,1]) minimising (weighted) squared error of q*scale+lo, q in 0..7. scale/lo are rounded to f16."""
    clips = CLIPS_I3 if clips is None else clips
    lo0, hi0 = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    mid, half = (lo0 + hi0) / 2, (hi0 - lo0) / 2
    amax = w.abs().amax(-1, keepdim=True)
    best = None
    wt = hw if hw is not None else None
    for c in clips:
        if SYM:
            s = (amax * (c / 7.5)).clamp_min(2.0 ** -14)
            if f16: s = s.half().float()
            lo = -7.5 * s
        else:
            lo = mid - half * c
            s = (half * (2 * c / QMAX)).clamp_min(2.0 ** -14)
            if f16:
                s = s.half().float(); lo = lo.half().float()
        q = ((w - lo) / s).round_().clamp_(0, QMAX)
        e = (q * s + lo - w).pow_(2)
        err = (e * wt).sum(-1, keepdim=True) if wt is not None else e.sum(-1, keepdim=True)
        if best is None:
            best = (err, s, lo)
        else:
            take = err < best[0]
            best = tuple(torch.where(take, a, b) for a, b in zip((err, s, lo), best))
    return best[1], best[2]


def quant_rtn_i3(W: torch.Tensor, hdiag: torch.Tensor | None = None):
    m, n = W.shape
    w = W.reshape(m, n // G, G)
    hw = hdiag.reshape(1, n // G, G) if hdiag is not None else None
    # params per (row, group)
    s_l, lo_l, q_l = [], [], []
    for g in range(n // G):
        s, lo = find_params(w[:, g], hw[:, g] if hw is not None else None)
        q = ((w[:, g] - lo) / s).round_().clamp_(0, QMAX)
        s_l.append(s); lo_l.append(lo); q_l.append(q)
    q = torch.cat(q_l, 1).to(torch.uint8)
    sm = torch.stack([torch.cat(s_l, 1), torch.cat(lo_l, 1)], -1).half()
    return q, sm


def dequant_i3(q: torch.Tensor, sm: torch.Tensor) -> torch.Tensor:
    m, n = q.shape
    s = sm[..., 0].float().repeat_interleave(G, 1)
    lo = sm[..., 1].float().repeat_interleave(G, 1)
    return q.float() * s + lo


def dequant_n4(W: torch.Tensor) -> torch.Tensor:
    j, gs, _, _ = quant_n4(W, DEFAULT_CLIPS_N4)
    j = torch.from_numpy(j).float() - 7.5
    return j * torch.from_numpy(gs).float().repeat_interleave(G, 1)


# ----------------------------------------------------------------------------- GPTQ
def gptq_i3(W: torch.Tensor, H: torch.Tensor, damp: float = 0.01, wclip: bool = True, block: int = G):
    """W [m, n] f32, H [n, n] f32 (sum x x^T / N) -> (q uint8, sm f16, Wq f32). Column-sequential GPTQ, groups of 128 fixed in plain order."""
    m, n = W.shape
    W = W.clone()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    d = damp * torch.mean(torch.diag(H))
    H += d * torch.eye(n)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    hdiag = torch.diag(H).clone()
    Q = torch.zeros_like(W)
    qs = torch.zeros((m, n), dtype=torch.uint8)
    sm = torch.zeros((m, n // G, 2), dtype=torch.float16)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        g = i1 // G
        s, lo = find_params(W1, hdiag[i1:i2].reshape(1, -1) if wclip else None)
        sm[:, g, 0] = s[:, 0].half(); sm[:, g, 1] = lo[:, 0].half()
        s = s.reshape(m); lo = lo.reshape(m)
        for i in range(i2 - i1):
            w = W1[:, i]
            dd = Hinv1[i, i]
            q = ((w - lo) / s).round_().clamp_(0, QMAX)
            wq = q * s + lo
            Q[:, i1 + i] = wq
            qs[:, i1 + i] = q.to(torch.uint8)
            err = (w - wq) / dd
            if i + 1 < i2 - i1:
                W1[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)
            Err1[:, i] = err
        if i2 < n:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return qs, sm, Q


# ----------------------------------------------------------------------------- activations
def load_x(path: str, n: int, max_rows: int | None = None) -> np.ndarray:
    a = np.memmap(path, dtype=np.float16, mode='r').reshape(-1, n)
    return a if max_rows is None else a[:max_rows]


def gram(x: np.ndarray, chunk: int = 8192) -> tuple[torch.Tensor, int]:
    n = x.shape[1]
    Hs = torch.zeros((n, n), dtype=torch.float32)
    N = 0
    for r in range(0, x.shape[0], chunk):
        xb = torch.from_numpy(np.asarray(x[r:r + chunk], dtype=np.float32))
        xb = torch.nan_to_num(xb, nan=0.0, posinf=60000.0, neginf=-60000.0)
        Hs += xb.T @ xb
        N += xb.shape[0]
    return Hs, N


def rel_err(W: torch.Tensor, Wq: torch.Tensor, Hh: torch.Tensor) -> float:
    D = W - Wq
    return float(((D @ Hh) * D).sum() / ((W @ Hh) * W).sum())


MATS = {'q': ('q_proj', 'h', 2048), 'k': ('k_proj', 'h', 2048), 'v': ('v_proj', 'h', 2048), 'o': ('o_proj', 'a', 4096)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--cal', required=True); ap.add_argument('--held', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--layers', default='0:48'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--damp', type=float, default=0.01); ap.add_argument('--mats', default='q,k,v,o')
    ap.add_argument('--no-wclip', action='store_true'); ap.add_argument('--skip-sink', type=int, default=0, help='drop the first N positions of every 2048-token calibration sequence from H (sink tokens)')
    ap.add_argument('--seqlen', type=int, default=2048)
    ap.add_argument('--fmt', default='i3', choices=['i3', 'n4'], help='n4: control run, GPTQ onto the CURRENT nibble grid (W = (q-7.5)*scale, q 0..15); writes q + scale only')
    a = ap.parse_args()
    global QMAX, SYM, CLIPS_I3
    if a.fmt == 'n4':
        QMAX, SYM = 15, True
        CLIPS_I3 = tuple(np.round(np.arange(0.6, 1.0001, 0.025), 4))
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    src = HFSource(a.model)
    l0, l1 = (int(x) for x in a.layers.split(':'))
    stats = open(os.path.join(a.out, 'stats.jsonl'), 'a')
    for l in range(l0, l1):
        t0 = time.time()
        xs = {}
        for kind, n in (('h', 2048), ('a', 4096)):
            if not any(MATS[m][1] == kind for m in a.mats.split(',')):
                continue
            xc = load_x(f'{a.cal}/L{l:02d}_{kind}.f16', n)
            if a.skip_sink:
                idx = np.arange(xc.shape[0]); keep = (idx % a.seqlen) >= a.skip_sink
                xc = xc[keep]
            H, N = gram(xc)
            xh = load_x(f'{a.held}/L{l:02d}_{kind}.f16', n)
            Hh, Nh = gram(xh)
            xs[kind] = (H / N, Hh / Nh, N, Nh)
        for mk in a.mats.split(','):
            hf, kind, n = MATS[mk]
            W = src.get(f'model.layers.{l}.self_attn.{hf}.weight')
            H, Hh, N, Nh = xs[kind]
            ts = time.time()
            Wn4 = dequant_n4(W)
            e_n4 = rel_err(W, Wn4, Hh)
            q0, sm0 = quant_rtn_i3(W)
            e_rtn = rel_err(W, dequant_i3(q0, sm0), Hh)
            q1, sm1, Wq = gptq_i3(W, H, a.damp, wclip=not a.no_wclip)
            e_gptq = rel_err(W, dequant_i3(q1, sm1), Hh)
            e_gptq_cal = rel_err(W, dequant_i3(q1, sm1), H)
            e_n4_cal = rel_err(W, Wn4, H)
            np.savez(os.path.join(a.out, f'L{l:02d}_{mk}.npz'), q=q1.numpy(), sm=sm1.numpy())
            rec = {'layer': l, 'mat': mk, 'm': W.shape[0], 'n': W.shape[1], 'err_n4_rtn': e_n4, 'err_i3_rtn': e_rtn, 'err_i3_gptq': e_gptq,
                   'cal_err_n4': e_n4_cal, 'cal_err_i3_gptq': e_gptq_cal, 'ratio_gptq_over_n4': e_gptq / e_n4, 'N': N, 'Nh': Nh, 'sec': round(time.time() - ts, 1)}
            stats.write(json.dumps(rec) + '\n'); stats.flush()
            log(f"L{l:02d}.{mk} held rel err n4 {e_n4:.3e} | i3 rtn {e_rtn:.3e} | i3 gptq {e_gptq:.3e} (x{e_gptq / e_n4:.2f} of n4) | cal gptq {e_gptq_cal:.3e} n4 {e_n4_cal:.3e}")
        log(f'layer {l} done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
