#!/usr/bin/env python3
"""Layer-by-layer STREAMING exporter: HF Qwen3-MoE safetensors (e.g. Qwen3-30B-A3B-Instruct-2507) -> Veltra `.vlt`.

Never materialises the model: tensors are read lazily (safetensors safe_open), quantised in a thread pool and appended to the
file by lbt_stream.LbtStreamWriter. Peak RAM ~ 2 layers of quantised output (~1.5 GB) + the f32 working set of the workers.

  python export_qwen3_moe_stream.py --model <dir>/hf30 --out <dir>/q30b_n4.vlt [--fmt n4] [--threads 8]

Formats (docs/qwen3_moe_export.md):
  n4    nibble-INT4 g128 (docs/bitplane_spec.md section 5): `.n4` U8 + `.gs` F16 [m, n/128]; RTN + per-group MSE-optimal clip search.
  i2a   INT2 asym g128 (unsigned {0..3}, f16 scale+min per group; layout of scripts/bench_repr/int2_kern.inl): `.i2` U8 + `.i2s` F16.
        NOT yet readable by the engine - only produced through `--fmt mixed`.
  mixed per-tensor selector (`select_fmt`): by default everything n4; `--i2-layers A:B --i2-mats gate,up` switch those experts to i2a.
Tensor names follow the engine contract: blk.i.{attn_norm,ffn_norm,q_norm,k_norm,q,k,v,o,moe_gate,e{e}.{gate,up,down}},
tok_embeddings(+.scale, i8), UNTIED output(+.scale, i8; --head f16 for f16), output_norm, tokenizer.json, tokenizer_config.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys
import threading
import time

import numpy as np
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"      # parallelism comes from our worker threads, not from intra-op pools
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lbt_format as lbt                      # noqa: E402
import nibble_lbt as nib                      # noqa: E402
from lbt_stream import LbtStreamWriter        # noqa: E402

G = 128
DEFAULT_CLIPS_N4 = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
DEFAULT_CLIPS_I2 = (1.0, 0.9, 0.8, 0.7, 0.6)
ELEM_CHUNK = 1 << 20          # elements per quantisation chunk (cache friendly)
BPW = {"n4": 4.0 + 16.0 / G, "i2a": 2.0 + 32.0 / G}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- encoders (W f32 [m, n] -> tensors)
def _chunks(m: int, n: int, mult: int = 1):
    rows = max(mult, (ELEM_CHUNK // n) // mult * mult)
    return [(r, min(m, r + rows)) for r in range(0, m, rows)]


def quant_n4(W: torch.Tensor, clips=DEFAULT_CLIPS_N4):
    """-> (j uint8 [m, n] level 0..15, gs f16 [m, n/128], sq_err, sq_norm). W_hat = gs * (j - 7.5)."""
    m, n = W.shape
    assert n % G == 0, "in_features must be a multiple of 128"
    js, gss, se, sn = [], [], 0.0, 0.0
    for r0, r1 in _chunks(m, n):
        w = W[r0:r1].reshape(r1 - r0, n // G, G)
        amax = w.abs().amax(-1, keepdim=True)
        best_err = best_s = best_j = None
        for c in clips:
            s = (amax * (c / 7.5)).clamp_min(2.0 ** -14).half().float()
            j = (w / s + 7.5).round_().clamp_(0, 15)
            err = ((j - 7.5) * s - w).pow_(2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_s, best_j = err, s, j
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s = torch.where(take, s, best_s)
                best_j = torch.where(take, j, best_j)
        js.append(best_j.to(torch.uint8).reshape(r1 - r0, n))
        gss.append(best_s.reshape(r1 - r0, n // G).half())
        se += float(best_err.sum())
        sn += float(w.pow(2).sum())
    return torch.cat(js).numpy(), torch.cat(gss).numpy(), se, sn


def enc_n4(base: str, W: torch.Tensor, clips):
    j, gs, se, sn = quant_n4(W, clips)
    m, n = j.shape
    return [(f"{base}.n4", nib.pack_n4(j), lbt.U8, (m, n)), (f"{base}.gs", gs, lbt.F16, gs.shape)], se, sn


def quant_i2a(W: torch.Tensor, clips=DEFAULT_CLIPS_I2):
    """-> (q uint8 [m, n] in 0..3, sm f16 [m, n/128, 2] = (scale, min), sq_err, sq_norm). W_hat = q * scale + min."""
    m, n = W.shape
    assert n % G == 0
    qs, sms, se, sn = [], [], 0.0, 0.0
    for r0, r1 in _chunks(m, n):
        w = W[r0:r1].reshape(r1 - r0, n // G, G)
        lo0, hi0 = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
        mid, half = (lo0 + hi0) / 2, (hi0 - lo0) / 2
        best = None
        for c in clips:
            lo = mid - half * c
            s = (half * (2 * c / 3)).clamp_min(2.0 ** -14).half().float()
            lo = lo.half().float()
            q = ((w - lo) / s).round_().clamp_(0, 3)
            err = (q * s + lo - w).pow_(2).sum(-1, keepdim=True)
            if best is None:
                best = (err, s, lo, q)
            else:
                take = err < best[0]
                best = tuple(torch.where(take, a, b) for a, b in zip((err, s, lo, q), best))
        err, s, lo, q = best
        qs.append(q.to(torch.uint8).reshape(r1 - r0, n))
        sms.append(torch.cat([s, lo], -1).half())
        se += float(err.sum())
        sn += float(w.pow(2).sum())
    return torch.cat(qs).numpy(), torch.cat(sms).numpy(), se, sn


def pack_i2a(q: np.ndarray, sm: np.ndarray):
    """q uint8 [m, n] (0..3), sm f16 [m, nt, 2] -> (bytes [m/4, nt, 4, 32], scales [m/4, nt, 4, 2]).
    Byte i of a row-tile = q[i] | q[32+i] << 2 | q[64+i] << 4 | q[96+i] << 6 (int2_kern.inl: w0..w3 = v >> 0/2/4/6 & 3)."""
    m, n = q.shape
    assert m % 4 == 0 and n % G == 0, "i2a needs rows % 4 == 0"
    nt = n // G
    t = q.reshape(m // 4, 4, nt, 4, 32)                                    # [grp, row, tile, quarter, i]
    b = (t[:, :, :, 0] | (t[:, :, :, 1] << 2) | (t[:, :, :, 2] << 4) | (t[:, :, :, 3] << 6)).astype(np.uint8)
    b = np.ascontiguousarray(b.transpose(0, 2, 1, 3))                      # [grp, tile, row, 32]
    s = np.ascontiguousarray(sm.reshape(m // 4, 4, nt, 2).transpose(0, 2, 1, 3))   # [grp, tile, row, 2]
    return b, s


def unpack_i2a(b: np.ndarray, s: np.ndarray, m: int, n: int) -> np.ndarray:
    """Inverse of pack_i2a -> dequantised f32 [m, n] (reference reader)."""
    nt = n // G
    b = np.asarray(b, dtype=np.uint8).reshape(m // 4, nt, 4, 32)
    q = np.stack([(b >> (2 * k)) & 3 for k in range(4)], axis=3)           # [grp, tile, row, quarter, i]
    q = q.transpose(0, 2, 1, 3, 4).reshape(m, nt, G).astype(np.float32)
    sm = np.asarray(s, dtype=np.float32).reshape(m // 4, nt, 4, 2).transpose(0, 2, 1, 3).reshape(m, nt, 2)
    return (q * sm[:, :, :1] + sm[:, :, 1:]).reshape(m, n)


def enc_i2a(base: str, W: torch.Tensor, clips):
    q, sm, se, sn = quant_i2a(W, clips)
    m, n = q.shape
    b, s = pack_i2a(q, sm)
    return [(f"{base}.i2", b, lbt.U8, (m, n)), (f"{base}.i2s", s, lbt.F16, s.shape)], se, sn


ENCODERS = {"n4": enc_n4, "i2a": enc_i2a}


def select_fmt(args, layer: int, kind: str, mat: str, num_layers: int) -> str:
    """Per-tensor format selector. kind in {attn, expert, dense}; mat in {q,k,v,o,gate,up,down}."""
    if args.fmt != "mixed":
        return args.fmt
    if kind == "expert" and args.i2_lo <= layer < (args.i2_hi if args.i2_hi > 0 else num_layers + args.i2_hi) \
            and mat in args.i2_mats:
        return "i2a"
    return "n4"


# ----------------------------------------------------------------------------- HF checkpoint access
class HFSource:
    """Lazy tensor access over (sharded) safetensors via pread (no mmap: mapped pages would count as RSS and grow to the file size)."""

    _DT = {"BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4)}

    def __init__(self, d: str):
        self.dir = d
        idx = os.path.join(d, "model.safetensors.index.json")
        files = ([os.path.join(d, s) for s in sorted(set(json.load(open(idx))["weight_map"].values()))]
                 if os.path.exists(idx) else sorted(glob.glob(os.path.join(d, "*.safetensors"))))
        assert files, f"no safetensors in {d}"
        self.info = {}      # name -> (path, dtype, shape, abs_offset, nbytes)
        for fp in files:
            with open(fp, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                o0, o1 = v["data_offsets"]
                self.info[k] = (fp, v["dtype"], tuple(v["shape"]), 8 + n + o0, o1 - o0)
        self.map = self.info
        self.tl = threading.local()

    def _fd(self, path):
        fds = getattr(self.tl, "fds", None)
        if fds is None:
            fds = self.tl.fds = {}
        if path not in fds:
            fds[path] = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        return fds[path]

    def has(self, name):
        return name in self.info

    def get(self, name, rows=None) -> torch.Tensor:
        path, dt, shape, off, nbytes = self.info[name]
        tdt, esz = self._DT[dt]
        if rows is not None:
            per = nbytes // shape[0]
            off += rows[0] * per
            nbytes = (rows[1] - rows[0]) * per
            shape = (rows[1] - rows[0],) + shape[1:]
        buf = bytearray(nbytes)
        mv, got, fd = memoryview(buf), 0, self._fd(path)
        while got < nbytes:
            r = os.preadv(fd, [mv[got:]], off + got) if hasattr(os, "preadv") else None
            if r is None:  # Windows fallback
                os.lseek(fd, off + got, os.SEEK_SET); chunk = os.read(fd, nbytes - got); mv[got:got + len(chunk)] = chunk; r = len(chunk)
            assert r > 0, "short read"
            got += r
        return torch.frombuffer(buf, dtype=tdt).reshape(shape).float()

    def shape(self, name):
        return self.info[name][2]


# ----------------------------------------------------------------------------- jobs (run in worker threads)
def job_linear(src, hf_name, base, fmt, clips):
    torch.set_num_threads(1)
    W = src.get(hf_name)
    enc = ENCODERS[fmt]
    tensors, se, sn = enc(base, W, clips[fmt])
    return tensors, (fmt, se, sn)


def job_rows_i8(src, hf_name, base, r0, r1):
    w = src.get(hf_name, (r0, r1)).numpy()
    amax = np.maximum(np.abs(w).max(axis=1), 1e-8)
    scale = (amax / 127.0).astype(np.float16)
    q = np.clip(np.rint(w / scale.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
    return q, scale


def job_rows_f16(src, hf_name, r0, r1):
    return src.get(hf_name, (r0, r1)).numpy().astype(np.float16)


# ----------------------------------------------------------------------------- main export
def read_meta(model_dir, cfg, args):
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    tj_path = os.path.join(model_dir, "tokenizer.json")
    tc = json.load(open(tc_path, encoding="utf-8")) if os.path.exists(tc_path) else {}
    tj = json.load(open(tj_path, encoding="utf-8"))
    chat = tc.get("chat_template")
    if isinstance(chat, list):  # [{"name": "default", "template": ...}]
        chat = next((c["template"] for c in chat if c.get("name") == "default"), chat[0]["template"])
    if not chat and os.path.exists(os.path.join(model_dir, "chat_template.jinja")):
        chat = open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8").read()
    added = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    eos = []

    def add(x):
        for v in (x if isinstance(x, (list, tuple)) else [x]):
            if v is not None and int(v) >= 0 and int(v) not in eos:
                eos.append(int(v))
    gp = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gp):
        add(json.load(open(gp)).get("eos_token_id"))
    add(cfg.get("eos_token_id"))
    et = tc.get("eos_token")
    if isinstance(et, dict):
        et = et.get("content")
    if et in added:
        add(added[et])
    if "<|im_end|>" in added and not eos:
        add(added["<|im_end|>"])
    bos = cfg.get("bos_token_id")
    return tc_path, tj_path, chat or "", eos, -1 if bos is None else int(bos)


def export(args):
    t_start = time.time()
    torch.set_num_threads(1)           # parallelism = worker threads (each op is small); total threads = --threads
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    assert cfg.get("model_type") in ("qwen3_moe", "qwen3"), cfg.get("model_type")
    L, H = cfg["num_hidden_layers"], cfg["hidden_size"]
    heads, kvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or H // heads
    E = int(cfg.get("num_experts", 0))
    sparse = [E > 0 and i not in cfg.get("mlp_only_layers", []) and (i + 1) % cfg.get("decoder_sparse_step", 1) == 0
              for i in range(L)]
    tied = bool(cfg.get("tie_word_embeddings", False))
    src = HFSource(args.model)
    tc_path, tj_path, chat, eos, bos = read_meta(args.model, cfg, args)
    clips = {"n4": tuple(float(x) for x in args.clips_n4.split(",")),
             "i2a": tuple(float(x) for x in args.clips_i2.split(","))}
    args.i2_mats = set(args.i2_mats.split(","))
    lo, hi = args.i2_layers.split(":") if args.i2_layers else ("2", "-2")
    args.i2_lo, args.i2_hi = int(lo), int(hi)

    max_tensors = 16 + sum((4 + 8 + (1 + 6 * E if s else 6)) for s in sparse)
    out_tmp = args.out + ".part"
    w = LbtStreamWriter(out_tmp, max_tensors)
    nlayers_run = L if args.max_layers <= 0 else min(L, args.max_layers)
    log(f"export {args.model} -> {args.out}: L={L} H={H} E={E} fmt={args.fmt} threads={args.threads} "
        f"layers_run={nlayers_run} max_tensors={max_tensors}")

    pool = cf.ThreadPoolExecutor(args.threads)
    stats = {}            # (fmt, kind) -> [sq_err, sq_norm, n_matrices]
    bits = {"bits": 0.0, "w": 0.0}
    fmt_count = {}

    def layer_jobs(i):
        pre = f"model.layers.{i}."
        jobs = []   # (out_base, kind, future)
        mats = [("q", "attn", pre + "self_attn.q_proj.weight"), ("k", "attn", pre + "self_attn.k_proj.weight"),
                ("v", "attn", pre + "self_attn.v_proj.weight"), ("o", "attn", pre + "self_attn.o_proj.weight")]
        if sparse[i]:
            for e in range(E):
                for m_, hf in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                    mats.append((f"e{e}.{m_}", "expert", pre + f"mlp.experts.{e}.{hf}.weight"))
        else:
            for m_, hf in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                mats.append((m_, "dense", pre + f"mlp.{hf}.weight"))
        for short, kind, hf_name in mats:
            mat = short.split(".")[-1]
            fmt = select_fmt(args, i, kind, mat, L)
            jobs.append((f"blk.{i}.{short}", kind, pool.submit(job_linear, src, hf_name, f"blk.{i}.{short}", fmt, clips)))
        return jobs

    # --- head / embeddings first (cheap relative to layers; also validates the checkpoint early)
    def add_rows(name, hf_name, i8):
        vocab, dim = src.shape(hf_name)
        step = max(1, (8 << 20) // dim)
        spans = [(r, min(vocab, r + step)) for r in range(0, vocab, step)]
        if i8:
            futs = [pool.submit(job_rows_i8, src, hf_name, name, a, b) for a, b in spans]
            qs, ss = [], []
            for f in futs:
                q, s = f.result()
                qs.append(q); ss.append(s)
            w.add_tensor(name, np.concatenate(qs), lbt.I8, (vocab, dim))
            w.add_tensor(name + ".scale", np.concatenate(ss), lbt.F16, (vocab,))
        else:
            futs = [pool.submit(job_rows_f16, src, hf_name, a, b) for a, b in spans]
            w.add_tensor(name, np.concatenate([f.result() for f in futs]), lbt.F16, (vocab, dim))
        return vocab

    vocab = add_rows("tok_embeddings", "model.embed_tokens.weight", True)
    w.add_tensor("output_norm", src.get("model.norm.weight").numpy(), lbt.F32)
    if not tied:
        add_rows("output", "lm_head.weight", args.head == "i8")
    log("embeddings + head done")

    # --- layers (jobs of the next layer are queued while the current one is drained)
    ahead = {}
    for i in range(min(2, nlayers_run)):
        ahead[i] = layer_jobs(i)
    for i in range(nlayers_run):
        t0 = time.time()
        pre = f"model.layers.{i}."
        w.add_tensor(f"blk.{i}.attn_norm", src.get(pre + "input_layernorm.weight").numpy(), lbt.F32)
        w.add_tensor(f"blk.{i}.ffn_norm", src.get(pre + "post_attention_layernorm.weight").numpy(), lbt.F32)
        w.add_tensor(f"blk.{i}.q_norm", src.get(pre + "self_attn.q_norm.weight").numpy(), lbt.F32)
        w.add_tensor(f"blk.{i}.k_norm", src.get(pre + "self_attn.k_norm.weight").numpy(), lbt.F32)
        if sparse[i]:
            w.add_tensor(f"blk.{i}.moe_gate", src.get(pre + "mlp.gate.weight").numpy().astype(np.float16), lbt.F16)
        for base, kind, fut in ahead.pop(i):
            tensors, (fmt, se, sn) = fut.result()
            for name, arr, dt, shp in tensors:
                w.add_tensor(name, arr, dt, shp)
            s = stats.setdefault((fmt, kind), [0.0, 0.0, 0])
            s[0] += se; s[1] += sn; s[2] += 1
            m_, n_ = tensors[0][3]
            bits["bits"] += BPW[fmt] * m_ * n_; bits["w"] += m_ * n_
            fmt_count[fmt] = fmt_count.get(fmt, 0) + 1
        if i + 2 < nlayers_run:
            ahead[i + 2] = layer_jobs(i + 2)
        log(f"layer {i + 1}/{nlayers_run} done ({time.time() - t0:.1f}s, file {w.pos / 2**30:.2f} GB)")

    # --- tokenizer + meta
    w.add_bytes("tokenizer.json", open(tj_path, "rb").read())
    if os.path.exists(tc_path):
        w.add_bytes("tokenizer_config.json", open(tc_path, "rb").read())
    else:
        w.add_bytes("tokenizer_config.json", json.dumps({"chat_template": chat}).encode())
    eff = bits["bits"] / max(bits["w"], 1.0)
    rel = {f"{f}/{k}": round(v[0] / max(v[1], 1e-30), 5) for (f, k), v in stats.items()}
    w.set_meta(format="lbt", version=1, arch="qwen3", model_name=args.model_name or cfg.get("_name_or_path") or os.path.basename(args.model.rstrip("/\\")),
               hidden_size=H, num_layers=nlayers_run, num_heads=heads, num_kv_heads=kvh, head_dim=hd,
               intermediate_size=cfg.get("intermediate_size", 0), vocab_size=cfg["vocab_size"],
               rope_theta=float(cfg.get("rope_theta", 10000.0)), rms_norm_eps=float(cfg["rms_norm_eps"]),
               max_position_embeddings=cfg.get("max_position_embeddings", 4096), tie_word_embeddings=tied, qk_norm=True,
               attn_bias=bool(cfg.get("attention_bias", False)), rope_scaling=cfg.get("rope_scaling"),
               bos_token_id=bos, eos_token_ids=eos, embed_dtype="i8", chat_template=chat,
               draft_layer_cos=[], bpw_linear_total=eff,
               grid={"version": 1, "group": G, "mode": "int4" if args.fmt == "n4" else args.fmt, "nibble": True,
                     "formats": fmt_count, "bpw_linear_total": eff, "rel_mse": rel})
    if E:
        w.set_meta(num_experts=E, num_experts_per_tok=int(cfg["num_experts_per_tok"]),
                   moe_intermediate_size=int(cfg["moe_intermediate_size"]),
                   norm_topk_prob=bool(cfg.get("norm_topk_prob", True)))
    w.set_draft_layer_mask([int(x) for x in args.draft_skip.split(",") if x != ""])
    info = w.close()
    pool.shutdown()
    os.replace(out_tmp, args.out)
    dt = time.time() - t_start
    log(f"DONE {args.out}: {info['bytes'] / 2**30:.2f} GB, {info['tensors']} tensors, {dt / 60:.1f} min, "
        f"bpw_linear {eff:.3f}, rel_mse {rel}")
    return {"bytes": info["bytes"], "tensors": info["tensors"], "seconds": dt, "rel_mse": rel, "bpw": eff}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local HF dir (config.json, *.safetensors, tokenizer*)")
    ap.add_argument("--out", required=True, help="output model file (.vlt; a legacy .lbt name works too; env VELTRA_LEGACY_MAGIC=1 writes the old magic)")
    ap.add_argument("--fmt", default="n4", choices=["n4", "i2a", "mixed"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--head", default="i8", choices=["i8", "f16"], help="lm_head storage (untied models)")
    ap.add_argument("--clips-n4", default=",".join(map(str, DEFAULT_CLIPS_N4)))
    ap.add_argument("--clips-i2", default=",".join(map(str, DEFAULT_CLIPS_I2)))
    ap.add_argument("--i2-layers", default="", help="mixed: A:B layer range for i2a experts (B<0 counts from the end; default 2:-2)")
    ap.add_argument("--i2-mats", default="gate,up", help="mixed: which expert matrices use i2a")
    ap.add_argument("--draft-skip", default="", help="comma list of layers skipped in draft mode (default none -> engine fallback)")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--max-layers", type=int, default=0, help="debug: export only the first N layers")
    args = ap.parse_args(argv)
    export(args)


if __name__ == "__main__":
    main()
