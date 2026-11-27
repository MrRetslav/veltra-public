#!/usr/bin/env python3
"""Streaming exporter: Qwen3.6-35B-A3B (HF bf16 + unsloth GGUF) -> Veltra `.vlt` per docs/qwen36_lbt_contract.md.

  python export_qwen36_stream.py --model <dir>/hf36 --gguf <dir>/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf \
         --out <dir>/q36_35b_qk.vlt [--threads 8] [--head i8|f16] [--no-mtp] [--colnorm] [--verify 200]

Sources
  routed experts (256 x gate/up/down x 40 layers): BIT-EXACT from the GGUF (`blk.i.ffn_{gate,up,down}_exps`), per-tensor type kept
      (K-quants -> `.q2k/.q3k/.q4k`, i-quants IQ2_XS / IQ3_XXS / IQ4_XS -> `.iq2xs/.iq3xxs/.iq4xs`, gguf_qk2.py).
      Fused HF `gate_up_proj` = rows 0..511 gate, 512..1023 up (checked against the GGUF at start, `--no-check` skips).
  everything else (attention, DeltaNet projections, shared expert, MTP block): HF bf16 -> n4 g128 RTN + per-group MSE clip search.
  small tensors: router + shared_expert_gate + DeltaNet a/b projections F16; norms (1 + w, contract 0.1), conv1d, A_log, dt_bias, dn.norm F32.
  embeddings / head: I8 per-row (`--head f16`: plain F16 head).  Vision tower is dropped.  MTP block: `--no-mtp` skips it.
Memory: all reads are pread() (HF shards + GGUF), never mmap'd pages; ~2 layers of output in flight; peak RSS is logged at the end.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import resource
import sys
import time

import numpy as np

try:    # glibc: big buffers via mmap (returned to the OS on free) and few arenas -> peak RSS follows the live working set, not fragmentation
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    _libc.mallopt(-8, 2)            # M_ARENA_MAX
    _libc.mallopt(-3, 1 << 17)      # M_MMAP_THRESHOLD (fixed => no dynamic growth)
    _libc.mallopt(-1, 1 << 20)      # M_TRIM_THRESHOLD
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                   # noqa: E402  (export_qwen3_moe_stream pins the BLAS thread env vars first)
import lbt_format as lbt                       # noqa: E402
import nibble_lbt as nib                       # noqa: E402
import gguf_qk2 as qk                          # noqa: E402
from lbt_stream import LbtStreamWriter         # noqa: E402
from export_qwen3_moe_stream import (HFSource, enc_n4, job_linear, job_rows_f16, job_rows_i8, BPW,  # noqa: E402
                                     DEFAULT_CLIPS_N4, G)

LM = "model.language_model."
PROJ = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- writer with chunked tensors
class Stream2(LbtStreamWriter):
    def add_chunked(self, name, dtype, shape, chunks):
        """Append one tensor whose bytes arrive as an iterable of arrays (peak RAM = a few chunks, not the tensor)."""
        assert name not in self.names and len(self.entries) < self.max_tensors
        assert self.f.tell() == self.pos
        off, total = self.pos, 0
        for c in chunks:
            raw = np.ascontiguousarray(c, dtype=lbt._NP[dtype]).reshape(-1).view(np.uint8)
            self.f.write(memoryview(raw))
            total += raw.nbytes
        end = lbt._align(off + total)
        if end > off + total:
            self.f.write(b"\0" * (end - off - total))
        self.pos = end
        self.names.add(name)
        self.entries.append((name, dtype, tuple(shape), off, int(total)))


class HFSourceP(HFSource):
    """HFSource that tolerates missing shards (verification / order check while the download is still running)."""

    def __init__(self, d, allow_partial=False):
        import glob, threading
        if not allow_partial:
            return super().__init__(d)
        self.dir, self.info, self.tl = d, {}, threading.local()
        for fp in sorted(glob.glob(os.path.join(d, "model-*.safetensors"))):
            with open(fp, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    o0, o1 = v["data_offsets"]
                    self.info[k] = (fp, v["dtype"], tuple(v["shape"]), 8 + n + o0, o1 - o0)
        self.map = self.info


# ----------------------------------------------------------------------------- GGUF access (pread, not mmap)
class GgufExperts:
    def __init__(self, path, gguf_py=""):
        if gguf_py:
            sys.path.insert(0, gguf_py)
        import gguf
        self.gguf = gguf
        r = gguf.GGUFReader(path)
        self.T = {t.name: t for t in r.tensors}
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self.meta = {}
        for k in ("general.architecture", "general.name", "general.file_type"):
            f = r.fields.get(k)
            if f is not None:
                try:
                    self.meta[k] = f.contents()
                except Exception:
                    pass
        self._r = r     # keeps the (lazy) mmap alive; pages are never touched

    def info(self, layer, proj):
        t = self.T[f"blk.{layer}.{PROJ[proj]}.weight"]
        qt = t.tensor_type.name
        assert qt in qk.SUFFIX, (t.name, qt)
        cols, rows, E = (int(x) for x in t.shape)
        bs, ts = self.gguf.GGML_QUANT_SIZES[t.tensor_type]
        assert bs == qk.QK_K and ts == qk.GGML_BYTES[qt], (qt, bs, ts)
        bpr = cols // bs * ts
        assert int(t.n_bytes) == E * rows * bpr, (t.name, int(t.n_bytes), E * rows * bpr)
        return t, qt, rows, cols, bpr, E

    def raw(self, layer, proj, e):
        t, qt, rows, cols, bpr, E = self.info(layer, proj)
        n, off = rows * bpr, int(t.data_offset) + e * rows * bpr
        buf = bytearray(n)
        mv, got = memoryview(buf), 0
        while got < n:
            r = os.preadv(self.fd, [mv[got:]], off + got) if hasattr(os, "preadv") else None
            if r is None:
                os.lseek(self.fd, off + got, os.SEEK_SET)
                c = os.read(self.fd, n - got)
                mv[got:got + len(c)] = c
                r = len(c)
            assert r > 0, "short read"
            got += r
        return np.frombuffer(buf, np.uint8).reshape(rows, bpr), qt, rows, cols

    def dequant(self, layer, proj, e):
        a, qt, rows, cols = self.raw(layer, proj, e)
        return self.gguf.quants.dequantize(a, getattr(self.gguf.GGMLQuantizationType, qt))


def job_expert(gg, layer, e, colnorm):
    out, cn = [], None
    for proj in PROJ:
        a, qt, rows, cols = gg.raw(layer, proj, e)
        buf = qk.pack_any(a, qt)
        out.append((f"blk.{layer}.e{e}.{proj}.{qk.SUFFIX[qt]}", buf, lbt.U8, (rows, cols // qk.QK_K * qk.QK_BYTES[qt])))
        if colnorm and proj == "down":
            w = gg.gguf.quants.dequantize(a, getattr(gg.gguf.GGMLQuantizationType, qt))     # [2048, 512]
            cn = np.sqrt((w.astype(np.float64) ** 2).sum(0)).astype(np.float16)
            out.append((f"blk.{layer}.e{e}.down.colnorm", cn, lbt.F16, cn.shape))
    return out


def job_mtp_expert(src, pre, e, base, clips):
    gu = src.get(pre + "mlp.experts.gate_up_proj", (e, e + 1))[0]         # [1024, 2048]: gate rows 0..511, up rows 512..1023
    dn = src.get(pre + "mlp.experts.down_proj", (e, e + 1))[0]            # [2048, 512]
    tensors, se, sn = [], 0.0, 0.0
    for nm, W in (("gate", gu[:512]), ("up", gu[512:]), ("down", dn)):
        t, a, b = enc_n4(f"{base}.e{e}.{nm}", W.contiguous(), clips["n4"])
        tensors += t
        se += a; sn += b
    return tensors, ("n4", se, sn)


def check_order(src, gg, layer=0, e=0):
    """GGUF gate/up dequant vs the two halves of HF gate_up_proj: gate must be the FIRST half."""
    pre = f"{LM}layers.{layer}.mlp.experts."
    gu = src.get(pre + "gate_up_proj", (e, e + 1))[0].numpy()
    cc = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    g, u = gg.dequant(layer, "gate", e), gg.dequant(layer, "up", e)
    r = dict(g_first=cc(g, gu[:512]), g_second=cc(g, gu[512:]), u_first=cc(u, gu[:512]), u_second=cc(u, gu[512:]))
    ok = r["g_first"] > 0.9 and r["u_second"] > 0.9 and abs(r["g_second"]) < 0.5 and abs(r["u_first"]) < 0.5
    log("gate/up order check (layer %d expert %d): %s -> %s" % (layer, e, {k: round(v, 4) for k, v in r.items()},
                                                               "gate = FIRST half, up = SECOND half" if ok else "UNEXPECTED"))
    if not ok:
        raise SystemExit("fused gate_up order is not gate|up; refusing to export")
    return r


# ----------------------------------------------------------------------------- small helpers
def f32(src, name):
    return src.get(name).numpy().astype(np.float32)


def one_plus(src, name):
    return (1.0 + src.get(name).numpy().astype(np.float32)).astype(np.float32)


def read_chat_meta(model_dir, tc_cfg):
    tj_path = os.path.join(model_dir, "tokenizer.json")
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    tc = json.load(open(tc_path, encoding="utf-8")) if os.path.exists(tc_path) else {}
    chat = tc.get("chat_template")
    if isinstance(chat, list):
        chat = next((c["template"] for c in chat if c.get("name") == "default"), chat[0]["template"])
    jp = os.path.join(model_dir, "chat_template.jinja")
    if not chat and os.path.exists(jp):
        chat = open(jp, encoding="utf-8").read()
    tc["chat_template"] = chat or ""
    tj = json.load(open(tj_path, encoding="utf-8"))
    added = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    eos = []
    gp = os.path.join(model_dir, "generation_config.json")
    ids = json.load(open(gp)).get("eos_token_id") if os.path.exists(gp) else tc_cfg.get("eos_token_id")
    for v in (ids if isinstance(ids, list) else [ids]):
        if v is not None and int(v) not in eos:
            eos.append(int(v))
    if "<|im_end|>" in added and added["<|im_end|>"] not in eos:
        eos.insert(0, added["<|im_end|>"])
    return tj_path, json.dumps(tc, ensure_ascii=False).encode("utf-8"), tc["chat_template"], eos


def thinking_meta(chat):
    """Qwen3.6: the generation prompt is `<think>\\n` (thinking ON by default); `enable_thinking=false` renders `<think>\\n\\n</think>\\n\\n`."""
    return {"default": "on" if ("enable_thinking is defined and enable_thinking is false" in chat) else "unknown",
            "template_variable": "enable_thinking",
            "generation_suffix_on": "<think>\n", "generation_suffix_off": "<think>\n\n</think>\n\n",
            "eval_harness_mode": "off"}


# ----------------------------------------------------------------------------- export
def export(args):
    t_start = time.time()
    torch.set_num_threads(1)
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    tc = cfg.get("text_config", cfg)
    assert tc.get("model_type") == "qwen3_5_moe_text", tc.get("model_type")
    L, H, E = tc["num_hidden_layers"], tc["hidden_size"], tc["num_experts"]
    lt = tc["layer_types"]
    heads, kvh, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    rp = tc.get("rope_parameters", {})
    theta = float(rp.get("rope_theta", tc.get("rope_theta", 1e7)))
    prf = float(rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
    nL = L if args.max_layers <= 0 else min(L, args.max_layers)
    src = HFSourceP(args.model, allow_partial=args.partial)
    gg = GgufExperts(args.gguf, args.gguf_py)
    with_mtp = (not args.no_mtp) and src.has("mtp.fc.weight") and nL == L
    tj_path, tc_blob, chat, eos = read_chat_meta(args.model, tc)
    clips = {"n4": tuple(float(x) for x in args.clips_n4.split(","))}
    log(f"export {args.model} + {os.path.basename(args.gguf)} -> {args.out}: L={nL}/{L} E={E} threads={args.threads} "
        f"mtp={with_mtp} colnorm={args.colnorm} head={args.head}")
    if not args.no_check:
        check_order(src, gg)

    per_layer = 2 + 1 + E * (3 + (1 if args.colnorm else 0)) + 6 + 1 + 20
    max_tensors = 16 + nL * per_layer + (E * 6 + 40 if with_mtp else 0)
    out_tmp = args.out + ".part"
    w = Stream2(out_tmp, max_tensors)
    pool = cf.ThreadPoolExecutor(args.threads)
    stats, fmt_count, counts, types = {}, {}, {}, {}
    bits = {"bits": 0.0, "w": 0.0}

    def n4_job(hf_name, base):
        return pool.submit(job_linear, src, hf_name, base, "n4", clips)

    def layer_plan(i, pre, out_pre, full, hf_pre=None):
        """-> list of (kind, future-or-None, immediate tensors) in file order."""
        P = hf_pre or f"{LM}layers.{i}."
        o = out_pre
        items = []
        small = [(f"{o}attn_norm", one_plus(src, P + "input_layernorm.weight"), lbt.F32),
                 (f"{o}ffn_norm", one_plus(src, P + "post_attention_layernorm.weight"), lbt.F32)]
        if full:
            small += [(f"{o}q_norm", one_plus(src, P + "self_attn.q_norm.weight"), lbt.F32),
                      (f"{o}k_norm", one_plus(src, P + "self_attn.k_norm.weight"), lbt.F32)]
        else:
            la = P + "linear_attn."
            small += [(f"{o}dn.a", src.get(la + "in_proj_a.weight").numpy().astype(np.float16), lbt.F16),
                      (f"{o}dn.b", src.get(la + "in_proj_b.weight").numpy().astype(np.float16), lbt.F16),
                      (f"{o}dn.conv", src.get(la + "conv1d.weight").numpy().astype(np.float32).reshape(-1, tc["linear_conv_kernel_dim"]), lbt.F32),
                      (f"{o}dn.A_log", f32(src, la + "A_log"), lbt.F32),
                      (f"{o}dn.dt_bias", f32(src, la + "dt_bias"), lbt.F32),
                      (f"{o}dn.norm", f32(src, la + "norm.weight"), lbt.F32)]
        small += [(f"{o}moe_gate", src.get(P + "mlp.gate.weight").numpy().astype(np.float16), lbt.F16),
                  (f"{o}sh_gate", src.get(P + "mlp.shared_expert_gate.weight").numpy().astype(np.float16), lbt.F16)]
        items.append(("small", small))
        lin = []
        if full:
            for s, h in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
                lin.append(("attn", f"{o}{s}", P + f"self_attn.{h}.weight"))
        else:
            for s, h in (("dn.qkv", "in_proj_qkv"), ("dn.z", "in_proj_z"), ("dn.out", "out_proj")):
                lin.append(("dn", f"{o}{s}", P + f"linear_attn.{h}.weight"))
        for s, h in (("sh.gate", "gate_proj"), ("sh.up", "up_proj"), ("sh.down", "down_proj")):
            lin.append(("shared", f"{o}{s}", P + f"mlp.shared_expert.{h}.weight"))
        items.append(("lin", [(kind, base, n4_job(hf, base)) for kind, base, hf in lin]))
        return items

    exq, ex_pos = collections.deque(), [0, 0]          # bounded window of in-flight expert jobs (~1 MB of packed bytes each)

    def ex_fill():
        while len(exq) < args.expert_window and ex_pos[0] < nL:
            exq.append(pool.submit(job_expert, gg, ex_pos[0], ex_pos[1], args.colnorm))
            ex_pos[1] += 1
            if ex_pos[1] == E:
                ex_pos[0], ex_pos[1] = ex_pos[0] + 1, 0

    def drain_layer(items, ex_futs, tag):
        for kind, payload in items:
            if kind == "small":
                for name, arr, dt in payload:
                    w.add_tensor(name, arr, dt)
            else:
                for k_, base, fut in payload:
                    tensors, (fmt, se, sn) = fut.result()
                    for name, arr, dt, shp in tensors:
                        w.add_tensor(name, arr, dt, shp)
                    s = stats.setdefault(f"{fmt}/{k_}", [0.0, 0.0, 0])
                    s[0] += se; s[1] += sn; s[2] += 1
                    m_, n_ = tensors[0][3]
                    bits["bits"] += BPW[fmt] * m_ * n_; bits["w"] += m_ * n_
                    fmt_count[fmt] = fmt_count.get(fmt, 0) + 1
        for e in range(E if ex_futs else 0):
            fut = exq.popleft()                          # global FIFO of expert jobs (layer-major), refilled to a bounded window
            ex_fill()
            for name, arr, dt, shp in fut.result():
                w.add_tensor(name, arr, dt, shp)
                sfx = name.rsplit(".", 1)[1]
                typ = next((t for t, s_ in qk.SUFFIX.items() if s_ == sfx), None)
                if typ:
                    counts[sfx] = counts.get(sfx, 0) + 1
                    bits["bits"] += shp[0] * shp[1] * 8.0
                    bits["w"] += shp[0] * (shp[1] // qk.QK_BYTES[typ]) * qk.QK_K

    # --- top level (embeddings + head): chunked, bounded in flight
    def add_rows(name, hf_name, i8):
        vocab, dim = src.shape(hf_name)
        step = max(1, (2 << 20) // dim)
        spans = [(r, min(vocab, r + step)) for r in range(0, vocab, step)]
        scales = []
        futs = {}

        def submit(k):
            a, b = spans[k]
            futs[k] = pool.submit(job_rows_i8, src, hf_name, name, a, b) if i8 else pool.submit(job_rows_f16, src, hf_name, a, b)

        win = args.threads + 2
        for k in range(min(win, len(spans))):
            submit(k)

        def gen():
            for k in range(len(spans)):
                res = futs.pop(k).result()
                if k + win < len(spans):
                    submit(k + win)
                if i8:
                    scales.append(res[1])
                    yield res[0]
                else:
                    yield res
        w.add_chunked(name, lbt.I8 if i8 else lbt.F16, (vocab, dim), gen())
        if i8:
            w.add_tensor(name + ".scale", np.concatenate(scales), lbt.F16, (vocab,))

    add_rows("tok_embeddings", LM + "embed_tokens.weight", True)
    w.add_tensor("output_norm", one_plus(src, LM + "norm.weight"), lbt.F32)
    add_rows("output", "lm_head.weight", args.head == "i8")
    log(f"embeddings + head done (peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB)")

    # --- layers: plan(i+1) is queued while layer i is drained
    ahead = {}

    def queue(i):
        ahead[i] = layer_plan(i, None, f"blk.{i}.", lt[i] == "full_attention")

    for i in range(min(2, nL)):
        queue(i)
    ex_fill()
    log(f"start rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB")
    for i in range(nL):
        t0 = time.time()
        items, exf = ahead.pop(i), True
        for proj in PROJ:
            _, qt, rows, cols, bpr, _E = gg.info(i, proj)
            types[f"{i}.{proj}"] = qt
        drain_layer(items, exf, i)
        if i + 2 < nL:
            queue(i + 2)           # n4 jobs of the layer after next (experts: bounded global window, ex_fill)
        log(f"layer {i + 1}/{nL} done ({time.time() - t0:.1f}s, file {w.pos / 2**30:.2f} GiB, "
            f"{'attn' if lt[i] == 'full_attention' else 'deltanet'}, {types[f'{i}.gate']}/{types[f'{i}.down']}, "
            f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB)")

    # --- MTP block (optional group `mtp.*`), experts from HF bf16 as n4
    if with_mtp:
        t0 = time.time()
        P = "mtp.layers.0."
        o = "mtp.blk.0."
        items = layer_plan(0, None, o, True, hf_pre=P)
        mtp_ex = {}
        win = 2 * args.threads                          # bounded in flight: each job holds ~3 f32 expert matrices
        for e in range(min(win, E)):
            mtp_ex[e] = pool.submit(job_mtp_expert, src, P, e, o[:-1], clips)
        mtp_fc = n4_job("mtp.fc.weight", "mtp.fc")
        for nm, hf in (("pre_fc_norm_embedding", "mtp.pre_fc_norm_embedding.weight"), ("pre_fc_norm_hidden", "mtp.pre_fc_norm_hidden.weight"),
                       ("norm", "mtp.norm.weight")):
            w.add_tensor(f"mtp.{nm}", one_plus(src, hf), lbt.F32)
        drain_layer(items, None, "mtp")
        tensors, (fmt, se, sn) = mtp_fc.result()
        for name, arr, dt, shp in tensors:
            w.add_tensor(name, arr, dt, shp)
        s = stats.setdefault("n4/mtp_fc", [0.0, 0.0, 0]); s[0] += se; s[1] += sn; s[2] += 1
        for e in range(E):
            tensors, (fmt, se, sn) = mtp_ex.pop(e).result()
            if e + win < E:
                mtp_ex[e + win] = pool.submit(job_mtp_expert, src, P, e + win, o[:-1], clips)
            for name, arr, dt, shp in tensors:
                w.add_tensor(name, arr, dt, shp)
            s = stats.setdefault("n4/mtp_expert", [0.0, 0.0, 0]); s[0] += se; s[1] += sn; s[2] += 3
        log(f"mtp block done ({time.time() - t0:.1f}s, file {w.pos / 2**30:.2f} GiB)")

    # --- tokenizer + meta
    w.add_bytes("tokenizer.json", open(tj_path, "rb").read())
    w.add_bytes("tokenizer_config.json", tc_blob)
    eff = bits["bits"] / max(bits["w"], 1.0)
    rel = {k: round(v[0] / max(v[1], 1e-30), 5) for k, v in stats.items() if v[1] > 0}
    meta = dict(format="lbt", version=1, arch="qwen3", model_name=args.model_name or "Qwen/Qwen3.6-35B-A3B",
                hidden_size=H, num_layers=nL, num_heads=heads, num_kv_heads=kvh, head_dim=hd, intermediate_size=0,
                vocab_size=tc["vocab_size"], rope_theta=theta, rms_norm_eps=float(tc["rms_norm_eps"]),
                max_position_embeddings=tc["max_position_embeddings"], tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
                qk_norm=True, attn_bias=bool(tc.get("attention_bias", False)), rope_scaling=None,
                bos_token_id=int(tc.get("bos_token_id", -1) if tc.get("bos_token_id") is not None else -1),
                eos_token_ids=eos, embed_dtype="i8", chat_template=chat, draft_layer_cos=[], bpw_linear_total=eff,
                num_experts=E, num_experts_per_tok=int(tc["num_experts_per_tok"]), moe_intermediate_size=int(tc["moe_intermediate_size"]),
                norm_topk_prob=True,
                layer_types=lt[:nL], rotary_dim=int(round(hd * prf)), partial_rotary_factor=prf, attn_output_gate=bool(tc.get("attn_output_gate", True)),
                mrope_section=rp.get("mrope_section"), mrope_interleaved=bool(rp.get("mrope_interleaved", False)),
                linear_num_key_heads=tc["linear_num_key_heads"], linear_num_value_heads=tc["linear_num_value_heads"],
                linear_key_head_dim=tc["linear_key_head_dim"], linear_value_head_dim=tc["linear_value_head_dim"],
                linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
                shared_expert_intermediate_size=int(tc.get("shared_expert_intermediate_size", 0)), shared_expert_gate=True,
                norm_one_plus_baked=True, thinking=thinking_meta(chat),
                grid={"version": 1, "group": G, "mode": "int4", "nibble": True, "formats": fmt_count,
                      "bpw_linear_total": eff, "rel_mse": rel},
                qk={"version": 2, "source": os.path.basename(args.gguf), "tensor_types": types, "counts": counts,
                    "suffix": qk.SUFFIX, "sb_bytes": qk.QK_BYTES, "expert_order": "gate=first half of HF gate_up_proj",
                    "colnorm": bool(args.colnorm)})
    if with_mtp:
        meta["mtp"] = {"layers": 1, "experts": "n4 (HF bf16; the GGUF has no MTP tensors)"}
    w.set_meta(**meta)
    w.set_draft_layer_mask([])
    info = w.close()
    pool.shutdown()
    os.replace(out_tmp, args.out)
    dt = time.time() - t_start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    log(f"DONE {args.out}: {info['bytes'] / 2**30:.2f} GiB, {info['tensors']} tensors, {dt / 60:.1f} min, peak RSS {rss:.2f} GiB, "
        f"counts {counts}, rel_mse {rel}")
    return {"bytes": info["bytes"], "tensors": info["tensors"], "seconds": dt, "rss_gib": rss}


# ----------------------------------------------------------------------------- verification
def verify(args):
    r = lbt.LbtReader(args.out)
    m = r.meta
    cfg = json.load(open(os.path.join(args.model, "config.json")))["text_config"]
    src = HFSourceP(args.model, allow_partial=args.partial)
    gg = GgufExperts(args.gguf, args.gguf_py)
    L, E = m["num_layers"], m["num_experts"]
    lt = m["layer_types"]
    rnd = random.Random(args.seed)
    bad = 0

    def rel(a, b):
        return float(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-30))

    # 1. experts: bit-exact vs GGUF
    samples = [(i, rnd.randrange(E), p) for i in range(L) for p in PROJ]
    samples += [(rnd.randrange(L), rnd.randrange(E), rnd.choice(list(PROJ))) for _ in range(max(0, args.verify - len(samples)))]
    for (i, e, proj) in samples:
        a, qt, rows, cols = gg.raw(i, proj, e)
        name = f"blk.{i}.e{e}.{proj}.{qk.SUFFIX[qt]}"
        ref = gg.gguf.quants.dequantize(a, getattr(gg.gguf.GGMLQuantizationType, qt))
        got = qk.unpack_any(r.raw(name), qt, rows, cols, gg.gguf)
        raw_ok = True
        if qt in qk.RAW_TYPES:
            raw_ok = np.array_equal(qk.unpack_to_ggml(r.raw(name), qt, rows, cols), a)
        if not (ref.shape == got.shape and np.array_equal(ref, got) and raw_ok):
            bad += 1
            print("MISMATCH expert", name, qt)
        if m["qk"].get("colnorm") and proj == "down":
            cn = r.raw(f"blk.{i}.e{e}.down.colnorm").view(np.float16).astype(np.float32)
            exp = np.sqrt((ref.astype(np.float64) ** 2).sum(0)).astype(np.float16).astype(np.float32)
            if not np.array_equal(cn, exp):
                bad += 1
                print("MISMATCH colnorm", name)
    log(f"verify experts: {len(samples)} tensors vs gguf-py dequantize (array_equal), mismatches so far {bad}")

    # 2. n4 linears vs HF (RTN error), 3. small tensors exact
    def n4_check(base, hf_name, rows=None, tol=0.15, full_n4=True):
        nonlocal bad
        e0 = r.entries[base + ".n4"]
        mm, nn = e0.shape
        W = src.get(hf_name) if rows is None else src.get(*hf_name)[rows]
        W = W.numpy()
        What = nib.n4_dequant(r.raw(base + ".n4"), r.raw(base + ".gs").view(np.float16).reshape(mm, nn // G), mm, nn)
        x = rel(What, W)
        if not x < tol or What.shape != W.shape:
            bad += 1
            print("BAD n4", base, x, What.shape, W.shape)
        if full_n4:                         # the quantiser is deterministic: re-encoding the HF tensor must give the stored bytes
            t_, _, _ = enc_n4(base, torch.from_numpy(np.ascontiguousarray(W)), DEFAULT_CLIPS_N4)
            if not (np.array_equal(t_[0][1], r.raw(base + ".n4")) and np.array_equal(t_[1][1].view(np.uint8).reshape(-1), r.raw(base + ".gs"))):
                bad += 1
                print("BAD n4 not reproducible", base)
        return x

    errs = {}
    layers = sorted(set([0, 3, L - 1] + [rnd.randrange(L) for _ in range(max(0, args.verify // 40))]))
    for i in layers:
        P, o = f"{LM}layers.{i}.", f"blk.{i}."
        full = lt[i] == "full_attention"
        if full:
            specs = [(s, P + f"self_attn.{h}.weight") for s, h in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj"))]
        else:
            specs = [(s, P + f"linear_attn.{h}.weight") for s, h in (("dn.qkv", "in_proj_qkv"), ("dn.z", "in_proj_z"), ("dn.out", "out_proj"))]
        specs += [(f"sh.{s}", P + f"mlp.shared_expert.{h}.weight") for s, h in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj"))]
        for s, hf in specs:
            if not src.has(hf):
                continue
            errs.setdefault(s.split(".")[0], []).append(n4_check(o + s, hf))
        # small tensors
        chk = [("attn_norm", 1 + src.get(P + "input_layernorm.weight").numpy()), ("ffn_norm", 1 + src.get(P + "post_attention_layernorm.weight").numpy()),
               ("moe_gate", src.get(P + "mlp.gate.weight").numpy().astype(np.float16)),
               ("sh_gate", src.get(P + "mlp.shared_expert_gate.weight").numpy().astype(np.float16))]
        if full:
            chk += [("q_norm", 1 + src.get(P + "self_attn.q_norm.weight").numpy()), ("k_norm", 1 + src.get(P + "self_attn.k_norm.weight").numpy())]
        else:
            la = P + "linear_attn."
            chk += [("dn.a", src.get(la + "in_proj_a.weight").numpy().astype(np.float16)), ("dn.b", src.get(la + "in_proj_b.weight").numpy().astype(np.float16)),
                    ("dn.conv", src.get(la + "conv1d.weight").numpy().reshape(-1, 4)), ("dn.A_log", src.get(la + "A_log").numpy()),
                    ("dn.dt_bias", src.get(la + "dt_bias").numpy()), ("dn.norm", src.get(la + "norm.weight").numpy())]
        for nm, ref in chk:
            got = r.tensor(o + nm)
            if got.shape != ref.shape or not np.array_equal(got, ref.astype(got.dtype)):
                bad += 1
                print("MISMATCH small", o + nm, got.shape, ref.shape)
    log("verify n4 rel_mse by class: " + ", ".join(f"{k} mean {np.mean(v):.4f} max {np.max(v):.4f} (n={len(v)})" for k, v in errs.items()))

    # 4. embeddings / head / output_norm
    for nm, hf in (("tok_embeddings", LM + "embed_tokens.weight"), ("output", "lm_head.weight")):
        vocab, dim = src.shape(hf)
        for row in (0, 1000, vocab // 2, vocab - 1):
            ref = src.get(hf, (row, row + 1)).numpy()[0]
            e0 = r.entries[nm]
            if e0.dtype == lbt.I8:
                got = r.raw(nm).view(np.int8).reshape(vocab, dim)[row].astype(np.float32) * float(r.raw(nm + ".scale").view(np.float16)[row])
            else:
                got = r.raw(nm).view(np.float16).reshape(vocab, dim)[row].astype(np.float32)
            if float(np.abs(got - ref).max()) > float(np.abs(ref).max()) / 254.0 * 1.02 + 1e-7:      # i8 per-row: error <= half a step
                bad += 1
                print("BAD row", nm, row, float(np.abs(got - ref).max()), float(np.abs(ref).max()))
    if not np.array_equal(r.tensor("output_norm"), 1 + src.get(LM + "norm.weight").numpy()):
        bad += 1
        print("MISMATCH output_norm")

    # 5. MTP
    if "mtp" in m:
        P, o = "mtp.layers.0.", "mtp.blk.0."
        for e in rnd.sample(range(E), 4):
            gu = src.get(P + "mlp.experts.gate_up_proj", (e, e + 1))[0].numpy()
            dn = src.get(P + "mlp.experts.down_proj", (e, e + 1))[0].numpy()
            for nm, W in (("gate", gu[:512]), ("up", gu[512:]), ("down", dn)):
                b = f"{o}e{e}.{nm}"
                mm, nn = r.entries[b + ".n4"].shape
                What = nib.n4_dequant(r.raw(b + ".n4"), r.raw(b + ".gs").view(np.float16).reshape(mm, nn // G), mm, nn)
                x = rel(What, W)
                if not x < 0.15:
                    bad += 1
                    print("BAD mtp expert", b, x)
        n4_check("mtp.fc", "mtp.fc.weight")
        for s, h in (("q", "q_proj"), ("o", "o_proj")):
            n4_check(o + s, P + f"self_attn.{h}.weight")
    log(f"verify DONE: mismatches/bad {bad}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF dir: config.json, model-*.safetensors, tokenizer*, chat_template.jinja")
    ap.add_argument("--gguf", required=True, help="unsloth Qwen3.6-35B-A3B GGUF (UD-Q2_K_XL)")
    ap.add_argument("--out", required=True, help="output model file (.vlt; a legacy .lbt name works too; env VELTRA_LEGACY_MAGIC=1 writes the old magic)")
    ap.add_argument("--gguf-py", default="<dir>/gguf-py", help="llama.cpp/gguf-py if `gguf` is not installed")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--head", default="i8", choices=["i8", "f16"])
    ap.add_argument("--no-mtp", action="store_true", help="skip the MTP block (0.85 B params, ~0.5 GB at n4)")
    ap.add_argument("--colnorm", action="store_true", help="also write blk.i.e{e}.down.colnorm (F16 [512], L2 norm per input column of the dequantised down_proj)")
    ap.add_argument("--clips-n4", default=",".join(map(str, DEFAULT_CLIPS_N4)))
    ap.add_argument("--model-name", default="")
    ap.add_argument("--max-layers", type=int, default=0, help="debug: only the first N layers (no MTP)")
    ap.add_argument("--no-check", action="store_true", help="skip the gate/up order check against the GGUF")
    ap.add_argument("--verify", type=int, default=0, help="after export: N expert samples (bit-exact vs GGUF) + n4/small/embedding checks vs HF")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--partial", action="store_true", help="verify: tolerate missing HF shards")
    ap.add_argument("--expert-window", type=int, default=48, help="expert transcode jobs in flight")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    args.threads = min(args.threads, 8)
    if not args.verify_only:
        export(args)
    if args.verify or args.verify_only:
        args.verify = args.verify or 200
        sys.exit(1 if verify(args) else 0)


if __name__ == "__main__":
    main()
