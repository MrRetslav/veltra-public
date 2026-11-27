#!/usr/bin/env python3
"""Replace the attention linears' nibble tensors (`blk.i.{q,k,v,o}.n4` + `.gs`) of an `.lbt` by INT3 g128 asymmetric `.i3` tensors (engine/i3_gemv.h,
docs/i3_format.md). Every other tensor is copied bit-exact (same order, 64-byte aligned); meta gains an `i3` entry.

  python export_i3_attn.py --src <dir>/q30b_qk.lbt --q <dir>/i3q --out <dir>/q30b_qk_i3.vlt [--mats q,k,v,o] [--layers 0:48] [--skip-layers 0,1,46,47]

--q is the directory written by quant_i3.py (L<ll>_<mat>.npz with q uint8 [m, n] 0..7 and sm f16 [m, n/128, 2] = (scale, min)).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lbt_format as lbt  # noqa: E402
from lbt_stream import LbtStreamWriter  # noqa: E402

HDR, DATA, CHUNK = 512, 6144, 6656


def pack_i3(q: np.ndarray, sm: np.ndarray) -> np.ndarray:
    """q uint8 [m, n] (0..7), sm f16 [m, n/128, 2] -> uint8 [m, n/256*104] (stream layout of engine/i3_gemv.h)."""
    m, n = q.shape
    assert m % 64 == 0 and n % 256 == 0 and q.max() <= 7
    nt = n // 128; npair = nt // 2; rb = m // 64
    assert sm.shape == (m, nt, 2)
    h = np.ascontiguousarray(sm.astype(np.float16).reshape(rb, 64, npair, 2, 2).transpose(0, 2, 3, 4, 1))   # [rb, p, tile, field, row]
    hb = h.view(np.uint8).reshape(rb, npair, HDR)
    qq = q.reshape(rb, 64, npair, 2, 4, 32)                                                              # [rb, row, p, tile t, k, i]
    lo = np.zeros((rb, 64, npair, 2, 32), dtype=np.uint8)
    for k in range(4):
        lo |= ((qq[:, :, :, :, k, :] & 3) << (2 * k)).astype(np.uint8)
    hi = np.zeros((rb, 64, npair, 32), dtype=np.uint8)
    for t in range(2):
        for k in range(4):
            hi |= (((qq[:, :, :, t, k, :] >> 2) & 1) << (4 * t + k)).astype(np.uint8)
    row = np.concatenate([lo[:, :, :, 0, :], lo[:, :, :, 1, :], hi], axis=-1)                            # [rb, row, p, 96]
    db = np.ascontiguousarray(row.transpose(0, 2, 1, 3)).reshape(rb, npair, DATA)                        # [rb, p, row*96]
    out = np.concatenate([hb, db], axis=-1).reshape(m, npair * (CHUNK // 64))
    return out


def unpack_i3(buf: np.ndarray, m: int, n: int) -> np.ndarray:
    """-> dequantised f32 [m, n] (independent re-implementation of the format doc, used by --verify)."""
    nt = n // 128; npair = nt // 2; rb = m // 64
    c = buf.reshape(rb, npair, CHUNK)
    sm = np.ascontiguousarray(c[:, :, :HDR]).view(np.float16).reshape(rb, npair, 2, 2, 64).astype(np.float32)       # [rb, p, t, f, r]
    d = c[:, :, HDR:].reshape(rb, npair, 64, 96)
    out = np.empty((rb, 64, npair, 2, 4, 32), dtype=np.float32)
    for t in range(2):
        lo = d[..., 32 * t:32 * t + 32]; hi = d[..., 64:96]
        for k in range(4):
            qv = ((lo >> (2 * k)) & 3) | (((hi >> (4 * t + k)) & 1) << 2)                                          # [rb, p, row, i]
            scale = sm[:, :, t, 0, :][..., None]; mn = sm[:, :, t, 1, :][..., None]                               # [rb, p, row, 1]
            out[:, :, :, t, k, :] = (qv.astype(np.float32) * scale + mn).transpose(0, 2, 1, 3)
    return out.reshape(m, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--q', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--mats', default='q,k,v,o'); ap.add_argument('--layers', default='0:48'); ap.add_argument('--skip-layers', default='')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--n4', action='store_true', help='control: write the GPTQ levels (quant_i3.py --fmt n4) back as `.n4` + `.gs` (current nibble format) instead of `.i3`')
    ap.add_argument('--n4q', default='', help='dir of quant_i3.py --fmt n4 output; with --n4-spec the listed matrices become n4-GPTQ in the same file (mixed i3 / n4-GPTQ model)')
    ap.add_argument('--n4-spec', default='', help='e.g. "k:*,v:*,o:6+7+12" -> matrix:layers (* = all selected layers) written as n4-GPTQ; the rest of --mats stays i3')
    a = ap.parse_args()
    l0, l1 = (int(x) for x in a.layers.split(':'))
    skip = {int(x) for x in a.skip_layers.split(',') if x}
    mats = a.mats.split(',')
    sel = {(l, m) for l in range(l0, l1) if l not in skip for m in mats}
    src = lbt.LbtReader(a.src)
    n4set = set()
    for item in [x for x in a.n4_spec.split(',') if x]:
        mk, ls = item.split(':')
        for l in range(l0, l1):
            if l not in skip and (ls == '*' or l in {int(x) for x in ls.split('+')}): n4set.add((l, mk))
    kind = {lm: ('n4' if (a.n4 or lm in n4set) else 'i3') for lm in sel}
    repl = {}
    for (l, m) in sel:
        repl[f'blk.{l}.{m}.n4'] = (l, m); repl[f'blk.{l}.{m}.gs'] = (l, m)
        assert f'blk.{l}.{m}.n4' in src.entries, f'blk.{l}.{m}.n4 not in source'
    n_out = len(src.entries) - sum(1 for lm in sel if kind[lm] == 'i3')   # n4 + gs -> i3
    meta = dict(src.meta)
    meta['i3'] = {'version': 1, 'group': 128, 'mats': mats, 'layers': sorted({l for l, _ in sel}), 'count': sum(1 for lm in sel if kind[lm] == 'i3'), 'n4_gptq': sorted(f'{l}.{m}' for l, m in sel if kind[(l, m)] == 'n4'), 'source': os.path.basename(a.src),
                  'format': 'docs/i3_format.md', 'bpw': 3.25}
    meta_len = len(json.dumps(meta, ensure_ascii=False).encode('utf-8'))
    w = LbtStreamWriter(a.out, max_tensors=n_out + 8, meta_reserve=max(1 << 20, ((meta_len + 65536) // 4096 + 1) * 4096))
    w.meta = meta
    w.draft_layer_mask = src.draft_layer_mask
    w.flags = src.flags
    t0 = time.time(); done = 0; saved = 0
    for e in sorted(src.entries.values(), key=lambda e: e.offset):
        if e.name in repl and kind[repl[e.name]] == 'n4':
            import nibble_lbt as nib
            l, mk = repl[e.name]
            z = np.load(os.path.join(a.n4q or a.q, f'L{l:02d}_{mk}.npz'))
            if e.name.endswith('.n4'):
                w.add_tensor(e.name, nib.pack_n4(z['q']), lbt.U8, (e.shape))
            else:
                w.add_tensor(e.name, np.ascontiguousarray(z['sm'][..., 0]).astype(np.float16), lbt.F16, e.shape)
            done += 1
            continue
        if e.name in repl:
            if e.name.endswith('.gs'):
                saved += e.nbytes
                continue
            l, mk = repl[e.name]
            z = np.load(os.path.join(a.q, f'L{l:02d}_{mk}.npz'))
            q, sm = z['q'], z['sm']
            m, n = e.shape
            assert q.shape == (m, n), (q.shape, e.shape)
            packed = pack_i3(q, sm)
            if a.verify:
                ref = (q.astype(np.float32) * np.repeat(sm[..., 0].astype(np.float32), 128, 1) + np.repeat(sm[..., 1].astype(np.float32), 128, 1))
                assert np.array_equal(unpack_i3(packed, m, n), ref), f'verify failed {e.name}'
            w.add_tensor(f'blk.{l}.{mk}.i3', packed, lbt.U8, packed.shape)
            saved += e.nbytes - packed.nbytes
            done += 1
            continue
        raw = src.raw(e.name)
        if e.dtype == lbt.BF16:
            w.add_tensor(e.name, raw, lbt.BF16, e.shape)
        else:
            w.add_tensor(e.name, raw.view(lbt._NP[e.dtype]), e.dtype, e.shape)
    info = w.close()
    print(f'{time.strftime("%H:%M:%S")} wrote {a.out}: {info}, {done} i3 tensors, -{saved / 1e6:.1f} MB vs source ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
