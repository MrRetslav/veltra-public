"""Nibble-INT4 (.n4) repack of P=4 grid linears; contract: docs/bitplane_spec.md section 5."""
from __future__ import annotations
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lbt_format as lbt
import grid_lbt as gl

TILE, RB = 128, 64


def pack_n4(j: np.ndarray) -> np.ndarray:
    """int level index [m, n] (0..15, n % 128 == 0) -> flat uint8 [pad_rows(m)/64 * (n/128) * 4096], row-block-major."""
    m, n = j.shape
    assert n % TILE == 0
    rp = lbt.pad_rows(m)
    jp = np.zeros((rp, n), dtype=np.uint8); jp[:m] = j
    t = jp.reshape(rp // RB, RB, n // TILE, 2, 64)               # [rb, row, tile, half, i]
    b = (t[:, :, :, 0, :] | (t[:, :, :, 1, :] << 4)).astype(np.uint8)   # [rb, row, tile, i]
    return np.ascontiguousarray(b.transpose(0, 2, 1, 3)).reshape(-1)    # [rb, tile, row, i]


def unpack_n4(flat: np.ndarray, m: int, n: int) -> np.ndarray:
    rp = lbt.pad_rows(m)
    b = np.asarray(flat, dtype=np.uint8).reshape(rp // RB, n // TILE, RB, 64).transpose(0, 2, 1, 3)   # [rb, row, tile, i]
    j = np.empty((rp // RB, RB, n // TILE, 2, 64), dtype=np.uint8)
    j[:, :, :, 0, :] = b & 15; j[:, :, :, 1, :] = b >> 4
    return j.reshape(rp, n)[:m]


def planes_to_j(planes) -> np.ndarray:
    assert len(planes) == 4
    return ((planes[0].astype(np.uint8) << 3) | (planes[1].astype(np.uint8) << 2) | (planes[2].astype(np.uint8) << 1) | planes[3].astype(np.uint8))


def n4_dequant(flat, gs, m, n) -> np.ndarray:
    q = unpack_n4(flat, m, n).astype(np.float32) - 7.5
    return q * np.repeat(np.asarray(gs, dtype=np.float32), TILE, axis=1)


def repack_file(src: str, dst: str, verbose: bool = True) -> dict:
    rd = lbt.LbtReader(src)
    w = lbt.LbtWriter()
    w.meta = dict(rd.meta); w.draft_layer_mask = rd.draft_layer_mask; w.flags = rd.flags
    conv = skipped = 0
    bases = sorted({n[:-len(".gp3_lut")] for n in rd.entries if n.endswith(".gp3_lut")})
    drop = set()
    for b in bases:
        for p in range(4): drop.add(f"{b}.gp{p}_lut")
    for name, e in rd.entries.items():
        if name in drop:
            continue
        raw = bytes(rd.raw(name))
        w._tensors.append((name, e.dtype, tuple(e.shape), raw)); w._names.add(name)
    for b in bases:
        planes = []
        for p in range(4):
            e = rd.entries[f"{b}.gp{p}_lut"]
            planes.append(gl.unpack_tilemajor(rd.raw(f"{b}.gp{p}_lut"), e.shape[0], e.shape[1]))
        m, n = planes[0].shape
        flat = pack_n4(planes_to_j(planes))
        w._tensors.append((f"{b}.n4", lbt.U8, (m, n), flat.tobytes())); w._names.add(f"{b}.n4")
        conv += 1
    w.meta.setdefault("grid", {})["nibble"] = True
    info = w.write(dst)
    return {"converted": conv, "bytes_in": os.path.getsize(src), "bytes_out": os.path.getsize(dst), **info}


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for m, n in ((64, 128), (100, 256), (2048, 384)):
        j = rng.integers(0, 16, size=(m, n)).astype(np.uint8)
        assert (unpack_n4(pack_n4(j), m, n) == j).all()
        pl = gl.levels_to_planes(j, 4)
        assert (planes_to_j(pl) == j).all()
        gs = (rng.random((m, n // 128)) + .1).astype(np.float16)
        assert np.allclose(n4_dequant(pack_n4(j), gs, m, n), gl.grid_dequant(pl, gs, 128))
    print("nibble_lbt self-check OK")
