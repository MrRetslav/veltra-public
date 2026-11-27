"""Veltra model container (`.vlt`, formerly `.lbt`) writer/reader + +-1 matrix packing. See docs/INTERFACES.md.

Magic: writers emit b"VELTRA\\0\\0" by default; pass legacy_magic=True (or set env VELTRA_LEGACY_MAGIC=1) to
write the old b"LITTLEBT" for engines built before the rename. Readers accept both; the layout is identical.

Layout (little endian, every tensor 64-byte aligned):
  [64B header][meta JSON][tensor table: n * 160B][tensor data ...]
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

MAGIC = b"VELTRA\0\0"          # current magic (8 bytes)
MAGIC_LEGACY = b"LITTLEBT"      # pre-rename magic, still accepted by every reader
MAGICS = (MAGIC, MAGIC_LEGACY)
DEFAULT_EXT = ".vlt"            # default model file extension (".lbt" files keep working)
VERSION = 1
ALIGN = 64
ENTRY_SIZE = 160
HEADER_SIZE = 64

# dtype codes
F32, F16, BF16, U8, U64, LUT8, I8, I32 = range(8)
_NP = {F32: np.float32, F16: np.float16, U8: np.uint8, U64: np.uint64, LUT8: np.uint8, I8: np.int8, I32: np.int32}

ROW_BLOCK = 64
K_TILE = 128
FLAG_HAS_U64 = 1


def write_magic(legacy_magic: Optional[bool] = None) -> bytes:
    """Magic a writer emits: MAGIC, or MAGIC_LEGACY when legacy_magic is True (None: env VELTRA_LEGACY_MAGIC=1)."""
    if legacy_magic is None:
        legacy_magic = os.environ.get("VELTRA_LEGACY_MAGIC", os.environ.get("LB_LEGACY_MAGIC", "0")) not in ("", "0")
    return MAGIC_LEGACY if legacy_magic else MAGIC


def is_model_magic(magic: bytes) -> bool:
    return bytes(magic[:8]) in MAGICS


def _align(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def pad_rows(r: int) -> int:
    return _align(r, ROW_BLOCK)


def pad_k(k: int) -> int:
    return _align(k, K_TILE)


# ----------------------------------------------------------------------------- packing
def signs_to_bool(signs) -> np.ndarray:
    """+-1 (any numeric array / torch tensor) -> bool matrix (True = +1, 0 maps to -1)."""
    if hasattr(signs, "detach"):
        signs = signs.detach().cpu().numpy()
    return np.asarray(signs) > 0


def pack_bits_u64(b: np.ndarray) -> np.ndarray:
    """bool [rows, cols] -> uint64 [rows, ceil(cols/64)], bit k of row j = word k//64, bit k%64."""
    rows, cols = b.shape
    words = (cols + 63) // 64
    padded = np.zeros((rows, words * 64), dtype=np.uint8)
    padded[:, :cols] = b
    by = np.packbits(padded, axis=1, bitorder="little")  # [rows, words*8] bytes, LSB first
    return np.ascontiguousarray(by).view("<u8").reshape(rows, words)


def pack_lut8(b: np.ndarray) -> np.ndarray:
    """bool [rows, cols] -> LUT8 uint8 [row_block][pair][64] (flattened).

    nibble(row, g) = b[4g] | b[4g+1]<<1 | b[4g+2]<<2 | b[4g+3]<<3 ; byte(row, p) = nib(2p) | nib(2p+1)<<4.
    Rows padded to 64, K padded to 128 (padding bits = 0).
    """
    rows, cols = b.shape
    rp, kp = pad_rows(rows), pad_k(cols)
    full = np.zeros((rp, kp), dtype=np.uint8)
    full[:rows, :cols] = b
    nib = (full[:, 0::4] | (full[:, 1::4] << 1) | (full[:, 2::4] << 2) | (full[:, 3::4] << 3)).astype(np.uint8)  # [rp, kp/4]
    byte = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)  # [rp, kp/8] (pairs)
    pairs = kp // 8
    out = byte.reshape(rp // ROW_BLOCK, ROW_BLOCK, pairs).transpose(0, 2, 1)  # [rb, pair, 64]
    return np.ascontiguousarray(out).reshape(-1)


def unpack_lut8(lut: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Inverse of pack_lut8 -> bool [rows, cols]."""
    rp, kp = pad_rows(rows), pad_k(cols)
    pairs = kp // 8
    byte = lut.reshape(rp // ROW_BLOCK, pairs, ROW_BLOCK).transpose(0, 2, 1).reshape(rp, pairs)
    nib = np.empty((rp, kp // 4), dtype=np.uint8)
    nib[:, 0::2] = byte & 0xF
    nib[:, 1::2] = byte >> 4
    full = np.empty((rp, kp), dtype=np.uint8)
    for j in range(4):
        full[:, j::4] = (nib >> j) & 1
    return full[:rows, :cols].astype(bool)


# ----------------------------------------------------------------------------- container
@dataclass
class TensorEntry:
    name: str
    dtype: int
    shape: tuple
    offset: int
    nbytes: int


class LbtWriter:
    def __init__(self, legacy_magic: Optional[bool] = None):
        self.magic = write_magic(legacy_magic)
        self.meta: dict = {}
        self.draft_layer_mask: int = 0
        self.flags: int = 0
        self._tensors: List[tuple] = []  # (name, dtype, shape, bytes)
        self._names = set()

    def set_meta(self, **kw):
        self.meta.update(kw)

    def set_draft_layer_mask(self, skip_layers) -> None:
        """skip_layers: iterable of layer indices (or bool list) skipped during the draft phase.
        Header u64 holds layers 0..63; the full list is also stored in meta['draft_layer_skip']."""
        if skip_layers and isinstance(next(iter(skip_layers)), (bool, np.bool_)):
            skip_layers = [i for i, v in enumerate(skip_layers) if v]
        skip = sorted(int(i) for i in skip_layers)
        self.draft_layer_mask = sum(1 << i for i in skip if i < 64)
        self.meta["draft_layer_skip"] = skip

    def add_tensor(self, name: str, arr: np.ndarray, dtype: Optional[int] = None, shape: Optional[tuple] = None):
        assert name not in self._names, f"duplicate tensor {name}"
        if dtype is None:
            dtype = {np.dtype(np.float32): F32, np.dtype(np.float16): F16, np.dtype(np.uint8): U8,
                     np.dtype(np.uint64): U64, np.dtype(np.int8): I8, np.dtype(np.int32): I32}[arr.dtype]
        if dtype == BF16:
            raw = arr.tobytes()
        else:
            raw = np.ascontiguousarray(arr, dtype=_NP[dtype]).tobytes()
        shp = tuple(shape if shape is not None else arr.shape)
        assert len(name.encode()) < 96 and len(shp) <= 4, name
        self._names.add(name)
        self._tensors.append((name, dtype, shp, raw))

    def add_bytes(self, name: str, data: bytes):
        self.add_tensor(name, np.frombuffer(data, dtype=np.uint8), U8, (len(data),))

    def add_binary_matrix(self, name: str, signs, with_u64: bool = True):
        """Add a +-1 matrix [rows, cols] as `{name}_lut` (LUT8) and optionally `{name}_bits` (uint64)."""
        b = signs_to_bool(signs)
        rows, cols = b.shape
        self.add_tensor(f"{name}_lut", pack_lut8(b), LUT8, (rows, cols))
        if with_u64:
            self.add_tensor(f"{name}_bits", pack_bits_u64(b), U64, (rows, cols))
            self.flags |= FLAG_HAS_U64

    def write(self, path: str):
        meta = json.dumps(self.meta, ensure_ascii=False).encode("utf-8")
        meta_off = HEADER_SIZE
        table_off = _align(meta_off + len(meta))
        data_off = _align(table_off + ENTRY_SIZE * len(self._tensors))
        off = data_off
        entries = []
        for name, dtype, shp, raw in self._tensors:
            entries.append((name, dtype, shp, off, len(raw)))
            off = _align(off + len(raw))
        with open(path, "wb") as f:
            hdr = struct.pack("<8sIIQQQQQII", self.magic, VERSION, len(self._tensors), meta_off, len(meta),
                              table_off, data_off, self.draft_layer_mask & (2**64 - 1), self.flags, 0)
            assert len(hdr) == HEADER_SIZE
            f.write(hdr)
            f.write(meta)
            f.write(b"\0" * (table_off - f.tell()))
            for name, dtype, shp, o, n in entries:
                shape4 = list(shp) + [0] * (4 - len(shp))
                e = struct.pack("<96sII4QQQ8x", name.encode(), dtype, len(shp), *shape4, o, n)
                assert len(e) == ENTRY_SIZE
                f.write(e)
            f.write(b"\0" * (data_off - f.tell()))
            for (name, dtype, shp, raw), (_, _, _, o, n) in zip(self._tensors, entries):
                assert f.tell() == o
                f.write(raw)
                f.write(b"\0" * (_align(o + n) - o - n))
        return {"bytes": off, "tensors": len(entries)}


class LbtReader:
    def __init__(self, path: str):
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        (magic, ver, n, meta_off, meta_len, table_off, data_off, mask, flags, _) = struct.unpack(
            "<8sIIQQQQQII", bytes(self.mm[:HEADER_SIZE]))
        assert magic in MAGICS and ver == VERSION, "not a Veltra v1 model file (.vlt / legacy .lbt)"
        self.magic = magic
        self.draft_layer_mask, self.flags = mask, flags
        self.meta = json.loads(bytes(self.mm[meta_off:meta_off + meta_len]).decode("utf-8"))
        self.entries: Dict[str, TensorEntry] = {}
        for i in range(n):
            raw = bytes(self.mm[table_off + i * ENTRY_SIZE: table_off + (i + 1) * ENTRY_SIZE])
            name, dtype, ndim, s0, s1, s2, s3, off, nb = struct.unpack("<96sII4QQQ8x", raw)
            name = name.split(b"\0", 1)[0].decode()
            self.entries[name] = TensorEntry(name, dtype, (s0, s1, s2, s3)[:ndim], off, nb)

    def raw(self, name: str) -> np.ndarray:
        e = self.entries[name]
        return np.asarray(self.mm[e.offset:e.offset + e.nbytes])

    def tensor(self, name: str) -> np.ndarray:
        e = self.entries[name]
        r = self.raw(name)
        if e.dtype == BF16:
            return (r.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(e.shape)
        if e.dtype == LUT8:
            return r  # flat packed bytes; use binary_matrix() for signs
        return r.view(_NP[e.dtype]).reshape(e.shape)

    def binary_matrix(self, name: str) -> np.ndarray:
        """+-1 float32 matrix from `{name}_lut` (works even when u64 bits were dropped)."""
        e = self.entries[f"{name}_lut"]
        b = unpack_lut8(self.raw(f"{name}_lut"), *e.shape)
        return np.where(b, 1.0, -1.0).astype(np.float32)


def _self_check() -> None:
    """Round trip with both magics: new writers emit MAGIC, legacy_magic=True emits MAGIC_LEGACY, readers accept both."""
    import tempfile
    d = tempfile.mkdtemp()
    signs = np.where(np.random.default_rng(0).standard_normal((70, 130)) > 0, 1.0, -1.0)
    for legacy in (False, True):
        w = LbtWriter(legacy_magic=legacy)
        w.set_meta(arch="selfcheck")
        w.add_tensor("x", np.arange(5, dtype=np.float32))
        w.add_binary_matrix("m", signs)
        path = os.path.join(d, "t" + (".lbt" if legacy else DEFAULT_EXT))
        w.write(path)
        with open(path, "rb") as f:
            assert f.read(8) == (MAGIC_LEGACY if legacy else MAGIC)
        r = LbtReader(path)
        assert r.meta["arch"] == "selfcheck" and np.array_equal(r.tensor("x"), np.arange(5, dtype=np.float32))
        assert np.array_equal(r.binary_matrix("m"), signs.astype(np.float32))
        del r
    print("lbt_format self-check OK (magics %r and %r)" % (MAGIC, MAGIC_LEGACY))


if __name__ == "__main__":
    _self_check()
