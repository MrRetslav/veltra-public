"""Streaming Veltra model (`.vlt`, formerly `.lbt`) writer: tensors are appended to the file as they are produced (peak RAM = one tensor).

Same on-disk format as lbt_format.LbtWriter (header / meta / table / 64B-aligned data) - the only difference is that the
tensor table and meta region are RESERVED up front (`max_tensors`, `meta_reserve`) and filled in by close(). The reader
only trusts the offsets in the header, so unused table slots / meta slack are just padding.
"""
from __future__ import annotations

import json
import struct
from typing import Optional

import numpy as np

import lbt_format as lbt


class LbtStreamWriter:
    def __init__(self, path: str, max_tensors: int, meta_reserve: int = 1 << 20, legacy_magic: Optional[bool] = None):
        self.magic = lbt.write_magic(legacy_magic)   # b"VELTRA\0\0" unless legacy_magic / env VELTRA_LEGACY_MAGIC=1
        self.path = path
        self.max_tensors = int(max_tensors)
        self.meta: dict = {}
        self.draft_layer_mask = 0
        self.flags = 0
        self.meta_off = lbt.HEADER_SIZE
        self.table_off = lbt._align(self.meta_off + meta_reserve)
        self.data_off = lbt._align(self.table_off + lbt.ENTRY_SIZE * self.max_tensors)
        self.meta_reserve = meta_reserve
        self.f = open(path, "wb")
        self.f.truncate(self.data_off)          # sparse header area, overwritten by close()
        self.f.seek(self.data_off)
        self.entries = []                       # (name, dtype, shape, offset, nbytes)
        self.names = set()
        self.pos = self.data_off

    def set_meta(self, **kw):
        self.meta.update(kw)

    def set_draft_layer_mask(self, skip_layers):
        skip = sorted(int(i) for i in skip_layers)
        self.draft_layer_mask = sum(1 << i for i in skip if i < 64)
        self.meta["draft_layer_skip"] = skip

    def add_tensor(self, name: str, arr: np.ndarray, dtype: int, shape: Optional[tuple] = None):
        assert name not in self.names, f"duplicate tensor {name}"
        assert len(self.entries) < self.max_tensors, "more tensors than reserved in the table"
        a = np.ascontiguousarray(arr)
        if dtype != lbt.BF16:
            a = np.ascontiguousarray(arr, dtype=lbt._NP[dtype])
        shp = tuple(shape if shape is not None else arr.shape)
        assert len(name.encode()) < 96 and len(shp) <= 4, name
        raw = a.reshape(-1).view(np.uint8)
        assert self.f.tell() == self.pos
        off = self.pos
        self.f.write(memoryview(raw))
        end = lbt._align(off + raw.nbytes)
        if end > off + raw.nbytes:
            self.f.write(b"\0" * (end - off - raw.nbytes))
        self.pos = end
        self.names.add(name)
        self.entries.append((name, dtype, shp, off, int(raw.nbytes)))

    def add_bytes(self, name: str, data: bytes):
        self.add_tensor(name, np.frombuffer(data, dtype=np.uint8), lbt.U8, (len(data),))

    def close(self) -> dict:
        meta = json.dumps(self.meta, ensure_ascii=False).encode("utf-8")
        assert len(meta) <= self.meta_reserve, f"meta ({len(meta)} B) exceeds reserve"
        f = self.f
        f.seek(0)
        f.write(struct.pack("<8sIIQQQQQII", self.magic, lbt.VERSION, len(self.entries), self.meta_off, len(meta),
                            self.table_off, self.data_off, self.draft_layer_mask & (2 ** 64 - 1), self.flags, 0))
        f.write(meta)
        f.seek(self.table_off)
        for name, dtype, shp, o, n in self.entries:
            shape4 = list(shp) + [0] * (4 - len(shp))
            f.write(struct.pack("<96sII4QQQ8x", name.encode(), dtype, len(shp), *shape4, o, n))
        f.flush()
        f.close()
        return {"bytes": self.pos, "tensors": len(self.entries)}
