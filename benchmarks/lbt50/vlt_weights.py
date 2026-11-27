"""Reads a .vlt file and dequantises its K-quant linears exactly as the engine does (engine/q2k_gemv.cpp sb_decode), plus the Q4_K quantiser the
exporter uses (engine/q4k_quant.cpp), so offline experiments compare against the weights the running model really uses."""
import json
import struct

import numpy as np

SB = {"q2k": 84, "q3k": 114, "q4k": 148}


class Vlt:
    def __init__(self, path):
        self.m = np.memmap(path, dtype=np.uint8, mode="r")
        _, _, n, meta_off, meta_len, table_off, _, _, _, _ = struct.unpack("<8sIIQQQQQII", self.m[:64].tobytes())
        self.meta = json.loads(self.m[meta_off:meta_off + meta_len].tobytes())
        self.t = {}
        for i in range(n):
            name, dtype, ndim, a, b, c, d, off, nbytes = struct.unpack("<96sII4QQQ8x", self.m[table_off + 160 * i:table_off + 160 * (i + 1)].tobytes())
            self.t[name.rstrip(b"\0").decode()] = (dtype, (a, b, c, d)[:ndim], off, nbytes)

    def linear(self, base):
        """-> (float32 [rows, cols], bytes in the file) of the K-quant linear `base`."""
        for kind, size in SB.items():
            name = base + "." + kind
            if name in self.t:
                _, shape, off, nbytes = self.t[name]
                rows = shape[0]
                nsb = nbytes // (rows * size)
                blk = np.asarray(self.m[off:off + nbytes]).reshape(rows // 4, nsb, 4, size).transpose(0, 2, 1, 3).reshape(rows, nsb, size)
                return decode(kind, blk).reshape(rows, nsb * 256), nbytes
        raise KeyError(base)


def decode(kind, blk):
    """blk [..., SB] uint8 -> [..., 256] float32"""
    f16 = lambda off: blk[..., off:off + 2].copy().view(np.float16).astype(np.float32)
    lead = blk.shape[:-1]
    if kind in ("q2k", "q3k"):
        qs = blk[..., 16:80] if kind == "q2k" else blk[..., 32:96]
        qs = qs.reshape(*lead, 2, 32)
        u = np.stack([(qs >> (2 * j)) & 3 for j in range(4)], axis=-2)            # [..., h, j, l]
        if kind == "q3k":
            hm = blk[..., 0:32]
            hb = np.stack([np.stack([(hm >> (4 * h + j)) & 1 for j in range(4)], axis=-2) for h in range(2)], axis=-3)
            u = u | (hb << 2)
        u = u.reshape(*lead, 256).astype(np.float32)
        if kind == "q2k":
            sc = np.repeat((blk[..., 0:16] & 15).astype(np.float32), 16, axis=-1)
            mn = np.repeat((blk[..., 0:16] >> 4).astype(np.float32), 16, axis=-1)
            return (f16(80) * sc) * u - f16(82) * mn
        sc = np.repeat(blk[..., 96:112].view(np.int8).astype(np.float32), 16, axis=-1)
        return (f16(112) * sc) * (u - 4.0)
    qs = blk[..., 20:148].reshape(*lead, 4, 32)
    u = np.concatenate([qs & 15, qs >> 4], axis=-1).reshape(*lead, 256).astype(np.float32)   # per 64 group: low nibbles, then high
    sc = np.repeat(blk[..., 4:12].astype(np.float32), 32, axis=-1)
    mn = np.repeat(blk[..., 12:20].astype(np.float32), 32, axis=-1)
    return (f16(0) * sc) * u - f16(2) * mn
