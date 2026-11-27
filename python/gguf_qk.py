"""llama.cpp K-quant superblocks (Q2_K / Q3_K / Q4_K) <-> the engine's `.qk` layout (engine/q2k_gemv.h, docs/qk_format.md).

The transcode is lossless: every (d, dmin, scale, min, quant) value is kept; only the byte arrangement changes
  * 4 consecutive rows are interleaved per superblock: [group of 4 rows][superblock][row 0..3][SB bytes]  (one sequential stream per group)
  * Q3_K / Q4_K 6-bit scales are unpacked to bytes (Q3_K: int8 scale - 32; Q4_K: u8 scale[8] + u8 min[8]).
Per (row, superblock) bytes:
  Q2_K  84: sc[16] | qs[64] | d f16 | dmin f16                       (= ggml block_q2_K)
  Q3_K 114: hmask[32] | qs[64] | sc[16] int8 | d f16
  Q4_K 148: d f16 | dmin f16 | sc[8] | mn[8] | qs[128]
`unpack_qk` is an independent inverse that dequantises with gguf-py's formulas (used by the verifier).
"""
from __future__ import annotations

import numpy as np

QK_K = 256
SUFFIX = {"Q2_K": "q2k", "Q3_K": "q3k", "Q4_K": "q4k"}
GGML_BYTES = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144}
QK_BYTES = {"Q2_K": 84, "Q3_K": 114, "Q4_K": 148}


def q4k_scales(s12: np.ndarray):
    """ggml get_scale_min_k4 for all blocks: s12 [nb, 12] u8 -> (sc [nb, 8], mn [nb, 8]) u8 (same as gguf-py Q4_K.get_scale_min)."""
    nb = s12.shape[0]
    s = s12.reshape(nb, 3, 4)
    d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.astype(np.uint8), mn.astype(np.uint8)


def q3k_scales(s12: np.ndarray) -> np.ndarray:
    """ggml Q3_K 12 packed bytes -> 16 int8 scales (already minus 32), as gguf-py Q3_K.dequantize_blocks."""
    nb = s12.shape[0]
    l = s12[:, :8].reshape(nb, 1, 8) >> np.array([0, 4], dtype=np.uint8).reshape(1, 2, 1)
    l = l.reshape(nb, 16)
    h = s12[:, 8:].reshape(nb, 1, 4) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 4, 1)
    h = h.reshape(nb, 16)
    sc = (l & np.uint8(0x0F)) | ((h & np.uint8(0x03)) << np.uint8(4))
    return (sc.astype(np.int8) - np.int8(32)).astype(np.int8)


def pack_qk(raw: np.ndarray, qtype: str) -> np.ndarray:
    """raw: ggml bytes [rows, nsb * GGML_BYTES] (rows % 4 == 0) -> flat u8 `.qk` stream [rows * nsb * QK_BYTES]."""
    gb, qb = GGML_BYTES[qtype], QK_BYTES[qtype]
    rows = raw.shape[0]
    assert rows % 4 == 0 and raw.shape[1] % gb == 0, (raw.shape, qtype)
    nsb = raw.shape[1] // gb
    b = np.ascontiguousarray(raw).reshape(rows * nsb, gb)
    out = np.empty((rows * nsb, qb), np.uint8)
    if qtype == "Q2_K":
        out[:] = b
    elif qtype == "Q3_K":
        out[:, 0:96] = b[:, 0:96]                                   # hmask + qs
        out[:, 96:112] = q3k_scales(b[:, 96:108]).view(np.uint8)
        out[:, 112:114] = b[:, 108:110]                             # d
    else:
        sc, mn = q4k_scales(b[:, 4:16])
        out[:, 0:4] = b[:, 0:4]                                     # d, dmin
        out[:, 4:12] = sc
        out[:, 12:20] = mn
        out[:, 20:148] = b[:, 16:144]
    out = out.reshape(rows // 4, 4, nsb, qb).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(out).reshape(-1)


def unpack_qk(buf: np.ndarray, qtype: str, rows: int, n: int) -> np.ndarray:
    """`.qk` stream -> float32 weights [rows, n]; formulas of gguf-py (dl = d*sc, then dl*q - dmin*mn)."""
    qb = QK_BYTES[qtype]
    nsb = n // QK_K
    b = np.asarray(buf, np.uint8).reshape(rows // 4, nsb, 4, qb).transpose(0, 2, 1, 3).reshape(rows * nsb, qb)
    nb = rows * nsb
    f16 = lambda a: np.ascontiguousarray(a).view(np.float16).astype(np.float32).reshape(nb, 1)
    if qtype == "Q2_K":
        sc, qs = b[:, :16], b[:, 16:80]
        d, dmin = f16(b[:, 80:82]), f16(b[:, 82:84])
        dl = (d * (sc & 0xF).astype(np.float32)).reshape(nb, 16, 1)
        ml = (dmin * (sc >> 4).astype(np.float32)).reshape(nb, 16, 1)
        q = ((qs.reshape(nb, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & np.uint8(3)).reshape(nb, 16, 16).astype(np.float32)
        w = dl * q - ml
    elif qtype == "Q3_K":
        hm, qs, sc = b[:, :32], b[:, 32:96], b[:, 96:112].view(np.int8).astype(np.float32)
        d = f16(b[:, 112:114])
        dl = (d * sc).reshape(nb, 16, 1)
        lo = ((qs.reshape(nb, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & np.uint8(3)).reshape(nb, 16, 16).astype(np.int32)
        hb = ((hm.reshape(nb, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & np.uint8(1)).reshape(nb, 16, 16).astype(np.int32)
        w = dl * ((lo + 4 * hb - 4).astype(np.float32))
    else:
        d, dmin = f16(b[:, 0:2]), f16(b[:, 2:4])
        sc, mn, qs = b[:, 4:12], b[:, 12:20], b[:, 20:148]
        dl = (d * sc.astype(np.float32)).reshape(nb, 8, 1)
        ml = (dmin * mn.astype(np.float32)).reshape(nb, 8, 1)
        q = ((qs.reshape(nb, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)).reshape(nb, 8, 32).astype(np.float32)
        w = dl * q - ml
    return w.reshape(rows, n)
