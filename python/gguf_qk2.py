"""`.qk` v2: gguf_qk.py (Q2_K / Q3_K / Q4_K) plus the llama.cpp i-quants used by the Qwen3.6-35B-A3B `UD-Q2_K_XL` GGUF experts
(IQ2_XS, IQ3_XXS, IQ4_XS).  For the i-quants the transcode is a PURE BYTE PERMUTATION of the ggml blocks (no scale unpacking):

  [group of 4 rows][superblock][row 0..3][ggml block bytes]      (same stream order as the K-quant `.qk` tensors)

Block bytes (ggml-common.h, QK_K = 256):
  IQ2_XS   74 B  d f16 | qs[32] u16 (9-bit grid index | 7-bit sign index) | scales[8] (two 4-bit scales per byte)   2.3125 bpw
  IQ3_XXS  98 B  d f16 | qs[96]  (64 grid indices u8, then 8 x u32 = 4x7-bit sign idx + 4-bit scale)               3.0625 bpw
  IQ4_XS  136 B  d f16 | scales_h u16 | scales_l[4] | qs[128] (4-bit indices into kvalues_iq4nl[16])              4.25   bpw
`unpack_any` = inverse permutation + gguf-py `dequantize`; K-quants keep using gguf_qk.unpack_qk (independent formulas).
"""
from __future__ import annotations

import numpy as np

import gguf_qk as k1

QK_K = 256
SUFFIX = dict(k1.SUFFIX, IQ2_XS="iq2xs", IQ3_XXS="iq3xxs", IQ4_XS="iq4xs")
GGML_BYTES = dict(k1.GGML_BYTES, IQ2_XS=74, IQ3_XXS=98, IQ4_XS=136)
QK_BYTES = dict(k1.QK_BYTES, IQ2_XS=74, IQ3_XXS=98, IQ4_XS=136)
RAW_TYPES = ("IQ2_XS", "IQ3_XXS", "IQ4_XS")
BPW = {t: QK_BYTES[t] * 8 / QK_K for t in SUFFIX}


def pack_any(raw: np.ndarray, qtype: str) -> np.ndarray:
    """ggml bytes [rows, nsb * GGML_BYTES] -> flat u8 `.qk` stream."""
    if qtype not in RAW_TYPES:
        return k1.pack_qk(raw, qtype)
    gb = GGML_BYTES[qtype]
    rows = raw.shape[0]
    assert rows % 4 == 0 and raw.shape[1] % gb == 0, (raw.shape, qtype)
    nsb = raw.shape[1] // gb
    b = np.ascontiguousarray(raw).reshape(rows // 4, 4, nsb, gb).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(b).reshape(-1)


def unpack_to_ggml(buf: np.ndarray, qtype: str, rows: int, n: int) -> np.ndarray:
    """Inverse byte permutation for the raw i-quant types -> ggml bytes [rows, nsb * GGML_BYTES]."""
    assert qtype in RAW_TYPES
    gb, nsb = GGML_BYTES[qtype], n // QK_K
    b = np.asarray(buf, np.uint8).reshape(rows // 4, nsb, 4, gb).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(b).reshape(rows, nsb * gb)


def unpack_any(buf: np.ndarray, qtype: str, rows: int, n: int, gguf=None) -> np.ndarray:
    """`.qk` stream -> f32 [rows, n].  K-quants: gguf_qk.unpack_qk; i-quants: de-permute + gguf-py dequantize."""
    if qtype not in RAW_TYPES:
        return k1.unpack_qk(buf, qtype, rows, n)
    if gguf is None:
        import gguf
    return gguf.quants.dequantize(unpack_to_ggml(buf, qtype, rows, n), getattr(gguf.GGMLQuantizationType, qtype))
