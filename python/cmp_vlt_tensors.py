#!/usr/bin/env python3
"""Compare two .vlt/.lbt files tensor by tensor (names, dtype, shape, bytes). usage: cmp_vlt_tensors.py A B"""
import struct
import sys

import numpy as np


def tensors(path):
    m = np.memmap(path, dtype=np.uint8, mode="r")
    _, _, n, _, _, table_off, _, _, _, _ = struct.unpack("<8sIIQQQQQII", m[:64].tobytes())
    out = {}
    for i in range(n):
        name, dtype, ndim, a, b, c, d, off, nbytes = struct.unpack("<96sII4QQQ8x", m[table_off + 160 * i:table_off + 160 * (i + 1)].tobytes())
        out[name.rstrip(b"\0").decode()] = (dtype, (a, b, c, d)[:ndim], m[off:off + nbytes])
    return out


a, b = tensors(sys.argv[1]), tensors(sys.argv[2])
missing = sorted(set(a) ^ set(b))
diff = []
for k in sorted(set(a) & set(b)):
    da, sa, xa = a[k]
    db, sb, xb = b[k]
    if da != db or sa != sb or len(xa) != len(xb):
        diff.append((k, "layout", da, sa, db, sb))
    elif not np.array_equal(xa, xb):
        diff.append((k, "bytes", int((xa != xb).sum()), len(xa)))
print("tensors: %d vs %d | only in one file: %d %s | different: %d" % (len(a), len(b), len(missing), missing[:4], len(diff)))
for d in diff[:3]:
    print(" ", d)
sys.exit(1 if missing or diff else 0)
