#!/usr/bin/env python3
"""Generate engine/unicode_tables.h: letters (L), numbers (N), whitespace, and NFC data.

Usage:
  python scripts/gen_unicode_tables.py [--ucd DIR] [--norm-ucd DIR|python] [--out PATH]

--ucd DIR  directory holding UnicodeData.txt, DerivedNormalizationProps.txt and PropList.txt of one Unicode
           release (e.g. https://www.unicode.org/Public/16.0.0/ucd/). Recommended: HuggingFace `tokenizers`
           ships a recent Oniguruma/unicode-normalization, so a recent release (16.0) matches it best.
           Without --ucd the (older) database of the running Python is used.
           Shipped tables: `--ucd <Unicode 16.0> --norm-ucd <Unicode 9.0>` (fetch UnicodeData.txt,
           DerivedNormalizationProps.txt, PropList.txt from https://www.unicode.org/Public/<ver>/ucd/ into
           tests/data/ucd16 and tests/data/ucd9). Verified: HF letters/numbers == Unicode 16 (Oniguruma), HF NFC == 9.0.
--norm-ucd same, but only for the NFC data (combining classes, decompositions, compositions). HF's
           `tokenizers` normalizers use an OLDER Unicode release than its regex engine, so for bit-exact
           parity the NFC part may need an older UCD ("python" = running Python's unicodedata).
"""
import argparse
import sys
import unicodedata
from pathlib import Path

MAXCP = 0x110000


def load_ucd(d):
    d = Path(d)
    cat = ["Cn"] * MAXCP
    ccc = [0] * MAXCP
    dm = {}
    first = None
    for line in (d / "UnicodeData.txt").read_text(encoding="utf-8").splitlines():
        f = line.split(";")
        cp = int(f[0], 16)
        name = f[1]
        if name.endswith(", First>"):
            first = cp
            continue
        lo = first if name.endswith(", Last>") else cp
        first = None
        for c in range(lo, cp + 1):
            cat[c] = f[2]
            ccc[c] = int(f[3])
        if f[5] and not f[5].startswith("<"):
            dm[cp] = [int(x, 16) for x in f[5].split()]
    excl = set()
    for line in (d / "DerivedNormalizationProps.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or "Full_Composition_Exclusion" not in line:
            continue
        rng = line.split(";")[0].strip()
        a, _, b = rng.partition("..")
        for c in range(int(a, 16), int(b or a, 16) + 1):
            excl.add(c)
    ws = []
    for line in (d / "PropList.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        rng, prop = [x.strip() for x in line.split(";")]
        if prop == "White_Space":
            a, _, b = rng.partition("..")
            ws.append((int(a, 16), int(b or a, 16)))
    return cat, ccc, dm, excl, sorted(ws), "UCD " + d.name + " (" + str((d / "UnicodeData.txt").stat().st_size) + " bytes UnicodeData)"


def load_python():
    ud = unicodedata
    cat = ["Cn"] * MAXCP
    ccc = [0] * MAXCP
    dm = {}
    excl = set()
    for cp in range(MAXCP):
        if 0xD800 <= cp <= 0xDFFF:
            cat[cp] = "Cs"
            continue
        ch = chr(cp)
        cat[cp] = ud.category(ch)
        ccc[cp] = ud.combining(ch)
        d = ud.decomposition(ch)
        if d and not d.startswith("<"):
            dm[cp] = [int(x, 16) for x in d.split()]
            if ud.normalize("NFC", ch) != ch:
                excl.add(cp)
    ws = [(0x9, 0xD), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
          (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]
    return cat, ccc, dm, excl, ws, "python unicodedata " + ud.unidata_version


def ranges_for(pred):
    out, start = [], None
    for cp in range(MAXCP):
        ok = pred(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAXCP - 1))
    return out


def fmt_rows(name, rows, cols, per_line, conv):
    lines = [f"inline constexpr uint32_t {name}[][{cols}] = {{"]
    row = []
    for r in rows:
        row.append("{" + ",".join(conv(x) for x in r) + "}")
        if len(row) == per_line:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    lines.append(f"inline constexpr size_t {name}Count = sizeof({name}) / sizeof({name}[0]);")
    return "\n".join(lines)


def hx(x):
    return f"0x{x:X}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucd")
    ap.add_argument("--norm-ucd")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "engine" / "unicode_tables.h"))
    a = ap.parse_args()
    cat, ccc, dm, excl, ws, src = load_ucd(a.ucd) if a.ucd else load_python()
    if a.norm_ucd:
        _, ccc, dm, excl, _, nsrc = load_python() if a.norm_ucd == "python" else load_ucd(a.norm_ucd)
        src += "; NFC data from " + nsrc

    letters = ranges_for(lambda cp: cat[cp][0] == "L")
    numbers = ranges_for(lambda cp: cat[cp][0] == "N")
    marks = ranges_for(lambda cp: cat[cp][0] == "M")

    ccc_ranges, run = [], None
    for cp in range(0x300, MAXCP):
        c = ccc[cp]
        if run is not None and c == run[2] and cp == run[1] + 1:
            run[1] = cp
            continue
        if run is not None:
            ccc_ranges.append(tuple(run))
            run = None
        if c:
            run = [cp, cp, c]
    if run is not None:
        ccc_ranges.append(tuple(run))

    def full(cp):
        if cp in dm:
            out = []
            for x in dm[cp]:
                out.extend(full(x))
            return out
        return [cp]

    decomp, pool = [], []
    for cp in sorted(dm):
        if 0xAC00 <= cp <= 0xD7A3:
            continue
        cps = full(cp)
        decomp.append((cp, len(pool), len(cps)))
        pool.extend(cps)
    comp = sorted((v[0], v[1], cp) for cp, v in dm.items()
                  if len(v) == 2 and cp not in excl and not (0xAC00 <= cp <= 0xD7A3))

    o = ["// GENERATED by scripts/gen_unicode_tables.py -- do not edit.",
         f"// Source: {src}",
         "#pragma once", "#include <cstddef>", "#include <cstdint>", "", "namespace lb::uni {", "",
         "// General category L (letters), inclusive ranges, sorted.",
         fmt_rows("kLetter", letters, 2, 6, hx),
         "// General category N (numbers).",
         fmt_rows("kNumber", numbers, 2, 6, hx),
         "// General category M (marks; Qwen3.5/3.6 pre-tokenizer [\\p{L}\\p{M}]).",
         fmt_rows("kMark", marks, 2, 6, hx),
         "// White_Space property (regex \\s).",
         fmt_rows("kSpace", ws, 2, 6, hx),
         "// Canonical combining class ranges {lo, hi, ccc}.",
         fmt_rows("kCcc", ccc_ranges, 3, 6, hx),
         "// Full canonical decomposition: {cp, offset into kDecompPool, length}, sorted by cp.",
         fmt_rows("kDecomp", decomp, 3, 6, hx),
         "inline constexpr uint32_t kDecompPool[] = {"]
    for i in range(0, len(pool), 12):
        o.append("    " + ", ".join(hx(x) for x in pool[i:i + 12]) + ",")
    o.append("};")
    o.append("// Primary composites {first, second, composite}, sorted by (first, second).")
    o.append(fmt_rows("kCompose", comp, 3, 5, hx))
    o += ["", "}  // namespace lb::uni", ""]
    out = Path(a.out)
    out.write_text("\n".join(o), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes) from {src}: L={len(letters)} N={len(numbers)} ccc={len(ccc_ranges)} "
          f"decomp={len(decomp)} pool={len(pool)} compose={len(comp)}")


if __name__ == "__main__":
    main()
