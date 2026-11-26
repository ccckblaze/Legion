#!/usr/bin/env python3
"""tools/nvfp4_scale_pow2.py - check whether an NVFP4 pack's per-expert scales are powers of two.

The scale fold at conversion (folding the per-expert fp32 scalar into the E4M3 block scale bytes)
is bit-exact only for powers of two. docs/ac922-nvfp4-residency.md: the probe over the converted
Strata pack found 0 of 8,192 pow2, which is what killed that route.

Usage: nvfp4_scale_pow2.py <shard.gguf> [max_values]
"""
import math
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from strata_pack_to_legion import Gguf  # noqa: E402


def main() -> None:
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    g = Gguf(path)
    f = open(path, "rb")
    checked = pow2 = 0
    lo = hi = None
    for name, dims, ty, toff in g.tensors:
        if not name.endswith("_exps.scale"):
            continue
        n = dims[0] if dims else 1
        f.seek(g.data_base + toff)
        for v in struct.unpack("<%df" % n, f.read(4 * n)):
            checked += 1
            lo = v if lo is None else min(lo, v)
            hi = v if hi is None else max(hi, v)
            if v > 0 and math.log2(v) % 1 == 0:
                pow2 += 1
        if checked >= cap:
            break
    print("checked %d, pow2 %d, min %s, max %s" % (checked, pow2, lo, hi))
    print("foldable: %s" % (checked > 0 and pow2 == checked))


if __name__ == "__main__":
    main()
