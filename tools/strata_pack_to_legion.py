#!/usr/bin/env python3
"""tools/strata_pack_to_legion.py - re-pack Strata's NVFP4 pack as a Legion split GGUF.

Inputs (read-only):
  --dense    qwen-nvfp4-gptq-dense.gguf   the pack's dense side (1079 tensors, q8_0/bf16/f32)
  --experts  pack/experts.bin             per-expert blobs: gate|up|down rows in ggml block_nvfp4 order
  --native   pack/native_experts.txt      the blob table (layer gu_type d_type offset blob_bytes)
  --ple-src  an existing UD split shard   donor of per_layer_token_embd.weight (IQ4_NL, copied as is)
  --out      split prefix                 writes <prefix>-0000X-of-00003.gguf

Shape: shard 1 = metadata only (the dense GGUF's KV section verbatim + split.* keys); shard 2 = the
dense tensors in their original order, then per layer the grouped expert tensors
blk.N.ffn_{gate,up,down}_exps.weight (GGML NVFP4 = 40) and their per-expert scale arrays
blk.N.ffn_*_exps.scale (F32[n_expert], the blob tail {s_gu, s_gu, s_d, 0}); shard 3 = the PLE table.

The blob rows are already the ggml block_nvfp4 layout (4 UE4M3 scale bytes + 32 code bytes per 64
values; nvfp4_codec.to_ggml() matches ggml's dequantize_row_nvfp4 nibble order), and gate|up are the
two halves of one 1280-row quantization (shared global scale s_gu), down the 2560 x 640-value rows
with s_d. Everything moves as whole-row byte copies: nothing here requantizes.
"""
from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys

T_F32, T_F16, T_Q8_0, T_IQ4_NL, T_BF16, T_NVFP4 = 0, 1, 8, 20, 30, 40
PLAIN = {T_F32: 4, T_F16: 2, T_BF16: 2}
BLOCKED = {T_Q8_0: (32, 34), T_IQ4_NL: (32, 18), T_NVFP4: (64, 36)}
SCALAR = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4), 6: ("f", 4),
          7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}
ALIGN = 32


def tbytes(ty: int, dims: list[int]) -> int:
    n = 1
    for x in dims:
        n *= x
    if ty in PLAIN:
        return n * PLAIN[ty]
    blk, sz = BLOCKED[ty]
    assert n % blk == 0, (ty, dims)
    return n // blk * sz


def pad_up(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


class Gguf:
    """One GGUF: header, raw KV bytes (re-emitted verbatim), scalar KV values, tensor infos."""

    def __init__(self, path: str):
        self.f = open(path, "rb")
        assert self.f.read(4) == b"GGUF", "%s: not GGUF" % path
        self.ver = struct.unpack("<I", self.f.read(4))[0]
        self.nt = struct.unpack("<Q", self.f.read(8))[0]
        self.nkv = struct.unpack("<Q", self.f.read(8))[0]
        kv0 = self.f.tell()
        self.kv = {}
        for _ in range(self.nkv):
            k = self._rstr()
            t = struct.unpack("<I", self.f.read(4))[0]
            self.kv[k] = self._rd(t)
        kv_end = self.f.tell()
        self.f.seek(kv0)
        self.kv_bytes = self.f.read(kv_end - kv0)
        self.f.seek(kv_end)
        self.tensors = []
        for _ in range(self.nt):
            name = self._rstr()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = [struct.unpack("<Q", self.f.read(8))[0] for _ in range(nd)]
            ty = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors.append((name, dims, ty, off))
        self.align = self.kv.get("general.alignment", ALIGN)
        self.data_base = pad_up(self.f.tell()) if self.nt else self.f.tell()

    def _rstr(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8")

    def _rd(self, t):
        if t == 8:
            n = struct.unpack("<Q", self.f.read(8))[0]
            b = self.f.read(n)
            try:
                return b.decode("utf-8")
            except UnicodeDecodeError:
                return b
        if t == 9:
            et = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            if et == 8:
                for _ in range(n):
                    m = struct.unpack("<Q", self.f.read(8))[0]
                    self.f.read(m)
            else:
                self.f.read(n * SCALAR[et][1])
            return None
        fmt, sz = SCALAR[t]
        return struct.unpack("<" + fmt, self.f.read(sz))[0]

    def kv_int(self, key) -> int:
        v = self.kv.get(key)
        assert isinstance(v, int), (key, type(v))
        return v


def wstr(o, s: str):
    e = s.encode("utf-8")
    o.write(struct.pack("<Q", len(e)) + e)


def wkv(o, k, fmt_tag, v):
    wstr(o, k)
    if fmt_tag == "u16":
        o.write(struct.pack("<I", 2) + struct.pack("<H", v))
    elif fmt_tag == "i32":
        o.write(struct.pack("<I", 5) + struct.pack("<i", v))
    elif fmt_tag == "str":
        o.write(struct.pack("<I", 8))
        wstr(o, v)


def wtinfo(o, name, dims, ty, off):
    wstr(o, name)
    o.write(struct.pack("<I", len(dims)))
    for x in dims:
        o.write(struct.pack("<Q", x))
    o.write(struct.pack("<I", ty) + struct.pack("<Q", off))


def copy_stream(src, dst, nbytes: int, chunk: int = 64 << 20):
    left = nbytes
    while left:
        b = src.read(min(chunk, left))
        assert len(b) > 0, "short read"
        dst.write(b)
        left -= len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--experts", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--ple-src", required=True)
    ap.add_argument("--out", required=True, help="split prefix without -0000X-of-0000N")
    ap.add_argument("--dry", action="store_true", help="manifest math only")
    a = ap.parse_args()

    d = Gguf(a.dense)
    embd = d.kv_int("qwen4exp.embedding_length")
    ff = d.kv_int("qwen4exp.expert_feed_forward_length")
    nexp = d.kv_int("qwen4exp.expert_count")
    layers = d.kv_int("qwen4exp.block_count")
    assert (embd, ff, nexp, layers) == (2560, 640, 512, 48), (embd, ff, nexp, layers)
    assert d.ver == 3 and len(d.tensors) == 1079, (d.ver, len(d.tensors))

    rows = []
    for line in open(a.native):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) == 5:
            rows.append(tuple(int(x) for x in p))
    assert len(rows) == layers, len(rows)
    blob_bytes = rows[0][4]
    gu_slice = ff * (embd // 64 * 36)
    d_slice = embd * (ff // 64 * 36)
    for (l, gt, dt, off, blob) in rows:
        assert gt == T_NVFP4 and dt == T_NVFP4, "layer %d not NVFP4/NVFP4" % l
        assert blob == 2 * gu_slice + d_slice + 16 and off == l * nexp * blob, "layer %d geometry" % l

    u = Gguf(a.ple_src)
    ple = [t for t in u.tensors if t[0] == "per_layer_token_embd.weight"]
    assert len(ple) == 1 and ple[0][2] == T_IQ4_NL, ple
    ple_name, ple_dims, ple_ty, ple_off = ple[0]
    ple_size = tbytes(ple_ty, ple_dims)

    m2 = []
    off = 0
    for (name, dims, ty, _o) in d.tensors:
        m2.append((name, dims, ty, off))
        off = pad_up(off + tbytes(ty, dims))
    eo = {}
    for l in range(layers):
        for (proj, dims, wsz) in (("ffn_gate_exps", [embd, ff, nexp], gu_slice),
                                  ("ffn_up_exps", [embd, ff, nexp], gu_slice),
                                  ("ffn_down_exps", [ff, embd, nexp], d_slice)):
            key = "%d.%s.weight" % (l, proj)
            m2.append((key.split(".", 1)[1], dims, T_NVFP4, off))
            m2[-1] = ("blk." + key, dims, T_NVFP4, off)
            eo[key] = off
            assert tbytes(T_NVFP4, dims) == nexp * wsz
            off = pad_up(off + nexp * wsz)
        for proj in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            key = "%d.%s.scale" % (l, proj)
            m2.append(("blk." + key, [nexp], T_F32, off))
            eo[key] = off
            off = pad_up(off + nexp * 4)
    ntot = len(m2) + 1
    print("MANIFEST dense=%d expert_tensors=%d shard2=%d tensors/%d B shard3=%d B total=%d"
          % (len(d.tensors), len(m2) - len(d.tensors), len(m2), off, ple_size, ntot))
    if a.dry:
        return

    p1, p2, p3 = ["%s-%05d-of-00003.gguf" % (a.out, i) for i in (1, 2, 3)]
    for p in (p1, p2, p3):
        assert not os.path.exists(p), "refusing to overwrite %s" % p

    def header(o, nt, no):
        o.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", nt) + struct.pack("<Q", d.nkv + 4))
        o.write(d.kv_bytes)
        wkv(o, "split.no", "u16", no)
        wkv(o, "split.tensors.count", "i32", ntot)
        wkv(o, "split.count", "u16", 3)
        wkv(o, "general.quantized_by", "str", "strata_pack_to_legion.py (nvfp4-gptq expert rows verbatim)")

    with open(p1, "wb") as o:
        header(o, 0, 0)

    with open(p2, "wb") as o:
        header(o, len(m2), 1)
        for (name, dims, ty, o2) in m2:
            wtinfo(o, name, dims, ty, o2)
        o.write(b"\0" * (pad_up(o.tell()) - o.tell()))
        base = o.tell()
        di = {t[0]: t for t in d.tensors}
        pos = base
        for (name, dims, ty, o2) in m2[:len(d.tensors)]:
            assert base + o2 == pos, (name, o2)
            src = di[name]
            d.f.seek(d.data_base + src[3])
            copy_stream(d.f, o, tbytes(ty, dims))
            pos = o.tell()
            o.write(b"\0" * (pad_up(pos + 0) - pos) if pos % ALIGN else b"")
            pos = o.tell()
        with open(a.experts, "rb") as ef:
            mm = mmap.mmap(ef.fileno(), 0, access=mmap.ACCESS_READ)
            for l in range(layers):
                lb = l * nexp * blob_bytes
                for (proj, slice_off, wsz) in (("ffn_gate_exps", 0, gu_slice),
                                               ("ffn_up_exps", gu_slice, gu_slice),
                                               ("ffn_down_exps", 2 * gu_slice, d_slice)):
                    o.seek(base + eo["%d.%s.weight" % (l, proj)])
                    for e in range(nexp):
                        o.write(mm[lb + e * blob_bytes + slice_off: lb + e * blob_bytes + slice_off + wsz])
                for (proj, word) in (("ffn_gate_exps", 0), ("ffn_up_exps", 1), ("ffn_down_exps", 2)):
                    o.seek(base + eo["%d.%s.scale" % (l, proj)])
                    buf = bytearray(nexp * 4)
                    for e in range(nexp):
                        t = lb + e * blob_bytes + blob_bytes - 16 + word * 4
                        buf[e * 4:e * 4 + 4] = mm[t:t + 4]
                    o.write(buf)
                print("LAYER %d packed" % l)
                sys.stdout.flush()
            mm.close()
        o.flush()
        assert o.tell() >= base + off - 1024

    with open(p3, "wb") as o:
        header(o, 1, 2)
        wtinfo(o, ple_name, ple_dims, ple_ty, 0)
        o.write(b"\0" * (pad_up(o.tell()) - o.tell()))
        u.f.seek(u.data_base + ple_off)
        copy_stream(u.f, o, ple_size)

    for p in (p1, p2, p3):
        print("WROTE %s %d B" % (os.path.basename(p), os.path.getsize(p)))
    print("CONV-MARK DONE tensors=%d" % ntot)


if __name__ == "__main__":
    main()
