#!/usr/bin/env python3
# Repair state-sidecar keys written before the Sha256::update tail fix: a sidecar whose
# length is not a multiple of 64 got its manifest key from the buggy tail path, so the
# fixed engine fails verification and aborts every revive on that conversation. Recompute
# each state key from the file's bytes and rewrite the shard atomically. Run with the
# lane's engine STOPPED (a live engine would flush its stale map back over the fix).
# Blocks in this geometry are 64-multiple sized and unaffected; --deep re-hashes them too.
import json, glob, hashlib, os, sys

d = sys.argv[1]
deep = "--deep" in sys.argv[2:]
fixed_s = fixed_b = checked = block_tail = 0
for shard in sorted(glob.glob(os.path.join(d, "manifest-*.json"))):
    m = json.load(open(shard))
    changed = False
    for e in m.get("states", []):
        sp = os.path.join(d, "states", e["chain"][:2], e["chain"] + ".bin")
        if not os.path.exists(sp):
            continue
        b = open(sp, "rb").read()
        h = hashlib.sha256(b).hexdigest()
        checked += 1
        if h != e["key"]:
            e["key"] = h
            fixed_s += 1
            changed = True
        if len(b) != e["bytes"]:
            e["bytes"] = len(b)
            changed = True
    for e in m.get("blocks", []):
        if e["bytes"] % 64:
            block_tail += 1
            if deep:
                bp = os.path.join(d, "blocks", e["chain"][:2], e["chain"] + ".bin")
                if os.path.exists(bp):
                    h = hashlib.sha256(open(bp, "rb").read()).hexdigest()
                    if h != e["key"]:
                        e["key"] = h
                        fixed_b += 1
                        changed = True
    if changed:
        tmp = shard + ".rekey"
        with open(tmp, "w") as f:
            json.dump(m, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, shard)
        dirfd = os.open(os.path.dirname(shard), os.O_RDONLY)
        os.fsync(dirfd)
        os.close(dirfd)
print("rekey states=%d/%d blocks=%d block_entries_with_tail=%d" % (fixed_s, checked, fixed_b, block_tail))
