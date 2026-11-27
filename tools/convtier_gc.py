#!/usr/bin/env python3
"""Report (and optionally reclaim) cold-tier blobs that no manifest can reach.

The engine's boot scan builds its index only from `manifest-*.json`, and a blob's token list
lives only in the manifest entry - so a blob with no entry is unreachable at any future revive,
however intact its bytes are. Under the pre-396 binaries a second writer could rename a shard
with its own smaller index over the first writer's blobs, which is how the production tier ended
up at 4,388 indexed against 111,605 files.

Default is dry-run. Deleting is irreversible and only worth doing once the running binary is
396-aware (verified: 396-aware boots keep index == blobs, see devq6 in docs/ac922-revive-decode-fix.md).
"""
import argparse, glob, json, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--apply", action="store_true", help="delete unreachable blobs (default: report only)")
    ap.add_argument("--max-delete", type=int, default=0, help="with --apply, stop after N files (0 = all)")
    a = ap.parse_args()

    indexed = set()
    shards = sorted(glob.glob(os.path.join(a.dir, "manifest-*.json")))
    bad_shards = 0
    for p in shards:
        try:
            m = json.load(open(p, "rb"))
        except Exception:
            bad_shards += 1
            continue
        for e in m.get("blocks", []):
            ch = e.get("chain") or ""
            if len(ch) >= 2:
                indexed.add(ch)
        for e in m.get("states", []):
            ch = e.get("chain") or ""
            if len(ch) >= 2:
                indexed.add(ch)

    orphan_bytes = 0
    orphan_n = 0
    live_n = 0
    victims = []
    for sub in ("blocks", "states"):
        for shard in sorted(os.listdir(os.path.join(a.dir, sub))) if os.path.isdir(os.path.join(a.dir, sub)) else []:
            d = os.path.join(a.dir, sub, shard)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.endswith(".bin"):
                    continue
                ch = fn[:-4]
                fp = os.path.join(d, fn)
                if ch in indexed:
                    live_n += 1
                    continue
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                orphan_n += 1
                orphan_bytes += sz
                if a.apply and (a.max_delete == 0 or len(victims) < a.max_delete):
                    victims.append(fp)

    print("tier=%s shards=%d bad_shards=%d indexed_keys=%d" % (a.dir, len(shards), bad_shards, len(indexed)))
    print("reachable_files=%d unreachable_files=%d unreachable_GiB=%.1f" % (live_n, orphan_n, orphan_bytes / 2**30))
    if not a.apply:
        print("DRY-RUN: nothing deleted. Pass --apply to reclaim.")
        return 0
    if not victims:
        print("APPLY: no victims selected.")
        return 0
    freed = 0
    for fp in victims:
        try:
            freed += os.path.getsize(fp)
            os.remove(fp)
        except OSError:
            pass
    print("APPLY deleted=%d freed_GiB=%.1f (remaining unreachable=%d)" % (len(victims), freed / 2**30, orphan_n - len(victims)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
