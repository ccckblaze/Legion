#!/usr/bin/env python3
"""Reclaim conv-tier blobs that no manifest shard attests.

The engine's writer rewrites a manifest shard from the entries this boot indexes, so an entry
belonging to another model or an older KV geometry could be dropped from the index while its
blob stayed on the fly (the AC922 incident: 264 GiB of files against a 128 GiB cap, /data at
99%). The writer-side fix keeps those entries named; this tool cleans up what earlier builds
already stranded, and is the pre-flight check for a tier directory.

Read-only by default. Unlinking is age-guarded so a dump that is mid-flight (the writer flushes
shards about every 2 s) is never mistaken for litter.
"""
import argparse
import glob
import json
import os
import time


def index(dirpath):
    """(kind, chain) pairs any manifest shard of this tier attests."""
    named = set()
    for path in glob.glob(os.path.join(dirpath, "manifest-*.json")):
        try:
            with open(path) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        for kind in ("blocks", "states"):
            for e in m.get(kind, []):
                named.add((kind, e.get("chain", "")))
    return named


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--apply", action="store_true", help="unlink the litter (default: report)")
    ap.add_argument("--min-age-min", type=float, default=30.0,
                    help="skip files newer than this: a fresh blob may simply be un-flushed")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    named = index(a.dir)
    cut = time.time() - a.min_age_min * 60
    keep_n = keep_b = 0          # attested by a manifest shard
    loose_n = loose_b = 0        # stranded: a file no shard names
    gone_n = gone_b = 0          # unlinked by this run (--apply only)
    young = 0
    empty = 0
    for kind in ("blocks", "states"):
        for root, _dirs, files in os.walk(os.path.join(a.dir, kind)):
            for name in files:
                path = os.path.join(root, name)
                try:
                    info = os.stat(path)
                except OSError:
                    continue
                if info.st_size == 0:
                    empty += 1
                    if a.apply:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                    continue
                if (kind, name.rsplit(".", 1)[0]) in named:
                    keep_n += 1
                    keep_b += info.st_size
                    continue
                if info.st_mtime > cut:
                    young += 1
                    continue
                loose_n += 1
                loose_b += info.st_size
                if a.apply:
                    try:
                        os.remove(path)
                        gone_n += 1
                        gone_b += info.st_size
                    except OSError:
                        pass
    gib = float(1 << 30)
    verb = f"deleted={gone_n} {gone_b / gib:.1f}G" if a.apply else f"stranded={loose_n}"
    print(
        f"CT-SWEEP {'APPLY' if a.apply else 'CHECK'} {a.dir}: "
        f"attested={keep_n} {keep_b / gib:.1f}G "
        f"{verb} "
        f"young-skipped={young} empty-dropped={empty}"
    )
    if not a.quiet and loose_n and not a.apply:
        print("CT-SWEEP-NOTE stranded blobs are unreachable through the index: --apply to reclaim")


if __name__ == "__main__":
    main()
