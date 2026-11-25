# AC922 cold tier L1: session KV + state on NVMe (oMLX semantics, Legion shape)

Motivation (measured 2026-11-25, 4-card S7, mb8192): a cold 4x123K prefill storm costs
**317.5 s**; the same sessions resumed from RAM cost **16.9 s**. Every restart or
eviction pays the full prefill again. A disk tier turns the worst case into a read:
at the AC922's measured 2.2 GB/s write / 2.4 GB/s O_DIRECT read, one 123K session is
~2.3 GiB of q8 KV (19,200 B/token) plus a few hundred MiB of recurrent state, i.e.
**single-digit seconds**.

## What already exists (the whole reason L1 is a pure addition)

- `src/kv/radix.rs` — `RadixCache`: nodes are `(parent, tokens, blocks)`;
  `insert(pool, tokens, blocks)` links ready-made block ids; `evict*` drops edges.
- `src/engine/saved_states.rs` — `SavedStates`: recurrent state per prefix boundary
  (`--prefix-cache-interval 8192` in production), host-backed already
  (`saved_states_host`, `exec.saved_states_to_host`).
- `src/engine/lm/host_tier.rs` — pinned host tier, block bytes read by the GPU in place
  over NVLink2; a hydrated edge written into host-tier blocks is warm, no promotion step.

## Design (aligned to oMLX main @ 6ff0b269, 2026-11-25 pull)

Keying is **content-addressed per KV block** (oMLX `PagedSSDBlockMetadata` shape), not
per session: blob key = `sha256(block bytes)`, and the header metadata carries the
compatibility signature (model digest, dims, quant, cache class, format version) —
a block whose signature mismatches never matches, and identical blocks shared by
several sessions are stored once (cross-session dedup is free at this granularity).

Four write disciplines oMLX learned the hard way (their fixes: #5543, #5544, #5497,
#5255) go into the spec up front:
1. superseded async writes must never replace a newer file (write to `pending/`, then
   link-or-refuse if the final path exists; a content-hash key is write-once);
2. an inline write-through that lands makes the entry clean (no double queue);
3. the size cap is **global across models** (`--conv-disk-gib` bounds the whole dir,
   LRU trim over blocks, `touch()` O(1) via an intrusive list, not per-model ledgers);
4. restart scan must re-register existing blobs and keep chains reachable in the hot
   index, or a long shared prefix half-restores and the cold storm returns.

Recurrent state is NOT sliceable, so it follows oMLX's `GDNCheckpointIndex` shape: a
sidecar keyed by (chain hash, token position) written at every
`--prefix-cache-interval` boundary; a restore needs the state blob for the deepest
chain boundary the block set can satisfy, else the restore stops earlier.

## Design (original sketch, superseded by the above where they differ)

Restore = allocate pool blocks, write the file bytes into them, `RadixCache::insert`,
`SavedStates::store` for the boundary state. From there the normal warm path runs.

Dump = on shutdown and on radix/save eviction that would free a block, copy the block
bytes into a pending buffer and flush in the background; bytes are immutable q8 blocks,
so content addressing is exact: blob key = `sha256(edge tokens || edge bytes)`.

Files under `--conv-disk-dir`:

- `index.json` — atomic-rewritten on changes: per session, model digest + dims +
  format/quant version, then the edge list (raw token ids, blob keys, state key).
  Key hygiene (oMLX discipline): a mismatched digest/dims/format never matches.
- `blobs/<sha256>.bin` — one edge: `block_bytes * n_blocks` of KV, verbatim.
- `states/<sha256>.bin` — the recurrent state after the edge, verbatim.
- `pending/` — write buffers; a flusher thread drains it (never on a request path).

Eviction: LRU over entries by `last_used`, `--conv-disk-ttl-min` (0: keep), and a cap
`--conv-disk-gib` enforced before writes; the index alone is negligible.

## Flags (serve)

- `--conv-disk <dir>` (off unless set), `--conv-disk-gib <n>` (default 100),
  `--conv-disk-ttl-min <n>` (default 0), `--conv-disk-restore` (on: hydrate at load;
  off: index only, dump-only mode for bisection).

## Verification gates

1. `revive-det`: same prompt -> 60K+ session -> stop -> start -> follow-up; the
   restored `prompt_eval` must show `cached_tokens >= 95%` of the prefix and the answer
   sha must equal the RAM-warm sha (bytes are verbatim, so this is a bitwise expectation).
2. det six-sha vs production baseline on the boot with the flag on (must be equal).
3. Steady state unchanged: throne 8K C4 and 123K C4 anchors within noise (writes are
   shutdown/eviction-time, decode windows untouched).
4. Space gate: refuse writes past the cap; report bytes/parked-session at /health stats.

Gate 1 note from implementation: `usage.cached_tokens` does not report hydrated or
even RAM-radix hits on this engine when the harness was written (observed 0 on every
repeat). It does report them now: a 261,917-token repeat after a restart logged
`cached_tokens=261888`. Keep all three signals anyway - the `conv tier: hydrated
blocks=N tokens=M` INFO line, the wall, and the sha equality; N and M must both be > 0
and the answer sha must equal the cold-computed sha.

## Cut plan

- C1 (this commit): flags + config + module + boot scan + health line. No-op otherwise.
- C2: dump at shutdown (radix walk + saved_states), index write, `pending/` flusher.
- C3: hydrate at load before the first prompt (pool alloc + insert + store), guarded by
  `--conv-disk-restore`; revive-det harness (`eval/revive.py`).
- C4: eviction-time spill (mid-run churn, not just restarts) + TTL/LRU trim.

## Landed status (2026-11-25)

- **C1/C2/C3 are implemented and verified; the cut differs from the sketch above where
  they diverge**: the dump is not shutdown-only. Every completed block and every saved
  state is queued behind the step (`ack`-gated, deduped, writer thread, O_DIRECT), so
  crashes and evictions lose at most the in-flight tail. Hydrate runs in `place()`:
  past what memory resumes, walk the on-disk chain run, cut back to the furthest
  boundary with a byte-length-matched saved state, verify every blob's sha256 at read
  (a mismatch or a trim race aborts the extension and falls back to plain memory
  resume; nothing is poisoned). FORMAT is now 3 (the state region carries the slot's
  token history window; FORMAT-2 fixtures are foreign and ignored by scan).
- **2-card dev verification (port 8097, Q4_K_XL + MTP, q8 KV)**: 48K-char probe ->
  A(12,844 tok)+B(13,477 tok) cold 17.6/18.2 s; dir holds 430 blocks @1,228,800 B +
  9 states (~113 MiB each), 430/430 sha_verified; stop -> start (memory empty) ->
  repeat A: `conv tier: hydrated blocks=200 tokens=12800` (full prefix, the tail 44
  tokens recomputed), wall 6.5 s (-63% vs cold), answer sha identical to warm
  (9081a2af...). A's sha also reproduced bit-identically across two independent boots
  before the restart, which is the det gate for the dump path itself.
- **Production (4 cards, port 8090, 2026-11-25 14:51 CST)**: the CT binary plus
  `--mmproj /data/work/vision-test/mmproj-BF16.gguf --conv-disk
  /data/cache/conv-tier-prod --conv-disk-gib 200` are live via lane.env
  (`LEGION_BIN`, `LEGION_EXTRA`; env backed up before the change). Boot scan
  `0 blocks 0 states ... restore on` (fresh dir; per-geometry digests keep 2-card
  fixtures out), vision tower on the first model GPU, text smoke OK, image probe
  `1,005.50` twice with identical sha, second pass `cached=384`.
- **Vision adoption evidence**: the GPU tower (sm_70 path) and the CPU oracle tower
  (`LEGION_DEBUG_VISION_CPU=1`) return the same invoice amount with 13-char content;
  greedy reasoning text drifts between arms as ggml's "close, not bitwise" contract
  predicts, while GPU-arm repeats are bit-identical (3x dev, 2x prod) and image
  encode is ~2.7 s vs ~32 s on CPU. Per-image cost is a one-time prefill, and the
  image prefix itself is radix-cached (`cached_tokens` on the repeat).
- **4-card revive verification (2026-11-25 18:20-19:05, `eval/ct-revive.py`, four
  distinct 261,917-token sessions, q8 KV, `--conv-disk-gib 200`)**:
  - cold compute, sequential: 133.2-231.1 s per session (pp 1,179-2,073 tok/s), answers
    and spans recorded as the reference.
  - restart (RAM empty), then the same four sessions one at a time: 4/4 answer sha equal
    to the cold reference - revive-det passes on the production geometry. Engine-side
    `prefill_seconds` 0.3-0.5 s and 5.4-6.2 s of decode at 35.6-40.4 tok/s, but 51-53 s
    of end-to-end span, because each request waits ~46 s in `queue_ms`.
  - that ~46 s is the per-blob sha256 verification of the chain, not the read: 5.05 GiB
    per session at ~112 MB/s single-threaded. Dropping the page cache beforehand changed
    nothing measurable (54.6 s vs 52.7 s, `prefill_seconds` 0.361 both times), so the NVMe
    is not the wall. Parallel or cheaper verification is the next cut and would take a
    full-window revive from ~46 s to single-digit seconds.
  - capacity price per parked full-window session: ~5.05 GiB of blocks plus ~3.6 GiB of
    state checkpoints, so the 200 GiB cap holds about 23 of them (/data free went
    620 to 576 GiB over four sessions).
  - C=4 revival burst: 267.6 s wall, 3.3 tok/s aggregate, per stream 1.8-40.5 tok/s.
    Repeating that burst with everything already RAM-resident reproduced it (264.5 s,
    3.3 tok/s, same collapse), and only one of four streams kept single-stream speed -
    the tier is not the cause, this is the decode-window composition issue tracked in
    [ac922-perf-seconddev.md](ac922-perf-seconddev.md). Greedy answers are not repeatable
    between two identical C=4 bursts (3 of 4 shas moved), while every C=1 answer was
    bit-identical to its cold reference: cite the C=1 numbers as the correctness gate.
- **Open**: C4 eviction-time spill beyond the behind-the-step queue (TTL/LRU trim is
  implemented; the churn-vs-cap behaviour under production mix is not yet measured);
  hydrated bytes/parked-session stats at `/health`; batched reader I/O, now with a price
  tag (verification is the ~112 MB/s floor above, reads are not); the ~45 s wait a
  full-window session pays before its first window even when nothing is being hydrated.
