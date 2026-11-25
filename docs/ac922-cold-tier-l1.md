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
  `--conv-disk-verify-threads <n>` (default 8): workers that read and sha256-verify blobs
  during hydrate; 1 reproduces the serial wall of the first production measurement.

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
  - that ~46 s is the per-blob sha256 verification of the chain, not the read: about
    4.1 GiB hydrated per session (3,254 blocks of 1,228,800 B plus one 112.6 MiB state)
    in ~46 s, i.e. ~90 MB/s single-threaded. Dropping the page cache beforehand changed
    nothing measurable (54.6 s vs 52.7 s, `prefill_seconds` 0.361 both times), so the NVMe
    is not the wall. Parallel or cheaper verification is the next cut and would take a
    full-window revive from ~46 s to single-digit seconds.
  - measured footprint per parked full-window session: 13,014 new block files over four
    sessions (3,254 each, 3.72 GiB) plus 128 state files (32 each at
    `--prefix-cache-interval 8192`, 3.53 GiB) = ~7.2 GiB, so the 200 GiB cap holds roughly
    27 full-window sessions. Two notes: the boot INFO line's GiB figure counts blocks only,
    and 3,254 is fewer than the naive 261,917/64 = 4,092 - unexplained, worth a look
    because it could be dedupe across head-salted chains or an unfilled tail.
  - C=4 revival burst: 267.6 s wall, 3.3 tok/s aggregate, per stream 1.8-40.5 tok/s.
    Repeating that burst with everything already RAM-resident reproduced it (264.5 s,
    3.3 tok/s, same collapse), and only one of four streams kept single-stream speed -
    the tier is not the cause, this is the decode-window composition issue tracked in
    [ac922-perf-seconddev.md](ac922-perf-seconddev.md). Greedy answers are not repeatable
    between two identical C=4 bursts (3 of 4 shas moved), while every C=1 answer was
    bit-identical to its cold reference: cite the C=1 numbers as the correctness gate.
  - **C5 (same evening, `8a32935`, the cut the paragraph above asked for)**: hydrate now
    verifies in windows of at most 512 blobs with `--conv-disk-verify-threads` (default 8)
    workers - the reads and the hashing move off the engine thread, uploads stay strictly
    in order, and the verdict of any blob is unchanged (a `None` still aborts the extension
    at the same place). Measured on the production shape, restart between rounds, same four
    sessions C=1: spans 51.2-53.2 s became 14.9-16.9 s (round wall 209.6 s -> 64.9/65.2 s),
    engine `queue_ms` ~46 s -> 9.1-10.4 s, `vs.cold` bit-identical [true x 4] in both rounds,
    one of them with the page cache dropped first. The state sidecar read stays single-thread
    (~110 MB, ~1 s, not the wall). Full-window revive is in single-digit-to-low-teens seconds.
  - C=4 burst on cold RAM after C5: wall 267.6 s -> 120.1 s, because the four hydrations,
    still serialized on the engine thread, each cost ~10 s instead of ~46 s. Aggregate stayed
    window-composition-bound (7.3 tok/s; one stream at 40.6 tg, the rest starved) and C=4
    shas stayed unreproducible between rounds - unchanged findings, now with the tier wait
    taken out of them. Cross-conversation hydrate overlap would be the next cut, not C5's.
  - this burst's per-session hydrated runs came out at the full 4,092 blocks (261,888
    tokens), not the 3,254 the C=1 rounds showed earlier - the open question flips shape:
    identical fixtures, different run lengths, so the short run was likely a walk stopping
    early (a state-boundary or trim-race condition), not missing blobs. Worth chasing
    against the saved boot logs before quoting revive spans as the tier's floor.
  - largely resolved the same evening: the C=1 rounds' own `prefill_seconds` 0.33-0.52 s
    already prove they served the full prefix from the tier (a walk stopping at 3,254 would
    have prefilled the 53K-token gap cold), so 3,254 was a file-count taken mid async-dump,
    never a walk stop. The true parked cost is 4,092 blocks (5.03 GiB) + 32 states
    (3.53 GiB) = ~8.6 GiB per full-window session, ~23 of them inside the 200 GiB cap;
    the earlier 3.72 GiB / 27-session numbers in this doc's footprint notes are superseded.
  - **the delta-encoding study (offline, `/tmp/delta_probe.py`, 231 consecutive-boundary
    pairs from the live tier): storage-only state deltas are DEAD on this model.** Adjacent
    boundary states share 11.4% of bytes (the GDN cumulative state updates densely every
    window), zeroing the equal bytes and lzma-ing the rest saves 1.6% - and full-file lzma
    on these blobs saves only ~6%, so the 3.53 GiB state column is incompressible, not
    unencoded. oMLX's `deepseek_v41_delta` wins where that state is sparse-delta-friendly;
    here it does not apply. The surviving capacity lever is POLICY, not format: hydrate only
    ever needs the deepest boundary at or below the walk end, so dump-time downsampling of
    older state snapshots (keep the last few, thin the ancient ones) cuts the parked cost
    toward the ~5.5 GiB the blocks alone cost, with zero numeric risk (a missing state
    already falls back to a shallower boundary) and no format change. Tracked as C6.
- **Open**: C4 eviction-time spill beyond the behind-the-step queue (TTL/LRU trim is
  implemented; the churn-vs-cap behaviour under production mix is not yet measured);
  batched reader I/O, now with a price
  tag (verification is the ~112 MB/s floor above, reads are not); the ~45 s wait a
  full-window session pays before its first window even when nothing is being hydrated.
  Post-C5 re-arms this last one: a RAM-resident full-window follow-up should be timed again
  - if its first-window wait is still ~45 s with no hydration involved, it is a separate
  scheduler issue and the C5 thread count is not its lever.

## Stats and C6 (2026-11-25 late)

- **Hydrate accounting is live** (`6e6f9f1`, fixed by `6b794ae`): `/health` carries a
  `conv_tier` object (walks, chains_present, blobs_ok/mismatch, states_ok/bad, alloc/upload
  fails, revive_aborts, tokens/bytes hydrated), the oMLX `cache/stats.py` counterpart. The
  first deploy kept saying null because every route reaches the engine through
  `MeteredEngine`, which swallowed the new trait method with its default - the wrapper now
  forwards it. Verified live: a three-revive sequence moved walks 3->7, tokens_hydrated
  +769,792 (3 x 261,888 minus tails), blobs_ok +12,028 (3 x 4,009), no failures.
- **The state column is incompressible, proven** (231 adjacent boundary-state pairs:
  11.4% equal bytes, diff-zeroing + lzma saves 1.6%, whole-file lzma ~6%). GDN recurrent
  state is dense-updated; storage-level deltas and generic compression are both dead. The
  surviving lever was always policy.
- **C6 landed** (`9db3a2d`): spaced checkpoints are thinned on the way to the disk
  (`--conv-disk-state-stride`, default 8) - every anchor, the last four spaced points of a
  prompt, one per stride older. The revive walk already cut back to the deepest boundary
  holding a state, so thinning costs at most one stride of re-prefill and never a miss;
  no format change. 2-card dev: a 131K session parked 6 states instead of 16 (depths
  65536/98304/106496/114688/122880/130944) and 2,065 blocks unchanged; after a restart the
  same request revived in 9.1 s (cold 135.5), hydrated exactly the deepest kept boundary
  (+130,944 tokens, +2,046 blobs), answer sha bit-identical, zero mismatches. Parked cost
  per full-window conversation: 8.6 GiB -> ~6.5 GiB (~30 sessions inside the 200 GiB cap).
  Production boot 22:5x CST carries it; gates (det x3, image, undeclared) all pass.

## D1: hydrate off the engine thread (2026-11-25 late night)

- The C6 numbers above are one conversation at a time. A C=4 burst paid an engine-thread
  wall instead: pre-D1, `place()` read and sha-verified the disk run inside the step, so
  four parked full-window resumes serialized ~10 s of disk+hashing each, head-blocking
  every other stream's decode (§366: the burst's 4x10 s stall, agg ~4 while one stream ran).
- D1 (`3f0999b`): a revival job gets a `HydrateTask` - a producer thread walks the chain
  run (the same `plan()` the inline path used), reads and verifies blobs in bounded windows
  (cap 2 windows ~ 1.2 GiB RAM per task), and holds the result. Each `place()` attempt
  uploads what is ready, up to `--conv-disk-hydrate-budget` blocks (default 256 ~ 0.3 GiB
  of engine-thread memcpy per window), parks the pool refs it took, and returns Wait; the
  rest arrives across following windows. Disk and hashing therefore overlap decode instead
  of stopping it, and different conversations' hydrations overlap each other. `0` = whole
  plan in one attempt (the pre-D1 one-shot shape); `--conv-disk-hydrate-tasks` (default 4)
  caps concurrent tasks - over-cap jobs place on memory alone, as before.
- Dead jobs (client gone, plan aborted mid-drain) are swept at the step head: their parked
  blocks are decref'd, their producers aborted. Bitwise: the bytes and their sha checks are
  unchanged from C6, only the thread doing them moved - verified against the same tier
  with the cold-baseline sha comparison in `eval/ct-revive.py` (bit-identical, see below).

### D1 verification (2026-11-26): the idle-sleep bug, then the A/B

- Boot hang (budget=0 twice, default once): D1 made `place()` park a job that advances at
  the step head, but the `decoding/prefilling both empty` branch of the worker loop still
  assumed the only cause was KV shortage - "which an idle engine never is" - and blocked
  the worker in `rx.recv()`. The boot warmup's first revive has no next request to wake
  it; the producer delivers its windows and exits, and nobody takes them. Signature: the
  log's last line is `conv tier: live`, 182 threads all futex, no producer. `tasks=0`
  booted because no task is ever born, which pinned the regression to the D1 path.
- Fix (`25a8678`): `Waiting::awaits_hydrate` + a read-only `WaitQueue::any`; the idle
  branch sleeps 5 ms on a hydrate in flight instead of sleeping on the channel.
- A/B (dev 2-card :8092, the 4×131K seeded tier, par burst after each reboot):
  pre-D1 C6 `wall 102.6 s` (spans 102.6/93.7/82.9/82.9) → D1 default (budget 256,
  tasks 4) `wall 66.0 s` (−35.7%, spans 54.3/59.9/66.0/47.2) → D1 budget=0 (one-shot
  upload) `wall 79.0 s` (−23%). The one-shot gain is the producers running ahead across
  jobs; the per-step budget adds ~16% more of streaming uploads on top.
- The burst's outputs are not bit-identical between the arms, nor between two par rounds
  on the same binary (`exact:false` throughout): the revive-vs-cold drift predates D1
  (§369's cold/cache asymmetry) and is not D1's doing; the adoption gate is the 4-card
  production `gate.sh` (det×3, IMG, undecl) instead, run on the same tier.
- Adopted: `lane.env LEGION_BIN=/data/work/legion/bin-D1fix`, prod boot 01:44
  (`legion-4-20261126-014405.log`), tier lines and warmup hydrations normal.

## L2: the oMLX convergence pass (2026-11-26, `aa1b25d`)

The audit against oMLX main (`12a577be`) found the cold tier already carries most of the
mechanisms: chain-keyed block identity, boot scan revival, the global capped LRU (#5497),
link-or-refuse staging (#5543), the signature gate, TTL, state sidecars (C6 stride),
write-through at prompt completion, and the stats surface. Four real gaps remained, all
closed in L2. None of them touches the numerics: key hygiene, write-side flow control,
crash consistency, and one hash-library bug.

### Mechanism matrix (oMLX → Legion)

| oMLX mechanism                    | pre-L2 Legion                                  | post-L2                                   |
| --------------------------------- | ---------------------------------------------- | ----------------------------------------- |
| chain-hash block keys             | had it                                         | unchanged                                 |
| boot scan revive                  | had it                                         | + `N v3 migrated` in the live line        |
| global cap LRU eviction           | had it                                         | unchanged                                 |
| bounded pending writer (#5543)    | unbounded queue                                | `--conv-disk-pending-gib` (default 64)    |
| signature gate                    | model/digest/quant/dims only                   | + pack content fingerprint (FORMAT=4)     |
| key hygiene on pack swap          | stale tier trusted                             | v3 re-signed once, v4 foreign on mismatch |
| fsync on publish                  | rename without sync                            | `sync_file` before, `sync_dir` after      |
| interrupted-write residue         | `pending/` orphans left forever                | boot clears them, logged                  |
| pooling-delta dumps               | considered, rejected (shape mismatch)          | unchanged                                 |

### Pack fingerprint (FORMAT 3 → 4)

`Signature.pack` is `pack_fingerprint(model_path)`: over the GGUF and its `-NNNNN-of-NNNNN`
shard siblings, per file name+size+mtime+first 64 KiB+last 64 KiB, sha256-aggregated. The
gate it closes: previously a re-quantized or re-exported pack at the same path kept serving
old blobs whose KV bytes no longer match the new weights - silently wrong continuations.
Migration is read-side: a v3 manifest's `pack`-less entries that otherwise agree with the
live signature enter the index re-signed (durable at the next shard flush), so the 45.7 GiB
production tier survives the upgrade at zero recompute. The legacy trust is scoped to the
v3 *format*: a v4 entry with an empty pack is corruption or a downgrade attempt, counted
foreign. A v4 scan against the same model but different pack bytes is foreign too - the
fingerprint test pins head/tail sensitivity.

### Write-side backpressure

`Inner.pending` counts staged dump bytes; over `pending_gib` a revive order steps aside
(ids acked, chain not occupied, `dumps_backpressured++` in `/health`) rather than piling
an unbounded writer queue during a big eviction storm. `0` restores the unbounded shape.

### The Sha256 tail bug (found here, largest quality win of the pass)

`Sha256::update` reset `buf_len` to the remainder instead of adding to it, so a stream
ending with a partly-filled block after a prior non-empty buffer dropped the tail bytes -
`sha256(b"abc")` was wrong. It is unreachable for the product paths: chain hashes run on
256-byte token blocks and blob keys on word-aligned byte counts, both multiples of the
64-byte block, so the buffer is never partly filled at finish. Verified: identical digests
before/after the fix across those producers - existing tier data stays valid, nothing
needs re-signing. It only becomes live the moment a new caller hashes arbitrary lengths,
which the added `sha256_vectors` test (FIPS vectors incl. the "abc" and multi-block cases)
now guarantees.

### L2 verification sequence (server)

1. `cargo test --release --features cuda --lib conv_tier` - all green (6 tests incl. the
   two new: `sha256_vectors`, `pack_fingerprint_tracks_content`; `scan_migrates_v3_and_gates_by_pack`
   pins the re-sign + v4-foreign semantics).
2. dev :8092 reboot: `conv tier: live ... (0 missing, 0 foreign, 18461 v3 migrated)`
   expected on the dev2 tier, plus the pending-clear line; one par revive round, wall in
   the 66 s band, errs 0.
3. prod 4-card cutover to the new binary, `gate.sh 4` (det×3, acc, IMG, undecl) before
   handback.
