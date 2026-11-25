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

## Design

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

## Cut plan

- C1 (this commit): flags + config + module + boot scan + health line. No-op otherwise.
- C2: dump at shutdown (radix walk + saved_states), index write, `pending/` flusher.
- C3: hydrate at load before the first prompt (pool alloc + insert + store), guarded by
  `--conv-disk-restore`; revive-det harness (`eval/revive.py`).
- C4: eviction-time spill (mid-run churn, not just restarts) + TTL/LRU trim.
