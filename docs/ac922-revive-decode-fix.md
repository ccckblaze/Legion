# AC922 · Revived-session decode collapse: mechanism candidates and fix shapes (2026-11-27)

Read-only design pass on the §390 lesion (runbook `AC922-bringup-runbook.md` §390/§391.5):
full-window sessions restored from the conv disk decode at 0.2-0.8 tok/s and degrade monotonically
within one boot, with `cached_tokens` at the whole prompt, while the same boot's in-pool reburst
decodes at ~20. This document pins the code anchors, ranks the mechanism candidates, and states
the fix shapes so the CV-wave data can select one immediately.

## Anchors (bin ~2130b13, file positions from the `coldtier` source tree)

- `src/engine/lm.rs:1692` — `conv_hydrate_budget: 256`, `conv_hydrate_tasks: 4` (defaults).
- `src/engine/lm.rs:7569-7576` — the hydrate spawn gate: fires when a resumed run crosses a
  block boundary and `parked.len() < hydrate_tasks`; the plan covers every remaining chain.
- `src/engine/lm.rs:7578-7600` — the placement cascade under pool pressure: `want = fresh + kv_keep + ext`
  (ext = the **whole remaining plan**, not this step's slice) → radix evict → warm evict →
  saved-tail drop → and when still short, **`hydrate_abort` ("place on memory alone")**.
- `src/engine/lm.rs:7601-7617` — `Placement::Wait` exists but only AFTER the abort branch: a job
  whose full plan cannot fit gives its disk part back and resumes as host-streaming decode.
- `src/engine/lm/conv_tier.rs:1345-1353` — `take(budget)`: 0 means the whole plan in one attempt
  (the documented `--conv-disk-hydrate-budget 0` shape).
- Boot sizing on this machine (4-card lane): `kv_pool_tokens=1044800`, host tier `786432`
  (`memory plan part="context"`), i.e. **four 262K windows ≈ 1048576 tokens > pool** — a
  four-way full-window revive is structurally past the device pool even before `kv_keep`.
  The NVFP4 lane measures `kv_pool_tokens=1057088` - headroom over 4x262K is **0.8%**: any
  parked hydrate plus `kv_keep` tips `want > available`, so the abort branch fires with even one
  revive overlapping the others. The UD lane (pool 1044800) is past capacity outright, yet its
  f262c4 survived (8 rows, agg 25.5) - consistent with abort-to-memory being *survivable* on
  the shallower geometry and *catastrophic* on the NV lane's deep-window expert streaming;
  the CV wave decides whether that split holds.

## Ranked candidates (fitted to the observed shape)

> **FIELD REVISION 2026-11-27 05:00 (supersedes the ranking below where they conflict).**
> The NV0/NV1 `f262c4` arms (bin 2130b13, two separate boots) produced a deterministic, byte-identical
> collapse table that falsifies candidate 1 *as written*:
>
> - Eight rows (4 prompts × 2 rounds), every row `prompt_tokens=261940 cached_tokens=261888`
>   (full prefix hit, `prefill_seconds≈0.7`), every row `completion_tokens=205`. Per-row
>   `decode_tok_s` on completion: **35.2, 36.3, 13.9, 3.5, 0.53, 0.26, (row7/8 still in flight at
>   timeout)**. Both boots: identical sequence. Same shape on the UD lane (S0) finished all 8 rows
>   in wall 64.2 s, tg_med 17.1, errs=0.
> - During the collapse, conv-tier counters are *healthy*: `revive_aborts=0`, `alloc_fails=0`,
>   `blobs_ok=41903` frozen (all hydration completed), `pending_bytes=0`,
>   `tokens_hydrated=2616256` ≈ 10 full rows. **The abort branch at `7578-7600` never fires.**
> - Engine-side counters during the collapse (`/metrics`, two 40 s windows): `kv_preemptions=0`,
>   `waiting=0`, `blocks_used≈10504/16517`. Differential sampling of `model_kind_bytes` shows
>   **all step traffic attributed to `phase="prefill"` — `phase="decode"` deltas are exactly zero**
>   while rows still emit tokens at 0.26-0.53 tok/s (ITL mean 0.271 s, tail multi-second).
> - No engine log lines exist between request boundaries during the collapse — there is **no
>   step-level window telemetry** in this build, which is what the P1-A arm (decode timing points,
>   the Strata `STRATA_DECODE_TIMING` analogue) is for.
>
> Revised mechanism (working): hydrated rows never enter a *decode* window. They are carried as
> **riders on prompt-phase steps** — each token costs a host-streamed forward over the parked
> context rather than a window decode step — which reproduces both the multi-second ITL and the
> all-prefill-phase byte attribution. The same demotion explains the intra-window gradient in
> round 1 (rows admitted 7/7/7/14 s after boot decode at 36 → 36 → 14 → 3.5 as pool occupancy
> climbs). Expert-cache `lend` to KV (`lm.rs` ~7903) remains the suspected trigger; `hit_rate`
> recovered to 0.979 during the collapse, so expert misses are *not* the rate limiter.
>
> Consequences for this document: candidate 1 is downgraded (the *abort* path is exonerated; the
> *placement/window-eligibility* path upstream of it is the new lesion). CV arm `NR`
> (`--conv-disk-no-restore`) is still the discriminator: if NR rows revive as cold prefill and
> then decode at 13-36, the lesion is specific to hydrated-row window eligibility; if they also
> collapse, it is pool-depth window formation itself. `B0/B2048/T8` predictions written under
> candidate 1 should be re-read through this lens.
>
> **Cross-lane contrast (06:10, waves v4, same 4-card shape, same bench, three arms each):**
> the UD lane (bin L2 @6101045: S0/H2/K2) completes `f262c4` at wall 64.2/70.0/53.8 s, errs=0;
> the NVFP4 lane (bin-nvscale @2130b13: NV0 mb4096, NV1 mb8192, NV3 micro4096) times out
> (>3000 s) in all three shapes. The lesion is therefore **NVFP4-lane-specific** — pool headroom
> is exonerated as the trigger (the UD pool 1044800 is *past* 4×262K and survives; the NV pool
> 1057088 has 0.8% margin and dies). Confounded axes: binary commit, pack (NVFP4 scale-remap +
> `LEGION_NVFP4_CACHE=1`), expert-cache sizing (25.7 vs 35.4 GiB) and KV_VRAM (274432 vs lane
> default). The dev-lane `host_time` capture (2 cards, same bin-nvscale @2130b13, C3@262K) is the
> phase-attribution discriminator: if revived rows' `exec_us` per step explodes while `hit_rate`
> stays high, the slow thing is the NVFP4 host-stream path itself, not the scheduler.

1. **~~Plan-fit starvation → hydrate_abort~~ (falsified by `revive_aborts=0`) → replaced by:
   hydrated-row rider demotion** — see field revision above. Original argument retained for
   provenance: the first deep revive(s) park most of
   the pool; every later revive's `want` (whole-plan ext) exceeds `available` even after evicting
   everything evictable, so it **aborts hydration and decodes from memory alone** — which on
   sm70 + host-tier KV is exactly 0.2-0.8 tok/s. Monotonic degradation = parked refs and
   radix state accumulate across the boot, so successive revives abort earlier. A boot with only
   one or two revives keeps good tg (the observed fast rows), matching the coexistence of
   tg 0.25 and tg 28 in one window.
2. Budget pacing (`256` blocks/step saturates PCIe per step) — contributes to the first ~16
   steps only; cannot explain minutes-long windows or monotonicity. Weak on its own.
3. Pure bandwidth ceiling — ruled out by §388 (roofline 8-32%).

## CV-wave mapping (arms → predictions under candidate 1)

- `B0` (`--conv-disk-hydrate-budget 0`): the first revive takes the whole plan in one attempt
  and wins; later revives still hit plan-fit starvation → B0 should fix the FIRST row's tg but
  not the tail of the run. If B0 fixes everything instead, candidate 2 dominates.
- `B2048`: same shape, slower ramp; expected ~ CTL.
- `T8` (`--conv-disk-hydrate-tasks 8`): more concurrent plans parked → **worse** under
  candidate 1 (more pool pressure), better under candidate 2.
- `NR` (`--conv-disk-no-restore`): no revive at all → tg back to the cold-prefill shape
  (fast decode, slow wall). Positive control for "revive path is the lesion".
- `CTL`: reproduces §390.

> **Dev-lane double capture (2-card, bin 2130b13, `host_time` on, boot 060732):**
> (1) C2/C3 revived rows are *healthy* (tg 10.3-19.7) even though 3×262K **oversubscribes** the
> ~541K-token device pool (rows run host-tier streamed) — the streaming decode path itself is
> fine; (2) a C4 run collapsed to tg 1.26/0.43/0.26 — but that batch contained two genuinely cold
> prompts, and the collapse window's `host_time` shows `steps=2/exec_us≈4.1-4.3e6/host_us≈230`
> with GPU util 0%/84%: those are **prefill-storm steps starving co-resident decode** (the same
> shape as the production "decode+PP concurrent, both crawl" complaint). The NV 4-card arm's
> f262c4 had *no* cold rows (all `cached=261888`), so its collapse has a second trigger;
> the leading theory is now **near-full resident-pool thrash** (4×262K = 99.2% of the 1057088
> pool: `back_slot`'s trim/demote cascade ping-pongs between co-resident revived rows).
> Discriminator armed (devf4c): re-run C4 after all 8 ordinals are primed — pool oversubscribed
> 194% (pure host-stream, no near-fit demote race). Healthy ⇒ NV lesion = resident-edge thrash
> (fixes F1/F3 pool hygiene); collapse ⇒ revived-row window ineligibility per se (fix F1b).
>
> **Rider-mechanism closure (dev host_time + per-row byte math):** collapsed rows carry
> `decode_model_gb/completion ≈ 0.7-7 GB/token` vs healthy `12-14 GB/token` — each token does
> *less* model work while taking ~100× longer. That is the signature of rows being **sampled once
> per foreign step**, i.e. `lm.rs:9044` `riders = big && n_stages>1 && kv_ring().is_none()`: while
> a multi-second prompt-phase step runs (dev: two genuinely cold 262K prompts; NV 4-card: the
> revived rows' prompt-phase resume replay, evidenced by all-collapse-bytes-on-`phase=prefill`
> with `phase=decode` Δ=0), every in-window decode row rides it at one sample per step
> (1/4.1 s ≈ 0.24 ≈ the measured 0.26). The collapse is not slow decode — it is **decode
> suppressed to rider rate by prompt-phase work sharing the loop**. devf4c (all-cached C4)
> discriminates the NV trigger: healthy ⇒ replay length was the trigger (cold storm co-residency
> is the whole story for dev); still collapsing ⇒ parked-row resume keeps riders latched, and the
> fix is F1b — never let a prompt-phase tail (cold or resumed) put live decode rows into rider
> mode; gate it behind `Placement::Wait` or finish the replay before windowing. Independent of the
> NV verdict, the dev capture stands on its own for the production complaint: **co-resident cold
> prefill costs co-running decode rows ~75× (tg 19.7 → 0.26)** — rider latching, not bandwidth.

## Fix shapes (all keep numeric outputs bit-identical: they move *when* work happens, not *what*)

> Post-revision priority (field data above): **F1b first** — the lesion is window eligibility of
> deep rows, not hydration aborts. F1/F2/F3 remain queue hygiene that F1b likely subsumes.

- **F1b · no-rider demotion for parked rows (preferred)**: a row whose KV lives on the host tier
  must not be sampled via prompt-phase forwards. Gate it into the decode-window admission set
  only when its window-resident share clears the `can` test (anchor: the `lend`/eligibility path
  near `lm.rs:7903`); otherwise it stays `Placement::Wait` in the queue and the caller sees queue
  latency instead of a 3-4 s ITL. Pair with a queue fairness bound so a parked row cannot starve
  behind an in-pool stream. Bit-parity holds: sampling points and token order are unchanged.

  **P0 code audit (2026-11-27 05:1x, coldtier ea17f0d)**: the `/metrics` phase split is per-step
  bookkeeping (`src/server/metrics/bandwidth.rs:229` maps step records to `st.prefill`/`st.decode`),
  so "zero `phase=decode` deltas while rows emit tokens" means the in-flight rows are literally
  *driven by prompt steps*, not decode steps. The two code shapes that can do that:
  (a) the pipeline-riders path (`lm.rs:9041-9137`): `big = lead.prompt.len() - lead.n_past` over
  threshold, multi-stage, `kv_ring().is_none()` — a revived row re-entering as a "lead" runs its
  262K as micro-chunks and shares graphs with decode rows;
  (b) the slot-backing churn (`back_slot` → `plan.trim/demote` → `exec.trim_kv_slot`,
  `lm.rs:7944-7975` + `free_ring_and_slot`'s `lends()` return path): a running row's KV slot room
  is trimmed to seat another row's hydrate, and if the trim lands between its decode steps its
  progress forces prompt-phase re-entry. The observed `blocks_used≈10504/16517` plateau with two
  rows in flight is consistent with (b) ping-ponging against parked 4092-block hydrates.
  Distinguisher without new code: the CV `NR` arm (no parked hydrate ⇒ no trim competition) plus
  a `/metrics` delta check for `phase=decode` being *nonzero* under NR at the same wall clock.
  **Correction (same audit pass)**: the step-timing brick already exists — `HostClock`
  (`src/engine/lm/host_time.rs`) times Plan/Pack/Exec/Sample/Device/Finish/Loop per step and logs
  µs-per-step window sums at debug level (`RUST_LOG=legion::engine::lm::host_time=debug`, 5 s
  windows; also per-step on the dashboard hub). No patch needed for attribution; the dev-lane
  job is to reproduce the lesion with this logger on and read which phase owns the 3-4 s ITL.

- **F1 · plan-fit admission gate (preferred if candidate 1 holds)**: before spawning a hydrate,
  require `resumed/KV_BLOCK + chains.len() + kv_keep <= free - reserved`; otherwise the job
  waits in the queue (the loop already retries via `awaits_hydrate`). One revive at a time past a
  window-depth line, e.g. only admit when no other `parked` plan is larger than half the pool.
  Cost: the last queued revive pays its prefill wait once; every admitted revive decodes fast.
- **F2 · window-adaptive budget**: `budget = clamp(available_blocks / (parked_plans + 1), 64, 1024)`
  — spreads the pool among concurrent plans so none aborts. Weaker alone (does not stop the
  whole-plan `want` overshoot).
- **F3 · no-abort placement**: make `want` overshoot return `Placement::Wait` **before** the
  abort branch when the pool is merely tight, and abort only when the tier is genuinely short
  (trim race). Minimal diff; risk: a plan that can never fit parks the queue — needs a
  starvation timeout (re-add abort after N seconds, log a line).

Acceptance for whichever shape is adopted: same CV harness — `ct-revive.py revive` tg_med ≥ 15
at 262K×4 with disk hits, zero `retire this hydrate`-class lines in the boot window, cold-entry
wall not worse by >3%, det-suite identical (the fix is on placement order, so bit-parity holds
against CTL by construction).

## §394 · The root cause: a hydrate plan double-counted a prefix the RAM match advanced

The pure-revive discriminator (dev lane, 2 cards, GPU 2/5, `bin-nvscale`, full-window C4 with
every order already in the cold tier) did not measure a slow decode: it **killed the worker**.

```
thread 'legion-qwen35' panicked at src/engine/lm.rs:8725:38:
range start index 417536 out of range for slice of length 261947
```

`8725` is `tokens: &s.prompt[s.n_past..end]`, so a prefilling sequence stood at
`n_past = 417536` on a 261947-token prompt. Both numbers are block multiples of `KV_BLOCK = 64`:
417536 = 6524 blocks, 261888 = 4092 blocks, and the RAM-side resume had advanced to
2432 blocks = 155648 tokens = 19 × `--prefix-cache-interval 8192`.

    417536 = (2432 + 4092) × 64        # resumed = (resumed/KV_BLOCK + ext) × KV_BLOCK

That is the completion branch of `place()` verbatim. `HydratePlan` stores an **absolute** run
(`chains = chains[start..j]`, i.e. blocks `[start, start + chains.len())`), but the update treats
`ext` as a delta on top of the *current* `resumed`, which every attempt re-derives from the radix
and warm match. When a saved state landed while the producer was reading the blobs, the plan's
first 2432 blocks were already in the table, and they got appended a second time.

Consequences, in increasing severity:

- Overflow past the prompt → worker panic → every in-flight request 500s ("engine stream ended
  without finishing", then "the model worker stopped"). Reproduced twice on the dev lane.
- Without the overflow the same overlap is **silent**: `n_past` no longer matches the table's
  block count, so the row re-enters prompt phase to repair and samples once per prompt step -
  exactly the 0.26-0.8 tok/s collapse shape measured on the 4-card NV lane and in dev C4, and the
  "one fast lane, the rest watching" GPU-utilisation signature (one stage carrying the replay).
- The disk state is copied over whatever the RAM match had reached, so an advanced resume can be
  rewound to a shallower recurrent state.

Two earlier readings of this file need correcting:

- `/health states_ok` is **not** zero on the dev lane: 14 with `states_bad = 0`. States are found
  and consumed; the `states_ok=0` note came from reading a lane before any revive. The tier is
  doing its dump/index job - the defect is in how the engine *applies* the result.
- Candidate 1 (`hydrate_abort` from pool pressure) stays refuted, and for the same reason:
  `revive_aborts = 0`, `alloc_fails = 0`.

The fix (`c709085`, coldtier) makes the run's overlap explicit in three places: `ext` counts only
blocks memory does not resume yet; at completion the duplicated blocks go back to the pool and
`resumed` becomes the plan's absolute end; a state deeper than the plan's end is not rewound. A
plan that would leave a *hole* (memory resumed less than the plan assumed) is handed back and
re-planned, counted as the new `states_stale`. Finally a resume at or past the prompt's last whole
block truncates the table and warns instead of panicking, so no future miscount can take the lane
down.

Side finding, separate bug (not fixed here): the kernel log on this box carries four
`conv-tier[...]: unhandled signal 11` entries - the cold tier's verify/hydrate producer threads
segfault out of process. The engine survives them (a failed read just aborts that plan), but it is
a memory-safety bug in the tier's own threads and deserves its own pass with `RUST_BACKTRACE` and
`valgrind`/ASan on a dev boot.

## §395 · The shutdown race behind `conv-tier: unhandled signal 11` (fixed, `b4a26a4`)

The side finding in §394 was not a verify/hydrate producer: it is the tier's own **writer**
thread, and it is a lifetime bug on the stop path. Six entries in the kernel log, and every one of
them sits 3-5 s after an engine process stopped (crash 08:25:26 / log ends 08:25:21; 08:28:48 /
08:28:43; 05:57:49 / 05:57:44; 02:29:15 / boot boundary 02:29:12; 02:05:06 / 02:05:00; and 23:08:47
the previous evening). A racing dump, not a corrupt file: it fires at teardown, on the way out.

Four facts make it inevitable:

- `Inner` owns the `Sender<Job>`, and the writer's closure owns `Arc<Inner>`: a reference cycle,
  so the channel can never disconnect and `Disconnected` can never end the loop.
- Nothing ever joins the thread. `ConvTier::start` documented the opposite ("the thread exits when
  the last `ConvTier` handle drops"), and `gather`'s comment repeated it ("pool tensors outlive it").
- `Worker` declares `exec` (which owns the KV pools and the pinned host arena) **before** `conv`, so
  dropping the worker frees the arena first.
- A queued `BlockJob` is a list of raw addresses into that arena, dereferenced whenever the writer
  gets to it.

So the engine thread exits, frees the arena, and a dump still sitting in the queue is gathered from
unmapped pages: SIGSEGV on POWER9 (`nip` always at libc's `memcpy`+0xeb7c, `code 2` = access error
past a valid mapping's last page). The corruption path is the worse half: when the read *does* land
in memory the next tenant already owns, the blob is written and its `sha256` recorded over bytes
that were never the block, so a later hydrate serves it as valid - the hash attests the garbage.

The fix is a real shutdown protocol, not a guard:

- `Job::Quit` ends the writer after landing what it already holds and flushing the shards.
- `ConvTier::close()` sets `closed`, sends `Quit`, and joins. `queue_blocks` hands its pool
  references straight back once `closed` is set (nothing may queue behind the join), and
  `queue_state` drops.
- `impl Drop for Worker` calls `close()` in its body, which runs before field drops, so the writer is
  gone while `exec` still owns the arena.

Durability improves as a side effect: dumps queued by the last requests of a session now reach the
disk before the process exits instead of being lost with it. What it does not fix: a process killed
by `SIGKILL` still drops the queue (that is what the `pending/` + rename order is for), and the
`states_stale`/re-plan work in §394 is unaffected.

Validation (dev lane, two boots of the same tree, only the binary differing): stop the engine
gracefully with blocks in the queue, count new `unhandled signal 11` entries in the 30 s window
after the stop. `bin-revive` (§394 fix, this bug present) is the control; `bin-ctquit` is the arm.

## Wave CV / NV accounting (the measurements behind §394)

CV wave - 4-card eval lane, one fresh tier directory and two boots per arm, `REV_N=2`,
131072-token conversations, 64 generated tokens; the cold pass runs the two sessions
sequentially, the revive pass runs them concurrently.

| arm (`--conv-disk-*`)   | revive tg | revive span | cold span | speedup | `sha_all_match` |
| ----------------------- | --------- | ----------- | --------- | ------- | --------------- |
| CTL (defaults)          | 33.8/22.4 | 16.5/10.2   | 119.8/186.6 | 7.3-18× | true            |
| `hydrate-budget 0`      | 22.6/34.0 | 10.0/15.8   | 119.8/185.1 | 11.7-12× | true            |
| `hydrate-budget 2048`   | 18.8/34.0 | 10.6/15.9   | 119.8/186.1 | 11.3-17× | true            |
| `hydrate-tasks 8`       | (same cell as above; repeated boot) | | | | true |
| `--conv-disk-no-restore` | 0.80/23.7 | 159.8/160.3 | 119.7/185.8 | 0.75-1.2× | **false**       |

Two conclusions:

- The hydrate-upload throttling hypothesis is **dead**. Budget 0 (one shot), the default budget,
  2048, and `tasks 8` all revive in the same 10-16 s band with identical outputs. Whatever made
  the 00:39 boot's revives collapse to 0.21-0.82 tg was not how the blocks were uploaded; it was
  §394's miscount of where the resumed prefix ends.
- The `no-restore` arm is the control, not a defect: with the tier read disabled a revive is a full
  re-prefill, so its span equals the cold pass and its tg is the cold-storm number. Its
  `sha_match = false` is the deep-window greedy fork between a sequential cold pass and a
  concurrent revive (the same shape sensitivity the DRIFT probes already track), not cold-tier
  corruption - the four arms that *did* read the tier all match bit for bit.

NV wave correction (changes a verdict): `NV0`/`NV1`/`NV3` `f262c4` were logged as PARSE-FAIL and
carried as "this shape does not work". The output files are **0 bytes**, the harness `timeout` is
1500 s, and all three boots show `panic=0` / `out of range=0`. So the failure is not the §394
overflow branch and not a harness parsing bug: at 262144 x 4 the un-revived rows drop to the
silent branch (one sample per step, tg 0.26-0.8), no request ever completes inside the window, and
the client is killed before it writes a line. Those three cells must be **re-measured on the fixed
binary** (the paired A/B's `FIX` arm is exactly `262144 4`) before any conclusion about batch shape
on 4 cards is drawn from them.

## §396 · Ten poisoned states in the production tier: two engines, one directory (fixed)

Symptom in the log, repeated on every deep revive:

```
conv tier: revive aborted, state sidecar verification failed for b73140c2a3a1...
```

Audit of `/data/cache/conv-tier-prod` (read-only, file hashes and sizes only): 35 state entries
indexed, **25 verify, 10 do not** - same 118,038,536-byte size, different `sha256`. Every other
tier on the box is clean (`nv4` 205/205, `cvCTL` 21/21, `devq` 5/5), so this is not the writer
thread from §395 and not a hashing bug: it is a second writer.

Mechanism:

- Blobs are write-once: `link_once` refuses to replace an existing file (#5543), which is right -
  keys are content addresses and a superseded writer must not rewrite history.
- Manifest shards are **not**: `flush` rebuilds a shard from the live index and renames it over.
- `legion-lane.sh restart` is `stop; sleep 5; start`, and `stop` is graceful: the old engine keeps
  draining its last request (a 185 s deep-window one was measured in §394.2) while the new boot has
  already scanned the directory and is queueing into the same files.
- The old boot then flushes its shard, containing a `key` for bytes *it* wrote - into a file the new
  boot had already linked with different bytes. Index says one thing, disk holds another, forever.

Consequence: those chains cost a full 112 MiB sidecar read on every revive and then abort to the
blocks-only path, so the recurrent state is recomputed. Safe, slow, and self-perpetuating: nothing in
the tier ever forgets or repairs a poisoned entry.

Fix, three parts:

- `take_writer`: an exclusive non-blocking `flock` on `<dir>/writer.lock`, held for the tier's life,
  retried for up to 30 s (the normal holder is the boot being restarted). A boot that cannot claim
  the directory runs with the **cold tier off** and says so, instead of writing a second index over
  one directory.
- `link_once` now returns `Linked::{Ours, Already}`. On `Already` the dump keeps the index honest: the
  surviving entry's `last_used_ms` moves, an orphan blob is adopted by hashing the file it actually
  holds, and the loser's key/size are never written into the manifest. Counted as `index_conflicts`.
- `repair_state` / `repair_block` on the read path's verification failure: drop the entry, free the
  chain so the next completion re-dumps it honestly, delete the blob, and queue the shard for flush
  through `extra_dirty` (only the writer thread renames a shard). Without this the ten poisoned
  entries would keep their verdict until the directory is wiped by hand.

Follow-ups filed, not done: `close()` (the §395 join) waits for the writer, so a boot holding the
full `--conv-disk-pending-gib` 64 GiB queue could take longer to stop than `legion-lane.sh stop`'s
60 s ceiling and get `SIGKILL`ed - the tier then loses exactly those dumps. Either bound the close
with a deadline that abandons the queue, or raise the stop ceiling. And the same double-writer window
applies to any two lanes pointed at one `--conv-disk` directory: the lock makes it loud, but lane
templates should keep one directory per lane (they mostly do).

## 397 - Verification of S395/S396 and the adoption decision (2026-11-27, 4-card + 2-card lanes)

### S395 (shutdown race) - confirmed, and the trigger is narrower than the fix assumes

The dmesg counter (`unhandled signal 11` with `conv-tier` in the fault line) moved exactly once, on
the arm that had work in flight at shutdown, and only on the pre-fix binary:

| arm | binary | in flight at stop | SEGV delta | blobs landed after stop |
|---|---|---|---|---|
| devq3 Q1 | `bin-revive` (S394 only) | channel jobs, `pending/` empty | **1** | **+2** |
| devq3 Q2 | `bin-ctlock` (S394+395+396) | same shape | **0** | **+302** |
| devq4 R1 | `bin-revive` | nothing (burst drained first) | 0 | 0 |
| devq4 R2 | `bin-ctlock` | nothing | 0 | 0 |

Two conclusions the earlier draft did not have:

- **`pending/` is not the indicator.** Q1 crashed with `pending=0`. A `BlockJob` sitting in the mpsc
  channel already carries raw pool addresses, so the disk directory can be empty while the race is
  live. The observable that matters is *work that lands after the stop* - Q1 lost it (2 of ~18),
  Q2 kept it (302).
- **The crash is conditional, not a background rate.** devq4 is the control: when the writer has
  caught up before shutdown, both binaries are clean. So the 6 historical faults were all
  "shutdown with a live queue", and the fix removes that class.

### S396 (two writers, one directory) - confirmed by test and by a real SIGKILL

- `cargo test --release --features cuda --lib conv_tier`: **7 passed, 0 failed**. The new
  `a_late_writer_keeps_the_index_on_disk` asserts the loser of a `link_once` race cannot rename its
  own key over a blob it did not write, and that the read path retires an entry that stopped
  matching its bytes.
  (First run failed for a test-harness reason, not a code reason: `ConvTier::start` deliberately does
  not create its directory layout - production builds the config through the opener that does - so a
  hand-rolled `ConvDiskCfg` made every dump fail in the silent `Err` arm. The test now creates
  `blocks/states/pending` itself.)
- `devq6`: boot a `bin-ctlock` tier, burst 8 x ~12k, then `kill -9` the engine mid-load. Independent
  re-scan (not the engine's own report): `states_indexed 7 ok 7 sha_mismatch 0 missing 0`,
  `blocks_scanned 221 ok 221 sha_mismatch 0 size_mismatch 0 missing 0`. A second boot on the same
  directory agrees (`221 blocks 7 states, 0 missing, 0 foreign`) and the revive completes. A killed
  writer can lose unindexed work; it cannot leave an index that misdescribes its bytes.
- Production baseline before adoption: `conv-tier-prod` = 111,600 blocks / 255 states / 212 GiB with
  **exactly 10 poisoned states**, all `want_bytes == file_bytes` and a differing sha, each with the
  manifest shard mtime *later* than the blob mtime - the signature in S396. Post-adoption gate:
  the boot log must show `states_repaired` reaching 10 and a re-scan must return `total_bad 0`.
  Do not hand-clean: the read path is the thing under test.

### S394 on the production shape - neutral, so it is safe to ship with the two tier fixes

`armud UD2` (`bin-revive`, `lane.env`, same cells, same night as `UD1` = `bin-ctl394`):

| cell | UD1 | UD2 | delta |
|---|---|---|---|
| 8192 x 4 conc - agg / tg_med | 82.6 / 36.97 | 82.5 / 36.75 | -0.1% / -0.6% |
| 122880 x 4 conc - wall / agg / tg_med | 276.1 / 7.4 / 29.92 | 275.4 / 7.4 / 29.83 | -0.3% / 0 / -0.3% |
| 262144 x 4 conc - wall / agg / tg_med | 652.4 / 2.5 / 22.09 | 645.4 / 2.5 / 22.24 | -1.1% / 0 / +0.7% |
| det | same, panic 0 | same, panic 0 | - |

These cells do not exercise the S390 pathology they were meant to cure (a *follow-up after eviction*
degrading to 0.2-0.8 tok/s), so read them as "no regression", not as "no benefit". The benefit is
already measured separately in the CV wave.

### Not a S394 result: the NV-scale lane's `262144 x 4` cell

`arm CTL`/`arm FIX` source `lane-nvfp4-4.env` (`MAX_BATCH=4096`, `KV_VRAM=274432`,
`EXPERT_CACHE_GIB=6.5`, tier-nv4), which is **not** the production shape. There, the same binary that
completes the cell in 652 s on the production lane produces **zero records** in 1500 s
(`reburst-abCTL-f.jsonl` and `reburst-abFIX-f.jsonl` do not exist at all - not a parse failure).
Both arms failing identically means the cell is a lane/VRAM-shape limit (same family as retry3's
`out of VRAM for a step's compute buffer ... need_mib=3035` -> `cudaMemcpy: invalid argument`),
not a property of the code under test. Rule: diff the two arms' lane envs before quoting any A/B.

### Adopted

`lane.env` -> `LEGION_BIN=/data/work/legion/bin-ctlock` (sha256 `24f72fdccb49`), dated `# ADOPTED`
header, backup kept. `lane-nvfp4-4.env` intentionally left on `bin-nvscale` (reproducibility baseline
for the NV-scale arms; its tier scanned 205/205 clean).

Filed for the next iteration, not folded into this one:

- `close()` joins the writer unboundedly, while `legion-lane.sh stop()` waits 60 s and then
  `kill -9`. Measured drain rate is >= 5 blocks/s, and a worst-case production queue is 8 x 4096 =
  32,768 blocks (~109 min), so a restart right after a heavy burst loses ~99.7% of the queue silently.
  The structural fix is to make `BlockJob` own an `Arc` of the KV arena instead of a raw address -
  then dropping the writer mid-flight is merely lost work, and shutdown never has to block.
- `--conv-disk-gib` counts blocks only, so the production tier reached 212 GiB against a 128 GiB
  nominal cap. Patch exists and `--check`s clean; it is not in the adopted binary.

## 400 - Measured blast radius of the two-writer bug, and the tool that counts it

`tools/convtier_gc.py` (dry-run by default) walks the tier's blobs and diffs them against every
chain named by any `manifest-*.json`. A blob with no manifest entry is unreachable forever, because
the token list that produced the chain is stored only in the entry - the filename is the chain hash,
not the tokens, so the engine cannot re-derive what a stray file was for.

Measured on the live production tier, one round after adopting `bin-ctlock`:

```
tier=/data/cache/conv-tier-prod shards=256 bad_shards=0 indexed_keys=4388
reachable_files=4436 unreachable_files=107932 unreachable_GiB=201.3
```

- 111,605 block files + 255 state files, 211.7 GiB total, but only 4,388 chains survive in the
  index. The boot log agrees with the tool: `conv tier: live 4388 blocks 47 states 5.0 GiB
  (0 missing, 0 foreign)` - so the loss is in the manifests, not in the blobs.
- 111,471 of the 111,605 blobs carry an mtime inside the last 24 h, which rules out TTL expiry as
  the cause. The writes were current; the index that described them was overwritten.
- **201.3 GiB is the cost of §396 on this machine**, and it is why full-window revive looks bimodal:
  a conversation whose chain survived comes back at page-cache speed, one whose chain was lost is a
  2.1M-token cold prefill. The same `262144 x 4` cell measured 18.9 s (chain present) and 652 s
  (chain absent) within ninety minutes of each other.
- `devq6` is the prevention proof: on the fixed binary a boot that queued 221 blocks ended with
  `indexed == blobs == 221`, and the following boot scanned the same numbers. The pre-fix binaries
  are what produce orphans.
- Nothing was deleted. With 516 G free there is no urgency, and the orphan set is the only physical
  evidence of the overwrite, so it stays until a deliberate `--apply` is scheduled.

Operational follow-ups, in order, none of them in the adopted binary:

1. Enforce the cap honestly: `--conv-disk-gib` counts blocks only, so 128 GiB nominal allowed
   211.7 GiB actual. The patch exists and `--check`s clean.
2. Reclaim the 201.3 GiB with `tools/convtier_gc.py --apply` once (1) is in, so the cap and the
   footprint agree afterwards.
3. Make the count observable at boot, not just in the archive: the scan already knows
   `blocks/bytes` but nothing tells the operator that blobs outnumber entries by 24x.

---

## Wave-M verdict (2026-11-27, runbook §403) — MTP stays off production

Six alternating boots on the 4-card lane, regime sentinel flat (rb-c1 wall 66-72 s), so every arm pair is
comparable. Against the warm control M0b:

| metric | ctl M0b | MTP k2/G1 | verdict |
|---|---|---|---|
| 8K C4 throne agg | 62.7 | 60.4 | neutral |
| 8K C16 throne agg | 167.6 | 253.2 | **+51%** |
| 123K C1 tg | 47.4 | 55.2 | **+16.4%** |
| 123K C4 agg | 116.0 | 104.6 | −9.8% |
| 262K C4 revive agg | 87.2 | 26.0 | **−70%** |
| 262K C4 revive tg | 25.5 | 6.9 | **−73%** |

The quality gate is the veto: `det_same` reproduced **true** twice for k2/min-group-1, and **false** for both
`--spec-k 3` and `--pipeline-min-group 2` - run-to-run nondeterminism inside a boot, which `acc 6/6` does not
excuse. So the only deterministic MTP shape is exactly the shape that collapses the deep-window revive path,
and the shapes that survive the revive path fail determinism. Production keeps `ud=0`.

Reopen conditions, in order: (1) locate the spec x min-group / k3 nondeterministic race; (2) fix the
revive -> lend -> rider collapse the deep rows hit once `1+k` verify rows land on restored state; (3) price
the one-time tier invalidation when a boot changes the layer count (48 -> 49), which is what rewrote the
night's waves into two shape families (§401) and is now prevented by the harness fix below.

## Harness root causes fixed (runbook §404)

- `lane19.sh` sourced the production lane env and therefore **shared `conv-tier-prod` with production**.
  The first 49-layer (MTP) boot overwrites the tier index, which is the mechanism behind §401's
  S4 -> K2 family flip, reproduced again in Wave-M (3,699 foreign entries left behind).
  Test lanes now use `conv-tier-lane`.
- `lane19.sh` stop could kill production through the shared `/tmp/legion-lane-4.pid` - it did, once, for
  ~2 minutes at 12:0x. Lanes now carry `LEGION_PIDF=/tmp/legion-lane-lane19.pid`.
- `bin-cap` (adopted binary + cap-counts-states patch) passed on the 2-card lane: 2,048 MiB nominal vs
  2,057 MiB actual (blocks 367 + states 1,689). The 128 -> 211.7 GiB cap miss is root-caused and closed
  for the next stable candidate. `tools/convtier_gc.py --apply` remains scheduled, not run.

## The 83 -> 55 "short-window regression" was a cross-caliber artifact (runbook 426)

A short-window single-stream median was reported as having fallen from 83 to 55 tok/s between the
11-27 and 11-28 production boots, which would have been the largest decode regression on the box.
It is not a regression. Recomputing the `<1K` band per boot, across six engine revisions:

| boot (CST) | cards | n | `<1K` median | version |
|---|---|---|---|---|
| 11-27 22:15 | 4 | 18 | 52.9 | `92730e0` |
| 11-27 23:10 | 4 | 18 | 53.2 | `92730e0` |
| 11-27 23:41 | 4 | 48 | 54.5 | `92730e0` |
| 11-28 08:41 | 4 | 18 | 52.6 | `a4c6c9e` |
| 11-28 11:32 | 4 | 59 | **54.2** | `a4c6c9e` + resident window budget |

52.6-54.5 over eight boots, and the current boot sits in the middle of that range. `92730e0..a4c6c9e`
is eight commits, four of them docs; the four code changes are the queue-based `--resident-window-budget`,
the conv-tier byte accounting, the starve guard default, and one re-queue on a lost `copy_states` race -
none of them touches the decode inner loop, and a single short row never reaches a row budget.

The 83 came from a different caliber: it is an MTP-on number (the upstream 79-101 headline is
`--spec mtp`; our reproduction on this box was 89.6), production runs `ud=0`. Same machine, same
fixture, draft on versus draft off. Rule added: **a regression claim must first match the spec state
and the card count of both samples**, otherwise a cross-boot comparison manufactures a defect. The
per-boot archived `legion starting version` line is what made this a five-minute close.

## MTP's veto mechanism is already removed upstream, so the ticket is re-priced by window band

Reopen condition (3) above - the one-time tier invalidation when a boot moves 48 layers to 49 - is
exactly what upstream `5ebb354` fixed, and it is an ancestor of our HEAD. The MTP head's layer is no
longer a tier layer: its experts load whole into the pinned host arena (~1.7 GiB for the Q4_K_M head,
2.79 GB for the Q8_0 file we hold), the tier's heat/prefill/drop/refill only touch the model's own
layers, and the memory plan reports `mtp_head_experts_mib`. Without `--spec mtp` nothing changes.

Upstream's own measurement, though, gives the cost model: on a 4090 with a bounded host tier,
`--spec mtp --spec-k 1` is *slower* - chats 40-42 against 47-51 tok/s, 8K prefill 1,931 against 2,097,
32K 2,012 against 2,949 - because "a verify step costs twice a one-token step when the experts come
from RAM and disk". Our box is that regime: the VRAM expert cache is small and experts stream from host
RAM. That is the same shape as our S1 arm (C1 +13.6%, C2 -24%), so MTP is not a defect and not a global
win: it is a **window-band sign flip**, and since the head layer is a boot-time geometry it cannot be
gated per request.

Open question, priced by a two-boot band sweep on the new binary (`wb_m.sh`, cards 2/5): where does the
sign flip? If short bands gain and deep bands lose, MTP becomes an operational choice per boot role
(agent lane on, long-context lane off); if it is negative out past 131K the Wave-M veto stands. Note the
counted fixture ends with an increment-by-ones tail, which inflates draft acceptance - read the gain as
an upper bound.

## The flip is measured, and the gate that should catch it already exists (runbook 427)

`bin-e1u` (`6ebfe1b`), cards 2/5, one boot per arm, counted fixture, C=1 single stream except the
last row. `--spec mtp` at its default `k=2` against the same boot shape without it:

| window | M0 (no MTP) | M1 (MTP) | delta |
|---|---|---|---|
| 1024 | 31.1 | 33.3 | **+7.1%** |
| 8192 | 30.7 | 32.3 | **+5.2%** |
| 32768 | 27.6 | 26.8 | -2.8% |
| 131072 | 24.3 | 23.8 | -2.4% |
| 261872, C=2 (tg) | 13.56 | 8.81 | **-35.0%** |
| 261872, C=2 (agg rows/s) | 24.08 | 16.68 | **-30.7%** |

So MTP is neither dead nor a win: it is positive up to ~8K and negative past ~32K, and the negative
side compounds with rows in the window - a single 131K row costs -2.4%, two 262K rows cost -35%.
That is the `1+k` verify rows landing on a deep, restored window, which is Wave-M's reopen condition
(2) seen from the other side.

The gate for this already exists and is on by default on our hardware class. The boot says so:

```
MTP drafts stop only from --spec-max-batch to this many decoding sequences on this V100 layer split
(--spec-gate-cost-model on: at any number above too; off: never) from=8 until=8
```

`vram::mtp_gate_until_default()` fixes `cfg.spec.gate_until` once at boot (`src/engine/lm.rs:5405`),
and `cost_model_gated()` (`src/spec/adaptive.rs:26`) tests only `batch`:

```rust
cfg.gate_cost_model && batch >= cfg.max_batch && (cfg.gate_until == 0 || batch <= cfg.gate_until)
```

Its two callers (`src/engine/lm.rs:9054`, `src/engine/lm.rs:9706`) sit inside the step loop, where the
window's sequences and their resident depths are in hand. **The gate therefore weighs the number of
sequences, while the cost it is meant to model scales with sequence count times resident depth.** Our
table is the price of that mismatch: the shape that gains +7% at 1K loses -35% at 262K, and one static
integer cannot separate them.

### Proposed patch (small, and it is the cheapest tg on the board)

Gate on window pressure, not batch width. The engine already accounts the same quantity for admission -
`--resident-window-budget` sums rows against a token budget (`1631acf`) - so the predicate can reuse it:

```
cost_model_gated(batch, resident_tokens, cfg) =
    gate_cost_model && batch >= max_batch && (gate_until == 0 || batch <= gate_until)
                 && resident_tokens <= spec_gate_window_tokens
```

with `--spec-gate-window-tokens` defaulting to the measured crossover (order 8-32K per sequence, so
the C=1 and C=2 rows above both land on the correct side), or equivalently as a fraction of
`resident-window-budget`. Effect: the short-window/agent lane keeps MTP's +5-7% (our 4-card S1 read
+13.6% on C1) and the deep lane never pays -35%, in one boot, no second binary.

Sequencing: re-run the band sweep on 4 cards first - the crossover moves right as residency grows, and
the 4-card number is what production ships against.
