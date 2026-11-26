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
