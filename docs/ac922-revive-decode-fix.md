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

## Ranked candidates (fitted to the observed shape)

1. **Plan-fit starvation → hydrate_abort** (`7578-7600`): the first deep revive(s) park most of
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
