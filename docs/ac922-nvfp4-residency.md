# NVFP4 expert residency: the scale-pool design (AC922 secondary development)

Status: implemented on `coldtier` (commit `2130b13`, `LEGION_NVFP4_CACHE=1`); the gates
(§5) run on the AC922 dev lane. Owner thread: the AC922 Strata↔Legion pack line
(runbook §372). Target: let an NVFP4 checkpoint's experts enter the VRAM expert cache,
bitwise, so the NVFP4 pack decodes at (close to) the UD pack's rate.

Implementation notes (what the code survey added to the design):

* The tiered branch of the MoE builder (`src/model/core.rs`) builds its products without
  ever consulting `.scale` — today's rejection is what keeps that true. So the pool rides
  through `TieredLayer::scale(k)` and the scaled products call
  `Linear::mul_mat_id` (the plain path's function): its dispatch on sm_70 lands on
  `ctx.mul_mat_id` + the Q8/BF16 hint + the exact `reshape/repeat/get_rows/mul` tail,
  with the POOL as the scale tensor and the remapped ids. No kernel, no ggml change.
* Subpaths that compute without consulting a scale gate out for scaled layers: the fused
  `moe_ffn` op (`supported_as`' type list already excludes NVFP4; the guard says it out
  loud), `expert_mv`/pad/skip rows (their `fits_as` is a Q4_K-family type list too), the
  host's `cpu_share`, and the overlap `glu_down_tiered` split.
* Staging is off for a scaled cache (`ExpertResidency::scaled` gates `stages_entry`):
  prompt steps read the tiered tensors in place (cached slots from VRAM, the rest over
  the link), which is the PP-can-yield trade. The staged closures' bare `mul_mat_id`
  never runs, so no staged-id scale view is needed (phase 2 may revisit).
* A layer whose pool cannot exist (bounded host tier, `Fetch::Copy`, CPU placement, a
  missing/non-F32 scale) is dropped from the cache entirely at `new()`, with one warn
  line: its reads stay exactly today's. No partially-scaled layer can exist.
* The pool is `[n_slots + n_expert]` F32 per kind, in a small device context per layer;
  the tail rows are the checkpoint's values (static), the slot rows are rewritten from
  the table inside `upload_tables` — the same call that makes a slot nameable, so ids
  and scalars always move together.

## 1. The block

`src/engine/lm.rs:3592-3597`: `ExpertResidency`'s `find()` returns `None` for any expert
stack that has a `.scale` companion tensor ("Per-expert scales would need remapping
too"). Consequence on the converted NVFP4 pack: the boot self-report is
`expert cache layers=0 slots=0 gib=0.0` - every expert is read from the host every
window, and decode loses the whole cache line that carries the UD pack (short-window
tg 20.3 vs UD ~31.2 on the same lane, runbook §372).

The concern is real. The cache works by remapping the router's ids into a pool of
`n_slots + n_expert` rows (slots first, the host-mapped experts after;
`src/residency/experts.rs:5-15`, `Layer::table`). Weight rows follow the remap because
the pool tensor IS the staged layout. But a per-expert scale is indexed by the *same*
ids: `w4a16(name, w, scale, n)` (`src/model/core/linear.rs:371`, reached through
`mmid_per_pair`, `:1038-1052`) reads `scale[ids]`. With ids remapped to slot numbers,
`scale[slot]` is the wrong expert's scalar. Hence the blanket rejection.

## 2. Why the fix is small

Three facts collapse the problem:

1. **The scale is one fp32 per expert.** In this pack `.scale` is `F32[512]`
   (`blk.N.ffn_{gate,up,down}_exps.scale`, gate and up sharing the `s_gu` value, down
   `s_d`, the `experts.bin` blob's f32x4 tail - see `tools/strata_pack_to_legion.py`'s
   docstring). 2 KiB per kind per layer, not a per-block array.
2. **The scale tensor resolves through the same name chain as weights.** `Model::w`
   (`src/model/core.rs:2320-2336`) tries `requant -> widened -> weights ->
   host_tier`; adding a scale-pool answer for `*.scale` names routes every existing
   call site (`linear::scale_name` users in `core.rs`) to the pool with **no model
   code changes**.
3. **The fold-at-conversion alternative is dead.** Folding the scalar into the E4M3
   block scales would need it to be a power of two to stay bit-exact; a probe over
   8,192 scale values found `pow2 = 0` (sample ~6e-5 to 1e-4, gptq magnitudes). So the
   engine-side pool is the only quality-neutral route.

## 3. Design: a per-layer, per-kind scale pool

Mirror of the weight pool, one tiny tensor per `(layer, kind)`:

```
scale_pool[k] : F32[n_slots + n_expert]   (CUDA buffer, ~ (64+512)*4B per kind)
  rows [0, n_slots)           written alongside each expert's bytes at
                              staging and at every swap-in: pool[k] = tail[e]
  rows [n_slots, +n_expert)   static copy of the checkpoint's .scale rows
```

Touch points:

* `src/residency/experts.rs`
  - allocate + initialize the pools in `Layer`'s construction (tail from the model's
    `.scale`, one D2D copy at load);
  - the staged byte copies (`stage_copies` / `packed_copies`, `:451+`, and the prefill
    staging buffers) get a parallel list of 4-byte destinations: fold the scalar into
    the existing per-swap job (a `cudaMemcpyAsync` on the same stream, or one 64-wide
    gather kernel per swap batch - the count is `n_slots`, so either is free);
  - expose `Layer::scale_tensor(k)` beside `Layer::tensor(k)` (`:429-432`).
* `src/engine/lm.rs`
  - register the pools in `w()`'s chain for the `blk.*.ffn_*_exps.scale` names (the
    same hook site that hands out the weight pool);
  - delete the `.scale` rejection in `find()` (`:3592-3597`); keep rejecting other
    companions (`.input_scale` etc.) and non-nvfp4 surprises;
  - the guard that must hold: the pool is registered only for layers whose stacks
    entered the cache. A rejected or `stage_below(n_layer)` layer (MTP head) keeps the
    raw tensor, whose ids are never remapped there.
* kernels: **no changes**. NVFP4 rows stay on the same dp4a/LUT
  (`vec_dot_nvfp4_q8_1`) and mmq paths, reading `scale[ids]` - now correct by
  construction because rows are slot-aligned.

Bitwise argument: the staged bytes equal the host bytes (the converter moves whole
rows, nothing requantizes), the pool's tail rows equal the original tensor, and the
kernel does the same loads and the same fp32 epilogue multiply with the same value.
Output must be bit-identical to the host-read path; the acceptance gate is exactly
that (§372's det suite plus a slot-vs-host diff: run one window with the pool forced
empty and one with the target expert staged, hashes equal).

Budget: ~288 KiB VRAM total (3 kinds x 48 layers x ~576 rows x 4B) - nothing. Host
footprint unchanged. No new pinned memory (the tail copy lands in the device pool).

## 4. Boundaries to respect

* **Slot cuts**: when the KV pools' mapped tail takes slots (`experts.rs:2762`,
  `--kv-vmm-keep`), the weight pool and the scale pool must shrink in lockstep - keep
  `n_slots` a single source, which the design already does by deriving rows from it.
* **Dual-lane memory policy (runbook §372 incidents)**: the NVFP4 dev lane hung the
  box twice while a production lane coexisted. Residency does not change the host
  footprint; until the cause is nailed, NVFP4 windows stay single-lane (production
  stopped) or after the fix ships, re-evaluate - the fix's whole point is that decode
  stops streaming 73 GiB of expert bytes from the host.
* **123K+0-cache deep prefills**: forbidden pre-fix; post-fix they exercise the
  prefill staging path (the `stage_copies` hook), not just swaps - keep the conservative
  65K arm first.
* **Profile**: the NVFP4 boots reported `profile=None`. The static expert prior is
  quantization-agnostic (keyed by layer/expert ids), so point the lane at the learned
  profile or regenerate it; without it the hit-rate measurement under-reads.
* **`--tp-moe-ep` / fused paths**: `ep_mv`/`fuses` require `!scaled` already
  (`core.rs:5021-5036`), the `ep_pairs` remap path is out of scope for 2-GPU geometry.
  Do not widen that surface in this change.

## 5. Plan and gates

1. Implementation (est. 300-400 lines with tests): pools + registration + rejection
   lift + swap hook; a unit test asserting pool rows equal `table`-indexed `.scale`
   values after simulated swaps.
2. Single-lane 2-GPU window: bitwise A/B (pool vs host-read) + det/acc/IMG gates +
   `nvtg` short1/short2/deep65 arms against the §372 NVFP4 numbers.
3. If tg lands in the UD band (~28-31): 4-GPU production candidate through the usual
   paired protocol; update runbook §373 and the front-page perf table.
4. Only after that, re-price the 123K/262K arms and the aggregation picture.

## 6. Rejected alternatives (record)

* Converter-side scalar fold into E4M3 block scales: kills bit-exactness (pow2=0).
* Kernel-side dual ids (slot ids for weights, raw ids for scales): touches every NVFP4
  mmid kernel and the grouped W4A4 path; larger diff, same outcome.
* Leaving scales on host-mapped memory indexed by slot: numerically wrong, the bug
  the rejection exists to prevent.
* Folding the scalar at load into the E4M3 block scale bytes of the staged VRAM copy:
  exact only for pow2 scalars; also makes the staged bytes differ from the arena's,
  breaking the "staged == host bytes" invariant the swap validator leans on.

## Validation record (AC922, 2026-11-26)

Two GPUs, `LEGION_NVFP4_CACHE=1` vs `0` on one binary, boots apart: the six
det hashes are identical (`e523274d…5309bdcf…`), equal to the pre-patch
NVFP4 baseline — the pool is bit-exact by construction and in measurement.
Throughput with the pool: 2K single 32.1 tok/s (20.3 without), 2K×2 mean
27.1 (14.6), 65K 25.6 with pp 203 (14.3/145).

Four GPUs at production shape (cache 6.5 GiB/card, KV_VRAM 274432,
max-batch 4096): 262144-token windows, four concurrent, eight streams —
wall 62.7 s, aggregate 16.3 rows/s, per-row decode median 11.5, zero
errors. The default shape (cache fill, batch 8192, KV 262144) collapses
at C4 deep: cudaMalloc failures until the cache is capped, then a
revive storm until the KV pool clears 4×262144 with slack. Per-row decode
falls at three rows (24 → 20.6 → 11.6 → 11.5 for C1..C4): the Volta
multi-row QSA cost, unchanged by the pool. Vision gate and the 6/6
accuracy battery pass on the same boot.
