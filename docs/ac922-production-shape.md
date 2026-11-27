# AC922 Production Shape

The exact thing running in production on the AC922, how it is launched and built, and what is
checked on every boot. This is an operations record. The flag-by-flag measurement papers stay
where they are: the author's shapes in [`run/ac922/`](run/ac922/), our campaign numbers in
[ac922-perf-seconddev.md](ac922-perf-seconddev.md) and
[ac922-cold-tier-l1.md](ac922-cold-tier-l1.md), the window anatomy in
[ac922-decode-window-anatomy.md](ac922-decode-window-anatomy.md).

## Machine and contract

| | |
|---|---|
| Box | IBM AC922, POWER9 ppc64le, **597.2 GiB host RAM** (DIMMs filled 2026-11-27; this paper's pre-11-27 numbers were taken on 256 GiB), RHEL 8 |
| GPUs | 4x Tesla V100-SXM2-16GB, physical cards `0,1,3,4` = two NVLink pairs, one pair per NUMA socket |
| Kernel | 4.18 plus the [power9-nvidia-nvlink2](https://github.com/eelgaev/power9-nvidia-nvlink2) patches, driver 550.54.15 |
| Model | Qwen3.8-Flash-Next UD-Q4_K_XL, 4-shard GGUF, BF16 checkpoint converted on load |
| Contract | 262144-token window, 4 concurrent sessions at that window, images accepted on the same endpoint |
| Endpoint | `0.0.0.0:8090`, `LEGION_API_KEY` in the environment |

## The command

```sh
CUDA_VISIBLE_DEVICES=0,1,3,4 legion serve \
  --model /data/work/legion-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --model-name qwen3.8-flash-next --port 8090 \
  --device cuda --gpus 4 --experts cache --max-seqs 16 \
  --ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0 \
  --kv-cache-tokens 262144 --kv-host-tokens 786432 \
  --qsa-attn-chain --qsa-scores-tc --ple-ops --hc-fuse \
  --weights-fp16 load --weights-fp16-gates \
  --long-dataflow --long-dataflow-peer --expert-heat-pass-cap 1 \
  --max-batch-tokens 8192 \
  --pipeline-groups 4 --pipeline-min-group 1 \
  --prefix-cache-states 4 --prefix-cache-interval 8192 \
  --long-batch-tokens 16384 --long-pipeline-micro 1024 \
  --mmproj /data/work/vision-test/mmproj-BF16.gguf \
  --conv-disk /data/cache/conv-tier-prod --conv-disk-gib 200 \
  --undeclared-tool-calls call
```

## What the non-obvious flags are doing

| Flag | Why it is here |
|---|---|
| `--experts cache` | The routed experts do not fit 16 GB of VRAM per card, so they live in pinned host memory on the GPU's own NUMA node and the hottest ones are cached in VRAM (35.4 GiB of cache over the four cards, 12157 slots, 48 layers). |
| `--weights-fp16 load` + `-gates` | The V100 has no BF16 tensor cores; convert on load and multiply on the FP16 ones. Opt-in because outputs shift slightly, so it is inside the boot gate below. |
| `--qsa-attn-chain`, `--qsa-scores-tc`, `--ple-ops`, `--hc-fuse` | The four Volta prompt-step kernels: chained `m8n8k4` accumulation in the sparse-attention kernel, indexer pool scores split into FP16 high/low and run on the tensor cores, the n-gram embedding as two ops instead of a thirty-node graph, and the gated shared expert folded into the fused MoE sum. The first two change sums' rounding, so they are gated, not free. |
| `--long-dataflow` + `--long-dataflow-peer` | While a long prompt borrows the cache, each stage starts its next micro-chunk when its own inputs are ready instead of waiting for the slowest stage, and the residual goes straight into the next GPU's mailbox over NVLink instead of through host memory. Same bytes, same outputs. |
| `--max-batch-tokens 8192` | The big prefill step. Prompts longer than a step run in other step sizes when the budget changes, so greedy output can move: this is a gated value, and 4096 is the launcher default. |
| `--kv-cache-tokens 262144` + `--kv-host-tokens 786432` | The window. 258368 tokens of q8 KV stay resident per card and 786432 tokens (14.06 GiB) sit in the host tier, so four 262144-token sessions are all live at once. |
| `--pipeline-groups 4 --pipeline-min-group 1` | Decode runs as four groups whose steps are in flight at once, so one GPU computes a group's step while the next GPU computes another's. The minimum of 1 lets groups collapse when fewer sequences are live. |
| `--prefix-cache-states 4 --prefix-cache-interval 8192` | Recurrent-state checkpoints every 8192 prompt tokens, four of them, so the second request over a shared system prefix does not replay it. |
| `--conv-disk ... --conv-disk-gib 200` | The cold tier: content-addressed KV blocks and state checkpoints on NVMe, so a restart or an eviction resumes from a disk read instead of a prefill storm. |
| `--mmproj ...` | Image input on the same endpoint. GPU tower on sm_70; the reference-image probe below is what makes this acceptable rather than assumed. |
| `--undeclared-tool-calls call` | A well-formed call that names a tool the request did not declare goes to the client on the tool channel instead of staying in the reply text. See [serving.md](serving.md#constrained-output-json-and-strict-tool-calls). |

## Binary of record (2026-11-27 10:45 CST)

`/data/work/legion/bin-ctlock`, sha256 `24f72fdccb49`, adopted into `lane.env` with a dated
`# ADOPTED` header and a timestamped backup beside it. `legion --version` stamps it `b4a26a4`, one
commit behind its source, for the reason described under Building: `build.rs` reads HEAD when the
build starts, and this build started before the commit that made it. The stamp is not the evidence
that the code is in the binary - the runtime is: the boot created `/data/cache/conv-tier-prod/writer.lock`,
which only the single-writer code path can create, and the tier's own counters plus `devq3`/`devq4`/
`devq6` in [ac922-revive-decode-fix.md](ac922-revive-decode-fix.md) reproduce all three fixes.

What it buys, each verified separately rather than assumed:

| fix | how it was proven | effect measured |
|---|---|---|
| S395 shutdown race | `bin-revive` vs `bin-ctlock` on the same shutdown shape | old: 1 SIGSEGV, backlog landed +2 of ~18; new: 0 SIGSEGV, +302 |
| S396 two writers | unit test 7/7, plus a real `kill -9` mid-load | after the kill: 7/7 states and 221/221 blocks match their index; next boot agrees |
| S394 revive decode | `armud UD2` vs `UD1`, production shape, same night | neutral: 8k -0.1%, 122880x4 0.0%, 262144x4 wall -1.1% / tg +0.7%, det same |

The NV-scale lane keeps `bin-nvscale` on purpose: it is the reproducibility baseline for the
NV-scale arms, and its own tier scanned clean (205/205).

## Launching

Everything is assembled by `/data/work/legion/legion-lane.sh`, which reads
`/data/work/legion/lane.env` for the overrides above. Every tunable is an env var, so an arm can
boot without editing the file; the launcher also refuses to start when any target card already
holds more than 1 GiB.

```sh
bash /data/work/legion/legion-lane.sh 4 status      # health + per-card memory
bash /data/work/legion/legion-lane.sh 4 restart     # stop, boot, wait for /health
bash /data/work/legion/legion-lane.sh 2 start       # dev lane, cards 2 and 5
```

`CARDS` selects the GPU set (`2` -> `2,5`, `4` -> `0,1,3,4`, `6` -> all), the host tier
(`786432` below 6 cards, `1310720` at 6) and the group count. Logs land in
`/data/work/legion/logs/legion-<cards>-<stamp>.log`, the pidfile in `/tmp/legion-lane-<cards>.pid`.

## Building

`/data/work/legion/eval/ct-build.sh`, which is [building.md](building.md#ibm-ac922-power9-ppc64le-v100)
with the machine's paths:

```sh
source /opt/rh/gcc-toolset-12/enable                 # the system gcc is too old
export PATH=/opt/rust/bin:$PATH                      # rustc 1.99.1; RHEL's module cargo is not
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc         #   rustup-driven and ignores the pin
export CMAKE_CUDA_ARCHITECTURES=70
CARGO_TARGET_DIR=/data/work/legion-ct-target cargo build --release --features cuda
```

`legion --version` prints the commit (`legion 0.1.0 (be91f3b, ggml unknown, cuda on)`), taken from
git metadata by `build.rs` at the start of the build. A docs-only commit changes nothing `build.rs`
declares as an input, so `cargo build` finishes in under a second and the binary keeps the previous
stamp: `touch build.rs` before building when the stamp has to prove the source, and note that this
makes the whole ggml/CUDA side rebuild too (about 9 minutes here). A stamp one docs commit behind
its source is not a functional gap, so no restart is spent on the stamp alone.

## Every boot, before traffic

```sh
. /data/work/legion/lane.env       # LB_API_KEY and PROBE_KEY both derive from LEGION_API_KEY
export LB_URL=http://127.0.0.1:8090
export PROBE_BASE=$LB_URL          # its own statement: bash expands `export A=x B=$A`
                                   # before A takes effect, so chaining them leaves PROBE_BASE empty
L=$(ls -t /data/work/legion/logs/legion-4-*.log | head -1)
grep -acE 'WARN|ERROR' "$L"                                    # expect 0
grep -aE 'conv tier: live|memory plan part="host"|listening' "$L"
python3 /data/work/legion/eval/qualLegion.py det       # three runs, six per-prompt shas, identical
python3 /data/work/vl-probe.py img                   # same amount, same answer sha
python3 /data/work/legion/eval/undecl_probe.py call  # drifted names leave on the tool channel
bash /data/work/legion/eval/plogarch.sh 4            # archive this boot, then hand back traffic
```

Every round is archived, not just read: `plogarch.sh` writes
`logs/archive/<boot-ts>-c<cards>.tar.gz` holding the raw engine log, both lane envs, the binary path
and sha, the argv flags, `nvidia-smi`, `meminfo`, and a derived summary (geometry, conv-tier scan and
live counts, NUMA failure lines, panic count, revive aborts, requests done, mean and slow-row decode
rate, tier footprint, and the index-vs-blob gap). Keys are redacted on the way in and the archive is
checked for leftovers, because a `lane.env` copy would otherwise put `LEGION_API_KEY` into every
archive. The last field is the one that has been wrong silently: on the adopted boot it printed
`indexed_blocks=4388 blobs_on_disk=111605 orphan_blobs=107217 orphan_pct=96.1` next to
`decode_rows=27 decode_mean=46.62 decode_slow_lt5=0` - a healthy decoder on a starved cold tier.

The probes read `PROBE_BASE`/`LB_URL` plus their own key variable, so a gate against a port other
than the default 8097 needs both lines above. Compare one prompt's sha across the three runs *and*
across boots: a boot that moves a sha is a numeric change, not a restart.

A healthy boot says, in this order: `model loaded ... stages=[0..12, 12..24, 24..36, 36..48]
experts=Cache ... kv_bytes_per_token=19200`, `conv tier: live N blocks M states ... restore on`
followed by `hydrated`, `warmup shapes done ... seconds=7`, `listening on http://0.0.0.0:8090`,
and serving 33 seconds after the log's first line (45 s from the launcher's stop to health). The
host line of the memory plan is the one to
read after any change to the cache flags: on this shape it reports `kv_host_tier_mib=14400
saved_states_mib=453 expert_host_mib=73450 ngram_table_mib=27465 host_available_mib=150295`, and
`VRAM after load` leaves 1.3-2.0 GiB free per card, so nothing else may be added to a card.

## Rollback

One variable at a time, in `lane.env`, then `restart`. `--max-batch-tokens`, the four Volta kernel
flags and both weights flags are the gated ones: revert the flag, re-run the det probe, and compare
the content sha with the anchor recorded above rather than eyeballing the answer. Cold-tier and
vision flags revert to their off defaults. `lane.env` keeps a timestamped backup beside it.

## Known anchors

123K at four concurrent sessions: 129.4 tokens/s aggregate (15.8 per stream), reproduced 113.1 on
a managed boot; 262K at four: 86.2 aggregate. A cold 4x123K prefill storm costs 317.5 s, the same
sessions resumed from RAM 16.9 s, and the cold tier turns the eviction case into the read path.
Both numbers carry the boot-to-boot caveat written up in the perf doc; read the deltas, not the
last digit.
**The same cell has two honest values, 35x apart, and the difference is page cache.** `262144 x 4`
measured 18.9 s wall / 87.0 agg when it ran 90 s after another boot had already touched those
chains, and 645-652 s / 2.5 agg when it did not, with `tg_med` staying ~22-26 throughout. With
597 GiB of RAM and a 212 GiB tier sitting inside a 301 GiB page cache, a "revive" is usually a RAM
read, not an NVMe read. Quote the warmth next to the number or the number means nothing.

**Why a chain goes missing:** `tools/convtier_gc.py` (dry-run) reports, on the live tier, 4,388
indexed keys against 111,605 block files - **107,932 unreachable blobs, 201.3 GiB** - because the
pre-S396 binaries let a second writer rename a manifest shard holding its own smaller index over
the first writer's blobs. TTL is ruled out: 111,471 of those blobs were written in the last 24 h.
The token list lives only in the manifest entry, so an orphaned blob cannot be reused, only deleted;
nothing has been deleted, since 516 G is free and the set is the physical evidence.
