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
| Box | IBM AC922, POWER9 ppc64le, 256 GiB host RAM, RHEL 8 |
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

`legion --version` prints the commit (`legion 0.1.0 (649eb40, ggml unknown, cuda on)`). The stamp
comes from build-time git metadata, so a rebuild that changes no Rust file can print the previous
stamp; `touch build.rs` before building when you need the stamp to prove the binary.

## Every boot, before traffic

```sh
. /data/work/legion/lane.env       # LB_API_KEY and PROBE_KEY both derive from LEGION_API_KEY
export LB_URL=http://127.0.0.1:8090 PROBE_BASE=$LB_URL
L=$(ls -t /data/work/legion/logs/legion-4-*.log | head -1)
grep -acE 'WARN|ERROR' "$L"                                    # expect 0
grep -aE 'conv tier: live|memory plan part="host"|listening' "$L"
python3 /data/work/legion/eval/qualLegion.py det       # three runs, six per-prompt shas, identical
python3 /data/work/vl-probe.py img                   # same amount, same answer sha
python3 /data/work/legion/eval/undecl_probe.py call  # drifted names leave on the tool channel
```

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
