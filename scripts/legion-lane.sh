#!/bin/bash
# legion-lane.sh <2|4|6> <start|stop|restart|status>  - AC922 Legion lane, S7 production shape.
#
# Shape of record 2026-11-25 02:0x (docs/ac922-perf-seconddev.md SS11, gates green):
#   q8 KV + VRAM pool 262144 + host tier 786432, heat-pass 1, batch-tokens 4096,
#   groups = cards / min-group 1, micro 1024, long-batch 16384,
#   prefix cache ON (states 4, interval 8192).  123K C4 reburst agg 129.4,
#   full-window 262K C4 agg 86.2, poison55 uniform, needle x3 + acc 6/6 + det sha equal.
# Every tunable is an env override so a wave can boot arms without editing the file.
LANE_ENV=${LEGION_LANE_ENV:-/data/work/legion/lane.env}
[ -r "$LANE_ENV" ] && . "$LANE_ENV"
set -uo pipefail

BIN=${LEGION_BIN:-/data/work/legion-src/target/release/legion}
MODEL_DIR=${LEGION_MODEL_DIR:-/data/work/legion-models}
MODEL=${LEGION_MODEL:-Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
DRAFTER=${LEGION_DRAFTER:-mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
USE_DRAFTER=${LEGION_USE_DRAFTER:-0}
SPEC_MODE=${LEGION_SPEC_MODE:-draft}
LOGD=${LEGION_LOGD:-/data/work/legion/logs}
PORT=${LEGION_PORT:-8090}
CTX=${LEGION_CTX:-262144}
KV_VRAM=${LEGION_KV_VRAM:-262144}
KV_HOST=${LEGION_KV_HOST:-}
PIPE_MIN_GROUP=${LEGION_PIPE_MIN_GROUP:-1}
PREFIX_STATES=${LEGION_PREFIX_STATES:-4}
PREFIX_INTERVAL=${LEGION_PREFIX_INTERVAL:-8192}
LONG_BATCH=${LEGION_LONG_BATCH:-16384}
LONG_ALONE=${LEGION_LONG_ALONE:-}
LONG_MICRO=${LEGION_LONG_MICRO:-1024}
SPEC_K=${LEGION_SPEC_K:-}
MAX_SEQS=${LEGION_MAX_SEQS:-16}
MAXB=${LEGION_MAX_BATCH:-4096}
EXTRA=${LEGION_EXTRA:-}
STATE=${LEGION_STATE:-/tmp/legion-lane.state}
PIDF=${LEGION_PIDF:-/tmp/legion-lane-PLACE.pid}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-3600}

CARDS=${1:-}; ACTION=${2:-start}
[ -n "$CARDS" ] && PIDF=${PIDF/PLACE/$CARDS}
[ -n "$CARDS" ] || { echo "usage: $0 <2|4|6> <start|stop|restart|status>" >&2; exit 2; }
mkdir -p "$LOGD"

case "$CARDS" in
  2) GPUS=${LEGION_GPUS:-2,5} ;;
  4) GPUS=${LEGION_GPUS:-0,1,3,4} ;;
  6) GPUS=${LEGION_GPUS:-0,1,2,3,4,5} ;;
  *) echo "cards must be 2, 4 or 6" >&2; exit 2 ;;
esac
[ -n "$KV_HOST" ] || KV_HOST=$(( CARDS >= 6 ? 1310720 : 786432 ))

LOG=$LOGD/legion-$CARDS-$(date +%Y%m%d-%H%M%S).log

stop() {
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    pid=$(cat "$PIDF"); echo "stop: TERM $pid"
    kill "$pid"; for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "stop: KILL $pid"; kill -9 "$pid"; }
  fi
  rm -f "$PIDF"
}

wait_health() {
  for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then return 0; fi
    [ -f "$PIDF" ] && ! kill -0 "$(cat "$PIDF")" 2>/dev/null && { echo "start: process died"; return 1; }
    sleep 1
  done; return 1
}

start() {
  local g bus short node seen_nodes="" ok=1 mtotal pinnable
  for g in ${GPUS//,/ }; do
    bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$g" '$1==i{print $2}')
    short=$(printf '%04x:%s' "0x${bus%%:*}" "${bus#*:}")
    node=$(cat "/sys/bus/pci/devices/${short,,}/numa_node" 2>/dev/null || echo '?')
    case " $seen_nodes " in *" $node "*) continue ;; esac
    seen_nodes="$seen_nodes $node"
    read -r mtotal pinnable < <(awk '
      /MemTotal:/ {t=$4} /MemFree:/ {f=$4} /Active\(file\):/ {af=$4} /Inactive\(file\):/ {i2=$4}
      /Mapped:/ {m=$4} /Dirty:/ {d=$4} /Writeback:/ {wb=$4}
      END { e=f+af+i2-m-d-wb; if (e<0) e=0; mg=(t/10>16777216)?t/10:16777216;
            printf "%d %d", t, (e>mg? e-mg : 0) }' "/sys/devices/system/node/node$node/meminfo" 2>/dev/null)
    echo "node$node (gpu$g): total=$(( mtotal / 1024 )) MiB pinnable=$(( pinnable / 1024 )) MiB"
  done

  echo "start: gpus=$GPUS ctx=$CTX kv=$KV_VRAM+$KV_HOST prefix=$PREFIX_STATES/$PREFIX_INTERVAL long=$LONG_BATCH groups=$CARDS/$PIPE_MIN_GROUP"
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' -v w="$GPUS" \
    'BEGIN{n=split(w,a,",");for(i=1;i<n+1;i++)g[a[i]]=1} $2+0>1024 && g[$1] {printf "%s(%s MiB) ",$1,$2}')
  if [ -n "$busy" ]; then echo "start: REFUSED, cards busy: $busy"; return 2; fi
  FLAGS=(--pipeline-groups "$CARDS" --pipeline-min-group "$PIPE_MIN_GROUP"
         --prefix-cache-states "$PREFIX_STATES" --prefix-cache-interval "$PREFIX_INTERVAL"
         --long-batch-tokens "$LONG_BATCH" --long-pipeline-micro "$LONG_MICRO")
  [ -n "$LONG_ALONE" ] && FLAGS+=(--long-alone-tokens "$LONG_ALONE")
  [ -n "$SPEC_K" ] && FLAGS+=(--spec-k "$SPEC_K")
  [ "$USE_DRAFTER" = 1 ] && FLAGS+=(--spec "$SPEC_MODE" --spec-draft "$MODEL_DIR/$DRAFTER")
  # shellcheck disable=SC2086
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPUS "$BIN" serve \
    --model "$MODEL_DIR/$MODEL" \
    --port "$PORT" --device cuda --gpus "$CARDS" --experts cache \
    --max-seqs "$MAX_SEQS" --ctx-size "$CTX" --qsa-attn-chain --qsa-scores-tc --ple-ops \
    --weights-fp16 load --weights-fp16-gates --hc-fuse \
    --long-dataflow --long-dataflow-peer \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --kv-cache-tokens "$KV_VRAM" --kv-host-tokens "$KV_HOST" \
    --expert-heat-pass-cap 1 --max-batch-tokens ${LEGION_MAX_BATCH:-4096} \
    --model-name qwen3.8-flash-next \
    "${FLAGS[@]}" $EXTRA >> "$LOG" 2>&1 </dev/null &
  echo $! > "$PIDF"
  echo "start: pid $(cat "$PIDF") log $LOG"
  if wait_health; then echo "HEALTH-OK"; else echo "HEALTH-FAIL"; tail -40 "$LOG"; return 1; fi
  grep -aE "memory plan|KV pool sizing|host tier|expert slots|step budget|loaded in|listening" "$LOG" | tail -12
}

status() {
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "RUNNING pid $(cat "$PIDF")"; curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  else echo "DOWN"; fi
}

case "$ACTION" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 5; start ;;
  status) status ;;
  *) echo "unknown action $ACTION" >&2; exit 2 ;;
esac
