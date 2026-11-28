#!/bin/bash
# usage: wb_arm_max.sh <TAG> <ENVDRAFT>
# Same shape as wb_arm.sh but with the counted fixture: every row legally fits
# ctx, so no arm loses a sequence to a 400 and agg is measured on true C4.
set -uo pipefail
TAG=$1; ENVF=$2
cd /data/work/legion || exit 1
LEGION_LANE_ENV=$ENVF bash /data/work/legion/legion-lane.sh 4 restart > "/data/work/boot-$TAG.log" 2>&1
sleep 5
export LB_URL=http://127.0.0.1:8090
P=$(cat /tmp/legion-lane-4.pid)
export LB_API_KEY=$(tr "\0" "\n" < /proc/$P/environ | grep ^LEGION_API_KEY= | cut -d= -f2-)
c=0
for i in $(seq 1 40); do
  c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' $LB_URL/health)
  [ "$c" = 200 ] && break
  sleep 5
done
L=$(ls -t /data/work/legion/logs/legion-4-*.log | head -1)
echo "== $TAG health=$c polls=$i log=$L"
B=$(grep -m1 "resident window budget" "$L" | sed -E 's/.*(resident window budget on.*)/\1/' | cut -c1-110)
echo "admission: ${B:-budget off (rows enter regardless of size)}"
python3 reburst_max.py 261872 4 "$TAG" | tee -a /data/work/wb_pair.jsonl
echo "$TAG traces: retry=$(grep -c 'retrying once' "$L") refused=$(grep -c 'admission refused' "$L") h400=$(grep -c ' 400 ' "$L")"
echo "$TAG per-row (engine decode rate):"
grep "request done" "$L" | sed -E 's/.*prompt_tokens=([0-9]+) cached_tokens=([0-9]+) completion_tokens=([0-9]+) queue_ms=([0-9.]+).* seconds=([0-9.]+) prefill_seconds=([0-9.]+).* decode_tok_s=([0-9.]+).*/  pt=\1 cached=\2 ct=\3 q_ms=\4 s=\5 pf=\6 dtg=\7/' | tail -8
echo ARM-END "$TAG"
