#!/bin/bash
# gate.sh <cards> <note>: the standing boot gate for the AC922 lane (docs/ac922-production-shape.md)
C=${1:-4}; NOTE=${2:-boot}
. /data/work/legion/lane.env
export LB_URL="http://127.0.0.1:${LEGION_PORT:-8090}"
export PROBE_BASE="$LB_URL"
[ -n "$PROBE_KEY" ] && [ -n "$LB_API_KEY" ] || { echo "GATE-ENV-FAIL keys"; exit 1; }
L=$(ls -t /data/work/legion/logs/legion-$C-*.log | head -1)
echo "GATE-START $(date +%H:%M:%S) $NOTE"
echo "LOG $(basename "$L") bytes=$(wc -c < "$L") warnerr=$(grep -acE ' WARN | ERROR ' "$L")"
grep -aE 'conv tier: live|VRAM after load|memory plan part="host"' "$L" | sed -e 's/^.*INFO //' | cut -c1-95
cd /data/work/legion/eval || exit 1
for i in 1 2 3; do echo -n "DET$i "; timeout 900 python3 -u qualLegion.py det 2>&1 | tail -1; done
echo -n "IMG "; timeout 300 python3 /data/work/vl-probe.py img 2>&1 | tail -1 | cut -c1-95
echo "--- undecl(call) ---"; timeout 600 python3 -u undecl_probe.py call 2>&1 | tail -4
python3 /data/work/legion/eval/boot-archive.py "$C"
echo "GATE-END $(date +%H:%M:%S)"
