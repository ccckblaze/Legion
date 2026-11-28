#!/bin/bash
# Decisive admission round: counted fixture (all 4 rows legal), fresh empty cold
# tier per boot, three shapes: off / 3x-full budget / 2x-full budget.
set -uo pipefail
cd /data/work/legion || exit 1
for arm in M0 M2 M1; do
  case "$arm" in
    M0) ENVF=/data/work/legion/wb4g.env ;;
    M2) ENVF=/data/work/legion/wb4m.env ;;
    M1) ENVF=/data/work/legion/wb4f.env ;;
  esac
  rm -rf /data/cache/conv-tier-fresh
  mkdir -p /data/cache/conv-tier-fresh
  bash /data/work/legion/wb_arm_max.sh "$arm" "$ENVF"
  sleep 10
done
echo MAX-END
