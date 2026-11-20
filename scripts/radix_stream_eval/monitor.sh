#!/usr/bin/env bash
# Watch the ten-talk stream job on a Radix node from the Mac every 5 minutes. Exits 0 when out/done appears,
# 1 when the container is gone, a log shows Traceback, out of memory or Killed, or nothing under the run
# directory changed for 20 minutes. Usage: monitor.sh <receipt> <container> <run dir under /data/luojiaxuan/home>
set -uo pipefail
RECEIPT=$1 NAME=$2 RUN=$3
HELPER=~/.claude/skills/radix-gpu-nodes/scripts/radix_node.py
while true; do
  out=$(timeout 120 python3 "$HELPER" ssh --receipt "$RECEIPT" -- "
    cd $RUN 2>/dev/null || { echo STATE missing-dir; exit; }
    echo STATE \$(podman ps -a --filter name=^$NAME\$ --format '{{.Status}}' | cut -d' ' -f1)
    echo DONE \$([ -f out/done ] && echo yes || echo no)
    echo AGE \$(( \$(date +%s) - \$(find . -type f -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1) ))
    echo ERR \$(grep -l -E 'Traceback|out of memory|OutOfMemoryError|Killed' logs/*.log download.log 2>/dev/null | tr '\n' ' ')
    echo OUTS \$(ls out 2>/dev/null | tr '\n' ' ')
    echo MEMFREE \$(free -g | awk '/Mem:/ {print \$7\"/\"\$2}')
    echo GPU \$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')
  " 2>&1)
  echo "$(date -u +%FT%TZ) $(echo "$out" | tr '\n' ' ')"
  if echo "$out" | grep -q "^DONE yes"; then echo RESULT done; exit 0; fi
  state=$(echo "$out" | awk '/^STATE/ {print $2}')
  age=$(echo "$out" | awk '/^AGE/ {print $2}')
  err=$(echo "$out" | awk '/^ERR/ {$1=""; print}' | tr -d ' ')
  if [[ -n "$state" && "$state" != "Up" ]]; then echo RESULT container-$state; exit 1; fi
  if [[ -n "$err" ]]; then echo RESULT error-in-$err; exit 1; fi
  if [[ -n "$age" && "$age" -gt 1200 ]]; then echo RESULT stalled-$age; exit 1; fi
  sleep 300
done
