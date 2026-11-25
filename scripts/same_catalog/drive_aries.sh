#!/usr/bin/env bash
# Run the 1m and then the 100k scale with run_aries.sh on three borrowed A6000 GPUs, and yield them:
# when a process of another user appears on one of them, stop the run, wait until the GPUs have been
# free of other users for 10 minutes, then resume (finished arms are skipped). Touch stop in the run
# root to end it. The run root holds env.sh, code/, rasst/, pyhook/, reference/ and the ACL copy.
#   drive_aries.sh <run root>
set -uo pipefail
ROOT=${1:?run root}
source "$ROOT/env.sh"
export CODE="$ROOT/code" RASST="$ROOT/rasst" PYHOOK="$ROOT/pyhook" REFERENCE="$ROOT/reference" ACL="$ROOT/acl6060_zh_segments"
LOG="$ROOT/driver.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

foreign() {  # prints the first process of another user on GPUS; fails when there is none
  local cards rows
  cards="$(nvidia-smi -i "$GPUS" --query-gpu=index,uuid --format=csv,noheader)" || { echo "GPU query failed"; return 0; }
  rows="$(nvidia-smi -i "$GPUS" --query-compute-apps=gpu_uuid,pid --format=csv,noheader)" || { echo "GPU query failed"; return 0; }
  while IFS=, read -r uuid pid; do
    [[ -n "$uuid" ]] || continue
    pid="${pid//[[:space:]]/}"
    owner="$(ps -o user:32= -p "$pid" 2>/dev/null | awk '{print $1}')"
    [[ "$owner" == "$(id -un)" ]] || { echo "GPU $(awk -F, -v u="$uuid" '$2 == " " u {print $1}' <<<"$cards") PID $pid owner ${owner:-unknown}"; return 0; }
  done <<<"$rows"
  return 1
}

run=0
stop_run() {
  ((run > 1)) || return 0
  kill -TERM -- "-$run" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 -- "-$run" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$run" 2>/dev/null || true
  wait "$run" 2>/dev/null || true
  run=0
}
trap 'stop_run; log "driver exiting"' EXIT

log "driver started"
for scale in 1m 100k; do
  while [[ ! -e "$ROOT/$scale/out/done" && ! -e "$ROOT/stop" ]]; do
    clean_since=$SECONDS
    while reason="$(foreign)"; do
      log "waiting: $reason"
      clean_since=$SECONDS
      sleep 60
      [[ ! -e "$ROOT/stop" ]] || break 3
    done
    if ((run == 0)); then
      mkdir -p "$ROOT/$scale"
      setsid bash "$CODE/scripts/same_catalog/run_aries.sh" "$scale" "$ROOT/$scale" >> "$ROOT/$scale/run.log" 2>&1 &
      run=$!
      log "started $scale, process group $run"
    fi
    while kill -0 "$run" 2>/dev/null; do
      sleep 15
      if [[ -e "$ROOT/stop" ]]; then log "stop marker"; stop_run; break 3; fi
      if reason="$(foreign)"; then
        log "yielding: $reason"
        stop_run
        until ! foreign >/dev/null && sleep 600 && ! foreign >/dev/null; do sleep 60; done
        continue 2
      fi
    done
    wait "$run"; rc=$?
    run=0
    log "$scale exited with $rc"
    if [[ ! -e "$ROOT/$scale/out/done" ]]; then
      log "$scale did not finish; stopping for inspection"
      break 2
    fi
  done
done
[[ -e "$ROOT/1m/out/done" && -e "$ROOT/100k/out/done" ]] && log "both scales done" && touch "$ROOT/done"
