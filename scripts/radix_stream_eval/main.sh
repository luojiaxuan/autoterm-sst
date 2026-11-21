#!/usr/bin/env bash
# Download the public models and inputs, then run a job script (default run.sh, the ten-talk stream phases).
set -euo pipefail
cd "$(dirname "$0")"
umask 077
python3 download.py > download.log 2>&1
exec bash "${1:-run.sh}"
