#!/usr/bin/env bash
# Download the public models, inputs and the prebuilt catalog, wait for the medicine lectures, then run the three arms
# with run_radix_arms.sh. The lectures are prepared as scripts/radix_stream_eval/download.py describes and copied
# into the container; an empty .complete file in their directory marks them ready.
#   bash main_radix_arms.sh <catalog repo> <revision>
set -euo pipefail
cd "$(dirname "$0")"
umask 077
# Xet transfers stalled for an hour on the ACL segments; plain HTTP with a timeout retries instead.
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
find /data/inputs -name '*.incomplete' -delete 2>/dev/null || true
python3 download_public.py > download.log 2>&1
python3 - "$1" "$2" >> download.log 2>&1 <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], repo_type="dataset", revision=sys.argv[2], local_dir="/data/inputs/catalog"), flush=True)
PY
MED=/data/inputs/main/main_result/audio/medicine
for _ in $(seq 1 240); do
  [[ $(find "$MED" -name '*.wav' 2>/dev/null | wc -l) -ge 5 && -e "$MED/.complete" ]] && break
  sleep 30
done
[[ -e "$MED/.complete" ]] || { echo "the medicine lectures did not arrive" >> download.log; exit 1; }
exec bash run_radix_arms.sh /data/inputs/catalog/catalog
