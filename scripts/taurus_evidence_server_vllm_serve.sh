#!/usr/bin/env bash
# note (luojiaxuan): Routed AutoTerm server whose generation goes to an external vllm serve (continuous batching).
# note (luojiaxuan): Usage: taurus_evidence_server_vllm_serve.sh <framework_port> <vllm_port>
# note (luojiaxuan): With two granted GPUs, vllm serve uses both and the retriever shares the first.
# note (luojiaxuan): With three, vllm serve uses the first two and the retriever and router model get the third.
set -euo pipefail

PORT="$1"
VLLM_PORT="$2"
ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
IO_ROOT=/home/jiaxuanluo/autoterm_taurus_20261024_io
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
MODEL=/mnt/data2/jiaxuanluo/RASST_release_runs/models/speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FUSED_MOE_GROUPED_TOPK=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
echo "slurm_step=${SLURM_STEP_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
IFS=, read -r -a GRANTED <<< "${CUDA_VISIBLE_DEVICES}"
VLLM_GPUS="${GRANTED[0]},${GRANTED[1]}"
FRAMEWORK_GPUS="${CUDA_VISIBLE_DEVICES}"
if [[ "${#GRANTED[@]}" -ge 3 ]]; then
  FRAMEWORK_GPUS="${GRANTED[2]}"
fi

CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name rasst-qwen3-omni \
  --host 127.0.0.1 --port "${VLLM_PORT}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${VLLM_UTIL:-0.72}" \
  --max-num-seqs 32 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"audio": 16}' \
  --enforce-eager \
  --disable-custom-all-reduce \
  --trust-remote-code \
  > "${IO_ROOT}/logs/vllm_serve_${VLLM_PORT}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  curl -fsS --max-time 3 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && break
  kill -0 "${VLLM_PID}"
  sleep 3
done
curl -fsS --max-time 3 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null

export RASST_BACKEND_KIND=vllm_serve
export RASST_VLLM_SERVE_URL="http://127.0.0.1:${VLLM_PORT}"
CUDA_VISIBLE_DEVICES="${FRAMEWORK_GPUS}" "${ROOT}/code/scripts/taurus_evidence_server.sh" "${PORT}" routed 32 0.72
