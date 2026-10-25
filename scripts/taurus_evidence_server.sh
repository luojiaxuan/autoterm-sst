#!/usr/bin/env bash
# Serve the RASST agent for the ten-talk evidence runs on Taurus.
# Usage: taurus_evidence_server.sh <port> <routed|fixed> <max_num_seqs> <gpu_memory_utilization>
# Run it inside a Slurm step that was granted two GPUs; the retriever shares the first one.
set -euo pipefail

PORT="$1"
MODE="$2"
MAX_SEQS="$3"
GPU_UTIL="$4"

ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
CODE="${ROOT}/code"
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
MODEL=/mnt/data2/jiaxuanluo/RASST_release_runs/models/speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf
RETRIEVER=/mnt/data1/jiaxuanluo/rasst_eval/autoterm_capacity_100topics_20260710/models/rasst-hn1024.pt
MANIFEST="${ROOT}/slices/manifest_taurus_1k.json"
SLICES=nlp_core_1k,medicine_core_1k,education_core_1k,finance_core_1k,legal_core_1k,environment_core_1k,entertainment_core_1k,science_core_1k,sports_core_1k,art_core_1k

case "${MODE}" in
  routed)
    POLICY_ARGS=(
      --required-presets "${SLICES}"
      --autoterm-policy budgeted_top_slices
      --autoterm-default-preset nlp_core_1k
      --autoterm-term-budget 4000
      --autoterm-max-active-slices 4
      --autoterm-context-window-chunks 4
      --autoterm-context-model /home/jiaxuanluo/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
      --autoterm-context-device cpu
      --autoterm-context-weight 1.0
      --autoterm-text-weight 0.0
      --autoterm-domain-probe-weight 0.0
      --autoterm-speech-weight 0.0
      --autoterm-metadata-weight 0.0
      --autoterm-ema-alpha 0.80
      --autoterm-update-sec 5.0
      --autoterm-warmup-sec 6.0
      --autoterm-cooldown-sec 30.0
    )
    ;;
  fixed)
    POLICY_ARGS=(
      --required-presets merged_10x1k,merged_nlp_med_2k,nlp_core_1k,medicine_core_1k
      --autoterm-policy fixed
    )
    ;;
  *)
    echo "unknown mode ${MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${ROOT}/tmp/server_${PORT}"
cd "${CODE}"
echo "slurm_step=${SLURM_STEP_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export HF_HUB_OFFLINE=1
exec "${PYTHON_BIN}" -u eval/streaming_sst/serve_glossary_capacity_sweep.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model-path "${MODEL}" \
  --manifest "${MANIFEST}" \
  --rag-model-path "${RETRIEVER}" \
  --rag-device cuda:0 \
  "${POLICY_ARGS[@]}" \
  --vllm-tp-size 2 \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs "${MAX_SEQS}" \
  --max-model-len 16384 \
  --vllm-limit-audio 16 \
  --vllm-enforce-eager 1 \
  --enable-prefix-caching 1 \
  --disable-custom-all-reduce 1 \
  --vllm-use-v1 1 \
  --vllm-enable-v1-multiprocessing 1 \
  --vllm-worker-multiproc-method spawn \
  --vllm-moe-use-deep-gemm 0 \
  --vllm-use-fused-moe-grouped-topk 0 \
  --nccl-p2p-disable 1 \
  --nccl-ib-disable 1 \
  --torch-nccl-enable-monitoring 0 \
  --extra-python-path /mnt/data2/jiaxuanluo/RASST/code/rasst/retriever \
  --scheduler-batch-size "${MAX_SEQS}" \
  --max-inflight-batches 2 \
  --max-new-tokens 40 \
  --rag-top-k 10 \
  --rag-score-threshold 0.78 \
  --retrieval-candidate-budget 100 \
  --term-map-format tagged \
  --empty-term-map-policy none_block \
  --tmp-dir "${ROOT}/tmp/server_${PORT}" \
  --log-level info
