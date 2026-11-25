#!/usr/bin/env bash
#SBATCH --job-name=rasst-framework
#SBATCH --partition=aries
#SBATCH --nodelist=aries
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/framework_vllm_aries_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/framework_vllm_aries_%j.err
#
# Run the THIN FRAMEWORK (framework.server) — NOT the legacy serve.rasst_server —
# with the real RASST Qwen3-Omni agent (in-process vLLM tp=2 + MaxSim RAG) on the
# aries node via SLURM. This is the SLURM counterpart of
# scripts/run_taurus_framework_vllm.sh (which pins free GPUs + setsid/nohup on the
# taurus login node); here SLURM allocates the GPUs via --gres.
#
#   sbatch scripts/slurm_framework_vllm_aries.sh
#   curl -s http://<aries>:8011/health | python -m json.tool   # after model load
set -euo pipefail

export PORT="${PORT:-8011}"
export HOST="${HOST:-0.0.0.0}"

# SLURM sets CUDA_VISIBLE_DEVICES from --gres; the framework launcher respects an
# already-set value (its 6,7,5 default only applies on the bare taurus node).
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-none} node=$(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

exec bash /mnt/taurus/home/jiaxuanluo/rasst-demo/scripts/run_taurus_framework_vllm.sh
