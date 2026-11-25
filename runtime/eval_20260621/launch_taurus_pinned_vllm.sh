#!/usr/bin/env bash
#SBATCH --job-name=rasst-vllm-pin
#SBATCH --partition=taurus
#SBATCH --nodelist=taurus
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/framework_vllm_pinned_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/framework_vllm_pinned_%j.err
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,5}"
export RASST_TERM_MEMORY_MANIFEST="${RASST_TERM_MEMORY_MANIFEST:-/home/jiaxuanluo/rasst-demo/runtime/eval_20260621/term_memory_combined_manifest.json}"
export RASST_AUTO_GLOSSARY_ENABLED="${RASST_AUTO_GLOSSARY_ENABLED:-1}"
export RASST_AUTO_GLOSSARY_DEFAULT="${RASST_AUTO_GLOSSARY_DEFAULT:-common_10k}"
export RASST_AUTO_GLOSSARY_PRESETS="${RASST_AUTO_GLOSSARY_PRESETS:-common_10k,nlp_core_10k,medicine_core_10k}"
export RASST_AUTO_GLOSSARY_PRELOAD_PRESETS="${RASST_AUTO_GLOSSARY_PRELOAD_PRESETS:-common_10k,nlp_core_10k,medicine_core_10k}"
export RASST_PROMPT_TOP_K="${RASST_PROMPT_TOP_K:-10}"
export RASST_UI_TOP_K="${RASST_UI_TOP_K:-10}"
export PORT="${PORT:-8011}"
export HOST="${HOST:-0.0.0.0}"

exec bash /mnt/taurus/home/jiaxuanluo/rasst-demo/scripts/run_taurus_framework_vllm.sh
