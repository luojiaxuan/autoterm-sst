#!/usr/bin/env bash
# LEGACY launcher: standalone RASST / SGLang-Omni demo server
# (serve.rasst_sglang_server).
#
# Superseded by the thin framework (`start_demo.sh` -> `python -m framework.server`),
# which serves RASST through `framework.agents.omni.OmniAgent`. Kept for
# direct/standalone use and A/B comparison. All tuning is via the server's own
# env-driven argparse defaults (RASST_SGLANG_BASE_URL, RASST_DEMO_LANGUAGE_PAIR,
# RASST_RAG_*, ...); extra CLI flags are passed through.
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/rasst-demo"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

echo "[legacy] Starting RASST SGLang-Omni server (serve.rasst_sglang_server)"
echo "  host: ${HOST}"
echo "  port: ${PORT}"
echo "  sglang base url: ${RASST_SGLANG_BASE_URL:-http://127.0.0.1:8100}"
echo "  language pair: ${RASST_DEMO_LANGUAGE_PAIR:-English -> Chinese}"

exec "${PYTHON_BIN}" -m serve.rasst_sglang_server --host "${HOST}" --port "${PORT}" "$@"
