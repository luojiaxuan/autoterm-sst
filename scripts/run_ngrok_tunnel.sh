#!/usr/bin/env bash
# Resident ngrok tunnel exposing the local framework demo server for remote
# access. Pairs with scripts/run_taurus_framework_vllm.sh (port 8011).
#
#   setsid nohup bash scripts/run_ngrok_tunnel.sh > logs/ngrok_tunnel.out 2>&1 &
#
# Free-tier ngrok allows only ONE simultaneous agent session per authtoken, so
# stop any other ngrok before starting this one.
set -euo pipefail

PORT="${PORT:-8011}"
# Set NGROK_DOMAIN="" to request an ephemeral (random) URL instead of the
# reserved static domain (useful if the static domain is held by another agent).
NGROK_DOMAIN="${NGROK_DOMAIN-amused-fleet-aardvark.ngrok-free.app}"
NGROK_BIN="${NGROK_BIN:-/mnt/aries/data6/jiaxuanluo/bin/ngrok}"
NGROK_CONFIG="${NGROK_CONFIG:-/mnt/taurus/home/jiaxuanluo/ngrok_jiaxuan.yml}"

if [ -n "${NGROK_DOMAIN}" ]; then
  echo "[INFO] ngrok $("${NGROK_BIN}" version 2>/dev/null) tunneling https://${NGROK_DOMAIN} -> http://127.0.0.1:${PORT}"
  exec "${NGROK_BIN}" http --url="${NGROK_DOMAIN}" --config "${NGROK_CONFIG}" "${PORT}"
else
  echo "[INFO] ngrok $("${NGROK_BIN}" version 2>/dev/null) tunneling EPHEMERAL url -> http://127.0.0.1:${PORT}"
  exec "${NGROK_BIN}" http --config "${NGROK_CONFIG}" "${PORT}"
fi
