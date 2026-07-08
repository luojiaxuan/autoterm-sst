#!/usr/bin/env bash
# Resident Cloudflare Tunnel exposing the local framework demo server for the
# public/live demo. Pairs with scripts/run_taurus_framework_vllm.sh (port 8011).
#
#   setsid nohup bash scripts/run_cloudflared_tunnel.sh \
#     > logs/cloudflared_tunnel.out 2>&1 &
#
# Why cloudflared instead of ngrok for the reviewer-facing demo:
#   * no interstitial "Visit Site" page,
#   * no single-concurrent-session limit (multiple reviewers can connect),
#   * WebSockets (/wss) proxied out of the box.
#
# Modes:
#   1. Quick tunnel (default, no account needed): prints an ephemeral
#      https://<random>.trycloudflare.com URL in the log. Fine for smoke tests;
#      the URL changes on every restart.
#   2. Named tunnel (stable URL — use this for the submission link):
#        cloudflared tunnel login
#        cloudflared tunnel create rasst-demo
#        cloudflared tunnel route dns rasst-demo <your-hostname>
#      then run with:
#        CF_TUNNEL_NAME=rasst-demo CF_HOSTNAME=<your-hostname> \
#          bash scripts/run_cloudflared_tunnel.sh
set -euo pipefail

PORT="${PORT:-8011}"
CF_BIN="${CF_BIN:-cloudflared}"
# Empty CF_TUNNEL_NAME = ephemeral quick tunnel (random trycloudflare.com URL).
CF_TUNNEL_NAME="${CF_TUNNEL_NAME:-}"
CF_HOSTNAME="${CF_HOSTNAME:-}"

if ! command -v "${CF_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] cloudflared not found (looked for '${CF_BIN}')." >&2
  echo "        Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
  echo "        or set CF_BIN=/path/to/cloudflared" >&2
  exit 1
fi

if [ -n "${CF_TUNNEL_NAME}" ]; then
  echo "[INFO] cloudflared $("${CF_BIN}" version 2>/dev/null | head -1)"
  echo "[INFO] named tunnel '${CF_TUNNEL_NAME}'${CF_HOSTNAME:+ (https://${CF_HOSTNAME})} -> http://127.0.0.1:${PORT}"
  exec "${CF_BIN}" tunnel run --url "http://127.0.0.1:${PORT}" "${CF_TUNNEL_NAME}"
else
  echo "[INFO] cloudflared $("${CF_BIN}" version 2>/dev/null | head -1)"
  echo "[INFO] quick tunnel -> http://127.0.0.1:${PORT} (ephemeral URL printed below)"
  exec "${CF_BIN}" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate
fi
