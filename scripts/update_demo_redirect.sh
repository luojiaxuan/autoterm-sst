#!/usr/bin/env bash
# Update the stable GitHub Pages entry (https://luojiaxuan.github.io/autoterm-sst/)
# to redirect to the current cloudflared quick-tunnel URL.
# Run from any machine with push access after (re)starting the tunnel:
#   scripts/update_demo_redirect.sh https://<new-subdomain>.trycloudflare.com
set -euo pipefail
URL=${1:?usage: update_demo_redirect.sh <tunnel-url>}
case "$URL" in https://*) ;; *) echo "expected https:// URL" >&2; exit 1;; esac
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 --branch gh-pages https://github.com/luojiaxuan/autoterm-sst.git "$WORK"
cat > "$WORK/index.html" <<HTML
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AutoTerm-SST Live Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="0; url=$URL/">
<script>location.replace("$URL/");</script>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:640px;margin:80px auto;padding:0 24px;color:#1a2433}
a{color:#2757a8}.muted{color:#6b7687;font-size:14px}
</style>
</head>
<body>
<h2>AutoTerm-SST Live Demo</h2>
<p>Redirecting to the live demo&hellip; If nothing happens,
<a href="$URL/">click here</a>.</p>
<p class="muted">If the demo host is being restarted, try again in a minute or use the fallbacks:
<a href="https://github.com/luojiaxuan/autoterm-sst">source repository</a> (mock mode runs without GPUs).</p>
</body>
</html>
HTML
git -C "$WORK" commit -qam "Point demo redirect at $URL"
git -C "$WORK" push -q origin gh-pages
echo "redirect updated -> $URL"
