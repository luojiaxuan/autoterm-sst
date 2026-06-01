#!/bin/bash

WATCH_DIR=./
REMOTE_PATH=jiaxuanluo@aries.cs.ucsb.edu:/mnt/data6/jiaxuanluo/new-infinisst/

echo "📡 Watching $WATCH_DIR for .py and .sh changes..."

fswatch -r \
  --exclude=".*\.swp$" \
  --exclude=".*~$" \
  --exclude=".*/\.idea/.*" \
  --exclude=".*/\.git/.*" \
  --exclude=".*\.pyc$" \
  --exclude=".*/node_modules/.*" \
  "$WATCH_DIR" | while read file
do
    echo "🔁 Detected change: $file"
    rsync -az -e "ssh" \
      --include='*/' \
      --include='*.py' \
      --include='*.sh' \
      --include='*.html' \
      --include='*.js' \
      --include='*.css' \
      --include='*.json' \
      --exclude='*' \
      "$WATCH_DIR/" "$REMOTE_PATH"
done