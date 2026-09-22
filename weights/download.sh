#!/usr/bin/env bash
# The weights are committed to the repository; this only restores them if they are missing.
set -euo pipefail
cd "$(dirname "$0")"
for m in yolo26n yolo26s yolo26m; do
  [ -f "$m.pt" ] || curl -L --fail -o "$m.pt" "https://github.com/ultralytics/assets/releases/download/v8.4.0/$m.pt"
done
ls -la *.pt
