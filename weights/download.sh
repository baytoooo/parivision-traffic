#!/usr/bin/env bash
# The weights are committed to the repository; this only restores them if they are missing,
# then checks every file against SHA256SUMS (the Ultralytics v8.4.0 release files).
set -euo pipefail
cd "$(dirname "$0")"
for m in yolo26n yolo26s yolo26m; do
  [ -f "$m.pt" ] || curl -L --fail -o "$m.pt" "https://github.com/ultralytics/assets/releases/download/v8.4.0/$m.pt"
done
check() {
  if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
}
check || { echo "A file above does not match SHA256SUMS: delete it and run this script again." >&2; exit 1; }
