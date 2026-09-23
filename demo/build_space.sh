#!/usr/bin/env bash
# Lay out the Hugging Face Space repo in $1 (default: out/space) from this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/out/space}"
rm -rf "$OUT" && mkdir -p "$OUT/demo" "$OUT/weights"
cp -R "$ROOT/src" "$OUT/src"
find "$OUT/src" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp "$ROOT/demo/app.py" "$ROOT/demo/worker.py" "$ROOT/demo/requirements.txt" "$OUT/demo/"
cp "$ROOT/demo/Dockerfile" "$OUT/Dockerfile"
cp "$ROOT/demo/space_README.md" "$OUT/README.md"
cp "$ROOT/weights/yolo26s.pt" "$OUT/weights/"  # the CPU profile (pipeline.PROFILES)
if [ -d "$ROOT/demo/samples" ]; then cp -R "$ROOT/demo/samples" "$OUT/demo/samples"; fi
echo "Space laid out in $OUT"
