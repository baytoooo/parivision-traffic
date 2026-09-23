#!/usr/bin/env bash
# Lay out the Hugging Face Space repo from this repository.
#
#   demo/build_space.sh [out_dir]            full tree, for a git push to the Space (default out/space)
#   demo/build_space.sh --archive [out_dir]  three files for the Space's web upload page (default out/space_upload):
#                                            Dockerfile, README.md and space.tar.gz with everything else
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE=0
if [ "${1:-}" = "--archive" ]; then ARCHIVE=1; shift; fi

TREE="$(mktemp -d)"
trap 'rm -rf "$TREE"' EXIT
mkdir -p "$TREE/demo" "$TREE/weights"
cp -R "$ROOT/src" "$TREE/src"
find "$TREE/src" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp "$ROOT/demo/app.py" "$ROOT/demo/worker.py" "$ROOT/demo/requirements.txt" "$TREE/demo/"
cp "$ROOT/weights/yolo26s.pt" "$TREE/weights/"  # the CPU profile (pipeline.PROFILES)
if [ -d "$ROOT/demo/samples" ]; then cp -R "$ROOT/demo/samples" "$TREE/demo/samples"; fi

if [ "$ARCHIVE" = 1 ]; then
  # the web upload takes files under 10 MB: see demo/Dockerfile.upload
  OUT="${1:-$ROOT/out/space_upload}"
  rm -rf "$OUT" && mkdir -p "$OUT"
  rm "$TREE/weights/yolo26s.pt"
  COPYFILE_DISABLE=1 tar -czf - -C "$TREE" . | split -b 9m - "$OUT/space.tar.gz.part-"
  cp "$ROOT/demo/space_README.md" "$OUT/README.md"
  cp "$ROOT/demo/Dockerfile.upload" "$OUT/Dockerfile"
else
  OUT="${1:-$ROOT/out/space}"
  rm -rf "$OUT" && mkdir -p "$OUT"
  cp -R "$TREE/." "$OUT/"
  cp "$ROOT/demo/Dockerfile" "$OUT/Dockerfile"
  cp "$ROOT/demo/space_README.md" "$OUT/README.md"
fi
echo "Space laid out in $OUT"
