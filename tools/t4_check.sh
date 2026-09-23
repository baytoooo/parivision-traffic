#!/usr/bin/env bash
# Timing check on a T4 (Colab or Kaggle), the same GPU the organisers use.
#
#   git clone <repo> && cd <repo> && bash tools/t4_check.sh [drive_file_id ...]
#
# Installs requirements.txt, downloads sample clips from the organisers' public
# Google Drive (all four by default), runs the official harness with the
# official 3x budget and prints its per-clip log (part_a_sec, part_b_sec,
# total_sec). Needs internet for the downloads only; the run itself is offline.
# Clips already in samples/ are not downloaded again. When Drive refuses the
# download ("Too many users have viewed or downloaded this file"), put the
# clips in samples/ some other way first; in Colab, the Drive API after
# google.colab.auth.authenticate_user() still works.
set -euo pipefail
cd "$(dirname "$0")/.."

IDS=("$@")
if [ ${#IDS[@]} -eq 0 ]; then
  IDS=(10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ 1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd
       1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_ 1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS)
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nproc
pip install -q -r requirements.txt gdown

mkdir -p samples
if ! ls samples/*.MP4 >/dev/null 2>&1; then
  for id in "${IDS[@]}"; do
    (cd samples && gdown --continue "$id")
  done
fi
ls -la samples

YOLO_OFFLINE=1 python run_submission.py --videos samples --out predictions_t4.json --team PariVision
python evaluate.py --pred predictions_t4.json --validate-only
python - <<'EOF'
import json
log = json.load(open("predictions_t4.json"))["log"]
for clip, r in log.items():
    print(f"{clip}: {r['duration']:.1f}s clip, Part A {r.get('part_a_sec')}s, Part B {r.get('part_b_sec')}s, "
          f"total {r.get('total_sec')}s = {r.get('total_sec', 0) / r['duration']:.2f}x, errors {r['errors']}")
EOF
