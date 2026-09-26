# Dev tools

The submission (`solution.py`, `run_submission.py`) imports nothing from this
folder. These scripts built the caches, labels, figures and website data, and
the committed `src/parivision/assets/drivable.png`, which the submission reads.
They need `pip install -r requirements.txt -r tools/requirements-dev.txt`, and
`ffmpeg` and `ffprobe` on `PATH`. The README's "Reproduce everything" section
runs them in order. Each script's docstring has its full usage.

`cache/`, `out/`, `samples/` and `external/` are gitignored, so everything the
tables below list under them has to be rebuilt on a fresh clone.

## Caches from the sample clips

| tool | what it does | reads | writes | submission needs it |
|---|---|---|---|---|
| `cache_detections.py` | runs one detector over the clips and stores every box | `samples/*.MP4`, `weights/` | `cache/det/<clip>__<weights>_<imgsz>_<width>_<fps>fps.npz` | no |
| `align_cache.py` | homography of each clip to the reference view, from a median background (its docstring has the ffmpeg command for the proxies) | `cache/proxy/<clip>.mp4` | `cache/align/<clip>.npz`, `<clip>_bg_ref.jpg` | no |
| `tracks_from_cache.py` | tracks cached detections into trajectories in the reference view | `cache/det`, `cache/align` | `cache/tracks/<clip>.pkl` | no |
| `signal_timeline.py` | vehicle signal phase every 0.2 s, with the lamp scores | `samples/*.MP4`, `cache/align` | `cache/signal/<clip>.json` | no |
| `signal_rederive.py` | phases again from the stored lamp scores, after a threshold change, without decoding video | `cache/signal` | `cache/signal` (rewritten) | no |
| `drivable_mask.py` | where moving vehicles were seen, as a mask; `--check` compares with the committed one | `cache/tracks` | `src/parivision/assets/drivable.png` | its output, yes |

## Rules, tuning and inspection

| tool | what it does | reads | writes | submission needs it |
|---|---|---|---|---|
| `run_rules.py` | runs the rules on cached trajectories (the fast tuning loop); its `load_context` is shared by other tools | `cache/tracks`, `cache/signal` | `out/pred_dev.json` | no |
| `tune.py` | grid search of one class's rule parameters against the dev labels | the same, `labels/dev_labels.json` | prints | no |
| `ablation.py` | detector, frame rate and design ablations, scored with `evaluate.py` | `cache/det` (several detectors), `cache/align`, `cache/signal`, `labels/dev_labels.json` | `out/ablations.json` | no |
| `inspect_evidence.py` | prints which road users made an event and draws them on proxy frames | `cache/tracks`, `cache/signal`, `cache/proxy` | `out/inspect/*.jpg` | no |
| `render_clip.py` | annotated MP4 of a sample clip from cached trajectories | `samples/*.MP4`, `cache/tracks`, `cache/signal` | `out/render/<clip>.mp4` | no |
| `sheet.py` | contact sheet of proxy frames, for labelling and checking events by eye | `cache/proxy` | `out/sheets/*.jpg` | no |
| `probe.py` | metadata of any video and evenly spaced frames, for a first look at a new clip | video files | `out/probe/<clip>/*.jpg` | no |
| `compare_predictions.py` | events per class and tIoU matches between two prediction files, and the largest risk difference | two prediction files | prints | no |

## Dev labels

| tool | what it does | reads | writes | submission needs it |
|---|---|---|---|---|
| `workflows/label_dev_set.js` | Claude Code workflow: labeller and verifier agents per 40 s window | `docs/scene.md`, `docs/labeling.md`, `cache/proxy` (through `sheet.py`) | a journal of claims and verdicts | no |
| `disagreements.py` | lists where the rules and the dev labels disagree | `cache/tracks`, `cache/signal`, `labels/dev_labels.json` | `out/adjudicate_items.json` | no |
| `workflows/adjudicate.js` | Claude Code workflow: one agent per disagreement decides who is right | `out/adjudicate_items.json` | `labels/adjudication.json` | no |
| `build_dev_labels.py` | builds the dev labels from the journals, or from `labels/dev_labels_verified.json`, plus the adjudication | journals or `labels/dev_labels_verified.json`, `labels/adjudication.json` | `labels/dev_labels.json` (and `dev_labels_rejected.json` from journals) | no |

`labels/dev_labels_verified.json` holds the 107 verified events before
adjudication. The journals are not in the repository, but this command rebuilds
the committed `labels/dev_labels.json` byte for byte:

```bash
python tools/build_dev_labels.py --verified labels/dev_labels_verified.json \
  --adjudication labels/adjudication.json --out labels/dev_labels.json
```

## Website and in-browser demo

| tool | what it does | reads | writes | submission needs it |
|---|---|---|---|---|
| `eda.py` | EDA figures and numbers | `cache/det`, `cache/tracks`, `cache/signal`, `cache/proxy`, `samples/*.MP4` (ffprobe) | `out/site_data/eda.json`, `out/site_data/media/eda/*.jpg` | no |
| `make_site_data.py` | the site's data files, annotated videos, posters, example frames and home-page loop | a prediction file (`--pred`), `out/analysis/*.pkl`, `samples/*.MP4`, `labels/dev_labels.json`, `out/metrics.json`, `out/ablations.json`, `out/site_data` | `site/public/data/`, `site/public/media/` | no |
| `export_browser_assets.py` | the detector as ONNX, and the scene rasters and constants, for the browser pipeline | `weights/yolo26n.pt`, `src/parivision/` | `site/public/pipeline/` | no |
| `export_parity_fixtures.py` | each Python stage's input and output for the site's parity tests, plus gzip copies | `cache/det`, `cache/align`, `cache/signal`, `samples/*.MP4`, `site/public/pipeline/` | `site/tests/fixtures/<clip>/` | no |

## Crash footage and timing

| tool | what it does | reads | writes | submission needs it |
|---|---|---|---|---|
| `crash_check.py` | accident rule and Part B on ACCIDENT benchmark clips: `fetch`, `cache`, `eval` | `labels/accident_*.csv`, Kaggle, `out/analysis/C3896.pkl`, `predictions_samples.json` | `external/accident/videos/`, `cache/ext/`, `out/crash_check_<set>.json` | no |
| `risk_from_cache.py` | replays cached detections through the Part B model and prints its alarms, to calibrate it without decoding video | `cache/det`, `cache/align` | prints | no |
| `t4_check.sh` | on a T4 (Colab or Kaggle): installs, downloads the samples, runs the harness with the official 3x budget and prints the timings | Google Drive | `samples/`, `predictions_t4.json` | no |

## Shared

| file | what it is |
|---|---|
| `common.py` | `ROOT` and `SAMPLES` (the clips folder, `samples/` or `PARIVISION_SAMPLES`) |
| `requirements-dev.txt` | what the tests and these tools need beyond `requirements.txt` |
