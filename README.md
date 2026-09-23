# PariVision: traffic events and accident anticipation from one fixed camera

WIUT Hackathon 2026, Computer Vision track, elimination task. Team PariVision
(Webster University in Tashkent): Amal Karimov, Komronbek Qodirov, Aziza Adizova.

Website with the live demo, EDA and results: <!-- SITE_URL -->

For a 4K clip of the Tashkent junction the organisers filmed, `solution.py` returns

* **Part A**: traffic events as `[start_sec, end_sec, label]` segments, and
* **Part B**: a causal per-frame probability that an accident starts within 5 s.

## Run it

Python 3.10 or newer. On the evaluation machine (NVIDIA GPU, no internet):

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

The model weights are in `weights/` (70 MB, committed to the repository), so
nothing is downloaded at run time. `weights/download.sh` only exists to
re-fetch the same files from Ultralytics if they ever go missing.

`run_submission.py` and `evaluate.py` are the organisers' files, unchanged
(sha256 in `docs/starter_kit.sha256`).

`requirements.txt` pins `torch==2.6.0`, whose PyPI wheel carries CUDA 12.4
kernels. They run on a T4 with any NVIDIA driver from 525 on. If neither CUDA
nor Apple MPS is available the pipeline switches to a lighter CPU profile
(Part A: YOLO26s at 960 px, 5 frames per second; Part B: YOLO26n at 640 px)
so that it still finishes inside the time budget, with lower accuracy.

To reproduce `predictions_samples.json` put the four sample clips in
`samples/` and run the same command with `--videos samples`.

## How it works

```
4K frame ─► every 3rd frame, decoded straight to 1920 px (PyAV, background thread)
          ├─► YOLO26m @1280 (COCO) ─► ByteTrack, separate trackers for vehicles / people / bicycles / animals
          ├─► vehicle signal phase (3-lamp head on the median, per-lamp colour contrast)
          └─► homography to the reference view (SIFT, day and dusk references; again on keyframes)
trajectories in reference coordinates + signal phase + scene layout
          └─► one rule per class ─► per-actor intervals ─► union per class ─► segments
```

**Learned** (pretrained, not trained by us): the YOLO26 detectors, COCO
weights from Ultralytics. **Rule-based**: everything after tracking. We did not
fine-tune anything: the organisers gave no labels, and 18.4 minutes of our own
dev labels are enough to tune thresholds, not to train a model without
overfitting to four clips.

**Data-derived, not hand-drawn**: the drivable area (`src/parivision/assets/drivable.png`,
from where moving vehicles were seen in the samples) and the metres-per-pixel
map (`scene.PERSON_HEIGHT_PX`, fitted on ~150k pedestrian boxes).

### Part A, class by class

| class | rule |
|---|---|
| jaywalking | a pedestrian (not a cyclist, not someone seen through a car window) on the drivable area outside every zebra, by at least 0.35 of their own height, for 1 s or more |
| failure_to_yield | a car, bus or truck drives across a zebra while a walking pedestrian is out on the same zebra (not waiting at the kerb) within 160 px of it (about 3 to 3.5 m on the north crossing) |
| red_light | a southbound vehicle's front crosses the stop line after the vehicle signal has been red for 1 s, and at least 1.5 s before it turns green |
| stop_line | a southbound vehicle stands still with its front past the stop line, between the line and the far side of the north crossing, while the signal is red |
| stopped_vehicle | a vehicle stands still 10 s or more on the northbound carriageway (not at the bus stop or at the right edge of the frame) or in the junction box; the southbound approach, where the red-light queue stands, does not count |
| wrong_way | a vehicle or bike moves against the lane direction on either carriageway for 1.5 s or more |
| congestion | southbound traffic stands still while it has green: at least 8 s into the green, 8 or more vehicles stand on the last stretch of the approach and past the stop line (or 5 past the stop line alone) for 6 s or more; it carries on into the red while 5 or more still stand past the stop line |
| illegal_u_turn | detected (SB traffic round the median nose into NB) and shown on the website, not submitted: nothing in view says these U-turns are prohibited, and a predicted class the test set lacks costs a zero in the macro average |
| others | no rule: accident, near_miss, illegal_turn, solid_line_crossing, road_obstacle, fire_smoke |

Segments of one class are merged when they overlap, as the task asks, and
when the gap between them is short (0 to 3 s depending on the class, 8 s for
congestion).
Every rule except congestion keeps the ids of the road users involved, and
the renderer uses them to draw who did what.

### Part B

`RiskEstimator.step` sees frames only in order. Every third frame (10 Hz) is
decimated to 1280 px and goes through YOLO26s @960 and the same tracker setup.
For each pair of road users on the carriageway it predicts constant-velocity
motion for 3 s (in metres, from the fitted scale map) and scores how soon and
how deeply their footprints would overlap. Hard braking raises the score. A
pair has to look dangerous for 0.6 s without a break, and pairs on opposite
sides of the median or a moving car next to a parked one are ignored. The score
is the worst pair, smoothed. If the machine is slow, `step` thins out the
frames it processes, down to 1 Hz, to keep its own work under 0.4x the clip
length and Parts A and B together under 2.8x (the harness allows 3x). Past
95% of that budget it stops processing and holds the last score.

### Things that shaped the design

* The four samples are not framed identically. The two afternoon clips are
  shifted by up to 60 px and scaled by about 2%, so fixed pixel polygons
  would have been wrong on them. Each clip is registered to a reference view.
  C3896's camera also drifts 9 px over its first 40 s, as a camera settling on
  its tripod does, so the view is registered again on keyframes (every 2 s for
  the first 30 s, then every 10 s): the signal lamps are read with the latest
  registration, and the rules use one registration of the keyframes' median.
* The signal head that faces the camera on the left pole is a pedestrian
  signal for the west crossing. The vehicle phase comes from the three-lamp
  head on the median nose: 36 s green (the last 3 s flashing), 3 s yellow,
  36 s red in the morning clips (a 75 s cycle), and 38 s green, 3 s yellow,
  39 s red in the afternoon ones (80 s).
* Kerbs, the bus stop and the median edge are where most pedestrians stand,
  and none of them is the carriageway. The walkable-road mask is built from
  where vehicles really drive, plus the junction box, and a 30 px band along
  the far kerb is excluded.

Details: `docs/scene.md` (our description of the scene, in place of the
missing `camera.md`) and `docs/labeling.md` (how we built the dev set).

## Results on our dev labels

Scored with the organisers' `evaluate.py` against our own labels of the four
samples (104 events; `labels/dev_labels.json`, built as `docs/labeling.md`
describes). F1 is the mean over tIoU 0.3, 0.5 and 0.7.

| class | F1 | TP / FP / FN at tIoU 0.5 |
|---|---:|---:|
| congestion | 0.800 | 2 / 0 / 1 |
| failure_to_yield | 0.518 | 27 / 28 / 21 |
| jaywalking | 0.538 | 12 / 12 / 16 |
| red_light | 0.667 | 1 / 0 / 1 |
| stop_line | 0.769 | 5 / 3 / 0 |
| stopped_vehicle | 0.905 | 6 / 1 / 1 |
| illegal_u_turn (not submitted) | 0 | 0 / 0 / 9 |
| illegal_turn (no rule) | 0 | 0 / 0 / 2 |

Score A is **0.525**; the mean over the six classes we emit is 0.70. The labels
were made by AI agents and checked by other agents, not by people, so treat
these numbers as indicative. The samples contain no accidents, so Part B is
not scored; its risk score crosses 0.5 once in the four clips, for 0.7 s
(C3905 at 57.6 s, a dense platoon of cars crossing the junction box side by
side; nothing happens).
`python evaluate.py --pred predictions_samples.json --gt labels/dev_labels.json --per-video`
prints the full report.

## Runtime

`predictions_samples.json` comes from the official harness on an Apple M5
laptop (16 GB, PyTorch on MPS). The laptop is slower than a T4, so for that
run we lifted the time guards (`PARIVISION_TIME_SHARE=12
PARIVISION_TOTAL_LIMIT=30 PARIVISION_RISK_SHARE=10`, `--time-factor 40`) to get
the output of the full pipeline, with no frames thinned out:

| clip | length | Part A | Part B | total |
|---|---:|---:|---:|---:|
| C3896 | 340 s | 595 s | 491 s | 3.2x |
| C3897 | 318 s | 919 s | 437 s | 4.3x |
| C3902 | 318 s | 540 s | 317 s | 2.7x |
| C3905 | 128 s | 200 s | 118 s | 2.5x |

With the default guards the pipeline always stays inside the harness budget:
Part A thins its frames and stops at 1.3x the clip length, and Part B keeps
both parts under 2.8x. `tools/t4_check.sh` runs the harness with the official
3x budget on a Colab or Kaggle T4 and prints the same table.

## Determinism

`src/parivision/seed.py` fixes the Python, NumPy and PyTorch seeds and turns
off cuDNN autotuning. Detection, tracking and the rules have no random steps,
so two runs on the same machine give the same `predictions.json`. The one
exception is floating-point noise from GPU fp16 inference, which can move a
box by a fraction of a pixel. Part A also has a wall-clock guard: if it falls
behind it analyses only every 2nd, then every 3rd sampled frame (down to
3.3 fps), and if it still passes 1.3x the clip length (or 60 s, whichever is
longer) it stops and reports what it has. That changes the output only on a
machine far slower than the target. The defaults fit the T4. On a slower
machine `PARIVISION_TIME_SHARE` (Part A, 1.3), `PARIVISION_RISK_SHARE`
(Part B's own work, 0.4) and `PARIVISION_TOTAL_LIMIT` (both parts, 2.8)
override these limits. `PARIVISION_CACHE_DIR` makes Part A save its full
analysis for the website; it is unset in the official run.

## Repository layout

```
solution.py            the interface: detect_events, RiskEstimator
run_submission.py      organisers' harness (unchanged)
evaluate.py            organisers' metric (unchanged)
src/parivision/        pipeline: video, detector, tracking, registration, signal, rules, risk, render
weights/               YOLO26 n/s/m COCO weights
labels/dev_labels.json our labels of the four sample clips (dev set)
tools/                 caching, EDA, dev-set labelling, tuning and site-data scripts; t4_check.sh times a run on a T4
tests/                 pytest checks for the core pieces and the solution interface
demo/                  FastAPI server that runs the full Python pipeline on a clip, for trying it locally (the website demo runs site/src/pipeline/ in the browser)
site/                  the team website (Astro)
docs/                  scene description, labelling guide, figures
predictions_samples.json  our output on the sample clips
LICENSE                AGPL-3.0
```

## Datasets, models and licences

* **YOLO26 n/s/m** detection weights, pretrained on COCO, from Ultralytics.
  Licence: AGPL-3.0. We use them as they are.
* **COCO 2017** (through those weights). Annotations: CC BY 4.0.
* **Sample clips** from the organisers. Used for scene analysis, our dev
  labels and tuning. The clips are not in this repository. The website shows
  annotated renders of them, as the task asks, and three 30 s cuts for the
  in-browser demo.
* No other datasets are used.

Open-source code we build on: Ultralytics (AGPL-3.0; YOLO inference and the
ByteTrack implementation), OpenCV (Apache-2.0), PyAV (BSD-3-Clause; its
wheels bundle an FFmpeg build that includes x264 and x265, which are GPL),
NumPy, SciPy (BSD). Because we ship Ultralytics weights and call its code, this
repository is released under AGPL-3.0 (`LICENSE`).

## Team

| member | role | did |
|---|---|---|
| Amal Karimov (captain) | scene layout, signal and vehicle rules | traced the junction in the reference view; found the vehicle signal head and wrote its phase reader; red_light, stop_line, stopped_vehicle and congestion rules |
| Komronbek Qodirov | pipeline, live demo, website | video decoding, detection, tracking, registration and the time budget; Part B; the in-browser port of the pipeline and the website |
| Aziza Adizova | dev set, pedestrian rules, report | the labelling guide, the labelling runs and the review of disputed events; jaywalking and failure_to_yield rules and their error analysis; EDA and the report |
