# PariVision: traffic events and accident anticipation from one fixed camera

WIUT Hackathon 2026, Computer Vision track, elimination task. Team PariVision
(Webster University in Tashkent): Amal Karimov, Komronbek Qodirov, Aziza Adizova.

Website with the in-browser demo (on a clip, or live on a camera or a shared stream at /live), EDA
and results: https://parivision-traffic.vercel.app

For a 4K clip of the Tashkent junction the organisers filmed, `solution.py` returns

* **Part A**: traffic events as `[start_sec, end_sec, label]` segments, and
* **Part B**: a causal per-frame probability that an accident starts within 5 s.

## Run it

Use Python 3.11 or 3.12. On the evaluation machine (NVIDIA GPU, no internet):

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

The model weights are in `weights/` (70 MB, committed to the repository), so
nothing is downloaded at run time. `weights/download.sh` only exists to
re-fetch the same files from Ultralytics if they ever go missing; it then
checks all three against `weights/SHA256SUMS`.

`run_submission.py` and `evaluate.py` are the organisers' files, unchanged
(sha256 in `docs/starter_kit.sha256`).

On Python 3.10 to 3.13, where we tested it, `requirements.txt` installs
`torch==2.6.0` and `torchvision==0.21.0`. The torch wheel carries CUDA 12.4
kernels, which run on a T4 with any NVIDIA driver from 525 on, and the T4
timings under Runtime were measured with it on Colab. torch 2.6.0 has no wheel
for Python 3.14, so there `requirements.txt` installs torch 2.9.1 and
torchvision 0.24.1 instead. We only smoke-tested that combination on a Mac; it
has not been timed on a T4. Ultralytics comes as `ultralytics-opencv-headless`,
the same package built against `opencv-python-headless`, so only one OpenCV
gets installed and it needs no libGL. If neither CUDA nor Apple MPS is
available the pipeline switches to a lighter CPU profile (Part A: YOLO26s at
960 px, 5 frames per second; Part B: YOLO26n at 640 px) that aims to finish
inside the time budget, at lower accuracy.

To check `predictions_samples.json`, put the four sample clips in `samples/`
(Reproduce everything below shows how to download them), run the harness into
a new file and compare the two:

```bash
PARIVISION_TIME_SHARE=12 PARIVISION_TOTAL_LIMIT=30 PARIVISION_RISK_SHARE=10 \
  python run_submission.py --videos samples --out predictions_samples_rerun.json --team PariVision --time-factor 40
python tools/compare_predictions.py predictions_samples.json predictions_samples_rerun.json
```

The three variables and `--time-factor 40` lift the time guards, so nothing is
thinned out (see Runtime). `tools/compare_predictions.py` prints, for each
clip, the events per class in both files, how many of them pair up at tIoU 0.3,
0.5 and 0.7 (with `evaluate.py`'s own matching) and the largest difference
between the two risk curves. We made `predictions_samples.json` on an Apple M5
laptop with PyTorch 2.14 on MPS, where the detector runs in fp32. On a CUDA GPU
it runs in fp16, so boxes and a few event boundaries can differ slightly: the
same command on a Colab T4 gave 24, 26 and 18 events on C3896, C3902 and C3905,
against 25, 26 and 18 in this file (C3897 could not be downloaded there because
of a Google Drive quota). (The pinned torch 2.6.0 is for CUDA. On MPS it is
several times slower, slow enough that Ultralytics' NMS time limit drops some
boxes, so on a Mac use a newer torch.)

`predictions_samples.json` was made at commit 98be7bd. Later commits changed
the code in `src/` (among other things the accident rule and the time guards)
and `requirements.txt`, but not the events on the four samples: running the
current rules on the analyses that run saved (`out/analysis`, not in the
repository) gives the same events on all four clips, and the accident rule
fires on none of them. `tests/test_regression.py` repeats this check for
C3905 on every `pytest` run, from its saved trajectories in `tests/data/`.

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
| accident | two road users meet at speed (closing at 3 m/s or more, the faster one doing 3 m/s or more), both velocities change at the contact, and both then stand together for 2 s; the event runs from the contact until they stand |
| others | no rule: near_miss, illegal_turn, solid_line_crossing, road_obstacle, fire_smoke |

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
frames it processes, down to 1 Hz, aiming to keep its own work under 0.4x the
clip length and Parts A and B together under 2.8x (the harness allows 3x). The
harness still decodes every frame it hands to `step`, and that time is outside
our control, so `step` measures it (the time between its calls) and stops
processing, holding the last score, as soon as the frames still to come would
take the pass past 95% of that budget. If the first frame does not register
(a dark or blank start), it tries again every 2 s.

### Things that shaped the design

* The four samples are not framed identically. The two afternoon clips are
  shifted by up to about 60 px and scaled by 1 to 2%, so fixed pixel polygons
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

Score A is **0.525**. We emit eight classes, and the mean F1 over the six that
fired on the samples is 0.70 (wrong_way and accident never fired and are not in
our labels, so `evaluate.py` leaves them out). The dev labels are committed in `labels/`. We
drafted them with Claude agents (a hosted model, used only to build the dev
set; `solution.py` never calls it), and other agents checked them; no person
labelled frames, so treat these numbers as indicative. The agent logs are not
in the repository, so the labels cannot be regenerated from it. The samples
contain no accidents, so Part B is not scored; its risk score crosses 0.5 once
in the four clips, for 0.7 s (C3905 at 57.6 s, a dense platoon of cars
crossing the junction box side by side; nothing happens).
`python evaluate.py --pred predictions_samples.json --gt labels/dev_labels.json --per-video`
prints the full report.

## Checking on crash footage

The samples have no crash, so we checked the accident rule and the Part B risk
model on the public ACCIDENT benchmark (CVPR 2026), whose clips come with the
moment of impact annotated. `labels/accident_real.csv` lists the 130 real CCTV
crash clips we picked (118 of them at intersections) and
`labels/accident_synthetic.csv` 100 of its CARLA crash clips;
`tools/crash_check.py` downloads, runs and scores them. The numbers below cover
the clips we had run the detectors on: 34 of the 130 real ones and 95 of the
100 synthetic ones (`crash_check.py eval` skips clips without a detection cache
and prints which). These are other cameras, so the check runs without our scene
layout and takes metres per pixel from the size of the vehicles. The rule finds
24 of the 95 synthetic crashes and 1 of the 34 real ones, starting between 1.5
s before and 2 s after the impact. The benchmark rates 25 of these 34 real
clips as poor quality, and in most of them the detector misses the striking
car. The rule fired early, more than 1.5 s before the impact, on 1 of the 34
real clips (8uCJX3Qp78g_00, a t-bone crash it then missed) and on none of the
synthetic ones. It fires nowhere in our 18 minutes of normal traffic, which is
why we submit the class: if the test set has no crash, a class we never predict
costs nothing. Part B raises an alarm in the 10 s before 11 of the synthetic
and 4 of the real impacts. Raising its gain catches more crashes but also sets
off many alarms in normal traffic, so we left it as it was.

## Runtime

`predictions_samples.json` comes from the official harness on an Apple M5
laptop (16 GB, PyTorch on MPS). For that run we lifted the time guards
(`PARIVISION_TIME_SHARE=12 PARIVISION_TOTAL_LIMIT=30
PARIVISION_RISK_SHARE=10`, `--time-factor 40`) to get the output of the full
pipeline, with no frames thinned out:

| clip | length | Part A | Part B | total |
|---|---:|---:|---:|---:|
| C3896 | 340 s | 595 s | 491 s | 3.2x |
| C3897 | 318 s | 919 s | 437 s | 4.3x |
| C3902 | 318 s | 540 s | 317 s | 2.7x |
| C3905 | 128 s | 200 s | 118 s | 2.5x |

**On a T4.** We ran the harness on a Colab T4 (2 vCPUs, torch 2.6.0+cu124,
installed from `requirements.txt`). The GPU work is small: YOLO26m at 1280 px
takes 34 ms a frame in fp16, 0.34x real time at 10 fps, and Part B's YOLO26s
at 960 px 19 ms, 0.19x. Decoding is what costs. These clips are 4K 10-bit
4:2:2 H.264 at 140 Mbit/s, which only a CPU decodes, and Colab's two vCPUs
manage about 12 frames per second in our decoder and about 9 in the harness's
own `cv2.VideoCapture` loop. There the harness alone needs about 3x the clip
length just to read the frames for Part B, so no pipeline fits the budget on
that machine; on C3896 our Part A reached 170 s of 340 s before its 1.3x stop.
A machine with more cores decodes proportionally faster (our M5 laptop reads
these files at about 78 frames per second), and with the default settings
the 60 s cut of C3896 took 1.4x its length on the laptop.

**The guards.** `solution.py` loads both detectors, runs each once and computes
the reference views' features when it is imported. The harness imports it
before it starts timing, so these one-off costs (several seconds on a cold GPU)
no longer come out of the first clip's budget. Part A thins its frames and stops
at 1.3x the clip length, checked on every frame, and keeps 2 s plus 0.03x the
clip free for the rules after the frame loop; and
Part B thins its own work to keep both parts under 2.8x. Because a clip that
goes over the budget scores nothing, Part A first times the harness's own
decoding loop (`cv2.VideoCapture.read`) on 40 frames of the clip, about a
second, and stops early enough to leave the harness 1.2 times that for all the
frames of Part B, plus 0.4x the clip for Part B's own work. (Timing our own
decoder while the detector runs overstated the harness's time about twice on
the M5: a 60 s 4K cut stopped at 31 s with 75 s of budget unused. With the
direct timing it analyses all 60 s and the harness finishes at 2.0x.) On a machine that decodes
a clip in less than about its own length this never binds; on a slow one it
trades the end of the clip for a result that is not empty. On short clips
there is a floor, set so that both parts still fit in 3x: Part A may take up
to 60 s but never more than 1.8x the clip length. Part B gets what Part A left
of the 2.8x and nothing more: a clip over 3x would lose Part A's events too.
On 10 edge-case clips (1 s to 60 s; 4K 10-bit, 1080p, 720p at 30 fps, HEVC,
black frames, another view, a file name with spaces and Cyrillic) the harness
stayed inside the budget on the M5, with other jobs running on it.
`tools/t4_check.sh` runs the harness with the official 3x budget on a T4 and
prints the same table.

## Determinism

`src/parivision/seed.py` fixes the Python, NumPy and PyTorch seeds and turns
off cuDNN autotuning. Detection, tracking and the rules have no random steps,
so two runs on the same machine give the same `predictions.json`. The one
exception is floating-point noise from GPU fp16 inference, which can move a
box by a fraction of a pixel. Part A also has a wall-clock guard: if it falls
behind it analyses only every 2nd, then every 3rd sampled frame (down to 3.3
fps), and if it still passes 1.3x the clip length (on a clip shorter than
about 46 s, 60 s or 1.8x the clip length, whichever is shorter) it stops and
reports what it has. That changes the output on any machine that cannot keep
up. The defaults are set for a T4 on a machine with enough cores to decode 4K
at speed (see Runtime). On a slower machine `PARIVISION_TIME_SHARE` (Part A,
1.3), `PARIVISION_RISK_SHARE` (Part B's own work, 0.4) and
`PARIVISION_TOTAL_LIMIT` (both parts, 2.8) override these limits.
`PARIVISION_CACHE_DIR` makes Part A save its full analysis for the website; it
is unset in the official run.

## Reproduce everything

The official run needs only the Run it section. These steps rebuild everything
else: the dev caches, EDA, ablations, website data, browser assets, test
fixtures and the crash check. They need the packages in
`tools/requirements-dev.txt` and `ffmpeg` and `ffprobe` on `PATH`. Several
steps decode the 4K clips or run a detector over them; step 9 alone took about
an hour on our M5 laptop. `cache/`, `out/` and `samples/` are gitignored.
`tools/README.md` lists every tool with what it reads and writes.

```bash
pip install -r requirements.txt -r tools/requirements-dev.txt

# 1. The four sample clips, from the organisers' public Google Drive (the ids tools/t4_check.sh uses)
mkdir -p samples && cd samples
for id in 10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ 1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd \
          1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_ 1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS; do
  gdown --continue "$id"
done
cd ..

# 2. 960x540, 10 fps proxies, which the alignment, EDA and labelling tools read
mkdir -p cache/proxy
for c in C3896 C3897 C3902 C3905; do
  ffmpeg -i samples/$c.MP4 -vf fps=10,scale=960:540 -an -c:v libx264 -preset veryfast -crf 22 -g 20 cache/proxy/$c.mp4
done

# 3. Detector caches: the submitted detector on every clip, the ablation detectors on C3897 and C3905
python tools/cache_detections.py samples/*.MP4 --fps 10 --width 1920
python tools/cache_detections.py samples/C3897.MP4 samples/C3905.MP4 --weights yolo26s.pt --imgsz 1280
python tools/cache_detections.py samples/C3897.MP4 samples/C3905.MP4 --weights yolo26m.pt --imgsz 960
python tools/cache_detections.py samples/C3897.MP4 samples/C3905.MP4 --weights yolo26n.pt --imgsz 960
python tools/cache_detections.py samples/C3897.MP4 samples/C3905.MP4 --weights yolo26s.pt --imgsz 960

# 4. Each clip's homography to the reference view
python tools/align_cache.py C3896 C3897 C3902 C3905

# 5. Trajectories from the submitted detector's cache
python tools/tracks_from_cache.py cache/det/*__yolo26m_1280_1920_10fps.npz --out cache/tracks

# 6. Signal phase timelines
python tools/signal_timeline.py C3896 C3897 C3902 C3905 --fps 5

# 7. EDA figures and numbers
python tools/eda.py --out out/site_data

# 8. Ablations
python tools/ablation.py --gt labels/dev_labels.json --out out/ablations.json

# 9. The sample run with the time guards lifted, saving Part A's full analysis to out/analysis
PARIVISION_CACHE_DIR=out/analysis PARIVISION_TIME_SHARE=12 PARIVISION_TOTAL_LIMIT=30 PARIVISION_RISK_SHARE=10 \
  python run_submission.py --videos samples --out predictions_samples_rerun.json --team PariVision --time-factor 40
python tools/compare_predictions.py predictions_samples.json predictions_samples_rerun.json

# 10. Scores on the dev labels
python evaluate.py --pred predictions_samples_rerun.json --gt labels/dev_labels.json --json out/metrics.json --per-video

# 11. Website data, annotated videos, posters and example frames
python tools/make_site_data.py --pred predictions_samples_rerun.json --site site/public \
  --machine "<the machine that made the run>" --runtime-note "<how it was made>"

# 12. The browser pipeline's model and scene files, then the site tests' parity fixtures
python tools/export_browser_assets.py
python tools/export_parity_fixtures.py --clip C3905
python tools/export_parity_fixtures.py --clip C3902

# 13. The crash check on the ACCIDENT benchmark (downloads from Kaggle)
for set in real synthetic; do
  python tools/crash_check.py fetch --set $set
  python tools/crash_check.py cache --set $set
  python tools/crash_check.py eval --set $set --ours
done
```

Two optional checks: `python tools/drivable_mask.py --check` compares the
drivable-area mask rebuilt from `cache/tracks` with the committed one, and

```bash
python tools/build_dev_labels.py --verified labels/dev_labels_verified.json \
  --adjudication labels/adjudication.json --out labels/dev_labels.json
```

rebuilds `labels/dev_labels.json` byte for byte from the 107 verified labels
and the adjudication verdicts (the agent runs behind them are not in the
repository, see `docs/labeling.md`).

The three 30 s clips of the live demo (`site/public/media/demo/`, gitignored)
are cuts of the samples. The commands were not recorded, so we found the start
times by matching frames against the samples. These commands rebuild them:

```bash
mkdir -p site/public/media/demo
ffmpeg -ss 10 -i samples/C3896.MP4 -t 30 -vf scale=1920:-2 -pix_fmt yuv420p -c:v libx264 -an -movflags +faststart site/public/media/demo/north_crossing_midday.mp4
ffmpeg -ss 195 -i samples/C3897.MP4 -t 30 -vf scale=1920:-2 -pix_fmt yuv420p -c:v libx264 -an -movflags +faststart site/public/media/demo/west_crossing_turns.mp4
ffmpeg -ss 75 -i samples/C3905.MP4 -t 30 -vf scale=1920:-2 -pix_fmt yuv420p -c:v libx264 -an -movflags +faststart site/public/media/demo/dusk_queue.mp4
```

`-movflags +faststart` puts the index at the start of the file, so the browser
can start playing a clip before all of it has arrived. With ffmpeg 9.0.2 the
north crossing and dusk clips come out pixel for pixel the same as ours. The
west crossing clip comes out one frame shorter (899 frames against 900) and
its last frames differ slightly. The annotated videos and the home-page loop
come from step 11.

The stored result the demo replays (with `?mock=1`, and in browsers without
WebAssembly) is the browser pipeline's own output on the dusk clip, made from
the command line with ffmpeg and onnxruntime-node:

```bash
cd site && pnpm replay public/media/demo/dusk_queue.mp4 public/data/demo/mock_result.json --video /media/demo/dusk_queue.mp4
```

## Tests

```bash
pip install pytest && pytest -q
cd site && pnpm install --frozen-lockfile && pnpm test
```

On a clean clone all 13 `pytest -q` tests pass and need nothing outside the
repository: most use a 6 s synthetic clip or hand-made scenes, and
`tests/test_regression.py` runs the rules on C3905's saved trajectories and
checks they give the submitted events. `pnpm test` needs Node 22.18 or
newer; on a clean clone it runs 66 tests, of which 51 pass and 15 are skipped.
Most site tests check the in-browser port of the pipeline
(`site/src/pipeline/`) against the Python pipeline, stage by stage, on fixtures
that `tools/export_parity_fixtures.py` writes from the dev caches. The C3905
fixtures and the two reference frames are committed, gzipped (6.9 MB
together); the C3902 fixtures are not. The 15 skipped tests are the C3902
ones, and each prints the command that writes their fixtures
(`python tools/export_parity_fixtures.py --clip C3902`, after steps 1 to 4 and
6 of Reproduce everything). With both clips' fixtures all 55 pass.

## Repository layout

```
solution.py            the interface: detect_events, RiskEstimator
run_submission.py      organisers' harness (unchanged)
evaluate.py            organisers' metric (unchanged)
src/parivision/        pipeline: video, detector, tracking, registration, signal, rules, risk, render
weights/               YOLO26 n/s/m COCO weights, their SHA256SUMS and download.sh
labels/                dev set (dev_labels.json = dev_labels_verified.json + adjudication.json), crash clip lists
tools/                 dev tools: caches, EDA, ablations, dev-set labelling, tuning, site data, crash check (index in tools/README.md)
tests/                 pytest checks for the core pieces and the solution interface
demo/                  FastAPI server that runs the full Python pipeline on a clip, for trying it locally (the website demo runs site/src/pipeline/ in the browser)
site/                  the team website (Astro)
docs/                  scene description, labelling guide, figures
predictions_samples.json  our output on the sample clips
LICENSE                AGPL-3.0
THIRD_PARTY_NOTICES.md licences of the models, packages and data we use
```

## Datasets, models and licences

* **YOLO26 n/s/m** detection weights, pretrained on COCO, from Ultralytics.
  Licence: AGPL-3.0. We use them as they are.
* **COCO 2017** (through those weights). Annotations: CC BY 4.0.
* **Sample clips** C3896, C3897, C3902 and C3905. No licence stated; the
  organisers gave them to participants for this hackathon. Not in this
  repository. Used for scene analysis, our dev labels and tuning. The website
  shows annotated renders of them, as the task asks, and three 30 s cuts for
  the in-browser demo.
* **ACCIDENT benchmark** (Picek et al., CVPR 2026; Kaggle `picekl/accident`):
  used only to check the accident rule and Part B, never for training.
  Licence: CC BY-NC-SA 4.0 for the data, CC BY 4.0 for the annotations. The
  clips are not in this repository; `labels/accident_real.csv` and
  `labels/accident_synthetic.csv` list the ones we used, with the benchmark's
  annotations.
* **Our dev labels** (`labels/`): 104 events on the four sample clips, drafted
  with Claude agents as `docs/labeling.md` describes. AGPL-3.0, with the rest
  of the repository.
* No other datasets are used.

Open-source code we build on: Ultralytics (AGPL-3.0; YOLO inference and the
ByteTrack implementation), ByteTrack itself (Zhang et al., MIT), PyTorch and
torchvision (BSD-3-Clause), lap (BSD-2-Clause), OpenCV (Apache-2.0), PyAV
(BSD-3-Clause; its wheels bundle an FFmpeg build that includes x264 and x265,
which are GPL), NumPy, SciPy (BSD), and for the website demo only
onnxruntime-web (MIT) and ffmpeg.wasm (MIT wrapper around an FFmpeg core under
GPL-2.0-or-later, which converts clips the browser cannot play). Because we
ship Ultralytics weights and call its code, this repository is released under
AGPL-3.0 (`LICENSE`). That includes `site/public/pipeline/model.onnx`, which
`tools/export_browser_assets.py` exports from `yolo26n.pt`.
`THIRD_PARTY_NOTICES.md` lists every third-party model, package and dataset
with its licence, and `weights/SHA256SUMS` holds the checksums of the weights,
which `weights/download.sh` checks.

## Team

| member | role | did |
|---|---|---|
| Amal Karimov (captain) | scene layout, signal and vehicle rules | traced the junction in the reference view; found the vehicle signal head and wrote its phase reader; red_light, stop_line, stopped_vehicle and congestion rules |
| Komronbek Qodirov | pipeline, live demo, website | video decoding, detection, tracking, registration and the time budget; Part B; the in-browser port of the pipeline and the website |
| Aziza Adizova | dev set, pedestrian rules, report | wrote the labelling guide, ran the labelling, verifier and adjudicator agents and checked part of their reasoning by hand; jaywalking and failure_to_yield rules and their error analysis; EDA and the report |

Links: Amal Karimov, [GitHub](https://github.com/Shen-de-Dia) and
[LinkedIn](https://www.linkedin.com/in/ka-a-a07690405/); Komronbek Qodirov,
[GitHub](https://github.com/baytoooo),
[LinkedIn](https://www.linkedin.com/in/kamron-kadirov-7a8149303/) and
[portfolio](https://bayto.uz); Aziza Adizova,
[LinkedIn](https://www.linkedin.com/in/aziza-adizova-033a712a0/). More on the
[team page](https://parivision-traffic.vercel.app/team).

## Licence

Copyright (C) 2026 Team PariVision: Amal Karimov, Komronbek Qodirov, Aziza
Adizova.

This program is free software: you can redistribute it and modify it under the
terms of the GNU Affero General Public License, version 3, as published by the
Free Software Foundation (`LICENSE`, SPDX `AGPL-3.0-only`). It comes with no
warranty; see sections 15 and 16 of the licence. Third-party models, code and
data keep their own licences, listed in `THIRD_PARTY_NOTICES.md`.
