# Traffic events and accident anticipation from one fixed camera

Team PariVision (Webster University in Tashkent): Amal Karimov, Komronbek
Qodirov, Aziza Adizova. WIUT Hackathon 2026, Computer Vision track,
elimination task.

## Summary

We built a system that watches the organisers' 4K camera over a Tashkent
junction and reports traffic events as time segments (Part A), and a causal
risk score that an accident starts within 5 s (Part B). Part A is a pretrained
detector and tracker followed by geometry and one hand-written rule per event
class, all in one reference view of the junction. On our own labels of the
four sample clips it scores a mean F1 of 0.70 over the six classes
it emits, and an official Score A of 0.525 once the classes it does not
emit are counted as zeros. Nothing is trained by us.

## The data, and a dev set without labels

The organisers gave four clips (C3896, C3897, C3902, C3905): 18.4 minutes of
3840x2160 video at 29.97 fps, filmed on 18 September 2026 between 11:18 and
17:22 local time, from one camera high on a building at a signalised four-way
junction. There were no labels, and the `camera.md` the task mentions was not
in the starter kit, so `docs/scene.md` is our own description of the scene.

To measure anything we needed labels, so we built a dev set with AI agents
(Claude) working from a written guide (`docs/labeling.md`) that follows the
task's start and end conventions:

1. Each clip was cut into 40 s windows. Two labelling agents looked at every
   window, one for pedestrian events, one for vehicle events, on contact sheets
   of frames at 0.2 to 1 s steps, with the signal phase given to them.
2. Every claimed event went to a separate verifier agent that tried to refute
   it on zoomed crops at 0.1 to 0.5 s steps, and corrected its class and
   boundaries when it held. Of 351 claims, 285 were kept and 66 rejected.
3. Once the model existed, every disagreement between model and labels on
   three clips went to an adjudicator agent (52 cases). The model was wrong in
   34, the labels in 13, and 5 differed only in boundaries. The labels were
   corrected where the adjudicator found them wrong.

No person labelled frames. We read the agents' reasoning for many of the events
and it holds up, but the numbers below are indicative, and step 3 only
re-examined places where the model disagreed, which favours the model.

The dev set has 104 events: mostly failure_to_yield and jaywalking,
then U-turns, stopped vehicles, stop-line and red-light events, three
congestion episodes and two illegal turns. There are no accidents and no near
misses in the samples.

## What the footage told us

* The clips are not framed the same. The afternoon clips are shifted by up to
  60 px and zoomed by about 2% against the morning ones, so fixed pixel
  polygons would be wrong on them. Every zone is drawn once in a 1920x1080
  reference view and each clip is registered onto it. C3896's camera also
  drifts 9 px over its first 40 s, so the view is registered again on
  keyframes through the clip.
* The signal head that faces the camera on the left pole is a pedestrian
  signal. Reading it as the vehicle signal made every car in the last platoon
  of a green look like a red-light runner. The vehicle phase comes from the
  three-lamp head on the median nose. Morning plan: 36 s green (the last 3 s
  flashing), 3 s yellow, 36 s red. Afternoon plan: 38 s green, 3 s yellow,
  39 s red. The rules read the phase from the lamps instead of assuming a plan.
* The southbound queue stands through every red. So a stopped vehicle has to
  be somewhere traffic should flow, and congestion has to hold 8 s into a green.
* About 5.7% of pedestrian positions are on the carriageway outside a zebra and
  21% are on a zebra. The rest wait at kerbs, islands and the median, which is
  exactly where a naive jaywalking rule fires.
* Drivers seen through car windows and delivery riders come out of the detector
  as pedestrians. Both have to be removed before any pedestrian rule runs.

## Part A: events

```
4K frame -> every 3rd frame, 1920 px (PyAV, background thread)
         -> YOLO26m @1280 (COCO weights) -> ByteTrack, one tracker per group
         -> homography to the reference view (SIFT, midday and dusk references,
            again on keyframes while the camera settles)
         -> signal phase (colour contrast of each lamp on the median head)
trajectories in the reference view + phase + scene layout
         -> one rule per class -> per-actor intervals -> union per class -> segments
```

**Detection and tracking.** YOLO26m with COCO weights at 1280 px, confidence
0.1, on every third frame (10 fps), batches of 8, fp16 on the GPU. ByteTrack
(the Ultralytics implementation) runs separately for vehicles, pedestrians,
bicycles and animals, so a pedestrian on a zebra never inherits the id of the
car passing in front of them. Each box becomes a foot point on the ground in
the reference view; speeds are smoothed finite differences. A scale map fitted
on about 150,000 pedestrian boxes gives metres per pixel anywhere in the view.

**Rules.** One function per class on trajectories, zones and phase:

| class | rule, in short |
|---|---|
| jaywalking | a pedestrian (riders and car occupants removed) on the area vehicles actually drive on, clearly outside every zebra and away from the kerb, for 1 s or more |
| failure_to_yield | a car, bus or truck drives across a zebra while a walking pedestrian is out on the same zebra within about 3 m of it |
| red_light | a southbound front crosses the stop line after 1 s of red and at least 1.5 s before green |
| stop_line | a southbound vehicle stands past the stop line, before the far edge of the north crossing, on red |
| stopped_vehicle | a vehicle stands 10 s or more in the northbound lanes or in the path through the junction, not at the bus stop |
| congestion | 8 or more southbound vehicles stand still on the approach and past the line, from 8 s into a green, for 6 s or more |
| wrong_way | a vehicle or bike moves more than 120 degrees against its carriageway for 1.5 s |

Per-actor intervals are merged per class: overlapping ones always (the task's
convention), and jaywalking across gaps of up to 3 s, because a pedestrian's
track breaks whenever a bus passes in front of them.

**Classes we do not emit.** Score A averages over every class that is in the
test labels or in our predictions, so a class we predict that never occurs in
the test set adds a zero. We emit a class only when we expect it to help. Our
U-turn rule finds U-turns round the median nose with F1 0.27 on the dev set,
but nothing in view marks them as prohibited. At F1 0.27, predicting them pays
off only if the organisers label them illegal with probability above about
0.7, so they are shown on the website and not submitted. Illegal turns (right
turns from a middle lane, two in the dev set), accidents, near misses, solid
line crossings, obstacles and fire have no rule.

**Time budget.** Part A aims to finish within 1.3x the clip length. It watches
its own pace and analyses every second, then every third sampled frame if it
falls behind, and stops at the deadline with what it has. The rules scale
their gap thresholds with the sampling step, so a thinned run still finds
short events.

## Part B: accident anticipation

`RiskEstimator.step` sees frames only in order. Every third frame (10 Hz) is
decimated to 1280 px and goes through YOLO26s at 960 px and the same tracker
(with a 1.5 s memory instead of 2 s). For each pair of road users on the
carriageway it predicts constant-velocity motion for 3 s, in metres at the
scale of the point between them, and scores how soon and how deeply
their footprints would overlap, weighted by closing speed; hard braking raises
the score. A pair has to look dangerous for 0.6 s without a break; pairs on
opposite sides of the median, and a moving car next to a parked one, are
ignored. The risk is the worst pair, smoothed with an exponential average. It
keeps its own work under 0.4x real time and both parts under 2.8x the clip
length by thinning frames.

The samples have no accidents, so Part B cannot be scored on them. What we can
check is that it stays calm in normal traffic: on the four samples
the score crosses the 0.5 alarm threshold once in 18.4 minutes, for 0.7 s in
C3905 at 57.6 s, when a dense platoon of cars crosses the junction box side by
side and nothing happens. On the other three clips it peaks at 0.499, just
under the threshold.
Before we measured each pair in a single metre scale and asked for 0.6 s of
danger instead of three updates, queues and following cars set off about one
false alarm per clip.

## Results on our dev set

| class | F1 @0.3 | F1 @0.5 | F1 @0.7 | mean | TP / FP / FN @0.5 |
|---|---:|---:|---:|---:|---:|
| congestion | 0.800 | 0.800 | 0.800 | 0.800 | 2 / 0 / 1 |
| failure_to_yield | 0.641 | 0.524 | 0.388 | 0.518 | 27 / 28 / 21 |
| illegal_turn (no rule) | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 / 2 |
| illegal_u_turn (not submitted) | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 / 9 |
| jaywalking | 0.808 | 0.462 | 0.346 | 0.538 | 12 / 12 / 16 |
| red_light | 0.667 | 0.667 | 0.667 | 0.667 | 1 / 0 / 1 |
| stop_line | 0.769 | 0.769 | 0.769 | 0.769 | 5 / 3 / 0 |
| stopped_vehicle | 1.000 | 0.857 | 0.857 | 0.905 | 6 / 1 / 1 |

Score A over these eight classes: **0.525**. Mean over the six classes we emit: 0.70.

Ablations (same rules, one change per row; detector variants on the two clips
we cached their detections for):

| configuration | clips | Score A | emitted classes | detector cost |
|---|---:|---:|---:|---:|
| YOLO26m @1280, 10 fps | 4 | 0.526 | 0.702 | 1.00x |
| YOLO26m @1280, 5 fps | 4 | 0.513 | 0.684 | 0.50x |
| YOLO26m @1280, 3.3 fps | 4 | 0.406 | 0.542 | 0.33x |
| YOLO26m @1280, 10 fps, same clips | 2 | 0.340 | 0.476 | 1.00x |
| YOLO26s @1280, 10 fps | 2 | 0.344 | 0.481 | 0.31x |
| YOLO26m @960, 10 fps | 2 | 0.330 | 0.462 | 0.56x |
| YOLO26n @960, 5 fps | 2 | 0.337 | 0.471 | 0.03x |
| YOLO26s @960, 5 fps | 2 | 0.350 | 0.490 | 0.09x |
| No registration | 4 | 0.393 | 0.506 | 1.00x |
| Hand-drawn road mask | 4 | 0.509 | 0.679 | 1.00x |

What the ablations say: registration matters; the learned drivable area is
better than the hand-drawn road for jaywalking; and 5 fps loses little, while
3.3 fps loses a fifth of the score even with the rules' gaps scaled to the
sampling step, because a car crosses a zebra in about a second. The small
detectors at 960 px do about as well as the submitted one on these two clips,
which is why the live demo can run YOLO26n in a browser.

**Runtime.** The submitted `predictions_samples.json` was made on an Apple M5
laptop with PyTorch on MPS and the time guards lifted, so that it is the output
of the full pipeline with no frames thinned: 3.3x the clip length over the four
clips (Part A 2.0x, Part B 1.2x). The laptop is slower than a T4. With the
default guards the pipeline stays inside the 3x budget on any machine by
thinning frames, and `tools/t4_check.sh` measures it on a Colab T4.

## What did not work, and why

* **Failure to yield is short and ambiguous.** A car crosses a zebra in about
  1 s, so boundaries decide tIoU 0.5 and 0.7. The definition does not say how
  close the pedestrian must be or whether someone standing on the stripes
  counts; our labellers and adjudicators disagreed on exactly that, and the
  rule's "walking" condition removes some real events along with the false ones.
* **Consecutive cars were chained into one segment.** Merging failure_to_yield
  segments less than 1 s apart turned four separate cars into one long
  segment. Merging only overlaps raised its F1 from 0.35 to 0.44.
* **Moped riders as jaywalkers.** When the detector misses the moped, the
  rider is a fast "pedestrian". A speed test in the person's own scale (more
  than 1.2 body heights per second; walkers stay under 0.9) removed them.
* **Stop line.** A car that crossed on green and got stuck past the line when
  the red came is a violation in some readings and not in others. Our labels
  and our adjudicator disagreed, so we kept the simpler rule.
* **Congestion is crawling, not only standing.** A jam that crawls at 1 to
  2 m/s is missed by our standing threshold, but raising it made normal slow
  traffic look like jams on two other clips.

## The live demo

The website's demo runs the pipeline in the visitor's browser: Hugging Face
now charges for Spaces that run code, and a demo that needs no server cannot
go down. The TypeScript port uses the same constants, exported from the Python
code, and is pinned to it by parity tests stage by stage (tracker, trajectories,
signal, rules, risk). It differs from the submission in three places: YOLO26n
at 960 px through ONNX Runtime Web (WebGPU or WASM), 5 fps, and a similarity
alignment found by searching edge maps instead of SIFT. It analyses the first
2 minutes of a clip; nothing leaves the visitor's machine.

## What we would do next

1. Fine-tune the detector on frames from this camera, starting with far and
   dusk pedestrians, now that the dev set shows where it misses them.
2. An illegal-turn rule from the lane a vehicle holds at the stop line.
3. More footage: night, rain, and anything with a near miss, which is what
   Part B needs to be calibrated at all.

## Reproducing

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
```

Weights are in the repository. `run_submission.py` and `evaluate.py` are the
organisers' files, unchanged. `tools/` rebuilds everything else: detection
caches, the dev labels from the agent runs, ablations, the EDA and the
website's data. Code is AGPL-3.0 because it ships Ultralytics weights.
