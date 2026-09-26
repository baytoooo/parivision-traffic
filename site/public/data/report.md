# Traffic events and accident anticipation from one fixed camera

Team PariVision (Webster University in Tashkent): Amal Karimov, Komronbek
Qodirov, Aziza Adizova. WIUT Hackathon 2026, Computer Vision track.
Website: https://parivision-traffic.vercel.app. Code:
https://github.com/baytoooo/parivision-traffic.

## What we built

Part A lists traffic events as time segments; Part B gives a causal risk that
an accident starts within 5 s. Part A runs YOLO26m (COCO weights) and
ByteTrack at 10 frames per second, maps each clip onto one reference view,
reads the signal lamps and applies one hand-written rule per class. Part B
projects each pair of road users 3 s ahead and scores how soon they would
meet. We trained nothing; the
[Approach page](https://parivision-traffic.vercel.app/approach) has every
threshold.

## A dev set made with Claude agents

The organisers gave four clips (18.4 minutes) and no labels, so Claude agents
labelled them from our guide (`docs/labeling.md`). A verifier agent kept 285
of 351 claims; an adjudicator agent reviewed 52 disagreements between model
and labels (the model was wrong in 34, the labels in 13). No person labelled
frames, so our numbers are indicative, and the adjudication favours the model.
The result is 104 events in eight classes, with no accidents or near misses.
`solution.py` never calls Claude.

## What worked

* **Score A 0.525 on our dev set.** The six classes that fired average an F1
  of 0.70. The rules were tuned on these labels, so we expect less on the test
  set.
* **Registration.** Framing shifts between clips and the camera drifts while
  it settles. Without registration Score A falls from 0.526 to 0.393
  (ablation figures; the official run gives 0.525).
* **The right signal head.** The head on the left corner pole is a pedestrian
  signal; read as the vehicle one, it turned the last platoon of every green
  into red-light runners. We read the median-nose head instead.
* **A road mask learned from traffic** raises jaywalking F1 from 0.415 to
  0.551 in the ablations (0.538 in the official run).
* **10 frames per second.** 5 fps costs 0.013 of Score A; 3.3 fps drops it
  from 0.526 to 0.406, since a car is on a zebra for only about 2 s.
* **Part B stays calm.** In 18.4 minutes of normal traffic it crosses the 0.5
  threshold once, for 0.7 s; elsewhere its highest value is 0.499.

## What did not work

* **Failure to yield (F1 0.518).** Boundaries decide tIoU 0.5 and 0.7, and the
  task does not say how close the pedestrian must be.
* **Jaywalking (F1 0.538).** A rider whose moped the detector loses looks like
  a pedestrian; a speed test removes most.
* **Smaller misses.** Congestion crawling at 1 to 2 m/s, and one of the two
  red-light events, because our lamp reader calls red plus amber "yellow".
* **Crashes.** The samples have none. On the public ACCIDENT benchmark the
  collision rule finds 24 of 95 synthetic and 1 of 34 real CCTV crashes
  (mostly low-resolution clips where the striking car is not detected); it
  never fires in our normal traffic. Part B warns before 11 synthetic and 4
  real impacts; rear-end hits on a standing car stay invisible to it.
* **Classes we leave out.** U-turns are found with F1 0.27 but not submitted,
  since nothing in view says they are prohibited. Five classes have no rule.
* **Decoding sets the runtime.** On a Colab T4 our detectors need 0.34x
  (Part A) and 0.19x (Part B) of real time, but two vCPUs decode these 4K
  10-bit clips at only 9 to 12 fps.

## What we would do next

1. Regenerate `predictions_samples.json` on a T4 with enough CPU cores to
   decode 4K at speed.
2. Have a person check the dev labels, starting with failure_to_yield.
3. Fine-tune the detector on this camera, for far and dusk pedestrians.
4. An illegal-turn rule from the lane a vehicle holds at the stop line.
5. Footage with near misses, so Part B can be calibrated.
