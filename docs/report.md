# Traffic events and accident anticipation from one fixed camera

Team PariVision (Webster University in Tashkent): Amal Karimov, Komronbek
Qodirov, Aziza Adizova. WIUT Hackathon 2026, Computer Vision track.
Website: https://parivision-traffic.vercel.app. Code and run commands:
https://github.com/baytoooo/parivision-traffic.

## What we built

Our pipeline reports traffic events in the organisers' 4K footage as time
segments (Part A) and gives a causal risk that an accident starts within 5 s
(Part B). Part A runs a COCO-pretrained YOLO26m and ByteTrack at 10 frames per
second, maps the tracks into one reference view of the junction, reads the
vehicle signal from its lamps and applies one hand-written rule per class.
Part B projects each pair of road users 3 s ahead and scores how soon they
would meet. We trained nothing; the
[Approach page](https://parivision-traffic.vercel.app/approach) has details.

## A dev set made with Claude agents

The organisers gave four clips (18.4 minutes) and no labels, so Claude agents
labelled them from our guide (`docs/labeling.md`). A verifier agent kept 285
of 351 claims, and an adjudicator agent reviewed 52 disagreements between
model and labels (the model was wrong in 34, the labels in 13). No person
labelled frames and we checked only part of the reasoning, so our numbers are
indicative; the adjudication favours the model, since it only looked where the
model disagreed.

The result is 104 events: 48 failure_to_yield, 28 jaywalking, 9 U-turns,
7 stopped vehicles, 5 stop-line, 3 congestion, 2 red-light and 2 illegal-turn,
and no accidents or near misses. Claude, a hosted model, was used only to
build these labels (`labels/`); `solution.py` never calls it.

## What worked

* **Score.** The official Score A on our dev set is 0.525. We emit eight
  classes; the six that fired on the samples average an F1 of 0.70 (wrong_way
  and accident never fired). We tuned the rules on these labels, so we expect less on the
  test set. The per-class tables and the ablations are on the
  [Results page](https://parivision-traffic.vercel.app/results).
* **Registration.** The afternoon clips are shifted by up to about 60 px and
  zoomed by 1 to 2%, and C3896's camera drifts 9 px while it settles, so each
  clip is registered onto one reference view. Without that, Score A falls from
  0.526 to 0.393 (ablations replay cached detections, hence 0.526 and not the
  official 0.525).
* **The right signal head.** The head on the left corner pole is a
  pedestrian signal. Read as the vehicle signal, it turned the last platoon of
  every green into red-light runners, so we read the median-nose head instead.
* **A road mask learned from traffic.** Checking jaywalking against where
  vehicles really drive, not our hand-traced carriageway, raises its F1 from
  0.415 to 0.551.
* **10 frames per second.** 5 fps costs 0.013 of Score A, but 3.3 fps loses
  almost a quarter (0.526 to 0.406), because a car is on a zebra for only
  about 2 s (median 1.9 s in our labels).
* **Part B stays calm.** In 18.4 minutes of normal traffic its score crosses
  the 0.5 alarm threshold once, for 0.7 s in C3905; on the other clips the
  highest is 0.499 (C3896). Measuring each pair in one metre scale and asking
  for 0.6 s of danger cut false alarms from about one per clip to this one.
* **A demo with no server.** The website's demo runs a parity-tested port of
  the pipeline in the visitor's browser (YOLO26n, 5 fps): no server to keep
  running, and a clip the visitor picks never leaves their machine.

## What did not work

* **Failure to yield (F1 0.518).** Boundaries decide tIoU 0.5 and 0.7, and
  the task does not say how close the pedestrian must be. Merging segments up
  to 1 s apart chained consecutive cars into one event, so we now merge only
  overlapping ones, as the task does.
* **Jaywalking (F1 0.538).** When the detector loses a moped, its rider looks
  like a pedestrian. A speed test in body heights per second removes most of
  them, not all.
* **Smaller misses.** Congestion that crawls at 1 to 2 m/s is missed. Our
  lamp reader calls red plus amber "yellow", and the rule ignores crossings in
  the last 1.5 s before green, so we miss one of the two red-light events.
* **Crashes we cannot see coming.** The samples have no crash, so we checked
  the accident rule and Part B on the public ACCIDENT benchmark. The rule finds
  24 of 95 synthetic crashes and 1 of 34 real CCTV crashes (mostly
  low-resolution clips where the detector misses the striking car), and never
  fires in our normal traffic, so we submit it. Part B warns before 11 of the
  synthetic and 4 of the real impacts; making it more eager costs more false
  alarms than it gains, and rear-end hits on a standing car stay invisible to it.
* **Classes we leave out.** U-turns are found with F1 0.27 but not submitted,
  because nothing in view says they are prohibited. Five classes, among them
  near misses, have no rule.
* **Decoding, not the model, sets the runtime.** On a Colab T4 our detectors
  cost 0.34x (Part A) and 0.19x (Part B) of real time, but Colab's two vCPUs
  decode these 4K 10-bit clips at only 9 to 12 frames per second, so the
  harness alone needs about 3x the clip length to read them for Part B. Part A
  now measures the decode speed and leaves the harness its share, so a slow
  machine gives a shorter result instead of an empty one.

## What we would do next

1. Regenerate `predictions_samples.json` on a T4 machine with enough CPU
   cores to decode 4K at speed, with the default settings.
2. Have a person check the dev labels, starting with failure_to_yield.
3. Fine-tune the detector on this camera, for far and dusk pedestrians.
4. An illegal-turn rule from the lane a vehicle holds at the stop line.
5. Footage with near misses, so Part B can be calibrated.
