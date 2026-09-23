# Traffic events and accident anticipation from one fixed camera

Team PariVision (Webster University in Tashkent): Amal Karimov, Komronbek
Qodirov, Aziza Adizova. WIUT Hackathon 2026, Computer Vision track.
Website: https://parivision-traffic.vercel.app. Code and run commands:
https://github.com/baytoooo/parivision-traffic.

## What we built

Our pipeline takes the organisers' 4K footage of a Tashkent junction and
reports traffic events as time segments (Part A) and a causal risk score that
an accident starts within 5 s (Part B). Part A runs a COCO-pretrained YOLO26m
and ByteTrack at 10 frames per second, maps the tracks into one reference
view of the junction, reads the vehicle signal from its lamps and applies one
hand-written rule per event class. Part B projects each pair of road users
3 s ahead and scores how soon they would meet. We trained nothing; the
[Approach page](https://parivision-traffic.vercel.app/approach) has the
details.

## A dev set made with Claude agents

The organisers gave four clips (18.4 minutes) and no labels, so we built a
dev set with Claude agents working from our guide (`docs/labeling.md`). A
verifier agent tried to refute each of the labelling agents' 351 claims on
zoomed crops and kept 285. An adjudicator agent then reviewed 52
disagreements between model and labels: the model was wrong in 34, the labels
in 13, and 5 differed only in boundaries. No person labelled frames. We
checked only part of the agents' reasoning by hand, so our numbers are
indicative, and the adjudication favours the model because it only looked
where the model disagreed.

The result is 104 events: 48 failure_to_yield, 28 jaywalking, 9 U-turns,
7 stopped vehicles, 5 stop-line, 3 congestion, 2 red-light and 2 illegal-turn,
and no accidents or near misses. The labels are in `labels/`. Claude, a
hosted model, was used only to build them; `solution.py` never calls it. The
agent logs are not in the repository, so the labels cannot be regenerated.

## What worked

* **Score.** The official Score A on our dev set is 0.525. We emit seven
  classes; the six that fired on the samples average an F1 of 0.70 (wrong_way
  never fired). We tuned the rules on these labels, so we expect less on the
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
* **Classes we leave out.** U-turns are found with F1 0.27 but not submitted,
  because nothing in view says they are prohibited. Six classes, among them
  accidents and near misses, have no rule, and Part B cannot be calibrated
  without accidents.
* **Runtime is not verified on a T4.** On an Apple M5 laptop with the time
  guards off the pipeline took 3.3x the clip length (Part A 2.0x, Part B
  1.2x). The laptop is probably slower than a T4, but we could not check: a
  Google Drive download quota blocked our Colab run. The default guards thin
  frames to stay inside the 3x budget, but the harness's own decoding of each
  4K frame is outside our control. `tools/t4_check.sh` times a T4 run.

## What we would do next

1. Time a T4 run and regenerate `predictions_samples.json` there with the
   default settings.
2. Have a person check the dev labels, starting with failure_to_yield.
3. Fine-tune the detector on this camera, for far and dusk pedestrians.
4. An illegal-turn rule from the lane a vehicle holds at the stop line.
5. Footage with near misses, so Part B can be calibrated.
