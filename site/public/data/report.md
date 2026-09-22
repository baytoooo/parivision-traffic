# What we built, what worked, what did not

PariVision, WIUT Hackathon 2026, CV track. Numbers marked TBD come from the
final run of `evaluate.py` on our dev labels and will be filled in before
judging.

## What we built

One fixed 4K camera looks up a Tashkent avenue from a building on the
south-west corner of a signalised junction. Our system turns that footage
into two outputs.

**Part A** lists traffic events as `[start, end, class]` segments. We decode
every 3rd frame (about 10 per second), resize it to 1920 px wide and run a
COCO-pretrained YOLO26m at 1280 px. ByteTrack links the boxes into tracks,
one tracker per object group so a pedestrian on a crossing is not glued to
the car in front of them. Each clip is registered to one reference view of
the junction with a SIFT homography, and every rule works in that view:
stop line, crossings, carriageways, bus stop. The signal phase comes from
the lamps of a head on the median nose. Hand-written rules on the
trajectories and the phase produce per-actor intervals, which are merged per
class into segments.

**Part B** gives a per-frame probability that an accident starts within
5 s. It is causal: YOLO26s at 10 Hz, the same tracker, a constant-velocity
look-ahead of 4 s for every road user, and a hazard for each pair that is
closing in on each other. Hazards are combined and smoothed.

We labelled the four sample clips ourselves (18.4 minutes) with the task's
start and end conventions, because the organisers' labels are hidden. That
dev set is what every number on this site is measured against.

## What worked

* **Registration first.** Drawing the scene once in a reference view and
  mapping each clip onto it removed a whole class of bugs. The afternoon
  clips are framed about 20 px differently, and nothing downstream had to
  know.
* **Reading the signal instead of learning it.** Two heads face the camera
  and stay readable at dusk. Lamp colour plus lamp position gives the phase
  every 0.2 s, and tail lights passing behind the head do not fool it.
  red_light and stop_line are simple rules on top of that.
* **One tracker per object group.** ByteTrack matches boxes by overlap only,
  so with one tracker a pedestrian on a crossing gets glued to the car
  passing in front of them.
* **Treating riders as vehicles.** A person over a bicycle or motorbike box,
  or moving faster than 110 px/s, is a rider. Without this, a cyclist on the
  carriageway reads as a jaywalker.
* **Keeping classes off until they work.** Score A is a macro average over
  classes, so a class we predict that never occurs in the test set adds a
  zero. We emit six classes and leave the rest off.

Dev-set Score A: TBD. Runtime: TBD minutes per minute of 4K video.

## What did not work

* **Long merged segments.** Same-class segments may not overlap, so when
  several people jaywalk at overlapping times we output one long segment.
  On C3896 that gave one 133 s jaywalking segment where our labels have five
  shorter ones, and at IoU 0.3 it matches none of them.
* **A stopped_vehicle segment covering a whole clip.** On C3896 one segment
  runs from 0.0 to 340.2 s with no matching label. We have not found the
  cause yet.
* **Part B has nothing to learn from.** The sample clips contain no
  accidents, so `evaluate.py` returns null for Part B on our dev set. The
  risk score is hand-calibrated so that 0.5 should mean contact is likely
  within 5 s, and we cannot check that on this data.
* **U-turns.** We detect them, but we could not confirm which U-turns are
  illegal at this junction, so the class stays off.
* **Seven classes have no rule.** accident, near_miss, illegal_turn,
  solid_line_crossing, congestion, road_obstacle and fire_smoke are never
  emitted.

## What we would do next

1. Split merged segments by actor before the per-class union, and score the
   change against the dev labels.
2. Find the cause of the whole-clip stopped_vehicle segment.
3. Congestion: a rule on the southbound queue that must hold through a whole
   green phase.
4. Fine-tune the detector on frames from this camera, starting with dusk
   pedestrians on the far crossing.
5. Label more footage, especially at dusk and at night, and anything with a
   near miss in it.
