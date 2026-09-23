# How we labelled the sample clips

The organisers keep their labels hidden, so we built our own dev set on the
four sample clips with the start/end conventions from the task. This page is
the guide the labelling, verifier and adjudicator agents (Claude) worked from;
`docs/report.md` describes the runs. Read `docs/scene.md` first for the
layout, the crossings and the signal.

## Output

One event is `[start_sec, end_sec, label]`. Two rules from the task shape
the whole dev set:

* Segments of the **same class never overlap**. If two jaywalkers are on
  the road at the same time, that is one `jaywalking` segment from the first
  one stepping on to the last one stepping off. We label each actor
  separately and merge per class at the end.
* Scoring matches segments by temporal IoU at 0.3, 0.5 and 0.7, so the
  boundaries matter as much as the label. We read them off the 10 fps proxy,
  which gives 0.1 s resolution.

Times are seconds from the first frame (`frame_index / 29.97`).

## Tools

```
python tools/sheet.py cache/proxy/C3896.mp4 --start 40 --end 48 --step 0.5 --cols 4
python tools/sheet.py cache/proxy/C3896.mp4 --start 40 --end 44 --step 0.2 --crop 0,250,480,540 --tile 640
```

The proxies are 960x540 at 10 fps, so proxy pixel = reference pixel / 2.
`cache/signal/<clip>.json` has the signal phase per 0.2 s.

## Classes, as applied to this junction

**jaywalking.** A pedestrian on the carriageway outside the zebra markings:
cutting diagonally across the junction box, crossing the avenue away from
the north crossing, walking in the road next to a crossing (more than
about a body width outside the stripes), standing in a traffic lane.
Not jaywalking: walking on a zebra, standing on a pink island or the median
nose, stepping off the kerb for a second while waiting. Start: the foot
leaves the kerb (or the zebra) onto the carriageway. End: back on a kerb,
island or zebra.

**failure_to_yield.** A vehicle drives across a zebra while a pedestrian is
on that zebra (or stepping onto it) in the vehicle's path or the lane next
to it. Typical here: right-turners over the west crossing, and vehicles
crossing the north crossing when pedestrians are still on it. A pedestrian
who is on the far half, well beyond a lane divider and not heading into the
vehicle's lane, does not count. Start: vehicle front enters the zebra.
End: vehicle rear leaves it.

**red_light.** A southbound vehicle whose front crosses the stop line while
the signal is red (not yellow). Start: front crosses the stop line. End:
the vehicle leaves the junction box or the frame.

**stop_line.** A southbound vehicle that stops (speed ~0 for at least a
second) with its front past the stop line while the signal is red, without
going on into the junction; usually standing on the north crossing. Start:
it stops. End: the signal turns green (or it leaves, if earlier).

**stopped_vehicle.** A vehicle standing still on the carriageway for 10 s
or more that is not queued at the red light: stopped in the junction box,
dropping off passengers in a lane, broken down. Buses at the NB bus stop
and parked cars off the carriageway do not count. Start: it stops. End: it
moves off or leaves.

**congestion.** All lanes of one direction at a standstill or crawling,
and not just a red-light queue: for example the SB lanes still stuck while
the light is green, or the junction box gridlocked. Start: the queue stops
moving. End: it clears.

**wrong_way.** A vehicle (including bicycles and scooters ridden on the
road) moving against its lane's direction: driving up the SB carriageway
away from the camera, down the NB carriageway towards the camera, or
through the junction on the wrong side of the median nose. Start: it
enters the opposing lane. End: it is back in a correct lane or gone.

**illegal_u_turn.** A U-turn where it is not allowed. At this junction
note every U-turn you see (for example SB traffic turning round the median
nose into the NB carriageway) with its time; whether it counts is decided
after looking at all of them. Start: the turn begins. End: it is done.

**illegal_turn.** A turn from the wrong lane (for example a right turn
into the west arm from a middle or left lane, cutting across other lanes)
or into a direction that is not allowed. Start: the turn begins. End: it
is done.

**solid_line_crossing.** A lane change across a solid line. The lane
lines in the last stretch before the SB stop line and the median edge are
solid. Start: the wheel crosses the line. End: fully in the new lane.

**near_miss.** Hard braking or a swerve to avoid hitting another road user,
no contact. Needs visible evasive action (nose dive, sudden stop, sharp
swerve, a pedestrian jumping back). Start: the evasive action starts. End:
the road users are clear of each other.

**accident.** Contact between road users, or with a kerb, pole or island.
Start: first frame of contact. End: everyone involved has stopped moving or
left the frame.

**road_obstacle.** Debris, a dropped object or an animal on the
carriageway. Start: it appears. End: it is removed.

**fire_smoke.** Visible fire or smoke from a vehicle or on the road.
Exhaust in cold air does not count. Start: first smoke. End: it clears.
