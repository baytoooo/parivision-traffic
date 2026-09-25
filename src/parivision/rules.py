"""Event rules on trajectories + scene layout.

Every rule returns a list of ``Evidence`` (one per actor and interval). The
pipeline unions them per class into the final segments; the renderer and the
website use the evidence to draw who did what.

Distances are in reference pixels (1920x1080 view). Around the north crossing
one metre across the view is about 50 px (``scene.metres_per_px``), and much
less along the avenue, which the camera sees end-on. Where it matters,
thresholds are scaled by the person's own box height instead. They were tuned
on our dev labels of the sample clips (tools/tune.py, tools/ablation.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import scene as S
from .registration import warp_points
from .segments import runs
from .detector import MOTORCYCLE
from .trajectories import Trajectory

PARAMS = {
    # pedestrians
    "rider_speed": 110.0,          # px/s median speed above which a "person" is on a bike
    "rider_overlap": 0.3,          # fraction of person box covered by a two-wheeler box
    "rider_rel_speed": 1.2,        # body heights per second (median): walkers stay under ~0.9
    "jay_margin_cw": 0.35,         # x person height outside the zebra before we call it off-crossing
    "jay_margin_kerb": 0.25,       # x person height into the carriageway
    "jay_min_dur": 1.0,            # s
    "jay_gap": 0.6,                # s, bridged inside one person's run
    "jay_min_height": 0.65,        # x expected person height; shorter boxes are occluded
    # failure to yield
    "fty_near_px": 160.0,          # lateral distance ped <-> vehicle along the crossing
    "fty_cw_dilate": 10.0,         # px, person counts as on the crossing within this margin
    "fty_min_speed": 20.0,         # px/s, the vehicle must actually be driving through
    "fty_kerb": 0.4,               # x person height: out on the zebra, not standing at its kerb end
    "fty_walk": 0.25,              # body heights per second: the pedestrian is walking, not standing
    "fty_pad_start": 0.3,          # s
    "fty_pad_end": 0.1,            # s
    "fty_moto_speed": float("inf"),  # px/s: scooters count only when ridden faster than this
    # signal
    "red_settle": 1.0,             # s of red before a crossing counts as red-light running
    "red_before_green": 1.5,       # s of red still to go: jumping the light by a fraction of a second is noise
    "red_max_len": 60.0,           # s, cap for a car that sits in the junction after running the red
    # stopping
    "stop_speed": 12.0,            # px/s below which a vehicle counts as stopped
    "stop_line_margin": 8.0,       # px past the line
    "stop_line_min": 1.0,          # s stopped
    "stopped_min": 10.0,           # s for stopped_vehicle
    "stopped_ne_corner_x": 1720.0, # east of this the NB lanes meet the plaza entrance: cars wait there to leave
    "cong_green_n": 8,             # standing SB vehicles (approach + box) that make a jam on green: five lanes and the box
    "cong_red_n": 5,               # standing vehicles left in the box after the green ended
    "cong_min": 6.0,               # s
    "cong_gap": 4.0,               # s
    "cong_after_green": 8.0,       # s into green before standing traffic counts as a jam
    "cong_speed": 15.0,            # px/s: at or below this a vehicle counts as standing in the jam
    # direction
    "ww_min_speed": 40.0,
    "ww_angle": 120.0,             # deg away from the lane direction
    "ww_min_dur": 1.5,
    # U-turns
    "ut_dir_tol": 30.0,            # deg: how close the first/last headings must be to SB/NB
    "ut_nose_px": 130.0,           # the path passes the median nose this closely
    "ut_turn_deg": 30.0,           # deg away from SB = turning started; within this of NB = done
    "ut_min_dur": 2.0,             # s
}

# Zones with one legal direction of travel (reference pixels) and that direction (deg, image axes, y down).
DIRECTION_ZONES = {
    "sb": ([(55, 40), (125, 40), (120, 118), (990, 452), (930, 457), (285, 527), (170, 380), (90, 200)], 33.0),
    "nb": ([(130, 112), (450, 95), (700, 185), (900, 240), (1100, 265), (1240, 295), (1480, 362),
            (1640, 440), (1820, 458), (1188, 500), (1000, 432)], 205.0),
}
BUS_STOP = [(450, 95), (1150, 250), (1130, 300), (430, 130)]
# Where SB traffic should keep moving on green: the last ~40 m of the approach, the
# stop zone and the path through the junction to the south exit (the queue of the
# south approach at the right edge and the left-turn pocket by the nose are left out).
SB_FLOW_APPROACH = [(160, 330), (700, 330), (990, 452), (930, 457), (285, 527), (220, 440)]
SB_FLOW_BOX = [(330, 650), (1160, 560), (1300, 600), (1650, 720), (1650, 1080), (900, 1080), (560, 760), (400, 700)]
# The stretch between the stop line and the far edge of the north crossing (+ a bit).
STOP_ZONE = [(285, 527), (930, 457), (1165, 540), (1170, 575), (390, 670), (330, 650)]


@dataclass
class Evidence:
    label: str
    start: float
    end: float
    actors: list[int]
    note: str = ""


@dataclass
class Context:
    trajectories: list[Trajectory]
    signal_t: np.ndarray               # sample times of the phase timeline
    signal_phase: np.ndarray           # array of "red"/"green"/"yellow"/"unknown"
    duration: float
    masks: dict = field(default_factory=S.masks)

    def __post_init__(self):
        road = self.masks["road"]
        self.road = road
        self.road_dist = cv2.distanceTransform(road, cv2.DIST_L2, 3)
        self.walk = self.masks["walk_check"]
        self.walk_dist = cv2.distanceTransform(self.walk, cv2.DIST_L2, 3)
        self.cw_masks = {}
        for name, polygon in S.CROSSWALKS.items():
            m = np.zeros_like(road)
            cv2.fillPoly(m, [S.poly(polygon).astype(np.int32)], 1)
            self.cw_masks[name] = m
        any_cw = np.clip(sum(self.cw_masks.values()), 0, 1).astype(np.uint8)
        self.cw_dist = cv2.distanceTransform(1 - any_cw, cv2.DIST_L2, 3)
        self.zones = {name: _raster(zone, road.shape) for name, (zone, _) in DIRECTION_ZONES.items()}
        self.zones["stop"] = _raster(STOP_ZONE, road.shape)
        self.zones["bus"] = _raster(BUS_STOP, road.shape)
        self.zones["flow_approach"] = _raster(SB_FLOW_APPROACH, road.shape)
        self.zones["flow_box"] = _raster(SB_FLOW_BOX, road.shape)
        self.by_id = {tr.tid: tr for tr in self.trajectories}
        # sampling step: 0.1 s at 10 fps, longer where Part A thinned frames on a slow machine
        frames = np.unique(np.concatenate([tr.t for tr in self.trajectories])) if self.trajectories else np.array([])
        self.dt = float(np.percentile(np.diff(frames), 90)) if len(frames) > 2 else 0.1
        self.vehicles = [tr for tr in self.trajectories if tr.is_vehicle]
        self.two_wheelers = [tr for tr in self.trajectories if tr.group == "bicycle" or tr.cls == MOTORCYCLE]
        self._boxes_at = _index_boxes(self.vehicles + self.two_wheelers)
        self.people = [tr for tr in self.trajectories if tr.is_person and not self._not_a_pedestrian(tr)]

    # -- lookups ---------------------------------------------------------
    def sample(self, img: np.ndarray, pts: np.ndarray, outside: float = 0.0) -> np.ndarray:
        pts = np.asarray(pts).reshape(-1, 2)
        x = np.round(pts[:, 0]).astype(int)
        y = np.round(pts[:, 1]).astype(int)
        ok = (x >= 0) & (y >= 0) & (x < img.shape[1]) & (y < img.shape[0])
        out = np.full(len(pts), outside, dtype=np.float64)
        out[ok] = img[y[ok], x[ok]]
        return out

    def phase_at(self, t: float) -> str:
        if len(self.signal_t) == 0:
            return "unknown"
        i = int(np.clip(np.searchsorted(self.signal_t, t), 0, len(self.signal_t) - 1))
        return str(self.signal_phase[i])

    def gap(self, seconds: float) -> float:
        """A gap to bridge between samples: never shorter than 1.5 sampling steps, so a
        thinned run (5 or 3.3 fps when the machine is slow) is not cut into single samples."""
        return max(seconds, 1.5 * self.dt)

    def phases_at(self, ts: np.ndarray) -> np.ndarray:
        if len(self.signal_t) == 0:
            return np.full(len(ts), "unknown", dtype=object)
        i = np.clip(np.searchsorted(self.signal_t, ts), 0, len(self.signal_t) - 1)
        return self.signal_phase[i]

    def red_since(self, t: float) -> float:
        """Seconds the phase has been red at time t (0 if it is not red)."""
        if self.phase_at(t) != "red":
            return 0.0
        i = int(np.clip(np.searchsorted(self.signal_t, t), 0, len(self.signal_t) - 1))
        j = i
        while j > 0 and self.signal_phase[j - 1] == "red":
            j -= 1
        return t - float(self.signal_t[j])

    def next_green(self, t: float) -> float:
        idx = np.nonzero((self.signal_t > t) & (self.signal_phase == "green"))[0]
        return float(self.signal_t[idx[0]]) if len(idx) else self.duration

    # -- riders and occupants --------------------------------------------
    def _not_a_pedestrian(self, person: Trajectory) -> bool:
        """Cyclists and scooter riders (person on a two-wheeler), and people seen through a car window."""
        if np.median(person.speed) > PARAMS["rider_speed"]:
            return True
        # the same test in the person's own scale: a moped rider far up the avenue is only ~40 px
        # tall, and when the detector misses the bike this is all that gives them away
        if np.median(person.speed / np.maximum(person.height, 20.0)) > PARAMS["rider_rel_speed"]:
            return True
        inside = 0
        for i, t in enumerate(person.t):
            pb = person.box[i]
            area = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
            for b, is_two_wheeler in self._boxes_at.get(int(round(t * 10)), ()):
                iw = max(0.0, min(pb[2], b[2]) - max(pb[0], b[0]))
                ih = max(0.0, min(pb[3], b[3]) - max(pb[1], b[1]))
                cover = iw * ih / area
                if cover > (PARAMS["rider_overlap"] if is_two_wheeler else 0.6):
                    inside += 1
                    break
        return inside > 0.4 * len(person.t)


def _index_boxes(trajectories: list[Trajectory]) -> dict[int, list[tuple[np.ndarray, bool]]]:
    """Boxes by 0.1 s time bin, flagged when they belong to a bicycle or motorbike."""
    out: dict[int, list[tuple[np.ndarray, bool]]] = {}
    for tr in trajectories:
        two = tr.group == "bicycle" or tr.cls == MOTORCYCLE
        for t, b in zip(tr.t, tr.box):
            out.setdefault(int(round(t * 10)), []).append((b, two))
    return out


def _raster(polygon, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [S.poly(polygon).astype(np.int32)], 1)
    return m


def _front_points(tr: Trajectory, H: np.ndarray) -> np.ndarray:
    """Bottom-centre of the box in reference pixels: the front bumper for traffic coming at the camera."""
    b = tr.box
    return warp_points(np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1), H)


def _footprint_points(tr: Trajectory, H: np.ndarray) -> np.ndarray:
    """Five points on the lower part of the box, (N, 5, 2) in reference pixels."""
    b = tr.box
    h = b[:, 3] - b[:, 1]
    xs = [b[:, 0] + 0.1 * (b[:, 2] - b[:, 0]), (b[:, 0] + b[:, 2]) / 2, b[:, 2] - 0.1 * (b[:, 2] - b[:, 0])]
    pts = [np.stack([x, b[:, 3]], axis=1) for x in xs]
    pts += [np.stack([x, b[:, 3] - 0.25 * h], axis=1) for x in (xs[0], xs[2])]
    flat = np.concatenate(pts)
    return warp_points(flat, H).reshape(5, -1, 2).transpose(1, 0, 2)


# ---------------------------------------------------------------------------
# pedestrians
# ---------------------------------------------------------------------------
def jaywalking(ctx: Context) -> list[Evidence]:
    p = PARAMS
    out = []
    for tr in ctx.people:
        # a box much shorter than a person standing there is cut off (legs hidden behind a car):
        # its bottom edge is not the feet, so those samples cannot put anyone on the road
        c0, cx, cy = S.PERSON_HEIGHT_PX
        expected = c0 + cx * tr.foot[:, 0] + cy * tr.foot[:, 1]
        whole = tr.height >= p["jay_min_height"] * expected
        road = (ctx.sample(ctx.walk, tr.foot) > 0) & whole
        kerb = ctx.sample(ctx.walk_dist, tr.foot)
        cw = ctx.sample(ctx.cw_dist, tr.foot)
        h = np.maximum(tr.height, 20.0)  # perspective: margins in units of the person's apparent height
        strict = road & (kerb > p["jay_margin_kerb"] * h) & (cw > p["jay_margin_cw"] * h)
        loose = road & (cw > 2.0)
        for s, e in runs(tr.t, strict, ctx.gap(p["jay_gap"])):
            if e - s < p["jay_min_dur"]:
                continue
            # boundaries from the enclosing loose run: the moment the foot left the kerb / zebra
            for ls, le in runs(tr.t, loose, ctx.gap(p["jay_gap"])):
                if ls <= s and le >= e:
                    s, e = ls, le
                    break
            out.append(Evidence("jaywalking", s, e, [tr.tid]))
    return out


def failure_to_yield(ctx: Context, H_work_to_ref: np.ndarray) -> list[Evidence]:
    p = PARAMS
    out = []
    k = int(2 * p["fty_cw_dilate"]) + 1
    for name, m in ctx.cw_masks.items():
        ped_zone = cv2.dilate(m, np.ones((k, k), np.uint8))
        # who is out on this crossing (not waiting on the kerb at its end), indexed by time (0.1 s bins)
        on_cw: dict[int, list[tuple[int, np.ndarray]]] = {}
        for ped in ctx.people:
            kerb = ctx.sample(ctx.road_dist, ped.foot)
            h = np.maximum(ped.height, 20.0)
            walking = ped.speed > p["fty_walk"] * h  # someone standing still beside the car's path is not being cut off
            hit = (ctx.sample(ped_zone, ped.foot) > 0) & (kerb > p["fty_kerb"] * h) & walking
            for t_, f_ in zip(ped.t[hit], ped.foot[hit]):
                on_cw.setdefault(int(round(t_ * 10)), []).append((ped.tid, f_))
        if not on_cw:
            continue
        for veh in ctx.vehicles:
            if veh.cls == MOTORCYCLE and np.median(veh.speed) < p["fty_moto_speed"]:
                continue  # scooters get walked along the zebras; only ones clearly being ridden count
            fp = _footprint_points(veh, H_work_to_ref)  # (N, 5, 2)
            on = (ctx.sample(m, fp.reshape(-1, 2)).reshape(len(veh.t), 5) > 0).any(axis=1)
            if not on.any():
                continue
            for s_, e_ in runs(veh.t, on, ctx.gap(0.3)):
                sel = (veh.t >= s_) & (veh.t <= e_)
                if e_ - s_ < 0.2 or np.median(veh.speed[sel]) < p["fty_min_speed"]:
                    continue
                victims = set()
                for i in np.nonzero((veh.t >= s_ - 0.3) & (veh.t <= e_))[0]:
                    for tid, f_ in on_cw.get(int(round(veh.t[i] * 10)), ()):
                        if np.linalg.norm(f_ - veh.foot[i]) < p["fty_near_px"]:
                            victims.add(tid)
                if victims:
                    # our footprint points sit low in the box; the convention runs from the front
                    # entering the zebra to the rear leaving it, which is a little longer
                    out.append(Evidence("failure_to_yield", s_ - p["fty_pad_start"], e_ + p["fty_pad_end"],
                                        [veh.tid, *sorted(victims)], note=name))
    return out


# ---------------------------------------------------------------------------
# signal-related
# ---------------------------------------------------------------------------
def _stop_line_crossings(ctx: Context, H: np.ndarray):
    """(trajectory, crossing time, front points) for SB vehicles crossing the stop line towards the camera."""
    (x1, _), (x2, _) = S.STOP_LINE_SB
    for veh in ctx.vehicles:
        front = _front_points(veh, H)
        side = S.signed_side(front, S.STOP_LINE_SB)
        for i in range(1, len(side)):
            if side[i - 1] < 0 <= side[i] and x1 - 20 <= front[i, 0] <= x2 + 20 and veh.vel[i, 1] > 0:
                a = side[i - 1] / (side[i - 1] - side[i])
                yield veh, float(veh.t[i - 1] + a * (veh.t[i] - veh.t[i - 1])), front
                break


def red_light(ctx: Context, H: np.ndarray) -> list[Evidence]:
    p = PARAMS
    out = []
    for veh, tc, _ in _stop_line_crossings(ctx, H):
        if ctx.red_since(tc) < p["red_settle"] or ctx.next_green(tc) - tc < p["red_before_green"]:
            continue
        # end: the car leaves the junction or the frame (its track ends), even if it waits inside first
        end = min(float(veh.t[-1]), tc + p["red_max_len"])
        out.append(Evidence("red_light", tc, end, [veh.tid], note=f"red for {ctx.red_since(tc):.1f}s"))
    return out


def stop_line(ctx: Context, H: np.ndarray) -> list[Evidence]:
    p = PARAMS
    out = []
    for veh in ctx.vehicles:
        front = _front_points(veh, H)
        past = S.signed_side(front, S.STOP_LINE_SB) > p["stop_line_margin"] * np.hypot(645, 70)
        in_zone = ctx.sample(ctx.zones["stop"], front) > 0
        stopped = veh.speed < p["stop_speed"]
        red = ctx.phases_at(veh.t) == "red"
        for s, e in runs(veh.t, past & in_zone & stopped & red, ctx.gap(0.4)):
            if e - s < p["stop_line_min"]:
                continue
            end = min(ctx.next_green(s), float(veh.t[-1]))
            out.append(Evidence("stop_line", s, end, [veh.tid]))
    return out


# ---------------------------------------------------------------------------
# stopped vehicles
# ---------------------------------------------------------------------------
def stopped_vehicle(ctx: Context) -> list[Evidence]:
    """A vehicle standing 10 s or more where traffic is supposed to flow, not in a signal queue.

    Where it stands decides what counts:
    * NB carriageway: there is no signal queue in view there, so a car standing in a lane
      (pick-ups, drop-offs, a car left in the kerb lane) counts. The bus stop does not.
    * Junction box (the SB path to the south exit): a car standing there is past its signal,
      alone or stuck in a jam, so it counts.
    * Everything else does not: the SB approach is the red-light queue (and the parking bay
      along its left edge), and at the right edge we only see the tail of queues waiting
      for their own signal or for people on the NB crossing.
    """
    p = PARAMS
    spans = []  # (start, end, x, y, tid)
    for veh in ctx.vehicles:
        on_road = ctx.sample(ctx.road, veh.foot) > 0
        for s, e in runs(veh.t, (veh.speed < p["stop_speed"]) & on_road, ctx.gap(0.5)):
            sel = (veh.t >= s) & (veh.t <= e)
            x, y = np.median(veh.foot[sel], axis=0)
            spans.append([s, e, float(x), float(y), veh.tid])
    # link fragments of the same standing car (tracker id switches while it stands still)
    spans.sort()
    linked: list[list] = []
    for sp in spans:
        for L in linked:
            if abs(L[2] - sp[2]) < 20 and abs(L[3] - sp[3]) < 20 and sp[0] - L[1] < 5.0:
                L[1] = max(L[1], sp[1])
                L[4].append(sp[4])
                break
        else:
            linked.append([sp[0], sp[1], sp[2], sp[3], [sp[4]]])
    out = []
    for s, e, x, y, tids in linked:
        if e - s < p["stopped_min"]:
            continue
        pt = np.array([[x, y]])
        zone = lambda name: ctx.sample(ctx.zones[name], pt)[0] > 0  # noqa: E731
        if zone("bus"):
            continue
        if not (zone("nb") or zone("flow_box")) or x > p["stopped_ne_corner_x"]:
            continue
        out.append(Evidence("stopped_vehicle", s, e, list(dict.fromkeys(tids)),
                            note="NB lane" if zone("nb") else "junction box"))
    return out


# ---------------------------------------------------------------------------
# congestion
# ---------------------------------------------------------------------------
def congestion(ctx: Context) -> list[Evidence]:
    """SB traffic standing still while it has green (the junction box is full and the south
    exit backs up), carried on into the red until the box clears."""
    p = PARAMS
    step = 0.5
    ts = np.arange(0.0, ctx.duration, step)
    approach = np.zeros(len(ts))
    box = np.zeros(len(ts))
    for v in ctx.vehicles:
        idx = np.round(v.t / step).astype(int)
        ok = (idx >= 0) & (idx < len(ts))
        still = v.speed < p["cong_speed"]
        in_app = ctx.sample(ctx.zones["flow_approach"], v.foot) > 0
        in_box = (ctx.sample(ctx.zones["flow_box"], v.foot) > 0) | (ctx.sample(ctx.zones["stop"], v.foot) > 0)
        for sel, acc in ((ok & still & in_app, approach), (ok & still & in_box, box)):
            np.add.at(acc, np.unique(idx[sel]), 1)
    phase = ctx.phases_at(ts)
    # seconds since the green started: the red-light queue needs a while to get going
    into_green = np.array([t - _green_start(ctx, t) for t in ts])
    settled = (phase == "green") & (into_green >= p["cong_after_green"])
    jam = (settled & (approach + box >= p["cong_green_n"])) | (box >= p["cong_red_n"])
    # a jam has to start on green; standing in the box on red only continues one
    out = []
    for s, e in runs(ts, jam, p["cong_gap"]):
        started_green = ctx.phase_at(s) in ("green", "yellow") and s - _green_start(ctx, s) >= p["cong_after_green"]
        if e - s >= p["cong_min"] and started_green:
            out.append(Evidence("congestion", s, e, [], note=f"max {int((approach + box)[(ts >= s) & (ts <= e)].max())} standing"))
    return out


def _green_start(ctx: Context, t: float) -> float:
    """Start of the green phase that is on (or last was on) at time t."""
    idx = np.nonzero((ctx.signal_t <= t) & (ctx.signal_phase == "green"))[0]
    if not len(idx):
        return -1e9
    j = idx[-1]
    while j > 0 and ctx.signal_phase[j - 1] == "green":
        j -= 1
    return float(ctx.signal_t[j])


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------
def wrong_way(ctx: Context) -> list[Evidence]:
    p = PARAMS
    out = []
    movers = ctx.vehicles + ctx.two_wheelers
    for name, (_, heading) in DIRECTION_ZONES.items():
        for tr in movers:
            inside = ctx.sample(ctx.zones[name], tr.foot) > 0
            if not inside.any():
                continue
            ang = np.degrees(np.arctan2(tr.vel[:, 1], tr.vel[:, 0]))
            diff = np.abs((ang - heading + 180) % 360 - 180)
            bad = inside & (tr.speed > p["ww_min_speed"]) & (diff > p["ww_angle"])
            for s, e in runs(tr.t, bad, ctx.gap(0.5)):
                if e - s >= p["ww_min_dur"]:
                    out.append(Evidence("wrong_way", s, e, [tr.tid], note=name))
    return out


def u_turns(ctx: Context) -> list[Evidence]:
    """Southbound vehicles that turn round the median nose and leave northbound.

    Start: the heading leaves the southbound direction (the turn begins).
    End: the heading settles on the northbound direction (the turn is done).
    Left turns into the east arm end heading east and are not counted; nor are
    cars from the cross street turning into the NB carriageway (they start
    heading east).
    """
    p = PARAMS
    out = []
    nose = np.array(S.MEDIAN_NOSE, float)
    sb, nb = DIRECTION_ZONES["sb"][1], DIRECTION_ZONES["nb"][1]
    for veh in ctx.vehicles:
        moving = veh.speed > 25
        if moving.sum() < 15:
            continue
        t = veh.t[moving]
        head = np.degrees(np.arctan2(veh.vel[moving, 1], veh.vel[moving, 0])) % 360
        dev = lambda h, ref: np.abs((h - ref + 180) % 360 - 180)  # noqa: E731
        first, last = head[: max(5, len(head) // 8)], head[-max(5, len(head) // 8):]
        if np.median(dev(first, sb)) > p["ut_dir_tol"] or np.median(dev(last, nb)) > p["ut_dir_tol"]:
            continue
        if np.min(np.linalg.norm(veh.foot - nose, axis=1)) > p["ut_nose_px"]:
            continue
        came_from_sb = (ctx.sample(ctx.zones["sb"], veh.foot[moving][:10]) > 0).any() or \
            (ctx.sample(ctx.zones["stop"], veh.foot[moving][:10]) > 0).any()
        if not came_from_sb:
            continue
        off_sb = np.nonzero(dev(head, sb) > p["ut_turn_deg"])[0]
        on_nb = np.nonzero(dev(head, nb) < p["ut_turn_deg"])[0]
        if not len(off_sb) or not len(on_nb):
            continue
        s_, e_ = float(t[off_sb[0]]), float(t[on_nb[on_nb >= off_sb[0]][0]]) if (on_nb >= off_sb[0]).any() else None
        if e_ is None or e_ - s_ < p["ut_min_dur"]:
            continue
        out.append(Evidence("illegal_u_turn", s_, e_, [veh.tid], note="U-turn round the median nose"))
    return out


# Collisions. The foot points in this oblique view are rough, so two cars in adjacent lanes often
# look in contact; what tells a crash from traffic is what follows: road users that met at speed
# and then stand still together. Checked on our samples (no crash, no evidence) and on public CCTV
# crash clips (tools/crash_check.py). Metres, not pixels, because the rule also runs on other
# cameras there: `mpp(x, y)` is metres per pixel at a point of the trajectories' plane.
CRASH = {
    "contact": 1.3,       # x the sum of the two footprint radii (risk.RADIUS_M): foot points are rough
    "closing": 3.0,       # m/s towards each other in the 0.5 s before contact
    "pre_speed": 3.0,     # m/s median speed of the faster one over the 1 s before: not a box jump
    "stop_speed": 1.0,    # m/s: both count as standing below this
    "stop_within": 3.0,   # s after contact by which both stand
    "stay": 2.0,          # s they then stand together
    "near": 4.0,          # m apart at most while they stand
    "kick": 1.5,          # m/s: both velocities change across the contact (a car stopping behind a
                          # standing one changes its own velocity only; an impact moves both)
    "end_pad": 1.0,       # s after they come to rest: the event ends when all involved stop moving
}


def collisions(trajectories: list[Trajectory], mpp, p: dict = CRASH) -> list[Evidence]:
    from .risk import RADIUS_M

    step = 0.1
    tracks = []
    for tr in trajectories:
        if len(tr.t) < 5 or not (tr.is_vehicle or tr.is_person or tr.group == "bicycle"):
            continue
        keys, first = np.unique(np.round(tr.t / step).astype(int), return_index=True)
        tracks.append((tr, keys, first))
    out = []
    for i in range(len(tracks)):
        A, ka, fa = tracks[i]
        for j in range(i + 1, len(tracks)):
            B, kb, fb = tracks[j]
            if not (A.is_vehicle or B.is_vehicle):
                continue
            common, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
            if len(common) < 10:
                continue
            ia, ib = fa[ia], fb[ib]  # sample indices on the shared 0.1 s grid
            pa, pb = A.foot[ia], B.foot[ib]
            if np.min(np.abs(pa - pb).max(axis=1)) > 400:  # never within 400 px of each other
                continue
            m = np.array([mpp(*q) for q in (pa + pb) / 2])
            d = np.linalg.norm(pa - pb, axis=1) * m
            sa = np.linalg.norm(A.vel[ia], axis=1) * m
            sb = np.linalg.norm(B.vel[ib], axis=1) * m
            sep = (pa - pb) * m[:, None]
            closing = -np.sum((A.vel[ia] - B.vel[ib]) * m[:, None] * sep, axis=1) / np.maximum(d, 1e-3)
            ts = common * step
            reach = (RADIUS_M.get(A.cls, 1.0) + RADIUS_M.get(B.cls, 1.0)) * p["contact"]
            for n in np.nonzero(d <= reach)[0]:
                before = (ts >= ts[n] - 1.0) & (ts < ts[n])
                if before.sum() < 3 or closing[(ts >= ts[n] - 0.5) & (ts < ts[n])].max(initial=0.0) < p["closing"]:
                    continue
                if max(np.median(sa[before]), np.median(sb[before])) < p["pre_speed"]:
                    continue
                after = (ts > ts[n]) & (ts <= ts[n] + 0.7)
                if after.sum() < 2:
                    continue
                va, vb = A.vel[ia] * m[:, None], B.vel[ib] * m[:, None]
                kick = [np.linalg.norm(v[after].mean(axis=0) - np.median(v[before], axis=0)) for v in (va, vb)]
                if min(kick) < p["kick"]:
                    continue
                still = (sa < p["stop_speed"]) & (sb < p["stop_speed"]) & (d < p["near"])
                rest = np.nonzero(still & (ts > ts[n]) & (ts <= ts[n] + p["stop_within"]))[0]
                if not len(rest):
                    continue
                t_rest = ts[rest[0]]
                window = (ts >= t_rest) & (ts <= t_rest + p["stay"])
                if ts[window].max() - t_rest < p["stay"] - 0.3 or still[window].mean() < 0.8:
                    continue
                out.append(Evidence("accident", float(ts[n]), float(t_rest + p["end_pad"]), [A.tid, B.tid],
                                    note=f"met at {closing[before].max():.0f} m/s"))
                break
    return out


def accident(ctx: Context) -> list[Evidence]:
    return collisions(ctx.trajectories, S.metres_per_px)
