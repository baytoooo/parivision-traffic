"""Exploratory analysis of the sample clips -> figures and eda.json for the website.

    python tools/eda.py --out out/site_data

Uses the cached detections/tracks (tools/cache_detections.py, tools/tracks_from_cache.py),
the per-clip alignment, the signal timelines and the proxies in cache/proxy (tools/align_cache.py).
Clip metadata comes from ffprobe, so ffmpeg must be on PATH (Homebrew or apt both ship it).
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision import scene as S  # noqa: E402
from parivision.signal import fill_phases  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from common import SAMPLES  # noqa: E402
from run_rules import load_context  # noqa: E402

CLIPS = ["C3896", "C3897", "C3902", "C3905"]
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BIN = 5.0


def clip_meta(clip: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=width,height,r_frame_rate,codec_name,profile,pix_fmt,bit_rate:format=duration:format_tags=creation_time",
                          "-of", "json", str(SAMPLES / f"{clip}.MP4")], capture_output=True, text=True)
    d = json.loads(out.stdout)
    v = d["streams"][0]
    num, den = map(int, v["r_frame_rate"].split("/"))
    return {"id": clip, "width": v["width"], "height": v["height"], "fps": round(num / den, 3),
            "duration": round(float(d["format"]["duration"]), 1), "codec": f"{v['codec_name']} {v.get('profile', '')}",
            "pix_fmt": v.get("pix_fmt"), "bitrate_mbps": round(int(v.get("bit_rate", 0)) / 1e6, 1),
            "created_utc": d["format"].get("tags", {}).get("creation_time")}


def signal_stats(t: np.ndarray, phases: list[str]) -> dict:
    """Phase lengths from onset to onset over complete phases, the cycle, and unreadable time.

    Blips under 1 s (a car passing in front of the head) are folded into the phase before them.
    """
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    segs: list[list] = []
    for ti, p in zip(t.tolist(), list(phases)):
        if segs and segs[-1][2] == p:
            segs[-1][1] = round(ti, 1)
        else:
            segs.append([round(ti, 1), round(ti, 1), p])
    steady: list[list] = []
    for a, b, p in segs:
        if steady and (b - a < 1.0 or steady[-1][2] == p):
            steady[-1][1] = b
        else:
            steady.append([a, b, p])
    # Only phases with a seen start and a seen end count: not the one already on when the clip
    # starts, and not one next to an unreadable spell. The afternoon plan's red-and-yellow before
    # green reads as yellow and belongs to red.
    kinds = [q for _, _, q in steady]
    greens, yellows, reds, g_on = [], [], [], []
    for i in range(1, len(steady)):
        a, _, q = steady[i]
        prev = kinds[i - 1]
        nxt, start_next = (kinds[i + 1], steady[i + 1][0]) if i + 1 < len(steady) else (None, None)
        if q == "green" and prev in ("red", "yellow"):
            g_on.append(a)
            if nxt == "yellow":
                greens.append(start_next - a)
        elif q == "yellow" and prev == "green" and nxt == "red":
            yellows.append(start_next - a)
        elif q == "red" and prev in ("yellow", "green"):
            if nxt == "green":
                reds.append(start_next - a)
            elif nxt == "yellow" and i + 2 < len(steady) and kinds[i + 2] == "green":
                reds.append(steady[i + 2][0] - a)
    g = np.array(g_on)
    mean = lambda v: round(float(np.mean(v)), 1) if len(v) else None  # noqa: E731
    return {"segments": segs, "green": mean(greens), "yellow": mean(yellows), "red": mean(reds),
            "cycle": round(float(np.median(np.diff(g))), 1) if len(g) > 1 else None, "cycles": len(reds),
            "unknown_sec": round(float(sum(b - a + dt for a, b, p in segs if p == "unknown")), 1)}


def heat(points: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    m = np.zeros((S.REF_SIZE[1], S.REF_SIZE[0]), np.float32)
    p = np.round(points).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < S.REF_SIZE[0]) & (p[:, 1] >= 0) & (p[:, 1] < S.REF_SIZE[1])
    np.add.at(m, (p[ok, 1], p[ok, 0]), 1)
    return cv2.GaussianBlur(m, (0, 0), sigma)


def overlay_heat(bg: np.ndarray, m: np.ndarray, cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    h = np.log1p(m * 40)
    h = (h / (h.max() + 1e-9) * 255).astype(np.uint8)
    col = cv2.applyColorMap(h, cmap)
    alpha = (h.astype(np.float32) / 255)[..., None] ** 0.6
    return (bg * (1 - alpha * 0.85) + col * alpha * 0.85).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/site_data")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "media/eda").mkdir(parents=True, exist_ok=True)
    bg = cv2.imread(str(ROOT / "src/parivision/assets/reference_day.jpg"))
    dark = (bg * 0.42).astype(np.uint8)

    eda = {"clips": [], "counts": {}, "density": {}, "signal": {}, "light": {}, "pedestrians": {}, "images": {}}
    all_veh, all_ped, veh_vel = [], [], []
    ped_stats = {"on_crossing": 0, "off_crossing_on_road": 0, "pavement": 0}
    for clip in CLIPS:
        meta = clip_meta(clip)
        eda["clips"].append(meta)
        ctx, _ = load_context(clip)
        tracks = ctx.trajectories
        pedestrians = {id(tr) for tr in ctx.people}  # riders and people seen through car windows left out
        z = np.load(ROOT / "cache/det" / f"{clip}__yolo26m_1280_1920_10fps.npz")  # the submitted detector
        det = z["det"]
        det = det[det[:, 6] >= 0.35]
        dur = meta["duration"]
        bins = np.arange(0, dur + BIN, BIN)
        counts = {"t": bins[:-1].tolist()}
        n_frames_bin = np.histogram(np.unique(det[:, 1]), bins)[0].clip(min=1)
        for cid, name in COCO.items():
            sel = det[det[:, 7] == cid]
            counts[name] = np.round(np.histogram(sel[:, 1], bins)[0] / n_frames_bin, 2).tolist()
        eda["counts"][clip] = counts

        # moving vs standing vehicles per bin, and people on the carriageway
        masks = S.masks()
        moving, standing = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
        for tr in tracks:
            if tr.is_vehicle:
                all_veh.append(tr.foot[tr.speed > 40])
                veh_vel.append(np.concatenate([tr.foot, tr.vel], axis=1)[tr.speed > 60])
                idx = np.clip(np.digitize(tr.t, bins) - 1, 0, len(bins) - 2)
                np.add.at(moving, idx[tr.speed > 40], 1)
                np.add.at(standing, idx[tr.speed <= 40], 1)
            elif id(tr) in pedestrians:
                all_ped.append(tr.foot)
                p = np.round(tr.foot).astype(int)
                ok = (p[:, 0] >= 0) & (p[:, 0] < 1920) & (p[:, 1] >= 0) & (p[:, 1] < 1080)
                p = p[ok]
                on_cw = masks["crosswalk"][p[:, 1], p[:, 0]] > 0
                on_road = masks["walk_check"][p[:, 1], p[:, 0]] > 0
                ped_stats["on_crossing"] += int(on_cw.sum())
                ped_stats["off_crossing_on_road"] += int((on_road & ~on_cw).sum())
                ped_stats["pavement"] += int((~on_road & ~on_cw).sum())
        per_bin = n_frames_bin
        eda["density"][clip] = {"t": bins[:-1].tolist(), "vehicles_moving": np.round(moving / per_bin, 2).tolist(),
                                "vehicles_standing": np.round(standing / per_bin, 2).tolist()}

        sig = json.loads((ROOT / "cache/signal" / f"{clip}.json").read_text())
        eda["signal"][clip] = signal_stats(np.array(sig["times"]), fill_phases(sig["phases"], np.array(sig["times"])))
        cap = cv2.VideoCapture(str(ROOT / "cache/proxy" / f"{clip}.mp4"))
        lum = []
        for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 100):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                lum.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
        eda["light"][clip] = {"mean_luma": round(float(np.mean(lum)), 1)}

    eda["pedestrians"] = {k: round(v / max(1, sum(ped_stats.values())), 4) for k, v in ped_stats.items()}

    # figures
    V, P = heat(np.concatenate(all_veh)), heat(np.concatenate(all_ped))
    cv2.imwrite(str(out / "media/eda/heatmap_vehicle.jpg"), overlay_heat(dark, V), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / "media/eda/heatmap_person.jpg"), overlay_heat(dark, P, cv2.COLORMAP_VIRIDIS), [cv2.IMWRITE_JPEG_QUALITY, 85])
    traj = dark.copy()
    for clip in CLIPS:
        for tr in pickle.load(open(ROOT / "cache/tracks" / f"{clip}.pkl", "rb"))["trajectories"]:
            if not (tr.is_vehicle or tr.is_person) or len(tr.t) < 15:
                continue
            ang = np.degrees(np.arctan2(tr.vel[:, 1], tr.vel[:, 0])) % 360
            pts = tr.foot.astype(np.int32)
            for i in range(1, len(pts)):
                if tr.speed[i] < 15:
                    continue
                col = cv2.cvtColor(np.uint8([[[int(ang[i] / 2), 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
                cv2.line(traj, tuple(pts[i - 1]), tuple(pts[i]), col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "media/eda/trajectories.jpg"), traj, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # direction field: mean heading per 40 px cell, arrow length = consistency
    field = dark.copy()
    vv = np.concatenate(veh_vel)
    cell = 40
    for gy in range(0, 1080, cell):
        for gx in range(0, 1920, cell):
            sel = (vv[:, 0] >= gx) & (vv[:, 0] < gx + cell) & (vv[:, 1] >= gy) & (vv[:, 1] < gy + cell)
            if sel.sum() < 30:
                continue
            u = vv[sel, 2:4] / (np.linalg.norm(vv[sel, 2:4], axis=1, keepdims=True) + 1e-6)
            m = u.mean(axis=0)
            r = float(np.linalg.norm(m))
            c = (gx + cell // 2, gy + cell // 2)
            ang = np.degrees(np.arctan2(m[1], m[0])) % 360
            col = cv2.cvtColor(np.uint8([[[int(ang / 2), 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            e = (int(c[0] + m[0] / (r + 1e-6) * 16 * r), int(c[1] + m[1] / (r + 1e-6) * 16 * r))
            cv2.arrowedLine(field, c, e, col, 2, cv2.LINE_AA, tipLength=0.35)
    cv2.imwrite(str(out / "media/eda/directions.jpg"), field, [cv2.IMWRITE_JPEG_QUALITY, 85])
    eda["images"] = {k: f"/media/eda/{k}.jpg" for k in ("heatmap_vehicle", "heatmap_person", "trajectories", "directions")}
    (out / "eda.json").write_text(json.dumps(eda))
    print(json.dumps({k: eda[k] for k in ("pedestrians", "light")}, indent=1))
    print({c: {k: v for k, v in eda["signal"][c].items() if k != "segments"} for c in CLIPS})


if __name__ == "__main__":
    main()
