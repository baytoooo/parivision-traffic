"""Exploratory analysis of the sample clips -> figures and eda.json for the website.

    python tools/eda.py --out out/site_data

Uses the cached detections/tracks (tools/cache_detections.py, tools/tracks_from_cache.py),
the per-clip alignment and the signal timelines.
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
from parivision.rules import DIRECTION_ZONES  # noqa: E402

CLIPS = ["C3896", "C3897", "C3902", "C3905"]
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BIN = 5.0


def clip_meta(clip: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=width,height,r_frame_rate,codec_name,profile,pix_fmt,bit_rate:format=duration:format_tags=creation_time",
                          "-of", "json", str(ROOT / "kit/samples" / f"{clip}.MP4")], capture_output=True, text=True)
    d = json.loads(out.stdout)
    v = d["streams"][0]
    num, den = map(int, v["r_frame_rate"].split("/"))
    return {"id": clip, "width": v["width"], "height": v["height"], "fps": round(num / den, 3),
            "duration": round(float(d["format"]["duration"]), 1), "codec": f"{v['codec_name']} {v.get('profile', '')}",
            "pix_fmt": v.get("pix_fmt"), "bitrate_mbps": round(int(v.get("bit_rate", 0)) / 1e6, 1),
            "created_utc": d["format"].get("tags", {}).get("creation_time")}


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

    eda = {"clips": [], "counts": {}, "density": {}, "speeds": {}, "signal": {}, "light": {}, "pedestrians": {},
           "images": {}, "findings": []}
    all_veh, all_ped, veh_vel = [], [], []
    ped_stats = {"on_crossing": 0, "off_crossing_on_road": 0, "pavement": 0}
    for clip in CLIPS:
        meta = clip_meta(clip)
        eda["clips"].append(meta)
        tracks = pickle.load(open(ROOT / "cache/tracks" / f"{clip}.pkl", "rb"))["trajectories"]
        z = np.load(next((ROOT / "cache/det").glob(f"{clip}__*.npz")))
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
            elif tr.is_person:
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

        # speeds by zone (m/s via the local scale)
        zones = {name: np.zeros((1080, 1920), np.uint8) for name in ("sb", "nb")}
        for name in zones:
            cv2.fillPoly(zones[name], [np.asarray(DIRECTION_ZONES[name][0], np.int32)], 1)
        box = np.zeros((1080, 1920), np.uint8)
        cv2.fillPoly(box, [np.asarray(S.JUNCTION_BOX, np.int32)], 1)
        zones["junction"] = box
        sp = {k: [] for k in zones}
        for tr in tracks:
            if not tr.is_vehicle:
                continue
            p = np.round(tr.foot).astype(int)
            ok = (p[:, 0] >= 0) & (p[:, 0] < 1920) & (p[:, 1] >= 0) & (p[:, 1] < 1080)
            for k, m in zones.items():
                inside = np.zeros(len(p), bool)
                inside[ok] = m[p[ok, 1], p[ok, 0]] > 0
                sel = inside & (tr.speed > 20)
                if sel.any():
                    mpp = np.array([S.metres_per_px(x, y) for x, y in tr.foot[sel]])
                    sp[k] += (tr.speed[sel] * mpp * 3.6).round(1).tolist()  # km/h
        eda["speeds"][clip] = {k: {"p10": round(float(np.percentile(v, 10)), 1), "p50": round(float(np.median(v)), 1),
                                   "p90": round(float(np.percentile(v, 90)), 1), "n": len(v)} if v else None
                               for k, v in sp.items()}

        sig = json.loads((ROOT / "cache/signal" / f"{clip}.json").read_text())
        segs = [s for s in sig["segments"] if s[1] - s[0] > 1.0]
        greens = [b - a for a, b, p in segs[1:-1] if p == "green"]
        reds = [b - a for a, b, p in segs[1:-1] if p == "red"]
        yellows = [b - a for a, b, p in segs[1:-1] if p == "yellow"]
        eda["signal"][clip] = {"segments": sig["segments"], "green": round(float(np.mean(greens)), 1) if greens else None,
                               "red": round(float(np.mean(reds)), 1) if reds else None,
                               "yellow": round(float(np.mean(yellows)), 1) if yellows else None}
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
    print(json.dumps({k: eda[k] for k in ("speeds", "pedestrians", "light")}, indent=1))
    print({c: {k: v for k, v in eda["signal"][c].items() if k != "segments"} for c in CLIPS})


if __name__ == "__main__":
    main()
