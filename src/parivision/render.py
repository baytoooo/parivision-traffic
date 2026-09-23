"""Annotated playback: boxes, tracks, scene layout, signal, active events and the risk curve.

Used for the sample-video renders on the website (tools/make_site_data.py) and by
the local demo server (demo/worker.py).
Output is H.264 (PyAV / libx264), small enough to stream in a browser.
"""
from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np

from . import scene as S
from .registration import warp_points
from .video import sample_frames

CLASS_COLORS = {  # BGR
    "accident": (40, 40, 230), "near_miss": (60, 120, 255), "red_light": (50, 50, 255),
    "wrong_way": (200, 60, 200), "illegal_u_turn": (220, 120, 180), "stopped_vehicle": (0, 190, 255),
    "jaywalking": (0, 220, 255), "failure_to_yield": (60, 170, 255), "illegal_turn": (180, 110, 255),
    "solid_line_crossing": (255, 180, 80), "stop_line": (80, 80, 255), "congestion": (120, 120, 120),
    "road_obstacle": (60, 200, 120), "fire_smoke": (30, 30, 180),
}
GROUP_COLORS = {"vehicle": (235, 200, 120), "person": (140, 235, 140), "bicycle": (230, 170, 250), "animal": (90, 200, 255)}
PHASE_COLORS = {"red": (60, 60, 235), "green": (90, 210, 90), "yellow": (40, 200, 240), "unknown": (150, 150, 150)}


def _scene_overlay(shape, H_ref_to_frame: np.ndarray) -> np.ndarray:
    """Crossings and stop line drawn once, blended into every frame."""
    over = np.zeros(shape, np.uint8)
    for poly in S.CROSSWALKS.values():
        pts = warp_points(np.asarray(poly, float), H_ref_to_frame).astype(np.int32)
        cv2.polylines(over, [pts], True, (255, 190, 90), 2, cv2.LINE_AA)
    line = warp_points(np.asarray(S.STOP_LINE_SB, float), H_ref_to_frame).astype(np.int32)
    cv2.polylines(over, [line], False, (80, 80, 255), 2, cv2.LINE_AA)
    return over


def _label(img, text, org, color, scale=0.45):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def render(video_path: str, out_path: str, trajectories, events, evidence, signal_t, signal_phase,
           H_work_to_ref: np.ndarray, work_width: int, risk: list | None = None,
           out_width: int = 1280, fps: float = 10.0, max_seconds: float | None = None, frames=None,
           duration: float | None = None) -> str:
    """Write an annotated MP4. Boxes in ``trajectories`` are in ``work_width`` pixels.

    ``frames`` may be an iterable of ``(t_sec, bgr)`` already at ``out_width``
    (the demo keeps them from the analysis pass); otherwise the video is read again.
    """
    k = out_width / work_width
    H_ref_to_out = np.diag([k, k, 1.0]) @ np.linalg.inv(H_work_to_ref)
    by_time = {}
    for tr in trajectories:
        for i, t in enumerate(tr.t):
            by_time.setdefault(round(float(t), 1), []).append((tr, i))
    actor_label = {}
    for ev in evidence:
        for tid in ev.actors:
            actor_label.setdefault(tid, []).append(ev)
    risk_t = np.array([r[0] for r in risk]) if risk else np.zeros(0)
    risk_v = np.array([r[1] for r in risk]) if risk else np.zeros(0)
    duration = duration or max([e[1] for e in events], default=0.0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(out_path, mode="w")
    stream = None
    overlay = None
    panel_h = 70
    source = frames if frames is not None else ((t, f) for _, t, f in sample_frames(video_path, fps, out_width))
    for t, frame in source:
        if max_seconds is not None and t > max_seconds:
            break
        if stream is None:
            h, w = frame.shape[:2]
            stream = container.add_stream("libx264", rate=int(round(fps)))
            stream.width, stream.height = w, h + panel_h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "26", "preset": "veryfast", "movflags": "+faststart"}
            overlay = _scene_overlay(frame.shape, H_ref_to_out)
            duration = max(duration, 1.0)
        img = cv2.addWeighted(frame, 1.0, overlay, 0.55, 0)
        for tr, i in by_time.get(round(float(t), 1), []):
            x1, y1, x2, y2 = (tr.box[i] * k).astype(int)
            active = [ev for ev in actor_label.get(tr.tid, []) if ev.start - 0.05 <= t <= ev.end + 0.05]
            color = CLASS_COLORS[active[0].label] if active else GROUP_COLORS.get(tr.group, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if active else 1, cv2.LINE_AA)
            if active:
                _label(img, active[0].label.replace("_", " "), (x1, max(14, y1 - 2)), color)
        # signal lamp
        j = int(np.clip(np.searchsorted(signal_t, t), 0, max(0, len(signal_t) - 1))) if len(signal_t) else None
        phase = str(signal_phase[j]) if j is not None else "unknown"
        cv2.circle(img, (24, 24), 11, PHASE_COLORS.get(phase, (150, 150, 150)), -1, cv2.LINE_AA)
        cv2.putText(img, f"{t:6.1f}s  signal {phase}", (42, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        # bottom panel: event timeline + risk
        panel = np.full((panel_h, img.shape[1], 3), 24, np.uint8)
        W = img.shape[1]
        for s, e, lab in events:
            x0, x1_ = int(s / duration * W), max(int(s / duration * W) + 2, int(e / duration * W))
            cv2.rectangle(panel, (x0, 8), (x1_, 30), CLASS_COLORS.get(lab, (200, 200, 200)), -1)
        live = [lab for s, e, lab in events if s <= t <= e]
        cv2.putText(panel, "  ".join(sorted(set(live))).replace("_", " ") or "no event", (8, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        if len(risk_t):
            sel = risk_t <= t
            if sel.any():
                pts = np.stack([risk_t[sel] / duration * W, 66 - risk_v[sel] * 30], axis=1).astype(np.int32)
                cv2.polylines(panel, [pts[:: max(1, len(pts) // 2000)]], False, (80, 80, 255), 1, cv2.LINE_AA)
        cv2.line(panel, (int(t / duration * W), 0), (int(t / duration * W), panel_h), (255, 255, 255), 1)
        out = np.vstack([img, panel])
        vf = av.VideoFrame.from_ndarray(out, format="bgr24")
        for packet in stream.encode(vf):
            container.mux(packet)
    if stream is not None:
        for packet in stream.encode():
            container.mux(packet)
    container.close()
    return out_path
