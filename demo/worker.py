"""Runs one demo job: events, causal risk curve and an annotated video for an uploaded clip."""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.detector import COCO_NAMES  # noqa: E402
from parivision.pipeline import analyse, profile  # noqa: E402
from parivision.registration import Alignment  # noqa: E402
from parivision.render import render  # noqa: E402
from parivision.risk import Anticipator  # noqa: E402

MAX_SECONDS = 120.0
RENDER_WIDTH = 960


@dataclass
class Job:
    id: str
    video: Path
    workdir: Path
    status: str = "queued"
    stage: str = "waiting in the queue"
    progress: float = 0.0
    started: float = 0.0
    error: str | None = None
    result: dict | None = field(default=None, repr=False)

    def public(self) -> dict:
        eta = None
        if self.status == "running" and self.progress > 0.05:
            spent = time.time() - self.started
            eta = int(spent / self.progress * (1 - self.progress))
        return {"status": self.status, "progress": round(self.progress, 3), "stage": self.stage,
                "eta_sec": eta, "error": self.error, "result": self.result}


def run(job: Job) -> None:
    job.status, job.started = "running", time.time()

    def progress(stage: str, frac: float) -> None:
        # detection is ~85% of the work, rendering the rest
        job.stage, job.progress = stage, 0.85 * frac

    fps = profile()[2]  # the rate analyse() samples at on this machine: 10 fps on a GPU, 5 on CPU
    risk_model: Anticipator | None = None
    risk: list[list[float]] = []
    kept: list[tuple[float, bytes]] = []  # JPEG copies of the analysed frames at render size
    per_bin: dict[int, dict[str, int]] = {}   # 5 s bin -> class -> detections
    frames_in_bin: dict[int, int] = {}

    def on_frame(t: float, det, img, H) -> None:
        # the risk model sees the same detections in time order: still causal, no second detector pass
        nonlocal risk_model
        shape = img.shape[:2]
        small = cv2.resize(img, (RENDER_WIDTH, int(round(img.shape[0] * RENDER_WIDTH / img.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        kept.append((t, cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()))
        if risk_model is None:
            risk_model = Anticipator({"fps": fps, "width": shape[1], "height": shape[0]})
            risk_model.alignment = Alignment(H, 1, True)
        risk.append([round(t, 2), round(risk_model.observe(det, shape, t), 4)])
        b = int(t // 5)
        frames_in_bin[b] = frames_in_bin.get(b, 0) + 1
        row = per_bin.setdefault(b, {})
        for c in det.cls[det.conf >= 0.35]:
            name = COCO_NAMES.get(int(c))
            if name:
                row[name] = row.get(name, 0) + 1

    try:
        a = analyse(str(job.video), progress=progress, max_seconds=MAX_SECONDS, on_frame=on_frame, time_share=6.0)
        job.stage, job.progress = "rendering the annotated video", 0.86
        out_video = job.workdir / "annotated.mp4"
        decoded = ((t, cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)) for t, b in kept)
        render(str(job.video), str(out_video), a.trajectories, a.events, a.evidence, a.signal_t, a.signal_phase,
               a.alignment.H, a.work_size[0], risk=risk, out_width=RENDER_WIDTH, fps=fps, frames=decoded,
               duration=min(a.info.duration, MAX_SECONDS))
        signal = []
        for t, p in zip(a.signal_t.tolist(), a.signal_phase.tolist()):
            if signal and signal[-1][2] == p:
                signal[-1][1] = round(t, 1)
            else:
                signal.append([round(t, 1), round(t, 1), p])
        job.result = {
            "clip": job.video.name,
            "duration": round(min(a.info.duration, MAX_SECONDS), 2),
            "events": a.events,
            "risk": risk,
            "signal": signal,
            "evidence": [{"label": e.label, "start": round(e.start, 2), "end": round(e.end, 2), "actors": e.actors,
                          "note": e.note} for e in a.evidence],
            "counts": {"t": [b * 5 for b in sorted(per_bin)],
                       **{name: [round(per_bin[b].get(name, 0) / frames_in_bin[b], 2) for b in sorted(per_bin)]
                          for name in ("car", "bus", "truck", "motorcycle", "person", "bicycle")}},
            "aligned": bool(a.alignment.ok),
            "video": f"/api/jobs/{job.id}/video",
        }
        (job.workdir / "result.json").write_text(json.dumps(job.result))
        job.status, job.stage, job.progress = "done", "done", 1.0
    except Exception as exc:  # reported to the page, never crashes the server
        traceback.print_exc()
        job.status, job.error, job.stage = "error", f"{type(exc).__name__}: {exc}", "failed"
