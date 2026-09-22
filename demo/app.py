"""Web API for the live demo (runs on a Hugging Face Space, CPU).

Endpoints match site/BRIEF.md:
  GET  /api/health
  GET  /api/samples
  POST /api/jobs                 multipart "file" (mp4, up to 500 MB, first 120 s analysed)
  POST /api/jobs/sample/{name}
  GET  /api/jobs/{id}
  GET  /api/jobs/{id}/video
  GET  /api/jobs/{id}/result.json

One worker thread processes jobs in arrival order; the page polls for progress.
"""
from __future__ import annotations

import os
import shutil
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from worker import Job, run

MAX_BYTES = 500 * 1024 * 1024
KEEP_JOBS = 40
WORK = Path(os.environ.get("DEMO_WORKDIR", "/tmp/parivision-demo"))
SAMPLES = Path(__file__).resolve().parent / "samples"
SAMPLE_LIST = [
    {"name": "north_crossing_midday", "label": "Midday, pedestrians on the north crossing", "file": "north_crossing_midday.mp4"},
    {"name": "west_crossing_turns", "label": "Right turns over the west crossing", "file": "west_crossing_turns.mp4"},
    {"name": "dusk_queue", "label": "Dusk, red-light queue and U-turn", "file": "dusk_queue.mp4"},
]

app = FastAPI(title="PariVision traffic events demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: "OrderedDict[str, Job]" = OrderedDict()
queue: "Queue[Job]" = Queue()


def _worker() -> None:
    while True:
        job = queue.get()
        run(job)
        queue.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _submit(video: Path, workdir: Path) -> dict:
    job = Job(id=workdir.name, video=video, workdir=workdir)
    jobs[job.id] = job
    while len(jobs) > KEEP_JOBS:  # forget the oldest jobs and their files
        _, old = jobs.popitem(last=False)
        shutil.rmtree(old.workdir, ignore_errors=True)
    queue.put(job)
    return {"job_id": job.id}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "queued": queue.qsize()}


@app.get("/api/samples")
def samples() -> list[dict]:
    out = []
    for s in SAMPLE_LIST:
        p = SAMPLES / s["file"]
        if p.exists():
            out.append({"name": s["name"], "label": s["label"], "seconds": _seconds(p)})
    return out


def _seconds(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return round(n / fps, 1)


@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile = File(...)) -> dict:
    length = int(request.headers.get("content-length") or 0)
    if length > MAX_BYTES + 1024 * 1024:
        raise HTTPException(413, "File too large: the demo accepts up to 500 MB.")
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".m4v")):
        raise HTTPException(415, "Please upload an .mp4 file.")
    workdir = WORK / uuid.uuid4().hex[:12]
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "input.mp4"
    size = 0
    with target.open("wb") as fh:
        while chunk := await file.read(4 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                fh.close()
                shutil.rmtree(workdir, ignore_errors=True)
                raise HTTPException(413, "File too large: the demo accepts up to 500 MB.")
            fh.write(chunk)
    return _submit(target, workdir)


@app.post("/api/jobs/sample/{name}")
def sample_job(name: str) -> dict:
    match = next((s for s in SAMPLE_LIST if s["name"] == name), None)
    if match is None or not (SAMPLES / match["file"]).exists():
        raise HTTPException(404, "Unknown sample.")
    workdir = WORK / uuid.uuid4().hex[:12]
    workdir.mkdir(parents=True, exist_ok=True)
    return _submit(SAMPLES / match["file"], workdir)


def _job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (the server may have restarted).")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = _job(job_id)
    body = job.public()
    if job.status == "queued":
        ahead = [j for j in list(jobs.values()) if j.status in ("queued", "running")]
        body["stage"] = f"waiting in the queue ({max(0, ahead.index(job))} ahead)"
    return JSONResponse(body)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = _job(job_id)
    path = job.workdir / "annotated.mp4"
    if job.status != "done" or not path.exists():
        raise HTTPException(409, "Video not ready.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/result.json")
def job_result(job_id: str) -> FileResponse:
    job = _job(job_id)
    path = job.workdir / "result.json"
    if not path.exists():
        raise HTTPException(409, "Result not ready.")
    return FileResponse(path, media_type="application/json", filename=f"parivision_{job_id}.json")
