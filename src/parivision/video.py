"""Frame sampling for Part A.

The camera files are 4K, 10-bit 4:2:2 H.264 at ~140 Mbit/s, so decoding is
the most expensive step of the whole pipeline. Two things keep it cheap:

* PyAV decodes every frame (H.264 has no cheap way to skip them) but only the
  sampled ones are converted, and the conversion goes straight to the working
  resolution inside swscale instead of producing a 4K BGR array first.
* Decoding runs in a background thread (PyAV releases the GIL), so the
  detector works on frame k while frame k+1 is being decoded.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import av.logging
import numpy as np

# Keep FFmpeg's log messages out of Python. With PyAV's Python log callback
# active (the default before PyAV 16), closing a frame-threaded H.264 decoder
# mid-stream can deadlock: the decoder's worker threads wait for the GIL to
# log while the closing thread holds it.
av.logging.set_level(None)
av.logging.set_libav_level(av.logging.ERROR)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    """Same numbers the harness reads through OpenCV, so our times line up with its clock."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = VideoInfo(
        path=str(path),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def sample_frames(
    path: str | Path,
    target_fps: float,
    out_width: int,
    prefetch: int = 16,
    stats: dict | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, t_sec, bgr)`` for roughly ``target_fps`` frames per second.

    ``t_sec`` is ``frame_index / fps`` (the harness convention), not the
    container PTS, so Part A and Part B agree on time even when the stream
    starts with a small PTS offset. Frames are resized to ``out_width``
    keeping the aspect ratio. ``stats``, if given, is kept up to date with
    ``frames`` decoded so far and the ``seconds`` spent decoding them (time
    waiting for the consumer not counted), which is how fast this machine
    decodes the clip.
    """
    info = probe(path)
    stride = max(1, int(round(info.fps / target_fps)))
    out_height = int(round(info.height * out_width / info.width / 2)) * 2

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def producer() -> None:
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                busy, t0 = 0.0, time.perf_counter()
                for idx, frame in enumerate(container.decode(stream)):
                    if stop.is_set():
                        break
                    if idx % stride == 0:
                        img = frame.to_ndarray(format="bgr24", width=out_width, height=out_height)
                        busy += time.perf_counter() - t0
                        q.put((idx, idx / info.fps, img))
                        t0 = time.perf_counter()
                    if stats is not None:
                        stats.update(frames=idx + 1, seconds=busy + time.perf_counter() - t0)
        except Exception as exc:  # surfaced in the consumer
            q.put(exc)
        finally:
            q.put(None)

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        while worker.is_alive():  # unblock a producer waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                worker.join(timeout=0.05)
