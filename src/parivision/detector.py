"""COCO-pretrained YOLO detector with batching and device selection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO ids we care about, grouped the way the rules use them.
PERSON = 0
BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 1, 2, 3, 5, 7
VEHICLES = (CAR, BUS, TRUCK, MOTORCYCLE)
TWO_WHEELERS = (BICYCLE, MOTORCYCLE)
ANIMALS = (14, 15, 16, 17, 18, 19)  # bird, cat, dog, horse, sheep, cow
KEEP = (PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK) + ANIMALS
COCO_NAMES = {PERSON: "person", BICYCLE: "bicycle", CAR: "car", MOTORCYCLE: "motorcycle", BUS: "bus", TRUCK: "truck",
              14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Detections:
    """Boxes for one frame, in the pixel space of the frame given to the detector."""

    xyxy: np.ndarray  # (N, 4) float32
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int16

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int16))


class Detector:
    def __init__(self, weights: str = "yolo26m.pt", imgsz: int = 1280, conf: float = 0.15,
                 device: str | None = None, classes: tuple[int, ...] = KEEP):
        from ultralytics import YOLO

        path = Path(weights)
        if not path.is_absolute() and not path.exists():
            path = WEIGHTS_DIR / weights
        self.model = YOLO(str(path))
        self.device = device or pick_device()
        self.half = self.device.startswith("cuda")
        self.imgsz = imgsz
        self.conf = conf
        self.classes = list(classes)

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        if not frames:
            return []
        results = self.model.predict(
            frames, imgsz=self.imgsz, conf=self.conf, classes=self.classes, device=self.device,
            quantize=16 if self.half else None, verbose=False, max_det=300,
        )
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(Detections.empty())
                continue
            out.append(Detections(
                b.xyxy.cpu().numpy().astype(np.float32),
                b.conf.cpu().numpy().astype(np.float32),
                b.cls.cpu().numpy().astype(np.int16),
            ))
        return out
