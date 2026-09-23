"""Export what the in-browser demo needs from the Python pipeline, so both share one source of truth.

    python tools/export_browser_assets.py [--model yolo26n --height 544 --width 960]

Writes site/public/pipeline/:

  model.onnx     the detector, end-to-end (1, 300, 6): x1 y1 x2 y2 score class, in input pixels
  scene.json     every constant the rules, tracker, signal reader and risk model use
  scene.bin.gz   rasters in the 1920x1080 reference view, gzip of back-to-back layers:
                   masks    uint16, one bit per mask (scene.json raster.mask_bits)
                   zones    uint8, one bit per zone (scene.json raster.zone_bits)
                   road_dist, walk_dist, cw_dist   uint16, distance in 1/16 reference px (cv2.DIST_L2, mask 3)
  reference_*.png  the midday and dusk references at 480x270, grey, for aligning an upload

The browser code (site/src/pipeline/) reads scene.json for the layer order, so adding a mask
here needs no change there beyond using it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision import events as E  # noqa: E402
from parivision import risk as RK  # noqa: E402
from parivision import rules as R  # noqa: E402
from parivision import scene as S  # noqa: E402
from parivision import signal as SG  # noqa: E402
from parivision import tracking as T  # noqa: E402
from parivision import trajectories as TJ  # noqa: E402
from parivision.detector import KEEP  # noqa: E402

OUT = ROOT / "site/public/pipeline"
DIST_SCALE = 16  # uint16 distances in 1/16 px


def rasters() -> tuple[dict, bytes]:
    w, h = S.REF_SIZE
    m = S.masks()
    ctx = R.Context([], np.array([]), np.array([]), 1.0)  # builds the same masks, distances and zones
    k = int(2 * R.PARAMS["fty_cw_dilate"]) + 1
    masks = {"road": ctx.road, "walk": ctx.walk, "crosswalk": m["crosswalk"]}
    for name, cw in ctx.cw_masks.items():
        masks[f"cw_{name}"] = cw
        masks[f"cw_{name}_zone"] = cv2.dilate(cw, np.ones((k, k), np.uint8))  # failure_to_yield's zebra + margin
    zones = dict(ctx.zones)
    box = np.zeros((h, w), np.uint8)
    cv2.fillPoly(box, [S.poly(S.JUNCTION_BOX).astype(np.int32)], 1)
    zones["junction_box"] = box
    assert len(masks) <= 16 and len(zones) <= 8
    mask_bits = np.zeros((h, w), np.uint16)
    for i, v in enumerate(masks.values()):
        mask_bits |= (v > 0).astype(np.uint16) << np.uint16(i)
    zone_bits = np.zeros((h, w), np.uint8)
    for i, v in enumerate(zones.values()):
        zone_bits |= (v > 0).astype(np.uint8) << np.uint8(i)
    dists = {"road_dist": ctx.road_dist, "walk_dist": ctx.walk_dist, "cw_dist": ctx.cw_dist}
    layers = [("masks", "uint16", mask_bits), ("zones", "uint8", zone_bits)]
    for name, d in dists.items():
        layers.append((name, "uint16", np.clip(np.round(d * DIST_SCALE), 0, 65535).astype(np.uint16)))
    blob, index, off = b"", [], 0
    for name, dtype, arr in layers:
        raw = np.ascontiguousarray(arr).astype("<" + ("u2" if dtype == "uint16" else "u1")).tobytes()
        index.append({"name": name, "dtype": dtype, "offset": off, "bytes": len(raw)})
        blob += raw
        off += len(raw)
    meta = {"width": w, "height": h, "layers": index, "dist_scale": DIST_SCALE,
            "mask_bits": list(masks), "zone_bits": list(zones)}
    return meta, blob


def constants() -> dict:
    return {
        "ref_size": list(S.REF_SIZE),
        "rules": R.PARAMS,
        "direction_zones": {k: {"polygon": v[0], "heading": v[1]} for k, v in R.DIRECTION_ZONES.items()},
        "crosswalks": {k: list(v) for k, v in S.CROSSWALKS.items()},
        "stop_line_sb": list(S.STOP_LINE_SB),
        "median_nose": list(S.MEDIAN_NOSE),
        "person_height_px": list(S.PERSON_HEIGHT_PX),
        "signal_lamps": S.SIGNAL_LAMPS,
        "signal_min_contrast": SG.MIN_CONTRAST.tolist(),
        "events": {"enabled": list(E.ENABLED), "shown": list(E.SHOWN), "gap": E.GAP, "min_len": E.MIN_LEN},
        "detector": {"keep_classes": sorted(int(c) for c in KEEP), "conf": 0.1},  # pipeline.make_detector
        "tracking": {**{name: getattr(T, name) for name in dir(T) if name.isupper() and not name.startswith("_")},
                     "bytetrack": {k: v for k, v in vars(T._byte_args(10.0, 2.0)).items()
                                   if k not in ("tracker_type", "track_buffer")},
                     "buffer_sec": 2.0},
        "trajectories": {name: getattr(TJ, name) for name in dir(TJ) if name.isupper() and not name.startswith("_")},
        "risk": {name: getattr(RK, name) for name in dir(RK) if name.isupper() and not name.startswith("_")
                 and name != "LAST_RUN" and isinstance(getattr(RK, name), (int, float, dict, list, tuple, str))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n")
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta, blob = rasters()
    (OUT / "scene.bin.gz").write_bytes(gzip.compress(blob, 9))
    const = constants()
    const["raster"] = meta
    const["detector"].update({"model": args.model, "input": [args.height, args.width]})
    (OUT / "scene.json").write_text(json.dumps(const, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

    for name in ("reference_day", "reference_dusk"):
        img = cv2.imread(str(ROOT / "src/parivision/assets" / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA))

    from ultralytics import YOLO

    tmp = OUT / f"{args.model}.pt"
    shutil.copy(ROOT / "weights" / f"{args.model}.pt", tmp)
    onnx = YOLO(str(tmp)).export(format="onnx", imgsz=[args.height, args.width], opset=17, simplify=True,
                                 dynamic=False, batch=1, nms=False)
    shutil.move(onnx, OUT / "model.onnx")
    tmp.unlink()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:22s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
