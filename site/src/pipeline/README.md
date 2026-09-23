# In-browser pipeline

The live demo runs the PariVision pipeline in the visitor's browser: no server,
nothing uploaded. It is a port of `src/parivision/` (Python) to TypeScript, and
every stage is pinned to the Python code by parity tests on fixtures that the
Python code writes (`tools/export_parity_fixtures.py`).

What differs from the submission, on purpose:

| stage | submission (Python) | browser |
|---|---|---|
| frames | every 3rd frame (10 fps), 4K decoded to 1920 px | 5 fps, drawn from a `<video>` element |
| detector | YOLO26m at 1280 px (PyTorch) | the ONNX export in `public/pipeline/model.onnx` (see `scene.json` `detector`) on WebGPU or WASM |
| registration | SIFT + RANSAC homography | similarity (scale + shift) found by searching edge maps against the two references |
| everything after | tracker, trajectories, signal, rules, risk | the same algorithms and constants |

## Files

```
types.ts         shared types (below)
scene.ts         loads public/pipeline/scene.json + scene.bin.gz; point lookups in the reference view
geometry.ts      homographies, polygons, signed side of a line, rounding like NumPy
segments.ts      runs / union / finalize (segments.py)
tracker.ts       ByteTrack as in Ultralytics 8.4 (Kalman filter, two-stage matching), one per object group
trajectories.ts  build(): foot points, smoothing, velocities (trajectories.py)
signal.ts        lamp patches, lamp scores, phase per frame, fill_phases (signal.py)
align.ts         the upload's first frame -> homography into the reference view
rules.ts         Context and one function per class, detectFromContext (rules.py, events.py)
risk.ts          Part B's Anticipator.observe path (risk.py)
detector.ts      letterbox, ONNX session, output parsing; works with onnxruntime-web and -node
analyse.ts       glue with no DOM: frames in, ClipResult + overlay out
worker.ts        Web Worker: owns the ONNX session and an Analyser
```

The UI side lives in `src/scripts/local_api.ts` (the demo page's `Api` for this
pipeline: frame extraction from a `<video>`, talking to the worker) and the
overlay drawing in `src/scripts/player.ts`.

## Rules for this folder

* TypeScript that Node runs as is (`node --test "tests/*.test.ts"`): only erasable syntax
  (no `enum`, no constructor parameter properties, no `namespace`), relative
  imports with the `.ts` extension, `import type` for types.
* No DOM in anything but `worker.ts`; the Node tests import the rest.
* Constants come from `scene.json`, never retyped. If a constant is missing,
  add it to `tools/export_browser_assets.py` and re-export.
* Match NumPy where it matters: `np.round` rounds half to even
  (`geometry.roundHalfEven`), `scipy.ndimage.uniform_filter1d` reflects at the
  edges, `np.percentile` interpolates linearly, `np.median` of an even count
  averages the middle two.

## Coordinates

* **work**: the frame scaled to 1920 px wide (1080 high for 16:9). Detector
  boxes, tracker boxes and `Trajectory.box` are in work pixels.
* **reference**: the 1920x1080 reference view. `Trajectory.foot`, all scene
  rasters, polygons and rule thresholds are in reference pixels.
* `H` (3x3, row-major) maps work pixels to reference pixels.
* The detector input is `scene.json` `detector.input` = [h, w] (544 x 960):
  the frame is drawn at 960 x 540 and letterboxed with 2 px of grey (114) at
  the top and bottom, as Ultralytics' `LetterBox(auto=False, center=True)`.
  Its boxes (input pixels) minus the padding, times 2, are work pixels.

## Tests

`site/tests/*.test.ts`, run from `site/` with `node --test "tests/*.test.ts"`. Fixtures are in
`site/tests/fixtures/<clip>/` (C3905 and C3902), made by
`python tools/export_parity_fixtures.py --clip <clip>`; see its docstring for
what each file holds. Each test feeds a module the Python input of that stage
and compares with the Python output of that stage:

| test | input | compared with | tolerance |
|---|---|---|---|
| scene | scene_samples.json points | the Python values | exact (distances 1/16 px) |
| segments | hand-written cases from tests/test_core.py | | exact |
| tracker | detections.json | tracks.json | >= 97% of rows matched by box, ids consistent |
| trajectories | tracks.json | trajectories.json | 1e-3 px |
| signal | lamp_*.rgb, signal.json raw | lamps.json scores, signal.json phases | 0.5, exact |
| align | frame0.gray + refs | alignment.json H | reference points within 4 px |
| rules | trajectories.json + signal.json phases | rules.json events and evidence | 0.01 s |
| risk | detections.json | risk.json | 0.02 per frame |
| detector | det_frame.rgb | det_expected.json | boxes within 1 px, same classes |
