# Third-party notices

PariVision traffic events. Copyright (C) 2026 Team PariVision: Amal Karimov,
Komronbek Qodirov, Aziza Adizova. Our own code, labels and documents are
released under the GNU Affero General Public License, version 3 (`LICENSE`).

The repository also contains, or is built on, the work of others listed below.
Each keeps its own licence.

## Model weights in this repository

| file | what it is | licence |
|---|---|---|
| `weights/yolo26n.pt`, `weights/yolo26s.pt`, `weights/yolo26m.pt` | YOLO26 detection weights pretrained on COCO, from the Ultralytics assets release v8.4.0, unchanged (checksums in `weights/SHA256SUMS`) | AGPL-3.0 (Ultralytics) |
| `site/public/pipeline/model.onnx` | `yolo26n.pt` exported to ONNX for the in-browser demo by `tools/export_browser_assets.py` | AGPL-3.0, as the weights it is made from |

The weights were trained by Ultralytics on COCO 2017, whose annotations are
under CC BY 4.0.

## Code ported into this repository

`site/src/pipeline/tracker.ts` is a TypeScript port of the ByteTrack tracker in
Ultralytics 8.4.159 (`trackers/byte_tracker.py`, `basetrack.py`,
`utils/kalman_filter.py`, `utils/matching.py`, `utils/stracks.py`; AGPL-3.0),
which builds on ByteTrack by Zhang et al. (MIT), and of the `lapjv` solver of
the lap package (BSD-2-Clause; its licence is reproduced at the end of this
file). The rest of `site/src/pipeline/` is a port of our own Python code in
`src/parivision/`.

## Python packages the submission installs

Installed by `requirements.txt`, not stored in the repository.

| package | licence |
|---|---|
| Ultralytics (`ultralytics-opencv-headless`) | AGPL-3.0 |
| PyTorch (`torch`), torchvision | BSD-3-Clause |
| lap | BSD-2-Clause |
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | Apache-2.0 |
| PyAV (`av`) | BSD-3-Clause; its wheels bundle an FFmpeg build that includes x264 and x265, which are GPL |

The dev tools add pytest (MIT), pandas (BSD-3-Clause), kagglehub (Apache-2.0),
gdown (MIT), onnx (Apache-2.0), onnxslim (MIT) and onnxruntime (MIT)
(`tools/requirements-dev.txt`). The local demo server adds FastAPI (MIT),
Uvicorn (BSD-3-Clause) and python-multipart (Apache-2.0)
(`demo/requirements.txt`).

## Website packages

Installed by pnpm from `site/pnpm-lock.yaml`. The built site ships
onnxruntime-web, ffmpeg.wasm and the fonts; marked and Astro run only when the
site is built. Versions and licences below are from each package's
`package.json` in `site/node_modules/`.

| package | version | licence | notes |
|---|---|---|---|
| onnxruntime-web | 1.30.0 | MIT | runs `model.onnx` in the browser |
| @ffmpeg/ffmpeg | 0.12.15 | MIT | ffmpeg.wasm wrapper |
| @ffmpeg/core-mt | 0.12.10 | GPL-2.0-or-later | the FFmpeg core compiled to WebAssembly, copied to `site/public/ffmpeg/` by `site/scripts/copy-ffmpeg.ts`; it converts clips the browser cannot play. Source: https://github.com/ffmpegwasm/ffmpeg.wasm |
| marked | 18.0.13 | MIT | turns the report into HTML at build time |
| @fontsource-variable/overpass, @fontsource/overpass-mono | 5.3.0 | OFL-1.1 | Overpass fonts, Copyright 2021 The Overpass Project Authors |
| astro | 7.3.3 | MIT | builds the site |

## Data

* **Sample clips** C3896, C3897, C3902 and C3905, given by the organisers to
  the participants of WIUT Hackathon 2026. No licence stated. The clips are not
  in the repository, but images made from them are: the reference views
  (`src/parivision/assets/reference_*.jpg`, `docs/background.jpg`), the
  website's posters, example frames and EDA figures (`site/public/media/`), and
  the C3905 test fixtures (`site/tests/fixtures/C3905/`: one 960x540 frame, one
  480x270 grey frame and four crops of the signal head).
* **ACCIDENT benchmark** (Picek et al., CVPR 2026; Kaggle `picekl/accident`).
  Data CC BY-NC-SA 4.0, annotations CC BY 4.0. `labels/accident_real.csv` and
  `labels/accident_synthetic.csv` copy the benchmark's annotations for the
  clips we picked. The clips themselves are not in the repository.

## lap licence

```
BSD 2-Clause License

Copyright (c) 2012-2025, Tomas Kazmar

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
