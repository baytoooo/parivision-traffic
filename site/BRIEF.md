# Website brief (team PariVision, WIUT Hackathon 2026, CV track)

Build the team website in this folder (`site/`). It is judged, and it is 25% of
our score. Read the rubric at the end before designing anything.

## What the project is

A system that watches a fixed 4K camera over a signalised junction in Tashkent
and (A) reports traffic events as time segments with a class, (B) outputs a
per-frame risk that an accident starts within 5 s. Pipeline: 4K frames ->
every 3rd frame at 1920 px -> YOLO26m detector -> ByteTrack -> tracks mapped
into a reference view of the junction by a homography (clips from the
afternoon session are framed ~20 px differently) -> hand-written rules on
trajectories + the vehicle signal phase (read from a 3-lamp head on the
median) -> segments per class. Part B: YOLO26s at 10 Hz, causal tracker,
time-to-collision between road users -> probability.

14 classes: accident, near_miss, red_light, wrong_way, illegal_u_turn,
stopped_vehicle, jaywalking, failure_to_yield, illegal_turn,
solid_line_crossing, stop_line, congestion, road_obstacle, fire_smoke.

Scene (the camera looks up the avenue from a building on the south-west
corner): southbound carriageway on the left of a median, northbound on the
right, a zebra across both carriageways just past the SB stop line (the
"north crossing"), a long diagonal zebra over the west arm at the bottom
left ("west crossing"), three pink channelising islands. Crossings have
yellow/white alternating stripes, which is a nice visual motif.

Real assets to use from the repo root (`..`):
* `docs/background.jpg` — median background of the junction (1920x1080)
* `docs/scene_overlay.jpg` — same with road mask, crossings, stop line drawn
* `docs/scene.md`, `docs/labeling.md` — text you can adapt (do not paste walls of it)

## Stack and hosting

* Astro (static output), TypeScript, no heavy UI kit. Plain CSS (custom
  properties) or a tiny utility layer; no Tailwind purple-gradient look.
* Deployed on Vercel from this folder (`site/` is the project root there).
  Keep every page static; the only dynamic part is the demo, which talks to a
  separate API (below).
* Charts: small, fast, interactive. Hand-rolled SVG where it is simple
  (timelines), a light library (uPlot or Observable Plot) where it helps.
  No Chart.js defaults look.
* Must work well on a phone (judges will open it on one). Fast: no multi-MB
  JS bundles, lazy-load videos.

## Pages

1. **Home** `/` — one screen that says what we built, with a looping
   annotated clip, three or four honest numbers (dev-set F1, runtime per
   minute of video, classes covered), and links into everything below.
2. **Approach** `/approach` — pipeline diagram (inline SVG you design, not an
   image), what is learned vs rule-based (a table), the rule for each class in
   one or two plain sentences, how the signal is read, why registration was
   needed, models and datasets with licences, frame sampling and runtime budget.
3. **EDA** `/eda` — clip table (resolution, fps, duration, local time,
   lighting), object counts over time by class (interactive line/area chart per
   clip), motion heatmap and trajectory images, lane direction field, traffic
   density by time, signal cycle stats, pedestrian desire lines, and a short
   "what this changed in our solution" list.
4. **Results** `/results` — per sample clip: annotated video player, event
   timeline under it (one lane per class; click a segment to seek the video),
   risk curve synced to playback, our dev labels vs predictions toggle,
   per-class F1 table from evaluate.py, examples of each detected class
   (thumbnails that open the video at that time), honest failure cases.
   Ablations table (detector variants, frame rates, with and without tracking).
5. **Demo** `/demo` — upload an .mp4 (state limits: up to 2 minutes, up to
   500 MB; 1080p recommended, 4K works but is slower), progress bar with stage
   names while it runs, then: event timeline, annotated playback, risk curve,
   event table, JSON download. Also "try a sample clip" buttons that run on
   short pre-cut clips hosted by the API. Handle errors and a cold-starting
   backend gracefully (show "waking up the server" state).
6. **Dashboard** `/dashboard` — operator view across all sample clips:
   events per class, per clip/time of day, per zone of the junction; a
   combined event log with filters. Think "what a city traffic centre would
   want on a wall screen".
7. **Report** `/report` — one page: what we built, what worked, what did not,
   what we would do next. Content comes from `data/report.md` (render markdown).
8. **Team** `/team` — members, roles, who did what, GitHub / LinkedIn /
   portfolio links, previous projects. Content from `data/team.json`.
   Placeholder values are marked `TODO` and must be easy to replace.

Footer on every page: repository link, weights link, predictions_samples.json
link, team name.

## Data contract (all under `public/data/`, loaded at build time or fetched)

Create realistic mock files with exactly these shapes now; the real ones will
replace them later without code changes.

`clips.json`
```json
[{"id": "C3896", "duration": 340.3, "fps": 29.97, "width": 3840, "height": 2160,
  "local_time": "2026-09-18 11:18", "light": "midday sun",
  "video": "/media/C3896_annotated.mp4", "poster": "/media/C3896_poster.jpg"}]
```

`results/<clip>.json`
```json
{"clip": "C3896", "duration": 340.3,
 "events": [[12.4, 18.9, "jaywalking"]],
 "labels": [[12.0, 19.0, "jaywalking"]],
 "risk": [[0.0, 0.01], [0.1, 0.01]],
 "signal": [[0.0, 27.1, "red"], [27.1, 59.0, "green"]],
 "evidence": [{"label": "jaywalking", "start": 12.4, "end": 18.9, "actors": [2000123], "note": ""}],
 "counts": {"t": [0, 5, 10], "car": [12, 14, 9], "bus": [1, 1, 0], "truck": [0, 1, 1],
            "motorcycle": [0, 0, 1], "person": [20, 25, 18], "bicycle": [0, 1, 0]}}
```
`risk` is downsampled to 10 Hz. `labels` are our own dev annotations.

`metrics.json` — the JSON written by `python evaluate.py --json`; the
fields used: `model_score`, `part_a.score_a`, `part_a.per_class[class]["0.3"|"0.5"|"0.7"].f1`,
`part_a.per_class[class].f1_mean`, `part_a.per_class[class]["0.5"].tp/fp/fn`,
`part_b` (may be null). Plus `ablations.json`:
`[{"name": "yolo26m @1280, 10 fps", "score_a": 0.41, "runtime_x": 0.9, "note": "..."}]`.

`eda.json`
```json
{"clips": ["C3896"], "density": {"C3896": {"t": [0, 10], "vehicles": [30, 28], "people": [12, 20]}},
 "speeds": {"sb": [...], "nb": [...]}, "signal_cycle": {"green": 32.0, "red": 43.0, "cycle": 75.0},
 "images": {"heatmap_vehicle": "/media/eda/heatmap_vehicle.jpg", "heatmap_person": "/media/eda/heatmap_person.jpg",
            "trajectories": "/media/eda/trajectories.jpg", "directions": "/media/eda/directions.jpg"},
 "findings": ["..."]}
```

`examples.json` — `[{"label": "jaywalking", "clip": "C3896", "t": 151.3, "thumb": "/media/examples/jay_1.jpg", "caption": "..."}]`

`team.json` — `{"team": "PariVision", "university": "Webster University in Tashkent",
"members": [{"name": "...", "role": "...", "did": ["..."], "github": "TODO", "linkedin": "TODO", "portfolio": "TODO", "projects": [{"name": "...", "url": "..."}]}]}`
Members: Amal Karimov (captain), Komronbek Qodirov, Aziza Adizova. Roles and
links are TODO for now.

`report.md` — markdown, one page.

Demo API (base URL in one config constant, default `https://parivision-traffic-demo.hf.space`):
* `POST /api/jobs` multipart field `file` -> `{"job_id": "..."}`; 413 if too big
* `POST /api/jobs/sample/<name>` -> `{"job_id": "..."}`
* `GET /api/samples` -> `[{"name": "c3896_crossing", "label": "Busy crossing, midday", "seconds": 30}]`
* `GET /api/jobs/<id>` -> `{"status": "queued|running|done|error", "progress": 0.42, "stage": "detecting and tracking", "eta_sec": 35, "error": null, "result": {same shape as results/<clip>.json without labels, plus "video": "/api/jobs/<id>/video"}}`
* `GET /api/health` -> `{"ok": true}`
Build the demo against a mock mode (`?mock=1` or when the API is unreachable
during development) that replays a fake job so the page can be developed now.

## Writing

The team asked for no AI slop. Copy must read like three students wrote it
about their own work: short, specific, concrete numbers, first person plural,
no hype words (no "revolutionary", "cutting-edge", "seamless", "leverage",
"empower", "robust solution"), no em dashes, no emoji, no rhetorical
questions, no "In conclusion". Say what did not work. Where you do not know
a number yet, write `TBD` in the mock data, not invented figures.

## Design

Make it look like it belongs to this project, not to a template. Ideas we
like: asphalt-dark background, road-marking white type, the yellow/white
zebra stripe as a recurring device (section rules, progress bars, the
timeline track), signal red/amber/green used only for states, monospace for
timestamps and numbers. A clean grotesk for text. Motion only where it
explains something (a timeline cursor, a playing clip), and respect
prefers-reduced-motion. Accessible contrast, visible focus states, keyboard
seekable timeline.

## Rubric the judges use (website, 0-1)

Live demo 30% (upload works, results visualised, no crashes). Sample-video
visualisations 20% (every sample annotated; timelines and risk curves
readable and correct). EDA 15% (goes beyond frame counts; findings that shaped
the solution). Approach and report 15% (a reader can rebuild the pipeline;
failures stated plainly). Team 10% (roles, contributions, links complete).
Design, UX, extras 10% (clean, fast, works on a phone; extras: interactive
charts, click-to-seek, ablations, error analysis, dashboard, live stream).

## Done means

`pnpm install && pnpm build` works in `site/`, `pnpm dev` shows every page
with the mock data, the demo page runs its mock job end to end, pages look
right at 390 px and 1440 px wide. Tell me which files hold the mock data and
the API base URL.
