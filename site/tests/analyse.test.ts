// Analyser (the analyse() loop plus demo/worker.py's extras) on the fixture detections and lamp
// scores. The same detections go through the same tracker, trajectories, signal clean-up and rules
// as in Python, so the events must be those of rules.json; the risk curve follows risk.json.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { Analyser } from "../src/pipeline/analyse.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import type { Detection, Seg } from "../src/pipeline/types.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;
const scene = loadScene(read);
const fixture = async (clip: string, name: string) =>
  JSON.parse(await readFile(new URL(`./fixtures/${clip}/${name}`, import.meta.url), "utf8"));

// Event and evidence times: rules.json keeps 4 decimals and the result 2 (as worker.py), and
// segments.round2 can differ from Python's round(x, 2) by 0.01 on a tie (115.115 -> 115.12).
const TOL = 0.015;
const PHASES = new Set(["red", "yellow", "green", "off", "unknown"]);

/** worker.py's signal segments from signal.json's times and filled phases. */
function segments(times: number[], phases: string[]): Seg[] {
  const out: Seg[] = [];
  const r1 = (v: number) => Number(v.toFixed(1));
  times.forEach((t, i) => {
    const last = out[out.length - 1];
    if (last && last[2] === phases[i]) last[1] = r1(t);
    else out.push([r1(t), r1(t), phases[i]]);
  });
  return out;
}

for (const clip of ["C3905", "C3902"]) {
  test(`${clip}: Analyser gives the events of rules.json and a well-formed result`, async (t) => {
    const det = await fixture(clip, "detections.json");
    const sig = await fixture(clip, "signal.json");
    const { H, work_size } = await fixture(clip, "alignment.json");
    const expected = await fixture(clip, "rules.json");
    const riskRef: [number, number][] = (await fixture(clip, "risk.json")).risk;

    const a = new Analyser(await scene, det.fps, work_size);
    a.setAlignment({ H: (H as number[][]).flat(), score: 1, ok: true, reference: "fixture" });
    let j = 0;
    for (const fr of det.frames as { t: number; boxes: number[][] }[]) {
      // the lamp scores are per 0.2 s sample of the clip: take the nearest one
      while (j + 1 < sig.times.length && Math.abs(sig.times[j + 1] - fr.t) <= Math.abs(sig.times[j] - fr.t)) j++;
      const dets: Detection[] = fr.boxes.map(([x1, y1, x2, y2, conf, cls]) => ({ x1, y1, x2, y2, conf, cls }));
      a.push(fr.t, dets, sig.scores[j]);
    }
    const n = det.frames.length;
    assert.equal(a.frameCount, n);
    const r = a.finish(det.duration, clip);

    // shape
    assert.equal(r.clip, clip);
    assert.equal(r.duration, Number(det.duration.toFixed(2)));
    assert.equal(r.fps, det.fps);
    assert.equal(r.aligned, true);
    assert.deepEqual(r.overlay.work, work_size);
    assert.equal(r.overlay.frames.length, n);
    r.overlay.frames.forEach((f, i) => {
      assert.equal(f.t, det.frames[i].t);
      for (const b of f.boxes) {
        assert.equal(b.length, 6);
        assert.ok(b.every(Number.isFinite) && b[0] <= b[2] && b[1] <= b[3], `frame ${i}: box ${b}`);
        assert.ok(b[4] >= 1_000_000 && b[4] < 5_000_000, `frame ${i}: track id ${b[4]}`);
      }
    });
    assert.equal(r.risk.length, n);
    r.risk.forEach(([rt, s], i) => {
      assert.ok(Math.abs(rt - det.frames[i].t) <= 0.005 + 1e-9 && s >= 0 && s <= 1, `risk ${i}: ${rt}, ${s}`);
    });
    const bins = r.counts.t;
    assert.deepEqual(bins, [...new Set(det.frames.map((f: { t: number }) => Math.floor(f.t / 5) * 5))]);
    for (const name of ["car", "bus", "truck", "motorcycle", "person", "bicycle"]) {
      assert.equal(r.counts[name].length, bins.length, name);
      assert.ok(r.counts[name].every((v) => v >= 0 && Number.isFinite(v)), name);
    }
    assert.ok(r.counts.car.some((v) => v > 0) && r.counts.person.some((v) => v > 0));
    r.signal.forEach(([s, e, p], i) => {
      assert.ok(s <= e && PHASES.has(p), `signal ${i}: ${s}-${e} ${p}`);
      if (i) assert.ok(s >= r.signal[i - 1][1] && p !== r.signal[i - 1][2], `signal ${i} after ${r.signal[i - 1]}`);
    });
    // the phases from the lamp scores are signal.json's filled phases
    assert.deepEqual(r.signal, segments(det.frames.map((f: { t: number }) => f.t), sig.phases));

    // events
    assert.equal(r.events.length, expected.events.length, "event count");
    let worst = 0;
    r.events.forEach(([s, e, label], i) => {
      const [xs, xe, xl] = expected.events[i];
      assert.equal(label, xl, `event ${i}`);
      worst = Math.max(worst, Math.abs(s - xs), Math.abs(e - xe));
      assert.ok(Math.abs(s - xs) <= TOL && Math.abs(e - xe) <= TOL, `event ${i} ${label}: ${s}-${e} vs ${xs}-${xe}`);
    });

    // evidence: the same items; track ids differ from rules.json only by a one-to-one renaming (the
    // exporter's risk tracker drew ids from the main tracker's counter, the Analyser's does not)
    assert.equal(r.evidence.length, expected.evidence.length, "evidence count");
    const ids = new Map<number, number>();
    const back = new Map<number, number>();
    r.evidence.forEach((ev, i) => {
      const e = expected.evidence[i];
      const what = `evidence ${i} (${e.label} ${e.start}-${e.end})`;
      assert.equal(ev.label, e.label, what);
      assert.equal(ev.note, e.note, what);
      assert.ok(Math.abs(ev.start - e.start) <= TOL && Math.abs(ev.end - e.end) <= TOL, `${what}: got ${ev.start}-${ev.end}`);
      assert.equal(ev.actors.length, e.actors.length, what);
      ev.actors.forEach((id, k) => {
        const want = e.actors[k] as number;
        assert.equal(ids.get(id) ?? want, want, `${what}: actor ${id}`);
        assert.equal(back.get(want) ?? id, id, `${what}: actor ${want}`);
        ids.set(id, want);
        back.set(want, id);
      });
    });

    // risk: risk.json replayed Part B over the same detections; the few frames where the fixture's
    // 3-decimal rounding changed which tracks exist (tests/risk.test.ts) may differ by more
    const diffs = r.risk.map(([, s], i) => Math.abs(s - riskRef[i][1]));
    const over = diffs.filter((d) => d > 0.02).length;
    const maxDiff = Math.max(...diffs);
    assert.ok(over <= Math.ceil(0.01 * n) && maxDiff < 0.05, `risk: ${over} frames over 0.02, max ${maxDiff}`);

    t.diagnostic(
      `${clip}: ${n} frames, ${r.events.length}/${expected.events.length} events matched, worst edge ${worst.toFixed(3)} s; ` +
        `${r.evidence.length}/${expected.evidence.length} evidence items matched (${ids.size} actors); ` +
        `risk max |diff| to risk.json ${maxDiff.toExponential(2)}, ${over} frames over 0.02`,
    );
  });
}

test("Analyser without an alignment refuses frames and finishes with a plain rescale", async () => {
  const a = new Analyser(await scene, 5, [1920, 1080]);
  assert.throws(() => a.push(0, [], null), /setAlignment/);
  const r = a.finish(0, "empty");
  assert.deepEqual([r.events, r.evidence, r.signal, r.risk, r.overlay.frames, r.counts.t], [[], [], [], [], [], []]);
  assert.equal(r.aligned, false);
});

test("Analyser reads a frame without lamp scores as an unknown phase", async () => {
  const a = new Analyser(await scene, 5, [1920, 1080]);
  a.setAlignment({ H: [1, 0, 0, 0, 1, 0, 0, 0, 1], score: 1, ok: true, reference: "identity" });
  const car: Detection = { x1: 100, y1: 100, x2: 200, y2: 180, conf: 0.9, cls: 2 };
  for (let k = 0; k < 30; k++) a.push(k / 5, [{ ...car, x1: car.x1 + k, x2: car.x2 + k }], k < 20 ? [200, -2, -2] : null);
  const r = a.finish(6, "lamps");
  assert.deepEqual(r.signal, [[0, 3.8, "red"], [4, 5.8, "unknown"]]);
  assert.deepEqual(r.counts.t, [0, 5]);
  assert.deepEqual(r.counts.car, [1, 1]);
  // one vehicle track (id 1 in the vehicle range) whose Kalman box follows the car
  const boxes = r.overlay.frames.map((f) => f.boxes);
  assert.ok(boxes.every((b) => b.length === 1 && b[0][4] === 1_000_001 && b[0][5] === 2));
  const [x1, y1, x2, y2] = boxes[29][0];
  assert.ok(Math.max(Math.abs(x1 - 129), Math.abs(y1 - 100), Math.abs(x2 - 229), Math.abs(y2 - 180)) < 1, `${boxes[29][0]}`);
});
