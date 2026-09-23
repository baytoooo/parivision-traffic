// Part B's Anticipator.observe on the cached detections gives risk.json: the exporter replays
// fixtures/<clip>/detections.json (full-precision float32 values) through risk.py with the clip's H
// (alignment.json) and writes the score after every frame, rounded to 5 decimals.
//
// Track ids differ from Python's (its risk tracker shares the id counter with the main one), which
// does not matter here: ids only name the history and the pair streaks.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import type { TestContext } from "node:test";
import { Anticipator, braking, pairHazard } from "../src/pipeline/risk.ts";
import type { RiskConstants } from "../src/pipeline/risk.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import type { Detection } from "../src/pipeline/types.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;
const fixture = async (clip: string, name: string) =>
  JSON.parse(await readFile(new URL(`./fixtures/${clip}/${name}`, import.meta.url), "utf8"));

const TOL = 1e-5; // risk.json is rounded to 5 decimals

for (const clip of ["C3905", "C3902"]) {
  test(`${clip}: Anticipator.observe reproduces risk.json`, async (t: TestContext) => {
    const scene = await loadScene(read);
    const d: { fps: number; width: number; height: number; frames: { t: number; boxes: number[][] }[] } =
      await fixture(clip, "detections.json");
    const { H } = await fixture(clip, "alignment.json");
    const { risk }: { risk: [number, number][] } = await fixture(clip, "risk.json");
    assert.equal(risk.length, d.frames.length);

    const model = new Anticipator({ fps: d.fps, width: d.width, height: d.height }, scene);
    model.setAlignment((H as number[][]).flat());
    let worst = 0, at = -1, over = 0, nonzero = 0;
    d.frames.forEach((f, k) => {
      const dets: Detection[] = f.boxes.map(([x1, y1, x2, y2, conf, cls]) => ({ x1, y1, x2, y2, conf, cls }));
      const score = model.observe(dets, [d.height, d.width], f.t);
      assert.ok(Math.abs(f.t - risk[k][0]) < 1e-3, `frame ${k}: t ${f.t} vs ${risk[k][0]}`);
      const diff = Math.abs(score - risk[k][1]);
      if (diff > worst) [worst, at] = [diff, k];
      if (risk[k][1] > 0) nonzero++;
      if (diff > TOL) over++;
    });
    const peak = Math.max(...risk.map((r) => r[1]));
    t.diagnostic(
      `${clip}: ${risk.length} frames (${nonzero} with risk > 0, peak ${peak}); max |diff| to risk.json ` +
        `${worst.toExponential(3)}` + (at >= 0 ? ` at frame ${at} (t ${risk[at][0]})` : "") +
        `; frames over ${TOL}: ${over}`,
    );
    assert.equal(over, 0, `${over} frames differ by more than ${TOL} (worst ${worst} at frame ${at})`);
  });
}

test("Anticipator on a hand-made scene matches risk.py (H = identity, so work pixels are reference pixels)", async () => {
  // A still pedestrian on the southbound carriageway with a car heading at them and braking, a truck
  // passing them towards a parked car (moving vs parked: never on its own), a pedestrian off the road
  // (never scored), a car on the northbound side head-on to the southbound truck (opposite sides of
  // the median: skipped until it leaves the nb zone), the pedestrian missing for two frames (stale:
  // skipped) and then gone for good (history dropped after 1 s). The car jumps 80 px a frame at
  // first, so ByteTrack keeps dropping it: new ids, one-sample histories. Scores from risk.py's
  // Anticipator.observe with alignment H = eye(3) on the same float32 detections, t = k / 5.
  const scene = await loadScene(read);
  const C1X = [300, 380, 460, 540, 620, 680, 720, 740, 750, 755, 758, 760];
  const boxes = (k: number): number[][] => {
    const b: number[][] = [];
    if (k <= 11 || (k >= 14 && k <= 17)) b.push([690, 350, 710, 400, 0.8, 0]);
    if (k <= 20) {
      const x = C1X[Math.min(k, C1X.length - 1)];
      b.push([x - 60, 312, x + 60, 402, 0.9, 2]);
    }
    b.push([190, 650, 210, 700, 0.7, 0], [500, 350, 620, 440, 0.85, 2], [100 + 60 * k - 70, 360, 100 + 60 * k + 70, 445, 0.9, 7]);
    const x3 = 1300 - 60 * k;
    b.push([x3 - 55, 250, x3 + 55, 330, 0.9, 2]);
    return b;
  };
  const expected = [
    0, 0, 0, 0, 0, 0.0673826982310318, 0.13431087834347027, 0.21330597973326276, 0.284879153214727, 0.3441984153941615,
    0.3849985708775281, 0.4104370516439064, 0.26678408356853917, 0.17340965431955047, 0.11271627530770781,
    0.07326557895001008, 0.04762262631750656, 0.030954707106379264, 0.020120559619146522, 0.013078363752445239,
    0.008500936439089406, 0.005525608685408114, 0.0035916456455152746, 0.0023345696695849286, 0.0015174702852302038,
  ];
  const model = new Anticipator({ fps: 5, width: 1920, height: 1080 }, scene);
  model.setAlignment([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  expected.forEach((e, k) => {
    const dets: Detection[] = boxes(k).map(([x1, y1, x2, y2, conf, cls]) => ({ x1, y1, x2, y2, conf, cls }));
    const s = model.observe(dets, [1080, 1920], k / 5);
    assert.ok(Math.abs(s - e) < 1e-12, `frame ${k}: ${s} vs risk.py ${e}`);
  });
});

test("a user unseen for exactly 1.0 s keeps its history (risk.py drops it only when t - last > 1.0)", async () => {
  // A car driving at a still pedestrian is hidden on frames 6-10: at frame 10, t - last = 2.0 - 1.0 = 1.0
  // exactly, so the history stays; ByteTrack finds the car again at frame 11 with the same id and the
  // pair can score from its third update (frame 13). Scores from risk.py, H = identity, t = k / 5.
  const scene = await loadScene(read);
  const expected = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04524631492194854, 0.09297266108385688, 0.1530913562683682,
    0.21105034009846868, 0.2650212342875765, 0.3132297364045877, 0.35184264288331446,
  ];
  const model = new Anticipator({ fps: 5, width: 1920, height: 1080 }, scene);
  model.setAlignment([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  expected.forEach((e, k) => {
    const dets: Detection[] = [{ x1: 850, y1: 390, x2: 870, y2: 440, conf: 0.8, cls: 0 }];
    if (k <= 5 || k >= 11) dets.push({ x1: 100 + 40 * k - 80, y1: 350, x2: 100 + 40 * k + 80, y2: 440, conf: 0.9, cls: 2 });
    const s = model.observe(dets, [1080, 1920], k / 5);
    assert.ok(Math.abs(s - e) < 1e-12, `frame ${k}: ${s} vs risk.py ${e}`);
  });
});

test("pairHazard and braking on hand-made cases", async () => {
  const k = (await loadScene(read)).c.risk as unknown as RiskConstants;
  // parallel users (same velocity) never score
  assert.equal(pairHazard([0, 0], [5, 0], 1.3, [0, 3], [5, 0], 1.3, k), 0);
  // slow approach (closing speed under CLOSING_MIN)
  assert.equal(pairHazard([0, 0], [2, 0], 1.3, [10, 0], [0, 0], 1.3, k), 0);
  // head-on at 20 m/s closing, 10 m apart: contact at once (gap 10 - 2.6 = 7.4 m -> 0.37 s -> ts 0.4)
  const h = pairHazard([0, 0], [10, 0], 1.3, [10, 0], [-10, 0], 1.3, k);
  const soon = 1 / (1 + Math.exp(2.5 * (0.4 - k.TTC_HALF)));
  assert.ok(Math.abs(h - soon * (0.4 + 0.6 * 1)) < 1e-6, `head-on hazard ${h}`);
  // values from risk.pair_hazard: the first touch on np.arange's 0.1 s grid, t_star clipped to HORIZON,
  // the 0.4x radii when one user stands still, and footprints that already overlap
  const py: [number[], number[], number, number[], number[], number, number][] = [
    [[0, 0], [5, 0], 1.3, [10, 0], [-5, 0], 1.3, 0.5621764316948514],
    [[0, 0], [2.5, 0], 1.3, [16, 0], [-2.5, 0], 1.3, 0.0042257469015035765],
    [[0, 0], [8, 0], 1.3, [6, 0.2], [0, 0], 0.35, 0.4072016976501382],
    [[0, 0], [8, 0], 1.3, [6, 0.8], [0, 0], 0.35, 0],
    [[0, 0], [6, 0], 1.8, [2, 0], [-1, 0], 1.3, 0.6332550529404212],
  ];
  for (const [p1, v1, r1, p2, v2, r2, e] of py) {
    const got = pairHazard(p1 as [number, number], v1 as [number, number], r1, p2 as [number, number], v2 as [number, number], r2, k);
    assert.ok(Math.abs(got - e) < 1e-12, `pairHazard(${p1}, ${v1}, ${r1}, ${p2}, ${v2}, ${r2}) = ${got}, risk.py ${e}`);
  }
  // steady speed: no braking; 10 m/s dropping to 0 within 0.35 s windows: full braking
  const ts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
  assert.equal(braking(ts, ts.map((x) => [10 * x, 0])), 0);
  assert.equal(braking(ts, ts.map((x) => [x <= 0.4 ? 10 * x : 4, 0])), 1);
  assert.equal(braking(ts.slice(0, 5), ts.slice(0, 5).map((x) => [x <= 0.2 ? 10 * x : 2, 0])), 0); // < 6 samples
});
