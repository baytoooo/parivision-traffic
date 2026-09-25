// rules.Context + events.detect_from_context on the Python trajectories and signal phases give rules.json.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { collisions, Context, detectFromContext } from "../src/pipeline/rules.ts";
import { loadScene } from "../src/pipeline/scene.ts";
import type { Trajectory } from "../src/pipeline/types.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;
const fixture = async (clip: string, name: string) =>
  JSON.parse(await readFile(new URL(`./fixtures/${clip}/${name}`, import.meta.url), "utf8"));

/** trajectories.json as Trajectory objects; box and conf are float32 arrays in Python. */
const toTrajectory = (d: Trajectory): Trajectory => ({
  ...d,
  box: d.box.map((b) => b.map(Math.fround)),
  conf: d.conf.map(Math.fround),
});

const TOL = 0.01; // s

for (const clip of ["C3905", "C3902"]) {
  test(`rules match rules.json (${clip})`, async () => {
    const scene = await loadScene(read);
    const trajs = (await fixture(clip, "trajectories.json")).map(toTrajectory);
    const signal = await fixture(clip, "signal.json");
    const { duration } = await fixture(clip, "detections.json");
    const { H } = await fixture(clip, "alignment.json");
    const expected = await fixture(clip, "rules.json");

    const ctx = new Context(trajs, signal.times, signal.phases, duration, scene);
    assert.ok(Math.abs(ctx.dt - expected.dt) < 1e-12, `dt ${ctx.dt} vs ${expected.dt}`);
    assert.deepEqual(ctx.people.map((tr) => tr.tid).sort((a, b) => a - b), expected.people);

    const { events, evidence } = detectFromContext(ctx, (H as number[][]).flat());
    assert.equal(evidence.length, expected.evidence.length, "evidence count");
    evidence.forEach((ev, i) => {
      const e = expected.evidence[i];
      const what = `evidence ${i} (${e.label} ${e.start}-${e.end})`;
      assert.equal(ev.label, e.label, what);
      assert.ok(Math.abs(ev.start - e.start) <= TOL && Math.abs(ev.end - e.end) <= TOL, `${what}: got ${ev.start}-${ev.end}`);
      assert.deepEqual(ev.actors, e.actors, what);
      assert.equal(ev.note, e.note, what);
    });
    assert.equal(events.length, expected.events.length, "event count");
    events.forEach(([s, e, label], i) => {
      const [xs, xe, xl] = expected.events[i];
      assert.equal(label, xl, `event ${i}`);
      assert.ok(Math.abs(s - xs) <= TOL && Math.abs(e - xe) <= TOL, `event ${i} ${label}: ${s}-${e} vs ${xs}-${xe}`);
    });
  });
}

// The two clips never produce red_light, congestion or wrong_way. The exporter perturbs the same inputs
// until they do and writes what rules.py gives in rules_perturbed.json; the same perturbations here must
// give the same evidence (times to 1e-6 s) and events.
type Row = [string, number, number, number[], string];
const reverse = (tr: Trajectory): Trajectory => ({
  ...tr,
  box: [...tr.box].reverse(),
  foot: [...tr.foot].reverse(),
  vel: [...tr.vel].reverse().map(([x, y]) => [-x, -y]),
  height: [...tr.height].reverse(),
  conf: [...tr.conf].reverse(),
});
const scaleVel = (tr: Trajectory, k: number): Trajectory => ({ ...tr, vel: tr.vel.map(([x, y]) => [x * k, y * k]) });
const roll = <T>(a: T[], k: number) => a.map((_, i) => a[(((i - k) % a.length) + a.length) % a.length]);

const PERTURBATIONS: Record<string, (trajs: Trajectory[], phases: string[]) => [Trajectory[], string[]]> = {
  phases_rolled_37: (trajs, phases) => [trajs, roll(phases, 37)],
  slow_on_green: (trajs, phases) => [trajs.map((tr) => scaleVel(tr, 0.3)), phases.map(() => "green")],
  every_10th_reversed: (trajs, phases) => [trajs.map((tr, i) => (i % 10 === 0 ? reverse(tr) : tr)), phases],
};

for (const clip of ["C3905", "C3902"]) {
  for (const [name, change] of Object.entries(PERTURBATIONS)) {
    test(`rules.py and the port agree on perturbed input (${clip}, ${name})`, async () => {
      const want = (await fixture(clip, "rules_perturbed.json"))[name];
      const scene = await loadScene(read);
      const trajs0 = (await fixture(clip, "trajectories.json")).map(toTrajectory);
      const signal = await fixture(clip, "signal.json");
      const { duration } = await fixture(clip, "detections.json");
      const { H } = await fixture(clip, "alignment.json");
      const [trajs, phases] = change(trajs0, signal.phases);
      const { events, evidence } = detectFromContext(new Context(trajs, signal.times, phases, duration, scene), (H as number[][]).flat());
      const counts: Record<string, number> = {};
      for (const ev of evidence) counts[ev.label] = (counts[ev.label] ?? 0) + 1;
      assert.deepEqual(counts, want.counts);
      const rows: Row[] = evidence.filter((ev) => want.labels.includes(ev.label)).map((ev) => [ev.label, ev.start, ev.end, ev.actors, ev.note]);
      assert.equal(rows.length, want.evidence.length, "evidence count");
      rows.forEach((g, i) => {
        const w: Row = want.evidence[i];
        assert.deepEqual([g[0], g[3], g[4]], [w[0], w[3], w[4]], `evidence ${i}`);
        assert.ok(Math.abs(g[1] - w[1]) <= 1e-6 && Math.abs(g[2] - w[2]) <= 1e-6, `evidence ${i}: ${g[1]}-${g[2]} vs ${w[1]}-${w[2]}`);
      });
      assert.deepEqual(events.filter((ev) => want.labels.includes(ev[2])), want.events);
    });
  }
}

// Two cars crossing the SB stop line (285,527)-(930,457) at x = 414, where the line is at y = 513:
// the front goes from y 503 to 523 (signed side -6450 -> +6450), so each crossing is exactly halfway
// between two samples. The light is red from 50 s to 90 s (phase samples every 0.25 s), so the notes
// read red for exactly 10.25 s and 11.25 s: f"{x:.1f}" rounds those ties to even ("10.2", "11.2")
// where toFixed would print "10.3" and "11.3". Expected: rules.red_light / detect_from_context.
async function redLightScene() {
  const scene = await loadScene(read);
  const car = (tid: number, t: number[]): Trajectory => ({
    tid, group: "vehicle", cls: 2, t,
    box: [[404, 473, 424, 503], [404, 493, 424, 523], [404, 600, 424, 630]],
    foot: [[414, 497], [414, 517], [414, 625.5]],
    vel: [[0, 40], [0, 40], [0, 30]],
    height: [30, 30, 30],
    conf: [0.9, 0.9, 0.9].map(Math.fround),
  });
  const times = Array.from({ length: 481 }, (_, i) => i * 0.25);
  const phases = times.map((t) => (t >= 50 && t < 90 ? "red" : "green"));
  const ctx = new Context([car(1000001, [60, 60.5, 75.175]), car(1000002, [61, 61.5, 63])], times, phases, 120, scene);
  return detectFromContext(ctx, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

test("red_light crossing time and note match rules.py on a hand-made crossing (.1f ties round to even)", async () => {
  const { evidence } = await redLightScene();
  assert.deepEqual(
    evidence.map((ev) => [ev.label, ev.start, ev.end, ev.actors, ev.note]),
    [["red_light", 60.25, 75.175, [1000001], "red for 10.2s"], ["red_light", 61.25, 63, [1000002], "red for 11.2s"]],
  );
});

// segments.finalize (segments.ts, not this module) rounds with roundHalfEven(x * 100) / 100, which is
// np.round; the evidence times in finalize are Python floats, so Python uses round(x, 2) on the exact
// binary value. They differ when x * 100 lands on .5 but x is below the tie: 75.175 is
// 75.17499999999999715..., Python gives 75.17, the port 75.18 (likewise 115.115 -> 115.11 vs 115.12).
// At 4.995 fps one frame time in 50 (k * 0.2002 with k = 25 mod 50) ends in 5 at the third decimal.
test("final events round like Python round(x, 2)", async () => {
  const { events } = await redLightScene();
  assert.deepEqual(events, [[60.25, 75.17, "red_light"]]);
});

// tests/test_core.py _crash_scene: two cars meet at t = 5 s (east at 6 m/s, north at 4 m/s, at
// 0.05 m/px) and stand. Expected: rules.collisions on the same arrays.
function crashScene(): Trajectory[] {
  const t = Array.from({ length: 101 }, (_, k) => k * 0.1);
  const car = (tid: number, foot: number[][], vel: number[][]): Trajectory => ({
    tid, group: "vehicle", cls: 2, t, box: t.map(() => [0, 0, 0, 0]), foot, vel,
    height: t.map(() => 40), conf: t.map(() => Math.fround(0.9)),
  });
  const tt = t.map((x) => Math.min(x, 5.0));
  return [
    car(1000001, tt.map((x) => [100 + 120 * x, 500]), t.map((x) => [x <= 5.0 ? 120 : 0, 0])),
    car(1000002, tt.map((x) => [740, 900 - 80 * x]), t.map((x) => [0, x <= 5.0 ? -80 : 0])),
  ];
}

test("collisions matches rules.py on a hand-made crash and ignores a car joining a queue", async () => {
  const scene = await loadScene(read);
  const radius = scene.c.risk.RADIUS_M as Record<string, number>;
  const ev = collisions(crashScene(), () => 0.05, scene.c.crash, radius);
  assert.deepEqual(ev.map((e) => [Number(e.start.toFixed(6)), Number(e.end.toFixed(6)), e.actors, e.note]),
    [[4.8, 6.1, [1000001, 1000002], "met at 7 m/s"]]);
  const [a, b] = crashScene();
  const last = b.foot[b.foot.length - 1];
  const queued: Trajectory = { ...b, foot: b.foot.map(() => [...last]), vel: b.vel.map(() => [0, 0]) };
  assert.deepEqual(collisions([a, queued], () => 0.05, scene.c.crash, radius), []);
});
