// The scene rasters, read through Scene, give the Python Context's values at random points.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { loadScene } from "../src/pipeline/scene.ts";

const PUB = new URL("../public/pipeline/", import.meta.url);
const FIX = new URL("./fixtures/C3905/", import.meta.url);
const read = async (name: string) => (await readFile(new URL(name, PUB))).buffer as ArrayBuffer;

test("scene lookups match rules.Context.sample", async () => {
  const scene = await loadScene(read);
  const s = JSON.parse(await readFile(new URL("scene_samples.json", FIX), "utf8"));
  const pts: [number, number][] = s.points;
  for (const [key, vals] of Object.entries(s) as [string, number[]][]) {
    if (key === "points") continue;
    const get = (x: number, y: number) =>
      key.endsWith("_dist") ? scene.dist(key as "road_dist" | "walk_dist" | "cw_dist", x, y)
      : key.startsWith("zone_") ? scene.zone(key.slice(5), x, y)
      : scene.mask(key === "road" || key === "walk" ? key : key, x, y);
    let bad = 0;
    pts.forEach(([x, y], i) => {
      const tol = key.endsWith("_dist") ? 1 / 32 + 1e-3 : 0;
      if (Math.abs(get(x, y) - vals[i]) > tol) bad++;
    });
    assert.equal(bad, 0, `${key}: ${bad} of ${pts.length} points differ`);
  }
});
