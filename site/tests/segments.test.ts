// The cases of tests/test_core.py for segments.py, against the port.
import assert from "node:assert/strict";
import { test } from "node:test";
import { finalize, runs, union } from "../src/pipeline/segments.ts";

test("union merges overlaps and small gaps", () => {
  assert.deepEqual(union([[0, 2], [1, 3], [5, 6]]), [[0, 3], [5, 6]]);
  assert.deepEqual(union([[0, 2], [2.4, 3]], 0.5), [[0, 3]]);
  assert.deepEqual(union([]), []);
});

test("runs bridges gaps", () => {
  const t = Array.from({ length: 30 }, (_, i) => i * 0.1);
  const flags = t.map((x) => x < 1 || (x > 1.25 && x < 2));
  const last = t.filter((_, i) => flags[i]).at(-1)!;
  assert.deepEqual(runs(t, flags, 0.45), [[0, last]]);
  assert.equal(runs(t, flags, 0.3).length, 2);
});

test("finalize output is valid for evaluate", () => {
  const events = finalize({ jaywalking: [[1, 3], [2.5, 4], [10, 10.3]], red_light: [[-1, 2]] }, 5, {}, { jaywalking: 0.5 });
  assert.deepEqual(events, [[0, 2, "red_light"], [1, 4, "jaywalking"]]);
});
