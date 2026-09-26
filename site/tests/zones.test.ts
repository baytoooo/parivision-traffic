// Where the demo table puts a piece of evidence (src/lib/classes.ts evidenceZone and readableNote),
// against where() in tools/make_site_data.py, which places the evidence of the Results page.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { evidenceZone, nearestZone, NOTE_ZONE, readableNote, ZONE_BY_KEY, ZONES } from "../src/lib/classes.ts";

/** The point of the 1920x1080 reference view at a zone marker, in reference pixels. */
const at = (key: string): [number, number] => [ZONE_BY_KEY[key].x * 19.2, ZONE_BY_KEY[key].y * 10.8];

test("notes that name a place get the Results page's zone and wording", () => {
  const cases: [label: string, note: string, zone: string, readable: string][] = [
    ["failure_to_yield", "north_sb", "north_crossing", "north crossing, SB half"],
    ["failure_to_yield", "north_nb", "north_crossing", "north crossing, NB half"],
    ["failure_to_yield", "west", "west_crossing", "west crossing"],
    ["stopped_vehicle", "NB lane", "nb_carriageway", "NB lane"],
    ["stopped_vehicle", "junction box", "junction_box", "junction box"],
    ["wrong_way", "sb", "sb_approach", "SB carriageway"],
    ["wrong_way", "nb", "nb_carriageway", "NB carriageway"],
  ];
  for (const [label, note, zone, readable] of cases) {
    // the note wins over the actor's position
    assert.equal(evidenceZone(label, note, at("median_nose")), zone, `${label} ${note}`);
    assert.equal(readableNote(note), readable, note);
  }
});

test("other notes keep their words; the rule's own place or the actor's position gives the zone", () => {
  assert.equal(evidenceZone("red_light", "red for 2.4s", at("west_crossing")), "stop_line");
  assert.equal(evidenceZone("stop_line", "", null), "stop_line");
  assert.equal(evidenceZone("congestion", "max 9 standing", null), "sb_approach");
  assert.equal(evidenceZone("illegal_u_turn", "U-turn round the median nose", null), "median_nose");
  // jaywalking and accidents have no place of their own: the marker nearest to the first actor
  assert.equal(evidenceZone("jaywalking", "", at("west_crossing")), "west_crossing");
  assert.equal(evidenceZone("jaywalking", "", [1300, 850]), "junction_box");
  assert.equal(evidenceZone("accident", "met at 7 m/s", [1180, 560]), "median_nose");
  assert.equal(evidenceZone("jaywalking", "", null), "junction_box", "no track: the junction box, as make_site_data.py");
  for (const note of ["red for 2.4s", "met at 7 m/s", "", "U-turn round the median nose"]) assert.equal(readableNote(note), note);
});

test("nearestZone picks the nearest marker", () => {
  for (const z of ZONES) assert.equal(nearestZone(...at(z.key)), z.key);
  // a little nearer the stop line (27, 42) than the SB approach (22, 28)
  assert.equal(nearestZone(25 * 19.2, 36 * 10.8), "stop_line");
  assert.equal(nearestZone(24 * 19.2, 34 * 10.8), "sb_approach");
});

test("the markers and the note table are those of tools/make_site_data.py", async () => {
  const py = await readFile(new URL("../../tools/make_site_data.py", import.meta.url), "utf8");
  const block = (name: string) => {
    const m = new RegExp(`^${name} = \\{([\\s\\S]*?)\\}\\n`, "m").exec(py);
    assert.ok(m, `${name} in make_site_data.py`);
    return m[1];
  };
  const zones = [...block("ZONES").matchAll(/"(\w+)": \((\d+), (\d+)\)/g)].map(([, key, x, y]) => ({ key, x: Number(x), y: Number(y) }));
  assert.deepEqual(zones, ZONES.map(({ key, x, y }) => ({ key, x, y })));
  const notes = Object.fromEntries([...block("NOTE_ZONE").matchAll(/"([^"]+)": \("(\w+)", "([^"]+)"\)/g)].map(([, note, zone, text]) => [note, [zone, text]]));
  assert.deepEqual(notes, NOTE_ZONE);
  const rules = Object.fromEntries([...block("CLASS_ZONE").matchAll(/"(\w+)": "(\w+)"/g)].map(([, label, zone]) => [label, zone]));
  assert.ok(Object.keys(rules).length >= 4);
  for (const [label, zone] of Object.entries(rules)) assert.equal(evidenceZone(label, "some note", null), zone, label);
});
