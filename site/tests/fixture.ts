// Reads the parity fixtures that tools/export_parity_fixtures.py writes to tests/fixtures/.
// The C3905 and refs fixtures are committed gzipped (x.json.gz); an export writes the plain file
// and its .gz copy. A test reads the plain file when it is there and the .gz copy otherwise.
// The C3902 fixtures are not committed, so the tests that need them are skipped on a fresh clone.
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";

const DIR = new URL("./fixtures/", import.meta.url);

/** The command that writes a clip's fixtures (it needs the dev caches, see README.md). */
export const exportCommand = (clip: string) => `python tools/export_parity_fixtures.py --clip ${clip}`;

/** Raw bytes of fixtures/<path>, from the plain file or from its .gz copy. */
export async function fixtureBytes(path: string): Promise<Buffer> {
  const plain = new URL(path, DIR);
  if (existsSync(plain)) return readFile(plain);
  const gz = new URL(`${path}.gz`, DIR);
  if (existsSync(gz)) return gunzipSync(await readFile(gz));
  throw new Error(`missing fixture ${path}: run ${exportCommand(path.split("/")[0])}`);
}

/** fixtures/<path> parsed as JSON. */
export const fixtureJson = async (path: string): Promise<any> => JSON.parse((await fixtureBytes(path)).toString("utf8"));

/** True when a clip's fixtures are present (plain or gzipped). */
export const hasFixtures = (clip: string) =>
  existsSync(new URL(`${clip}/detections.json`, DIR)) || existsSync(new URL(`${clip}/detections.json.gz`, DIR));

/** Options for a test that needs a clip's fixtures: skipped, with the command to make them, when they are missing. */
export const needs = (clip: string) => ({
  skip: hasFixtures(clip) ? false : `no ${clip} fixtures in tests/fixtures (not committed): run ${exportCommand(clip)}`,
});
