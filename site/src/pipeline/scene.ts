// The scene in the reference view, as exported by tools/export_browser_assets.py:
// constants (scene.json) and rasters (scene.bin.gz). Lookups behave like
// rules.Context.sample in Python: the point is rounded half to even, and a point
// outside the image reads as 0.

import { roundHalfEven } from "./geometry.ts";

export interface LayerInfo {
  name: string;
  dtype: "uint8" | "uint16";
  offset: number;
  bytes: number;
}

/** scene.json. Only the fields the pipeline reads are typed; see the exporter for the rest. */
export interface SceneConstants {
  ref_size: [number, number];
  rules: Record<string, number>;
  /** rules.CRASH: the collision rule's thresholds (metres, m/s, s). */
  crash: Record<string, number>;
  direction_zones: Record<string, { polygon: [number, number][]; heading: number }>;
  crosswalks: Record<string, [number, number][]>;
  stop_line_sb: [[number, number], [number, number]];
  median_nose: [number, number];
  person_height_px: [number, number, number];
  signal_lamps: Record<"red" | "yellow" | "green", [number, number]>;
  signal_min_contrast: [number, number, number];
  events: { enabled: string[]; shown: string[]; gap: Record<string, number>; min_len: Record<string, number> };
  detector: { keep_classes: number[]; conf: number; model: string; input: [number, number] };
  tracking: {
    GROUPS: Record<string, number[]>;
    bytetrack: { track_high_thresh: number; track_low_thresh: number; new_track_thresh: number; match_thresh: number; fuse_score: boolean };
    buffer_sec: number;
    [k: string]: unknown;
  };
  risk: Record<string, unknown>;
  raster: {
    width: number;
    height: number;
    layers: LayerInfo[];
    dist_scale: number;
    mask_bits: string[];
    zone_bits: string[];
  };
}

export type DistName = "road_dist" | "walk_dist" | "cw_dist";

export class Scene {
  readonly c: SceneConstants;
  readonly width: number;
  readonly height: number;
  private maskBits: Uint16Array;
  private zoneBits: Uint8Array;
  private dists: Record<DistName, Uint16Array>;
  private maskIndex: Map<string, number>;
  private zoneIndex: Map<string, number>;

  constructor(c: SceneConstants, raw: Uint8Array) {
    this.c = c;
    this.width = c.raster.width;
    this.height = c.raster.height;
    const layer = (name: string) => {
      const l = c.raster.layers.find((x) => x.name === name);
      if (!l) throw new Error(`scene.bin has no layer ${name}`);
      const buf = raw.buffer.slice(raw.byteOffset + l.offset, raw.byteOffset + l.offset + l.bytes);
      return l.dtype === "uint16" ? new Uint16Array(buf) : new Uint8Array(buf);
    };
    this.maskBits = layer("masks") as Uint16Array;
    this.zoneBits = layer("zones") as Uint8Array;
    this.dists = {
      road_dist: layer("road_dist") as Uint16Array,
      walk_dist: layer("walk_dist") as Uint16Array,
      cw_dist: layer("cw_dist") as Uint16Array,
    };
    this.maskIndex = new Map(c.raster.mask_bits.map((n, i) => [n, i]));
    this.zoneIndex = new Map(c.raster.zone_bits.map((n, i) => [n, i]));
  }

  /** Pixel index of a reference point, or -1 outside the view. */
  private index(x: number, y: number): number {
    const xi = roundHalfEven(x);
    const yi = roundHalfEven(y);
    if (xi < 0 || yi < 0 || xi >= this.width || yi >= this.height) return -1;
    return yi * this.width + xi;
  }

  /** 1 inside the named mask (road, walk, crosswalk, cw_<name>, cw_<name>_zone), else 0. */
  mask(name: string, x: number, y: number): number {
    const bit = this.maskIndex.get(name);
    if (bit === undefined) throw new Error(`unknown mask ${name}`);
    const i = this.index(x, y);
    return i < 0 ? 0 : (this.maskBits[i] >> bit) & 1;
  }

  /** 1 inside the named zone (sb, nb, stop, bus, flow_approach, flow_box, junction_box), else 0. */
  zone(name: string, x: number, y: number): number {
    const bit = this.zoneIndex.get(name);
    if (bit === undefined) throw new Error(`unknown zone ${name}`);
    const i = this.index(x, y);
    return i < 0 ? 0 : (this.zoneBits[i] >> bit) & 1;
  }

  /** Distance in reference pixels (cv2.distanceTransform, to 1/16 px); 0 outside the view. */
  dist(name: DistName, x: number, y: number): number {
    const i = this.index(x, y);
    return i < 0 ? 0 : this.dists[name][i] / this.c.raster.dist_scale;
  }

  hasMask(name: string): boolean {
    return this.maskIndex.has(name);
  }
}

/**
 * Loads the scene from any byte source: fetch in the browser, the file system in tests.
 * `read("scene.json")` and `read("scene.bin.gz")` must resolve to the files' bytes.
 */
export async function loadScene(read: (name: string) => Promise<ArrayBuffer>): Promise<Scene> {
  const [json, gz] = await Promise.all([read("scene.json"), read("scene.bin.gz")]);
  const constants = JSON.parse(new TextDecoder().decode(json)) as SceneConstants;
  const stream = new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());
  return new Scene(constants, raw);
}
