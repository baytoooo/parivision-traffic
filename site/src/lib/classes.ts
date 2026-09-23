// The 14 event classes, in the order the task lists them, with the text the
// site shows for each. `status` says what the pipeline does with the class:
//   on   - a rule exists and the class is emitted
//   off  - a rule exists but the class is not emitted yet
//   none - no rule yet
// Keep `status` in step with ENABLED in src/parivision/events.py.

export type ClassStatus = "on" | "off" | "none";

export interface ClassMeta {
  key: string;
  name: string;
  code: string;
  status: ClassStatus;
  rule: string;
}

export const CLASSES: ClassMeta[] = [
  {
    key: "accident",
    name: "Accident",
    code: "ACC",
    status: "none",
    rule: "Not emitted by Part A. Part B gives a per-frame probability that one starts within 5 s; the sample clips contain no accidents to tune it on.",
  },
  {
    key: "near_miss",
    name: "Near miss",
    code: "NM",
    status: "none",
    rule: "No rule yet. The Part B time-to-collision score is the closest signal we have, and we have not turned it into segments.",
  },
  {
    key: "red_light",
    name: "Red light",
    code: "RED",
    status: "on",
    rule: "A southbound vehicle's front crosses the stop line after the lamp has been red for at least 1 s, with at least 1.5 s of red still to go. Yellow does not count. The segment ends when its track ends, at most 60 s later.",
  },
  {
    key: "wrong_way",
    name: "Wrong way",
    code: "WW",
    status: "on",
    rule: "A vehicle faster than 40 px/s heads more than 120 degrees away from its carriageway's direction for at least 1.5 s. Each carriageway has one legal direction in the reference view.",
  },
  {
    key: "illegal_u_turn",
    name: "Illegal U-turn",
    code: "UT",
    status: "off",
    rule: "We find U-turns (a southbound vehicle turns round the median nose, within 130 px of it, and leaves northbound) but do not emit the class, because we have not confirmed which U-turns are illegal at this junction.",
  },
  {
    key: "stopped_vehicle",
    name: "Stopped vehicle",
    code: "SV",
    status: "on",
    rule: "A vehicle stays under 12 px/s for 10 s or more on the northbound carriageway or in the junction box on the southbound path to the south exit. The bus stop is excluded, and so are the southbound approach, where cars queue at the red, and the northbound lanes east of x = 1720 px.",
  },
  {
    key: "jaywalking",
    name: "Jaywalking",
    code: "JAY",
    status: "on",
    rule: "A person who is not riding a bike has their feet on the carriageway, clearly outside every zebra, for at least 1 s. Riders are found by a two-wheeler box over the person or a speed above 110 px/s.",
  },
  {
    key: "failure_to_yield",
    name: "Failure to yield",
    code: "FTY",
    status: "on",
    rule: "A car, bus or truck moving at 20 px/s or more drives across a zebra while a walking pedestrian is out on it, not at the kerb, within 160 px of the vehicle. Motorbikes are skipped. The segment runs from the front entering the zebra to the rear leaving it.",
  },
  {
    key: "illegal_turn",
    name: "Illegal turn",
    code: "IT",
    status: "none",
    rule: "No rule yet. The plan is to compare the lane a vehicle holds at the stop line with the arm it leaves by.",
  },
  {
    key: "solid_line_crossing",
    name: "Solid line crossing",
    code: "SLC",
    status: "none",
    rule: "No rule yet. The solid lane lines before the southbound stop line would be drawn once in the reference view and a lane change across them flagged.",
  },
  {
    key: "stop_line",
    name: "Stop line",
    code: "STOP",
    status: "on",
    rule: "A southbound vehicle stands for at least 1 s with its front more than 8 px past the stop line while the lamp is red. The segment ends at green, or earlier if its track ends.",
  },
  {
    key: "congestion",
    name: "Congestion",
    code: "CONG",
    status: "on",
    rule: "Southbound traffic stands still on green. From 8 s into the green, 6 or more vehicles under 15 px/s on roughly the last 40 m of the approach and past the stop line on the way to the south exit make a jam, and so do 5 or more past the stop line alone. A jam must start on green or yellow and last at least 6 s, and it carries on into the red while 5 or more still stand past the stop line.",
  },
  {
    key: "road_obstacle",
    name: "Road obstacle",
    code: "OBS",
    status: "none",
    rule: "No rule yet. The detector already keeps six COCO animal classes, so an animal on the road could feed it.",
  },
  {
    key: "fire_smoke",
    name: "Fire or smoke",
    code: "FIRE",
    status: "none",
    rule: "No rule yet, and nothing in the sample clips to test one on.",
  },
];

export const CLASS_KEYS = CLASSES.map((c) => c.key);
export const CLASS_BY_KEY: Record<string, ClassMeta> = Object.fromEntries(CLASSES.map((c) => [c.key, c]));

export function className(key: string): string {
  return CLASS_BY_KEY[key]?.name ?? key.replace(/_/g, " ");
}

/** Places on the junction, with their position on the 1920x1080 reference view (percent). */
export interface ZoneMeta {
  key: string;
  name: string;
  x: number;
  y: number;
}

export const ZONES: ZoneMeta[] = [
  { key: "sb_approach", name: "SB approach", x: 22, y: 28 },
  { key: "stop_line", name: "SB stop line", x: 27, y: 42 },
  { key: "north_crossing", name: "North crossing", x: 47, y: 56 },
  { key: "median_nose", name: "Median nose", x: 64, y: 49 },
  { key: "nb_carriageway", name: "NB carriageway", x: 62, y: 22 },
  { key: "west_crossing", name: "West crossing", x: 25, y: 83 },
  { key: "junction_box", name: "Junction box", x: 66, y: 76 },
];

export const ZONE_BY_KEY: Record<string, ZoneMeta> = Object.fromEntries(ZONES.map((z) => [z.key, z]));

/** Where each class's rule looks, used when an event has no zone of its own. */
const ZONE_FOR_CLASS: Record<string, string> = {
  red_light: "stop_line",
  stop_line: "stop_line",
  solid_line_crossing: "sb_approach",
  congestion: "sb_approach",
  failure_to_yield: "north_crossing",
  jaywalking: "junction_box",
  stopped_vehicle: "junction_box",
  wrong_way: "nb_carriageway",
  illegal_u_turn: "median_nose",
};

export function zoneFor(label: string, zone?: string): string {
  if (zone && ZONE_BY_KEY[zone]) return zone;
  return ZONE_FOR_CLASS[label] ?? "junction_box";
}
