export const meta = {
  name: 'adjudicate-dev-set',
  description: 'Look at every disagreement between our predictions and the dev labels and decide who is right',
  phases: [{ title: 'Adjudicate', detail: 'one agent per disagreement, frames at 0.2-0.5 s' }],
}

const ROOT = '/Users/bayto/Desktop/claude/traffic-events'
const CLASSES = ['accident', 'near_miss', 'red_light', 'wrong_way', 'illegal_u_turn', 'stopped_vehicle',
  'jaywalking', 'failure_to_yield', 'illegal_turn', 'solid_line_crossing', 'stop_line', 'congestion',
  'road_obstacle', 'fire_smoke']

const SCHEMA = {
  type: 'object',
  properties: {
    real_event: { type: 'boolean', description: 'is there really an event of this class here under the definition?' },
    label: { type: 'string', enum: CLASSES, description: 'the class if real (may differ from the claimed one)' },
    start: { type: 'number', description: 'correct start per the convention (if real)' },
    end: { type: 'number', description: 'correct end per the convention (if real)' },
    category: {
      type: 'string',
      enum: ['label_missed_it', 'label_wrong', 'true_detection', 'false_detection', 'missed_by_model',
        'boundary_only', 'ambiguous'],
      description: 'fp items: label_missed_it (real, labels lacked it) / false_detection; fn items: missed_by_model (real, we did not find it) / label_wrong (not real); boundary_only if both saw it but times differ a lot; ambiguous if the definition does not settle it',
    },
    cause: { type: 'string', description: 'for false detections or misses: the concrete cause, e.g. "person standing on the kerb counted as on the road", "tracker lost the car behind the bus"' },
    reason: { type: 'string' },
  },
  required: ['real_event', 'label', 'start', 'end', 'category', 'cause', 'reason'],
}

function prompt(it) {
  const what = it.kind === 'fp'
    ? `Our model predicted a "${it.label}" segment ${it.start.toFixed(1)}-${it.end.toFixed(1)} s that our dev labels do not have. Actors the model used: ${it.actors}.`
    : `Our dev labels have a "${it.label}" segment ${it.start.toFixed(1)}-${it.end.toFixed(1)} s that the model did not find. ${it.note || ''}${args.items_file ? `The entry with id "${it.id}" in ${ROOT}/${args.items_file} has a "note" field with what the dev-set verifier saw (it may be wrong); read it first.` : ''}`
  return `You are adjudicating a traffic-event dev set. Project root: ${ROOT}. Read ${ROOT}/docs/scene.md and
${ROOT}/docs/labeling.md first (layout, directions, crossings, class definitions, start/end conventions) and look at
${ROOT}/docs/scene_overlay.jpg. Clip ${it.clip}: proxy cache/proxy/${it.clip}.mp4 (960x540, 10 fps; proxy pixel =
reference pixel / 2, but afternoon clips C3902/C3905 are framed ~20-60 px differently from the reference).
Signal phase (SB vehicles) around this time: ${it.signal}.
Make contact sheets with:
  cd ${ROOT} && .venv/bin/python tools/sheet.py cache/proxy/${it.clip}.mp4 --start A --end B --step S --cols C --tile W [--crop x0,y0,x1,y1] --out out/sheets/adj/${it.id}_<n>.jpg
and look at them with Read (use crops and 0.2-0.5 s steps around the segment, from ${Math.max(0, it.start - 4).toFixed(1)} to ${(it.end + 4).toFixed(1)} s).

${what}

Decide from the frames whether a real "${it.label}" event (as defined in docs/labeling.md) happens here, and if it
does, its correct start and end. Be strict and concrete: name who did what and where. If it is real but a different
class fits better, say so in label. Precision matters: do not call something real unless you can see it.`
}

const results = await parallel(args.items.map(it => () =>
  agent(prompt(it), { label: `adj:${it.clip}:${it.kind}:${it.label}:${it.start.toFixed(0)}`, phase: 'Adjudicate', schema: SCHEMA })
    .then(v => ({ ...it, verdict: v }))))
const done = results.filter(Boolean)
const tally = {}
for (const r of done) {
  if (!r.verdict) continue
  const k = `${r.kind}:${r.verdict.category}`
  tally[k] = (tally[k] || 0) + 1
}
log(JSON.stringify(tally))
return done
