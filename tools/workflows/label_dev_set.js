export const meta = {
  name: 'label-dev-set',
  description: 'Label traffic events in the sample clips (two specialist passes per window), verify each claim, merge per class',
  phases: [
    { title: 'Label', detail: 'pedestrian and vehicle specialists per 40 s window' },
    { title: 'Verify', detail: 'one skeptical verifier per claimed event, refines boundaries' },
  ],
}

const ROOT = '/Users/bayto/Desktop/claude/traffic-events'
const CLASSES = ['accident', 'near_miss', 'red_light', 'wrong_way', 'illegal_u_turn', 'stopped_vehicle',
  'jaywalking', 'failure_to_yield', 'illegal_turn', 'solid_line_crossing', 'stop_line', 'congestion',
  'road_obstacle', 'fire_smoke']
const VIDEOS = args.videos  // [{id, duration}]
const WINDOW = 40

const SPECIALTIES = {
  pedestrian: {
    classes: ['jaywalking', 'failure_to_yield', 'near_miss'],
    focus: `Watch the PEOPLE. Every pedestrian who is on the carriageway outside the zebra stripes is a jaywalking
candidate: people cutting across the junction box, crossing the avenue away from the north crossing, walking in
the road beside a crossing, crossing the west arm outside the west zebra. Then watch every zebra while people are
on it: any vehicle that drives across that zebra while someone is on it (or stepping onto it) in or next to its
path is failure_to_yield. Near misses involving pedestrians (someone jumps back, a car brakes hard for them) are
near_miss. Cyclists and scooter riders are vehicles, not pedestrians.`,
  },
  vehicle: {
    classes: CLASSES.filter(c => c !== 'jaywalking' && c !== 'failure_to_yield'),
    focus: `Watch the VEHICLES (cars, buses, trucks, motorbikes, bicycles and scooters on the road). Use the signal
timeline: any southbound vehicle whose front crosses the stop line while the phase is red is red_light; a
southbound vehicle that stops with its front past the stop line on red is stop_line. Look for vehicles standing
still 10 s or more on the carriageway that are not in the red-light queue (stopped_vehicle), jams that persist on
green (congestion), anything moving against the traffic direction (wrong_way), U-turns (label them
illegal_u_turn and say exactly where the turn happened), turns from the wrong lane (illegal_turn), lane changes
over solid lines (solid_line_crossing), hard braking or swerves (near_miss), collisions (accident), debris or
animals on the road (road_obstacle), smoke or fire (fire_smoke). Ignore buses at the NB bus stop and parked cars.`,
  },
}

const LABEL_SCHEMA = {
  type: 'object',
  properties: {
    events: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          label: { type: 'string', enum: CLASSES },
          start: { type: 'number' },
          end: { type: 'number' },
          confidence: { type: 'number', description: '0..1, how sure you are the event is real under the definition' },
          actor: { type: 'string', description: 'who: e.g. "white sedan, SB lane 2", "man in black shirt"' },
          evidence: { type: 'string', description: 'what you saw, with times and positions in proxy pixels' },
        },
        required: ['label', 'start', 'end', 'confidence', 'actor', 'evidence'],
      },
    },
    notes: { type: 'array', items: { type: 'string' }, description: 'ambiguous things worth a human look' },
  },
  required: ['events', 'notes'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'rejected', 'relabelled'] },
    label: { type: 'string', enum: CLASSES },
    start: { type: 'number' },
    end: { type: 'number' },
    reason: { type: 'string' },
  },
  required: ['verdict', 'label', 'start', 'end', 'reason'],
}

function windowsFor(v) {
  const out = []
  for (let t = 0; t < v.duration - 1; t += WINDOW) out.push([t, Math.min(v.duration, t + WINDOW)])
  return out
}

function signalText(v, t0, t1) {
  const segs = (args.signals[v.id] || []).filter(s => s[1] >= t0 - 10 && s[0] <= t1 + 25)
  return segs.map(s => `${s[0].toFixed(1)}-${s[1].toFixed(1)} s: ${s[2]}`).join('; ') || 'unknown'
}

const common = `You are building a dev set for a traffic-event detection task. Project root: ${ROOT}.
Read ${ROOT}/docs/scene.md (layout, directions, crossings, signal) and ${ROOT}/docs/labeling.md (class
definitions and start/end conventions) before anything else, and look at ${ROOT}/docs/scene_overlay.jpg.
Video proxies are 960x540 at 10 fps (proxy pixel = reference pixel / 2). Make contact sheets with
  cd ${ROOT} && .venv/bin/python tools/sheet.py cache/proxy/<clip>.mp4 --start A --end B --step S --cols C --tile W [--crop x0,y0,x1,y1] --out <file.jpg>
and look at them with the Read tool. Always pass a unique --out under out/sheets/<your-own-folder>/.
Tiles wider than 480 px keep pedestrians visible; use --crop to zoom on an area and a small --step
(0.2 to 0.5 s) to pin down boundaries. Times are seconds from the first frame.
Only report what you actually saw in frames. Precision matters more than recall: an invented event costs as
much as a missed one, but do not skip real events because they are small or brief.`

function labelPrompt(v, t0, t1, spec) {
  const s = SPECIALTIES[spec]
  return `${common}

Clip ${v.id} (cache/proxy/${v.id}.mp4, ${v.duration.toFixed(1)} s). Your window: ${t0.toFixed(0)} to ${t1.toFixed(0)} s.
Signal phase (SB approach) around this window: ${signalText(v, t0, t1)}.

${s.focus}

Classes you are responsible for: ${s.classes.join(', ')}. Ignore the others (another labeller handles them).

Method: first scan the whole window with sheets at 1 s steps (for example 2 cols x tile 960, or 3 cols x tile 640,
eight to twelve frames per sheet). Then, for every candidate, zoom in (crop + 0.2-0.5 s steps) to confirm it and to
find start and end exactly per the conventions. Report an event only if it STARTS inside [${t0.toFixed(0)}, ${t1.toFixed(0)}) s;
follow it past the window end (up to 30 s) to find where it ends. One entry per actor (merging happens later).
Put anything uncertain (possible U-turns, unclear jaywalks, odd manoeuvres) in notes with times.`
}

function verifyPrompt(v, ev) {
  return `${common}

A labeller claims this event in clip ${v.id} (cache/proxy/${v.id}.mp4):
  label: ${ev.label}
  time: ${ev.start.toFixed(1)} to ${ev.end.toFixed(1)} s
  actor: ${ev.actor}
  evidence: ${ev.evidence}
Signal phase around it: ${signalText(v, ev.start, ev.end)}.

Your job is to try to REFUTE it, then fix its boundaries if it survives. Look at frames from ${Math.max(0, ev.start - 4).toFixed(1)}
to ${(ev.end + 4).toFixed(1)} s (crop to the actor, 0.2-0.5 s steps). Check: did it really happen as described; does it meet the
class definition in docs/labeling.md (for jaywalking: really on the carriageway and outside the stripes; for
failure_to_yield: a pedestrian really on that zebra in or next to the vehicle's path while the vehicle crossed it;
for red_light: front crossed the stop line while red, not yellow); is another class a better fit.
Verdict: "confirmed" (keep, with corrected start/end), "relabelled" (real, but a different class; give it and the
boundaries for that class), or "rejected" (did not happen or does not meet the definition). If unsure after
looking, reject. Boundaries follow the start/end conventions to 0.1 s.`
}

const jobs = []
for (const v of VIDEOS) for (const [t0, t1] of windowsFor(v)) for (const spec of Object.keys(SPECIALTIES)) jobs.push({ v, t0, t1, spec })
log(`${jobs.length} labelling jobs over ${VIDEOS.length} clips`)

const results = await pipeline(
  jobs,
  job => agent(labelPrompt(job.v, job.t0, job.t1, job.spec), {
    label: `label:${job.v.id}:${job.t0}:${job.spec}`, phase: 'Label', schema: LABEL_SCHEMA,
  }),
  (lab, job) => {
    if (!lab) return { job, raw: [], verified: [], notes: [] }
    const claims = lab.events.filter(e => e.end > e.start && SPECIALTIES[job.spec].classes.includes(e.label))
    return parallel(claims.map(ev => () => agent(verifyPrompt(job.v, ev), {
      label: `verify:${job.v.id}:${ev.label}:${ev.start.toFixed(0)}`, phase: 'Verify', schema: VERDICT_SCHEMA,
    }).then(ver => ({ claim: ev, verdict: ver })))).then(vs => ({
      job, raw: claims, notes: lab.notes, verified: vs.filter(Boolean),
    }))
  },
)

const perVideo = {}
for (const r of results.filter(Boolean)) {
  const id = r.job.v.id
  perVideo[id] = perVideo[id] || { claims: [], notes: [] }
  perVideo[id].notes.push(...r.notes.map(n => `[${r.job.t0}-${r.job.t1} ${r.job.spec}] ${n}`))
  for (const x of r.verified) perVideo[id].claims.push({ ...x.claim, verdict: x.verdict })
  for (const c of r.raw) if (!r.verified.find(x => x.claim === c)) perVideo[id].claims.push({ ...c, verdict: null })
}
const summary = Object.fromEntries(Object.entries(perVideo).map(([id, p]) => [id, {
  claims: p.claims.length,
  confirmed: p.claims.filter(c => c.verdict && c.verdict.verdict !== 'rejected').length,
}]))
log(JSON.stringify(summary))
return perVideo
