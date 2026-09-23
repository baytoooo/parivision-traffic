// Site-wide settings. Everything a teammate may need to change before the
// final deploy lives here; nothing else in the code hard-codes these values.

/** Links in the footer. A value of "TODO" renders as a greyed-out label instead of a link. */
export const REPO_URL = "https://github.com/baytoooo/parivision-traffic";
export const WEIGHTS_URL = "https://github.com/baytoooo/parivision-traffic/tree/main/weights";
export const PREDICTIONS_URL = "/data/predictions_samples.json";

export const TEAM_NAME = "PariVision";
export const EVENT_NAME = "WIUT Hackathon 2026, CV track";

/** Alarm threshold on the risk score, same as THETA in evaluate.py. */
export const RISK_THETA = 0.5;
/** Alarm runs closer than this are merged (MERGE_GAP in evaluate.py). */
export const RISK_MERGE_GAP = 2.0;

/** The demo analyses this many seconds from the start of a clip, at most (src/scripts/local_api.ts stops reading frames there). */
export const UPLOAD_MAX_SECONDS = 120;
