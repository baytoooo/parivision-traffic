// Site-wide settings. Everything a teammate may need to change before the
// final deploy lives here; nothing else in the code hard-codes these values.

/** Links in the footer. A value of "TODO" renders as a greyed-out label instead of a link. */
export const REPO_URL = "TODO";
export const WEIGHTS_URL = "TODO";
export const PREDICTIONS_URL = "/data/predictions_samples.json";

export const TEAM_NAME = "PariVision";
export const EVENT_NAME = "WIUT Hackathon 2026, CV track";

/**
 * While the files in public/data are mock data, pages show a small notice and
 * the EDA images carry a "placeholder" tag. Set to false once the real files are in.
 */
export const DATA_IS_MOCK = true;

/** Alarm threshold on the risk score, same as THETA in evaluate.py. */
export const RISK_THETA = 0.5;
/** Alarm runs closer than this are merged (MERGE_GAP in evaluate.py). */
export const RISK_MERGE_GAP = 2.0;

/** The demo analyses this many seconds from the start of a clip, at most (demo/worker.py MAX_SECONDS). */
export const UPLOAD_MAX_SECONDS = 120;
