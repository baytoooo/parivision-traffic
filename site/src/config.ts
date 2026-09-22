// Site-wide settings. Everything a teammate may need to change before the
// final deploy lives here; nothing else in the code hard-codes these values.

/** Base URL of the demo API (FastAPI on a Hugging Face Space). No trailing slash. */
export const API_BASE_URL = "https://parivision-traffic-demo.hf.space";

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

/** Upload limits shown on the demo page and checked in the browser before upload. */
export const UPLOAD_MAX_SECONDS = 120;
export const UPLOAD_MAX_BYTES = 500 * 1024 * 1024;
