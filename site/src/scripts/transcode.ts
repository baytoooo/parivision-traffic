// Converts a clip this browser cannot decode into one it can, in the browser, with ffmpeg.wasm.
// local_api.ts imports this module only when a clip needs it. The multi-threaded core needs
// SharedArrayBuffer, so the page must be cross-origin isolated (public/vercel.json, and
// astro.config.mjs for the dev server); scripts/copy-ffmpeg.ts puts the core in public/ffmpeg/.
// The clip is mounted, not copied: ffmpeg reads it in slices through WORKERFS, so a camera file of
// several GB takes no memory, and only its first `maxSeconds` are read. The result is 8-bit 4:2:0
// H.264 that every browser plays, 1920 px wide, at the clip's own frame rate, without audio.

import { FFFSType, FFmpeg, type LogEvent, type ProgressEvent } from "@ffmpeg/ffmpeg";
import { CONVERT_CMD } from "../config";

const CORE = "/ffmpeg/";
// The pipeline's work width, so it reads the lamps at their own size. On the first 120 s of C3896
// a 1920 px copy gave the events of the 4K original, but for a stopped vehicle split in two; a
// 1280 px copy, about 20% faster to make and analyse, also missed a jaywalking and a failure to
// yield.
const WIDTH = 1920;
// At CRF 28 the signal reader missed a yellow phase that it finds in the 4K original; at 23 it
// matched the original's phases on a 60 s cut of C3896.
const CRF = "23";
// The core starts a pool of 32 threads and cannot add one while ffmpeg runs (it would hang), so
// the thread counts stay well below that on any machine; ffmpeg gives the scale filter the
// encoder's count (with navigator.hardwareConcurrency faked to 64, a 4K clip still converts).
// Decoding 4K is most of the work; x264 at "ultrafast" needs little.
const DECODE_THREADS = Math.max(2, Math.min(8, navigator.hardwareConcurrency || 4));
const ENCODE_THREADS = Math.max(1, Math.min(4, Math.floor(DECODE_THREADS / 2)));
// @ffmpeg/ffmpeg reports no error of the core's threads, nor a core that cannot start them: its
// call then never returns. The core starts in seconds once it has its bytes, and a running ffmpeg
// reports its progress twice a second, so this long without a word means it has stopped.
const QUIET_MS = 60000;

function failed(why: string): Error {
  return new Error(`We could not convert this clip in the browser (${why}). Convert it on a computer with ffmpeg and pick the result: ${CONVERT_CMD}`);
}

/** Seconds of an ffmpeg time "hh:mm:ss.cc" in `line` after `key`, or null. */
function seconds(line: string, key: string): number | null {
  const m = new RegExp(`${key}\\s*(\\d+):(\\d+):(\\d+(?:\\.\\d+)?)`).exec(line);
  return m ? Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]) : null;
}

/** The first `maxSeconds` of `clip` as an .mp4 this browser can decode; `onProgress` gets the
 * fraction done. Aborting `signal` stops ffmpeg. */
export async function transcode(clip: Blob, maxSeconds: number, onProgress: (f: number) => void, signal: AbortSignal): Promise<Blob> {
  if (!crossOriginIsolated) throw failed("the page is not cross-origin isolated, so ffmpeg cannot run its threads");
  const ffmpeg = new FFmpeg();
  const stop = () => ffmpeg.terminate();
  signal.addEventListener("abort", stop, { once: true });
  // ffmpeg logs the clip's duration when it opens it; progress events carry the output's time
  let duration = maxSeconds;
  const lastLines: string[] = [];
  let heard = performance.now(); // the core's last log line or progress event
  let stalled!: (e: Error) => void;
  const stall = new Promise<never>((_, reject) => (stalled = reject));
  stall.catch(() => undefined); // when it fires with no call pending
  let watchdog = 0;
  const onLog = ({ message }: LogEvent) => {
    heard = performance.now();
    const d = seconds(message, "Duration:");
    if (d) duration = Math.min(d, maxSeconds);
    if (message === "Aborted()") return; // how the core logs ffmpeg's exit, even a clean one
    lastLines.push(message.replace(/^\[\S+ @ 0x[0-9a-f]+\] /, "")); // without the "[mov,mp4,... @ 0xdf0b30]" of the reader
    if (lastLines.length > 4) lastLines.shift();
    if (message.startsWith("worker sent an error!")) stalled(new Error(message)); // a thread of the core crashed
  };
  const onTime = ({ time }: ProgressEvent) => {
    heard = performance.now();
    onProgress(Math.min(1, Math.max(0, time / 1e6 / duration)));
  };
  ffmpeg.on("log", onLog);
  ffmpeg.on("progress", onTime);
  try {
    const base = new URL(CORE, location.href).href;
    let wasmURL = "";
    try {
      // the 33 MB first, however long the network takes; then the core has QUIET_MS to start
      const r = await fetch(`${base}ffmpeg-core.wasm`, { signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      wasmURL = URL.createObjectURL(new Blob([await r.arrayBuffer()], { type: "application/wasm" }));
      heard = performance.now();
      watchdog = window.setInterval(() => performance.now() - heard > QUIET_MS && stalled(new Error(`no word from it in ${QUIET_MS / 1000} s`)), 2000);
      await Promise.race([ffmpeg.load({ coreURL: `${base}ffmpeg-core.js`, wasmURL, workerURL: `${base}ffmpeg-core.worker.js` }), stall]);
    } catch (e) {
      throw failed(`ffmpeg did not load: ${e instanceof Error ? e.message : e}`);
    } finally {
      if (wasmURL) URL.revokeObjectURL(wasmURL); // the core has compiled it and hands the module to its threads
    }
    let code: number;
    try {
      await ffmpeg.createDir("/in");
      await ffmpeg.mount(FFFSType.WORKERFS, { blobs: [{ name: "clip.mp4", data: clip }] }, "/in");
      const run = ffmpeg.exec([
        "-hide_banner", // so the last lines of the log, which an error shows, say what went wrong
        "-threads", String(DECODE_THREADS),
        "-t", String(maxSeconds),
        "-i", "/in/clip.mp4",
        "-map", "0:v:0", // the picture only: no audio, no timecode or metadata tracks
        // area averaging is the right filter for shrinking and costs no more than bicubic here
        "-vf", `scale='min(${WIDTH},trunc(iw/2)*2)':-2:flags=area`,
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", CRF,
        "-threads", String(ENCODE_THREADS),
        "/out.mp4",
      ]);
      code = await Promise.race([run, stall]);
    } catch (e) {
      // the core rejects with a bare string when ffmpeg itself crashes, e.g. out of memory
      throw failed(`ffmpeg stopped: ${e instanceof Error ? e.message : e}. ${lastLines.join(" ").trim()}`.trim());
    }
    if (code !== 0) throw failed(`ffmpeg stopped with code ${code}: ${lastLines.join(" ").trim() || "no message"}`);
    const data = await ffmpeg.readFile("/out.mp4");
    if (!(data instanceof Uint8Array) || !data.length) throw failed("ffmpeg wrote no video");
    onProgress(1);
    return new Blob([data as Uint8Array<ArrayBuffer>], { type: "video/mp4" });
  } finally {
    clearInterval(watchdog);
    signal.removeEventListener("abort", stop);
    // the core holds 1 GB of memory and 32 threads; the pipeline needs them more
    ffmpeg.terminate();
  }
}
