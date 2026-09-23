// Small numeric helpers shared by the pipeline, written to match NumPy/OpenCV where
// the Python code relies on their exact behaviour.

/** np.round: halves go to the nearest even integer. */
export function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** A 3x3 homography as a row-major array of 9 numbers. */
export type Mat3 = number[];

/** registration.warp_points: apply H to [x, y] points. */
export function warpPoints(pts: number[][], H: Mat3): number[][] {
  return pts.map(([x, y]) => {
    const w = H[6] * x + H[7] * y + H[8];
    return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
  });
}

export function matMul(A: Mat3, B: Mat3): Mat3 {
  const out = new Array(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) out[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
  return out;
}

export function matInv(m: Mat3): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map(
    (v) => v / det,
  );
}

export function diag(sx: number, sy: number): Mat3 {
  return [sx, 0, 0, 0, sy, 0, 0, 0, 1];
}

/** scene.signed_side: > 0 on the camera side of a left-to-right line, < 0 beyond it. Not normalised
 * (a cross product in px^2), as in Python; divide by the line length for a distance. */
export function signedSide(pts: number[][], line: [[number, number], [number, number]]): number[] {
  const [[x1, y1], [x2, y2]] = line;
  return pts.map(([x, y]) => (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1));
}

export function median(a: number[]): number {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** np.percentile with the default linear interpolation. */
export function percentile(a: number[], q: number): number {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const pos = ((s.length - 1) * q) / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
