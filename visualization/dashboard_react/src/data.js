// src/data.js — Project-GP Synthetic Data Generators

const sd = (s) => {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return () => { h ^= h << 13; h ^= h >> 17; h ^= h << 5; return (h >>> 0) / 4294967296; };
};
const RN = sd("projectgp2026ter26fsg");

export const PN = [
  "k_f","k_r","arb_f","arb_r","c_low_f","c_low_r","c_high_f","c_high_r",
  "camber_f","camber_r","toe_f","toe_r","castor","anti_dive_f","anti_squat",
  "diff_lock","heave_k_f","heave_k_r","bumpstop_f","bumpstop_r",
  "h_rc_f","h_rc_r","h_cg","brake_bias","tire_p_f","tire_p_r","ackermann","steer_ratio",
];
export const PU = [
  "N/m","N/m","N/m","N/m","Ns/m","Ns/m","Ns/m","Ns/m",
  "deg","deg","deg","deg","deg","","","",
  "N/m","N/m","mm","mm","mm","mm","mm","","kPa","kPa","","",
];

// Existing generators
export function gP(n = 52) { const p = []; for (let i = 0; i < n; i++) { const w = .5 * (1 - Math.cos(i * Math.PI / (n - 1))); p.push({ grip: +(1.25 + .3 * w + .04 * (RN() - .5)).toFixed(4), stability: +(.5 + 2.8 * (1 - w) + .15 * (RN() - .5)).toFixed(2), gen: Math.floor(RN() * 200), omega: +w.toFixed(3), params: PN.map(() => RN()) }); } return p.sort((a, b) => a.grip - b.grip); }
export function gCV(n = 200) { const d = []; let b = 1.1; for (let i = 0; i < n; i++) { b = Math.max(b, 1.1 + .45 * (1 - Math.exp(-i / 40)) + .02 * RN()); d.push({ iter: i, bestGrip: +b.toFixed(4), kl: +(.012 * Math.exp(-i / 80) + .001 * RN()).toFixed(5), entropy: +(-1 + .3 * Math.exp(-i / 60) + .05 * RN()).toFixed(3) }); } return d; }
export function gTK(n = 360) { const v = []; let cx = 0, cy = 0, psi = 0; for (let i = 0; i < n; i++) { const si = i * .5, ki = .08 * Math.sin(si / 15) + .04 * Math.sin(si / 7) + .02 * Math.sin(si / 3.5); psi += ki * .5; cx += Math.cos(psi) * .5; cy += Math.sin(psi) * .5; const vi = 12 + 8 / (1 + 3 * Math.abs(ki)) + 1.5 * RN(); v.push({ s: +si.toFixed(1), x: +cx.toFixed(2), y: +cy.toFixed(2), speed: +vi.toFixed(1), lat_g: +(vi * vi * ki / 9.81).toFixed(3), lon_g: +(i > 0 ? (vi - (v[i - 1]?.speed || vi)) / (.5 / vi) / 9.81 : 0).toFixed(3), curvature: +ki.toFixed(5) }); } return v; }
export function gTT(n = 200) { const d = []; let a = 25, b = 25, c = 25, e = 25; for (let i = 0; i < n; i++) { const h = .4 + .3 * Math.sin(i / 20) + .1 * RN(), cl = .12; a = Math.min(128, Math.max(20, a + h * (.9 + .2 * RN()) - cl * (a - 25) / 60)); b = Math.min(128, Math.max(20, b + h * (1 + .15 * RN()) - cl * (b - 25) / 60)); c = Math.min(128, Math.max(20, c + h * (.75 + .2 * RN()) - cl * (c - 25) / 60)); e = Math.min(128, Math.max(20, e + h * (.8 + .15 * RN()) - cl * (e - 25) / 60)); d.push({ s: i, tfl: +a.toFixed(1), tfr: +b.toFixed(1), trl: +c.toFixed(1), trr: +e.toFixed(1) }); } return d; }
export function gSN() { return PN.slice(0, 14).map(p => ({ param: p, dGrip: +((RN() - .3) * .6).toFixed(3), dStab: +((RN() - .4) * .4).toFixed(3) })); }
export function gSU() { const d = []; for (let i = 0; i < 150; i++) { const t = i * .02; d.push({ t: +t.toFixed(3), steer: +(12 * Math.sin(t * 2.5)).toFixed(2), roll: +(12 * Math.sin(t * 2.5) * .15 + .3 * RN()).toFixed(2), heave_f: +(-12.8 + 2.5 * Math.sin(t * 3) + .5 * RN()).toFixed(2), heave_r: +(-14.2 + 2 * Math.sin(t * 3 + .5) + .4 * RN()).toFixed(2) }); } return d; }

// ── NEW: Load transfer per distance step ────────────────────────────
export function gLoadTransfer(track) {
  return track.map(p => {
    const m = 300, hcg = 0.33, lf = 0.8525, lr = 0.6975, L = lf + lr, tw = 1.2;
    const latG = Number(p.lat_g) || 0, lonG = Number(p.lon_g) || 0;
    const dFzLat = m * 9.81 * Math.abs(latG) * hcg / tw;
    const dFzLon = m * 9.81 * lonG * hcg / L;
    const staticF = m * 9.81 * lr / L / 2, staticR = m * 9.81 * lf / L / 2;
    return {
      s: p.s,
      Fz_fl: +(staticF - dFzLon / 2 + (latG > 0 ? -dFzLat / 2 : dFzLat / 2)).toFixed(0),
      Fz_fr: +(staticF - dFzLon / 2 + (latG > 0 ? dFzLat / 2 : -dFzLat / 2)).toFixed(0),
      Fz_rl: +(staticR + dFzLon / 2 + (latG > 0 ? -dFzLat / 2 : dFzLat / 2)).toFixed(0),
      Fz_rr: +(staticR + dFzLon / 2 + (latG > 0 ? dFzLat / 2 : -dFzLat / 2)).toFixed(0),
    };
  });
}

// ── NEW: Damper velocity histogram ──────────────────────────────────
export function gDamperHist() {
  const bins = [];
  for (let v = -0.3; v <= 0.3; v += 0.025) {
    const x = v / 0.12;
    const countF = Math.round(800 * Math.exp(-x * x / 2) * (1 + 0.3 * RN()));
    const countR = Math.round(600 * Math.exp(-x * x / 1.8) * (1 + 0.25 * RN()));
    bins.push({ vel: +v.toFixed(3), front: countF, rear: countR });
  }
  return bins;
}

// ── NEW: Frequency response (Bode magnitude) ───────────────────────
export function gFreqResponse() {
  const d = [];
  for (let f = 0.5; f <= 25; f += 0.5) {
    const wn_f = 2 * Math.PI * 1.8, wn_r = 2 * Math.PI * 2.1, zeta = 0.35;
    const w = 2 * Math.PI * f;
    const magF = 1 / Math.sqrt(Math.pow(1 - (w / wn_f) ** 2, 2) + (2 * zeta * w / wn_f) ** 2);
    const magR = 1 / Math.sqrt(Math.pow(1 - (w / wn_r) ** 2, 2) + (2 * zeta * w / wn_r) ** 2);
    d.push({ freq: f, front_dB: +(20 * Math.log10(magF)).toFixed(1), rear_dB: +(20 * Math.log10(magR)).toFixed(1) });
  }
  return d;
}

// ── NEW: Lap delta (actual vs optimal) ──────────────────────────────
export function gLapDelta(track) {
  let cumDelta = 0;
  return track.map((p, i) => {
    const optSpeed = 12 + 10 / (1 + 3 * Math.abs(Number(p.curvature) || 0));
    const actual = Number(p.speed) || 0;
    const dt = 0.5 / Math.max(actual, 1);
    const dtOpt = 0.5 / Math.max(optSpeed, 1);
    cumDelta += (dt - dtOpt);
    return { s: p.s, delta: +cumDelta.toFixed(4), vActual: actual, vOptimal: +optSpeed.toFixed(1) };
  });
}

// ── NEW: Grip utilisation per corner ────────────────────────────────
export function gGripUtil(track) {
  return track.filter((_, i) => i % 4 === 0).map(p => {
    const lg = Math.abs(Number(p.lat_g) || 0), longG = Math.abs(Number(p.lon_g) || 0);
    const mu = 1.35;
    const combined = Math.sqrt(lg * lg + longG * longG);
    return { s: p.s, utilisation: +Math.min(combined / mu, 1.0).toFixed(3), combined: +combined.toFixed(3) };
  });
}
