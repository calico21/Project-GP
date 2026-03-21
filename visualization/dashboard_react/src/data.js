// src/data.js — Synthetic Data Generators (mirrors real CSV output schemas)

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

export function gP(n = 52) {
  const p = [];
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos(i * Math.PI / (n - 1)));
    p.push({
      grip: +(1.25 + 0.3 * w + 0.04 * (RN() - 0.5)).toFixed(4),
      stability: +(0.5 + 2.8 * (1 - w) + 0.15 * (RN() - 0.5)).toFixed(2),
      gen: Math.floor(RN() * 200),
      omega: +w.toFixed(3),
      params: PN.map(() => RN()),
    });
  }
  return p.sort((a, b) => a.grip - b.grip);
}

export function gCV(n = 200) {
  const d = []; let b = 1.1;
  for (let i = 0; i < n; i++) {
    b = Math.max(b, 1.1 + 0.45 * (1 - Math.exp(-i / 40)) + 0.02 * RN());
    d.push({
      iter: i,
      bestGrip: +b.toFixed(4),
      kl: +(0.012 * Math.exp(-i / 80) + 0.001 * RN()).toFixed(5),
      entropy: +(-1 + 0.3 * Math.exp(-i / 60) + 0.05 * RN()).toFixed(3),
    });
  }
  return d;
}

export function gTK(n = 360) {
  const v = []; let cx = 0, cy = 0, psi = 0;
  for (let i = 0; i < n; i++) {
    const si = i * 0.5;
    const ki = 0.08 * Math.sin(si / 15) + 0.04 * Math.sin(si / 7) + 0.02 * Math.sin(si / 3.5);
    psi += ki * 0.5; cx += Math.cos(psi) * 0.5; cy += Math.sin(psi) * 0.5;
    const vi = 12 + 8 / (1 + 3 * Math.abs(ki)) + 1.5 * RN();
    v.push({
      s: +si.toFixed(1), x: +cx.toFixed(2), y: +cy.toFixed(2),
      speed: +vi.toFixed(1),
      lat_g: +(vi * vi * ki / 9.81).toFixed(3),
      lon_g: +(i > 0 ? (vi - (v[i - 1]?.speed || vi)) / (0.5 / vi) / 9.81 : 0).toFixed(3),
      curvature: +ki.toFixed(5),
    });
  }
  return v;
}

export function gTT(n = 200) {
  const d = []; let a = 25, b = 25, c = 25, e = 25;
  for (let i = 0; i < n; i++) {
    const h = 0.4 + 0.3 * Math.sin(i / 20) + 0.1 * RN();
    const cl = 0.12;
    a = Math.min(128, Math.max(20, a + h * (0.9 + 0.2 * RN()) - cl * (a - 25) / 60));
    b = Math.min(128, Math.max(20, b + h * (1.0 + 0.15 * RN()) - cl * (b - 25) / 60));
    c = Math.min(128, Math.max(20, c + h * (0.75 + 0.2 * RN()) - cl * (c - 25) / 60));
    e = Math.min(128, Math.max(20, e + h * (0.80 + 0.15 * RN()) - cl * (e - 25) / 60));
    d.push({ s: i, tfl: +a.toFixed(1), tfr: +b.toFixed(1), trl: +c.toFixed(1), trr: +e.toFixed(1) });
  }
  return d;
}

export function gSN() {
  return PN.slice(0, 14).map(p => ({
    param: p,
    dGrip: +((RN() - 0.3) * 0.6).toFixed(3),
    dStab: +((RN() - 0.4) * 0.4).toFixed(3),
  }));
}

export function gSU() {
  const d = [];
  for (let i = 0; i < 150; i++) {
    const t = i * 0.02;
    d.push({
      t: +t.toFixed(3),
      steer: +(12 * Math.sin(t * 2.5)).toFixed(2),
      roll: +(12 * Math.sin(t * 2.5) * 0.15 + 0.3 * RN()).toFixed(2),
      heave_f: +(-12.8 + 2.5 * Math.sin(t * 3) + 0.5 * RN()).toFixed(2),
      heave_r: +(-14.2 + 2.0 * Math.sin(t * 3 + 0.5) + 0.4 * RN()).toFixed(2),
    });
  }
  return d;
}
