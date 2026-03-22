// src/theme.js — Project-GP Design System
// ═══════════════════════════════════════════════════════════════════════
// v4.0 — ADDED: Thermal palette (5-node tire viz), constraint status,
//         energy flow palette, axis shorthand, common chart axis preset.
// ═══════════════════════════════════════════════════════════════════════

export const C = {
  // ── Surface & Layout (unchanged) ──────────────────────────────────
  bg: "#05070b", bg2: "#080c14", card: "rgba(10,15,25,0.55)",
  panel: "rgba(8,12,20,0.72)", surf: "rgba(14,20,32,0.6)",
  hover: "rgba(20,30,50,0.55)", glassB: "rgba(70,100,160,0.10)",
  b1: "rgba(35,48,75,0.45)", b2: "rgba(50,68,105,0.40)", b3: "rgba(75,95,140,0.35)",

  // ── Accent (unchanged) ────────────────────────────────────────────
  red: "#e10600", redG: "rgba(225,6,0,0.12)",
  cy: "#00d2ff", cyG: "rgba(0,210,255,0.10)",
  gn: "#00e676", gnG: "rgba(0,230,118,0.10)",
  am: "#ffab00", amG: "rgba(255,171,0,0.10)",
  pr: "#b388ff", prG: "rgba(179,136,255,0.10)",

  // ── Text (unchanged) ──────────────────────────────────────────────
  w: "#eef0f6", br: "#c0c8da", md: "#6e7a96", dm: "#3e4a64",

  // ── Typography (unchanged) ────────────────────────────────────────
  hd: "'Outfit', sans-serif",
  dt: "'Azeret Mono', 'IBM Plex Mono', monospace",
  bd: "'Outfit', sans-serif",

  // ══════════════════════════════════════════════════════════════════
  // v4.0 ADDITIONS BELOW
  // ══════════════════════════════════════════════════════════════════

  // ── 5-Node Tire Thermal Palette ───────────────────────────────────
  // Maps to Jaeger flash → inflation gas thermal stack.
  // Continuous interpolation handled by thermalHex() in canvas components;
  // these are the anchor stops for legends, annotations, threshold markers.
  th_cold:   "#0d47a1",   // < 70°C   deep blue  — cold tire, no grip
  th_cool:   "#00bcd4",   //   85°C   cyan       — warming up
  th_nom:    "#4caf50",   //  100°C   green      — optimal grip window
  th_warm:   "#ff9800",   //  120°C   amber      — getting hot
  th_hot:    "#ff5722",   //  140°C   orange-red — degradation onset
  th_crit:   "#f44336",   // > 155°C  red        — thermal abuse / blister risk

  // Gradient-safe alpha variants for area fills on thermal charts
  th_nomG:  "rgba(76,175,80,0.10)",
  th_critG: "rgba(244,67,54,0.10)",

  // ── Augmented Lagrangian Constraint Status ────────────────────────
  // Used by AL constraint monitor, track boundary warnings, solver health.
  cn_inactive: "#00e676",   // slack > 0.1  — constraint irrelevant
  cn_marginal: "#ffab00",   // 0 < slack < 0.1 — approaching active
  cn_binding:  "#e10600",   // slack ≈ 0 — binding, AL multiplier nonzero
  cn_inactiveG: "rgba(0,230,118,0.10)",
  cn_marginalG: "rgba(255,171,0,0.10)",
  cn_bindingG:  "rgba(225,6,0,0.10)",

  // ── Port-Hamiltonian Energy Flow Palette ──────────────────────────
  // One color per energy compartment for Sankey, stacked area, dH/dt audit.
  // Order follows the Hamiltonian decomposition: H = T + V_s + V_arb + H_res
  e_ke:     "#29b6f6",   // Kinetic energy  ½mv² + ½Iω²
  e_spe:    "#66bb6a",   // Spring PE       ½k·δ²
  e_arb:    "#9ccc65",   // ARB PE          ½k_arb·Δδ²
  e_hnet:   "#ab47bc",   // H_net residual  (neural, should be bounded)
  e_diss:   "#ef5350",   // R_net dissipation rate
  e_therm:  "#ffa726",   // Tire thermal energy absorption
  e_aero:   "#78909c",   // Aerodynamic drag work

  // Alpha variants for stacked area fills
  e_keG:    "rgba(41,182,246,0.15)",
  e_speG:   "rgba(102,187,106,0.15)",
  e_arbG:   "rgba(156,204,101,0.15)",
  e_hnetG:  "rgba(171,71,188,0.20)",
  e_dissG:  "rgba(239,83,80,0.15)",
  e_thermG: "rgba(255,167,38,0.15)",
  e_aeroG:  "rgba(120,144,156,0.12)",

  // ── Tire Slip / Force Visualization ───────────────────────────────
  // For friction carpet, hysteresis Lissajous, GP uncertainty band.
  sl_lat:  "#00d2ff",   // Lateral force Fy  (cyan family)
  sl_lon:  "#00e676",   // Longitudinal Fx   (green family)
  sl_comb: "#ffab00",   // Combined |F|      (amber)
  sl_gp:   "#b388ff",   // GP uncertainty σ  (purple family)

  // ── Trust Region / FIM Visualization ──────────────────────────────
  tr_accepted: "#00e676",
  tr_rejected: "#e10600",
  tr_boundary: "rgba(50,68,105,0.40)",   // δ circle stroke
};

// ── Glass morphism preset (unchanged) ───────────────────────────────
export const GL = {
  background: C.card,
  backdropFilter: "blur(24px) saturate(1.5)",
  WebkitBackdropFilter: "blur(24px) saturate(1.5)",
  border: `1px solid ${C.glassB}`,
  borderRadius: 12,
  boxShadow: "0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)",
};

// ── Chart grid stroke (unchanged) ───────────────────────────────────
export const GS = "rgba(25,35,55,0.5)";

// ── Tooltip style (unchanged) ───────────────────────────────────────
export const TT = {
  backgroundColor: "rgba(8,12,20,0.95)", backdropFilter: "blur(16px)",
  border: `1px solid ${C.b2}`, borderRadius: 8,
  fontSize: 11, fontFamily: C.dt, color: C.br, padding: "10px 12px",
};

// ── v4.0: Reusable Recharts axis preset ─────────────────────────────
// Drop into any <XAxis {...AX}/> or <YAxis {...AX}/> to get consistent
// tick styling without repeating the object literal in every module.
export const AX = {
  tick: { fontSize: 9, fill: C.dm, fontFamily: C.dt },
  stroke: C.b1,
};

// ── Fonts (unchanged) ───────────────────────────────────────────────
export const FONTS_URL =
  "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Azeret+Mono:wght@400;500;600;700&display=swap";

// ══════════════════════════════════════════════════════════════════════
// v4.0: Thermal color interpolation utilities
// ══════════════════════════════════════════════════════════════════════
// Used by ThermalQuintet.jsx and TirePhysicsModule.jsx for continuous
// temperature-to-color mapping across the 5-node Jaeger stack.

const THERMAL_STOPS = [
  [60,  0x0d, 0x47, 0xa1],   // deep blue
  [85,  0x00, 0xbc, 0xd4],   // cyan
  [100, 0x4c, 0xaf, 0x50],   // green
  [120, 0xff, 0x98, 0x00],   // amber
  [140, 0xff, 0x57, 0x22],   // orange-red
  [160, 0xf4, 0x43, 0x36],   // red
];

/**
 * Continuous temperature → hex color. Linearly interpolates between the
 * anchor stops defined in THERMAL_STOPS. Clamps at the boundaries.
 *
 * @param {number} t  Temperature in °C
 * @returns {string}  CSS hex color, e.g. "#4caf50"
 */
export function thermalHex(t) {
  if (t <= THERMAL_STOPS[0][0]) return `#${THERMAL_STOPS[0].slice(1).map(c => c.toString(16).padStart(2, "0")).join("")}`;
  for (let i = 1; i < THERMAL_STOPS.length; i++) {
    if (t <= THERMAL_STOPS[i][0]) {
      const f = (t - THERMAL_STOPS[i - 1][0]) / (THERMAL_STOPS[i][0] - THERMAL_STOPS[i - 1][0]);
      const r = Math.round(THERMAL_STOPS[i - 1][1] * (1 - f) + THERMAL_STOPS[i][1] * f);
      const g = Math.round(THERMAL_STOPS[i - 1][2] * (1 - f) + THERMAL_STOPS[i][2] * f);
      const b = Math.round(THERMAL_STOPS[i - 1][3] * (1 - f) + THERMAL_STOPS[i][3] * f);
      return `#${(r << 16 | g << 8 | b).toString(16).padStart(6, "0")}`;
    }
  }
  return `#${THERMAL_STOPS[THERMAL_STOPS.length - 1].slice(1).map(c => c.toString(16).padStart(2, "0")).join("")}`;
}

/**
 * Temperature → discrete status color (for badges, dots, text).
 * Uses the semantic anchor tokens from C.th_*.
 *
 * @param {number} t  Temperature in °C
 * @returns {string}  One of C.th_cold … C.th_crit
 */
export function thermalColor(t) {
  if (t < 70)  return C.th_cold;
  if (t < 85)  return C.th_cool;
  if (t < 100) return C.th_nom;
  if (t < 120) return C.th_warm;
  if (t < 140) return C.th_hot;
  return C.th_crit;
}

/**
 * AL constraint slack → status color.
 * @param {number} slack  Constraint slack value (>0 = feasible)
 * @returns {string}
 */
export function constraintColor(slack) {
  if (slack > 0.1) return C.cn_inactive;
  if (slack > 0.0) return C.cn_marginal;
  return C.cn_binding;
}