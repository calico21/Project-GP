// ═══════════════════════════════════════════════════════════════════════════
// src/TorqueVectoringSimModule.jsx  —  Project-GP Dashboard v5.1
// ═══════════════════════════════════════════════════════════════════════════
// Interactive torque vectoring simulator. Top-down canvas physics sim that
// lets the user compare Simple (open-diff, equal split) vs Advanced
// (SOCP-inspired PI yaw-moment) TV strategies at the wheel level.
//
// Physics: linearised single-track (bicycle) model, Pacejka-linear tire
//          forces, 4× substep Euler integration per frame.
// Controls: WASD / Arrow keys (with auto-centre on release) + sliders.
// Rendering: Canvas 2D, 60 fps rAF loop, React state updated at ~15 fps.
//
// Integration (3 lines in App.jsx — see patches below):
//   NAV : { key:"tvsim", label:"TV Simulator", icon:"◎" }
//   Import : import TorqueVectoringSimModule from "./TorqueVectoringSimModule.jsx"
//   Route  : case "tvsim": return <TorqueVectoringSimModule />;
// ═══════════════════════════════════════════════════════════════════════════

import React, { useRef, useEffect, useState, useCallback } from "react";
import { C, GL } from "./theme.js";
import { KPI, Pill } from "./components.jsx";

// ─────────────────────────────────────────────────────────────────────────────
// §1  Vehicle parameters  (Ter26 RWD, single-track linearisation)
// ─────────────────────────────────────────────────────────────────────────────
const V = {
  m:    280,      // kg   total sprung+unsprung
  Iz:   120,      // kg·m²  yaw inertia
  L:    1.53,     // m   wheelbase
  lf:   0.76,     // m   CG → front axle
  lr:   0.77,     // m   CG → rear  axle
  tw:   1.20,     // m   rear track width
  rw:   0.254,    // m   wheel radius
  Cf:   25000,    // N/rad  front cornering stiffness (linearised Pacejka)
  Cr:   28000,    // N/rad  rear  cornering stiffness
  Tp:   120,      // Nm  peak torque per rear wheel
  Cd:   1.3,      // aero drag coefficient
  Af:   1.0,      // m²  frontal area
  rho:  1.225,    // kg/m³ air density
  dMax: 0.40,     // rad max steer angle (≈23°)
  Kus:  0.008,    // rad·s²/m²  understeer gradient
};

// PI TV controller gains  (tuned to match SOCP bandwidth at ~50 Hz)
const KP_TV = 1500;   // Nm/(rad/s)  — faster snap, corrects exit oscillation
const KI_TV = 14;     // Nm/rad      — low: avoids windup on corner exit

// ─────────────────────────────────────────────────────────────────────────────
// §2  Track geometry  (ellipse, rendered with ctx.ellipse → pixel-perfect)
// ─────────────────────────────────────────────────────────────────────────────
const OW  = 50;   // m  semi-axis x (outer centreline)
const OH  = 22;   // m  semi-axis y
const THW = 5.0;  // m  half track-width
const SCALE = 5.0; // px / m   canvas→world transform

// Initial state: west side of track, heading north, gentle entry speed
const S0 = Object.freeze({
  x:  -(OW - THW * 0.5),
  y:  0,
  psi: Math.PI / 2,   // heading north (+y in world)
  vx: 9,              // m/s  longitudinal speed
  vy: 0,
  r:  0,              // rad/s  yaw rate
  wz_int: 0,          // TV integral state
});

// ─────────────────────────────────────────────────────────────────────────────
// §3  Physics  —  single Euler substep
// ─────────────────────────────────────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function physicsStep(s, steer, throttle, brake, mode, dt) {
  const vx = Math.max(0.5, s.vx);
  const v  = Math.hypot(vx, s.vy);

  // Ackermann single-track reference yaw rate
  const wzRef = (v * steer) / (V.L * (1 + V.Kus * v * v));

  // Small-angle slip  (valid for |α| < ~10°, well-satisfied here)
  const alphaF =  steer - (s.vy + V.lf * s.r) / vx;
  const alphaR  = -(s.vy - V.lr * s.r) / vx;

  // Linearised lateral forces  (friction-limited)
  const FyF = clamp(V.Cf * alphaF, -4500, 4500);
  const FyR  = clamp(V.Cr * alphaR, -5000, 5000);

  // Drive / brake
  const Tdrive = throttle * V.Tp * 2;   // rear axle total [Nm]
  const Fbrake = brake    * 4000;        // [N]  longitudinal

  let Trl, Trr, wz_int = s.wz_int;

  if (mode === "simple") {
    // Simple TV: steer-proportional feedforward — outer wheel bias, no yaw feedback.
    // Positive steer = left turn → outer wheel = RR → dT_s > 0 → Trr > Trl.
    const dT_s = clamp(steer * Tdrive * 0.30, -V.Tp * 0.45, V.Tp * 0.45);
    Trl = clamp(Tdrive / 2 - dT_s, 0, V.Tp);
    Trr = clamp(Tdrive / 2 + dT_s, 0, V.Tp);
  } else {
    const err = wzRef - s.r;
    // Anti-windup: exponential decay scales with inverse error magnitude.
    // When |err| < 0.08 rad/s (straight/exit), integral bleeds down fast.
    const decayRate = 1 - 14 * dt * Math.max(0, 1 - Math.abs(err) / 0.08);
    wz_int = clamp(s.wz_int * decayRate + err * dt, -3.5, 3.5);
    const dM = KP_TV * err + KI_TV * wz_int;
    const dT = clamp(dM / (V.tw / V.rw), -V.Tp * 0.55, V.Tp * 0.55);
    // Thrust-preserving allocation: recover any floor-clamp deficit.
    // Mirrors SOCP behaviour — TV is a differential overlay, not a thrust reduction.
    const rawRl = Tdrive / 2 - dT;
    const rawRr = Tdrive / 2 + dT;
    Trl = clamp(rawRl, 0, V.Tp);
    Trr = clamp(rawRr, 0, V.Tp);
    const deficit = Math.max(0, -rawRl) + Math.max(0, -rawRr);
    if (deficit > 0) {
      Trl = Math.min(V.Tp, Trl + deficit * 0.5);
      Trr = Math.min(V.Tp, Trr + deficit * 0.5);
    }
  }

  const Fx    = (Trl + Trr) / V.rw - Fbrake;
  const Mtv   = ((Trr - Trl) / V.rw) * (V.tw / 2);  // [Nm] total TV yaw moment
  const Fdrag = 0.5 * V.rho * V.Cd * V.Af * v * v;

  // Body-frame accelerations  (centripetal coupling terms included)
  const ax = (Fx - Fdrag - FyF * Math.sin(steer)) / V.m  +  s.vy * s.r;
  const ay = (FyF * Math.cos(steer) + FyR)         / V.m  -  vx   * s.r;
  const az = (FyF * Math.cos(steer) * V.lf - FyR * V.lr + Mtv) / V.Iz;

  // World-frame position update
  const sinP = Math.sin(s.psi), cosP = Math.cos(s.psi);
  return {
    x:      s.x   + (vx * cosP - s.vy * sinP) * dt,
    y:      s.y   + (vx * sinP + s.vy * cosP) * dt,
    psi:    s.psi + s.r * dt,
    vx:     Math.max(0, s.vx + ax * dt),
    vy:     s.vy + ay * dt,
    r:      s.r  + az * dt,
    wz_int,
    // ── Diagnostics carried in state for zero-allocation display reads ──
    Trl, Trr,
    wzRef,
    wzErr:  wzRef - s.r,
    Mtv,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Canvas helpers
// ─────────────────────────────────────────────────────────────────────────────

/** World (m, y-up) → canvas (px, y-down) */
function w2c(wx, wy, cw, ch) {
  return [cw / 2 + wx * SCALE, ch / 2 - wy * SCALE];
}

function drawTrack(ctx, cw, ch) {
  const cx0 = cw / 2, cy0 = ch / 2;

  // Grass background
  ctx.fillStyle = "#04090404";
  ctx.fillRect(0, 0, cw, ch);

  // Subtle grid (gives spatial reference)
  ctx.strokeStyle = "rgba(20,35,60,0.35)";
  ctx.lineWidth = 0.5;
  const step = 10 * SCALE;
  for (let gx = cx0 % step; gx < cw; gx += step) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, ch); ctx.stroke(); }
  for (let gy = cy0 % step; gy < ch; gy += step) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cw, gy); ctx.stroke(); }

  // Track surface
  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW + THW) * SCALE, (OH + THW) * SCALE, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#0c1220";
  ctx.fill();

  // Infield punch-out
  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW - THW) * SCALE, (OH - THW) * SCALE, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#050b08";
  ctx.fill();

  // Infield grass texture dot
  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW - THW - 2) * SCALE, (OH - THW - 2) * SCALE, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#060d07";
  ctx.fill();

  // Outer kerb line
  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW + THW) * SCALE, (OH + THW) * SCALE, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(200,220,255,0.14)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Inner kerb line
  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW - THW) * SCALE, (OH - THW) * SCALE, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(200,220,255,0.14)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Centreline dashes
  ctx.beginPath();
  ctx.ellipse(cx0, cy0, OW * SCALE, OH * SCALE, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  ctx.setLineDash([10, 14]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Start/finish line
  const [slx, sly] = w2c(-(OW), 0, cw, ch);
  ctx.beginPath();
  ctx.moveTo(slx, sly - THW * SCALE);
  ctx.lineTo(slx, sly + THW * SCALE);
  ctx.strokeStyle = "rgba(255,255,255,0.30)";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawTrail(ctx, trail, mode) {
  if (trail.length < 2) return;
  ctx.beginPath();
  trail.forEach(([tx, ty], i) => {
    i === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty);
  });
  ctx.strokeStyle = mode === "advanced" ? "rgba(0,210,255,0.50)" : "rgba(225,6,0,0.50)";
  ctx.lineWidth   = 2.5;
  ctx.lineJoin    = "round";
  ctx.stroke();
}

function drawCar(ctx, cw, ch, state, mode) {
  const [px, py] = w2c(state.x, state.y, cw, ch);
  const { psi } = state;
  const Trl = state.Trl ?? 0;
  const Trr = state.Trr ?? 0;

  ctx.save();
  ctx.translate(px, py);
  // Canvas rotation derivation: nose drawn at (0, -BL), which should point in
  // world heading direction. World→canvas: rot_canvas = π/2 - psi.
  ctx.rotate(Math.PI / 2 - psi);

  const BL = 14, BW = 6;  // px half-dims (body length / width)

  // ── Chassis ───────────────────────────────────────────────────────────
  ctx.beginPath();
  ctx.rect(-BW, -BL, BW * 2, BL * 2);
  ctx.fillStyle = mode === "advanced" ? "rgba(4,20,38,0.92)" : "rgba(28,6,6,0.92)";
  ctx.fill();
  ctx.strokeStyle = mode === "advanced" ? C.cy : C.red;
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  // ── Nose cone ─────────────────────────────────────────────────────────
  ctx.beginPath();
  ctx.moveTo(0,         -(BL + 6));
  ctx.lineTo(-BW * 0.55, -BL + 2);
  ctx.lineTo( BW * 0.55, -BL + 2);
  ctx.closePath();
  ctx.fillStyle = mode === "advanced" ? C.cy : C.red;
  ctx.fill();

  // ── Rear wing (decorative bar) ─────────────────────────────────────────
  ctx.beginPath();
  ctx.rect(-(BW + 3), BL - 4, (BW + 3) * 2, 2.5);
  ctx.fillStyle = mode === "advanced" ? C.cy : C.red;
  ctx.fill();

  // ── Cockpit highlight ─────────────────────────────────────────────────
  ctx.beginPath();
  ctx.ellipse(0, 0, BW * 0.55, BL * 0.35, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.fill();
  ctx.strokeStyle = "rgba(150,180,220,0.2)";
  ctx.lineWidth = 1;
  ctx.stroke();

  // ── Wheels ────────────────────────────────────────────────────────────
  // Rendered as rectangles (top-down realistic view)
  const wheels = [
    { x: -(BW + 3.5), y: -BL * 0.50, T: 0,   driven: false, label: "FL" },
    { x:  (BW + 3.5), y: -BL * 0.50, T: 0,   driven: false, label: "FR" },
    { x: -(BW + 3.5), y:  BL * 0.50, T: Trl, driven: true,  label: "RL" },
    { x:  (BW + 3.5), y:  BL * 0.50, T: Trr, driven: true,  label: "RR" },
  ];

  wheels.forEach(w => {
    const ratio = w.driven ? Math.min(1, w.T / V.Tp) : 0;

    // Torque → colour  (green→yellow→orange→red as load increases)
    let borderColor;
    if (!w.driven || ratio < 0.05) {
      borderColor = "rgba(80,100,130,0.55)";
    } else if (ratio < 0.4) {
      const t = ratio / 0.4;
      borderColor = `rgba(${Math.round(t * 255)},230,0,${0.55 + t * 0.35})`;
    } else if (ratio < 0.75) {
      const t = (ratio - 0.4) / 0.35;
      borderColor = `rgba(255,${Math.round((1 - t) * 180)},0,0.9)`;
    } else {
      borderColor = `rgba(255,${Math.round((1 - (ratio - 0.75) / 0.25) * 80)},0,1.0)`;
    }

    // Tyre body
    ctx.beginPath();
    ctx.rect(w.x - 3, w.y - 5, 6, 10);
    ctx.fillStyle = "rgba(20,25,35,0.92)";
    ctx.fill();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth   = 1.8;
    ctx.stroke();

    // Glow for heavy torque
    if (ratio > 0.45) {
      ctx.shadowBlur  = 9;
      ctx.shadowColor = borderColor;
      ctx.beginPath();
      ctx.rect(w.x - 3, w.y - 5, 6, 10);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  });

  ctx.restore();
}

function renderHUD(ctx, state, mode, steer) {
  const spd = Math.hypot(state.vx, state.vy) * 3.6;
  const modeColor = mode === "advanced" ? "#00d2ff" : "#e10600";

  ctx.font      = "bold 11px 'Azeret Mono', monospace";
  ctx.fillStyle = "rgba(192,200,218,0.85)";
  ctx.fillText(`${spd.toFixed(0)} km/h`, 10, 20);

  ctx.font      = "bold 9px 'Azeret Mono', monospace";
  ctx.fillStyle = modeColor;
  ctx.fillText(mode === "advanced" ? "◎ ADVANCED TV" : "⊗ SIMPLE TV", 10, 34);

  // Steering indicator (arc at bottom-left)
  const arcX = 28, arcY = 72, arcR = 18;
  ctx.beginPath();
  ctx.arc(arcX, arcY, arcR, Math.PI, 2 * Math.PI, false);
  ctx.strokeStyle = "rgba(70,90,130,0.4)";
  ctx.lineWidth   = 2.5;
  ctx.stroke();
  // Pointer
  const angle = Math.PI + (steer / V.dMax) * (Math.PI / 2);
  ctx.beginPath();
  ctx.moveTo(arcX, arcY);
  ctx.lineTo(arcX + Math.cos(angle) * arcR, arcY + Math.sin(angle) * arcR);
  ctx.strokeStyle = modeColor;
  ctx.lineWidth   = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(arcX, arcY, 3, 0, Math.PI * 2);
  ctx.fillStyle = modeColor;
  ctx.fill();
  ctx.font = "8px 'Azeret Mono', monospace";
  ctx.fillStyle = "rgba(100,120,160,0.8)";
  ctx.fillText("STEER", arcX - 13, arcY + 11);
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function TorqueBar({ label, value, max, color, driven }) {
  const pct = driven ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
      <div style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, fontWeight: 700 }}>{label}</div>
      <div style={{ width: 28, height: 68, background: "rgba(20,30,50,0.6)", borderRadius: 4, position: "relative", overflow: "hidden", border: "1px solid rgba(50,68,105,0.5)" }}>
        <div style={{
          position: "absolute", bottom: 0, width: "100%", height: `${pct}%`,
          background: driven
            ? `linear-gradient(to top, ${color}, ${color}88)`
            : "rgba(50,65,95,0.4)",
          transition: "height 0.05s linear",
        }} />
        {/* Peak marker at 80% */}
        <div style={{ position: "absolute", bottom: "80%", width: "100%", height: 1, background: "rgba(255,255,255,0.12)" }} />
      </div>
      <div style={{ fontSize: 8, fontWeight: 700, color: driven ? color : C.dm, fontFamily: C.dt }}>
        {driven ? value.toFixed(0) : "—"}
        <span style={{ fontSize: 6.5, opacity: 0.6 }}>{driven ? " Nm" : ""}</span>
      </div>
      {!driven && <div style={{ fontSize: 6.5, color: C.dm, fontFamily: C.dt }}>coast</div>}
    </div>
  );
}

const F = ({ c, children }) => (
  <span style={{ color: c, fontWeight: 700 }}>{children}</span>
);

function TheoryBlock({ mode }) {
  const isAdv = mode === "advanced";
  const ac    = isAdv ? C.cy : C.red;

  return (
    <div style={{ ...GL, padding: "16px 20px", borderLeft: `3px solid ${ac}`, transition: "border-color 0.4s" }}>
      <div style={{ fontSize: 10, fontWeight: 800, color: ac, fontFamily: C.dt, letterSpacing: 2.5, marginBottom: 10 }}>
        {isAdv ? "ADVANCED TV — SOCP-INSPIRED PI YAW-MOMENT CONTROLLER" : "SIMPLE TV — OPEN DIFFERENTIAL EQUIVALENT"}
      </div>

      {!isAdv ? (
        <>
          <p style={{ fontSize: 9, color: C.br, fontFamily: C.dt, lineHeight: 1.85, margin: 0 }}>
            Equal rear torque split: <F c={C.red}>T_rl = T_rr = T_total / 2</F>. No yaw moment generated by
            the drivetrain. In a corner, lateral load transfer shifts vertical load to the outer wheel
            (raising its friction budget), but both rear wheels receive identical drive torque — the
            inner wheel saturates its friction envelope first, inducing <F c={C.am}>understeer</F>.
            The car pushes wide of the apex. Driver must trail-brake to rotate, sacrificing exit speed.
            Under high-speed corners with heavy throttle, loss of yaw moment is pronounced.
          </p>
          <pre style={{ fontFamily: "monospace", fontSize: 8, color: C.dm, marginTop: 12, padding: "10px 14px", background: "rgba(5,8,15,0.85)", borderRadius: 6, lineHeight: 2.1 }}>
{`M_yaw_TV  =  0 Nm                        (no differential moment)
T_rl      =  T_rr  =  T_total / 2        (fixed equal split)`}
          </pre>
        </>
      ) : (
        <>
          <p style={{ fontSize: 9, color: C.br, fontFamily: C.dt, lineHeight: 1.85, margin: 0 }}>
            The WMPC generates a reference yaw rate from the Ackermann single-track model:
            <F c={C.cy}> ψ̇_ref = v·δ / (L·(1 + K_us·v²))</F>. The deviation
            <F c={C.gn}> ε = ψ̇_ref − ψ̇_actual </F>
            drives a PI controller producing a yaw-moment demand
            <F c={C.am}> ΔM = Kp·ε + Ki·∫ε dt</F>. This is decomposed into a differential torque
            ΔT = ΔM·r_w / t_w and applied asymmetrically: the outer wheel receives surplus torque
            (acting as a <F c={C.cy}>mechanical yaw actuator</F>), the inner wheel is reduced.
            The SOCP allocator guarantees a globally optimal solution inside the friction ellipse each timestep.
          </p>
          <pre style={{ fontFamily: "monospace", fontSize: 8, color: C.dm, marginTop: 12, padding: "10px 14px", background: "rgba(5,8,15,0.85)", borderRadius: 6, lineHeight: 2.1 }}>
{`ψ̇_ref  =  v · δ / (L · (1 + 0.008 · v²))     [rad/s]
ε       =  ψ̇_ref  −  ψ̇_actual
ΔT      =  clamp( (Kp·ε + Ki·∫ε) · rw / tw,  ±0.65·T_peak )
T_rr    =  T/2 + ΔT    │    T_rl  =  T/2  −  ΔT`}
          </pre>
        </>
      )}

      {/* Comparison callout */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 14 }}>
        {[
          { label: "Corner entry",   simple: "Understeer — pushes wide",        adv: "ψ̇ tracked — follows apex" },
          { label: "Mid-corner",     simple: "Inner wheel spins, grip lost",     adv: "Outer wheel surplus → rotation" },
          { label: "Exit throttle",  simple: "Yaw collapses under power",        adv: "PI clamps yaw error instantly" },
          { label: "Yaw moment",     simple: "0 Nm (passive)",                   adv: "Up to ±50 Nm active" },
        ].map(row => (
          <div key={row.label} style={{ ...GL, padding: "8px 12px" }}>
            <div style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, marginBottom: 5, fontWeight: 700 }}>{row.label}</div>
            <div style={{ fontSize: 8, color: C.red, fontFamily: C.dt, marginBottom: 3 }}>⊗ {row.simple}</div>
            <div style={{ fontSize: 8, color: C.cy, fontFamily: C.dt }}>◎ {row.adv}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Main module export
// ─────────────────────────────────────────────────────────────────────────────
export default function TorqueVectoringSimModule() {
  const canvasRef = useRef(null);

  // Physics state in a ref — avoids React re-render overhead inside the rAF loop
  const stateRef  = useRef({ ...S0 });

  // Dual-source control: keyboard (auto-centring) + slider (sticky)
  const kbRef     = useRef({ steer: 0, throttle: 0, brake: 0 });
  const slRef     = useRef({ steer: 0, throttle: 0 });

  const keysRef   = useRef(new Set());
  const modeRef   = useRef("simple");
  const trailRef  = useRef([]);  // stores [canvasX, canvasY] pairs (pre-transformed)
  const rafRef    = useRef(null);
  const lastTRef  = useRef(null);

  // React state — updated at ~15 fps for UI, does not drive physics
  const [mode, setMode]       = useState("simple");
  const [slSteer, setSlSteer] = useState(0);     // normalised [-1, 1]
  const [slThr,   setSlThr]   = useState(0);     // normalised [ 0, 1]
  const [disp, setDisp]       = useState({
    speed: 0, wzRef: 0, wzAct: 0, wzErr: 0, Mtv: 0, Trl: 0, Trr: 0,
  });

  // Keep mode ref in sync without stale closure inside rAF
  useEffect(() => { modeRef.current = mode; }, [mode]);

  // Global keyboard listeners (attached once, independent of mode)
  useEffect(() => {
    const dn = e => { keysRef.current.add(e.key); };
    const up = e => { keysRef.current.delete(e.key); };
    window.addEventListener("keydown", dn);
    window.addEventListener("keyup",   up);
    return () => {
      window.removeEventListener("keydown", dn);
      window.removeEventListener("keyup",   up);
    };
  }, []);

  const reset = useCallback(() => {
    stateRef.current = { ...S0 };
    kbRef.current    = { steer: 0, throttle: 0, brake: 0 };
    slRef.current    = { steer: 0, throttle: 0 };
    trailRef.current = [];
    setSlSteer(0);
    setSlThr(0);
    setDisp({ speed: 0, wzRef: 0, wzAct: 0, wzErr: 0, Mtv: 0, Trl: 0, Trr: 0 });
  }, []);

  // ── rAF game loop ─────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // DPR scaling for sharp rendering on retina displays
    const dpr  = window.devicePixelRatio || 1;
    const logW = canvas.clientWidth  || 660;
    const logH = canvas.clientHeight || 370;
    canvas.width  = logW * dpr;
    canvas.height = logH * dpr;
    ctx.scale(dpr, dpr);

    let tick = 0;

    const loop = (ts) => {
      if (!lastTRef.current) lastTRef.current = ts;
      const rawDt = Math.min((ts - lastTRef.current) / 1000, 0.035);  // cap at ~28fps equiv
      lastTRef.current = ts;

      const keys = keysRef.current;
      const kb   = kbRef.current;
      const sl   = slRef.current;

      // ── Keyboard contributions (independent auto-centre on release) ──────
      const steerLeft  = keys.has("ArrowLeft")  || keys.has("a");
      const steerRight = keys.has("ArrowRight") || keys.has("d");
      if      (steerLeft)  kb.steer = clamp(kb.steer + 2.2 * rawDt, -V.dMax, V.dMax);
      else if (steerRight) kb.steer = clamp(kb.steer - 2.2 * rawDt, -V.dMax, V.dMax);
      else                 kb.steer *= (1 - 9 * rawDt);   // fast return to 0

      if (keys.has("ArrowUp") || keys.has("w"))
        kb.throttle = clamp(kb.throttle + 2.5 * rawDt, 0, 1);
      else
        kb.throttle = Math.max(0, kb.throttle - 3.5 * rawDt);

      if (keys.has("ArrowDown") || keys.has("s"))
        kb.brake = clamp(kb.brake + 3.5 * rawDt, 0, 1);
      else
        kb.brake = Math.max(0, kb.brake - 5.0 * rawDt);

      // ── Merge sources (additive steer, max-select throttle) ─────────────
      const steer    = clamp(kb.steer + sl.steer * V.dMax, -V.dMax, V.dMax);
      const throttle = Math.min(1, Math.max(kb.throttle, sl.throttle));
      const brake    = kb.brake;

      // ── 4× substep Euler ─────────────────────────────────────────────────
      let s  = stateRef.current;
      const dt = rawDt / 4;
      for (let i = 0; i < 4; i++)
        s = physicsStep(s, steer, throttle, brake, modeRef.current, dt);
      stateRef.current = s;

      // Store trail as canvas coordinates (avoids repeated w2c in draw)
      const cw = logW, ch = logH;
      const [tx, ty] = w2c(s.x, s.y, cw, ch);
      trailRef.current.push([tx, ty]);
      if (trailRef.current.length > 400) trailRef.current.shift();

      // ── Render ────────────────────────────────────────────────────────────
      ctx.clearRect(0, 0, cw, ch);
      drawTrack(ctx, cw, ch);
      drawTrail(ctx, trailRef.current, modeRef.current);
      drawCar(ctx, cw, ch, s, modeRef.current);
      renderHUD(ctx, s, modeRef.current, steer);

      // ── React state sync at ≈15 fps ───────────────────────────────────────
      if (++tick % 4 === 0) {
        setDisp({
          speed: Math.hypot(s.vx, s.vy) * 3.6,
          wzRef: s.wzRef  ?? 0,
          wzAct: s.r,
          wzErr: s.wzErr  ?? 0,
          Mtv:   s.Mtv    ?? 0,
          Trl:   s.Trl    ?? 0,
          Trr:   s.Trr    ?? 0,
        });
        setSlSteer(kb.steer / V.dMax);  // keep slider visually in sync with keyboard
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(rafRef.current);
      lastTRef.current = null;
    };
  }, []);  // intentionally empty — loop reads everything through refs

  // ── Slider handlers ────────────────────────────────────────────────────────
  const onSlSteer = useCallback(e => {
    const v = parseFloat(e.target.value);
    slRef.current.steer = v;
    setSlSteer(v);
  }, []);

  const onSlThr = useCallback(e => {
    const v = parseFloat(e.target.value);
    slRef.current.throttle = v;
    setSlThr(v);
  }, []);

  const onModeChange = useCallback((m) => {
    setMode(m);
    // Defer reset by one tick so modeRef updates first
    setTimeout(reset, 0);
  }, [reset]);

  const ac = mode === "advanced" ? C.cy : C.red;

  return (
    <div>
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{
        ...GL, padding: "12px 18px", marginBottom: 14,
        borderLeft: `3px solid ${ac}`,
        background: `linear-gradient(90deg, ${ac}10, transparent)`,
        transition: "all 0.4s",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 20, color: ac }}>◎</span>
            <div>
              <span style={{ fontSize: 12, fontWeight: 800, color: ac, fontFamily: C.dt, letterSpacing: 2 }}>
                TORQUE VECTORING SIMULATOR
              </span>
              <span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 12 }}>
                Ter26 RWD · Bicycle model · PI yaw-moment controller
              </span>
            </div>
          </div>
          <button
            onClick={reset}
            style={{
              ...GL, border: `1px solid ${C.b2}`, borderRadius: 8,
              padding: "5px 16px", color: C.md, fontFamily: C.dt,
              fontSize: 9, cursor: "pointer", letterSpacing: 1.5, fontWeight: 700,
            }}
          >
            ↺ RESET
          </button>
        </div>
      </div>

      {/* ── Mode selector ──────────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14, alignItems: "center", flexWrap: "wrap" }}>
        <Pill active={mode === "simple"}   label="⊗  Simple TV"   onClick={() => onModeChange("simple")}   color={C.red} />
        <Pill active={mode === "advanced"} label="◎  Advanced TV"  onClick={() => onModeChange("advanced")} color={C.cy}  />
        <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, marginLeft: 8 }}>
          WASD / Arrow Keys  ·  or use sliders below
        </span>
      </div>

      {/* ── Simulation canvas + torque bars ────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 148px", gap: 10, marginBottom: 10 }}>

        {/* Canvas */}
        <div style={{
          ...GL, padding: 0, overflow: "hidden",
          border: `1px solid ${ac}28`,
          transition: "border-color 0.4s",
          position: "relative",
        }}>
          <canvas
            ref={canvasRef}
            style={{ display: "block", width: "100%", height: "auto", aspectRatio: "660/370" }}
          />
          {/* Overlay legend */}
          <div style={{
            position: "absolute", bottom: 10, right: 12,
            display: "flex", flexDirection: "column", gap: 4,
          }}>
            {[
              { c: "rgba(0,200,0,0.8)",   l: "Low torque"  },
              { c: "rgba(255,200,0,0.8)", l: "Mid torque"  },
              { c: "rgba(255,80,0,0.85)", l: "Peak torque" },
            ].map(e => (
              <div key={e.l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: e.c }} />
                <span style={{ fontSize: 7.5, color: "rgba(150,165,190,0.7)", fontFamily: C.dt }}>{e.l}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Per-wheel torque panel */}
        <div style={{ ...GL, padding: "14px 10px", display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontSize: 7.5, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 2, textAlign: "center" }}>
            PER-WHEEL
          </div>

          {/* 2×2 torque bars */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, flex: 1 }}>
            <TorqueBar label="FL" value={0}       max={V.Tp} color={C.dm} driven={false} />
            <TorqueBar label="FR" value={0}       max={V.Tp} color={C.dm} driven={false} />
            <TorqueBar label="RL" value={disp.Trl} max={V.Tp} color={ac}   driven={true}  />
            <TorqueBar label="RR" value={disp.Trr} max={V.Tp} color={ac}   driven={true}  />
          </div>

          {/* Yaw moment readout */}
          <div style={{ ...GL, padding: "8px 10px", textAlign: "center" }}>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, marginBottom: 4, letterSpacing: 1.5 }}>M_TV [Nm]</div>
            <div style={{
              fontSize: 16, fontWeight: 800, fontFamily: C.hd,
              color: Math.abs(disp.Mtv) > 20 ? ac : C.dm,
              transition: "color 0.2s",
            }}>
              {disp.Mtv.toFixed(0)}
            </div>
          </div>

          {/* ΔT differential */}
          <div style={{ ...GL, padding: "6px 10px", textAlign: "center" }}>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, marginBottom: 2, letterSpacing: 1.5 }}>ΔT [Nm]</div>
            <div style={{
              fontSize: 13, fontWeight: 700, fontFamily: C.hd,
              color: Math.abs(disp.Trr - disp.Trl) > 10 ? ac : C.dm,
            }}>
              {(disp.Trr - disp.Trl).toFixed(0)}
            </div>
          </div>
        </div>
      </div>

      {/* ── Input sliders ──────────────────────────────────────────────────── */}
      <div style={{ ...GL, padding: "12px 18px", marginBottom: 10 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22 }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>STEERING ← →</span>
              <span style={{ fontSize: 8, color: C.br, fontFamily: C.dt, fontWeight: 700 }}>
                {(slSteer * V.dMax * 180 / Math.PI).toFixed(1)}°
              </span>
            </div>
            <input
              type="range" min={-1} max={1} step={0.01} value={slSteer}
              onChange={onSlSteer}
              style={{ width: "100%", accentColor: ac, cursor: "pointer", height: 4 }}
            />
          </div>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>THROTTLE ↑</span>
              <span style={{ fontSize: 8, color: C.br, fontFamily: C.dt, fontWeight: 700 }}>
                {(slThr * 100).toFixed(0)} %
              </span>
            </div>
            <input
              type="range" min={0} max={1} step={0.01} value={slThr}
              onChange={onSlThr}
              style={{ width: "100%", accentColor: C.gn, cursor: "pointer", height: 4 }}
            />
          </div>
        </div>
      </div>

      {/* ── KPI row ────────────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI
          label="Speed"
          value={`${disp.speed.toFixed(1)} km/h`}
          sub="body speed"
          sentiment={disp.speed > 10 ? "positive" : "neutral"}
          delay={0}
        />
        <KPI
          label="ψ̇  ref"
          value={`${disp.wzRef.toFixed(3)} r/s`}
          sub="Ackermann target"
          sentiment="neutral"
          delay={1}
        />
        <KPI
          label="ψ̇  actual"
          value={`${disp.wzAct.toFixed(3)} r/s`}
          sub="measured yaw rate"
          sentiment={Math.abs(disp.wzErr) < 0.06 ? "positive" : "amber"}
          delay={2}
        />
        <KPI
          label="Yaw Error"
          value={`${(disp.wzErr * 1000).toFixed(1)} mr/s`}
          sub={mode === "advanced" ? "PI corrected" : "open-loop"}
          sentiment={
            Math.abs(disp.wzErr) < 0.06  ? "positive" :
            mode === "advanced"           ? "amber"    : "negative"
          }
          delay={3}
        />
      </div>

      {/* ── Theory panel ───────────────────────────────────────────────────── */}
      <TheoryBlock mode={mode} />
    </div>
  );
}