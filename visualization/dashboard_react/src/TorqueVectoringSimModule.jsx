// ═══════════════════════════════════════════════════════════════════════════
// src/TorqueVectoringSimModule.jsx  —  Project-GP Dashboard v5.1 rev-2
// ═══════════════════════════════════════════════════════════════════════════
// Changelog vs rev-1:
//   • V.Tp  120 → 540 Nm  (wheel torque = motor × 4.5 gear ratio)
//   • S0.vx 9   → 14 m/s  (realistic FS autocross entry speed)
//   • Lap timer: finish-line crossing detection, best lap, lap history
//   • Checkered finish line rendered on canvas
//   • Anti-windup and thrust-preservation patches preserved from rev-1
// ═══════════════════════════════════════════════════════════════════════════

import React, { useRef, useEffect, useState, useCallback } from "react";
import { C, GL } from "./theme.js";
import { KPI, Pill } from "./components.jsx";

// ─────────────────────────────────────────────────────────────────────────────
// §1  Vehicle parameters
// ─────────────────────────────────────────────────────────────────────────────
const V = {
  m:    280,      // kg
  Iz:   120,      // kg·m²  yaw inertia
  L:    1.53,     // m   wheelbase
  lf:   0.76,     // m   CG → front axle
  lr:   0.77,     // m   CG → rear axle
  tw:   1.20,     // m   rear track width
  rw:   0.254,    // m   wheel radius
  Cf:   28000,    // N/rad  front cornering stiffness
  Cr:   32000,    // N/rad  rear  cornering stiffness
  // FIXED: wheel torque = motor torque × gear ratio (120 Nm × 4.5)
  Tp:   540,      // Nm per rear wheel at contact patch
  Cd:   1.1,
  Af:   1.0,      // m²
  rho:  1.225,    // kg/m³
  dMax: 0.38,     // rad max steer
  Kus:  0.006,    // rad·s²/m²
};

const KP_TV = 1500;
const KI_TV = 14;

// ─────────────────────────────────────────────────────────────────────────────
// §2  Track geometry
// ─────────────────────────────────────────────────────────────────────────────
const OW  = 50;
const OH  = 22;
const THW = 5.0;
const SCALE = 5.0;

// Finish line: left vertex of ellipse (x = −OW, y = 0).
// Lap counted when car transitions from y < −FL_Y_ARM to y >= −FL_Y_ARM
// while x is within FL_X_TOL of −OW.
const FL_X_TOL = THW * 1.8;
const FL_Y_ARM = 1.5;

const S0 = Object.freeze({
  x:   -(OW - THW * 0.5),
  y:   0,
  psi: Math.PI / 2,
  vx:  14,      // FIXED: was 9 m/s
  vy:  0,
  r:   0,
  wz_int: 0,
});

// ─────────────────────────────────────────────────────────────────────────────
// §3  Physics — single-track bicycle model, 4× substep Euler
// ─────────────────────────────────────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function physicsStep(s, steer, throttle, brake, mode, dt) {
  const vx = Math.max(0.5, s.vx);
  const v  = Math.hypot(vx, s.vy);

  const wzRef  = (v * steer) / (V.L * (1 + V.Kus * v * v));
  const alphaF =  steer - (s.vy + V.lf * s.r) / vx;
  const alphaR  = -(s.vy - V.lr * s.r) / vx;

  const FyF = clamp(V.Cf * alphaF, -6500, 6500);
  const FyR  = clamp(V.Cr * alphaR, -7000, 7000);

  const Tdrive = throttle * V.Tp * 2;
  const Fbrake = brake * 6000;

  let Trl, Trr, wz_int = s.wz_int;

  if (mode === "simple") {
    // Steer-proportional feedforward: outer wheel bias proportional to δ
    const dT_s = clamp(steer * Tdrive * 0.30, -V.Tp * 0.42, V.Tp * 0.42);
    Trl = clamp(Tdrive / 2 - dT_s, 0, V.Tp);
    Trr = clamp(Tdrive / 2 + dT_s, 0, V.Tp);
  } else {
    // PI yaw-moment feedback + anti-windup + thrust preservation
    const err       = wzRef - s.r;
    const decayRate = 1 - 14 * dt * Math.max(0, 1 - Math.abs(err) / 0.08);
    wz_int = clamp(s.wz_int * decayRate + err * dt, -3.5, 3.5);
    const dM = KP_TV * err + KI_TV * wz_int;
    const dT = clamp(dM / (V.tw / V.rw), -V.Tp * 0.55, V.Tp * 0.55);

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
  const Mtv   = ((Trr - Trl) / V.rw) * (V.tw / 2);
  const Fdrag = 0.5 * V.rho * V.Cd * V.Af * v * v;

  const ax = (Fx - Fdrag - FyF * Math.sin(steer)) / V.m + s.vy * s.r;
  const ay = (FyF * Math.cos(steer) + FyR)         / V.m - vx   * s.r;
  const az = (FyF * Math.cos(steer) * V.lf - FyR * V.lr + Mtv) / V.Iz;

  const sinP = Math.sin(s.psi), cosP = Math.cos(s.psi);
  return {
    x:   s.x   + (vx * cosP - s.vy * sinP) * dt,
    y:   s.y   + (vx * sinP + s.vy * cosP) * dt,
    psi: s.psi + s.r * dt,
    vx:  Math.max(0, s.vx + ax * dt),
    vy:  s.vy + ay * dt,
    r:   s.r  + az * dt,
    wz_int,
    Trl, Trr, wzRef, wzErr: wzRef - s.r, Mtv,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Canvas rendering
// ─────────────────────────────────────────────────────────────────────────────
function w2c(wx, wy, cw, ch) {
  return [cw / 2 + wx * SCALE, ch / 2 - wy * SCALE];
}

function drawTrack(ctx, cw, ch) {
  const cx0 = cw / 2, cy0 = ch / 2;

  ctx.fillStyle = "#04060c";
  ctx.fillRect(0, 0, cw, ch);

  ctx.strokeStyle = "rgba(18,30,55,0.40)";
  ctx.lineWidth   = 0.5;
  const step = 10 * SCALE;
  for (let gx = cx0 % step; gx < cw; gx += step) {
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, ch); ctx.stroke();
  }
  for (let gy = cy0 % step; gy < ch; gy += step) {
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cw, gy); ctx.stroke();
  }

  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW + THW) * SCALE, (OH + THW) * SCALE, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#0d1520";
  ctx.fill();

  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW - THW) * SCALE, (OH - THW) * SCALE, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#060d08";
  ctx.fill();

  ctx.beginPath();
  ctx.ellipse(cx0, cy0, (OW - THW - 2) * SCALE, (OH - THW - 2) * SCALE, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#050b07";
  ctx.fill();

  [[OW + THW, 1.5], [OW - THW, 1.5]].forEach(([r, lw]) => {
    ctx.beginPath();
    ctx.ellipse(cx0, cy0, r * SCALE, (r === OW + THW ? OH + THW : OH - THW) * SCALE, 0, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(200,220,255,0.13)";
    ctx.lineWidth   = lw;
    ctx.stroke();
  });

  ctx.beginPath();
  ctx.ellipse(cx0, cy0, OW * SCALE, OH * SCALE, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.055)";
  ctx.lineWidth   = 1;
  ctx.setLineDash([10, 14]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawFinishLine(ctx, cw, ch) {
  const [flx, flyTop] = w2c(-OW, +THW, cw, ch);
  const [,    flyBot] = w2c(-OW, -THW, cw, ch);
  const lineH = flyBot - flyTop;
  const tileH = lineH / 8;
  const tileW = 6;

  for (let i = 0; i < 8; i++) {
    for (let col = 0; col < 2; col++) {
      ctx.fillStyle = (i + col) % 2 === 0
        ? "rgba(255,255,255,0.52)"
        : "rgba(0,0,0,0.42)";
      ctx.fillRect(flx - tileW + col * tileW, flyTop + i * tileH, tileW, tileH);
    }
  }

  ctx.beginPath();
  ctx.moveTo(flx, flyTop - 2);
  ctx.lineTo(flx, flyBot + 2);
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.lineWidth   = 1;
  ctx.stroke();
}

function drawTrail(ctx, trail, mode) {
  if (trail.length < 2) return;
  ctx.beginPath();
  trail.forEach(([tx, ty], i) => (i === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty)));
  ctx.strokeStyle = mode === "advanced" ? "rgba(0,210,255,0.52)" : "rgba(225,6,0,0.52)";
  ctx.lineWidth   = 2.5;
  ctx.lineJoin    = "round";
  ctx.stroke();
}

function drawCar(ctx, cw, ch, state, mode) {
  const [px, py] = w2c(state.x, state.y, cw, ch);
  const Trl = state.Trl ?? 0;
  const Trr = state.Trr ?? 0;

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(Math.PI / 2 - state.psi);

  const BL = 14, BW = 6;

  ctx.beginPath();
  ctx.rect(-BW, -BL, BW * 2, BL * 2);
  ctx.fillStyle = mode === "advanced" ? "rgba(4,20,38,0.93)" : "rgba(28,6,6,0.93)";
  ctx.fill();
  ctx.strokeStyle = mode === "advanced" ? C.cy : C.red;
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, -(BL + 6));
  ctx.lineTo(-BW * 0.55, -BL + 2);
  ctx.lineTo( BW * 0.55, -BL + 2);
  ctx.closePath();
  ctx.fillStyle = mode === "advanced" ? C.cy : C.red;
  ctx.fill();

  ctx.beginPath();
  ctx.rect(-(BW + 3), BL - 4, (BW + 3) * 2, 2.5);
  ctx.fillStyle = mode === "advanced" ? C.cy : C.red;
  ctx.fill();

  ctx.beginPath();
  ctx.ellipse(0, 0, BW * 0.55, BL * 0.35, 0, 0, Math.PI * 2);
  ctx.fillStyle   = "rgba(0,0,0,0.50)";
  ctx.fill();
  ctx.strokeStyle = "rgba(150,180,220,0.18)";
  ctx.lineWidth   = 1;
  ctx.stroke();

  const wheels = [
    { x: -(BW + 3.5), y: -BL * 0.50, T: 0,   driven: false },
    { x:  (BW + 3.5), y: -BL * 0.50, T: 0,   driven: false },
    { x: -(BW + 3.5), y:  BL * 0.50, T: Trl, driven: true  },
    { x:  (BW + 3.5), y:  BL * 0.50, T: Trr, driven: true  },
  ];

  wheels.forEach(w => {
    const ratio = w.driven ? Math.min(1, w.T / V.Tp) : 0;
    let borderColor;
    if (!w.driven || ratio < 0.05) {
      borderColor = "rgba(80,100,130,0.55)";
    } else if (ratio < 0.40) {
      const t = ratio / 0.40;
      borderColor = `rgba(${Math.round(t * 255)},230,0,${0.55 + t * 0.35})`;
    } else if (ratio < 0.75) {
      const t = (ratio - 0.40) / 0.35;
      borderColor = `rgba(255,${Math.round((1 - t) * 180)},0,0.90)`;
    } else {
      borderColor = `rgba(255,${Math.round((1 - (ratio - 0.75) / 0.25) * 80)},0,1.0)`;
    }

    ctx.beginPath();
    ctx.rect(w.x - 3, w.y - 5, 6, 10);
    ctx.fillStyle   = "rgba(20,25,35,0.92)";
    ctx.fill();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth   = 1.8;
    ctx.stroke();

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
  const spd       = Math.hypot(state.vx, state.vy) * 3.6;
  const modeColor = mode === "advanced" ? "#00d2ff" : "#e10600";

  ctx.font      = "bold 11px 'Azeret Mono', monospace";
  ctx.fillStyle = "rgba(192,200,218,0.85)";
  ctx.fillText(`${spd.toFixed(0)} km/h`, 10, 20);

  ctx.font      = "bold 9px 'Azeret Mono', monospace";
  ctx.fillStyle = modeColor;
  ctx.fillText(mode === "advanced" ? "◎ ADVANCED TV" : "⊗ SIMPLE TV", 10, 34);

  const arcX = 28, arcY = 72, arcR = 18;
  ctx.beginPath();
  ctx.arc(arcX, arcY, arcR, Math.PI, 2 * Math.PI, false);
  ctx.strokeStyle = "rgba(70,90,130,0.40)";
  ctx.lineWidth   = 2.5;
  ctx.stroke();

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

  ctx.font      = "8px 'Azeret Mono', monospace";
  ctx.fillStyle = "rgba(100,120,160,0.80)";
  ctx.fillText("STEER", arcX - 13, arcY + 11);
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Lap timer helpers
// ─────────────────────────────────────────────────────────────────────────────
function fmtLap(ms) {
  if (!ms || ms <= 0) return "--:--.---";
  const m   = Math.floor(ms / 60000);
  const s   = Math.floor((ms % 60000) / 1000);
  const ms3 = Math.floor(ms % 1000);
  return `${m}:${String(s).padStart(2, "0")}.${String(ms3).padStart(3, "0")}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Sub-components
// ─────────────────────────────────────────────────────────────────────────────
function TorqueBar({ label, value, max, color, driven }) {
  const pct = driven ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
      <div style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, fontWeight: 700 }}>
        {label}
      </div>
      <div style={{
        width: 28, height: 68, background: "rgba(20,30,50,0.6)", borderRadius: 4,
        position: "relative", overflow: "hidden", border: "1px solid rgba(50,68,105,0.5)",
      }}>
        <div style={{
          position: "absolute", bottom: 0, width: "100%", height: `${pct}%`,
          background: driven ? `linear-gradient(to top, ${color}, ${color}88)` : "rgba(50,65,95,0.4)",
          transition: "height 0.05s linear",
        }} />
        <div style={{ position: "absolute", bottom: "80%", width: "100%", height: 1, background: "rgba(255,255,255,0.12)" }} />
      </div>
      <div style={{ fontSize: 8, fontWeight: 700, color: driven ? color : C.dm, fontFamily: C.dt }}>
        {driven ? value.toFixed(0) : "—"}
        {driven && <span style={{ fontSize: 6.5, opacity: 0.6 }}> Nm</span>}
      </div>
    </div>
  );
}

function LapTimer({ current, best, history }) {
  const isNewBest = best > 0 && history.length >= 1 && history[0] === best;

  return (
    <div style={{ ...GL, padding: "12px 14px", flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 7.5, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 2 }}>
        LAP TIMER
      </div>

      {/* Current lap */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, letterSpacing: 1 }}>CURRENT LAP</span>
        <span style={{ fontSize: 20, fontWeight: 800, color: C.w, fontFamily: C.hd, letterSpacing: 1, lineHeight: 1.1 }}>
          {fmtLap(current)}
        </span>
      </div>

      {/* Best lap */}
      <div style={{
        padding: "7px 10px", borderRadius: 6,
        background: best > 0 ? `${C.gn}12` : "rgba(20,30,50,0.5)",
        border: `1px solid ${best > 0 ? C.gn + "30" : "rgba(50,68,105,0.4)"}`,
        transition: "all 0.4s",
      }}>
        <div style={{ fontSize: 7, color: C.gn, fontFamily: C.dt, letterSpacing: 1, marginBottom: 3 }}>
          {isNewBest ? "★ NEW BEST" : "BEST LAP"}
        </div>
        <div style={{ fontSize: 15, fontWeight: 800, color: best > 0 ? C.gn : C.dm, fontFamily: C.hd }}>
          {fmtLap(best)}
        </div>
      </div>

      {/* History */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
        {history.slice(0, 5).map((t, i) => (
          <div key={i} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "3px 6px", borderRadius: 4,
            background: t === best ? `${C.gn}09` : "transparent",
          }}>
            <span style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt }}>
              LAP {history.length - i}
            </span>
            <span style={{ fontSize: 10, fontWeight: 700, fontFamily: C.hd, color: t === best ? C.gn : C.br }}>
              {fmtLap(t)}
            </span>
          </div>
        ))}
        {history.length === 0 && (
          <div style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, textAlign: "center", paddingTop: 6, lineHeight: 1.7 }}>
            Cross the checkered<br />line to start timing
          </div>
        )}
      </div>
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
        {isAdv
          ? "ADVANCED TV — SOCP-INSPIRED PI YAW-MOMENT CONTROLLER"
          : "SIMPLE TV — STEER-PROPORTIONAL FEEDFORWARD"}
      </div>

      {!isAdv ? (
        <>
          <p style={{ fontSize: 9, color: C.br, fontFamily: C.dt, lineHeight: 1.85, margin: 0 }}>
            Open-loop: outer-wheel torque is biased by <F c={C.red}>ΔT_s ∝ δ × T_total</F>. No yaw
            feedback — the allocation is purely geometric. Under moderate cornering the outer wheel
            loads up correctly, but any mismatch between the assumed and actual tyre state produces
            uncorrected yaw error. On corner exit with heavy throttle the static gain
            over-drives the outer wheel, introducing <F c={C.am}>power-induced oversteer</F> that
            cascades without any corrective moment.
          </p>
          <pre style={{ fontFamily: "monospace", fontSize: 8, color: C.dm, marginTop: 12, padding: "10px 14px", background: "rgba(5,8,15,0.85)", borderRadius: 6, lineHeight: 2.1 }}>
{`ΔT_s   =  clamp( δ × T_total × 0.30,  ±0.42·T_peak )
T_rl   =  T/2 − ΔT_s      (inner — reduced)
T_rr   =  T/2 + ΔT_s      (outer — boosted)
M_yaw  =  open-loop, no closed-loop correction`}
          </pre>
        </>
      ) : (
        <>
          <p style={{ fontSize: 9, color: C.br, fontFamily: C.dt, lineHeight: 1.85, margin: 0 }}>
            Closed-loop PI on yaw-rate error. Reference: <F c={C.cy}>ψ̇_ref = v·δ / (L·(1 + K_us·v²))</F>.
            Error <F c={C.gn}>ε = ψ̇_ref − ψ̇_actual</F> drives the differential moment
            <F c={C.am}> ΔM = Kp·ε + Ki·∫ε</F>. Anti-windup decay bleeds the integral
            when |ε| &lt; 0.08 rad/s (straight/exit), eliminating corner-exit torque suppression.
            SOCP-style thrust preservation redistributes any floor-clamped deficit — total Fx
            is invariant to the TV overlay.
          </p>
          <pre style={{ fontFamily: "monospace", fontSize: 8, color: C.dm, marginTop: 12, padding: "10px 14px", background: "rgba(5,8,15,0.85)", borderRadius: 6, lineHeight: 2.1 }}>
{`ψ̇_ref  =  v · δ / (L · (1 + 0.006 · v²))
ε       =  ψ̇_ref − ψ̇_actual
ΔT      =  clamp( (1500·ε + 14·∫ε) · rw/tw,  ±0.55·T_peak )
T_rr    =  T/2 + ΔT  │  T_rl = T/2 − ΔT
deficit → redistributed equally → Fx preserved`}
          </pre>
        </>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 14 }}>
        {[
          { label: "Corner entry",  simple: "Understeer if δ-gain mismatch",    adv: "ψ̇ closed-loop — follows apex"   },
          { label: "Mid-corner",    simple: "Static gain, no tyre state update", adv: "PI corrects slip in real time"   },
          { label: "Exit throttle", simple: "Oversteer from static outer bias",  adv: "Integral decays fast, full Fx"  },
          { label: "Yaw moment",    simple: "Feed-forward only",                 adv: "Active ±50 Nm correction"       },
        ].map(row => (
          <div key={row.label} style={{ ...GL, padding: "8px 12px" }}>
            <div style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, marginBottom: 5, fontWeight: 700 }}>
              {row.label}
            </div>
            <div style={{ fontSize: 8, color: C.red, fontFamily: C.dt, marginBottom: 3 }}>⊗ {row.simple}</div>
            <div style={{ fontSize: 8, color: C.cy,  fontFamily: C.dt }}>◎ {row.adv}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Main export
// ─────────────────────────────────────────────────────────────────────────────
export default function TorqueVectoringSimModule() {
  const canvasRef = useRef(null);
  const stateRef  = useRef({ ...S0 });
  const kbRef     = useRef({ steer: 0, throttle: 0, brake: 0 });
  const slRef     = useRef({ steer: 0, throttle: 0 });
  const keysRef   = useRef(new Set());
  const modeRef   = useRef("simple");
  const trailRef  = useRef([]);
  const rafRef    = useRef(null);
  const lastTRef  = useRef(null);

  // Lap timing — all in refs, zero React involvement inside rAF
  const prevYRef       = useRef(S0.y);
  const lapStartRef    = useRef(null);
  const bestLapRef     = useRef(0);
  const lapHistoryRef  = useRef([]);    // [ms] newest first

  const [mode, setMode]       = useState("simple");
  const [slSteer, setSlSteer] = useState(0);
  const [slThr,   setSlThr]   = useState(0);
  const [disp, setDisp]       = useState({ speed: 0, wzRef: 0, wzAct: 0, wzErr: 0, Mtv: 0, Trl: 0, Trr: 0 });
  const [lapDisp, setLapDisp] = useState({ current: 0, best: 0, history: [] });

  useEffect(() => { modeRef.current = mode; }, [mode]);

  useEffect(() => {
    const dn = e => keysRef.current.add(e.key);
    const up = e => keysRef.current.delete(e.key);
    window.addEventListener("keydown", dn);
    window.addEventListener("keyup",   up);
    return () => {
      window.removeEventListener("keydown", dn);
      window.removeEventListener("keyup",   up);
    };
  }, []);

  const reset = useCallback(() => {
    stateRef.current     = { ...S0 };
    kbRef.current        = { steer: 0, throttle: 0, brake: 0 };
    slRef.current        = { steer: 0, throttle: 0 };
    trailRef.current     = [];
    prevYRef.current     = S0.y;
    lapStartRef.current  = null;
    bestLapRef.current   = 0;
    lapHistoryRef.current = [];
    setSlSteer(0); setSlThr(0);
    setDisp({ speed: 0, wzRef: 0, wzAct: 0, wzErr: 0, Mtv: 0, Trl: 0, Trr: 0 });
    setLapDisp({ current: 0, best: 0, history: [] });
  }, []);

  // ── rAF loop ────────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx  = canvas.getContext("2d");
    const dpr  = window.devicePixelRatio || 1;
    const logW = canvas.clientWidth  || 680;
    const logH = canvas.clientHeight || 380;
    canvas.width  = logW * dpr;
    canvas.height = logH * dpr;
    ctx.scale(dpr, dpr);

    let tick = 0;

    const loop = (ts) => {
      if (!lastTRef.current) lastTRef.current = ts;
      const rawDt = Math.min((ts - lastTRef.current) / 1000, 0.035);
      lastTRef.current = ts;

      const keys = keysRef.current;
      const kb   = kbRef.current;
      const sl   = slRef.current;

      // Keyboard
      const steerLeft  = keys.has("ArrowLeft")  || keys.has("a");
      const steerRight = keys.has("ArrowRight") || keys.has("d");
      if      (steerLeft)  kb.steer = clamp(kb.steer + 2.2 * rawDt, -V.dMax, V.dMax);
      else if (steerRight) kb.steer = clamp(kb.steer - 2.2 * rawDt, -V.dMax, V.dMax);
      else                 kb.steer *= (1 - 9 * rawDt);

      if (keys.has("ArrowUp") || keys.has("w"))
        kb.throttle = clamp(kb.throttle + 2.5 * rawDt, 0, 1);
      else
        kb.throttle = Math.max(0, kb.throttle - 3.5 * rawDt);

      if (keys.has("ArrowDown") || keys.has("s"))
        kb.brake = clamp(kb.brake + 3.5 * rawDt, 0, 1);
      else
        kb.brake = Math.max(0, kb.brake - 5.0 * rawDt);

      const steer    = clamp(kb.steer + sl.steer * V.dMax, -V.dMax, V.dMax);
      const throttle = Math.min(1, Math.max(kb.throttle, sl.throttle));
      const brake    = kb.brake;

      // 4× substep
      let s  = stateRef.current;
      const dt = rawDt / 4;
      for (let i = 0; i < 4; i++)
        s = physicsStep(s, steer, throttle, brake, modeRef.current, dt);
      stateRef.current = s;

      // ── Lap crossing detection ───────────────────────────────────────────
      // Car starts at y=0 heading north → goes to positive y first.
      // Finish line crossing: approaching from south (prevY < −FL_Y_ARM)
      // crossing to y ≥ −FL_Y_ARM while near x = −OW.
      const prevY = prevYRef.current;
      const currY = s.y;
      if (
        Math.abs(s.x - (-OW)) < FL_X_TOL &&
        prevY < -FL_Y_ARM &&
        currY >= -FL_Y_ARM
      ) {
        if (lapStartRef.current !== null) {
          const lapMs = ts - lapStartRef.current;
          if (lapMs > 5000) {   // reject shortcuts / resets
            lapHistoryRef.current = [lapMs, ...lapHistoryRef.current].slice(0, 10);
            if (bestLapRef.current === 0 || lapMs < bestLapRef.current)
              bestLapRef.current = lapMs;
          }
        }
        lapStartRef.current = ts;
      }
      prevYRef.current = currY;

      // Trail
      const cw = logW, ch = logH;
      const [tx, ty] = w2c(s.x, s.y, cw, ch);
      trailRef.current.push([tx, ty]);
      if (trailRef.current.length > 500) trailRef.current.shift();

      // Render
      ctx.clearRect(0, 0, cw, ch);
      drawTrack(ctx, cw, ch);
      drawFinishLine(ctx, cw, ch);
      drawTrail(ctx, trailRef.current, modeRef.current);
      drawCar(ctx, cw, ch, s, modeRef.current);
      renderHUD(ctx, s, modeRef.current, steer);

      // React sync ~15 fps
      if (++tick % 4 === 0) {
        const currentLapMs = lapStartRef.current !== null ? (ts - lapStartRef.current) : 0;
        setDisp({
          speed: Math.hypot(s.vx, s.vy) * 3.6,
          wzRef: s.wzRef ?? 0,
          wzAct: s.r,
          wzErr: s.wzErr ?? 0,
          Mtv:   s.Mtv   ?? 0,
          Trl:   s.Trl   ?? 0,
          Trr:   s.Trr   ?? 0,
        });
        setLapDisp({
          current: currentLapMs,
          best:    bestLapRef.current,
          history: lapHistoryRef.current,
        });
        setSlSteer(kb.steer / V.dMax);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(rafRef.current); lastTRef.current = null; };
  }, []);

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
    setTimeout(reset, 0);
  }, [reset]);

  const ac = mode === "advanced" ? C.cy : C.red;

  return (
    <div>
      {/* Header */}
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
                Ter26 RWD · Bicycle model · 540 Nm/wheel · PI yaw-moment controller
              </span>
            </div>
          </div>
          <button onClick={reset} style={{
            ...GL, border: `1px solid ${C.b2}`, borderRadius: 8,
            padding: "5px 16px", color: C.md, fontFamily: C.dt,
            fontSize: 9, cursor: "pointer", letterSpacing: 1.5, fontWeight: 700,
          }}>
            ↺ RESET
          </button>
        </div>
      </div>

      {/* Mode selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14, alignItems: "center", flexWrap: "wrap" }}>
        <Pill active={mode === "simple"}   label="⊗  Simple TV"  onClick={() => onModeChange("simple")}   color={C.red} />
        <Pill active={mode === "advanced"} label="◎  Advanced TV" onClick={() => onModeChange("advanced")} color={C.cy}  />
        <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, marginLeft: 8 }}>
          WASD / Arrow Keys · or sliders · cross the checkered line to record laps
        </span>
      </div>

      {/* Canvas + right panel */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: 10, marginBottom: 10 }}>

        <div style={{
          ...GL, padding: 0, overflow: "hidden",
          border: `1px solid ${ac}28`,
          transition: "border-color 0.4s",
          position: "relative",
        }}>
          <canvas
            ref={canvasRef}
            style={{ display: "block", width: "100%", height: "auto", aspectRatio: "680/380" }}
          />
          {/* Wheel colour legend */}
          <div style={{ position: "absolute", bottom: 10, right: 12, display: "flex", flexDirection: "column", gap: 4 }}>
            {[
              { c: "rgba(0,200,0,0.8)",   l: "Low"  },
              { c: "rgba(255,200,0,0.8)", l: "Mid"  },
              { c: "rgba(255,80,0,0.85)", l: "Peak" },
            ].map(e => (
              <div key={e.l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: e.c }} />
                <span style={{ fontSize: 7.5, color: "rgba(150,165,190,0.7)", fontFamily: C.dt }}>{e.l}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

          {/* Torque bars */}
          <div style={{ ...GL, padding: "12px 10px", flex: "0 0 auto" }}>
            <div style={{ fontSize: 7.5, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 2, textAlign: "center", marginBottom: 8 }}>
              PER-WHEEL
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <TorqueBar label="FL" value={0}        max={V.Tp} color={C.dm} driven={false} />
              <TorqueBar label="FR" value={0}        max={V.Tp} color={C.dm} driven={false} />
              <TorqueBar label="RL" value={disp.Trl} max={V.Tp} color={ac}   driven={true}  />
              <TorqueBar label="RR" value={disp.Trr} max={V.Tp} color={ac}   driven={true}  />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 10 }}>
              <div style={{ ...GL, padding: "6px 8px", textAlign: "center" }}>
                <div style={{ fontSize: 6.5, color: C.dm, fontFamily: C.dt, marginBottom: 2 }}>M_TV [Nm]</div>
                <div style={{ fontSize: 14, fontWeight: 800, fontFamily: C.hd, color: Math.abs(disp.Mtv) > 20 ? ac : C.dm }}>
                  {disp.Mtv.toFixed(0)}
                </div>
              </div>
              <div style={{ ...GL, padding: "6px 8px", textAlign: "center" }}>
                <div style={{ fontSize: 6.5, color: C.dm, fontFamily: C.dt, marginBottom: 2 }}>ΔT [Nm]</div>
                <div style={{ fontSize: 14, fontWeight: 700, fontFamily: C.hd, color: Math.abs(disp.Trr - disp.Trl) > 10 ? ac : C.dm }}>
                  {(disp.Trr - disp.Trl).toFixed(0)}
                </div>
              </div>
            </div>
          </div>

          {/* Lap timer — takes remaining height */}
          <LapTimer current={lapDisp.current} best={lapDisp.best} history={lapDisp.history} />
        </div>
      </div>

      {/* Sliders */}
      <div style={{ ...GL, padding: "12px 18px", marginBottom: 10 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22 }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>STEERING ← →</span>
              <span style={{ fontSize: 8, color: C.br, fontFamily: C.dt, fontWeight: 700 }}>
                {(slSteer * V.dMax * 180 / Math.PI).toFixed(1)}°
              </span>
            </div>
            <input type="range" min={-1} max={1} step={0.01} value={slSteer}
              onChange={onSlSteer}
              style={{ width: "100%", accentColor: ac, cursor: "pointer", height: 4 }} />
          </div>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>THROTTLE ↑</span>
              <span style={{ fontSize: 8, color: C.br, fontFamily: C.dt, fontWeight: 700 }}>
                {(slThr * 100).toFixed(0)} %
              </span>
            </div>
            <input type="range" min={0} max={1} step={0.01} value={slThr}
              onChange={onSlThr}
              style={{ width: "100%", accentColor: C.gn, cursor: "pointer", height: 4 }} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Speed"    value={`${disp.speed.toFixed(1)} km/h`}       sub="body speed"        sentiment={disp.speed > 20 ? "positive" : "neutral"} delay={0} />
        <KPI label="ψ̇  ref"  value={`${disp.wzRef.toFixed(3)} r/s`}        sub="Ackermann target"  sentiment="neutral"                                    delay={1} />
        <KPI label="ψ̇  act"  value={`${disp.wzAct.toFixed(3)} r/s`}        sub="measured yaw rate" sentiment={Math.abs(disp.wzErr) < 0.06 ? "positive" : "amber"} delay={2} />
        <KPI label="Yaw Err"  value={`${(disp.wzErr * 1000).toFixed(1)} mr/s`} sub={mode === "advanced" ? "PI corrected" : "feedforward"}
          sentiment={Math.abs(disp.wzErr) < 0.06 ? "positive" : mode === "advanced" ? "amber" : "negative"} delay={3} />
      </div>

      {/* Theory */}
      <TheoryBlock mode={mode} />
    </div>
  );
}