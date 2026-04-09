// ═══════════════════════════════════════════════════════════════════════════
// src/TorqueVectoringSimModule.jsx  —  Project-GP Dashboard v5.1 rev-4
// ═══════════════════════════════════════════════════════════════════════════
// rev-4 fixes vs rev-3:
//   • LEFT ARC: anticlockwise=true → false  (was drawing east side, not west)
//   • FINISH LINE: moved to rightmost point (FL_X = +STRA+CR), car heading N
//   • TRACK SIZE: STRA 28→22, CR 10→9, THW 4.5→4  (fits narrower canvas)
//   • SLIDER SIGN: slRef.steer = -v  (slider-right now = car-right)
//   • S0: starts at rightmost arc, just south of finish, heading north
// ═══════════════════════════════════════════════════════════════════════════

import React, { useRef, useEffect, useState, useCallback } from "react";
import { C, GL } from "./theme.js";
import { KPI, Pill } from "./components.jsx";

// ─────────────────────────────────────────────────────────────────────────────
// §1  Vehicle parameters  (Ter26 RWD)
// ─────────────────────────────────────────────────────────────────────────────
const V = {
  m: 280, Iz: 120,
  L: 1.53, lf: 0.76, lr: 0.77,
  tw: 1.20, rw: 0.254,
  Cf: 28000, Cr: 32000,
  Tp: 540,        // Nm/wheel at patch (motor 120 Nm × 4.5 gear ratio)
  Cd: 1.1, Af: 1.0, rho: 1.225,
  dMax: 0.38,     // rad
  Kus: 0.006,
};
const KP_TV = 1500;
const KI_TV = 14;

// ─────────────────────────────────────────────────────────────────────────────
// §2  Track geometry — stadium (rounded rectangle)
// ─────────────────────────────────────────────────────────────────────────────
const STRA  = 22;    // m  half-length of straight sections
const CR    = 9;     // m  corner radius (centreline)
const THW   = 4.0;   // m  half track-width
const SCALE = 5.0;   // px/m

// Finish line at the RIGHTMOST point — car passes heading north, line is horizontal
const FL_X     = STRA + CR;    // +31 m
const FL_X_TOL = THW * 1.5;   // m  lateral gate
const FL_Y_ARM = 1.5;          // m  must have been south before counting

// Car starts just south of finish line, on the right arc, heading north
const S0 = Object.freeze({
  x:   FL_X,
  y:  -4,
  psi: Math.PI / 2,
  vx:  14,
  vy:  0,
  r:   0,
  wz_int: 0,
});

// ─────────────────────────────────────────────────────────────────────────────
// §3  Physics
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
    // Steer-proportional feedforward.
    // right turn → steer < 0 → −steer > 0 → dT_s > 0 → Trr > Trl  ✓
    const dT_s = clamp(-steer * Tdrive * 0.30, -V.Tp * 0.42, V.Tp * 0.42);
    Trl = clamp(Tdrive / 2 - dT_s, 0, V.Tp);
    Trr = clamp(Tdrive / 2 + dT_s, 0, V.Tp);
  } else {
    // PI yaw-moment feedback.
    // right-turn understeer → err < 0 → dT < 0 → rawRr = T/2 − dT > T/2  ✓
    const err       = wzRef - s.r;
    const decayRate = 1 - 14 * dt * Math.max(0, 1 - Math.abs(err) / 0.08);
    wz_int = clamp(s.wz_int * decayRate + err * dt, -3.5, 3.5);
    const dM = KP_TV * err + KI_TV * wz_int;
    const dT = clamp(dM / (V.tw / V.rw), -V.Tp * 0.55, V.Tp * 0.55);

    const rawRl = Tdrive / 2 + dT;
    const rawRr = Tdrive / 2 - dT;
    Trl = clamp(rawRl, 0, V.Tp);
    Trr = clamp(rawRr, 0, V.Tp);
    const deficit = Math.max(0, -rawRl) + Math.max(0, -rawRr);
    if (deficit > 0) {
      Trl = Math.min(V.Tp, Trl + deficit * 0.5);
      Trr = Math.min(V.Tp, Trr + deficit * 0.5);
    }
  }

  const Fx    = (Trl + Trr) / V.rw - Fbrake;
  const Fdrag = 0.5 * V.rho * V.Cd * V.Af * v * v;
  // M_z = (Trl − Trr)·tw/(2·rw): right turn → Trr > Trl → Mtv < 0 → right yaw ✓
  const Mtv   = ((Trl - Trr) / V.rw) * (V.tw / 2);

  const ax = (Fx - Fdrag - FyF * Math.sin(steer)) / V.m + s.vy * s.r;
  const ay = (FyF * Math.cos(steer) + FyR)         / V.m - vx  * s.r;
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
// §4  Canvas helpers
// ─────────────────────────────────────────────────────────────────────────────
function w2c(wx, wy, cw, ch) {
  return [cw / 2 + wx * SCALE, ch / 2 - wy * SCALE];
}

// Stadium path (rounded rectangle) at centreline radius R.
// Arc direction: clockwise on screen (anticlockwise=false) for BOTH arcs.
//   Right arc: from top (−π/2) → east side → bottom (+π/2)   [anticlockwise=false] ✓
//   Left  arc: from bottom (π/2) → WEST side → top (−π/2)    [anticlockwise=false] ✓
//   Bug in rev-3: left arc used `true` → swept east side instead of west.
function stadiumPath(ctx, cw, ch, R) {
  const lcx = cw / 2 - STRA * SCALE;   // left  arc centre (world x = −STRA)
  const rcx = cw / 2 + STRA * SCALE;   // right arc centre (world x = +STRA)
  const cy  = ch / 2;
  const rs  = R * SCALE;

  ctx.beginPath();
  ctx.moveTo(lcx, cy - rs);                                        // top-left
  ctx.lineTo(rcx, cy - rs);                                        // top-right
  ctx.arc(rcx, cy, rs, -Math.PI / 2, Math.PI / 2,  false);        // right arc ✓
  ctx.lineTo(lcx, cy + rs);                                        // bottom-right → bottom-left
  ctx.arc(lcx, cy, rs,  Math.PI / 2, -Math.PI / 2, false);        // left arc ✓ (was true — BUG)
  ctx.closePath();
}

function drawTrack(ctx, cw, ch) {
  ctx.fillStyle = "#04060c";
  ctx.fillRect(0, 0, cw, ch);

  // Grid
  const step = 10 * SCALE;
  ctx.strokeStyle = "rgba(18,30,55,0.40)";
  ctx.lineWidth = 0.5;
  for (let gx = (cw / 2) % step; gx < cw; gx += step) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, ch); ctx.stroke(); }
  for (let gy = (ch / 2) % step; gy < ch; gy += step) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cw, gy); ctx.stroke(); }

  // Track surface
  stadiumPath(ctx, cw, ch, CR + THW);
  ctx.fillStyle = "#0d1520"; ctx.fill();

  // Infield punch-out
  stadiumPath(ctx, cw, ch, CR - THW);
  ctx.fillStyle = "#060d08"; ctx.fill();

  // Inner shade
  if (CR - THW > 2) {
    stadiumPath(ctx, cw, ch, CR - THW - 1.2);
    ctx.fillStyle = "#050b07"; ctx.fill();
  }

  // Outer kerb
  stadiumPath(ctx, cw, ch, CR + THW);
  ctx.strokeStyle = "rgba(200,220,255,0.14)"; ctx.lineWidth = 1.5; ctx.stroke();

  // Inner kerb
  stadiumPath(ctx, cw, ch, CR - THW);
  ctx.strokeStyle = "rgba(200,220,255,0.14)"; ctx.lineWidth = 1.5; ctx.stroke();

  // Centreline dashes
  stadiumPath(ctx, cw, ch, CR);
  ctx.strokeStyle = "rgba(255,255,255,0.055)"; ctx.lineWidth = 1;
  ctx.setLineDash([10, 14]); ctx.stroke(); ctx.setLineDash([]);
}

function drawFinishLine(ctx, cw, ch) {
  // HORIZONTAL finish line at rightmost arc point (FL_X = +STRA+CR, y=0).
  // Car travels NORTH here → perpendicular = EAST-WEST = horizontal on screen.
  // Spans x ∈ [FL_X − THW, FL_X + THW] at world y=0.
  const [flxL, fly] = w2c(FL_X - THW, 0, cw, ch);
  const [flxR,    ] = w2c(FL_X + THW, 0, cw, ch);

  const lineW = flxR - flxL;
  const cols  = 8;
  const tileW = lineW / cols;
  const tileH = 7;

  for (let col = 0; col < cols; col++) {
    for (let row = 0; row < 2; row++) {
      ctx.fillStyle = (col + row) % 2 === 0
        ? "rgba(255,255,255,0.55)"
        : "rgba(0,0,0,0.45)";
      ctx.fillRect(flxL + col * tileW, fly - tileH + row * tileH, tileW, tileH);
    }
  }
  ctx.beginPath();
  ctx.moveTo(flxL - 3, fly); ctx.lineTo(flxR + 3, fly);
  ctx.strokeStyle = "rgba(255,255,255,0.30)"; ctx.lineWidth = 1.5; ctx.stroke();
}

function drawTrail(ctx, trail, mode) {
  if (trail.length < 2) return;
  ctx.beginPath();
  trail.forEach(([tx, ty], i) => (i === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty)));
  ctx.strokeStyle = mode === "advanced" ? "rgba(0,210,255,0.52)" : "rgba(225,6,0,0.52)";
  ctx.lineWidth = 2.5; ctx.lineJoin = "round"; ctx.stroke();
}

function drawCar(ctx, cw, ch, state, mode) {
  const [px, py] = w2c(state.x, state.y, cw, ch);
  const Trl = state.Trl ?? 0, Trr = state.Trr ?? 0;

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(Math.PI / 2 - state.psi);

  const BL = 14, BW = 6;
  const ac = mode === "advanced" ? C.cy : C.red;

  ctx.beginPath(); ctx.rect(-BW, -BL, BW * 2, BL * 2);
  ctx.fillStyle = mode === "advanced" ? "rgba(4,20,38,0.93)" : "rgba(28,6,6,0.93)";
  ctx.fill(); ctx.strokeStyle = ac; ctx.lineWidth = 1.5; ctx.stroke();

  ctx.beginPath(); ctx.moveTo(0, -(BL+6)); ctx.lineTo(-BW*0.55, -BL+2); ctx.lineTo(BW*0.55, -BL+2); ctx.closePath();
  ctx.fillStyle = ac; ctx.fill();

  ctx.beginPath(); ctx.rect(-(BW+3), BL-4, (BW+3)*2, 2.5); ctx.fillStyle = ac; ctx.fill();

  ctx.beginPath(); ctx.ellipse(0, 0, BW*0.55, BL*0.35, 0, 0, Math.PI*2);
  ctx.fillStyle = "rgba(0,0,0,0.50)"; ctx.fill();

  // Wheels: RL at −x (left), RR at +x (right). RR glows for right turn ✓
  [
    { x: -(BW+3.5), y: -BL*0.50, T: 0,   driven: false },
    { x:  (BW+3.5), y: -BL*0.50, T: 0,   driven: false },
    { x: -(BW+3.5), y:  BL*0.50, T: Trl, driven: true  },
    { x:  (BW+3.5), y:  BL*0.50, T: Trr, driven: true  },
  ].forEach(w => {
    const ratio = w.driven ? Math.min(1, w.T / V.Tp) : 0;
    let bc;
    if (!w.driven || ratio < 0.05) bc = "rgba(80,100,130,0.55)";
    else if (ratio < 0.40) { const t = ratio / 0.40; bc = `rgba(${Math.round(t*255)},230,0,${0.55+t*0.35})`; }
    else if (ratio < 0.75) { const t = (ratio-0.40)/0.35; bc = `rgba(255,${Math.round((1-t)*180)},0,0.90)`; }
    else bc = `rgba(255,${Math.round((1-(ratio-0.75)/0.25)*80)},0,1.0)`;

    ctx.beginPath(); ctx.rect(w.x-3, w.y-5, 6, 10);
    ctx.fillStyle = "rgba(20,25,35,0.92)"; ctx.fill();
    ctx.strokeStyle = bc; ctx.lineWidth = 1.8; ctx.stroke();
    if (ratio > 0.45) {
      ctx.shadowBlur = 9; ctx.shadowColor = bc;
      ctx.beginPath(); ctx.rect(w.x-3, w.y-5, 6, 10); ctx.stroke();
      ctx.shadowBlur = 0;
    }
  });

  ctx.restore();
}

function renderHUD(ctx, state, mode, steer) {
  const spd = Math.hypot(state.vx, state.vy) * 3.6;
  const mc  = mode === "advanced" ? "#00d2ff" : "#e10600";

  ctx.font = "bold 11px 'Azeret Mono', monospace";
  ctx.fillStyle = "rgba(192,200,218,0.85)";
  ctx.fillText(`${spd.toFixed(0)} km/h`, 10, 20);

  ctx.font = "bold 9px 'Azeret Mono', monospace";
  ctx.fillStyle = mc;
  ctx.fillText(mode === "advanced" ? "◎ ADVANCED TV" : "⊗ SIMPLE TV", 10, 34);

  const arcX = 28, arcY = 72, arcR = 18;
  ctx.beginPath(); ctx.arc(arcX, arcY, arcR, Math.PI, 2*Math.PI, false);
  ctx.strokeStyle = "rgba(70,90,130,0.40)"; ctx.lineWidth = 2.5; ctx.stroke();

  const angle = Math.PI + (steer / V.dMax) * (Math.PI / 2);
  ctx.beginPath(); ctx.moveTo(arcX, arcY);
  ctx.lineTo(arcX + Math.cos(angle)*arcR, arcY + Math.sin(angle)*arcR);
  ctx.strokeStyle = mc; ctx.lineWidth = 2; ctx.stroke();

  ctx.beginPath(); ctx.arc(arcX, arcY, 3, 0, Math.PI*2);
  ctx.fillStyle = mc; ctx.fill();

  ctx.font = "8px 'Azeret Mono', monospace";
  ctx.fillStyle = "rgba(100,120,160,0.80)";
  ctx.fillText("STEER", arcX - 13, arcY + 11);
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Lap timing
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
      <div style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, fontWeight: 700 }}>{label}</div>
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
        {driven ? value.toFixed(0) : "—"}{driven && <span style={{ fontSize: 6.5, opacity: 0.6 }}> Nm</span>}
      </div>
    </div>
  );
}

function LapTimer({ current, best, history }) {
  const isNewBest = best > 0 && history.length >= 1 && history[0] === best;
  return (
    <div style={{ ...GL, padding: "12px 14px", flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 7.5, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 2 }}>LAP TIMER</div>

      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt }}>CURRENT LAP</span>
        <span style={{ fontSize: 20, fontWeight: 800, color: C.w, fontFamily: C.hd, letterSpacing: 1, lineHeight: 1.1 }}>
          {fmtLap(current)}
        </span>
      </div>

      <div style={{
        padding: "7px 10px", borderRadius: 6,
        background: best > 0 ? `${C.gn}12` : "rgba(20,30,50,0.5)",
        border: `1px solid ${best > 0 ? C.gn + "30" : "rgba(50,68,105,0.4)"}`,
        transition: "all 0.4s",
      }}>
        <div style={{ fontSize: 7, color: C.gn, fontFamily: C.dt, marginBottom: 3 }}>
          {isNewBest ? "★ NEW BEST" : "BEST LAP"}
        </div>
        <div style={{ fontSize: 15, fontWeight: 800, color: best > 0 ? C.gn : C.dm, fontFamily: C.hd }}>
          {fmtLap(best)}
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
        {history.slice(0, 5).map((t, i) => (
          <div key={i} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "3px 6px", borderRadius: 4,
            background: t === best ? `${C.gn}09` : "transparent",
          }}>
            <span style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt }}>LAP {history.length - i}</span>
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

const F = ({ c, children }) => <span style={{ color: c, fontWeight: 700 }}>{children}</span>;

function TheoryBlock({ mode }) {
  const isAdv = mode === "advanced";
  const ac    = isAdv ? C.cy : C.red;
  return (
    <div style={{ ...GL, padding: "16px 20px", borderLeft: `3px solid ${ac}`, transition: "border-color 0.4s" }}>
      <div style={{ fontSize: 10, fontWeight: 800, color: ac, fontFamily: C.dt, letterSpacing: 2.5, marginBottom: 10 }}>
        {isAdv ? "ADVANCED TV — SOCP-INSPIRED PI YAW-MOMENT CONTROLLER" : "SIMPLE TV — STEER-PROPORTIONAL FEEDFORWARD"}
      </div>
      {!isAdv ? (
        <>
          <p style={{ fontSize: 9, color: C.br, fontFamily: C.dt, lineHeight: 1.85, margin: 0 }}>
            Open-loop: outer-wheel bias <F c={C.red}>ΔT_s ∝ −δ × T_total</F>.
            Turning right → RR gets more torque, no closed-loop correction.
            Under heavy throttle-on, static gain overshoots → <F c={C.am}>torque-induced oversteer</F>.
          </p>
          <pre style={{ fontFamily: "monospace", fontSize: 8, color: C.dm, marginTop: 12, padding: "10px 14px", background: "rgba(5,8,15,0.85)", borderRadius: 6, lineHeight: 2.1 }}>
{`ΔT_s   =  clamp(−δ × T_total × 0.30,  ±0.42·Tp)
T_rl   =  T/2 − ΔT_s   (inner for right turn → less)
T_rr   =  T/2 + ΔT_s   (outer for right turn → more)
M_yaw  =  (T_rl − T_rr)·tw/(2·rw)   [open-loop]`}
          </pre>
        </>
      ) : (
        <>
          <p style={{ fontSize: 9, color: C.br, fontFamily: C.dt, lineHeight: 1.85, margin: 0 }}>
            Closed-loop PI: <F c={C.cy}>ψ̇_ref = v·δ/(L·(1 + K_us·v²))</F>.
            Error <F c={C.gn}>ε = ψ̇_ref − ψ̇</F> drives
            <F c={C.am}> ΔM = Kp·ε + Ki·∫ε</F>.
            Anti-windup decay + SOCP-style thrust preservation keeps total Fx invariant.
          </p>
          <pre style={{ fontFamily: "monospace", fontSize: 8, color: C.dm, marginTop: 12, padding: "10px 14px", background: "rgba(5,8,15,0.85)", borderRadius: 6, lineHeight: 2.1 }}>
{`ε       =  ψ̇_ref − ψ̇_actual
ΔT      =  clamp((1500·ε + 14·∫ε)·rw/tw,  ±0.55·Tp)
T_rl    =  T/2 + ΔT   │   T_rr = T/2 − ΔT
M_yaw   =  (T_rl − T_rr)·tw/(2·rw)   [closed-loop, right turn: M<0 ✓]`}
          </pre>
        </>
      )}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 14 }}>
        {[
          { label: "Corner entry",  s: "Understeer if δ-gain mismatch",   a: "ψ̇ closed-loop → follows apex"  },
          { label: "Mid-corner",    s: "Static gain, no tyre update",      a: "PI corrects slip in real time"  },
          { label: "Exit throttle", s: "Oversteer from outer-bias",        a: "Integral decay → full Fx"       },
          { label: "Yaw moment",    s: "Feed-forward only",                a: "Active ±50 Nm correction"       },
        ].map(row => (
          <div key={row.label} style={{ ...GL, padding: "8px 12px" }}>
            <div style={{ fontSize: 7.5, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, marginBottom: 5, fontWeight: 700 }}>{row.label}</div>
            <div style={{ fontSize: 8, color: C.red, fontFamily: C.dt, marginBottom: 3 }}>⊗ {row.s}</div>
            <div style={{ fontSize: 8, color: C.cy,  fontFamily: C.dt }}>◎ {row.a}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Main component
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

  const prevYRef      = useRef(S0.y);
  const lapStartRef   = useRef(null);
  const bestLapRef    = useRef(0);
  const lapHistoryRef = useRef([]);

  const [mode,    setMode]    = useState("simple");
  const [slSteer, setSlSteer] = useState(0);
  const [slThr,   setSlThr]   = useState(0);
  const [disp,    setDisp]    = useState({ speed: 0, wzRef: 0, wzAct: 0, wzErr: 0, Mtv: 0, Trl: 0, Trr: 0 });
  const [lapDisp, setLapDisp] = useState({ current: 0, best: 0, history: [] });

  useEffect(() => { modeRef.current = mode; }, [mode]);

  useEffect(() => {
    const dn = e => keysRef.current.add(e.key);
    const up = e => keysRef.current.delete(e.key);
    window.addEventListener("keydown", dn);
    window.addEventListener("keyup",   up);
    return () => { window.removeEventListener("keydown", dn); window.removeEventListener("keyup", up); };
  }, []);

  const reset = useCallback(() => {
    stateRef.current      = { ...S0 };
    kbRef.current         = { steer: 0, throttle: 0, brake: 0 };
    slRef.current         = { steer: 0, throttle: 0 };
    trailRef.current      = [];
    prevYRef.current      = S0.y;
    lapStartRef.current   = null;
    bestLapRef.current    = 0;
    lapHistoryRef.current = [];
    setSlSteer(0); setSlThr(0);
    setDisp({ speed: 0, wzRef: 0, wzAct: 0, wzErr: 0, Mtv: 0, Trl: 0, Trr: 0 });
    setLapDisp({ current: 0, best: 0, history: [] });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx  = canvas.getContext("2d");
    const dpr  = window.devicePixelRatio || 1;
    const logW = canvas.clientWidth  || 640;
    const logH = canvas.clientHeight || 320;
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

      // Keyboard: left-arrow = positive steer = left turn; right-arrow = negative = right turn
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

      let s  = stateRef.current;
      const dt = rawDt / 4;
      for (let i = 0; i < 4; i++)
        s = physicsStep(s, steer, throttle, brake, modeRef.current, dt);
      stateRef.current = s;

      // Lap crossing: car near rightmost point (x ≈ FL_X), y crossing north
      const prevY = prevYRef.current;
      const currY = s.y;
      if (
        Math.abs(s.x - FL_X) < FL_X_TOL &&
        prevY < -FL_Y_ARM &&
        currY >= -FL_Y_ARM
      ) {
        if (lapStartRef.current !== null) {
          const lapMs = ts - lapStartRef.current;
          if (lapMs > 5000) {
            lapHistoryRef.current = [lapMs, ...lapHistoryRef.current].slice(0, 10);
            if (bestLapRef.current === 0 || lapMs < bestLapRef.current)
              bestLapRef.current = lapMs;
          }
        }
        lapStartRef.current = ts;
      }
      prevYRef.current = currY;

      const cw = logW, ch = logH;
      const [tx, ty] = w2c(s.x, s.y, cw, ch);
      trailRef.current.push([tx, ty]);
      if (trailRef.current.length > 600) trailRef.current.shift();

      ctx.clearRect(0, 0, cw, ch);
      drawTrack(ctx, cw, ch);
      drawFinishLine(ctx, cw, ch);
      drawTrail(ctx, trailRef.current, modeRef.current);
      drawCar(ctx, cw, ch, s, modeRef.current);
      renderHUD(ctx, s, modeRef.current, steer);

      if (++tick % 4 === 0) {
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
          current: lapStartRef.current !== null ? (ts - lapStartRef.current) : 0,
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

  // SLIDER SIGN FIX: slider-right → -v → negative steer → right turn ✓
  const onSlSteer = useCallback(e => {
    const v = parseFloat(e.target.value);
    slRef.current.steer = -v;   // ← negated so slider-right = car-right
    setSlSteer(v);
  }, []);
  const onSlThr = useCallback(e => {
    const v = parseFloat(e.target.value);
    slRef.current.throttle = v;
    setSlThr(v);
  }, []);
  const onModeChange = useCallback(m => {
    setMode(m); setTimeout(reset, 0);
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
                Ter26 RWD · Stadium 44m × 26m · 10m corner radius · 540 Nm/wheel
              </span>
            </div>
          </div>
          <button onClick={reset} style={{
            ...GL, border: `1px solid ${C.b2}`, borderRadius: 8,
            padding: "5px 16px", color: C.md, fontFamily: C.dt,
            fontSize: 9, cursor: "pointer", letterSpacing: 1.5, fontWeight: 700,
          }}>↺ RESET</button>
        </div>
      </div>

      {/* Mode */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14, alignItems: "center", flexWrap: "wrap" }}>
        <Pill active={mode === "simple"}   label="⊗  Simple TV"  onClick={() => onModeChange("simple")}   color={C.red} />
        <Pill active={mode === "advanced"} label="◎  Advanced TV" onClick={() => onModeChange("advanced")} color={C.cy}  />
        <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, marginLeft: 8 }}>
          WASD / Arrows · sliders · cross checkered line (right arc) to record laps
        </span>
      </div>

      {/* Canvas + panel */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: 10, marginBottom: 10 }}>
        <div style={{
          ...GL, padding: 0, overflow: "hidden",
          border: `1px solid ${ac}28`, transition: "border-color 0.4s",
          position: "relative",
        }}>
          <canvas
            ref={canvasRef}
            style={{ display: "block", width: "100%", height: "auto", aspectRatio: "640/320" }}
          />
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

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ ...GL, padding: "12px 10px" }}>
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
          <LapTimer current={lapDisp.current} best={lapDisp.best} history={lapDisp.history} />
        </div>
      </div>

      {/* Sliders */}
      <div style={{ ...GL, padding: "12px 18px", marginBottom: 10 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22 }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>← LEFT · STEER · RIGHT →</span>
              <span style={{ fontSize: 8, color: C.br, fontFamily: C.dt, fontWeight: 700 }}>
                {Math.abs(slSteer * V.dMax * 180 / Math.PI).toFixed(1)}°{slSteer > 0.02 ? " R" : slSteer < -0.02 ? " L" : ""}
              </span>
            </div>
            <input type="range" min={-1} max={1} step={0.01} value={slSteer} onChange={onSlSteer}
              style={{ width: "100%", accentColor: ac, cursor: "pointer", height: 4 }} />
          </div>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>THROTTLE ↑</span>
              <span style={{ fontSize: 8, color: C.br, fontFamily: C.dt, fontWeight: 700 }}>{(slThr * 100).toFixed(0)} %</span>
            </div>
            <input type="range" min={0} max={1} step={0.01} value={slThr} onChange={onSlThr}
              style={{ width: "100%", accentColor: C.gn, cursor: "pointer", height: 4 }} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Speed"   value={`${disp.speed.toFixed(1)} km/h`}          sub="body speed"        sentiment={disp.speed > 20 ? "positive" : "neutral"}                                    delay={0} />
        <KPI label="ψ̇  ref" value={`${disp.wzRef.toFixed(3)} r/s`}           sub="Ackermann target"  sentiment="neutral"                                                                     delay={1} />
        <KPI label="ψ̇  act" value={`${disp.wzAct.toFixed(3)} r/s`}           sub="measured yaw rate" sentiment={Math.abs(disp.wzErr) < 0.06 ? "positive" : "amber"}                         delay={2} />
        <KPI label="Yaw Err" value={`${(disp.wzErr * 1000).toFixed(1)} mr/s`} sub={mode === "advanced" ? "PI corrected" : "feedforward"}
          sentiment={Math.abs(disp.wzErr) < 0.06 ? "positive" : mode === "advanced" ? "amber" : "negative"} delay={3} />
      </div>

      <TheoryBlock mode={mode} />
    </div>
  );
}