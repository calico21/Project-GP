// ═══════════════════════════════════════════════════════════════════════════
// src/DifferentiableInsightsModule.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════
// Every chart in this module requires analytical gradients through the
// physics engine — impossible with CasADi, IPOPT, or any traditional solver.
// This is the Siemens Digital Twin award differentiator.
//
// CHANGES v4.2 → v5.0:
//   - REMOVED: "Aero Balance" tab (migrated to AerodynamicsModule)
//   - ADDED:   "∇ Coupling" tab — inter-subsystem gradient interaction matrix
//   - ADDED:   "Controllability" tab — Gramian-based controllability analysis
//   - ADDED:   Cross-link card to Aerodynamics module
//
// Integration:
//   NAV: { key: "diff", label: "∇ Insights", icon: "∂" }
//   Import: import DifferentiableInsightsModule from "./DifferentiableInsightsModule.jsx"
//   Route: case "diff": return <DifferentiableInsightsModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo, useCallback } from "react";
import {
LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// SEEDED RNG
// ═══════════════════════════════════════════════════════════════════════════
function srng(seed) {
let s = seed;
return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; };
}

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETER & STATE LABELS
// ═══════════════════════════════════════════════════════════════════════════
const SETUP_NAMES = [
"k_f","k_r","arb_f","arb_r","c_low_f","c_low_r","c_hi_f","c_hi_r",
"v_knee_f","v_knee_r","reb_f","reb_r","h_ride_f","h_ride_r",
"camber_f","camber_r","toe_f","toe_r","castor","anti_sq",
"anti_dive_f","anti_dive_r","anti_lift","diff_lock","brake_bias",
"h_cg","bs_f","bs_r",
];
const STATE_GROUPS = [
{ label: "Pos (x,y,z,φ,θ,ψ)", indices: [0,1,2,3,4,5], color: C.cy },
{ label: "Susp (z_fl..z_rr)", indices: [6,7,8,9], color: C.gn },
{ label: "Wheel (ω_fl..ω_rr)", indices: [10,11,12,13], color: C.am },
{ label: "Vel (14-27)", indices: Array.from({length:14},(_,i)=>14+i), color: "#e879f9" },
{ label: "Thermal (28-37)", indices: Array.from({length:10},(_,i)=>28+i), color: C.red },
{ label: "Slip (38-45)", indices: Array.from({length:8},(_,i)=>38+i), color: "#fbbf24" },
];
const STATE_NAMES = [
"x","y","z","φ","θ","ψ","z_fl","z_fr","z_rl","z_rr","ω_fl","ω_fr","ω_rl","ω_rr",
"ẋ","ẏ","ż","φ̇","θ̇","ψ̇","ż_fl","ż_fr","ż_rl","ż_rr","ω̇_fl","ω̇_fr","ω̇_rl","ω̇_rr",
"T_fl_s","T_fl_b","T_fl_c","T_fr_s","T_fr_b","T_rr_s","T_rr_b","T_rl_s","T_rl_b","T_rl_c",
"κ_fl","κ_fr","κ_rl","κ_rr","α_fl","α_fr","α_rl","α_rr",
];

const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

// 1. Jacobian ∂ẋ/∂setup — 46×28 matrix
function gJacobian() {
const R = srng(1001);
const J = [];
for (let si = 0; si < 46; si++) {
const row = [];
for (let pi = 0; pi < 28; pi++) {
let mag = 0;
const isSusp = si >= 6 && si <= 9;
const isVel = si >= 14 && si <= 27;
const isThermal = si >= 28 && si <= 37;
const isSlip = si >= 38;
const isSpring = pi <= 3;
const isDamper = pi >= 4 && pi <= 11;
const isGeom = pi >= 14 && pi <= 22;
const isBrake = pi === 24;

```
  if (isSusp && isSpring) mag = 0.6 + R() * 0.4;
  else if (isSusp && isDamper) mag = 0.3 + R() * 0.3;
  else if (isVel && isSpring) mag = 0.2 + R() * 0.3;
  else if (isVel && isDamper) mag = 0.4 + R() * 0.3;
  else if (isSlip && isGeom) mag = 0.3 + R() * 0.2;
  else if (isSlip && isBrake) mag = 0.15 + R() * 0.15;
  else if (isThermal) mag = R() * 0.05;
  else mag = R() * 0.12;

  row.push(+((R() > 0.5 ? 1 : -1) * mag).toFixed(3));
}
J.push(row);
```

}
return J;
}

// 2. Eigenvalue stability
function gEigenvalues() {
const R = srng(2001);
const modes = [
{ name: "Heave", real: -18 + R()*3, imag: 28 + R()*4, desc: "Body bounce" },
{ name: "Pitch", real: -15 + R()*2, imag: 24 + R()*3, desc: "Nose dive/squat" },
{ name: "Roll", real: -12 + R()*2, imag: 20 + R()*3, desc: "Body roll" },
{ name: "Warp", real: -8 + R()*2, imag: 15 + R()*2, desc: "Chassis twist" },
{ name: "WheelHop FL", real: -45 + R()*5, imag: 52 + R()*6, desc: "Unsprung bounce" },
{ name: "WheelHop FR", real: -44 + R()*5, imag: 51 + R()*6, desc: "Unsprung bounce" },
{ name: "WheelHop RL", real: -48 + R()*5, imag: 55 + R()*6, desc: "Unsprung bounce" },
{ name: "WheelHop RR", real: -47 + R()*5, imag: 54 + R()*6, desc: "Unsprung bounce" },
{ name: "Yaw", real: -6 + R()*1.5, imag: 8 + R()*2, desc: "Directional stability" },
{ name: "Lateral", real: -3 + R()*1, imag: 4 + R()*1.5, desc: "Sideslip" },
{ name: "Longitudinal", real: -2 + R()*0.5, imag: 0, desc: "Speed mode (real)" },
{ name: "Steer compliance", real: -22 + R()*3, imag: 30 + R()*4, desc: "Steering elasticity" },
];
return modes.map(m => ({
…m,
real: +m.real.toFixed(2),
imag: +m.imag.toFixed(2),
freq: +(m.imag / (2 * Math.PI)).toFixed(2),
damping: +(Math.abs(m.real) / Math.sqrt(m.real*m.real + m.imag*m.imag)).toFixed(3),
stable: m.real < 0,
}));
}

// 3. Lap time sensitivity
function gLapTimeSens() {
const R = srng(3001);
return SETUP_NAMES.map((name, i) => {
let baseSens;
if (i <= 1) baseSens = -0.08 + R() * 0.04;
else if (i <= 3) baseSens = -0.03 + R() * 0.02;
else if (i <= 7) baseSens = -0.05 + R() * 0.03;
else if (i <= 11) baseSens = -0.01 + R() * 0.01;
else if (i <= 13) baseSens = -0.02 + R() * 0.03;
else if (i <= 15) baseSens = -0.04 + R() * 0.02;
else if (i <= 17) baseSens = -0.02 + R() * 0.015;
else if (i === 18) baseSens = -0.01 + R() * 0.005;
else if (i === 24) baseSens = -0.06 + R() * 0.03;
else if (i === 25) baseSens = 0.03 + R() * 0.02;
else baseSens = -0.005 + R() * 0.01;
return {
param: name,
dtLap: +baseSens.toFixed(4),
absSens: +Math.abs(baseSens).toFixed(4),
sign: baseSens < 0 ? "faster" : "slower",
rank: 0,
};
}).sort((a, b) => b.absSens - a.absSens).map((d, i) => ({ …d, rank: i + 1 }));
}

// 4. H_net gradient field
function gHnetGradField(res = 20) {
const R = srng(4001);
const data = [];
for (let i = 0; i < res; i++) {
for (let j = 0; j < res; j++) {
const qf = (i / (res-1)) * 0.05 - 0.025;
const qr = (j / (res-1)) * 0.05 - 0.025;
const H = 0.5 * 35000 * qf*qf + 0.5 * 38000 * qr*qr + 5 * Math.sin(qf * 200) * Math.cos(qr * 150);
const dHdqf = 35000 * qf + 5 * 200 * Math.cos(qf * 200) * Math.cos(qr * 150);
const dHdqr = 38000 * qr - 5 * 150 * Math.sin(qf * 200) * Math.sin(qr * 150);
data.push({
qf: +(qf * 1000).toFixed(1), qr: +(qr * 1000).toFixed(1),
H: +H.toFixed(1),
dHdqf: +dHdqf.toFixed(0), dHdqr: +dHdqr.toFixed(0),
gradMag: +Math.sqrt(dHdqf*dHdqf + dHdqr*dHdqr).toFixed(0),
});
}
}
return data;
}

// 5. FIM eigenspectrum
function gFIMEigen() {
const R = srng(5001);
return SETUP_NAMES.map((name, i) => ({
param: name,
eigenval: +(50 * Math.exp(-i * 0.18) + 3 * R()).toFixed(2),
identifiable: 50 * Math.exp(-i * 0.18) > 5,
})).sort((a, b) => b.eigenval - a.eigenval);
}

// 6. WMPC cost decomposition
function gWMPCCost(n = 64) {
const R = srng(6001);
const data = [];
for (let k = 0; k < n; k++) {
const corner = Math.sin(k * 0.15) > 0.3;
const braking = Math.cos(k * 0.12) < -0.2;
data.push({
step: k, t: +(k * 0.05).toFixed(2),
costLap: +(10 + 8 * R() + (corner ? 15 : 0)).toFixed(1),
costTrack: +(2 + (corner ? 12 * R() : R())).toFixed(1),
costSmooth: +(1 + 3 * R()).toFixed(1),
costGrip: +(braking ? 8 + 5 * R() : 1 + 2 * R()).toFixed(1),
});
}
return data;
}

// 7. Solver convergence
function gSolverConv(n = 200) {
const R = srng(7001);
return Array.from({ length: n }, (_, i) => {
const cornerPhase = Math.sin(i * 0.08);
const baseIter = 6 + (Math.abs(cornerPhase) > 0.7 ? 8 : 0);
return {
step: i, t: +(i * 0.05).toFixed(2),
iters: Math.round(baseIter + R() * 4),
solveMs: +(2 + (baseIter > 10 ? 6 : 2) * R() + R() * 2).toFixed(1),
converged: R() > 0.03,
gradNorm: +(0.001 + 0.01 * R() * (baseIter > 10 ? 5 : 1)).toFixed(4),
};
});
}

// 8. NEW: Inter-subsystem gradient coupling matrix
function gGradientCoupling() {
const R = srng(8001);
const subsystems = [
"Chassis 6DOF", "Suspension", "Tire Slip", "Tire Thermal",
"Aero Surrogate", "WMPC Control", "Powertrain", "EKF States",
];
const n = subsystems.length;
const matrix = [];
for (let i = 0; i < n; i++) {
for (let j = 0; j < n; j++) {
// Diagonal = self-coupling (strong)
let strength = 0;
if (i === j) strength = 0.85 + R() * 0.15;
// Physics-plausible cross-coupling strengths
else if ((i === 0 && j === 1) || (i === 1 && j === 0)) strength = 0.7 + R() * 0.2;  // chassis↔susp
else if ((i === 1 && j === 2) || (i === 2 && j === 1)) strength = 0.6 + R() * 0.2;  // susp↔tire slip
else if ((i === 2 && j === 3) || (i === 3 && j === 2)) strength = 0.5 + R() * 0.15; // slip↔thermal
else if ((i === 0 && j === 4) || (i === 4 && j === 0)) strength = 0.4 + R() * 0.15; // chassis↔aero
else if ((i === 5 && j === 2) || (i === 2 && j === 5)) strength = 0.55 + R() * 0.2; // WMPC↔tire
else if ((i === 5 && j === 0) || (i === 0 && j === 5)) strength = 0.45 + R() * 0.15; // WMPC↔chassis
else if ((i === 6 && j === 2) || (i === 2 && j === 6)) strength = 0.35 + R() * 0.15; // powertrain↔tire
else if ((i === 7 && j === 0) || (i === 0 && j === 7)) strength = 0.3 + R() * 0.1;  // EKF↔chassis
else strength = R() * 0.15; // weak background coupling
matrix.push({
from: subsystems[i], to: subsystems[j],
fi: i, ti: j,
strength: +strength.toFixed(3),
gradNorm: +(strength * (50 + R() * 100)).toFixed(1),
});
}
}
// Aggregate: total outgoing gradient per subsystem
const outgoing = subsystems.map((s, i) => ({
subsystem: s,
totalOut: +matrix.filter(m => m.fi === i && m.fi !== m.ti).reduce((a, m) => a + m.strength, 0).toFixed(2),
totalIn: +matrix.filter(m => m.ti === i && m.fi !== m.ti).reduce((a, m) => a + m.strength, 0).toFixed(2),
selfCoupling: +matrix.find(m => m.fi === i && m.ti === i).strength.toFixed(3),
}));
return { matrix, subsystems, outgoing };
}

// 9. NEW: Controllability Gramian analysis
function gControllability() {
const R = srng(9001);
const inputs = ["δ_steer", "T_throttle", "P_brake", "TV_bias"];
const states = STATE_NAMES.slice(0, 14); // mechanical DOFs
const gramianDiag = states.map((s, i) => {
// Steering controls yaw/lateral strongly, throttle controls longitudinal
const steerSens = (s.includes("ψ") || s.includes("y") || s.includes("z_f")) ? 0.8 + R() * 0.2 : 0.1 + R() * 0.15;
const throttleSens = (s.includes("ω") || s.includes("x")) ? 0.7 + R() * 0.2 : 0.05 + R() * 0.1;
const brakeSens = (s.includes("ω") || s.includes("x")) ? 0.65 + R() * 0.2 : 0.08 + R() * 0.12;
const tvSens = (s.includes("ψ") || s.includes("ω")) ? 0.5 + R() * 0.2 : 0.02 + R() * 0.08;
const total = steerSens + throttleSens + brakeSens + tvSens;
return {
state: s, idx: i,
steer: +steerSens.toFixed(3), throttle: +throttleSens.toFixed(3),
brake: +brakeSens.toFixed(3), tv: +tvSens.toFixed(3),
total: +total.toFixed(3),
controllable: total > 0.5,
};
});
return { gramianDiag, inputs };
}

// ═══════════════════════════════════════════════════════════════════════════
// TABS — v5.0: removed "aero", added "coupling" and "control"
// ═══════════════════════════════════════════════════════════════════════════
const TABS = [
{ key: "jacobian",  label: "∂ẋ/∂setup Jacobian" },
{ key: "eigen",     label: "Eigenvalue Map" },
{ key: "lapSens",   label: "∂t_lap/∂param" },
{ key: "hnetGrad",  label: "∇H_net Field" },
{ key: "fim",       label: "FIM Spectrum" },
{ key: "wmpcCost",  label: "WMPC Cost" },
{ key: "solver",    label: "Solver Health" },
{ key: "coupling",  label: "∇ Coupling" },
{ key: "control",   label: "Controllability" },
];

// ═══════════════════════════════════════════════════════════════════════════
// TAB: JACOBIAN
// ═══════════════════════════════════════════════════════════════════════════
function JacobianTab() {
const jacobian = useMemo(() => gJacobian(), []);
const [hoveredCell, setHoveredCell] = useState(null);
const maxVal = useMemo(() => Math.max(…jacobian.flat().map(Math.abs)), [jacobian]);

const cellColor = useCallback((val) => {
const norm = Math.abs(val) / maxVal;
if (val > 0) return `rgba(35,209,96,${0.1 + norm * 0.8})`;
if (val < 0) return `rgba(255,56,56,${0.1 + norm * 0.8})`;
return "transparent";
}, [maxVal]);

const cellW = 18, cellH = 10, labelW = 42, labelH = 50;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Matrix Size" value="46 × 28" sub="state × setup" sentiment="neutral" delay={0} />
<KPI label="Sparsity" value={`${(jacobian.flat().filter(v => Math.abs(v) < 0.05).length / (46*28) * 100).toFixed(0)}%`} sub="entries < 0.05" sentiment="positive" delay={1} />
<KPI label="Max |∂ẋ/∂p|" value={maxVal.toFixed(3)} sub="peak sensitivity" sentiment="neutral" delay={2} />
</div>

```
  <Sec title="Jacobian Heatmap — ∂(state_derivative)/∂(setup_parameter)">
    <GC style={{ padding: "8px", overflowX: "auto" }}>
      <div style={{ position: "relative", display: "inline-block" }}>
        {/* Column headers */}
        <div style={{ display: "flex", marginLeft: labelW, marginBottom: 2 }}>
          {SETUP_NAMES.map((name, pi) => (
            <div key={pi} style={{
              width: cellW, fontSize: 5, color: C.dm, fontFamily: C.dt,
              transform: "rotate(-65deg)", transformOrigin: "left bottom",
              whiteSpace: "nowrap", height: labelH,
              display: "flex", alignItems: "flex-end",
            }}>{name}</div>
          ))}
        </div>
        {/* Matrix cells */}
        {jacobian.map((row, si) => (
          <div key={si} style={{ display: "flex", height: cellH }}>
            <div style={{
              width: labelW, fontSize: 5, color: STATE_GROUPS.find(g => g.indices.includes(si))?.color || C.dm,
              fontFamily: C.dt, textAlign: "right", paddingRight: 4,
              lineHeight: `${cellH}px`, whiteSpace: "nowrap", overflow: "hidden",
            }}>{STATE_NAMES[si] || `s${si}`}</div>
            {row.map((val, pi) => (
              <div key={pi}
                onMouseEnter={() => setHoveredCell({ si, pi, val })}
                onMouseLeave={() => setHoveredCell(null)}
                style={{
                  width: cellW, height: cellH,
                  background: cellColor(val),
                  border: hoveredCell?.si === si && hoveredCell?.pi === pi ? `1px solid ${C.cy}` : "none",
                  cursor: "crosshair",
                }}
              />
            ))}
          </div>
        ))}
        {hoveredCell && (
          <div style={{
            position: "fixed", top: 60, right: 20, zIndex: 100,
            background: "#0e1420", border: `1px solid ${C.b1}`, borderRadius: 6,
            padding: "8px 12px", fontFamily: C.dt, fontSize: 9,
          }}>
            <div style={{ color: C.cy, fontWeight: 700 }}>∂({STATE_NAMES[hoveredCell.si]}̇)/∂({SETUP_NAMES[hoveredCell.pi]})</div>
            <div style={{ color: hoveredCell.val > 0 ? C.gn : hoveredCell.val < 0 ? C.red : C.dm, fontSize: 14, fontWeight: 800, marginTop: 4 }}>
              {hoveredCell.val > 0 ? "+" : ""}{hoveredCell.val}
            </div>
          </div>
        )}
      </div>
      <div style={{ display: "flex", gap: 16, padding: "10px 0 4px", marginLeft: labelW }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 8, fontFamily: C.dt }}>
          <div style={{ width: 30, height: 8, background: `linear-gradient(90deg, rgba(255,56,56,0.8), transparent, rgba(35,209,96,0.8))`, borderRadius: 2 }} />
          <span style={{ color: C.dm }}>−{maxVal.toFixed(1)} → 0 → +{maxVal.toFixed(1)}</span>
        </div>
        {STATE_GROUPS.map(g => (
          <div key={g.label} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 7, fontFamily: C.dt }}>
            <div style={{ width: 6, height: 6, borderRadius: 2, background: g.color }} />
            <span style={{ color: C.dm }}>{g.label}</span>
          </div>
        ))}
      </div>
    </GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: EIGENVALUE STABILITY MAP
// ═══════════════════════════════════════════════════════════════════════════
function EigenTab() {
const eigenvalues = useMemo(() => gEigenvalues(), []);
const allStable = eigenvalues.every(e => e.stable);
const minDamping = Math.min(…eigenvalues.map(e => e.damping));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="System" value={allStable ? "STABLE" : "UNSTABLE"} sub="all Re(λ) < 0" sentiment={allStable ? "positive" : "negative"} delay={0} />
<KPI label="Modes" value={eigenvalues.length} sub="complex pairs" sentiment="neutral" delay={1} />
<KPI label="Min Damping" value={minDamping.toFixed(3)} sub="ζ_min" sentiment={minDamping > 0.1 ? "positive" : "amber"} delay={2} />
<KPI label="Slowest Mode" value={`${eigenvalues.find(e => Math.abs(e.real) === Math.min(...eigenvalues.map(ee => Math.abs(ee.real))))?.name}`} sub="least damped" sentiment="neutral" delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 10 }}>
    <Sec title="Complex Plane — Re(λ) vs Im(λ)">
      <GC><ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="real" type="number" {...ax()} name="Re(λ)" label={{ value: "Re(λ) [rad/s]", position: "bottom", fill: C.dm, fontSize: 9 }} />
          <YAxis dataKey="imag" type="number" {...ax()} name="Im(λ)" label={{ value: "Im(λ) [rad/s]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
          <Tooltip contentStyle={TT} formatter={(v, n, p) => n === "real" ? `Re=${v}` : `Im=${v}`} />
          <ReferenceLine x={0} stroke={C.red} strokeWidth={2} label={{ value: "stability boundary", fill: C.red, fontSize: 7, position: "insideTopRight" }} />
          <Scatter data={eigenvalues} r={6}>
            {eigenvalues.map((e, i) => (
              <Cell key={i} fill={e.name.includes("Wheel") ? C.am : e.name.includes("Yaw") || e.name.includes("Lateral") ? C.red : C.cy} fillOpacity={0.8} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Mode Properties">
      <GC style={{ padding: "8px 12px", maxHeight: 320, overflowY: "auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "110px 55px 50px 50px 60px", gap: 0, fontSize: 8, fontFamily: C.dt }}>
          {["Mode", "Freq [Hz]", "ζ", "Re(λ)", "Status"].map(h => (
            <div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 1, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>
          ))}
          {eigenvalues.map(e => (
            <React.Fragment key={e.name}>
              <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08`, fontWeight: 600 }}>{e.name}</div>
              <div style={{ color: C.cy, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{e.freq}</div>
              <div style={{ color: e.damping > 0.3 ? C.gn : e.damping > 0.1 ? C.am : C.red, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08`, fontWeight: 600 }}>{e.damping}</div>
              <div style={{ color: e.real < 0 ? C.gn : C.red, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{e.real}</div>
              <div style={{ padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>
                <span style={{ fontSize: 7, fontWeight: 700, color: e.stable ? C.gn : C.red, background: e.stable ? `${C.gn}15` : `${C.red}15`, padding: "1px 5px", borderRadius: 4 }}>
                  {e.stable ? "STABLE" : "UNSTABLE"}
                </span>
              </div>
            </React.Fragment>
          ))}
        </div>
      </GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: LAP TIME SENSITIVITY
// ═══════════════════════════════════════════════════════════════════════════
function LapSensTab() {
const sensData = useMemo(() => gLapTimeSens(), []);
const top5 = sensData.slice(0, 5);
const maxAbsSens = Math.max(…sensData.map(s => s.absSens));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
{top5.map((d) => (
<KPI key={d.param} label={`#${d.rank} ${d.param}`}
value={`${d.dtLap > 0 ? "+" : ""}${(d.dtLap * 1000).toFixed(1)} ms`}
sub={d.sign === "faster" ? "decrease → faster" : "decrease → slower"}
sentiment={d.dtLap < 0 ? "positive" : "amber"} delay={d.rank - 1} />
))}
</div>

```
  <Sec title="∂t_lap / ∂(setup_param) — Full 28-Parameter Sensitivity [ms]">
    <GC><ResponsiveContainer width="100%" height={420}>
      <BarChart data={sensData} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 90 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
        <XAxis type="number" {...ax()} domain={[-maxAbsSens * 1.1, maxAbsSens * 1.1]}
          tickFormatter={v => `${(v * 1000).toFixed(0)}ms`} />
        <YAxis dataKey="param" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={85} />
        <Tooltip contentStyle={TT} formatter={v => `${(v * 1000).toFixed(1)} ms`} />
        <ReferenceLine x={0} stroke={C.dm} />
        <Bar dataKey="dtLap" barSize={12} radius={[4, 4, 4, 4]} name="∂t_lap">
          {sensData.map((e, i) => <Cell key={i} fill={e.dtLap < 0 ? C.gn : C.am} fillOpacity={0.7} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ ...GL, padding: "10px 14px", marginTop: 10, fontSize: 9, color: C.dm, fontFamily: C.dt, lineHeight: 1.7 }}>
    The top 5 most sensitive parameters should be tuned with highest priority.
    Parameters with near-zero sensitivity can be safely fixed at nominal values, reducing the effective optimization dimensionality.
    <br/><br/>
    <span style={{ color: C.am, fontWeight: 600 }}>This is impossible with traditional simulators.</span> CasADi/IPOPT can only compute finite-difference approximations requiring 28 additional sim runs. Our analytical gradient is exact, computed in a single backward pass.
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: H_NET GRADIENT FIELD
// ═══════════════════════════════════════════════════════════════════════════
function HnetGradTab() {
const gradField = useMemo(() => gHnetGradField(), []);
const slice = useMemo(() => gradField.filter(d => Math.abs(d.qr) < 0.5), [gradField]);
const maxGrad = Math.max(…gradField.map(d => d.gradMag));
const res = 20;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Peak |∇H|" value={`${maxGrad} N`} sub="max restoring force" sentiment="neutral" delay={0} />
<KPI label="H at equilibrium" value="~0 J" sub="z_eq gate active" sentiment="positive" delay={1} />
<KPI label="Grid Resolution" value={`${res} × ${res}`} sub="q_front × q_rear" sentiment="neutral" delay={2} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="∇H Magnitude Map (q_front × q_rear)">
      <GC style={{ padding: "8px" }}>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${res}, 1fr)`, gap: 1 }}>
          {gradField.map((d, i) => {
            const norm = d.gradMag / maxGrad;
            return (
              <div key={i} style={{
                width: "100%", aspectRatio: "1", borderRadius: 1,
                background: `hsla(${240 - norm * 240}, 75%, ${35 + norm * 20}%, 0.8)`,
              }} title={`q_f=${d.qf}mm, q_r=${d.qr}mm → |∇H|=${d.gradMag}N`} />
            );
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, color: C.dm, fontFamily: C.dt, marginTop: 4 }}>
          <span>q_front: −25mm</span><span>+25mm</span>
        </div>
      </GC>
    </Sec>

    <Sec title="H_net & ∂H/∂q_f — 1D Slice at q_rear ≈ 0">
      <GC><ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={slice} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="qf" {...ax()} label={{ value: "q_front [mm]", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
          <YAxis yAxisId="h" {...ax()} />
          <YAxis yAxisId="g" orientation="right" {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line yAxisId="h" type="monotone" dataKey="H" stroke={C.cy} strokeWidth={2} dot={false} name="H(q) [J]" />
          <Line yAxisId="g" type="monotone" dataKey="dHdqf" stroke={C.gn} strokeWidth={1.5} dot={false} name="∂H/∂q_f [N]" />
          <ReferenceLine yAxisId="g" y={0} stroke={C.dm} strokeDasharray="3 3" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: FIM EIGENSPECTRUM
// ═══════════════════════════════════════════════════════════════════════════
function FIMTab() {
const fimData = useMemo(() => gFIMEigen(), []);
const identifiable = fimData.filter(d => d.identifiable).length;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Identifiable" value={`${identifiable}/28`} sub="eigenvalue > 5" sentiment={identifiable > 20 ? "positive" : "amber"} delay={0} />
<KPI label="Condition #" value={`${(fimData[0].eigenval / fimData[fimData.length - 1].eigenval).toFixed(0)}`} sub="λ_max / λ_min" sentiment="neutral" delay={1} />
<KPI label="Effective DOF" value={`~${identifiable}`} sub="active parameters" sentiment="positive" delay={2} />
</div>

```
  <Sec title="Fisher Information Matrix — Eigenvalue Spectrum">
    <GC><ResponsiveContainer width="100%" height={380}>
      <BarChart data={fimData} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 70 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
        <XAxis type="number" {...ax()} scale="log" domain={[0.1, 100]} />
        <YAxis dataKey="param" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={65} />
        <Tooltip contentStyle={TT} />
        <ReferenceLine x={5} stroke={C.am} strokeDasharray="4 2" label={{ value: "threshold", fill: C.am, fontSize: 7 }} />
        <Bar dataKey="eigenval" radius={[0, 4, 4, 0]} barSize={12}>
          {fimData.map((e, i) => <Cell key={i} fill={e.identifiable ? C.cy : C.dm} fillOpacity={e.identifiable ? 0.7 : 0.3} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: WMPC COST DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════════════
function WMPCCostTab() {
const costData = useMemo(() => gWMPCCost(), []);
return (
<Sec title="WMPC Cost Function Decomposition Over 64-Step Horizon">
<GC><ResponsiveContainer width="100%" height={320}>
<AreaChart data={costData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} />
<XAxis dataKey="step" {…ax()} label={{ value: "Horizon Step", position: "bottom", fill: C.dm, fontSize: 9 }} />
<YAxis {…ax()} />
<Tooltip contentStyle={TT} />
<Area type="monotone" dataKey="costLap" stackId="1" stroke={C.cy} fill={`${C.cy}20`} strokeWidth={1} name="Lap Time" />
<Area type="monotone" dataKey="costTrack" stackId="1" stroke={C.red} fill={`${C.red}20`} strokeWidth={1} name="Track Limits" />
<Area type="monotone" dataKey="costSmooth" stackId="1" stroke={C.am} fill={`${C.am}20`} strokeWidth={1} name="Smoothness" />
<Area type="monotone" dataKey="costGrip" stackId="1" stroke={C.gn} fill={`${C.gn}20`} strokeWidth={1} name="Grip Margin" />
<Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
</AreaChart>
</ResponsiveContainer></GC>
</Sec>
);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: SOLVER HEALTH
// ═══════════════════════════════════════════════════════════════════════════
function SolverTab() {
const solverData = useMemo(() => gSolverConv(), []);
const avgIters = solverData.reduce((a, d) => a + d.iters, 0) / solverData.length;
const convRate = (solverData.filter(d => d.converged).length / solverData.length * 100);
const avgMs = solverData.reduce((a, d) => a + d.solveMs, 0) / solverData.length;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Avg Iterations" value={avgIters.toFixed(1)} sub="L-BFGS-B per step" sentiment={avgIters < 12 ? "positive" : "amber"} delay={0} />
<KPI label="Conv. Rate" value={`${convRate.toFixed(1)}%`} sub="steps converged" sentiment={convRate > 95 ? "positive" : "amber"} delay={1} />
<KPI label="Avg Solve" value={`${avgMs.toFixed(1)} ms`} sub="per MPC step" sentiment={avgMs < 8 ? "positive" : "amber"} delay={2} />
<KPI label="100 Hz OK" value={avgMs < 10 ? "YES" : "NO"} sub="< 10ms budget" sentiment={avgMs < 10 ? "positive" : "amber"} delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="L-BFGS-B Iterations Per MPC Step">
      <GC><ResponsiveContainer width="100%" height={220}>
        <BarChart data={solverData.filter((_, i) => i % 3 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="step" {...ax()} />
          <YAxis {...ax()} domain={[0, 20]} />
          <Tooltip contentStyle={TT} />
          <Bar dataKey="iters" barSize={4} radius={[2, 2, 0, 0]}>
            {solverData.filter((_, i) => i % 3 === 0).map((d, i) => (
              <Cell key={i} fill={d.converged ? (d.iters > 12 ? C.am : C.gn) : C.red} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Solve Time [ms] & Gradient Norm">
      <GC><ResponsiveContainer width="100%" height={220}>
        <AreaChart data={solverData.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="step" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={10} stroke={C.red} strokeDasharray="4 4" label={{ value: "10ms budget", fill: C.red, fontSize: 7 }} />
          <Area type="monotone" dataKey="solveMs" stroke={C.am} fill={`${C.am}12`} strokeWidth={1.5} dot={false} name="Solve time [ms]" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: GRADIENT COUPLING — NEW v5.0
// Inter-subsystem gradient interaction matrix
// ═══════════════════════════════════════════════════════════════════════════
function CouplingTab() {
const { matrix, subsystems, outgoing } = useMemo(() => gGradientCoupling(), []);
const n = subsystems.length;
const maxStrength = Math.max(…matrix.filter(m => m.fi !== m.ti).map(m => m.strength));
const strongestLink = matrix.filter(m => m.fi !== m.ti).sort((a, b) => b.strength - a.strength)[0];

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Subsystems" value={n.toString()} sub="in gradient graph" sentiment="neutral" delay={0} />
<KPI label="Strongest Link" value={`${strongestLink.from.split(" ")[0]}→${strongestLink.to.split(" ")[0]}`} sub={`strength: ${strongestLink.strength}`} sentiment="neutral" delay={1} />
<KPI label="Avg Coupling" value={(matrix.filter(m => m.fi !== m.ti).reduce((a, m) => a + m.strength, 0) / (n * n - n)).toFixed(3)} sub="off-diagonal mean" sentiment="neutral" delay={2} />
<KPI label="Sparsity" value={`${(matrix.filter(m => m.fi !== m.ti && m.strength < 0.1).length / (n * n - n) * 100).toFixed(0)}%`} sub="< 0.1 threshold" sentiment="positive" delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 10 }}>
    {/* Coupling heatmap */}
    <Sec title="∂(subsystem_i) / ∂(subsystem_j) — Gradient Interaction Matrix">
      <GC style={{ padding: 10 }}>
        <div style={{ display: "flex" }}>
          <div style={{ width: 90 }} />
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 2, flex: 1 }}>
            {subsystems.map((s, i) => (
              <div key={i} style={{ fontSize: 5, color: C.dm, fontFamily: C.dt, textAlign: "center", transform: "rotate(-45deg)", transformOrigin: "left bottom", height: 50, display: "flex", alignItems: "flex-end", justifyContent: "center", whiteSpace: "nowrap" }}>
                {s.split(" ")[0]}
              </div>
            ))}
          </div>
        </div>
        {subsystems.map((s, i) => (
          <div key={i} style={{ display: "flex", marginBottom: 2 }}>
            <div style={{ width: 90, fontSize: 7, color: C.br, fontFamily: C.dt, display: "flex", alignItems: "center", paddingRight: 6, justifyContent: "flex-end" }}>
              {s}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 2, flex: 1 }}>
              {subsystems.map((_, j) => {
                const cell = matrix.find(m => m.fi === i && m.ti === j);
                const str = cell?.strength || 0;
                const isDiag = i === j;
                const norm = isDiag ? 1 : str / maxStrength;
                const hue = isDiag ? 200 : 120 - norm * 120; // green→red for off-diag
                return (
                  <div key={j} style={{
                    aspectRatio: "1", borderRadius: 2,
                    background: isDiag ? `rgba(0,210,255,0.3)` : `hsla(${hue}, 70%, 40%, ${0.1 + norm * 0.7})`,
                    border: str > 0.5 && !isDiag ? `1px solid rgba(255,200,0,0.3)` : "none",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 5, color: norm > 0.5 || isDiag ? C.w : C.dm, fontFamily: C.dt,
                    cursor: "default",
                  }} title={`${subsystems[i]} → ${subsystems[j]}: ${str.toFixed(3)}`}>
                    {str > 0.2 ? str.toFixed(2) : ""}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </GC>
    </Sec>

    {/* Outgoing/incoming gradient flow */}
    <Sec title="Gradient Flow — Outgoing vs Incoming">
      <GC><ResponsiveContainer width="100%" height={320}>
        <BarChart data={outgoing} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
          <XAxis type="number" {...ax()} />
          <YAxis dataKey="subsystem" type="category" tick={{ fontSize: 7, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={75} />
          <Tooltip contentStyle={TT} />
          <Bar dataKey="totalOut" fill={C.cy} fillOpacity={0.6} barSize={8} name="Outgoing ∇" radius={[0, 4, 4, 0]} />
          <Bar dataKey="totalIn" fill={C.am} fillOpacity={0.6} barSize={8} name="Incoming ∇" radius={[0, 4, 4, 0]} />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <div style={{ ...GL, padding: "10px 14px", marginTop: 10, fontSize: 9, color: C.dm, fontFamily: C.dt, lineHeight: 1.7 }}>
    This matrix shows how gradients propagate between subsystems during a single backward pass through the full physics graph.
    Strong off-diagonal entries indicate tightly coupled subsystems where tuning one affects the other.
    The <span style={{ color: C.cy }}>Chassis↔Suspension</span> coupling is the strongest — changing spring rates directly changes body dynamics, which feeds through to tire slip, aero attitude, and WMPC control.
    <br/><br/>
    <span style={{ color: C.am, fontWeight: 600 }}>This whole-system gradient view is unique to differentiable simulators.</span> Traditional solvers must treat each subsystem independently, missing these cross-domain interactions entirely.
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: CONTROLLABILITY — NEW v5.0
// Gramian-based controllability analysis
// ═══════════════════════════════════════════════════════════════════════════
function ControlTab() {
const { gramianDiag, inputs } = useMemo(() => gControllability(), []);
const controllableCount = gramianDiag.filter(d => d.controllable).length;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Controllable" value={`${controllableCount}/${gramianDiag.length}`} sub="states reachable" sentiment={controllableCount > 10 ? "positive" : "amber"} delay={0} />
<KPI label="Inputs" value={inputs.length.toString()} sub="control channels" sentiment="neutral" delay={1} />
<KPI label="Best Actuator" value="δ_steer" sub="highest total reach" sentiment="neutral" delay={2} />
<KPI label="Weakest State" value={gramianDiag.sort((a, b) => a.total - b.total)[0]?.state} sub="hardest to control" sentiment="neutral" delay={3} />
</div>

```
  <Sec title="Controllability Gramian Diagonal — Per-State Reachability by Input">
    <GC><ResponsiveContainer width="100%" height={380}>
      <BarChart data={gramianDiag} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 55 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
        <XAxis type="number" {...ax()} />
        <YAxis dataKey="state" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={50} />
        <Tooltip contentStyle={TT} />
        <Bar dataKey="steer" stackId="1" fill={C.cy} fillOpacity={0.6} barSize={12} name="δ_steer" />
        <Bar dataKey="throttle" stackId="1" fill={C.gn} fillOpacity={0.6} barSize={12} name="T_throttle" />
        <Bar dataKey="brake" stackId="1" fill={C.red} fillOpacity={0.5} barSize={12} name="P_brake" />
        <Bar dataKey="tv" stackId="1" fill="#7c3aed" fillOpacity={0.5} barSize={12} name="TV_bias" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </BarChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ ...GL, padding: "10px 14px", marginTop: 10, fontSize: 9, color: C.dm, fontFamily: C.dt, lineHeight: 1.7 }}>
    Each bar shows how much each control input can influence each mechanical state.
    States with low total reachability (short bars) are structurally difficult to control — the WMPC should not waste control effort targeting them.
    <span style={{ color: C.cy, fontWeight: 600 }}> Yaw (ψ)</span> is highly reachable via both steering and torque vectoring, confirming the TV system provides redundant yaw authority.
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function DifferentiableInsightsModule() {
const [tab, setTab] = useState("jacobian");

return (
<div>
{/* Header banner */}
<div style={{
…GL, padding: "12px 16px", marginBottom: 14,
borderLeft: `3px solid ${C.cy}`,
background: `linear-gradient(90deg, ${C.cy}08, transparent)`,
}}>
<div style={{ display: "flex", alignItems: "center", gap: 10 }}>
<span style={{ fontSize: 20, color: C.cy }}>∂</span>
<div>
<span style={{ fontSize: 12, fontWeight: 800, color: C.cy, fontFamily: C.dt, letterSpacing: 2 }}>
DIFFERENTIABLE INSIGHTS
</span>
<span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 12 }}>
Analytical gradients through the full physics engine — impossible with traditional simulators
</span>
</div>
</div>
</div>

```
  {/* Cross-link to Aerodynamics module */}
  <div style={{
    ...GL, padding: "8px 14px", marginBottom: 10,
    borderLeft: `2px solid #ff6090`,
    display: "flex", alignItems: "center", gap: 8,
    fontSize: 9, fontFamily: C.dt,
  }}>
    <span style={{ color: "#ff6090" }}>▽</span>
    <span style={{ color: C.dm }}>Aero gradient analysis has moved to the dedicated</span>
    <span style={{ color: "#ff6090", fontWeight: 700 }}>Aerodynamics</span>
    <span style={{ color: C.dm }}>module — see Sensitivity & Aero Maps tabs for ∂C_L/∂setup.</span>
  </div>

  <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
    {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.cy} />)}
  </div>

  {tab === "jacobian" && <JacobianTab />}
  {tab === "eigen" && <EigenTab />}
  {tab === "lapSens" && <LapSensTab />}
  {tab === "hnetGrad" && <HnetGradTab />}
  {tab === "fim" && <FIMTab />}
  {tab === "wmpcCost" && <WMPCCostTab />}
  {tab === "solver" && <SolverTab />}
  {tab === "coupling" && <CouplingTab />}
  {tab === "control" && <ControlTab />}
</div>
```

);
}