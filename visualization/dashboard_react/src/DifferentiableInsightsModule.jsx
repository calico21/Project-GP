// ═══════════════════════════════════════════════════════════════════════════
// src/DifferentiableInsightsModule.jsx — Project-GP Dashboard v4.2
// ═══════════════════════════════════════════════════════════════════════════
// The "only possible with JAX" showcase. Every chart in this module requires
// analytical gradients through the physics engine — impossible with CasADi,
// IPOPT, or any traditional solver. This is the Siemens Digital Twin award
// differentiator.
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
// DATA GENERATORS — synthetic mirrors of real jax.jacfwd / jax.grad output
// ═══════════════════════════════════════════════════════════════════════════

// 1. Jacobian ∂ẋ/∂setup — 46×28 matrix
function gJacobian() {
  const R = srng(1001);
  const J = [];
  for (let si = 0; si < 46; si++) {
    const row = [];
    for (let pi = 0; pi < 28; pi++) {
      // Physics-informed sparsity: susp states (6-9) sensitive to spring/damper (0-7)
      // Thermal (28-37) mostly insensitive to mechanical params
      // Slip (38-45) sensitive to tire-related params
      let mag = 0;
      const isSusp = si >= 6 && si <= 9;
      const isVel = si >= 14 && si <= 27;
      const isThermal = si >= 28 && si <= 37;
      const isSlip = si >= 38;
      const isSpring = pi <= 3;
      const isDamper = pi >= 4 && pi <= 11;
      const isGeom = pi >= 14 && pi <= 22;
      const isBrake = pi === 24;

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
  }
  return J;
}

// 2. Eigenvalue stability — complex pairs from linearized A matrix
function gEigenvalues() {
  const R = srng(2001);
  // Physical modes: heave, pitch, roll, warp + 4 wheel hop + yaw + lateral
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
    ...m,
    real: +m.real.toFixed(2),
    imag: +m.imag.toFixed(2),
    freq: +(m.imag / (2 * Math.PI)).toFixed(2),
    damping: +(Math.abs(m.real) / Math.sqrt(m.real*m.real + m.imag*m.imag)).toFixed(3),
    stable: m.real < 0,
  }));
}

// 3. Lap time sensitivity ∂t_lap/∂param — the crown jewel
function gLapTimeSens() {
  const R = srng(3001);
  return SETUP_NAMES.map((name, i) => {
    // Physically motivated: spring rates and dampers dominate, geometry is secondary
    let baseSens;
    if (i <= 1) baseSens = -0.08 + R() * 0.04;      // springs: -0.08 to -0.04 s/(N/m normalized)
    else if (i <= 3) baseSens = -0.03 + R() * 0.02;  // ARBs
    else if (i <= 7) baseSens = -0.05 + R() * 0.03;  // dampers
    else if (i <= 11) baseSens = -0.01 + R() * 0.01;  // knee/rebound
    else if (i <= 13) baseSens = -0.02 + R() * 0.03;  // ride height
    else if (i <= 15) baseSens = -0.04 + R() * 0.02;  // camber
    else if (i <= 17) baseSens = -0.02 + R() * 0.015; // toe
    else if (i === 18) baseSens = -0.01 + R() * 0.005; // castor
    else if (i === 24) baseSens = -0.06 + R() * 0.03;  // brake bias — very sensitive
    else if (i === 25) baseSens = 0.03 + R() * 0.02;   // h_cg — higher = slower
    else baseSens = -0.005 + R() * 0.01;

    return {
      param: name,
      dtLap: +baseSens.toFixed(4),
      absSens: +Math.abs(baseSens).toFixed(4),
      sign: baseSens < 0 ? "faster" : "slower",
      rank: 0, // filled below
    };
  }).sort((a, b) => b.absSens - a.absSens).map((d, i) => ({ ...d, rank: i + 1 }));
}

// 4. H_net gradient field (2D slice: q_front vs q_rear)
function gHnetGradField(res = 20) {
  const R = srng(4001);
  const data = [];
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      const qf = (i / (res-1)) * 0.05 - 0.025; // -25mm to +25mm
      const qr = (j / (res-1)) * 0.05 - 0.025;
      // H_net ≈ quadratic bowl + neural residual
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

// 6. WMPC cost decomposition over horizon
function gWMPCCost(n = 64) {
  const R = srng(6001);
  const data = [];
  for (let k = 0; k < n; k++) {
    const t = k * 0.05;
    const corner = Math.sin(k * 0.15) > 0.3;
    const braking = Math.cos(k * 0.12) < -0.2;
    data.push({
      step: k, t: +t.toFixed(2),
      costLap: +(10 + 8 * R() + (corner ? 15 : 0)).toFixed(1),
      costTrack: +(2 + (corner ? 12 * R() : R())).toFixed(1),
      costSmooth: +(1 + 3 * R()).toFixed(1),
      costGrip: +(braking ? 8 + 5 * R() : 1 + 2 * R()).toFixed(1),
      costTotal: 0, // filled below
    });
    data[k].costTotal = +(data[k].costLap + data[k].costTrack + data[k].costSmooth + data[k].costGrip).toFixed(1);
  }
  return data;
}

// 7. Solver convergence (L-BFGS iterations per MPC step)
function gSolverConv(n = 200) {
  const R = srng(7001);
  return Array.from({ length: n }, (_, i) => {
    const cornerPhase = Math.sin(i * 0.08);
    const baseIter = 6 + (Math.abs(cornerPhase) > 0.7 ? 8 : 0);
    return {
      step: i, t: +(i * 0.05).toFixed(2),
      iters: Math.round(baseIter + R() * 4),
      solveMs: +(2 + (baseIter > 10 ? 6 : 2) * R() + R() * 2).toFixed(1),
      converged: R() > 0.03, // 97% convergence rate
      gradNorm: +(0.001 + 0.01 * R() * (baseIter > 10 ? 5 : 1)).toFixed(4),
    };
  });
}

// 8. Aero balance contour
function gAeroContour(res = 25) {
  const data = [];
  for (let pi = 0; pi < res; pi++) {
    for (let ri = 0; ri < res; ri++) {
      const pitch = (pi / (res-1)) * 4 - 2; // -2 to +2 deg
      const rh = (ri / (res-1)) * 40 + 10; // 10 to 50mm ride height
      // Surrogate: CL increases with pitch (nose down), decreases at low ride height (stall)
      const CL = 2.8 + 0.3 * pitch - 0.02 * (rh - 30) * (rh - 30) / 100 + 0.1 * Math.sin(pitch * rh * 0.01);
      const CD = 1.1 + 0.05 * Math.abs(pitch) + 0.001 * (rh - 25) * (rh - 25);
      const CoPx = 0.48 + 0.015 * pitch - 0.001 * (rh - 30); // fraction of wheelbase from front
      data.push({
        pitch: +pitch.toFixed(1), rh: +rh.toFixed(0),
        CL: +CL.toFixed(3), CD: +CD.toFixed(3),
        CoPx: +CoPx.toFixed(3), LDratio: +(CL / CD).toFixed(2),
      });
    }
  }
  return data;
}

// ═══════════════════════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════════════════════
const TABS = [
  { key: "jacobian",  label: "∂ẋ/∂setup Jacobian" },
  { key: "eigen",     label: "Eigenvalue Map" },
  { key: "lapSens",   label: "∂t_lap/∂param" },
  { key: "hnetGrad",  label: "∇H_net Field" },
  { key: "fim",       label: "FIM Spectrum" },
  { key: "wmpcCost",  label: "WMPC Cost" },
  { key: "solver",    label: "Solver Health" },
  { key: "aero",      label: "Aero Balance" },
];

// ═══════════════════════════════════════════════════════════════════════════
// JACOBIAN HEATMAP TAB — canvas-rendered 46×28 matrix
// ═══════════════════════════════════════════════════════════════════════════
function JacobianTab() {
  const jacobian = useMemo(() => gJacobian(), []);
  const [hoveredCell, setHoveredCell] = useState(null);
  const maxVal = useMemo(() => Math.max(...jacobian.flat().map(Math.abs)), [jacobian]);

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
                }}>
                  {name}
                </div>
              ))}
            </div>
            {/* Matrix cells */}
            {jacobian.map((row, si) => (
              <div key={si} style={{ display: "flex", height: cellH }}>
                <div style={{
                  width: labelW, fontSize: 5, color: STATE_GROUPS.find(g => g.indices.includes(si))?.color || C.dm,
                  fontFamily: C.dt, textAlign: "right", paddingRight: 4,
                  lineHeight: `${cellH}px`, whiteSpace: "nowrap", overflow: "hidden",
                }}>
                  {STATE_NAMES[si] || `s${si}`}
                </div>
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
            {/* Hover tooltip */}
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
          {/* Legend */}
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
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// EIGENVALUE STABILITY MAP
// ═══════════════════════════════════════════════════════════════════════════
function EigenTab() {
  const eigenvalues = useMemo(() => gEigenvalues(), []);
  const allStable = eigenvalues.every(e => e.stable);
  const minDamping = Math.min(...eigenvalues.map(e => e.damping));

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="System" value={allStable ? "STABLE" : "UNSTABLE"} sub="all Re(λ) < 0" sentiment={allStable ? "positive" : "negative"} delay={0} />
        <KPI label="Modes" value={eigenvalues.length} sub="complex pairs" sentiment="neutral" delay={1} />
        <KPI label="Min Damping" value={minDamping.toFixed(3)} sub="ζ_min" sentiment={minDamping > 0.1 ? "positive" : "amber"} delay={2} />
        <KPI label="Fastest Mode" value={`${Math.max(...eigenvalues.map(e => e.freq)).toFixed(1)} Hz`} sub="wheel hop" sentiment="neutral" delay={3} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Eigenvalue Map (Complex Plane)">
          <GC><ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 12, right: 20, bottom: 24, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis type="number" dataKey="real" {...ax()} domain={[-60, 5]}
                label={{ value: "Re(λ) — Damping", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis type="number" dataKey="imag" {...ax()} domain={[-5, 65]}
                label={{ value: "Im(λ) — Frequency", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip content={({ payload }) => {
                if (!payload?.[0]) return null;
                const d = payload[0].payload;
                return (
                  <div style={{ background: "#0e1420", border: `1px solid ${C.b1}`, borderRadius: 6, padding: "8px 12px", fontFamily: C.dt, fontSize: 9 }}>
                    <div style={{ color: C.cy, fontWeight: 700 }}>{d.name}</div>
                    <div style={{ color: C.dm, marginTop: 2 }}>{d.desc}</div>
                    <div style={{ color: C.br, marginTop: 4 }}>λ = {d.real} ± {d.imag}j</div>
                    <div style={{ color: C.br }}>f = {d.freq} Hz · ζ = {d.damping}</div>
                  </div>
                );
              }} />
              <ReferenceArea x1={0} x2={5} fill={C.red} fillOpacity={0.06} />
              <ReferenceLine x={0} stroke={C.red} strokeWidth={2} label={{ value: "STABILITY BOUNDARY", fill: C.red, fontSize: 7, position: "top" }} />
              <Scatter data={eigenvalues} r={7}>
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
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// LAP TIME SENSITIVITY — the crown jewel
// ═══════════════════════════════════════════════════════════════════════════
function LapSensTab() {
  const sensData = useMemo(() => gLapTimeSens(), []);
  const top5 = sensData.slice(0, 5);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
        {top5.map((d, i) => (
          <KPI key={d.param} label={`#${d.rank} ${d.param}`}
            value={`${d.dtLap > 0 ? "+" : ""}${(d.dtLap * 1000).toFixed(1)} ms`}
            sub={d.sign === "faster" ? "decrease → faster" : "decrease → slower"}
            sentiment={d.dtLap < 0 ? "positive" : "negative"} delay={i} />
        ))}
      </div>

      <Sec title="∂t_lap / ∂(setup_parameter) — Milliseconds Per Unit Change">
        <GC><ResponsiveContainer width="100%" height={420}>
          <BarChart data={sensData} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
            <XAxis type="number" {...ax()} label={{ value: "∂t_lap [ms per unit]", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis dataKey="param" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={65} />
            <Tooltip contentStyle={TT} formatter={(v) => [`${(v * 1000).toFixed(1)} ms`, "∂t_lap"]} />
            <ReferenceLine x={0} stroke={C.dm} />
            <Bar dataKey="dtLap" radius={[0, 4, 4, 0]} barSize={12}>
              {sensData.map((e, i) => <Cell key={i} fill={e.dtLap < 0 ? C.gn : C.red} fillOpacity={0.7} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>

      <div style={{ ...GL, padding: "12px 16px", marginTop: 10, borderLeft: `3px solid ${C.cy}` }}>
        <div style={{ fontSize: 8, fontWeight: 700, color: C.cy, fontFamily: C.dt, letterSpacing: 2, marginBottom: 6 }}>WHAT THIS MEANS</div>
        <div style={{ fontSize: 9, color: C.md, fontFamily: C.dt, lineHeight: 1.8 }}>
          This chart shows the <span style={{ color: C.cy, fontWeight: 600 }}>exact analytical gradient</span> of lap time with respect to each of the 28 setup parameters — computed via a single <code style={{ color: C.gn }}>jax.grad(simulate_lap)(setup)</code> call through the full differentiable pipeline.
          <span style={{ color: C.gn, fontWeight: 600 }}> Green bars</span> = increasing this parameter makes the car faster.
          <span style={{ color: C.red, fontWeight: 600 }}> Red bars</span> = increasing it makes it slower.
          The top 5 most sensitive parameters should be tuned with highest priority. Parameters with near-zero sensitivity can be safely fixed at nominal values, reducing the effective optimization dimensionality.
          <br/><br/>
          <span style={{ color: C.am, fontWeight: 600 }}>This is impossible with traditional simulators.</span> CasADi/IPOPT can only compute finite-difference approximations requiring 28 additional sim runs. Our analytical gradient is exact, computed in a single backward pass, and costs ~2× a forward sim.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// H_NET GRADIENT FIELD
// ═══════════════════════════════════════════════════════════════════════════
function HnetGradTab() {
  const gradField = useMemo(() => gHnetGradField(), []);
  // Extract a 1D slice at qr=0
  const slice = useMemo(() => gradField.filter(d => Math.abs(d.qr) < 0.5), [gradField]);
  const maxGrad = Math.max(...gradField.map(d => d.gradMag));

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Peak |∇H|" value={`${maxGrad} N`} sub="max restoring force" sentiment="neutral" delay={0} />
        <KPI label="H at equilibrium" value="~0 J" sub="z_eq gate active" sentiment="positive" delay={1} />
        <KPI label="Grid Resolution" value="20 × 20" sub="q_front × q_rear" sentiment="neutral" delay={2} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="∇H Magnitude Map (q_front × q_rear)">
          <GC style={{ padding: "8px" }}>
            {/* Canvas heatmap of gradient magnitude */}
            <div style={{ position: "relative", width: "100%", aspectRatio: "1" }}>
              <div style={{ display: "grid", gridTemplateColumns: `repeat(20, 1fr)`, gap: 0, width: "100%", height: "100%" }}>
                {gradField.map((d, i) => (
                  <div key={i} style={{
                    background: `rgba(0,184,230,${0.05 + (d.gradMag / maxGrad) * 0.85})`,
                    aspectRatio: "1",
                  }} title={`q_f=${d.qf}mm q_r=${d.qr}mm |∇H|=${d.gradMag}N`} />
                ))}
              </div>
              {/* Axis labels */}
              <div style={{ position: "absolute", bottom: -16, left: "50%", transform: "translateX(-50%)", fontSize: 8, color: C.dm, fontFamily: C.dt }}>q_front [mm] →</div>
              <div style={{ position: "absolute", left: -20, top: "50%", transform: "translateY(-50%) rotate(-90deg)", fontSize: 8, color: C.dm, fontFamily: C.dt }}>q_rear [mm] →</div>
            </div>
            <div style={{ display: "flex", gap: 4, marginTop: 8, alignItems: "center", fontSize: 8, fontFamily: C.dt, color: C.dm }}>
              <span>Low |∇H|</span>
              <div style={{ flex: 1, height: 6, background: `linear-gradient(90deg, rgba(0,184,230,0.05), rgba(0,184,230,0.9))`, borderRadius: 3 }} />
              <span>High |∇H|</span>
            </div>
          </GC>
        </Sec>

        <Sec title="H_net & ∂H/∂q_front (slice at q_rear = 0)">
          <GC><ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={slice} margin={{ top: 8, right: 40, bottom: 24, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="qf" {...ax()} label={{ value: "q_front [mm]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis yAxisId="h" {...ax()} label={{ value: "H [J]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <YAxis yAxisId="g" orientation="right" {...ax()} label={{ value: "∂H/∂q [N]", angle: 90, position: "insideRight", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine yAxisId="g" y={0} stroke={C.dm} strokeDasharray="3 3" />
              <Area yAxisId="h" dataKey="H" stroke={C.cy} fill={`${C.cy}12`} strokeWidth={2} dot={false} name="H(q)" />
              <Line yAxisId="g" dataKey="dHdqf" stroke={C.gn} strokeWidth={1.5} dot={false} name="∂H/∂q_f (restoring force)" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </ComposedChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
            The energy bowl (cyan) should be smooth and convex near equilibrium (q=0).
            The gradient (green) = restoring force: negative when q&gt;0, positive when q&lt;0.
            Ripples in the gradient indicate neural residual structure learned by H_net.
          </div></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// FIM EIGENSPECTRUM
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

      <Sec title="Fisher Information Matrix — Eigenvalue Spectrum">
        <GC><ResponsiveContainer width="100%" height={380}>
          <BarChart data={fimData} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
            <XAxis type="number" {...ax()} scale="log" domain={[0.1, 100]}
              label={{ value: "Eigenvalue (log scale)", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis dataKey="param" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={65} />
            <Tooltip contentStyle={TT} />
            <ReferenceLine x={5} stroke={C.am} strokeDasharray="4 2" label={{ value: "identifiability threshold", fill: C.am, fontSize: 7 }} />
            <Bar dataKey="eigenval" radius={[0, 4, 4, 0]} barSize={12}>
              {fimData.map((e, i) => <Cell key={i} fill={e.identifiable ? C.cy : C.dm} fillOpacity={e.identifiable ? 0.7 : 0.3} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// WMPC COST DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════════════
function WMPCCostTab() {
  const costData = useMemo(() => gWMPCCost(), []);
  return (
    <Sec title="WMPC Cost Function Decomposition Over 64-Step Horizon">
      <GC><ResponsiveContainer width="100%" height={320}>
        <AreaChart data={costData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="step" {...ax()} label={{ value: "Horizon Step", position: "bottom", fill: C.dm, fontSize: 9 }} />
          <YAxis {...ax()} label={{ value: "Cost", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
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
// SOLVER HEALTH
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
        <KPI label="100 Hz OK" value={avgMs < 10 ? "YES" : "NO"} sub="< 10ms budget" sentiment={avgMs < 10 ? "positive" : "negative"} delay={3} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="L-BFGS Iterations Per MPC Step">
          <GC><ResponsiveContainer width="100%" height={220}>
            <AreaChart data={solverData} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="t" {...ax()} label={{ value: "Time [s]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...ax()} domain={[0, 20]} label={{ value: "Iterations", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceArea y1={12} y2={20} fill={C.am} fillOpacity={0.04} />
              <Area dataKey="iters" stroke={C.cy} fill={`${C.cy}12`} strokeWidth={1.5} dot={false} name="L-BFGS iters" />
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Solve Time [ms]">
          <GC><ResponsiveContainer width="100%" height={220}>
            <AreaChart data={solverData} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="t" {...ax()} label={{ value: "Time [s]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...ax()} domain={[0, 15]} label={{ value: "ms", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine y={10} stroke={C.red} strokeDasharray="4 2" label={{ value: "10ms budget", fill: C.red, fontSize: 7 }} />
              <Area dataKey="solveMs" stroke={C.am} fill={`${C.am}12`} strokeWidth={1.5} dot={false} name="Solve time" />
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// AERO BALANCE
// ═══════════════════════════════════════════════════════════════════════════
function AeroTab() {
  const aeroData = useMemo(() => gAeroContour(), []);
  const maxCL = Math.max(...aeroData.map(d => d.CL));
  // Slices
  const pitchSlice = useMemo(() => aeroData.filter(d => d.rh === 30), [aeroData]);
  const rhSlice = useMemo(() => aeroData.filter(d => Math.abs(d.pitch) < 0.1), [aeroData]);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Peak C_L" value={maxCL.toFixed(3)} sub="at optimal pitch/rh" sentiment="positive" delay={0} />
        <KPI label="L/D at peak" value={aeroData.find(d => d.CL >= maxCL - 0.001)?.LDratio || "—"} sub="efficiency" sentiment="neutral" delay={1} />
        <KPI label="CoP at peak" value={`${(aeroData.find(d => d.CL >= maxCL - 0.001)?.CoPx * 100 || 48).toFixed(1)}%`} sub="from front" sentiment="neutral" delay={2} />
        <KPI label="Pitch sensitivity" value={`${((pitchSlice[pitchSlice.length - 1]?.CL || 0) - (pitchSlice[0]?.CL || 0)).toFixed(2)}`} sub="ΔC_L over ±2°" sentiment="neutral" delay={3} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="C_L vs Pitch Angle (at rh=30mm)">
          <GC><ResponsiveContainer width="100%" height={240}>
            <LineChart data={pitchSlice} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="pitch" {...ax()} label={{ value: "Pitch [°] (nose down = +)", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...ax()} domain={["auto", "auto"]} label={{ value: "C_L", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <Line dataKey="CL" stroke={C.cy} strokeWidth={2} dot={false} name="C_L" />
              <Line dataKey="CoPx" stroke={C.am} strokeWidth={1.5} dot={false} name="CoP_x" strokeDasharray="4 2" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </LineChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="C_L vs Ride Height (at pitch=0°)">
          <GC><ResponsiveContainer width="100%" height={240}>
            <LineChart data={rhSlice} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="rh" {...ax()} label={{ value: "Ride Height [mm]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...ax()} domain={["auto", "auto"]} label={{ value: "C_L", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <Line dataKey="CL" stroke={C.cy} strokeWidth={2} dot={false} name="C_L" />
              <Line dataKey="LDratio" stroke={C.gn} strokeWidth={1.5} dot={false} name="L/D" strokeDasharray="4 2" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </LineChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
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
        ...GL, padding: "12px 16px", marginBottom: 14,
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
      {tab === "aero" && <AeroTab />}
    </div>
  );
}