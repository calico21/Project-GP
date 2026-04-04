// ═══════════════════════════════════════════════════════════════════════════
// src/PowertrainControlModule.jsx — Project-GP Dashboard v5.1
// ═══════════════════════════════════════════════════════════════════════════
// Dedicated powertrain control visualization hub for the Ter27 4WD stack.
// Covers every subsystem in the powertrain_manager pipeline.
//
// Sub-tabs (8):
//   1. Allocation    — Per-wheel torque distribution, friction circle util
//   2. Yaw Control   — ψ̇ reference tracking, Mz demand, counter-steer
//   3. Traction      — DESC convergence, κ* tracking, combined-slip ellipse
//   4. Launch        — State machine phases, B-spline profile, μ probe
//   5. CBF Safety    — Barrier values, intervention events, safe set
//   6. Impedance     — Raw vs filtered pedals, frequency response
//   7. Motor Thermal — 4-motor temps, SoC, derating, power budget
//   8. Diagnostics   — Sensor confidence, degradation, timing, cost
//
// Integration (3 lines in App.jsx):
//   NAV: { key: "powertrain", label: "Powertrain Control", icon: "⚙" }
//   Import: import PowertrainControlModule from "./PowertrainControlModule.jsx"
//   Route: case "powertrain": return <PowertrainControlModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
  ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import { C, GL, GS, TT, AX } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const PT = "#f97316"; // powertrain accent (orange)
const PT_G = "rgba(249,115,22,0.10)";
const CBF_C = "#ef4444";
const DESC_C = "#22d3ee";
const LAUNCH_C = "#a855f7";
const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });
const tt = () => ({ contentStyle: { background: C.panel, border: `1px solid ${C.b1}`, borderRadius: 4, fontSize: 9, padding: "6px 10px" }, labelStyle: { fontSize: 8, color: C.dm } });const CORNERS = ["FL", "FR", "RL", "RR"];
const CORNER_C = [C.cy, C.gn, C.am, C.red];

const TABS = [
  { key: "alloc",    label: "Allocation" },
  { key: "yaw",      label: "Yaw Control" },
  { key: "traction", label: "Traction" },
  { key: "launch",   label: "Launch" },
  { key: "cbf",      label: "CBF Safety" },
  { key: "impedance",label: "Impedance" },
  { key: "thermal",  label: "Motor Thermal" },
  { key: "diag",     label: "Diagnostics" },
];

// ═══════════════════════════════════════════════════════════════════════════
// SEEDED RNG
// ═══════════════════════════════════════════════════════════════════════════
function srng(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; };
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

function genAllocData(n = 300) {
  const r = srng(42);
  const d = [];
  for (let i = 0; i < n; i++) {
    const t = i * 0.02;
    const phase = t * 0.8;
    const ay = 1.3 * Math.sin(phase) + 0.3 * Math.sin(phase * 2.7);
    const ax = 0.5 * Math.cos(phase * 0.6) + r() * 0.1;
    const base = 120 + 80 * ax;
    const yawSplit = 40 * ay;
    d.push({
      t: +t.toFixed(2),
      T_fl: +(base - yawSplit * 0.6 + r() * 8).toFixed(1),
      T_fr: +(base + yawSplit * 0.6 + r() * 8).toFixed(1),
      T_rl: +(base * 1.1 - yawSplit * 0.4 + r() * 8).toFixed(1),
      T_rr: +(base * 1.1 + yawSplit * 0.4 + r() * 8).toFixed(1),
      Fx_target: +(ax * 2500 + r() * 50).toFixed(0),
      Fx_actual: +(ax * 2500 + r() * 80 - 15).toFixed(0),
      util_fl: +(0.4 + 0.3 * Math.abs(ay) + r() * 0.08).toFixed(3),
      util_fr: +(0.4 + 0.35 * Math.abs(ay) + r() * 0.08).toFixed(3),
      util_rl: +(0.35 + 0.25 * Math.abs(ay) + r() * 0.08).toFixed(3),
      util_rr: +(0.35 + 0.3 * Math.abs(ay) + r() * 0.08).toFixed(3),
    });
  }
  return d;
}

function genYawData(n = 300) {
  const r = srng(137);
  return Array.from({ length: n }, (_, i) => {
    const t = i * 0.02;
    const wz_ref = 0.6 * Math.sin(t * 0.8) + 0.2 * Math.sin(t * 2.1);
    const wz_act = wz_ref + (r() - 0.5) * 0.06 - 0.01 * Math.sin(t * 5);
    const Mz_t = 60 * (wz_ref - wz_act) + 5 * Math.cos(t * 2);
    return {
      t: +t.toFixed(2), wz_ref: +wz_ref.toFixed(4), wz_act: +wz_act.toFixed(4),
      wz_err: +(wz_ref - wz_act).toFixed(4),
      Mz_target: +Mz_t.toFixed(1), Mz_actual: +(Mz_t + (r() - 0.5) * 8).toFixed(1),
      counter_steer: Math.sin(t * 0.3) > 0.8 ? 1 : 0,
    };
  });
}

function genTractionData(n = 300) {
  const r = srng(271);
  return Array.from({ length: n }, (_, i) => {
    const t = i * 0.02;
    const k_model = 0.11 + 0.02 * Math.sin(t * 0.3);
    const k_esc = k_model + 0.005 * Math.sin(t * 0.1) + (r() - 0.5) * 0.008;
    const sigma = 0.04 + 0.03 * Math.sin(t * 0.15);
    const alpha_fuse = 0.3 + 0.7 / (1 + Math.exp(-3 * (sigma / 0.08 - 1)));
    const k_fused = alpha_fuse * k_esc + (1 - alpha_fuse) * k_model;
    const k_meas = k_fused + (r() - 0.5) * 0.015 + 0.01 * Math.sin(t * 15);
    return {
      t: +t.toFixed(2),
      k_model: +k_model.toFixed(4), k_esc: +k_esc.toFixed(4),
      k_fused: +k_fused.toFixed(4), k_meas: +k_meas.toFixed(4),
      gp_sigma: +sigma.toFixed(4), fusion_alpha: +alpha_fuse.toFixed(3),
      desc_grad: +(Math.sin(t * 15) * 200 + r() * 40).toFixed(0),
    };
  });
}

function genLaunchData() {
  const r = srng(404);
  const d = [];
  const phases = [
    { name: "IDLE", t0: 0, t1: 0.5, phase: 0 },
    { name: "ARMED", t0: 0.5, t1: 0.8, phase: 1 },
    { name: "PROBE", t0: 0.8, t1: 0.95, phase: 2 },
    { name: "LAUNCH", t0: 0.95, t1: 2.8, phase: 3 },
    { name: "HANDOFF", t0: 2.8, t1: 3.1, phase: 4 },
    { name: "TC", t0: 3.1, t1: 4.0, phase: 5 },
  ];
  for (let i = 0; i < 400; i++) {
    const t = i * 0.01;
    const ph = phases.find(p => t >= p.t0 && t < p.t1) || phases[phases.length - 1];
    const inLaunch = t >= 0.95 && t < 3.1;
    const tL = Math.max(0, t - 0.95);
    const profile = inLaunch ? Math.min(1, 0.7 * Math.exp(-tL * 3) + 0.5 * (1 - Math.exp(-tL * 2)) + 0.5 * Math.min(tL / 1.5, 1)) : 0;
    const vx = inLaunch ? 20 * (1 - Math.exp(-tL * 1.5)) : t > 3.1 ? 18 + r() * 0.5 : 0;
    const f_front = inLaunch ? 0.40 - 0.10 * (1 - Math.exp(-tL / 0.5)) : 0.5;
    d.push({
      t: +t.toFixed(3), phase: ph.phase, phaseName: ph.name,
      profile: +profile.toFixed(3), vx: +vx.toFixed(2),
      T_total: +(profile * 1800).toFixed(0),
      f_front: +f_front.toFixed(3),
      mu_est: +(1.35 + r() * 0.05).toFixed(3),
    });
  }
  return d;
}

function genCBFData(n = 300) {
  const r = srng(555);
  return Array.from({ length: n }, (_, i) => {
    const t = i * 0.02;
    const beta = 0.08 * Math.sin(t * 0.7) + 0.04 * Math.sin(t * 1.9);
    const B_beta = 0.15 * 0.15 - beta * beta;
    const wz = 0.8 * Math.sin(t * 0.6);
    const B_wz = 1.5 * 1.5 - wz * wz;
    const intervention = B_beta < 0.005 || B_wz < 0.3 ? 20 + r() * 30 : 0;
    return {
      t: +t.toFixed(2), beta: +beta.toFixed(4),
      B_beta: +B_beta.toFixed(5), B_wz: +B_wz.toFixed(3),
      wz: +wz.toFixed(3), intervention: +intervention.toFixed(1),
      beta_limit: 0.15, wz_limit: 1.5,
    };
  });
}

function genImpedanceData(n = 400) {
  const r = srng(678);
  return Array.from({ length: n }, (_, i) => {
    const t = i * 0.005;
    const step = t > 0.2 ? 0.7 : 0;
    const pio = 0.15 * Math.sin(2 * Math.PI * 3 * t) * (t > 0.8 ? 1 : 0);
    const raw = Math.min(1, Math.max(0, step + pio + r() * 0.02));
    const J = 0.05, Cv = 2.1, K = 45;
    const wn = Math.sqrt(K / J), zeta = Cv / (2 * Math.sqrt(K * J));
    const wd = wn * Math.sqrt(1 - zeta * zeta);
    const tStep = Math.max(0, t - 0.2);
    const filtered_step = t > 0.2 ? 0.7 * (1 - Math.exp(-zeta * wn * tStep) * Math.cos(wd * tStep)) : 0;
    const filtered_pio = 0.15 * Math.sin(2 * Math.PI * 3 * t - 55 * Math.PI / 180) * 0.93 * (t > 0.8 ? 1 : 0);
    const filtered = Math.min(1, Math.max(0, filtered_step + filtered_pio));
    return { t: +t.toFixed(4), raw: +raw.toFixed(4), filtered: +filtered.toFixed(4), pio_zone: t > 0.8 ? 1 : 0 };
  });
}

function genThermalData(n = 200) {
  const r = srng(789);
  return Array.from({ length: n }, (_, i) => {
    const t = i * 0.5;
    const base = 55 + 25 * (1 - Math.exp(-t / 40));
    const load = 8 * Math.sin(t * 0.08);
    return {
      t: +t.toFixed(1),
      M_fl: +(base + load * 0.8 + r() * 1.5).toFixed(1),
      M_fr: +(base + load * 1.2 + r() * 1.5).toFixed(1),
      M_rl: +(base + 5 + load * 1.0 + r() * 1.5).toFixed(1),
      M_rr: +(base + 5 + load * 1.4 + r() * 1.5).toFixed(1),
      I_fl: +(40 + 12 * (1 - Math.exp(-t / 50)) + r() * 0.8).toFixed(1),
      I_fr: +(40 + 13 * (1 - Math.exp(-t / 50)) + r() * 0.8).toFixed(1),
      I_rl: +(40 + 11 * (1 - Math.exp(-t / 50)) + r() * 0.8).toFixed(1),
      I_rr: +(40 + 14 * (1 - Math.exp(-t / 50)) + r() * 0.8).toFixed(1),
      SoC: +(95 - t * 0.08 + r() * 0.1).toFixed(2),
      V_bus: +(650 - t * 0.12 + r() * 0.5).toFixed(1),
      P_total: +(15000 + 8000 * Math.sin(t * 0.1) + r() * 500).toFixed(0),
    };
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// FRICTION CIRCLE SCATTER (special component)
// ═══════════════════════════════════════════════════════════════════════════
function FrictionCircleMini({ data, idx, label, color }) {
  const last30 = data.slice(-30);
  const pts = last30.map(d => ({
    fx: (d[`T_${label.toLowerCase()}`] || 0) / 0.2032 / 10,
    fy: (Math.sin(d.t * 1.5) * 600 * (idx < 2 ? 1.2 : 0.8)) / 10,
  }));
  const muFz = 1.4 * (idx < 2 ? 700 : 800) / 10;
  const circPts = Array.from({ length: 37 }, (_, i) => {
    const a = i * Math.PI * 2 / 36;
    return { fx: muFz * Math.cos(a), fy: muFz * Math.sin(a) };
  });
  // LA SOLUCIÓN:
return (
    <GC style={{ padding: "8px 6px", textAlign: "center" }}>
      <div style={{ fontSize: 8, fontWeight: 700, color, letterSpacing: 2, fontFamily: C.dt, marginBottom: 4 }}>{label}</div>
      <ResponsiveContainer width="100%" height={110}>
        <ScatterChart margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <XAxis type="number" dataKey="fx" domain={[-140, 140]} hide />
          <YAxis type="number" dataKey="fy" domain={[-140, 140]} hide />
          <Scatter data={circPts} fill="none" stroke={`${color}30`} strokeWidth={1} line />
          <Scatter data={pts} fill={color} r={2} />
        </ScatterChart>
      </ResponsiveContainer>
      <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>Fx/Fy [×10 N]</div>
    </GC>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════
function AllocationTab({ data }) {
  const last = data[data.length - 1];
  const chart = useMemo(() => data.filter((_, i) => i % 2 === 0), [data]);
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
      {CORNERS.map((c, i) => <KPI key={c} label={`T ${c}`} value={`${last[`T_${c.toLowerCase()}`]} Nm`} sub={`util: ${(last[`util_${c.toLowerCase()}`] * 100).toFixed(0)}%`} sentiment={last[`util_${c.toLowerCase()}`] < 0.85 ? "positive" : "amber"} delay={i} />)}
      <KPI label="Fx Track" value={`${(last.Fx_actual / last.Fx_target * 100).toFixed(0)}%`} sub={`${last.Fx_actual}/${last.Fx_target} N`} sentiment={Math.abs(last.Fx_actual - last.Fx_target) < 200 ? "positive" : "amber"} delay={4} />
    </div>
    <Sec title="Per-Wheel Torque Distribution">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} unit=" Nm" />
          <Tooltip {...tt()} />
          {CORNERS.map((c, i) => <Line key={c} type="monotone" dataKey={`T_${c.toLowerCase()}`} stroke={CORNER_C[i]} strokeWidth={1.5} dot={false} name={c} />)}
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
    <Sec title="Friction Circle Utilization">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
        {CORNERS.map((c, i) => <FrictionCircleMini key={c} data={data} idx={i} label={c} color={CORNER_C[i]} />)}
      </div>
    </Sec>
    <Sec title="Force Tracking">
      <GC><ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} unit=" N" />
          <Tooltip {...tt()} />
          <Area type="monotone" dataKey="Fx_target" fill={`${PT}15`} stroke={PT} strokeWidth={1} strokeDasharray="4 2" name="Fx target" />
          <Line type="monotone" dataKey="Fx_actual" stroke={C.gn} strokeWidth={1.5} dot={false} name="Fx actual" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: YAW CONTROL
// ═══════════════════════════════════════════════════════════════════════════
function YawTab({ data }) {
  const last = data[data.length - 1];
  const chart = useMemo(() => data.filter((_, i) => i % 2 === 0), [data]);
  const rmsErr = Math.sqrt(data.reduce((s, d) => s + d.wz_err ** 2, 0) / data.length);
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
      <KPI label="ψ̇ ref" value={`${last.wz_ref.toFixed(3)} r/s`} sub="reference" sentiment="neutral" delay={0} />
      <KPI label="ψ̇ actual" value={`${last.wz_act.toFixed(3)} r/s`} sub="measured" sentiment="neutral" delay={1} />
      <KPI label="RMS Error" value={`${(rmsErr * 1000).toFixed(1)} mr/s`} sub={rmsErr < 0.02 ? "excellent" : "acceptable"} sentiment={rmsErr < 0.02 ? "positive" : "amber"} delay={2} />
      <KPI label="Counter-Steer" value={last.counter_steer ? "ACTIVE" : "—"} sub="driver override" sentiment={last.counter_steer ? "amber" : "neutral"} delay={3} />
    </div>
    <Sec title="Yaw Rate Tracking">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} unit=" r/s" />
          <Tooltip {...tt()} />
          <Area type="monotone" dataKey="wz_ref" fill={`${C.cy}10`} stroke={C.cy} strokeWidth={1} strokeDasharray="4 2" name="ψ̇ ref" />
          <Line type="monotone" dataKey="wz_act" stroke={C.gn} strokeWidth={1.5} dot={false} name="ψ̇ actual" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
    <Sec title="Yaw Moment Demand">
      <GC><ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} unit=" Nm" />
          <Tooltip {...tt()} /><ReferenceLine y={0} stroke={C.dm} strokeDasharray="2 2" />
          <Line type="monotone" dataKey="Mz_target" stroke={PT} strokeWidth={1.5} dot={false} name="Mz demand" />
          <Line type="monotone" dataKey="Mz_actual" stroke={C.gn} strokeWidth={1} dot={false} name="Mz achieved" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: TRACTION
// ═══════════════════════════════════════════════════════════════════════════
function TractionTab({ data }) {
  const last = data[data.length - 1];
  const chart = useMemo(() => data.filter((_, i) => i % 2 === 0), [data]);
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
      <KPI label="κ* model" value={last.k_model.toFixed(4)} sub="Pacejka Newton" sentiment="neutral" delay={0} />
      <KPI label="κ* ESC" value={last.k_esc.toFixed(4)} sub="DESC lock-in" sentiment="neutral" delay={1} />
      <KPI label="κ* fused" value={last.k_fused.toFixed(4)} sub="GP-weighted" sentiment="positive" delay={2} />
      <KPI label="GP σ" value={last.gp_sigma.toFixed(4)} sub={last.gp_sigma > 0.08 ? "high → trust ESC" : "low → trust model"} sentiment={last.gp_sigma < 0.08 ? "positive" : "amber"} delay={3} />
      <KPI label="Fusion α" value={`${(last.fusion_alpha * 100).toFixed(0)}%`} sub="ESC weight" sentiment="neutral" delay={4} />
    </div>
    <Sec title="Dual-Path κ* Estimation">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} domain={[0.05, 0.18]} />
          <Tooltip {...tt()} />
          <Line type="monotone" dataKey="k_model" stroke={C.am} strokeWidth={1} dot={false} strokeDasharray="4 2" name="κ* model" />
          <Line type="monotone" dataKey="k_esc" stroke={DESC_C} strokeWidth={1} dot={false} strokeDasharray="2 2" name="κ* ESC" />
          <Line type="monotone" dataKey="k_fused" stroke={C.gn} strokeWidth={2} dot={false} name="κ* fused" />
          <Line type="monotone" dataKey="k_meas" stroke={C.red} strokeWidth={0.8} dot={false} opacity={0.5} name="κ measured" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
    <Sec title="DESC Gradient (lock-in demodulator output)">
      <GC><ResponsiveContainer width="100%" height={120}>
        <AreaChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} />
          <Tooltip {...tt()} /><ReferenceLine y={0} stroke={C.dm} strokeDasharray="2 2" />
          <Area type="monotone" dataKey="desc_grad" fill={`${DESC_C}15`} stroke={DESC_C} strokeWidth={1} name="∂Fx/∂κ est." />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: LAUNCH
// ═══════════════════════════════════════════════════════════════════════════
function LaunchTab({ data }) {
  const last = data[data.length - 1];
  const chart = useMemo(() => data.filter((_, i) => i % 3 === 0), [data]);
  const phaseNames = ["IDLE", "ARMED", "PROBE", "LAUNCH", "HANDOFF", "TC"];
  const phaseColors = [C.dm, C.am, LAUNCH_C, C.gn, C.cy, PT];
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
      <KPI label="Phase" value={phaseNames[last.phase]} sub={`code ${last.phase}`} sentiment={last.phase === 3 ? "positive" : "neutral"} delay={0} />
      <KPI label="Speed" value={`${last.vx} m/s`} sub={`${(last.vx * 3.6).toFixed(0)} km/h`} sentiment={last.vx > 5 ? "positive" : "neutral"} delay={1} />
      <KPI label="Torque" value={`${last.T_total} Nm`} sub="total 4-wheel" sentiment={last.T_total > 0 ? "positive" : "neutral"} delay={2} />
      <KPI label="μ probe" value={last.mu_est} sub="surface estimate" sentiment={last.mu_est > 1.2 ? "positive" : "amber"} delay={3} />
      <KPI label="F/R Split" value={`${(last.f_front * 100).toFixed(0)}/${((1 - last.f_front) * 100).toFixed(0)}`} sub="front/rear %" sentiment="neutral" delay={4} />
    </div>
    <Sec title="Launch Profile & Velocity">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} label={{ value: "Time [s]", position: "insideBottomRight", style: { fontSize: 8, fill: C.dm } }} />
          <YAxis yAxisId="left" {...ax()} /><YAxis yAxisId="right" orientation="right" {...ax()} />
          <Tooltip {...tt()} />
          <Area yAxisId="left" type="monotone" dataKey="T_total" fill={`${LAUNCH_C}15`} stroke={LAUNCH_C} strokeWidth={1.5} name="T total [Nm]" />
          <Line yAxisId="right" type="monotone" dataKey="vx" stroke={C.gn} strokeWidth={2} dot={false} name="Speed [m/s]" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
    <Sec title="State Machine Phase Timeline">
      <GC style={{ padding: "10px 14px" }}>
        <div style={{ display: "flex", gap: 0, height: 28, borderRadius: 4, overflow: "hidden" }}>
          {phaseNames.map((name, pi) => {
            const pts = data.filter(d => d.phase === pi);
            const frac = pts.length / data.length;
            if (frac < 0.01) return null;
            return (<div key={name} style={{ flex: frac, background: `${phaseColors[pi]}30`, borderRight: `1px solid ${C.b0}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ fontSize: 7, fontWeight: 700, color: phaseColors[pi], fontFamily: C.dt, letterSpacing: 1 }}>{name}</span>
            </div>);
          })}
        </div>
      </GC>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: CBF SAFETY
// ═══════════════════════════════════════════════════════════════════════════
function CBFTab({ data }) {
  const last = data[data.length - 1];
  const chart = useMemo(() => data.filter((_, i) => i % 2 === 0), [data]);
  const interventions = data.filter(d => d.intervention > 5).length;
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
      <KPI label="B_β" value={last.B_beta.toFixed(4)} sub={last.B_beta < 0.005 ? "CRITICAL" : last.B_beta < 0.01 ? "near limit" : "safe"} sentiment={last.B_beta > 0.01 ? "positive" : last.B_beta > 0.003 ? "amber" : "negative"} delay={0} />
      <KPI label="B_ψ̇" value={last.B_wz.toFixed(3)} sub={last.B_wz < 0.3 ? "near limit" : "safe"} sentiment={last.B_wz > 0.5 ? "positive" : "amber"} delay={1} />
      <KPI label="β" value={`${(last.beta * 180 / Math.PI).toFixed(1)}°`} sub={`limit: ${(last.beta_limit * 180 / Math.PI).toFixed(1)}°`} sentiment={Math.abs(last.beta) < last.beta_limit * 0.8 ? "positive" : "amber"} delay={2} />
      <KPI label="Interventions" value={interventions} sub={`of ${data.length} steps`} sentiment={interventions < 10 ? "positive" : "amber"} delay={3} />
    </div>
    <Sec title="Sideslip Barrier B_β = β_max² − β²">
      <GC><ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} />
          <Tooltip {...tt()} /><ReferenceLine y={0} stroke={CBF_C} strokeWidth={2} label={{ value: "UNSAFE", position: "insideTopRight", style: { fontSize: 7, fill: CBF_C } }} />
          <Area type="monotone" dataKey="B_beta" fill={`${C.gn}15`} stroke={C.gn} strokeWidth={1.5} name="B_β" />
          <Line type="monotone" dataKey="intervention" stroke={CBF_C} strokeWidth={1} dot={false} opacity={0.7} name="CBF ΔT [Nm]" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: IMPEDANCE
// ═══════════════════════════════════════════════════════════════════════════
function ImpedanceTab({ data }) {
  const chart = useMemo(() => data.filter((_, i) => i % 3 === 0), [data]);
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
      <KPI label="ωₙ" value="4.8 Hz" sub="natural frequency" sentiment="neutral" delay={0} />
      <KPI label="ζ" value="0.70" sub="damping ratio" sentiment="positive" delay={1} />
      <KPI label="φ @ 3Hz" value="−55°" sub="PIO phase shift" sentiment="positive" delay={2} />
    </div>
    <Sec title="Throttle: Raw vs Impedance-Filtered">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} label={{ value: "Time [s]", position: "insideBottomRight", style: { fontSize: 8, fill: C.dm } }} />
          <YAxis {...ax()} domain={[0, 1]} />
          <Tooltip {...tt()} />
          {data[0]?.pio_zone !== undefined && <ReferenceArea x1={0.8} x2={2.0} fill={`${CBF_C}08`} label={{ value: "PIO zone", position: "insideTop", style: { fontSize: 7, fill: CBF_C } }} />}
          <Line type="monotone" dataKey="raw" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 2" name="Raw pedal" />
          <Line type="monotone" dataKey="filtered" stroke={PT} strokeWidth={2} dot={false} name="Filtered" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
    <Sec title="Impedance Advantage vs Low-Pass Filter">
      <GC style={{ padding: "12px 14px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, fontSize: 9, fontFamily: C.dt }}>
          <div><div style={{ color: C.dm, fontSize: 7, letterSpacing: 1, marginBottom: 4 }}>METRIC</div>
            {["Phase @ 3Hz", "Attenuation @ 3Hz", "DC lag", "Step 90% rise"].map(m => <div key={m} style={{ color: C.br, padding: "3px 0", borderBottom: `1px solid ${C.b1}08` }}>{m}</div>)}
          </div>
          <div><div style={{ color: PT, fontSize: 7, letterSpacing: 1, marginBottom: 4 }}>IMPEDANCE</div>
            {["−55°", "7%", "0 ms", "50 ms"].map(v => <div key={v} style={{ color: C.gn, padding: "3px 0", borderBottom: `1px solid ${C.b1}08` }}>{v}</div>)}
          </div>
          <div><div style={{ color: C.dm, fontSize: 7, letterSpacing: 1, marginBottom: 4 }}>5Hz LPF</div>
            {["−31°", "14%", "~30 ms", "~45 ms"].map(v => <div key={v} style={{ color: C.am, padding: "3px 0", borderBottom: `1px solid ${C.b1}08` }}>{v}</div>)}
          </div>
        </div>
      </GC>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: MOTOR THERMAL
// ═══════════════════════════════════════════════════════════════════════════
function ThermalTab({ data }) {
  const last = data[data.length - 1];
  const chart = useMemo(() => data.filter((_, i) => i % 2 === 0), [data]);
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
      {CORNERS.map((c, i) => <KPI key={c} label={`Motor ${c}`} value={`${last[`M_${c.toLowerCase()}`]}°C`} sub={`inv: ${last[`I_${c.toLowerCase()}`]}°C`} sentiment={last[`M_${c.toLowerCase()}`] < 120 ? "positive" : last[`M_${c.toLowerCase()}`] < 140 ? "amber" : "negative"} delay={i} />)}
      <KPI label="SoC" value={`${last.SoC}%`} sub={`${last.V_bus} V`} sentiment={last.SoC > 20 ? "positive" : last.SoC > 10 ? "amber" : "negative"} delay={4} />
    </div>
    <Sec title="Motor Temperature Evolution">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} unit="°C" domain={[40, 120]} />
          <Tooltip {...tt()} /><ReferenceLine y={130} stroke={C.am} strokeDasharray="4 2" label={{ value: "derate onset", position: "insideTopRight", style: { fontSize: 7, fill: C.am } }} />
          <ReferenceLine y={150} stroke={C.red} strokeDasharray="2 2" label={{ value: "LIMIT", position: "insideTopRight", style: { fontSize: 7, fill: C.red } }} />
          {CORNERS.map((c, i) => <Line key={c} type="monotone" dataKey={`M_${c.toLowerCase()}`} stroke={CORNER_C[i]} strokeWidth={1.5} dot={false} name={`Motor ${c}`} />)}
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
    <Sec title="Pack SoC & Power Draw">
      <GC><ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={`${C.b1}15`} /><XAxis dataKey="t" {...ax()} />
          <YAxis yAxisId="l" {...ax()} unit="%" domain={[60, 100]} /><YAxis yAxisId="r" orientation="right" {...ax()} unit=" W" />
          <Tooltip {...tt()} />
          <Area yAxisId="l" type="monotone" dataKey="SoC" fill={`${C.gn}15`} stroke={C.gn} strokeWidth={1.5} name="SoC %" />
          <Line yAxisId="r" type="monotone" dataKey="P_total" stroke={PT} strokeWidth={1} dot={false} name="P total [W]" />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB: DIAGNOSTICS
// ═══════════════════════════════════════════════════════════════════════════
function DiagnosticsTab() {
  const r = srng(999);
  const systems = [
    { name: "SOCP Allocator", status: "nominal", timing: "0.18", budget: "1.0" },
    { name: "CBF Filter", status: "nominal", timing: "0.05", budget: "0.5" },
    { name: "DESC (×4)", status: "nominal", timing: "0.03", budget: "0.2" },
    { name: "Launch FSM", status: "idle", timing: "0.01", budget: "0.1" },
    { name: "Impedance (×2)", status: "nominal", timing: "0.01", budget: "0.1" },
    { name: "Thermal ODE", status: "nominal", timing: "0.02", budget: "0.2" },
    { name: "Total Pipeline", status: "nominal", timing: "0.30", budget: "5.0" },
  ];
  const conf = [0.98, 0.97, 0.99, 0.96].map(c => +(c + (r() - 0.5) * 0.02).toFixed(3));
  return (<div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
      <KPI label="Pipeline" value="0.30 ms" sub="per step (200 Hz)" sentiment="positive" delay={0} />
      <KPI label="Margin" value="94%" sub="vs 5ms budget" sentiment="positive" delay={1} />
      <KPI label="Degradation" value="Level 0" sub="all sensors nominal" sentiment="positive" delay={2} />
      <KPI label="XLA Graph" value="1" sub="fused subgraph" sentiment="positive" delay={3} />
    </div>
    <Sec title="Subsystem Timing Budget">
      <GC style={{ padding: "10px 14px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "180px 80px 80px 80px 1fr", gap: 0, fontSize: 8, fontFamily: C.dt }}>
          {["Subsystem", "Status", "Time [ms]", "Budget [ms]", ""].map(h => <div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 1, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>)}
          {systems.map(s => {
            const pct = (parseFloat(s.timing) / parseFloat(s.budget)) * 100;
            return (<React.Fragment key={s.name}>
              <div style={{ color: s.name === "Total Pipeline" ? PT : C.br, fontWeight: s.name === "Total Pipeline" ? 700 : 400, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.name}</div>
              <div style={{ color: s.status === "nominal" ? C.gn : C.am, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.status.toUpperCase()}</div>
              <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.timing}</div>
              <div style={{ color: C.dm, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.budget}</div>
              <div style={{ padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>
                <div style={{ height: 6, borderRadius: 3, background: `${C.b1}20`, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min(pct, 100)}%`, height: "100%", background: pct < 60 ? C.gn : pct < 85 ? C.am : C.red, borderRadius: 3 }} />
                </div>
              </div>
            </React.Fragment>);
          })}
        </div>
      </GC>
    </Sec>
    <Sec title="Sensor Confidence Scores">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
        {CORNERS.map((c, i) => (
          <GC key={c} style={{ padding: "12px", textAlign: "center" }}>
            <div style={{ fontSize: 8, fontWeight: 700, color: CORNER_C[i], letterSpacing: 2, fontFamily: C.dt }}>{c}</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: conf[i] > 0.95 ? C.gn : C.am, fontFamily: C.hd, margin: "6px 0 2px" }}>{(conf[i] * 100).toFixed(1)}%</div>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>{conf[i] > 0.95 ? "HEALTHY" : "DEGRADED"}</div>
          </GC>
        ))}
      </div>
    </Sec>
  </div>);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN MODULE EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function PowertrainControlModule() {
  const [tab, setTab] = useState("alloc");

  const allocData = useMemo(() => genAllocData(), []);
  const yawData = useMemo(() => genYawData(), []);
  const tractionData = useMemo(() => genTractionData(), []);
  const launchData = useMemo(() => genLaunchData(), []);
  const cbfData = useMemo(() => genCBFData(), []);
  const impedanceData = useMemo(() => genImpedanceData(), []);
  const thermalData = useMemo(() => genThermalData(), []);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Mode" value="TV + TC" sub="unified allocation" sentiment="positive" delay={0} />
        <KPI label="SOCP Iters" value="12" sub="projected gradient" sentiment="neutral" delay={1} />
        <KPI label="CBF" value="ARMED" sub="input-delay DCBF" sentiment="positive" delay={2} />
        <KPI label="Stack" value="2,797 LOC" sub="100% JAX — 0% numpy" sentiment="positive" delay={3} />
      </div>

      <div style={{ display: "flex", gap: 4, marginBottom: 14, flexWrap: "wrap" }}>
        {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={PT} />)}
      </div>

      {tab === "alloc" && <AllocationTab data={allocData} />}
      {tab === "yaw" && <YawTab data={yawData} />}
      {tab === "traction" && <TractionTab data={tractionData} />}
      {tab === "launch" && <LaunchTab data={launchData} />}
      {tab === "cbf" && <CBFTab data={cbfData} />}
      {tab === "impedance" && <ImpedanceTab data={impedanceData} />}
      {tab === "thermal" && <ThermalTab data={thermalData} />}
      {tab === "diag" && <DiagnosticsTab />}
    </div>
  );
}