// ═══════════════════════════════════════════════════════════════════════════
// src/OverviewModule.jsx — Project-GP Dashboard v6.1
// ═══════════════════════════════════════════════════════════════════════════
// Bloomberg-terminal density: system KPI header + subsystem status grid +
// Pareto front + convergence + architecture pillar deep-dives.
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
  ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend, Cell,
} from "recharts";
import { C, GL } from "./theme.js";

// ── Helpers ──────────────────────────────────────────────────────────────────
const dt = C.dt || "'JetBrains Mono', monospace";
const hd = C.hd || "'Space Grotesk', sans-serif";
const bd = C.bd || "'Inter', sans-serif";
const F = ({ color, children }) => <span style={{ color: color || C.cy, fontWeight: 600 }}>{children}</span>;
const gl = { ...GL, overflow: "hidden" };

// ═══════════════════════════════════════════════════════════════════════════
// SYSTEM KPI HEADER — top-level metrics across entire stack
// ═══════════════════════════════════════════════════════════════════════════

function SystemKPIHeader({ pareto, conv }) {
  const bestGrip = pareto?.length ? Math.max(...pareto.map(p => p.grip)) : 1.0;
  const estLapTime = (64.0 / Math.max(0.5, bestGrip)).toFixed(2);
  const bestStab = pareto?.length ? Math.min(...pareto.filter(p => p.grip > bestGrip - 0.05).map(p => p.stability)) : 5.0;
  const convergedGrip = conv?.length ? conv[conv.length - 1]?.bestGrip?.toFixed(3) : "—";
  const hvol = conv?.length ? conv[conv.length - 1]?.hvol?.toFixed(3) : "—";
  const kl = conv?.length ? conv[conv.length - 1]?.kl : 0;
  const paretoSize = pareto?.length || 0;

  const kpis = [
    { label: "EST LAP TIME", value: `${estLapTime}s`, sub: "FSG Autocross", color: C.cy, large: true },
    { label: "BEST GRIP μ", value: bestGrip.toFixed(3), sub: "Pareto front", color: C.gn },
    { label: "MIN OVERSHOOT", value: `${bestStab.toFixed(2)}`, sub: "rad/s yaw", color: C.am },
    { label: "PARETO SIZE", value: `${paretoSize}`, sub: "non-dominated", color: "#a78bfa" },
    { label: "CONVERGED μ", value: convergedGrip, sub: "last iteration", color: C.gn },
    { label: "HYPERVOLUME", value: hvol, sub: "indicator", color: C.cy },
    { label: "KL DIVERGENCE", value: kl ? kl.toExponential(2) : "—", sub: "trust region", color: kl > 0.01 ? C.am : C.gn },
    { label: "SETUP DIM", value: "40", sub: "28+12 ext Ter27", color: C.cy },
    { label: "STATE DIM", value: "46", sub: "14 mech + thermal", color: C.am },
    { label: "PHYSICS", value: "200Hz", sub: "5-substep LF", color: C.gn },
    { label: "STACK", value: "100%", sub: "pure JAX/XLA", color: C.gn },
    { label: "SOCP", value: "265ms", sub: "target <5ms", color: C.red },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(12, 1fr)", gap: 6, marginBottom: 12 }}>
      {kpis.map((k, i) => (
        <div key={k.label} style={{
          ...gl, padding: "8px 8px 6px",
          borderLeft: `2px solid ${k.color}30`,
          gridColumn: k.large ? "span 1" : undefined,
        }}>
          <div style={{ fontSize: 6.5, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: dt, marginBottom: 3 }}>{k.label}</div>
          <div style={{ fontSize: k.large ? 16 : 14, fontWeight: 800, color: k.color, fontFamily: dt, lineHeight: 1 }}>{k.value}</div>
          <div style={{ fontSize: 7, color: C.dm, fontFamily: dt, marginTop: 2 }}>{k.sub}</div>
        </div>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBSYSTEM STATUS GRID — dense 4-col matrix of all subsystem states
// ═══════════════════════════════════════════════════════════════════════════

const SUBSYSTEMS = [
  {
    name: "Neural Port-Hamiltonian", tag: "H_net", status: "OPERATIONAL", color: C.cy,
    metrics: [
      { l: "DOF", v: "46 (14 mech)" },
      { l: "Integrator", v: "Symplectic LF ×5" },
      { l: "H_res gate", v: "softplus·susp_sq" },
      { l: "R_net", v: "LLᵀ+diag PSD" },
      { l: "Fidelity", v: "90.6%" },
    ],
  },
  {
    name: "Multi-Fidelity Tire", tag: "PINN+GP", status: "OPERATIONAL", color: C.gn,
    metrics: [
      { l: "Base", v: "Pacejka MF6.2" },
      { l: "Thermal", v: "5-node Jaeger" },
      { l: "PINN", v: "Residual drift" },
      { l: "GP", v: "Matérn 5/2 sparse" },
      { l: "Turn slip", v: "Corrected" },
    ],
  },
  {
    name: "Diff-WMPC", tag: "OCP", status: "OPERATIONAL", color: C.am,
    metrics: [
      { l: "Wavelets", v: "3-level Db4" },
      { l: "Safety", v: "UT stochastic tubes" },
      { l: "Solver", v: "AL + L-BFGS-B" },
      { l: "Regularizer", v: "Pseudo-Huber" },
      { l: "Horizon", v: "N=64 (320ms)" },
    ],
  },
  {
    name: "MORL-SB-TRPO", tag: "OPT", status: "OPERATIONAL", color: C.red,
    metrics: [
      { l: "Search dim", v: "40 (28+12)" },
      { l: "Objectives", v: "Grip + Stab + LTE" },
      { l: "Policy grad", v: "RNPG natural" },
      { l: "Archive", v: "SMS-EMOA HV" },
      { l: "Cold start", v: "ARD-BO EI" },
    ],
  },
  {
    name: "Powertrain Stack", tag: "PT", status: "OPERATIONAL", color: "#f97316",
    metrics: [
      { l: "Stages", v: "13-stage pipeline" },
      { l: "TV", v: "SOCP convex alloc" },
      { l: "TC", v: "DESC extremum-seek" },
      { l: "CBF", v: "Input-delay DCBF" },
      { l: "Entry", v: "powertrain_step()" },
    ],
  },
  {
    name: "EKF Online Estimation", tag: "EKF", status: "OPERATIONAL", color: "#22d3ee",
    metrics: [
      { l: "States", v: "5 (λ_μ,Cα,h_cg,T,α_pk)" },
      { l: "Rate", v: "100 Hz" },
      { l: "Q,P dims", v: "5×5 extended" },
      { l: "Innovation", v: "ỹ→0 convergence" },
      { l: "Differentiable", v: "jax.grad(ỹ)" },
    ],
  },
  {
    name: "Aero Map", tag: "AERO", status: "OPERATIONAL", color: "#ff6090",
    metrics: [
      { l: "Model", v: "Physics-informed NN" },
      { l: "Constraint", v: "Fz ∝ v² enforced" },
      { l: "Inputs", v: "v, h_ride, rake" },
      { l: "Outputs", v: "Fz_f, Fz_r, Cd" },
      { l: "Floor", v: "softplus everywhere" },
    ],
  },
  {
    name: "Differentiable Track", tag: "TRACK", status: "OPERATIONAL", color: "#a78bfa",
    metrics: [
      { l: "Spline", v: "JAX cubic SE(3)" },
      { l: "Curvature", v: "Analytical κ(s)" },
      { l: "Lap sim", v: "∂(t_lap)/∂(setup)" },
      { l: "CMA-ES", v: "Crossover validator" },
      { l: "Tracks", v: "FSG autocross +4" },
    ],
  },
];

function SubsystemGrid() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 12 }}>
      {SUBSYSTEMS.map(sys => (
        <div key={sys.tag} style={{ ...gl, padding: "8px 10px" }}>
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
            <div>
              <span style={{ fontSize: 8, fontWeight: 700, color: sys.color, fontFamily: dt }}>{sys.tag}</span>
              <span style={{ fontSize: 7, color: C.dm, fontFamily: dt, marginLeft: 6 }}>{sys.name}</span>
            </div>
            <span style={{ fontSize: 6.5, fontWeight: 700, color: sys.status === "OPERATIONAL" ? C.gn : C.am, fontFamily: dt, background: sys.status === "OPERATIONAL" ? `${C.gn}12` : `${C.am}12`, padding: "1px 5px", borderRadius: 2 }}>
              {sys.status}
            </span>
          </div>
          {/* Metrics */}
          {sys.metrics.map(m => (
            <div key={m.l} style={{ display: "flex", justifyContent: "space-between", fontSize: 7.5, fontFamily: dt, padding: "1px 0" }}>
              <span style={{ color: C.dm }}>{m.l}</span>
              <span style={{ color: C.br, fontWeight: 500 }}>{m.v}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// PARETO + CONVERGENCE CHARTS — inline mini-charts
// ═══════════════════════════════════════════════════════════════════════════

function ParetoConvergenceRow({ pareto, conv }) {
  const safePareto = pareto || [];
  const safeConv = (conv || []).filter((_, i) => i % 2 === 0);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
      {/* Pareto Front */}
      <div style={{ ...gl, padding: "8px 8px 4px" }}>
        <div style={{ fontSize: 7, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: dt, marginBottom: 4 }}>PARETO FRONT — GRIP vs STABILITY</div>
        <ResponsiveContainer width="100%" height={160}>
          <ScatterChart margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="grip" type="number" name="Grip μ" tick={{ fontSize: 7, fill: C.dm, fontFamily: dt }} stroke={C.b1} />
            <YAxis dataKey="stability" type="number" name="Stability" tick={{ fontSize: 7, fill: C.dm, fontFamily: dt }} stroke={C.b1} />
            <Tooltip contentStyle={{ background: C.panel || "#0e1420", border: `1px solid ${C.b1}`, borderRadius: 4, fontSize: 8, fontFamily: dt, color: C.br }} />
            <Scatter data={safePareto} fill={C.cy} fillOpacity={0.6} r={3}>
              {safePareto.map((_, i) => <Cell key={i} fill={i === safePareto.length - 1 ? C.gn : C.cy} />)}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Convergence */}
      <div style={{ ...gl, padding: "8px 8px 4px" }}>
        <div style={{ fontSize: 7, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: dt, marginBottom: 4 }}>MORL CONVERGENCE — BEST GRIP</div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={safeConv} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="iter" tick={{ fontSize: 7, fill: C.dm, fontFamily: dt }} stroke={C.b1} />
            <YAxis domain={["auto", "auto"]} tick={{ fontSize: 7, fill: C.dm, fontFamily: dt }} stroke={C.b1} />
            <Tooltip contentStyle={{ background: C.panel || "#0e1420", border: `1px solid ${C.b1}`, borderRadius: 4, fontSize: 8, fontFamily: dt, color: C.br }} />
            <Line dataKey="bestGrip" stroke={C.gn} dot={false} strokeWidth={1.5} name="Best Grip" />
            {safeConv.length > 0 && <Line dataKey="hvol" stroke={C.cy} dot={false} strokeWidth={1} strokeDasharray="4 2" name="HV Indicator" />}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* KL + Policy Gradient */}
      <div style={{ ...gl, padding: "8px 8px 4px" }}>
        <div style={{ fontSize: 7, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: dt, marginBottom: 4 }}>TRUST REGION — KL DIVERGENCE</div>
        <ResponsiveContainer width="100%" height={160}>
          <ComposedChart data={safeConv} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="iter" tick={{ fontSize: 7, fill: C.dm, fontFamily: dt }} stroke={C.b1} />
            <YAxis tick={{ fontSize: 7, fill: C.dm, fontFamily: dt }} stroke={C.b1} />
            <Tooltip contentStyle={{ background: C.panel || "#0e1420", border: `1px solid ${C.b1}`, borderRadius: 4, fontSize: 8, fontFamily: dt, color: C.br }} />
            <Line dataKey="kl" stroke={C.am} dot={false} strokeWidth={1.5} name="KL" />
            {safeConv[0]?.pgNorm != null && <Line dataKey="pgNorm" stroke={C.red} dot={false} strokeWidth={1} strokeDasharray="3 2" name="PG Norm" />}
            <ReferenceLine y={0.005} stroke={C.gn} strokeDasharray="4 4" label={{ value: "δ_KL", fontSize: 7, fill: C.gn }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SANITY CHECK RESULTS — test status matrix
// ═══════════════════════════════════════════════════════════════════════════

function SanityCheckMatrix() {
  const tests = [
    { id: 1, name: "Static Equilibrium", status: "PASS", time: "0.3s" },
    { id: 2, name: "PH Energy Conservation", status: "PASS", time: "1.2s" },
    { id: 3, name: "WMPC Circular Track", status: "PASS", time: "45s" },
    { id: 4, name: "Friction Circle", status: "PASS", time: "0.1s" },
    { id: 5, name: "Pacejka Load Sensitivity", status: "PASS", time: "0.1s" },
    { id: 6, name: "Diagonal Load Transfer", status: "PASS", time: "0.2s" },
    { id: 7, name: "MORL 28-dim Pareto", status: "PASS", time: "120s" },
    { id: 8, name: "GP Uncertainty Bounds", status: "PASS", time: "2.1s" },
    { id: 9, name: "Wavelet Reconstruction", status: "PASS", time: "0.3s" },
    { id: 10, name: "Motor Torque Limits", status: "PASS", time: "0.4s" },
    { id: 11, name: "SOCP Solve Time", status: "WARN", time: "265ms" },
    { id: 12, name: "CBF Safety Filter", status: "PASS", time: "0.8s" },
    { id: 13, name: "DESC Convergence", status: "PASS", time: "3.2s" },
    { id: 14, name: "Launch State Machine", status: "PASS", time: "1.1s" },
    { id: 15, name: "Virtual Impedance", status: "PASS", time: "0.5s" },
    { id: 16, name: "Full Pipeline", status: "PASS", time: "180s" },
  ];

  const pass = tests.filter(t => t.status === "PASS").length;
  const warn = tests.filter(t => t.status === "WARN").length;

  return (
    <div style={{ ...gl, padding: "8px 10px", marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <div style={{ fontSize: 7, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: dt }}>SANITY CHECK SUITE — 16 TESTS</div>
        <div style={{ fontSize: 7, fontFamily: dt }}>
          <span style={{ color: C.gn, fontWeight: 700 }}>{pass} PASS</span>
          {warn > 0 && <span style={{ color: C.am, fontWeight: 700, marginLeft: 8 }}>{warn} WARN</span>}
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 4 }}>
        {tests.map(t => (
          <div key={t.id} style={{
            padding: "4px 6px", borderRadius: 3,
            background: t.status === "PASS" ? `${C.gn}08` : `${C.am}12`,
            border: `1px solid ${t.status === "PASS" ? `${C.gn}20` : `${C.am}30`}`,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 7, fontWeight: 700, color: t.status === "PASS" ? C.gn : C.am, fontFamily: dt }}>T{t.id}</span>
              <span style={{ fontSize: 6, color: C.dm, fontFamily: dt }}>{t.time}</span>
            </div>
            <div style={{ fontSize: 6.5, color: C.md, fontFamily: dt, marginTop: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{t.name}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// ARCHITECTURE SUMMARY — compact version of pillar descriptions
// ═══════════════════════════════════════════════════════════════════════════

function ArchitectureSummary() {
  const pillars = [
    { num: "01", name: "Neural Port-Hamiltonian", icon: "⬡", color: C.cy,
      eq: "ṗ = −∂H/∂q − R(q,p)·∂H/∂p + B·u",
      desc: "46-DOF energy-conserving dynamics. H_net residual MLP + softplus gate ensures zero ghost forces at equilibrium. R_net = LLᵀ + diag(softplus) guarantees PSD dissipation.",
      files: "vehicle_dynamics.py · 1,200 LOC" },
    { num: "02", name: "Multi-Fidelity Tire", icon: "⊗", color: C.gn,
      eq: "Fy = Pacejka(α,κ,Fz,γ) + PINN(·) ± 2σ_GP",
      desc: "Pacejka MF6.2 base + 5-node Jaeger thermal + PINN residual + Matérn 5/2 Sparse GP uncertainty envelope. Turn-slip corrected.",
      files: "tire_model.py · 1,800 LOC" },
    { num: "03", name: "Diff-WMPC", icon: "∿", color: C.am,
      eq: "min Σ ψ_k · |cD_k|₁ + ρ/2 · ||g(x)||²",
      desc: "3-level Db4 wavelet decomposition of control signals. Augmented Lagrangian with Pseudo-Huber smoothing. UT-generated stochastic safety tubes.",
      files: "ocp_solver.py · 900 LOC" },
    { num: "04", name: "MORL-SB-TRPO", icon: "◆", color: C.red,
      eq: "max E[R(grip,stab,lte)] s.t. D_KL(π||π_old) ≤ δ",
      desc: "Natural policy gradient over 40-dim setup. SMS-EMOA hypervolume pruning. ARD Bayesian optimisation cold-start. 3 Pareto axes: grip, stability, endurance LTE.",
      files: "evolutionary.py · 1,100 LOC" },
    { num: "05", name: "Powertrain Control", icon: "⏣", color: "#f97316",
      eq: "T* = argmin ||Mz − Mz_ref||² s.t. T ∈ friction_cone",
      desc: "13-stage pipeline: SOCP torque vectoring → DESC traction control → CBF safety filter → Launch B-spline → Virtual impedance. Single JIT entry point.",
      files: "powertrain_manager.py · 2,800 LOC" },
    { num: "06", name: "EKF Online Estimation", icon: "∂", color: "#22d3ee",
      eq: "x̂⁺ = x̂⁻ + K·(z − H·x̂⁻), α_peak 5th state",
      desc: "5-state differentiable EKF at 100 Hz. Learns friction scaling λ_μ, cornering stiffness C_α, CG height, and peak slip angle live from innovation error.",
      files: "differentiable_ekf.py · 450 LOC" },
  ];

  return (
    <div style={{ ...gl, padding: "10px 12px" }}>
      <div style={{ fontSize: 7, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: dt, marginBottom: 8 }}>ARCHITECTURE PILLARS — 100% DIFFERENTIABLE JAX/XLA</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {pillars.map(p => (
          <div key={p.num} style={{ borderLeft: `2px solid ${p.color}30`, paddingLeft: 10, paddingBottom: 6 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 3 }}>
              <span style={{ fontSize: 8, color: C.dm, fontFamily: dt }}>{p.num}</span>
              <span style={{ fontSize: 20, color: p.color, filter: `drop-shadow(0 0 4px ${p.color}30)` }}>{p.icon}</span>
              <span style={{ fontSize: 10, fontWeight: 700, color: C.w, fontFamily: hd }}>{p.name}</span>
            </div>
            <div style={{ fontSize: 8, color: p.color, fontFamily: dt, background: `${p.color}08`, padding: "3px 6px", borderRadius: 3, marginBottom: 4, whiteSpace: "pre", overflow: "auto" }}>
              {p.eq}
            </div>
            <div style={{ fontSize: 8, color: C.md, fontFamily: bd, lineHeight: 1.5, marginBottom: 3 }}>{p.desc}</div>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: dt }}>{p.files}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// FILE STRUCTURE — codebase overview
// ═══════════════════════════════════════════════════════════════════════════

function CodebaseOverview() {
  const modules = [
    { dir: "models/", files: 7, loc: 4200, desc: "Physics engine + tire + EKF + aero" },
    { dir: "optimization/", files: 7, loc: 3800, desc: "WMPC + MORL + track + lap sim" },
    { dir: "powertrain/", files: 8, loc: 2800, desc: "TV + TC + CBF + launch + impedance" },
    { dir: "suspension/", files: 5, loc: 1600, desc: "Kinematics + compliance + sweep" },
    { dir: "simulator/", files: 8, loc: 1900, desc: "Physics server + WS + ROS 2" },
    { dir: "visualization/", files: 18, loc: 5200, desc: "React dashboard 18 modules" },
    { dir: "config/", files: 4, loc: 800, desc: "Ter26/27 vehicle params" },
    { dir: "tests/", files: 2, loc: 900, desc: "Sanity checks (16 tests)" },
  ];
  const totalLOC = modules.reduce((s, m) => s + m.loc, 0);
  const totalFiles = modules.reduce((s, m) => s + m.files, 0);

  return (
    <div style={{ ...gl, padding: "8px 10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <div style={{ fontSize: 7, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: dt }}>CODEBASE</div>
        <div style={{ fontSize: 7, fontFamily: dt }}>
          <span style={{ color: C.cy }}>{totalFiles} files</span>
          <span style={{ color: C.dm, margin: "0 4px" }}>·</span>
          <span style={{ color: C.gn }}>{(totalLOC / 1000).toFixed(1)}k LOC</span>
          <span style={{ color: C.dm, margin: "0 4px" }}>·</span>
          <span style={{ color: C.am }}>0% NumPy</span>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6 }}>
        {modules.map(m => (
          <div key={m.dir} style={{ padding: "4px 6px", background: `${C.cy}05`, borderRadius: 3 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
              <span style={{ fontSize: 8, fontWeight: 700, color: C.cy, fontFamily: dt }}>{m.dir}</span>
              <span style={{ fontSize: 7, color: C.dm, fontFamily: dt }}>{m.loc} LOC</span>
            </div>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: dt }}>{m.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════

export default function OverviewModule({ pareto, conv }) {
  return (
    <div>
      {/* System-wide KPI strip */}
      <SystemKPIHeader pareto={pareto} conv={conv} />

      {/* Subsystem status grid */}
      <SubsystemGrid />

      {/* Pareto + Convergence charts */}
      <ParetoConvergenceRow pareto={pareto} conv={conv} />

      {/* Sanity check results */}
      <SanityCheckMatrix />

      {/* Architecture pillars */}
      <ArchitectureSummary />

      {/* Codebase overview */}
      <CodebaseOverview />
    </div>
  );
}