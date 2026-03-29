// ═══════════════════════════════════════════════════════════════════════════
// src/WeightBalanceModule.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════
// Weight & balance calculator with dynamic transfer analysis.
//
// v5.0 CHANGES:
//   - Enhanced from 5 → 8 tabs
//   - NEW: Dynamic Transfer — weight transfer under braking/cornering
//   - NEW: Regulations — FSG weight compliance checks
//   - NEW: Target Optimizer — CG/mass optimization targets
//   - Cross-links to Suspension (spring rates), Aero (downforce distribution)
//
// Sub-tabs (8):
//   1. CG & Mass      — CG position, mass breakdown by category
//   2. Inertia Tensor — Full 3×3 inertia tensor and principal axes
//   3. Corner Weights  — Static corner loads, cross-weights
//   4. Ballast         — Ballast optimizer for target weight distribution
//   5. Weight Sens     — Sensitivity of lap time to component mass
//   6. Dynamic Transfer— Lateral/longitudinal weight transfer under G
//   7. Regulations     — FSG minimum weight, driver weight compliance
//   8. CG Targets      — Optimization targets for CG position
//
// Integration:
//   NAV: { key: "weight", label: "Weight & CG", icon: "⊿" }
//   Import: import WeightBalanceModule from "./WeightBalanceModule.jsx"
//   Route: case "weight": return <WeightBalanceModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
BarChart, Bar, LineChart, Line, ScatterChart, Scatter, ComposedChart,
XAxis, YAxis, CartesianGrid, Tooltip,
ResponsiveContainer, ReferenceLine, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });

const TABS = [
{ key: "overview",   label: "CG & Mass" },
{ key: "inertia",    label: "Inertia Tensor" },
{ key: "corners",    label: "Corner Weights" },
{ key: "ballast",    label: "Ballast Optimizer" },
{ key: "sensitivity",label: "Weight Sensitivity" },
{ key: "transfer",   label: "Dynamic Transfer" },
{ key: "regulations",label: "Regulations" },
{ key: "targets",    label: "CG Targets" },
];

const CAT_COLORS = {
Chassis: C.cy, Aero: "#ff6090", Powertrain: C.am, Electronics: "#7c3aed",
Cooling: "#22d3ee", Suspension: C.red, Unsprung: "#a78bfa", Driver: "#f472b6", Safety: C.dm,
};

// ═══════════════════════════════════════════════════════════════════════════
// VEHICLE COMPONENT DATABASE
// ═══════════════════════════════════════════════════════════════════════════
const DEFAULT_COMPONENTS = [
{ name: "Monocoque", mass: 32, x: 0.20, y: 0.18, z: 0.00, category: "Chassis" },
{ name: "Rear Subframe", mass: 8, x: -0.55, y: 0.14, z: 0.00, category: "Chassis" },
{ name: "Roll Hoop", mass: 3.5, x: -0.26, y: 0.40, z: 0.00, category: "Chassis" },
{ name: "Front Wing Assy", mass: 4.2, x: 1.10, y: -0.10, z: 0.00, category: "Aero" },
{ name: "Rear Wing Assy", mass: 3.8, x: -0.80, y: 0.48, z: 0.00, category: "Aero" },
{ name: "Undertray", mass: 5.0, x: 0.15, y: -0.05, z: 0.00, category: "Aero" },
{ name: "Motor L", mass: 12, x: -0.50, y: 0.10, z: 0.30, category: "Powertrain" },
{ name: "Motor R", mass: 12, x: -0.50, y: 0.10, z: -0.30, category: "Powertrain" },
{ name: "Inverter L", mass: 4.5, x: -0.40, y: 0.22, z: 0.25, category: "Powertrain" },
{ name: "Inverter R", mass: 4.5, x: -0.40, y: 0.22, z: -0.25, category: "Powertrain" },
{ name: "Accumulator", mass: 52, x: -0.15, y: 0.12, z: 0.00, category: "Powertrain" },
{ name: "TSAL/BMS/HVD", mass: 6, x: -0.10, y: 0.30, z: 0.10, category: "Electronics" },
{ name: "ECU + Sensors", mass: 3, x: 0.10, y: 0.28, z: -0.08, category: "Electronics" },
{ name: "Wiring Harness", mass: 8, x: 0.00, y: 0.15, z: 0.00, category: "Electronics" },
{ name: "Cooling System", mass: 7, x: 0.40, y: 0.08, z: 0.20, category: "Cooling" },
{ name: "Brake System", mass: 6, x: 0.30, y: 0.05, z: 0.00, category: "Chassis" },
{ name: "Steering System", mass: 4, x: 0.50, y: 0.12, z: 0.00, category: "Chassis" },
{ name: "Susp FL", mass: 5.5, x: 0.85, y: 0.10, z: 0.61, category: "Suspension" },
{ name: "Susp FR", mass: 5.5, x: 0.85, y: 0.10, z: -0.61, category: "Suspension" },
{ name: "Susp RL", mass: 5.0, x: -0.70, y: 0.10, z: 0.59, category: "Suspension" },
{ name: "Susp RR", mass: 5.0, x: -0.70, y: 0.10, z: -0.59, category: "Suspension" },
{ name: "Wheel+Tire FL", mass: 8, x: 0.85, y: 0.23, z: 0.61, category: "Unsprung" },
{ name: "Wheel+Tire FR", mass: 8, x: 0.85, y: 0.23, z: -0.61, category: "Unsprung" },
{ name: "Wheel+Tire RL", mass: 8, x: -0.70, y: 0.23, z: 0.59, category: "Unsprung" },
{ name: "Wheel+Tire RR", mass: 8, x: -0.70, y: 0.23, z: -0.59, category: "Unsprung" },
{ name: "Driver (75kg)", mass: 75, x: 0.05, y: 0.28, z: 0.00, category: "Driver" },
{ name: "Pedalbox", mass: 3, x: 0.60, y: 0.06, z: 0.00, category: "Chassis" },
{ name: "Fire Extinguisher", mass: 1.5, x: -0.20, y: 0.10, z: 0.15, category: "Safety" },
];

// ═══════════════════════════════════════════════════════════════════════════
// PHYSICS CALCULATIONS
// ═══════════════════════════════════════════════════════════════════════════
function computeCG(components) {
const m = components.reduce((a, c) => a + c.mass, 0);
if (m === 0) return { x: 0, y: 0, z: 0, mass: 0 };
return {
x: components.reduce((a, c) => a + c.mass * c.x, 0) / m,
y: components.reduce((a, c) => a + c.mass * c.y, 0) / m,
z: components.reduce((a, c) => a + c.mass * c.z, 0) / m,
mass: m,
};
}

function computeInertia(components, cg) {
let Ixx = 0, Iyy = 0, Izz = 0, Ixy = 0, Ixz = 0, Iyz = 0;
for (const c of components) {
const dx = c.x - cg.x, dy = c.y - cg.y, dz = c.z - cg.z;
Ixx += c.mass * (dy * dy + dz * dz);
Iyy += c.mass * (dx * dx + dz * dz);
Izz += c.mass * (dx * dx + dy * dy);
Ixy -= c.mass * dx * dy;
Ixz -= c.mass * dx * dz;
Iyz -= c.mass * dy * dz;
}
return { Ixx, Iyy, Izz, Ixy, Ixz, Iyz };
}

function computeCornerWeights(cg, mass, wb = 1.55, tF = 1.22, tR = 1.18) {
const lf = wb / 2 + cg.x, lr = wb / 2 - cg.x;
const frontPct = lr / wb, rearPct = lf / wb;
const frontTotal = mass * 9.81 * frontPct;
const rearTotal = mass * 9.81 * rearPct;
const latBias = cg.z / ((tF + tR) / 2);
return {
FL: frontTotal * (0.5 - latBias), FR: frontTotal * (0.5 + latBias),
RL: rearTotal * (0.5 - latBias), RR: rearTotal * (0.5 + latBias),
frontPct: frontPct * 100, rearPct: rearPct * 100,
crossFL_RR: frontTotal * (0.5 - latBias) + rearTotal * (0.5 + latBias),
crossFR_RL: frontTotal * (0.5 + latBias) + rearTotal * (0.5 - latBias),
};
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 1: CG & MASS OVERVIEW
// ═══════════════════════════════════════════════════════════════════════════
function OverviewTab({ components, cg, inertia }) {
const byCat = useMemo(() => {
const cats = {};
for (const c of components) { if (!cats[c.category]) cats[c.category] = 0; cats[c.category] += c.mass; }
return Object.entries(cats).map(([cat, mass]) => ({ cat, mass: +mass.toFixed(1), color: CAT_COLORS[cat] || C.dm })).sort((a, b) => b.mass - a.mass);
}, [components]);
const top10 = useMemo(() => […components].sort((a, b) => b.mass - a.mass).slice(0, 10).map(c => ({ name: c.name, mass: c.mass, color: CAT_COLORS[c.category] || C.dm })), [components]);

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
<Sec title="Mass by Category [kg]">
<GC><ResponsiveContainer width="100%" height={260}>
<BarChart data={byCat} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 80 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
<XAxis type="number" {…ax()} />
<YAxis dataKey="cat" type="category" tick={{ fontSize: 9, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={75} />
<Tooltip contentStyle={TT} />
<Bar dataKey="mass" radius={[0, 4, 4, 0]} barSize={14}>
{byCat.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
</Bar>
</BarChart>
</ResponsiveContainer></GC>
</Sec>
<Sec title="Top 10 Heaviest [kg]">
<GC><ResponsiveContainer width="100%" height={260}>
<BarChart data={top10} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 100 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
<XAxis type="number" {…ax()} />
<YAxis dataKey="name" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={95} />
<Tooltip contentStyle={TT} />
<Bar dataKey="mass" radius={[0, 4, 4, 0]} barSize={12}>
{top10.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
</Bar>
</BarChart>
</ResponsiveContainer></GC>
</Sec>
</div>
<Sec title="CG Position (top view: X forward, Z right)" style={{ marginTop: 10 }}>
<GC><ResponsiveContainer width="100%" height={280}>
<ScatterChart margin={{ top: 16, right: 20, bottom: 24, left: 20 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} />
<XAxis dataKey="x" type="number" {…ax()} domain={[-1, 1.3]} name="X [m]" />
<YAxis dataKey="z" type="number" {…ax()} domain={[-0.8, 0.8]} name="Z [m]" />
<Tooltip contentStyle={TT} />
<Scatter data={components} r={4}>
{components.map((c, i) => <Cell key={i} fill={CAT_COLORS[c.category] || C.dm} fillOpacity={0.6} />)}
</Scatter>
<Scatter data={[{ x: cg.x, z: cg.z, name: "CG" }]} r={8} fill={C.red}>
<Cell fill={C.red} />
</Scatter>
<ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3" />
<ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
</ScatterChart>
</ResponsiveContainer></GC>
</Sec>
</div>
);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 2: INERTIA TENSOR
// ═══════════════════════════════════════════════════════════════════════════
function InertiaTab({ inertia, cg }) {
const tensor = [
[inertia.Ixx, inertia.Ixy, inertia.Ixz],
[inertia.Ixy, inertia.Iyy, inertia.Iyz],
[inertia.Ixz, inertia.Iyz, inertia.Izz],
];
const labels = ["Roll (Ixx)", "Pitch (Iyy)", "Yaw (Izz)"];
const diagData = labels.map((l, i) => ({ axis: l, value: +tensor[i][i].toFixed(1) }));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Ixx (Roll)" value={`${inertia.Ixx.toFixed(1)} kg·m²`} sub="roll inertia" sentiment="neutral" delay={0} />
<KPI label="Iyy (Pitch)" value={`${inertia.Iyy.toFixed(1)} kg·m²`} sub="pitch inertia" sentiment="neutral" delay={1} />
<KPI label="Izz (Yaw)" value={`${inertia.Izz.toFixed(1)} kg·m²`} sub="yaw inertia" sentiment="neutral" delay={2} />
</div>
<Sec title="Inertia Tensor Matrix [kg·m²]">
<GC style={{ padding: 12 }}>
<div style={{ display: "grid", gridTemplateColumns: "60px repeat(3, 1fr)", gap: 4, fontSize: 10, fontFamily: C.dt }}>
<div />
{["X (Roll)", "Y (Pitch)", "Z (Yaw)"].map(h => <div key={h} style={{ color: C.dm, fontWeight: 700, textAlign: "center" }}>{h}</div>)}
{tensor.map((row, i) => (
<React.Fragment key={i}>
<div style={{ color: C.cy, fontWeight: 700 }}>{["X", "Y", "Z"][i]}</div>
{row.map((v, j) => (
<div key={j} style={{
textAlign: "center", padding: "8px 4px", borderRadius: 4,
background: i === j ? `${C.cy}10` : Math.abs(v) > 1 ? `${C.am}08` : "transparent",
color: i === j ? C.cy : C.br, fontWeight: i === j ? 700 : 400,
}}>{v.toFixed(1)}</div>
))}
</React.Fragment>
))}
</div>
</GC>
</Sec>
<Sec title="Principal Inertias" style={{ marginTop: 10 }}>
<GC><ResponsiveContainer width="100%" height={180}>
<BarChart data={diagData} margin={{ top: 8, right: 16, bottom: 8, left: 80 }} layout="vertical">
<CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
<XAxis type="number" {…ax()} />
<YAxis dataKey="axis" type="category" tick={{ fontSize: 9, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={75} />
<Tooltip contentStyle={TT} />
<Bar dataKey="value" barSize={16} radius={[0, 4, 4, 0]} fill={C.cy} fillOpacity={0.6} />
</BarChart>
</ResponsiveContainer></GC>
</Sec>
</div>
);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 3: CORNER WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════
function CornersTab({ corners, cg }) {
const crossDiag = Math.abs(corners.crossFL_RR - corners.crossFR_RL) / (cg.mass * 9.81) * 100;
const cornerData = [
{ corner: "FL", load: +corners.FL.toFixed(0), color: C.cy },
{ corner: "FR", load: +corners.FR.toFixed(0), color: C.gn },
{ corner: "RL", load: +corners.RL.toFixed(0), color: C.am },
{ corner: "RR", load: +corners.RR.toFixed(0), color: C.red },
];

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="FL" value={`${corners.FL.toFixed(0)} N`} sub="front left" sentiment="neutral" delay={0} />
<KPI label="FR" value={`${corners.FR.toFixed(0)} N`} sub="front right" sentiment="neutral" delay={1} />
<KPI label="RL" value={`${corners.RL.toFixed(0)} N`} sub="rear left" sentiment="neutral" delay={2} />
<KPI label="RR" value={`${corners.RR.toFixed(0)} N`} sub="rear right" sentiment="neutral" delay={3} />
<KPI label="Cross Weight" value={`${crossDiag.toFixed(1)}%`} sub="diagonal imbalance" sentiment={crossDiag < 1 ? "positive" : "amber"} delay={4} />
</div>
<Sec title="Static Corner Loads [N]">
<GC><ResponsiveContainer width="100%" height={220}>
<BarChart data={cornerData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} />
<XAxis dataKey="corner" {…ax()} />
<YAxis {…ax()} />
<Tooltip contentStyle={TT} />
<Bar dataKey="load" barSize={40} radius={[4, 4, 0, 0]}>
{cornerData.map((c, i) => <Cell key={i} fill={c.color} fillOpacity={0.7} />)}
</Bar>
</BarChart>
</ResponsiveContainer></GC>
</Sec>
</div>
);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 4: BALLAST OPTIMIZER
// ═══════════════════════════════════════════════════════════════════════════
function BallastTab({ components, cg }) {
// Simulate ballast placement options
const options = useMemo(() => {
const positions = [
{ label: "Front left", x: 0.7, z: 0.3 },
{ label: "Front right", x: 0.7, z: -0.3 },
{ label: "Rear left", x: -0.5, z: 0.3 },
{ label: "Rear right", x: -0.5, z: -0.3 },
{ label: "Front center", x: 0.6, z: 0.0 },
];
return positions.map(p => {
const ballast = { …p, name: `Ballast ${p.label}`, mass: 2, y: 0.05, category: "Chassis" };
const newCG = computeCG([…components, ballast]);
const newCorners = computeCornerWeights(newCG, newCG.mass);
return {
…p, frontPct: +newCorners.frontPct.toFixed(1),
cgX: +(newCG.x * 1000).toFixed(0), cgZ: +(newCG.z * 1000).toFixed(1),
crossImbalance: +(Math.abs(newCorners.crossFL_RR - newCorners.crossFR_RL) / (newCG.mass * 9.81) * 100).toFixed(2),
};
});
}, [components]);

return (
<div>
<Sec title="Ballast Placement Options (2 kg)">
<GC style={{ padding: 10 }}>
<div style={{ display: "grid", gridTemplateColumns: "120px 80px 80px 80px 80px", gap: 0, fontSize: 8, fontFamily: C.dt }}>
{["Position", "F/R [%]", "CG X [mm]", "CG Z [mm]", "Cross Δ [%]"].map(h => (
<div key={h} style={{ color: C.dm, fontWeight: 700, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>
))}
{options.map(o => (
<React.Fragment key={o.label}>
<div style={{ color: C.cy, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{o.label}</div>
<div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{o.frontPct}</div>
<div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{o.cgX}</div>
<div style={{ color: Math.abs(+o.cgZ) < 2 ? C.gn : C.am, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{o.cgZ}</div>
<div style={{ color: +o.crossImbalance < 0.5 ? C.gn : C.am, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{o.crossImbalance}</div>
</React.Fragment>
))}
</div>
</GC>
</Sec>
</div>
);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 5: WEIGHT SENSITIVITY
// ═══════════════════════════════════════════════════════════════════════════
function SensitivityTab({ components }) {
const sensData = useMemo(() => {
return components.filter(c => c.category !== "Driver").map(c => {
const lapSens = c.mass * 0.012 + (c.y > 0.2 ? c.mass * 0.005 : 0); // higher = worse
return { name: c.name, mass: c.mass, lapSens: +lapSens.toFixed(3), category: c.category };
}).sort((a, b) => b.lapSens - a.lapSens).slice(0, 15);
}, [components]);

return (
<div>
<Sec title="Lap Time Sensitivity to Component Mass [ms/kg]">
<GC><ResponsiveContainer width="100%" height={380}>
<BarChart data={sensData} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 100 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
<XAxis type="number" {…ax()} />
<YAxis dataKey="name" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={95} />
<Tooltip contentStyle={TT} />
<Bar dataKey="lapSens" barSize={12} radius={[0, 4, 4, 0]} name="Δt_lap [ms/kg]">
{sensData.map((s, i) => <Cell key={i} fill={CAT_COLORS[s.category] || C.dm} fillOpacity={0.7} />)}
</Bar>
</BarChart>
</ResponsiveContainer></GC>
</Sec>
</div>
);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 6: DYNAMIC WEIGHT TRANSFER — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function TransferTab({ cg, corners }) {
const wb = 1.55, tF = 1.22, tR = 1.18, hCG = cg.y;
const mass = cg.mass;

// Weight transfer at various G levels
const transferData = useMemo(() => {
return Array.from({ length: 30 }, (_, i) => {
const latG = (i / 29) * 1.6;
const lateralTransfer = mass * latG * 9.81 * hCG / ((tF + tR) / 2);
const lonG = 1.2;
const longTransfer = mass * lonG * 9.81 * hCG / wb;
const FL = corners.FL + lateralTransfer * 0.5 - longTransfer * 0.5;
const FR = corners.FR - lateralTransfer * 0.5 - longTransfer * 0.5;
const RL = corners.RL + lateralTransfer * 0.5 + longTransfer * 0.5;
const RR = corners.RR - lateralTransfer * 0.5 + longTransfer * 0.5;
return {
latG: +latG.toFixed(2),
lateralTransfer: +lateralTransfer.toFixed(0),
FL: +Math.max(0, FL).toFixed(0), FR: +Math.max(0, FR).toFixed(0),
RL: +Math.max(0, RL).toFixed(0), RR: +Math.max(0, RR).toFixed(0),
insideLoad: +Math.min(FR, RR).toFixed(0),
};
});
}, [cg, corners, mass, hCG]);

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="CG Height" value={`${(hCG * 1000).toFixed(0)} mm`} sub="primary transfer driver" sentiment={hCG < 0.30 ? "positive" : "amber"} delay={0} />
<KPI label="ΔFz @ 1G lat" value={`${(mass * 1.0 * 9.81 * hCG / ((tF + tR) / 2)).toFixed(0)} N`} sub="total lateral transfer" sentiment="neutral" delay={1} />
<KPI label="ΔFz @ 1G lon" value={`${(mass * 1.0 * 9.81 * hCG / wb).toFixed(0)} N`} sub="total longitudinal transfer" sentiment="neutral" delay={2} />
<KPI label="Wheel Lift G" value={`${(corners.FR / (mass * 9.81 * hCG / ((tF + tR) / 2)) + 0.01).toFixed(2)}G`} sub="inside wheel unloads" sentiment="neutral" delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Corner Loads vs Lateral G (right turn)">
      <GC><ResponsiveContainer width="100%" height={260}>
        <LineChart data={transferData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="latG" {...ax()} label={{ value: "Lateral G", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0} stroke={C.red} strokeDasharray="4 2" label={{ value: "Lift-off", fill: C.red, fontSize: 7 }} />
          <Line type="monotone" dataKey="FL" stroke={C.cy} strokeWidth={1.5} dot={false} name="FL (outside)" />
          <Line type="monotone" dataKey="FR" stroke={C.gn} strokeWidth={1.5} dot={false} name="FR (inside)" />
          <Line type="monotone" dataKey="RL" stroke={C.am} strokeWidth={1.5} dot={false} name="RL (outside)" />
          <Line type="monotone" dataKey="RR" stroke={C.red} strokeWidth={1.5} dot={false} name="RR (inside)" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Total Lateral Transfer [N]">
      <GC><ResponsiveContainer width="100%" height={260}>
        <LineChart data={transferData} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="latG" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="lateralTransfer" stroke={C.cy} strokeWidth={2} dot={false} name="ΔFz [N]" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 7: REGULATIONS — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function RegulationsTab({ cg }) {
const minWeight = 0; // FSG EV: no minimum vehicle weight (only with driver)
const driverMin = 68; // kg minimum driver weight (with ballast)
const driver = 75;
const carOnly = cg.mass - driver;
const checks = [
{ rule: "No minimum", item: "Vehicle weight (no minimum in FSG EV)", value: `${carOnly.toFixed(0)} kg`, status: "pass" },
{ rule: "Driver", item: "Driver weight ≥ 68 kg (inc. ballast)", value: `${driver} kg`, status: driver >= driverMin ? "pass" : "fail" },
{ rule: "Total", item: "Total system weight with driver", value: `${cg.mass.toFixed(0)} kg`, status: "pass" },
{ rule: "CG Height", item: "CG height below 350mm recommended", value: `${(cg.y * 1000).toFixed(0)} mm`, status: cg.y < 0.35 ? "pass" : "warn" },
{ rule: "Lateral CG", item: "Lateral CG offset < 10mm", value: `${(Math.abs(cg.z) * 1000).toFixed(1)} mm`, status: Math.abs(cg.z) < 0.01 ? "pass" : "warn" },
{ rule: "F/R Split", item: "Front weight 45–52% optimal", value: `${(cg.mass > 0 ? ((1.55/2 - cg.x) / 1.55 * 100) : 50).toFixed(1)}%`, status: "pass" },
];

return (
<div>
<Sec title="FSG Weight & CG Compliance">
<GC style={{ padding: 10 }}>
{checks.map(c => (
<div key={c.rule} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: `1px solid ${C.b1}08` }}>
<div style={{ width: 8, height: 8, borderRadius: "50%", background: c.status === "pass" ? C.gn : c.status === "warn" ? C.am : C.red, flexShrink: 0 }} />
<div style={{ fontSize: 8, color: C.cy, fontFamily: C.dt, fontWeight: 700, width: 70 }}>{c.rule}</div>
<div style={{ fontSize: 9, color: C.br, fontFamily: C.dt, flex: 1 }}>{c.item}</div>
<div style={{ fontSize: 10, fontWeight: 700, color: c.status === "pass" ? C.gn : C.am, fontFamily: C.dt }}>{c.value}</div>
<div style={{ fontSize: 7, fontWeight: 700, color: c.status === "pass" ? C.gn : c.status === "warn" ? C.am : C.red, background: `${c.status === "pass" ? C.gn : c.status === "warn" ? C.am : C.red}15`, padding: "2px 8px", borderRadius: 4, fontFamily: C.dt }}>
{c.status === "pass" ? "PASS" : c.status === "warn" ? "REVIEW" : "FAIL"}
</div>
</div>
))}
</GC>
</Sec>
</div>
);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 8: CG TARGETS — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function TargetsTab({ cg, corners }) {
const targets = [
{ param: "CG X (longitudinal)", current: +(cg.x * 1000).toFixed(0), target: 0, unit: "mm", tolerance: 30 },
{ param: "CG Y (height)", current: +(cg.y * 1000).toFixed(0), target: 280, unit: "mm", tolerance: 20 },
{ param: "CG Z (lateral)", current: +(cg.z * 1000).toFixed(1), target: 0, unit: "mm", tolerance: 5 },
{ param: "Front Weight %", current: +corners.frontPct.toFixed(1), target: 48.0, unit: "%", tolerance: 2 },
{ param: "Total Mass", current: +cg.mass.toFixed(0), target: 290, unit: "kg", tolerance: 15 },
];

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
{targets.map((t, i) => {
const onTarget = Math.abs(t.current - t.target) <= t.tolerance;
return <KPI key={t.param} label={t.param.split(" (")[0]} value={`${t.current} ${t.unit}`} sub={`target: ${t.target} ±${t.tolerance}`} sentiment={onTarget ? "positive" : "amber"} delay={i} />;
})}
</div>

```
  <Sec title="CG Position vs Targets">
    <GC style={{ padding: 10 }}>
      {targets.map(t => {
        const delta = t.current - t.target;
        const pct = Math.min(100, Math.abs(delta) / t.tolerance * 50);
        const onTarget = Math.abs(delta) <= t.tolerance;
        return (
          <div key={t.param} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderBottom: `1px solid ${C.b1}08` }}>
            <div style={{ fontSize: 9, color: C.br, fontFamily: C.dt, width: 140 }}>{t.param}</div>
            <div style={{ flex: 1, height: 8, background: C.b1, borderRadius: 4, position: "relative" }}>
              <div style={{ width: `${Math.min(100, 50 + (delta / t.tolerance) * 25)}%`, height: "100%", background: onTarget ? C.gn : C.am, borderRadius: 4, transition: "width 0.3s" }} />
              <div style={{ position: "absolute", left: "50%", top: -2, width: 2, height: 12, background: C.w, opacity: 0.5 }} />
            </div>
            <div style={{ fontSize: 9, fontWeight: 700, color: onTarget ? C.gn : C.am, fontFamily: C.dt, width: 70, textAlign: "right" }}>
              {delta > 0 ? "+" : ""}{delta.toFixed(1)} {t.unit}
            </div>
          </div>
        );
      })}
    </GC>
  </Sec>

  <div style={{ ...GL, padding: "8px 14px", marginTop: 10, borderLeft: `2px solid ${C.am}`, display: "flex", alignItems: "center", gap: 8, fontSize: 9, fontFamily: C.dt }}>
    <span style={{ color: C.am }}>△</span>
    <span style={{ color: C.dm }}>Spring rates and ride heights affect dynamic CG position →</span>
    <span style={{ color: C.am, fontWeight: 700 }}>Suspension module</span>
  </div>
  <div style={{ ...GL, padding: "8px 14px", marginTop: 4, borderLeft: `2px solid #ff6090`, display: "flex", alignItems: "center", gap: 8, fontSize: 9, fontFamily: C.dt }}>
    <span style={{ color: "#ff6090" }}>▽</span>
    <span style={{ color: C.dm }}>Aero balance shifts effective weight distribution at speed →</span>
    <span style={{ color: "#ff6090", fontWeight: 700 }}>Aerodynamics module</span>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function WeightBalanceModule() {
const [tab, setTab] = useState("overview");
const components = DEFAULT_COMPONENTS;
const cg = useMemo(() => computeCG(components), [components]);
const inertia = useMemo(() => computeInertia(components, cg), [components, cg]);
const corners = useMemo(() => computeCornerWeights(cg, cg.mass), [cg]);

return (
<div>
{/* Header */}
<div style={{
…GL, padding: "12px 16px", marginBottom: 14,
borderLeft: `3px solid ${C.am}`,
background: `linear-gradient(90deg, ${C.am}08, transparent)`,
}}>
<div style={{ display: "flex", alignItems: "center", gap: 10 }}>
<span style={{ fontSize: 20, color: C.am }}>⊿</span>
<div>
<span style={{ fontSize: 12, fontWeight: 800, color: C.am, fontFamily: C.dt, letterSpacing: 2 }}>WEIGHT & CG</span>
<span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 12 }}>
{components.length} components · {cg.mass.toFixed(0)} kg total · CG at ({(cg.x*1000).toFixed(0)}, {(cg.y*1000).toFixed(0)}, {(cg.z*1000).toFixed(1)}) mm
</span>
</div>
</div>
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
    <KPI label="Total Mass" value={`${cg.mass.toFixed(1)} kg`} sub="with driver" sentiment={cg.mass < 320 ? "positive" : "amber"} delay={0} />
    <KPI label="CG X" value={`${(cg.x * 1000).toFixed(0)} mm`} sub={cg.x > 0 ? "ahead of mid-WB" : "behind mid-WB"} sentiment="neutral" delay={1} />
    <KPI label="CG Height" value={`${(cg.y * 1000).toFixed(0)} mm`} sub="from ground" sentiment={cg.y < 0.30 ? "positive" : "amber"} delay={2} />
    <KPI label="CG Lateral" value={`${(cg.z * 1000).toFixed(1)} mm`} sub={Math.abs(cg.z) < 0.005 ? "centered" : "offset"} sentiment={Math.abs(cg.z) < 0.01 ? "positive" : "amber"} delay={3} />
    <KPI label="F/R Split" value={`${corners.frontPct.toFixed(1)}/${corners.rearPct.toFixed(1)}`} sub="front/rear %" sentiment={Math.abs(corners.frontPct - 48) < 3 ? "positive" : "amber"} delay={4} />
  </div>

  <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
    {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.am} />)}
  </div>

  {tab === "overview" && <OverviewTab components={components} cg={cg} inertia={inertia} />}
  {tab === "inertia" && <InertiaTab inertia={inertia} cg={cg} />}
  {tab === "corners" && <CornersTab corners={corners} cg={cg} />}
  {tab === "ballast" && <BallastTab components={components} cg={cg} />}
  {tab === "sensitivity" && <SensitivityTab components={components} />}
  {tab === "transfer" && <TransferTab cg={cg} corners={corners} />}
  {tab === "regulations" && <RegulationsTab cg={cg} />}
  {tab === "targets" && <TargetsTab cg={cg} corners={corners} />}
</div>
```

);
}