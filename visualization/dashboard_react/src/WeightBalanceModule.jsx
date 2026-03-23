// ═══════════════════════════════════════════════════════════════════════════
// src/WeightBalanceModule.jsx — Project-GP Dashboard v4.2
// ═══════════════════════════════════════════════════════════════════════════
// Weight & balance calculator: CG position, inertia tensor, corner weights,
// ballast optimization, weight sensitivity analysis.
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
const Lbl = ({children, color}) => <span style={{fontSize:8, fontWeight:700, color:color||C.dm, fontFamily:C.dt, letterSpacing:1.5, textTransform:"uppercase"}}>{children}</span>;
// ═══════════════════════════════════════════════════════════════════════════
// VEHICLE COMPONENT DATABASE
// ═══════════════════════════════════════════════════════════════════════════
const DEFAULT_COMPONENTS = [
  { name: "Monocoque",          mass: 32,  x: 0.20,  y: 0.18,  z: 0.00, category: "Chassis" },
  { name: "Rear Subframe",      mass: 8,   x: -0.55, y: 0.14,  z: 0.00, category: "Chassis" },
  { name: "Roll Hoop",          mass: 3.5, x: -0.26, y: 0.40,  z: 0.00, category: "Chassis" },
  { name: "Front Wing Assy",    mass: 4.2, x: 1.10,  y: -0.10, z: 0.00, category: "Aero" },
  { name: "Rear Wing Assy",     mass: 3.8, x: -0.80, y: 0.48,  z: 0.00, category: "Aero" },
  { name: "Undertray",          mass: 5.0, x: 0.15,  y: -0.05, z: 0.00, category: "Aero" },
  { name: "Motor L",            mass: 12,  x: -0.50, y: 0.10,  z: 0.30, category: "Powertrain" },
  { name: "Motor R",            mass: 12,  x: -0.50, y: 0.10,  z: -0.30, category: "Powertrain" },
  { name: "Inverter L",         mass: 4.5, x: -0.40, y: 0.22,  z: 0.25, category: "Powertrain" },
  { name: "Inverter R",         mass: 4.5, x: -0.40, y: 0.22,  z: -0.25, category: "Powertrain" },
  { name: "Accumulator",        mass: 52,  x: -0.15, y: 0.12,  z: 0.00, category: "Powertrain" },
  { name: "TSAL/BMS/HVD",       mass: 6,   x: -0.10, y: 0.30,  z: 0.10, category: "Electronics" },
  { name: "ECU + Sensors",      mass: 3,   x: 0.10,  y: 0.28,  z: -0.08, category: "Electronics" },
  { name: "Wiring Harness",     mass: 8,   x: 0.00,  y: 0.15,  z: 0.00, category: "Electronics" },
  { name: "Cooling System",     mass: 7,   x: 0.40,  y: 0.08,  z: 0.20, category: "Cooling" },
  { name: "Brake System",       mass: 6,   x: 0.30,  y: 0.05,  z: 0.00, category: "Chassis" },
  { name: "Steering System",    mass: 4,   x: 0.50,  y: 0.12,  z: 0.00, category: "Chassis" },
  { name: "Susp FL",            mass: 5.5, x: 0.85,  y: 0.10,  z: 0.61, category: "Suspension" },
  { name: "Susp FR",            mass: 5.5, x: 0.85,  y: 0.10,  z: -0.61, category: "Suspension" },
  { name: "Susp RL",            mass: 5.0, x: -0.70, y: 0.10,  z: 0.59, category: "Suspension" },
  { name: "Susp RR",            mass: 5.0, x: -0.70, y: 0.10,  z: -0.59, category: "Suspension" },
  { name: "Wheel+Tire FL",      mass: 8,   x: 0.85,  y: 0.23,  z: 0.61, category: "Unsprung" },
  { name: "Wheel+Tire FR",      mass: 8,   x: 0.85,  y: 0.23,  z: -0.61, category: "Unsprung" },
  { name: "Wheel+Tire RL",      mass: 8,   x: -0.70, y: 0.23,  z: 0.59, category: "Unsprung" },
  { name: "Wheel+Tire RR",      mass: 8,   x: -0.70, y: 0.23,  z: -0.59, category: "Unsprung" },
  { name: "Driver (75kg)",      mass: 75,  x: 0.05,  y: 0.28,  z: 0.00, category: "Driver" },
  { name: "Pedalbox",           mass: 3,   x: 0.60,  y: 0.06,  z: 0.00, category: "Chassis" },
  { name: "Fire Extinguisher",  mass: 1.5, x: -0.20, y: 0.10,  z: 0.15, category: "Safety" },
];

function computeCG(components) {
  const totalMass = components.reduce((a, c) => a + c.mass, 0);
  if (totalMass === 0) return { x: 0, y: 0, z: 0, mass: 0 };
  const x = components.reduce((a, c) => a + c.mass * c.x, 0) / totalMass;
  const y = components.reduce((a, c) => a + c.mass * c.y, 0) / totalMass;
  const z = components.reduce((a, c) => a + c.mass * c.z, 0) / totalMass;
  return { x, y, z, mass: totalMass };
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
    crossFL_RR: (frontTotal * (0.5 - latBias) + rearTotal * (0.5 + latBias)),
    crossFR_RL: (frontTotal * (0.5 + latBias) + rearTotal * (0.5 - latBias)),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════════════════════
const TABS = [
  { key: "overview", label: "CG & Mass" },
  { key: "inertia", label: "Inertia Tensor" },
  { key: "corners", label: "Corner Weights" },
  { key: "ballast", label: "Ballast Optimizer" },
  { key: "sensitivity", label: "Weight Sensitivity" },
];

const CAT_COLORS = {
  Chassis: C.cy, Aero: "#ff6090", Powertrain: C.am, Electronics: C.gn,
  Cooling: "#22d3ee", Suspension: C.red, Unsprung: "#a78bfa", Driver: "#f472b6", Safety: C.dm,
};

// ═══════════════════════════════════════════════════════════════════════════
// CG & MASS OVERVIEW
// ═══════════════════════════════════════════════════════════════════════════
function OverviewTab({ components, cg, inertia }) {
  const byCat = useMemo(() => {
    const cats = {};
    for (const c of components) {
      if (!cats[c.category]) cats[c.category] = 0;
      cats[c.category] += c.mass;
    }
    return Object.entries(cats).map(([cat, mass]) => ({ cat, mass: +mass.toFixed(1), color: CAT_COLORS[cat] || C.dm })).sort((a, b) => b.mass - a.mass);
  }, [components]);

  const top10 = useMemo(() =>
    [...components].sort((a, b) => b.mass - a.mass).slice(0, 10).map(c => ({ name: c.name, mass: c.mass, color: CAT_COLORS[c.category] || C.dm })),
    [components]
  );

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Mass by Category [kg]">
          <GC><ResponsiveContainer width="100%" height={260}>
            <BarChart data={byCat} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
              <XAxis type="number" {...ax()} />
              <YAxis dataKey="cat" type="category" tick={{ fontSize: 9, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={75} />
              <Tooltip contentStyle={TT} />
              <Bar dataKey="mass" radius={[0, 4, 4, 0]} barSize={14}>
                {byCat.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Heaviest Components [kg]">
          <GC><ResponsiveContainer width="100%" height={260}>
            <BarChart data={top10} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
              <XAxis type="number" {...ax()} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={95} />
              <Tooltip contentStyle={TT} />
              <Bar dataKey="mass" radius={[0, 4, 4, 0]} barSize={12}>
                {top10.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>

      <Sec title="CG Position (top view: X forward, Z right)">
        <GC><ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 16, right: 20, bottom: 24, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis type="number" dataKey="x" {...ax()} domain={[-1.2, 1.4]} label={{ value: "X [m] →forward", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis type="number" dataKey="z" {...ax()} domain={[-0.8, 0.8]} label={{ value: "Z [m] →right", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
            <Tooltip contentStyle={TT} />
            <Scatter data={components.map(c => ({ ...c, r: Math.max(3, c.mass / 8) }))} name="Components">
              {components.map((c, i) => <Cell key={i} fill={CAT_COLORS[c.category] || C.dm} fillOpacity={0.5} r={Math.max(3, c.mass / 8)} />)}
            </Scatter>
            <Scatter data={[{ x: cg.x, z: cg.z, name: "CG" }]} fill={C.red} r={8} name="CG" legendType="diamond" />
            <ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </ScatterChart>
        </ResponsiveContainer></GC>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// INERTIA TENSOR TAB
// ═══════════════════════════════════════════════════════════════════════════
function InertiaTab({ inertia, cg }) {
  const tensor = [
    [inertia.Ixx, inertia.Ixy, inertia.Ixz],
    [inertia.Ixy, inertia.Iyy, inertia.Iyz],
    [inertia.Ixz, inertia.Iyz, inertia.Izz],
  ];
  const labels = ["Roll (Ixx)", "Pitch (Iyy)", "Yaw (Izz)"];
  const diag = [inertia.Ixx, inertia.Iyy, inertia.Izz];
  const yawRollRatio = inertia.Izz / inertia.Ixx;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Ixx (Roll)" value={`${inertia.Ixx.toFixed(1)}`} sub="kg·m²" sentiment="neutral" delay={0} />
        <KPI label="Iyy (Pitch)" value={`${inertia.Iyy.toFixed(1)}`} sub="kg·m²" sentiment="neutral" delay={1} />
        <KPI label="Izz (Yaw)" value={`${inertia.Izz.toFixed(1)}`} sub="kg·m²" sentiment="neutral" delay={2} />
        <KPI label="Izz/Ixx" value={`${yawRollRatio.toFixed(2)}`} sub={yawRollRatio < 1.5 ? "compact" : "elongated"} sentiment={yawRollRatio < 1.8 ? "positive" : "amber"} delay={3} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Inertia Tensor Matrix [kg·m²]">
          <GC style={{ padding: "16px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "60px repeat(3, 1fr)", gap: 0, fontFamily: C.dt, fontSize: 10 }}>
              <div />
              {["X (roll)", "Y (pitch)", "Z (yaw)"].map(h => (
                <div key={h} style={{ textAlign: "center", color: C.dm, fontSize: 8, fontWeight: 700, letterSpacing: 1, padding: "6px 0" }}>{h}</div>
              ))}
              {tensor.map((row, i) => (
                <React.Fragment key={i}>
                  <div style={{ color: C.dm, fontSize: 8, fontWeight: 700, padding: "10px 4px", textAlign: "right" }}>{["X", "Y", "Z"][i]}</div>
                  {row.map((val, j) => (
                    <div key={j} style={{
                      textAlign: "center", padding: "10px 4px",
                      color: i === j ? C.cy : Math.abs(val) > 1 ? C.am : C.dm,
                      fontWeight: i === j ? 700 : 400,
                      background: i === j ? `${C.cy}08` : "transparent",
                      borderRadius: 4,
                    }}>
                      {val.toFixed(1)}
                    </div>
                  ))}
                </React.Fragment>
              ))}
            </div>
          </GC>
        </Sec>

        <Sec title="Principal Moments">
          <GC><ResponsiveContainer width="100%" height={200}>
            <BarChart data={labels.map((l, i) => ({ axis: l, I: +diag[i].toFixed(1), color: [C.cy, C.am, C.gn][i] }))} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
              <XAxis dataKey="axis" {...ax()} />
              <YAxis {...ax()} label={{ value: "kg·m²", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }} />
              <Tooltip contentStyle={TT} />
              <Bar dataKey="I" radius={[4, 4, 0, 0]} barSize={32}>
                {diag.map((_, i) => <Cell key={i} fill={[C.cy, C.am, C.gn][i]} fillOpacity={0.7} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// CORNER WEIGHTS TAB
// ═══════════════════════════════════════════════════════════════════════════
function CornersTab({ corners, cg, mass }) {
  const total = corners.FL + corners.FR + corners.RL + corners.RR;
  const crossDiag = Math.abs(corners.crossFL_RR - corners.crossFR_RL) / total * 100;
  const cornerData = [
    { corner: "FL", weight: +corners.FL.toFixed(1), color: C.cy },
    { corner: "FR", weight: +corners.FR.toFixed(1), color: C.gn },
    { corner: "RL", weight: +corners.RL.toFixed(1), color: C.am },
    { corner: "RR", weight: +corners.RR.toFixed(1), color: C.red },
  ];

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="FL" value={`${corners.FL.toFixed(1)} N`} sub={`${(corners.FL / total * 100).toFixed(1)}%`} sentiment="neutral" delay={0} />
        <KPI label="FR" value={`${corners.FR.toFixed(1)} N`} sub={`${(corners.FR / total * 100).toFixed(1)}%`} sentiment="neutral" delay={1} />
        <KPI label="RL" value={`${corners.RL.toFixed(1)} N`} sub={`${(corners.RL / total * 100).toFixed(1)}%`} sentiment="neutral" delay={2} />
        <KPI label="RR" value={`${corners.RR.toFixed(1)} N`} sub={`${(corners.RR / total * 100).toFixed(1)}%`} sentiment="neutral" delay={3} />
        <KPI label="F/R Split" value={`${corners.frontPct.toFixed(1)}/${corners.rearPct.toFixed(1)}`} sub="front/rear %" sentiment={Math.abs(corners.frontPct - 50) < 5 ? "positive" : "amber"} delay={4} />
        <KPI label="Cross Δ" value={`${crossDiag.toFixed(1)}%`} sub={crossDiag < 2 ? "excellent" : crossDiag < 5 ? "acceptable" : "adjust ballast"} sentiment={crossDiag < 2 ? "positive" : crossDiag < 5 ? "amber" : "negative"} delay={5} />
      </div>

      <Sec title="Static Corner Weights [N]">
        <GC><ResponsiveContainer width="100%" height={200}>
          <BarChart data={cornerData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="corner" {...ax()} />
            <YAxis {...ax()} label={{ value: "N", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }} />
            <Tooltip contentStyle={TT} />
            <Bar dataKey="weight" radius={[4, 4, 0, 0]} barSize={40}>
              {cornerData.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// BALLAST OPTIMIZER
// ═══════════════════════════════════════════════════════════════════════════
function BallastTab({ components, cg }) {
  const [targetFR, setTargetFR] = useState(48); // target front %
  const wb = 1.55, tF = 1.22, tR = 1.18;
  const mass = components.reduce((a, c) => a + c.mass, 0);

  const result = useMemo(() => {
    const currentFrontPct = ((wb / 2 - cg.x) / wb) * 100;
    const deltaPct = targetFR - currentFrontPct;
    const ballastMass = Math.abs(deltaPct * mass / 100) * (wb / 0.8); // rough estimate
    const ballastX = deltaPct > 0 ? 0.7 : -0.6; // front or rear
    const ballastY = 0.05; // low as possible
    return {
      currentFrontPct, deltaPct,
      ballastMass: +Math.min(15, ballastMass).toFixed(1),
      ballastX, ballastY,
      position: deltaPct > 0 ? "FRONT (nose cone area)" : "REAR (behind driver)",
    };
  }, [targetFR, cg, mass]);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 14 }}>
        <Lbl>TARGET FRONT %</Lbl>
        <input type="range" min={40} max={55} step={0.5} value={targetFR}
          onChange={e => setTargetFR(+e.target.value)}
          style={{ width: 200, accentColor: C.cy }} />
        <span style={{ fontSize: 14, fontWeight: 700, color: C.cy, fontFamily: C.dt }}>{targetFR}%</span>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt }}>
          Current: {result.currentFrontPct.toFixed(1)}% · Δ: {result.deltaPct > 0 ? "+" : ""}{result.deltaPct.toFixed(1)}%
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Ballast Needed" value={`${result.ballastMass} kg`} sub={Math.abs(result.deltaPct) < 0.5 ? "already optimal" : "to reach target"} sentiment={Math.abs(result.deltaPct) < 0.5 ? "positive" : "amber"} delay={0} />
        <KPI label="Position" value={result.position.split(" ")[0]} sub={result.position} sentiment="neutral" delay={1} />
        <KPI label="Height" value="50 mm" sub="as low as possible" sentiment="positive" delay={2} />
        <KPI label="Mass Penalty" value={`+${result.ballastMass} kg`} sub={`${((mass + result.ballastMass) / mass * 100 - 100).toFixed(1)}% total`} sentiment={result.ballastMass < 5 ? "positive" : "amber"} delay={3} />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// WEIGHT SENSITIVITY
// ═══════════════════════════════════════════════════════════════════════════
function SensitivityTab({ components }) {
  const sensData = useMemo(() => {
    const baseline = computeCG(components);
    return components.filter(c => c.mass > 2).map(c => {
      const reduced = components.map(cc => cc.name === c.name ? { ...cc, mass: cc.mass * 0.9 } : cc);
      const newCG = computeCG(reduced);
      const massSaved = c.mass * 0.1;
      const cgShiftX = (newCG.x - baseline.x) * 1000;
      const cgShiftY = (newCG.y - baseline.y) * 1000;
      return { name: c.name, mass: c.mass, massSaved: +massSaved.toFixed(1), cgShiftX: +cgShiftX.toFixed(2), cgShiftY: +cgShiftY.toFixed(2), category: c.category };
    }).sort((a, b) => b.massSaved - a.massSaved);
  }, [components]);

  return (
    <div>
      <Sec title="10% Mass Reduction Sensitivity — CG Shift [mm]">
        <GC><ResponsiveContainer width="100%" height={350}>
          <BarChart data={sensData.slice(0, 12)} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
            <XAxis type="number" {...ax()} label={{ value: "CG shift X [mm]", position: "bottom", fill: C.dm, fontSize: 8 }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={95} />
            <Tooltip contentStyle={TT} />
            <ReferenceLine x={0} stroke={C.dm} />
            <Bar dataKey="cgShiftX" radius={[0, 4, 4, 0]} barSize={12}>
              {sensData.slice(0, 12).map((e, i) => <Cell key={i} fill={e.cgShiftX > 0 ? C.gn : C.red} fillOpacity={0.7} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>
    </div>
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
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Total Mass" value={`${cg.mass.toFixed(1)} kg`} sub="with driver" sentiment={cg.mass < 320 ? "positive" : "amber"} delay={0} />
        <KPI label="CG X" value={`${(cg.x * 1000).toFixed(0)} mm`} sub={cg.x > 0 ? "ahead of mid-WB" : "behind mid-WB"} sentiment="neutral" delay={1} />
        <KPI label="CG Y (height)" value={`${(cg.y * 1000).toFixed(0)} mm`} sub="from ground" sentiment={cg.y < 0.30 ? "positive" : "amber"} delay={2} />
        <KPI label="CG Z (lateral)" value={`${(cg.z * 1000).toFixed(1)} mm`} sub={Math.abs(cg.z) < 0.005 ? "centered" : "offset"} sentiment={Math.abs(cg.z) < 0.01 ? "positive" : "amber"} delay={3} />
        <KPI label="F/R Split" value={`${corners.frontPct.toFixed(1)}/${corners.rearPct.toFixed(1)}`} sub="front / rear %" sentiment={Math.abs(corners.frontPct - 48) < 3 ? "positive" : "amber"} delay={4} />
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.cy} />)}
      </div>

      {tab === "overview" && <OverviewTab components={components} cg={cg} inertia={inertia} />}
      {tab === "inertia" && <InertiaTab inertia={inertia} cg={cg} />}
      {tab === "corners" && <CornersTab corners={corners} cg={cg} mass={cg.mass} />}
      {tab === "ballast" && <BallastTab components={components} cg={cg} />}
      {tab === "sensitivity" && <SensitivityTab components={components} />}
    </div>
  );
}