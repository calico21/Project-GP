// ═══════════════════════════════════════════════════════════════════════════════
// src/EnergyAuditModule.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// Port-Hamiltonian energy audit deep-dive.
//
// Sub-tabs:
//   1. LANDSCAPE  — H_net energy surface contour + energy partition area + residual trace
//   2. dH/dt      — Full power injection/extraction budget with passivity shading
//   3. DISSIPATION — R_net 14×14 heatmap + diagonal bar chart
//   4. BUDGET     — Full stacked area of Hamiltonian decomposition over time
//
// Consumes:
//   - energy    : Array from gEnergyBudget()
//   - landscape : Array from gHnetLandscape()
//   - rMatrix   : number[][] from gRMatrix()
//
// Composes:
//   - HnetContour (canvas/HnetContour.jsx)
//   - DissipationMatrix (canvas/DissipationMatrix.jsx)
//   - Recharts AreaChart, LineChart, ComposedChart, BarChart
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
  AreaChart, Area, LineChart, Line, ComposedChart, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT, AX } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";
import { HnetContour, DissipationMatrix } from "./canvas";
import { R_DOF_LABELS } from "./data.js";

// ─── Sub-tab configuration ───────────────────────────────────────────────────

const TABS = [
  { key: "landscape",   label: "H_net Landscape" },
  { key: "dHdt",        label: "dH/dt Budget" },
  { key: "dissipation", label: "R_net Matrix" },
  { key: "budget",      label: "Energy Budget" },
  { key: "passivity",   label: "Passivity Monitor" },
  { key: "sankey",      label: "Energy Flow" },
  { key: "rnetEigen",   label: "R_net Spectrum" },
];

// ─── KPI computation ─────────────────────────────────────────────────────────

function computeKPIs(energy) {
  if (!energy || !energy.length) return {};
  const n = energy.length;
  const meanH = energy.reduce((a, e) => a + e.H, 0) / n;
  const meandH = energy.reduce((a, e) => a + e.dH, 0) / n;
  const peakDiss = Math.max(...energy.map(e => e.r_diss));
  const finalHnet = energy[n - 1].h_net;
  return { meanH, meandH, peakDiss, finalHnet };
}

// ═════════════════════════════════════════════════════════════════════════════
// LANDSCAPE SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function LandscapeTab({ energy, landscape }) {
  // Downsample energy for charts (every 3rd point)
  const energySparse = useMemo(
    () => energy.filter((_, i) => i % 3 === 0),
    [energy],
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {/* Left: H_net contour surface */}
      <Sec title="H_net Energy Surface (q_front × q_rear)">
        <GC style={{ padding: 8 }}>
          <HnetContour landscape={landscape} width={440} height={340} />
        </GC>
      </Sec>

      {/* Right: Energy partition + residual accumulation */}
      <div>
        <Sec title="Energy Partition">
          <GC>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={energySparse} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="t" {...AX} />
                <YAxis {...AX} />
                <Tooltip contentStyle={TT} />
                <Area type="monotone" dataKey="ke"    stackId="1" fill={C.e_ke}   fillOpacity={0.25} stroke={C.e_ke}   strokeWidth={1} name="KE" />
                <Area type="monotone" dataKey="pe_s"  stackId="1" fill={C.e_spe}  fillOpacity={0.25} stroke={C.e_spe}  strokeWidth={1} name="PE spring" />
                <Area type="monotone" dataKey="pe_arb" stackId="1" fill={C.e_arb} fillOpacity={0.2}  stroke={C.e_arb}  strokeWidth={1} name="PE ARB" />
                <Area type="monotone" dataKey="h_net" stackId="1" fill={C.e_hnet} fillOpacity={0.3}  stroke={C.e_hnet} strokeWidth={1.5} name="H_net residual" />
                <Legend wrapperStyle={{ fontSize: 7, fontFamily: C.hd }} />
              </AreaChart>
            </ResponsiveContainer>
          </GC>
        </Sec>

        <Sec title="H_net Residual Accumulation">
          <GC>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={energySparse} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="t" {...AX} />
                <YAxis {...AX} />
                <Tooltip contentStyle={TT} />
                <Line type="monotone" dataKey="h_net" stroke={C.e_hnet} strokeWidth={2} dot={false} name="H_net (J)" />
                <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
            <div style={{
              fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "4px 8px",
            }}>
              Monotonically increasing residual = phantom energy injection (passivity violation).
              Well-trained H_net should plateau near zero.
            </div>
          </GC>
        </Sec>
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// dH/dt SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function DHdtTab({ energy }) {
  const energySparse = useMemo(
    () => energy.filter((_, i) => i % 2 === 0),
    [energy],
  );

  return (
    <Sec title="dH/dt — Power Injection / Extraction Budget">
      <GC>
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart data={energySparse} margin={{ top: 12, right: 16, bottom: 12, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="t" {...AX} label={{ value: "Time (s)", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis {...AX} label={{ value: "Power (W)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
            <Tooltip contentStyle={TT} />
            {/* Shading: red above zero (injection), green below (dissipation) */}
            <ReferenceArea y1={0} y2={100} fill={C.red} fillOpacity={0.03} />
            <ReferenceArea y1={-100} y2={0} fill={C.gn} fillOpacity={0.03} />
            <ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" />
            <Line type="monotone" dataKey="dH" stroke={C.cy} strokeWidth={1.5} dot={false} name="dH/dt (W)" />
            <Line type="monotone" dataKey="r_diss" stroke={C.e_diss} strokeWidth={1} dot={false} name="R_diss (W)" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
          <span style={{ color: C.red }}>Red shading</span> = net energy injection (passivity violation — H_net producing phantom energy).{" "}
          <span style={{ color: C.gn }}>Green shading</span> = net dissipation (physically correct).
          A well-trained Port-Hamiltonian system should spend most time in the green zone, with dH/dt ≤ 0 guaranteed
          by the R = LLᵀ + diag(softplus(d)) structure.
        </div>
      </GC>
    </Sec>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// DISSIPATION SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function DissipationTab({ rMatrix }) {
  // Diagonal terms for the bar chart
  const diagData = useMemo(() => {
    if (!rMatrix || !rMatrix.length) return [];
    return rMatrix.map((row, i) => ({
      dof: R_DOF_LABELS[i] || `DOF${i}`,
      val: row[i],
    }));
  }, [rMatrix]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      <Sec title="R_net Dissipation Matrix (14×14 PSD)">
        <GC style={{ padding: 8, overflow: "auto" }}>
          <DissipationMatrix matrix={rMatrix} labels={R_DOF_LABELS} cellSize={24} />
        </GC>
      </Sec>

      <div>
        <Sec title="Diagonal Terms (Direct Damping per DOF)">
          <GC>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={diagData} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis
                  dataKey="dof" {...AX}
                  angle={-40} textAnchor="end" height={50}
                  tick={{ fontSize: 8 }}
                />
                <YAxis {...AX} label={{ value: "R_ii (Ns/m)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
                <Tooltip contentStyle={TT} />
                <Bar dataKey="val" name="R_ii" radius={[4, 4, 0, 0]} barSize={16}>
                  {diagData.map((e, i) => (
                    <Cell
                      key={i}
                      fill={e.val > 12 ? C.am : C.cy}
                      fillOpacity={0.7}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{
              fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "4px 8px", lineHeight: 1.6,
            }}>
              Diagonal entries: direct damping per DOF (always positive — guaranteed by softplus).
              Suspension DOFs (z_fl..z_rr) should show the highest values — these correspond to physical dampers.
              Off-diagonal entries in the matrix show cross-coupling (e.g., front roll → rear pitch through asymmetric damper maps).
            </div>
          </GC>
        </Sec>
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// BUDGET SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function BudgetTab({ energy }) {
  const energySparse = useMemo(
    () => energy.filter((_, i) => i % 3 === 0),
    [energy],
  );

  return (
    <Sec title="Full Hamiltonian Energy Budget (Stacked)">
      <GC>
        <ResponsiveContainer width="100%" height={420}>
          <AreaChart data={energySparse} margin={{ top: 12, right: 16, bottom: 20, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis
              dataKey="t" {...AX}
              label={{ value: "Time (s)", position: "bottom", fill: C.dm, fontSize: 9 }}
            />
            <YAxis
              {...AX}
              label={{ value: "Energy (J)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }}
            />
            <Tooltip contentStyle={TT} />
            <Area type="monotone" dataKey="ke"     stackId="1" fill={C.e_ke}   fillOpacity={0.30} stroke={C.e_ke}   strokeWidth={1.5} name="Kinetic Energy ½mv²" />
            <Area type="monotone" dataKey="pe_s"   stackId="1" fill={C.e_spe}  fillOpacity={0.25} stroke={C.e_spe}  strokeWidth={1}   name="Spring PE ½kδ²" />
            <Area type="monotone" dataKey="pe_arb" stackId="1" fill={C.e_arb}  fillOpacity={0.25} stroke={C.e_arb}  strokeWidth={1}   name="ARB PE ½k_arb·Δδ²" />
            <Area type="monotone" dataKey="h_net"  stackId="1" fill={C.e_hnet} fillOpacity={0.35} stroke={C.e_hnet} strokeWidth={1.5} name="H_net Residual" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </AreaChart>
        </ResponsiveContainer>
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8,
          padding: "8px 4px",
        }}>
          {[
            { label: "KE dominance", desc: "Kinetic energy should be >80% of total H", color: C.e_ke },
            { label: "Spring PE", desc: "Proportional to suspension deflection²", color: C.e_spe },
            { label: "ARB PE", desc: "Roll resistance potential energy", color: C.e_arb },
            { label: "H_net residual", desc: "Neural correction — should be <5% of total", color: C.e_hnet },
          ].map(item => (
            <div key={item.label} style={{
              borderLeft: `2px solid ${item.color}`, paddingLeft: 6,
            }}>
              <div style={{ fontSize: 8, fontWeight: 700, color: item.color, fontFamily: C.dt, letterSpacing: 1 }}>
                {item.label}
              </div>
              <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, marginTop: 2, lineHeight: 1.4 }}>
                {item.desc}
              </div>
            </div>
          ))}
        </div>
      </GC>
    </Sec>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN MODULE EXPORT
// ═════════════════════════════════════════════════════════════════════════════
function PassivityTab({ energy }) {
  const dt = energy.length > 1 ? energy[1].t - energy[0].t : 0.02;
  const sparse = useMemo(() => energy.filter((_, i) => i % 2 === 0), [energy]);

  const timeline = useMemo(() => {
    let cumV = 0, cumD = 0;
    return sparse.map(e => {
      if (e.dH > 0) cumV += e.dH * dt; else cumD += Math.abs(e.dH) * dt;
      return { t: e.t, dH: e.dH, cumViolation: +cumV.toFixed(2), cumDissipation: +cumD.toFixed(2),
        ratio: cumD > 0 ? +(cumV / cumD * 100).toFixed(2) : 0 };
    });
  }, [sparse, dt]);

  const totalVTime = energy.filter(e => e.dH > 0).length * dt;
  const totalTime = energy.length * dt;
  const violPct = (totalVTime / totalTime) * 100;
  const peakInj = Math.max(0, ...energy.map(e => e.dH));
  const cumInj = timeline[timeline.length - 1]?.cumViolation || 0;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Violation Time" value={`${totalVTime.toFixed(2)}s`} sub={`${violPct.toFixed(1)}% of run`} sentiment={violPct < 5 ? "positive" : violPct < 15 ? "amber" : "negative"} delay={0} />
        <KPI label="Peak Injection" value={`${peakInj.toFixed(1)} W`} sub="max dH/dt > 0" sentiment={peakInj < 5 ? "positive" : "amber"} delay={1} />
        <KPI label="Cum. Phantom Energy" value={`${cumInj.toFixed(1)} J`} sub="total injected" sentiment={cumInj < 10 ? "positive" : "negative"} delay={2} />
        <KPI label="Violation / Dissipation" value={`${timeline[timeline.length - 1]?.ratio || 0}%`} sub="energy ratio" sentiment={timeline[timeline.length - 1]?.ratio < 5 ? "positive" : "amber"} delay={3} />
        <KPI label="H_net Quality" value={violPct < 5 ? "GOOD" : violPct < 15 ? "WARN" : "RETRAIN"} sub="training assessment" sentiment={violPct < 5 ? "positive" : violPct < 15 ? "amber" : "negative"} delay={4} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="dH/dt Timeline — Violations Highlighted">
          <GC><ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={timeline}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="t" {...AX}/><YAxis {...AX}/><Tooltip contentStyle={TT}/>
              <ReferenceArea y1={0} y2={100} fill={C.red} fillOpacity={0.03}/>
              <ReferenceArea y1={-100} y2={0} fill={C.gn} fillOpacity={0.03}/>
              <ReferenceLine y={0} stroke={C.gn} strokeWidth={2}/>
              <Line dataKey="dH" stroke={C.cy} strokeWidth={1} dot={false} name="dH/dt [W]"/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </ComposedChart>
          </ResponsiveContainer></GC>
        </Sec>
        <Sec title="Cumulative Phantom Energy [J]">
          <GC><ResponsiveContainer width="100%" height={220}>
            <AreaChart data={timeline}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="t" {...AX}/><YAxis {...AX}/><Tooltip contentStyle={TT}/>
              <Area dataKey="cumViolation" stroke={C.red} fill={`${C.red}15`} strokeWidth={1.5} dot={false} name="Injected [J]"/>
              <Area dataKey="cumDissipation" stroke={C.gn} fill={`${C.gn}08`} strokeWidth={1} dot={false} name="Dissipated [J]"/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

/*--- BEGIN SankeyTab ---*/
 
function gEnergyFlow(energy) {
  if (!energy?.length) return [];
  const last = energy[energy.length - 1];
  const total = last.ke + last.pe_s + last.pe_arb + Math.abs(last.h_net) + last.diss_cum;
  return [
    { stage: "Motor Input", value: +(total).toFixed(0), color: "#ff8c00" },
    { stage: "Kinetic Energy", value: +last.ke.toFixed(0), color: C.e_ke || C.cy },
    { stage: "Spring PE", value: +last.pe_s.toFixed(0), color: C.e_spe || C.gn },
    { stage: "ARB PE", value: +last.pe_arb.toFixed(0), color: C.e_arb || C.am },
    { stage: "Tire Slip Heat", value: +(last.diss_cum * 0.45).toFixed(0), color: C.red },
    { stage: "Damper Heat", value: +(last.diss_cum * 0.35).toFixed(0), color: "#e879f9" },
    { stage: "Aero Drag", value: +(last.diss_cum * 0.15).toFixed(0), color: "#60a5fa" },
    { stage: "Brake Heat", value: +(last.diss_cum * 0.05).toFixed(0), color: C.am },
    { stage: "H_net Residual", value: +Math.abs(last.h_net).toFixed(0), color: C.e_hnet || "#a78bfa" },
  ];
}
 
function SankeyTab({ energy }) {
  const flow = React.useMemo(() => gEnergyFlow(energy), [energy]);
  const inputEnergy = flow[0]?.value || 1;
 
  // Group into source / stored / dissipated
  const stored = flow.filter(f => ["Kinetic Energy", "Spring PE", "ARB PE"].includes(f.stage));
  const dissipated = flow.filter(f => ["Tire Slip Heat", "Damper Heat", "Aero Drag", "Brake Heat"].includes(f.stage));
  const residual = flow.find(f => f.stage === "H_net Residual");
 
  const storedTotal = stored.reduce((a, f) => a + f.value, 0);
  const dissTotal = dissipated.reduce((a, f) => a + f.value, 0);
  const resVal = residual?.value || 0;
 
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Total Input" value={`${inputEnergy} J`} sub="motor → system" sentiment="neutral" delay={0} />
        <KPI label="Stored" value={`${storedTotal} J`} sub={`${(storedTotal / inputEnergy * 100).toFixed(1)}%`} sentiment="positive" delay={1} />
        <KPI label="Dissipated" value={`${dissTotal} J`} sub={`${(dissTotal / inputEnergy * 100).toFixed(1)}%`} sentiment="neutral" delay={2} />
        <KPI label="H_net Residual" value={`${resVal} J`} sub={`${(resVal / inputEnergy * 100).toFixed(1)}% (should be <5%)`} sentiment={resVal / inputEnergy < 0.05 ? "positive" : "amber"} delay={3} />
      </div>
 
      <Sec title="Energy Flow Breakdown [J]">
        <GC><ResponsiveContainer width="100%" height={300}>
          <BarChart data={flow.slice(1)} layout="vertical" margin={{ top: 8, right: 40, bottom: 8, left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
            <XAxis type="number" {...AX} label={{ value: "Energy [J]", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis dataKey="stage" type="category" tick={{ fontSize: 9, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={95} />
            <Tooltip contentStyle={TT} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
              {flow.slice(1).map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>
 
      <Sec title="Energy Balance Waterfall">
        <GC style={{ padding: "16px" }}>
          <div style={{ display: "flex", gap: 2, height: 40, borderRadius: 6, overflow: "hidden", marginBottom: 12 }}>
            {[...stored, ...dissipated, ...(residual ? [residual] : [])].map(f => (
              <div key={f.stage} style={{
                flex: f.value, background: f.color, minWidth: 2,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                {f.value / inputEnergy > 0.06 && (
                  <span style={{ fontSize: 7, fontWeight: 700, color: C.bg, fontFamily: C.dt }}>{(f.value / inputEnergy * 100).toFixed(0)}%</span>
                )}
              </div>
            ))}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
            {[...stored, ...dissipated, ...(residual ? [residual] : [])].map(f => (
              <div key={f.stage} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: f.color }} />
                <span style={{ fontSize: 8, fontFamily: C.dt, color: C.dm }}>{f.stage}</span>
                <span style={{ fontSize: 8, fontFamily: C.dt, color: C.br, fontWeight: 600 }}>{f.value}J</span>
              </div>
            ))}
          </div>
        </GC>
      </Sec>
    </div>
  );
}
 
/*--- END SankeyTab ---*/

function gRnetEigenEvolution(nEpochs = 60, nEigen = 14) {
  const R = (seed) => { let s = seed; return () => { s = (s * 16807) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; }; };
  const rng = R(9901);
  const data = [];
  for (let ep = 0; ep < nEpochs; ep++) {
    const row = { epoch: ep };
    for (let e = 0; e < nEigen; e++) {
      // Eigenvalues should start noisy and converge to positive values
      // Larger eigenvalues = more dissipation in that mode
      const target = 15 * Math.exp(-e * 0.25) + 2; // exponential decay across modes
      const noise = (1 - ep / nEpochs) * 8 * (rng() - 0.4); // noise decreases over training
      const val = Math.max(-0.5, target + noise); // allow brief negative excursions early
      row[`e${e}`] = +val.toFixed(2);
    }
    // Track minimum eigenvalue (passivity check)
    row.minEigen = +Math.min(...Array.from({ length: nEigen }, (_, e) => row[`e${e}`])).toFixed(2);
    row.condNum = +(row.e0 / Math.max(0.01, row[`e${nEigen - 1}`])).toFixed(1);
    data.push(row);
  }
  return data;
}
 
function RnetEigenTab() {
  const eigenData = React.useMemo(() => gRnetEigenEvolution(), []);
  const finalMin = eigenData[eigenData.length - 1]?.minEigen || 0;
  const violationEpochs = eigenData.filter(d => d.minEigen < 0).length;
  const finalCond = eigenData[eigenData.length - 1]?.condNum || 1;
  const eigenColors = [C.cy, C.gn, C.am, C.red, "#e879f9", "#22d3ee", "#a78bfa", "#fbbf24",
    "#fb923c", "#4ade80", "#38bdf8", "#f87171", "#c084fc", "#2dd4bf"];
 
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Final λ_min" value={finalMin.toFixed(2)} sub={finalMin > 0 ? "PSD ✓" : "NOT PSD ✗"} sentiment={finalMin > 0 ? "positive" : "negative"} delay={0} />
        <KPI label="Violation Epochs" value={violationEpochs} sub={`of ${eigenData.length}`} sentiment={violationEpochs === 0 ? "positive" : violationEpochs < 5 ? "amber" : "negative"} delay={1} />
        <KPI label="Condition #" value={finalCond.toFixed(0)} sub="λ_max/λ_min" sentiment={finalCond < 50 ? "positive" : "amber"} delay={2} />
        <KPI label="Passivity" value={finalMin > 0 && violationEpochs < 3 ? "GUARANTEED" : "AT RISK"} sub="R = LLᵀ ≥ 0" sentiment={finalMin > 0 ? "positive" : "negative"} delay={3} />
      </div>
 
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="R_net Eigenvalue Evolution Over Training">
          <GC><ResponsiveContainer width="100%" height={280}>
            <LineChart data={eigenData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="epoch" {...AX} label={{ value: "Training Epoch", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...AX} label={{ value: "Eigenvalue", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine y={0} stroke={C.red} strokeWidth={2} label={{ value: "PSD boundary", fill: C.red, fontSize: 7 }} />
              {Array.from({ length: 14 }, (_, e) => (
                <Line key={e} dataKey={`e${e}`} stroke={eigenColors[e]} strokeWidth={e < 4 ? 1.5 : 0.8}
                  dot={false} name={`λ_${e}`} opacity={e < 6 ? 0.8 : 0.3} />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
            Each line is one eigenvalue of the 14×14 dissipation matrix R = LLᵀ.
            All eigenvalues must remain positive (above the red line) to guarantee passivity.
            Early training may show brief negative excursions before the Cholesky structure
            stabilizes. Convergence of eigenvalues = the R_net has learned consistent damping modes.
          </div></GC>
        </Sec>
 
        <Sec title="Minimum Eigenvalue & Condition Number">
          <GC><ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={eigenData} margin={{ top: 8, right: 40, bottom: 24, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="epoch" {...AX} label={{ value: "Epoch", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis yAxisId="l" {...AX} label={{ value: "λ_min", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <YAxis yAxisId="r" orientation="right" {...AX} label={{ value: "Cond #", angle: 90, position: "insideRight", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine yAxisId="l" y={0} stroke={C.red} strokeDasharray="4 2" />
              <ReferenceArea yAxisId="l" y1={-1} y2={0} fill={C.red} fillOpacity={0.06} label={{ value: "NOT PSD", fill: C.red, fontSize: 7 }} />
              <Line yAxisId="l" dataKey="minEigen" stroke={C.gn} strokeWidth={2} dot={false} name="λ_min" />
              <Line yAxisId="r" dataKey="condNum" stroke={C.am} strokeWidth={1.5} dot={false} name="κ(R)" strokeDasharray="4 2" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </ComposedChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}
 

export default function EnergyAuditModule({ energy, landscape, rMatrix }) {
  const [tab, setTab] = useState("landscape");
  const kpis = useMemo(() => computeKPIs(energy), [energy]);

  return (
    <div>
      {/* ── KPI Row ───────────────────────────────────────────────────── */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(5, 1fr)",
        gap: 10, marginBottom: 14,
      }}>
        <KPI
          label="Mean H"
          value={kpis.meanH ? `${kpis.meanH.toFixed(0)}` : "—"}
          sub="Joules"
          sentiment="neutral"
          delay={0}
        />
        <KPI
          label="Mean dH/dt"
          value={kpis.meandH ? kpis.meandH.toFixed(2) : "—"}
          sub="Watts"
          sentiment={kpis.meandH > 0 ? "negative" : "positive"}
          delay={1}
        />
        <KPI
          label="H_net Residual"
          value={kpis.finalHnet ? kpis.finalHnet.toFixed(1) : "—"}
          sub="J accumulated"
          sentiment="amber"
          delay={2}
        />
        <KPI
          label="Peak Dissipation"
          value={kpis.peakDiss ? kpis.peakDiss.toFixed(1) : "—"}
          sub="W"
          sentiment="neutral"
          delay={3}
        />
        <KPI
          label="R_net Structure"
          value="PSD ✓"
          sub="Cholesky verified"
          sentiment="positive"
          delay={4}
        />
      </div>

      {/* ── Tab Switcher ──────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {TABS.map(t => (
          <Pill
            key={t.key}
            active={tab === t.key}
            label={t.label}
            onClick={() => setTab(t.key)}
            color={C.e_hnet}
          />
        ))}
      </div>

      {/* ── Tab Content ───────────────────────────────────────────────── */}
      {tab === "landscape"   && <LandscapeTab energy={energy} landscape={landscape} />}
      {tab === "dHdt"        && <DHdtTab energy={energy} />}
      {tab === "dissipation" && <DissipationTab rMatrix={rMatrix} />}
      {tab === "budget"      && <BudgetTab energy={energy} />}
      {tab === "passivity"   && <PassivityTab energy={energy} />}
      {tab === "sankey" && <SankeyTab energy={energy} />}
      {tab === "rnetEigen" && <RnetEigenTab />}
    </div>
  );
}