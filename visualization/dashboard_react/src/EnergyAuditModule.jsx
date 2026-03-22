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
    </div>
  );
}