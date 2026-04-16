// ═══════════════════════════════════════════════════════════════════════════
// src/SuspensionExplorerModule.jsx — Project-GP Dashboard v6.0
// ═══════════════════════════════════════════════════════════════════════════
// Interactive 28-parameter suspension setup explorer with:
//   1. 3D Pareto front scatter (grip × stability × LTE)
//   2. Car schematic with per-corner parameter overlays
//   3. Side-by-side setup comparison with delta highlighting
//   4. Sensitivity bar chart (∂lap_time/∂param)
//   5. Live gradient connection (when physics server is running)
//
// INTEGRATION:
//   Add to App.jsx NAV_GROUPS under "VEHICLE DYNAMICS":
//     { key: "explorer", label: "Setup Explorer", icon: "⬢" }
//   Add routing case:
//     case "explorer": return <SuspensionExplorerModule mode={mode} />;
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo, useCallback } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip,
  BarChart, Bar, Cell, CartesianGrid,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { C, GL, GS, TT, AX } from "./theme.js";

// ── Parameter definitions (canonical 28D) ────────────────────────────────

const PARAMS = [
  { key: "k_f",         label: "Spring Rate Front",      unit: "N/mm",  group: "springs",  corner: "F",  idx: 0 },
  { key: "k_r",         label: "Spring Rate Rear",       unit: "N/mm",  group: "springs",  corner: "R",  idx: 1 },
  { key: "arb_f",       label: "ARB Front",              unit: "N/mm",  group: "springs",  corner: "F",  idx: 2 },
  { key: "arb_r",       label: "ARB Rear",               unit: "N/mm",  group: "springs",  corner: "R",  idx: 3 },
  { key: "c_ls_bump_f", label: "LS Bump Front",          unit: "Ns/m",  group: "dampers",  corner: "F",  idx: 4 },
  { key: "c_ls_bump_r", label: "LS Bump Rear",           unit: "Ns/m",  group: "dampers",  corner: "R",  idx: 5 },
  { key: "c_hs_bump_f", label: "HS Bump Front",          unit: "Ns/m",  group: "dampers",  corner: "F",  idx: 6 },
  { key: "c_hs_bump_r", label: "HS Bump Rear",           unit: "Ns/m",  group: "dampers",  corner: "R",  idx: 7 },
  { key: "c_ls_reb_f",  label: "LS Rebound Front",       unit: "Ns/m",  group: "dampers",  corner: "F",  idx: 8 },
  { key: "c_ls_reb_r",  label: "LS Rebound Rear",        unit: "Ns/m",  group: "dampers",  corner: "R",  idx: 9 },
  { key: "c_hs_reb_f",  label: "HS Rebound Front",       unit: "Ns/m",  group: "dampers",  corner: "F",  idx: 10 },
  { key: "c_hs_reb_r",  label: "HS Rebound Rear",        unit: "Ns/m",  group: "dampers",  corner: "R",  idx: 11 },
  { key: "camber_f",    label: "Camber Front",            unit: "deg",   group: "geometry", corner: "F",  idx: 12 },
  { key: "camber_r",    label: "Camber Rear",             unit: "deg",   group: "geometry", corner: "R",  idx: 13 },
  { key: "toe_f",       label: "Toe Front",               unit: "deg",   group: "geometry", corner: "F",  idx: 14 },
  { key: "toe_r",       label: "Toe Rear",                unit: "deg",   group: "geometry", corner: "R",  idx: 15 },
  { key: "castor",      label: "Castor Trail",            unit: "mm",    group: "geometry", corner: "F",  idx: 16 },
  { key: "ackermann",   label: "Ackermann %",             unit: "%",     group: "geometry", corner: "F",  idx: 17 },
  { key: "h_cg",        label: "CG Height",               unit: "mm",    group: "mass",     corner: "C",  idx: 18 },
  { key: "brake_bias",  label: "Brake Bias Front",        unit: "%",     group: "brakes",   corner: "C",  idx: 19 },
  { key: "diff_preload", label: "Diff Preload",           unit: "Nm",    group: "drivetrain", corner: "R", idx: 20 },
  { key: "diff_coast",  label: "Diff Coast Lock",         unit: "%",     group: "drivetrain", corner: "R", idx: 21 },
  { key: "diff_power",  label: "Diff Power Lock",         unit: "%",     group: "drivetrain", corner: "R", idx: 22 },
  { key: "k_heave_f",   label: "Heave Spring Front",      unit: "N/mm",  group: "springs",  corner: "F",  idx: 23 },
  { key: "k_heave_r",   label: "Heave Spring Rear",       unit: "N/mm",  group: "springs",  corner: "R",  idx: 24 },
  { key: "bumpstop_f",  label: "Bumpstop Gap Front",      unit: "mm",    group: "springs",  corner: "F",  idx: 25 },
  { key: "bumpstop_r",  label: "Bumpstop Gap Rear",       unit: "mm",    group: "springs",  corner: "R",  idx: 26 },
  { key: "anti_squat",  label: "Anti-Squat %",            unit: "%",     group: "geometry", corner: "R",  idx: 27 },
];

const GROUPS = [
  { key: "springs",     label: "Springs & ARBs",  color: C.am },
  { key: "dampers",     label: "4-Way Dampers",   color: C.cy },
  { key: "geometry",    label: "Geometry",         color: C.gn },
  { key: "mass",        label: "Mass & CG",       color: "#a78bfa" },
  { key: "brakes",      label: "Brakes",          color: C.red },
  { key: "drivetrain",  label: "Drivetrain",       color: "#ff6090" },
];

// ── Generate demo Pareto front (replace with real data in LIVE mode) ─────

function generateDemoPareto(n = 50) {
  const rng = (s) => {
    let x = Math.sin(s * 12.9898 + 78.233) * 43758.5453;
    return x - Math.floor(x);
  };
  return Array.from({ length: n }, (_, i) => {
    const grip = 1.2 + rng(i) * 0.4;
    const stability = 2.0 + rng(i + 100) * 3.5;
    const lte = 85 + rng(i + 200) * 12;
    const gen = Math.floor(rng(i + 300) * 400);
    return { grip, stability, lte, gen, id: i };
  });
}

function generateDemoSensitivity() {
  return PARAMS.map(p => ({
    key: p.key,
    label: p.label,
    sensitivity: (Math.random() - 0.5) * 0.1,  // ∂lap_time/∂param [s/unit]
    unit: p.unit,
  })).sort((a, b) => Math.abs(b.sensitivity) - Math.abs(a.sensitivity));
}

// ── Shared sub-components ────────────────────────────────────────────────

const Sec = ({ title, children }) => (
  <div style={{ ...GL, borderColor: C.b2, marginBottom: 10 }}>
    <div style={{
      padding: "8px 12px", borderBottom: `1px solid ${C.b1}`,
      fontSize: 9, fontWeight: 700, color: C.dm, letterSpacing: 1.5,
      fontFamily: "'JetBrains Mono', monospace",
    }}>{title}</div>
    <div style={{ padding: "12px" }}>{children}</div>
  </div>
);

const GC = ({ children, style = {} }) => (
  <div style={{
    background: `${C.bg}80`, borderRadius: 6,
    border: `1px solid ${C.b1}`, ...style,
  }}>{children}</div>
);

const ax = () => ({ tick: { fontSize: 8, fill: C.dm }, stroke: C.b1 });

// ── Main Module ─────────────────────────────────────────────────────────

export default function SuspensionExplorerModule({ mode }) {
  const [tab, setTab] = useState("pareto");
  const [selectedA, setSelectedA] = useState(null);
  const [selectedB, setSelectedB] = useState(null);
  const [activeGroup, setActiveGroup] = useState("springs");

  const pareto = useMemo(generateDemoPareto, []);
  const sensitivity = useMemo(generateDemoSensitivity, []);

  const TABS = [
    { key: "pareto",  label: "Pareto Front" },
    { key: "schematic", label: "Car Schematic" },
    { key: "compare", label: "Compare Setups" },
    { key: "sensitivity", label: "∂t/∂s Sensitivity" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, height: "100%" }}>
      {/* Tab bar */}
      <div style={{
        display: "flex", gap: 0, borderBottom: `1px solid ${C.b1}`,
        background: `${C.bg}80`, borderRadius: "6px 6px 0 0",
      }}>
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            padding: "10px 20px", border: "none", cursor: "pointer",
            background: tab === t.key ? `${C.cy}15` : "transparent",
            borderBottom: tab === t.key ? `2px solid ${C.cy}` : "2px solid transparent",
            color: tab === t.key ? C.cy : C.dm,
            fontSize: 10, fontWeight: 700, letterSpacing: 1.2,
            fontFamily: "'JetBrains Mono', monospace",
            transition: "all 0.15s ease",
          }}>{t.label}</button>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{
          padding: "10px 16px", fontSize: 8, color: C.dm,
          fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1,
          display: "flex", alignItems: "center", gap: 6,
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: "50%",
            background: mode === "LIVE" ? C.gn : C.am,
            boxShadow: mode === "LIVE" ? `0 0 8px ${C.gn}` : "none",
          }} />
          {mode === "LIVE" ? "LIVE GRADIENTS" : "DEMO DATA"}
        </div>
      </div>

      {/* Tab content */}
      {tab === "pareto" && <ParetoTab pareto={pareto} onSelectA={setSelectedA} onSelectB={setSelectedB} />}
      {tab === "schematic" && <SchematicTab activeGroup={activeGroup} setActiveGroup={setActiveGroup} />}
      {tab === "compare" && <CompareTab a={selectedA} b={selectedB} pareto={pareto} />}
      {tab === "sensitivity" && <SensitivityTab data={sensitivity} />}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: Pareto Front — 3-objective scatter
// ═══════════════════════════════════════════════════════════════════════════

function ParetoTab({ pareto, onSelectA, onSelectB }) {
  const [hovered, setHovered] = useState(null);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, flex: 1 }}>
      <Sec title="GRIP vs STABILITY (PARETO FRONT)">
        <GC>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="grip" name="Grip" unit=" G" {...ax()} label={{ value: "Grip [G]", fill: C.dm, fontSize: 8, position: "bottom" }} />
              <YAxis dataKey="stability" name="Stability" unit=" rad/s" {...ax()} label={{ value: "Overshoot [rad/s]", fill: C.dm, fontSize: 8, angle: -90, position: "left" }} />
              <ZAxis dataKey="lte" range={[40, 200]} name="LTE" unit="%" />
              <Tooltip contentStyle={TT} formatter={(v, n) => [typeof v === "number" ? v.toFixed(3) : v, n]} />
              <ReferenceLine y={5.0} stroke={C.red} strokeDasharray="4 4" label={{ value: "5.0 cap", fill: C.red, fontSize: 7 }} />
              <Scatter data={pareto} onMouseEnter={(_, i) => setHovered(i)}
                       onMouseLeave={() => setHovered(null)}
                       onClick={(_, i) => {
                         // Alternate between A and B selection
                         onSelectA(prev => prev === i ? null : i);
                       }}>
                {pareto.map((d, i) => (
                  <Cell key={i} fill={d.stability > 5 ? C.red : d.grip > 1.4 ? C.gn : C.cy}
                        fillOpacity={hovered === i ? 1.0 : 0.6}
                        stroke={hovered === i ? "#fff" : "none"}
                        strokeWidth={2} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </GC>
      </Sec>

      <Sec title="GRIP vs ENDURANCE EFFICIENCY">
        <GC>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="grip" name="Grip" unit=" G" {...ax()} />
              <YAxis dataKey="lte" name="LTE" unit="%" {...ax()} label={{ value: "LTE [%]", fill: C.dm, fontSize: 8, angle: -90, position: "left" }} />
              <Tooltip contentStyle={TT} />
              <Scatter data={pareto}>
                {pareto.map((d, i) => (
                  <Cell key={i} fill={d.lte > 92 ? C.gn : d.lte > 88 ? C.am : C.red}
                        fillOpacity={0.6} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </GC>
      </Sec>

      <div style={{ gridColumn: "1 / -1" }}>
        <Sec title="PARETO FRONT SUMMARY">
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10,
            fontSize: 9, fontFamily: "'JetBrains Mono', monospace",
          }}>
            {[
              { label: "Solutions", value: pareto.length, color: C.cy },
              { label: "Max Grip", value: Math.max(...pareto.map(p => p.grip)).toFixed(3) + " G", color: C.gn },
              { label: "Min Overshoot", value: Math.min(...pareto.map(p => p.stability)).toFixed(2) + " rad/s", color: C.am },
              { label: "Max LTE", value: Math.max(...pareto.map(p => p.lte)).toFixed(1) + "%", color: "#a78bfa" },
              { label: "Feasible", value: pareto.filter(p => p.stability <= 5).length + "/" + pareto.length, color: pareto.filter(p => p.stability <= 5).length === pareto.length ? C.gn : C.red },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ ...GL, padding: "10px 12px", textAlign: "center" }}>
                <div style={{ fontSize: 7, color: C.dm, letterSpacing: 1.5, fontWeight: 700 }}>{label}</div>
                <div style={{ fontSize: 16, fontWeight: 800, color, marginTop: 4 }}>{value}</div>
              </div>
            ))}
          </div>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: Sensitivity — ∂lap_time/∂param bar chart
// ═══════════════════════════════════════════════════════════════════════════

function SensitivityTab({ data }) {
  const top15 = data.slice(0, 15);

  return (
    <div style={{ flex: 1 }}>
      <Sec title="SETUP SENSITIVITY: ∂(LAP TIME) / ∂(PARAMETER)">
        <div style={{ fontSize: 9, color: C.md, marginBottom: 10, fontFamily: "'JetBrains Mono', monospace" }}>
          Negative = faster lap time. Largest magnitude = highest leverage parameter.
        </div>
        <GC>
          <ResponsiveContainer width="100%" height={450}>
            <BarChart data={top15} layout="vertical" margin={{ top: 8, right: 30, bottom: 8, left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis type="number" {...ax()} />
              <YAxis type="category" dataKey="label" {...ax()} width={110} tick={{ fontSize: 7.5, fill: C.dm }} />
              <Tooltip contentStyle={TT} formatter={(v) => [`${v > 0 ? "+" : ""}${v.toFixed(4)} s/unit`, "∂t/∂s"]} />
              <ReferenceLine x={0} stroke={C.dm} />
              <Bar dataKey="sensitivity" barSize={14} radius={[0, 3, 3, 0]}>
                {top15.map((d, i) => (
                  <Cell key={i} fill={d.sensitivity < 0 ? C.gn : C.red} fillOpacity={0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </GC>
        <div style={{
          marginTop: 8, fontSize: 8, color: C.dm,
          fontFamily: "'JetBrains Mono', monospace", textAlign: "center",
        }}>
          Computed via jax.grad(simulate_full_lap)(setup) — single backward pass through entire lap
        </div>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: Car Schematic — parameter overlays by group
// ═══════════════════════════════════════════════════════════════════════════

function SchematicTab({ activeGroup, setActiveGroup }) {
  const groupParams = PARAMS.filter(p => p.group === activeGroup);
  const groupColor = GROUPS.find(g => g.key === activeGroup)?.color || C.cy;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: 10, flex: 1 }}>
      {/* Group selector */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {GROUPS.map(g => (
          <button key={g.key} onClick={() => setActiveGroup(g.key)} style={{
            padding: "10px 12px", border: "none", cursor: "pointer",
            borderLeft: activeGroup === g.key ? `3px solid ${g.color}` : "3px solid transparent",
            background: activeGroup === g.key ? `${g.color}10` : "transparent",
            color: activeGroup === g.key ? g.color : C.dm,
            fontSize: 10, fontWeight: 600, textAlign: "left",
            fontFamily: "'JetBrains Mono', monospace",
            borderRadius: 0, transition: "all 0.15s ease",
          }}>
            {g.label}
            <div style={{ fontSize: 7, color: C.dm, marginTop: 2 }}>
              {PARAMS.filter(p => p.group === g.key).length} params
            </div>
          </button>
        ))}
      </div>

      {/* Schematic + parameters */}
      <Sec title={`${activeGroup.toUpperCase()} PARAMETERS`}>
        <div style={{ position: "relative", minHeight: 400 }}>
          {/* Simplified top-view car outline */}
          <svg viewBox="0 0 300 500" style={{ width: "100%", maxWidth: 300, margin: "0 auto", display: "block" }}>
            {/* Car body */}
            <rect x="80" y="60" width="140" height="380" rx="30"
                  fill="none" stroke={C.b2} strokeWidth="2" />
            {/* Wheels */}
            {[[60, 100], [200, 100], [60, 380], [200, 380]].map(([cx, cy], i) => (
              <g key={i}>
                <rect x={cx - 15} y={cy - 30} width="30" height="60" rx="6"
                      fill={`${groupColor}20`} stroke={groupColor} strokeWidth="1.5" />
                <text x={cx} y={cy + 3} textAnchor="middle" fontSize="8"
                      fill={groupColor} fontWeight="700">
                  {["FL", "FR", "RL", "RR"][i]}
                </text>
              </g>
            ))}
            {/* CG marker */}
            <circle cx="150" cy="230" r="5" fill={C.am} opacity="0.6" />
            <text x="150" y="252" textAnchor="middle" fontSize="7" fill={C.dm}>CG</text>
            {/* Direction arrow */}
            <polygon points="150,40 142,55 158,55" fill={C.gn} opacity="0.5" />
            <text x="150" y="35" textAnchor="middle" fontSize="7" fill={C.gn}>FWD</text>
          </svg>

          {/* Parameter list overlay */}
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6,
            marginTop: 16,
          }}>
            {groupParams.map(p => (
              <div key={p.key} style={{
                ...GL, padding: "8px 10px",
                borderLeft: `3px solid ${groupColor}`,
              }}>
                <div style={{ fontSize: 7, color: C.dm, letterSpacing: 1, fontWeight: 700,
                              fontFamily: "'JetBrains Mono', monospace" }}>
                  {p.corner} · {p.label}
                </div>
                <div style={{ fontSize: 14, fontWeight: 800, color: groupColor, marginTop: 2,
                              fontFamily: "'JetBrains Mono', monospace" }}>
                  —
                  <span style={{ fontSize: 8, color: C.dm, marginLeft: 4 }}>{p.unit}</span>
                </div>
                <div style={{ fontSize: 7, color: C.dm, marginTop: 2 }}>
                  idx={p.idx}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: Compare — side-by-side setup diff
// ═══════════════════════════════════════════════════════════════════════════

function CompareTab({ a, b, pareto }) {
  if (a == null && b == null) {
    return (
      <Sec title="SETUP COMPARISON">
        <div style={{
          textAlign: "center", padding: "60px 20px",
          color: C.dm, fontSize: 11,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          Click two points on the Pareto Front tab to compare setups.
          <br />
          <span style={{ fontSize: 9, color: C.b2 }}>
            First click = Setup A (cyan) · Second click = Setup B (amber)
          </span>
        </div>
      </Sec>
    );
  }

  const setupA = a != null ? pareto[a] : null;
  const setupB = b != null ? pareto[b] : null;

  return (
    <Sec title="SETUP A vs SETUP B">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {[
          { setup: setupA, label: "SETUP A", color: C.cy, idx: a },
          { setup: setupB, label: "SETUP B", color: C.am, idx: b },
        ].map(({ setup, label, color, idx }) => (
          <div key={label} style={{ ...GL, padding: 12, borderColor: color }}>
            <div style={{
              fontSize: 9, fontWeight: 700, color, letterSpacing: 1.5,
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 8,
            }}>
              {label} {idx != null ? `(#${idx})` : "(not selected)"}
            </div>
            {setup ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4 }}>
                {[
                  { k: "Grip", v: setup.grip.toFixed(3) + " G", c: C.gn },
                  { k: "Stability", v: setup.stability.toFixed(2) + " rad/s", c: C.am },
                  { k: "LTE", v: setup.lte.toFixed(1) + "%", c: "#a78bfa" },
                ].map(({ k, v, c }) => (
                  <div key={k} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 7, color: C.dm, fontWeight: 700, letterSpacing: 1 }}>{k}</div>
                    <div style={{ fontSize: 14, fontWeight: 800, color: c }}>{v}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: C.dm, fontSize: 9, textAlign: "center", padding: 20 }}>
                Not selected
              </div>
            )}
          </div>
        ))}
      </div>
    </Sec>
  );
}
