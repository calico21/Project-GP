// ═══════════════════════════════════════════════════════════════════════════════
// src/App.jsx — Project-GP Dashboard v6.0
// ═══════════════════════════════════════════════════════════════════════════════
// CHANGES FROM v5.x → v6.0:
// ─────────────────────────────────────────────────────────────────────────────
// 1. NAV: 8 groups, 18 modules — flat nav items restructured for density.
//    Added: Powertrain Control, TV Simulator, Setup Explorer.
//    Reorganized CONTROLS & AI to separate POWERTRAIN group.
//
// 2. SUSPENSION: Routes to fully rebuilt SuspensionModule (Optimum-K style).
//    Passes `mode` prop for LIVE/ANALYZE telemetry switching.
//
// 3. HEADER: Bloomberg-style status bar with clock, mode, subsystem health.
//
// 4. SYSTEM HEALTH: Expanded to cover all subsystems incl. powertrain stack.
//
// 5. SIDEBAR: Collapsible groups, active state highlighting, module count.
//
// 6. All v5.x data hooks, props, and module routing are preserved exactly.
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useMemo } from "react";
import { C, GL, FONTS_URL } from "./theme.js";
import { FadeSlide } from "./components.jsx";
import { SelectionProvider } from "./context/SelectionContext.jsx";

// ── Data generators ──────────────────────────────────────────────────────────
import { gP, gCV, gTK, gTT, gSN, gSU } from "./data.js";
import {
  gP4, gCV4, gSN4,
  gThermal5, gEnergyBudgetPH, gALConstraints, gWaveletCoeffsMPC,
  gTubePoints, gHnetLandscape, gRMatrix,
  gGPEnvelope, gHysteresis, gLoadSensitivity, gEpisodes,
} from "./data.js";

// ── Module imports ───────────────────────────────────────────────────────────
import OverviewModule from "./OverviewModule.jsx";
import SetupModule from "./SetupModule.jsx";
import TelemetryModule from "./TelemetryModule.jsx";
import SuspensionModule from "./SuspensionModule.jsx";
import EnergyAuditModule from "./EnergyAuditModule.jsx";
import TirePhysicsModule from "./TirePhysicsModule.jsx";
import DriverCoachingModule from "./DriverCoachingModule.jsx";
import EnduranceStrategyModule from "./EnduranceStrategyModule.jsx";
import WeightBalanceModule from "./WeightBalanceModule.jsx";
import ComplianceModule from "./ComplianceModule.jsx";
import DifferentiableInsightsModule from "./DifferentiableInsightsModule.jsx";
import ResearchModule from "./ResearchModule.jsx";
import AerodynamicsModule from "./AerodynamicsModule.jsx";
import ElectronicsModule from "./ElectronicsModule.jsx";
import PowertrainControlModule from "./PowertrainControlModule.jsx";
import TorqueVectoringSimModule from "./TorqueVectoringSimModule.jsx";
import SuspensionExplorerModule from "./SuspensionExplorerModule.jsx";

// ═════════════════════════════════════════════════════════════════════════════
// GROUPED NAV CONFIGURATION — 8 groups, 18 modules
// ═════════════════════════════════════════════════════════════════════════════

const NAV_GROUPS = [
  {
    label: "OVERVIEW", accent: C.cy, items: [
      { key: "overview", label: "Overview", icon: "⬡" },
    ],
  },
  {
    label: "TELEMETRY", accent: C.gn, items: [
      { key: "telemetry", label: "Telemetry", icon: "◇" },
    ],
  },
  {
    label: "VEHICLE DYNAMICS", accent: C.am, items: [
      { key: "setup", label: "Setup Opt", icon: "◆" },
      { key: "suspension", label: "Suspension", icon: "△" },
      { key: "explorer", label: "Setup Explorer", icon: "⬢" },
      { key: "tire", label: "Tire Physics", icon: "⊗" },
      { key: "weight", label: "Weight & CG", icon: "⊿" },
    ],
  },
  {
    label: "AERODYNAMICS", accent: "#ff6090", items: [
      { key: "aero", label: "Aerodynamics", icon: "▽" },
    ],
  },
  {
    label: "ELECTRONICS", accent: "#7c3aed", items: [
      { key: "electronics", label: "Electronics", icon: "⌁" },
    ],
  },
  {
    label: "POWERTRAIN", accent: "#f97316", items: [
      { key: "powertrain", label: "Powertrain Ctrl", icon: "⏣" },
      { key: "tvsim", label: "TV Simulator", icon: "◎" },
    ],
  },
  {
    label: "CONTROLS & AI", accent: C.cy, items: [
      { key: "energy", label: "Energy Audit", icon: "⊕" },
      { key: "diff", label: "∇ Insights", icon: "∂" },
      { key: "research", label: "Research", icon: "⚙" },
    ],
  },
  {
    label: "PERFORMANCE", accent: C.red, items: [
      { key: "coaching", label: "Coaching", icon: "◈" },
      { key: "endurance", label: "Endurance", icon: "◷" },
      { key: "compliance", label: "Compliance", icon: "✓" },
    ],
  },
];

const ALL_NAV = NAV_GROUPS.flatMap(g => g.items);

// ═════════════════════════════════════════════════════════════════════════════
// MODE TOGGLE
// ═════════════════════════════════════════════════════════════════════════════

const ModeToggle = ({ mode, setMode }) => (
  <div style={{
    display: "flex", borderRadius: 20,
    border: `1px solid ${C.b2}`, overflow: "hidden",
  }}>
    {["LIVE", "ANALYZE"].map(m => (
      <button key={m} onClick={() => setMode(m)} style={{
        flex: 1, padding: "5px 16px", border: "none", cursor: "pointer",
        fontSize: 10, fontWeight: 700, letterSpacing: 1.5, fontFamily: C.dt,
        background: mode === m
          ? (m === "LIVE" ? `${C.gn}20` : `${C.cy}20`)
          : "transparent",
        color: mode === m
          ? (m === "LIVE" ? C.gn : C.cy)
          : C.dm,
        transition: "all 0.2s",
      }}>
        {m === "LIVE" && <span style={{ display: "inline-block", width: 5, height: 5, borderRadius: "50%", background: mode === "LIVE" ? C.gn : C.dm, marginRight: 6, boxShadow: mode === "LIVE" ? `0 0 6px ${C.gn}` : "none" }} />}
        {m}
      </button>
    ))}
  </div>
);

// ═════════════════════════════════════════════════════════════════════════════
// SYSTEM HEALTH — Bloomberg-style subsystem status
// ═════════════════════════════════════════════════════════════════════════════

const SystemHealth = () => {
  const systems = [
    { name: "H_net 46-DOF", status: "PASS", color: C.gn },
    { name: "Tire PINN", status: "PASS", color: C.gn },
    { name: "GP σ-bound", status: "PASS", color: C.gn },
    { name: "Diff-WMPC", status: "PASS", color: C.gn },
    { name: "MORL-TRPO", status: "PASS", color: C.gn },
    { name: "SOCP TV", status: "WARN", color: C.am },
    { name: "DESC TC", status: "PASS", color: C.gn },
    { name: "CBF Safety", status: "PASS", color: C.gn },
    { name: "Launch Ctrl", status: "PASS", color: C.gn },
    { name: "WS Bridge", status: "IDLE", color: C.dm },
  ];

  return (
    <div style={{ padding: "8px 14px", borderTop: `1px solid ${C.b1}` }}>
      <div style={{
        fontSize: 8, fontWeight: 700, letterSpacing: 1.8,
        color: C.dm, fontFamily: C.dt, marginBottom: 6,
      }}>
        SUBSYSTEM STATUS
      </div>
      {systems.map(s => (
        <div key={s.name} style={{
          display: "flex", justifyContent: "space-between",
          alignItems: "center", padding: "2px 0",
        }}>
          <span style={{ fontSize: 8, color: C.md, fontFamily: C.dt }}>{s.name}</span>
          <span style={{
            fontSize: 7, fontWeight: 700, color: s.color,
            fontFamily: C.dt, letterSpacing: 0.5,
          }}>
            {s.status}
          </span>
        </div>
      ))}
    </div>
  );
};

// ═════════════════════════════════════════════════════════════════════════════
// MAIN APPLICATION COMPONENT
// ═════════════════════════════════════════════════════════════════════════════

export default function App() {
  const [active, setActive] = useState("overview");
  const [mode, setMode] = useState("ANALYZE");
  const [time, setTime] = useState(new Date());
  const [collapsed, setCollapsed] = useState({});

  useEffect(() => {
    const iv = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(iv);
  }, []);

  // ── Data generation (memoized) ──────────────────────────────────────────
  const pareto = useMemo(() => gP4(), []);
  const conv = useMemo(() => gCV4(), []);
  const sens = useMemo(() => gSN4(), []);
  const track = useMemo(() => gTK(), []);
  const tireTemps = useMemo(() => gTT(), []);
  const susp = useMemo(() => gSU(), []);
  const episodes = useMemo(() => gEpisodes(), []);
  const thermal5 = useMemo(() => gThermal5(), []);
  const energy = useMemo(() => gEnergyBudgetPH(), []);
  const alData = useMemo(() => gALConstraints(), []);
  const wavelets = useMemo(() => gWaveletCoeffsMPC(), []);
  const tubes = useMemo(() => gTubePoints(), []);
  const landscape = useMemo(() => gHnetLandscape(), []);
  const rMatrix = useMemo(() => gRMatrix(), []);
  const gpEnv = useMemo(() => gGPEnvelope(), []);
  const hysteresis = useMemo(() => gHysteresis(), []);
  const loadSens = useMemo(() => gLoadSensitivity(), []);

  const activeGroup = NAV_GROUPS.find(g => g.items.some(i => i.key === active))?.label || "";

  const toggleGroup = (label) => {
    setCollapsed(prev => ({ ...prev, [label]: !prev[label] }));
  };

  // ── Content renderer ────────────────────────────────────────────────────
  const renderContent = () => {
    switch (active) {
      case "telemetry":
        return (
          <TelemetryModule
            mode={mode}
            track={track}
            tireTemps={tireTemps}
            thermal5={thermal5}
            tubes={tubes}
            wavelets={wavelets}
            alData={alData}
            energy={energy}
          />
        );
      case "setup":
        return (
          <SetupModule
            pareto={pareto}
            conv={conv}
            sens={sens}
            track={track}
            episodes={episodes}
          />
        );
      case "energy":
        return (
          <EnergyAuditModule
            energy={energy}
            landscape={landscape}
            rMatrix={rMatrix}
          />
        );
      case "tire":
        return (
          <TirePhysicsModule
            thermal5={thermal5}
            gpEnv={gpEnv}
            hysteresis={hysteresis}
            loadSens={loadSens}
          />
        );
      case "coaching":    return <DriverCoachingModule />;
      case "endurance":   return <EnduranceStrategyModule />;
      case "suspension":  return <SuspensionModule data={susp} mode={mode} />;
      case "explorer":    return <SuspensionExplorerModule mode={mode} />;
      case "weight":      return <WeightBalanceModule />;
      case "compliance":  return <ComplianceModule />;
      case "diff":        return <DifferentiableInsightsModule mode={mode} />;
      case "research":    return <ResearchModule />;
      case "powertrain":  return <PowertrainControlModule />;
      case "tvsim":       return <TorqueVectoringSimModule />;
      case "aero":        return <AerodynamicsModule />;
      case "electronics": return <ElectronicsModule />;
      default:
        return <OverviewModule pareto={pareto} conv={conv} />;
    }
  };

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <SelectionProvider>
      <div style={{
        background: C.bg, color: C.br, fontFamily: C.bd,
        minHeight: "100vh", display: "flex",
      }}>
        <link href={FONTS_URL} rel="stylesheet" />
        <style>{`* { margin: 0; padding: 0; box-sizing: border-box; }
          ::-webkit-scrollbar { width: 4px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: ${C.b2}; border-radius: 2px; }
          ::selection { background: ${C.cy}30; }
        `}</style>

        {/* ═══ SIDEBAR ═══════════════════════════════════════════════ */}
        <div style={{
          width: 210, minHeight: "100vh",
          borderRight: `1px solid ${C.b1}`,
          display: "flex", flexDirection: "column",
          background: `linear-gradient(180deg, ${C.panel || "#0c0e18"} 0%, ${C.bg} 100%)`,
          flexShrink: 0,
          overflowY: "auto",
        }}>
          {/* Logo */}
          <div style={{
            padding: "14px 16px 10px",
            borderBottom: `1px solid ${C.b1}`,
          }}>
            <div style={{
              fontSize: 13, fontWeight: 800, letterSpacing: 2,
              fontFamily: C.hd, color: C.w,
            }}>
              PROJECT-GP
            </div>
            <div style={{
              fontSize: 8, color: C.dm, fontFamily: C.dt,
              letterSpacing: 1.5, marginTop: 2,
            }}>
              TER27 · DIGITAL TWIN · v6.0
            </div>
            <div style={{ marginTop: 8 }}>
              <ModeToggle mode={mode} setMode={setMode} />
            </div>
          </div>

          {/* Navigation groups */}
          <div style={{ flex: 1, overflowY: "auto", padding: "6px 0" }}>
            {NAV_GROUPS.map(group => {
              const isCollapsed = collapsed[group.label];
              const hasActive = group.items.some(i => i.key === active);

              return (
                <div key={group.label} style={{ marginBottom: 2 }}>
                  {/* Group header */}
                  <button
                    onClick={() => toggleGroup(group.label)}
                    style={{
                      width: "100%", border: "none", cursor: "pointer",
                      display: "flex", alignItems: "center", gap: 6,
                      padding: "5px 14px",
                      background: hasActive ? `${group.accent}08` : "transparent",
                      borderLeft: hasActive ? `2px solid ${group.accent}` : "2px solid transparent",
                    }}
                  >
                    <span style={{
                      fontSize: 8, fontWeight: 800, letterSpacing: 2,
                      color: hasActive ? group.accent : C.dm,
                      fontFamily: C.dt, flex: 1, textAlign: "left",
                    }}>
                      {group.label}
                    </span>
                    <span style={{
                      fontSize: 8, color: C.dm, fontFamily: C.dt,
                      transform: isCollapsed ? "rotate(-90deg)" : "rotate(0)",
                      transition: "transform 0.15s",
                    }}>▾</span>
                  </button>

                  {/* Items */}
                  {!isCollapsed && group.items.map(item => {
                    const isA = active === item.key;
                    return (
                      <button
                        key={item.key}
                        onClick={() => setActive(item.key)}
                        style={{
                          width: "100%", border: "none", cursor: "pointer",
                          display: "flex", alignItems: "center", gap: 8,
                          padding: "4px 14px 4px 22px",
                          background: isA
                            ? `${group.accent}10`
                            : "transparent",
                          borderLeft: isA
                            ? `2px solid ${group.accent}`
                            : "2px solid transparent",
                          transition: "all 0.12s",
                        }}
                      >
                        <span style={{
                          fontSize: 13,
                          color: isA ? group.accent : C.dm,
                          fontFamily: C.hd,
                          width: 18, textAlign: "center",
                        }}>
                          {item.icon}
                        </span>
                        <span style={{
                          fontSize: 9.5,
                          fontWeight: isA ? 700 : 500,
                          color: isA ? group.accent : C.md,
                          letterSpacing: 0.8,
                          fontFamily: C.hd,
                        }}>
                          {item.label}
                        </span>
                      </button>
                    );
                  })}
                </div>
              );
            })}
          </div>

          {/* System health */}
          <SystemHealth />

          {/* Footer */}
          <div style={{
            padding: "8px 14px",
            borderTop: `1px solid ${C.b1}`,
            fontSize: 7, color: C.dm, fontFamily: C.dt,
            letterSpacing: 1,
          }}>
            <div>TECNUN eRACING · FSG 2026</div>
            <div style={{ marginTop: 2 }}>Siemens Digital Twin Award</div>
            <div style={{ marginTop: 2 }}>JAX/XLA · 46-DOF · 200 Hz</div>
          </div>
        </div>

        {/* ═══ MAIN CONTENT ══════════════════════════════════════════ */}
        <div style={{
          flex: 1, overflowY: "auto",
          display: "flex", flexDirection: "column", zIndex: 1,
        }}>
          {/* Header bar */}
          <div style={{
            display: "flex", justifyContent: "space-between",
            alignItems: "center", padding: "10px 24px",
            borderBottom: `1px solid ${C.b1}`,
            background: C.panel,
            backdropFilter: "blur(20px) saturate(1.2)",
            position: "sticky", top: 0, zIndex: 5,
          }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
              <span style={{
                fontSize: 14, fontWeight: 700,
                color: C.w, fontFamily: C.hd,
              }}>
                {ALL_NAV.find(n => n.key === active)?.label || "Overview"}
              </span>
              <span style={{
                fontSize: 8, color: C.dm,
                fontFamily: C.dt, letterSpacing: 1,
                textTransform: "uppercase",
              }}>
                {activeGroup}
              </span>
              <span style={{
                fontSize: 8, color: mode === "LIVE" ? C.gn : C.cy,
                fontFamily: C.dt, fontWeight: 600,
              }}>
                {mode === "LIVE"
                  ? "● LIVE — Real-time telemetry"
                  : "◇ ANALYZE — Post-run analysis"
                }
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              {/* Bloomberg-style ticker */}
              <div style={{ display: "flex", gap: 12, fontSize: 8, fontFamily: C.dt }}>
                <span style={{ color: C.dm }}>RT</span>
                <span style={{ color: C.gn }}>200 Hz</span>
                <span style={{ color: C.dm }}>│</span>
                <span style={{ color: C.dm }}>WS</span>
                <span style={{ color: mode === "LIVE" ? C.gn : C.dm }}>{mode === "LIVE" ? "60 Hz" : "—"}</span>
                <span style={{ color: C.dm }}>│</span>
                <span style={{ color: C.dm }}>DOF</span>
                <span style={{ color: C.cy }}>46</span>
              </div>
              <span style={{
                fontSize: 9, color: C.dm, fontFamily: C.dt,
                fontFeatureSettings: '"tnum"',
              }}>
                {time.toLocaleTimeString()}
              </span>
            </div>
          </div>

          {/* Module content */}
          <div style={{ flex: 1, padding: "14px 20px" }}>
            <FadeSlide keyVal={active + mode}>
              {renderContent()}
            </FadeSlide>
          </div>
        </div>
      </div>
    </SelectionProvider>
  );
}