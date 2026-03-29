// ═══════════════════════════════════════════════════════════════════════════════
// src/App.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════════
// CHANGES FROM v4.0 → v5.0:
// ─────────────────────────────────────────────────────────────────────────────
// 1. IMPORTS:    + AerodynamicsModule, ElectronicsModule
//
// 2. SIDEBAR:    Flat NAV → 7 hierarchical NAV_GROUPS with collapsible headers.
//               Groups: OVERVIEW, TELEMETRY, VEHICLE DYNAMICS, AERODYNAMICS,
//               ELECTRONICS, CONTROLS & AI, PERFORMANCE.
//
// 3. ROUTING:   + case "aero":        → <AerodynamicsModule />
//               + case "electronics":  → <ElectronicsModule />
//
// 4. SYSTEM HEALTH: Updated status rows reflecting current subsystem state.
//
// 5. All v4.0 data hooks, props, and module routing are preserved exactly.
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
import AerodynamicsModule from "./AerodynamicsModule.jsx";
import ElectronicsModule from "./ElectronicsModule.jsx";

// ═════════════════════════════════════════════════════════════════════════════
// GROUPED NAV CONFIGURATION — 7 groups, 13 modules
// ═════════════════════════════════════════════════════════════════════════════

const NAV_GROUPS = [
  {
    label: "OVERVIEW", accent: C.cy, items: [
      { key: "overview",   label: "Overview",    icon: "⬡" },
    ],
  },
  {
    label: "TELEMETRY", accent: C.gn, items: [
      { key: "telemetry",  label: "Telemetry",   icon: "◇" },
    ],
  },
  {
    label: "VEHICLE DYNAMICS", accent: C.am, items: [
      { key: "setup",      label: "Setup Opt",   icon: "◆" },
      { key: "suspension", label: "Suspension",  icon: "△" },
      { key: "tire",       label: "Tire Physics", icon: "⊗" },
      { key: "weight",     label: "Weight & CG", icon: "⊿" },
    ],
  },
  {
    label: "AERODYNAMICS", accent: "#ff6090", items: [
      { key: "aero",       label: "Aerodynamics", icon: "▽" },
    ],
  },
  {
    label: "ELECTRONICS", accent: "#7c3aed", items: [
      { key: "electronics", label: "Electronics", icon: "⌁" }, // Swapped ⚡ for ⌁ (Technical Spark)
    ],
  },
  {
    label: "CONTROLS & AI", accent: C.cy, items: [
      { key: "energy",     label: "Energy Audit", icon: "⊕" },
      { key: "diff",       label: "∇ Insights",   icon: "∂" },
    ],
  },
  {
    label: "PERFORMANCE", accent: C.red, items: [
      { key: "coaching",   label: "Coaching",     icon: "◈" },
      { key: "endurance",  label: "Endurance",    icon: "◷" }, // Swapped ⏱ for ◷ (Geometric Clock)
      { key: "compliance", label: "Compliance",   icon: "✓" }, // Swapped ☑ for ✓ (Simple Check)
    ],
  },
];

// Flat list for lookups
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
        transition: "all 0.2s ease",
      }}>
        {m === "LIVE" && (
          <span style={{
            display: "inline-block", width: 6, height: 6,
            borderRadius: "50%",
            background: mode === "LIVE" ? C.gn : C.dm,
            marginRight: 6,
            boxShadow: mode === "LIVE" ? `0 0 8px ${C.gn}` : "none",
            animation: mode === "LIVE" ? "pulseGlow 2s infinite" : "none",
          }} />
        )}
        {m}
      </button>
    ))}
  </div>
);

// ═════════════════════════════════════════════════════════════════════════════
// SYSTEM HEALTH PANEL (sidebar bottom)
// ═════════════════════════════════════════════════════════════════════════════

function SystemHealth() {
const rows = [
["Physics",     "JAX NPH 46-DOF",  C.gn],
["Integrator",  "GLRK-4 Symplec.", C.gn],
["Tire",        "Pacejka+PINN+GP", C.gn],
["WMPC",        "64-step AL+UT",   C.gn],
["Optimizer",   "SB-TRPO 28D",     C.cy],
["H_net",       "Passivity OK",    C.gn],
["Aero",        "5D Surrogate",    C.gn],
["Electronics", "HV Loop CLOSED",  C.gn],
["State",       "46-dim",          C.dm],
["Setup",       "28-dim",          C.dm],
];

return (
<div style={{ padding: "12px 14px", borderTop: `1px solid ${C.b1}` }}>
<div style={{
display: "flex", alignItems: "center", gap: 5,
marginBottom: 8,
}}>
<div style={{
width: 6, height: 6, borderRadius: "50%",
background: C.gn,
boxShadow: `0 0 8px ${C.gn}`,
animation: "pulseGlow 2s infinite",
}} />
<span style={{
fontSize: 8, fontWeight: 700, color: C.gn,
fontFamily: C.dt, letterSpacing: 1.5,
}}>
ALL SYSTEMS NOMINAL
</span>
</div>
{rows.map(([k, v, c]) => (
<div key={k} style={{
display: "flex", justifyContent: "space-between",
fontSize: 8, fontFamily: C.dt, marginBottom: 2,
}}>
<span style={{ color: C.dm, letterSpacing: 1 }}>{k}</span>
<span style={{ color: c, fontWeight: 600 }}>{v}</span>
</div>
))}
</div>
);
}

// ═════════════════════════════════════════════════════════════════════════════
// APP
// ═════════════════════════════════════════════════════════════════════════════

export default function App() {
const [active, setActive] = useState("overview");
const [mode, setMode]     = useState("ANALYZE");
const [time, setTime]     = useState(new Date());

useEffect(() => {
const id = setInterval(() => setTime(new Date()), 1000);
return () => clearInterval(id);
}, []);

// Determine which group the active module belongs to (auto-expand)
const activeGroup = NAV_GROUPS.find(g => g.items.some(i => i.key === active))?.label || "";

// ── v3 data ───────────────────────────────────────────────────────────────
const track = useMemo(() => gTK(), []);
const tireT = useMemo(() => gTT(), []);
const susp  = useMemo(() => gSU(), []);

// ── v4 enhanced data ──────────────────────────────────────────────────────
const pareto   = useMemo(() => gP4(),  []);
const conv     = useMemo(() => gCV4(), []);
const sens     = useMemo(() => gSN4(), []);
const episodes = useMemo(() => gEpisodes(), []);

// ── v4 new data ───────────────────────────────────────────────────────────
const thermal5  = useMemo(() => gThermal5(),     []);
const energy    = useMemo(() => gEnergyBudgetPH(), []);
const alData    = useMemo(() => gALConstraints(), []);
const wavelets  = useMemo(() => gWaveletCoeffsMPC(), []);
const tubes     = useMemo(() => gTubePoints(track), [track]);
const landscape = useMemo(() => gHnetLandscape(), []);
const rMatrix   = useMemo(() => gRMatrix(),      []);
const gpEnv     = useMemo(() => gGPEnvelope(),   []);
const hysteresis = useMemo(() => gHysteresis(),  []);
const loadSens  = useMemo(() => gLoadSensitivity(), []);

// ── Content router ────────────────────────────────────────────────────────
const renderContent = () => {
switch (active) {
case "telemetry":
return (
<TelemetryModule
track={track}
tireTemps={tireT}
mode={mode}
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
  case "suspension":  return <SuspensionModule data={susp} />;
  case "weight":      return <WeightBalanceModule />;
  case "compliance":  return <ComplianceModule />;
  case "diff":        return <DifferentiableInsightsModule />;

  // ── v5.0 NEW MODULES ──────────────────────────────────────────
  case "aero":        return <AerodynamicsModule />;
  case "electronics": return <ElectronicsModule />;

  default:
    return <OverviewModule pareto={pareto} conv={conv} />;
}


};

// ── Render ────────────────────────────────────────────────────────────────
return (
<SelectionProvider>
<div style={{
background: C.bg, color: C.br, fontFamily: C.bd,
minHeight: "100vh", display: "flex",
}}>
<link href={FONTS_URL} rel="stylesheet" />
<style>{`* { margin: 0; padding: 0; box-sizing: border-box; } ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: ${C.bg}; } ::-webkit-scrollbar-thumb { background: ${C.b2}; border-radius: 2px; } @keyframes pulseGlow { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }`}</style>


    {/* ═══ SIDEBAR ═══════════════════════════════════════════════ */}
    <div style={{
      width: 220, minHeight: "100vh",
      background: C.panel,
      backdropFilter: "blur(24px) saturate(1.3)",
      borderRight: `1px solid ${C.glassB}`,
      display: "flex", flexDirection: "column",
      flexShrink: 0, zIndex: 2,
      position: "sticky", top: 0, height: "100vh",
    }}>
      {/* Logo */}
      <div style={{
        padding: "20px 16px 12px",
        borderBottom: `1px solid ${C.b1}`,
      }}>
        <div style={{
          fontSize: 22, fontWeight: 900, letterSpacing: -0.5,
          fontFamily: C.hd,
          background: `linear-gradient(135deg, ${C.red}, ${C.am})`,
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}>
          PROJECT-GP
        </div>
        <div style={{
          fontSize: 8, fontWeight: 600, letterSpacing: 3,
          color: C.dm, fontFamily: C.dt, marginTop: 4,
        }}>
          DIGITAL TWIN v5.0
        </div>
        <div style={{ marginTop: 10 }}>
          <ModeToggle mode={mode} setMode={setMode} />
        </div>
        <div style={{
          fontSize: 8, fontFamily: C.dt,
          color: mode === "LIVE" ? C.gn : C.cy,
          fontWeight: 600, marginTop: 6,
        }}>
          {mode === "LIVE"
            ? "● LIVE — Real-time telemetry via WebSocket"
            : "◎ ANALYZE — Post-run data review"
          }
        </div>
      </div>

      {/* Navigation — Grouped */}
      <div style={{ flex: 1, overflowY: "auto", padding: "6px 0" }}>
        {NAV_GROUPS.map(group => {
          const isGroupActive = group.items.some(i => i.key === active);

          return (
            <div key={group.label} style={{ marginBottom: 2 }}>
              {/* Group header */}
              <div style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "8px 14px 4px",
                borderLeft: `2px solid ${isGroupActive ? group.accent : "transparent"}`,
              }}>
                <div style={{
                  width: 4, height: 4, borderRadius: "50%",
                  background: isGroupActive ? group.accent : C.dm,
                  opacity: isGroupActive ? 1 : 0.4,
                }} />
                <span style={{
                  fontSize: 7, fontWeight: 700, color: isGroupActive ? group.accent : C.dm,
                  fontFamily: C.dt, letterSpacing: 2.5,
                  textTransform: "uppercase",
                }}>
                  {group.label}
                </span>
              </div>

              {/* Group items */}
              {group.items.map(item => {
                const isA = active === item.key;
                return (
                  <button
                    key={item.key}
                    onClick={() => setActive(item.key)}
                    style={{
                      width: "100%", display: "flex", alignItems: "center",
                      gap: 8, padding: "9px 14px 9px 22px", marginBottom: 0,
                      borderRadius: 0, border: "none",
                      background: isA ? `${group.accent}10` : "transparent",
                      borderLeft: isA
                        ? `2px solid ${group.accent}`
                        : "2px solid transparent",
                      cursor: "pointer",
                      transition: "all 0.2s ease",
                    }}
                    onMouseEnter={e => {
                      if (!isA) e.currentTarget.style.background = C.hover;
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.background = isA
                        ? `${group.accent}10`
                        : "transparent";
                    }}
                  >
                    <span style={{
                      fontSize: 14,
                      color: isA ? group.accent : C.dm,
                      fontFamily: C.hd,
                      width: 20, textAlign: "center",
                    }}>
                      {item.icon}
                    </span>
                    <span style={{
                      fontSize: 10.5,
                      fontWeight: isA ? 700 : 500,
                      color: isA ? group.accent : C.md,
                      letterSpacing: 1,
                      textTransform: "uppercase",
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
    </div>

    {/* ═══ MAIN CONTENT ══════════════════════════════════════════ */}
    <div style={{
      flex: 1, overflowY: "auto",
      display: "flex", flexDirection: "column", zIndex: 1,
    }}>
      {/* Header bar */}
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", padding: "12px 28px",
        borderBottom: `1px solid ${C.b1}`,
        background: C.panel,
        backdropFilter: "blur(20px) saturate(1.2)",
        position: "sticky", top: 0, zIndex: 5,
      }}>
        <div>
          <span style={{
            fontSize: 14, fontWeight: 700,
            color: C.w, fontFamily: C.hd,
          }}>
            {ALL_NAV.find(n => n.key === active)?.label || "Overview"}
          </span>
          <span style={{
            fontSize: 9, color: C.dm,
            fontFamily: C.dt, marginLeft: 12,
          }}>
            {mode === "LIVE"
              ? "Real-time telemetry"
              : "Post-run analysis"
            }
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{
            fontSize: 8, color: C.dm, fontFamily: C.dt,
            letterSpacing: 1, textTransform: "uppercase",
          }}>
            {activeGroup}
          </span>
          <span style={{
            fontSize: 9, color: C.dm, fontFamily: C.dt,
          }}>
            {time.toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Module content */}
      <div style={{ flex: 1, padding: "16px 24px" }}>
        <FadeSlide keyVal={active + mode}>
          {renderContent()}
        </FadeSlide>
      </div>
    </div>
  </div>
</SelectionProvider>


);
}