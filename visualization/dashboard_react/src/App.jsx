// ═══════════════════════════════════════════════════════════════════════════════
// src/App.jsx — Project-GP Dashboard v4.0
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHANGES FROM v3.0:
// ─────────────────────────────────────────────────────────────────────────────
// 1. IMPORTS:    + EnergyAuditModule, TirePhysicsModule
//               + SelectionProvider from context/SelectionContext.jsx
//               + 10 new data generators from data.js (gThermal5, gEnergyBudget,
//                 gALConstraints, gWaveletCoeffs, gTubePoints, gHnetLandscape,
//                 gRMatrix, gGPEnvelope, gHysteresis, gLoadSensitivity,
//                 gP4, gCV4, gSN4, gEpisodes)
//
// 2. NAV:       + { key: "energy",  label: "Energy Audit",  icon: "⊕" }
//               + { key: "tire",    label: "Tire Physics",  icon: "⊗" }
//
// 3. DATA:      + 10 new useMemo hooks for new generators
//               + gP → gP4, gCV → gCV4, gSN → gSN4 (enhanced versions with
//                 hvContrib, pgNorm, hvol, dTherm)
//
// 4. ROUTING:   + case "energy": → <EnergyAuditModule ... />
//               + case "tire":   → <TirePhysicsModule ... />
//               + TelemetryModule now receives thermal5, tubes, wavelets,
//                 alData, energy as additional props
//               + SetupModule now receives episodes for MORL convergence
//               + OverviewModule now receives conv for convergence chart
//
// 5. LAYOUT:    + SelectionProvider wraps the entire app
//               + Sidebar system health panel expanded with 3 new status rows
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useMemo } from "react";
import { C, GL, FONTS_URL } from "./theme.js";
import { FadeSlide } from "./components.jsx";
import { SelectionProvider } from "./context/SelectionContext.jsx";

// ── Data generators ──────────────────────────────────────────────────────────
// Original v3 generators (keep for backward compat where modules still use them)
import { gP, gCV, gTK, gTT, gSN, gSU } from "./data.js";

// v4 enhanced/new generators
import {
  gP4, gCV4, gSN4,
  gThermal5, gEnergyBudget, gALConstraints, gWaveletCoeffs,
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


// ═════════════════════════════════════════════════════════════════════════════
// NAV CONFIGURATION
// ═════════════════════════════════════════════════════════════════════════════

const NAV = [
  { key: "overview",   label: "Overview",      icon: "⬡" },
  { key: "telemetry",  label: "Telemetry",     icon: "◇" },
  { key: "setup",      label: "Setup Opt",     icon: "◆" },
  { key: "energy",     label: "Energy Audit",  icon: "⊕" },
  { key: "tire",       label: "Tire Physics",  icon: "⊗" },
  { key: "suspension", label: "Suspension",    icon: "△" },
];


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
// SYSTEM HEALTH PANEL (sidebar)
// ═════════════════════════════════════════════════════════════════════════════

function SystemHealth() {
  const rows = [
    ["Physics",     "JAX NPH 46-DOF",  C.gn],
    ["Integrator",  "GLRK-4 Symplec.", C.gn],
    ["Tire",        "Pacejka+PINN+GP", C.gn],
    ["WMPC",        "64-step AL+UT",   C.gn],
    ["Optimizer",   "SB-TRPO 28D",     C.cy],
    ["H_net",       "~450 J/s warn",   C.am],
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
          ONLINE
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

  // ── v3 data (kept for OverviewModule / SuspensionModule backward compat) ──
  const track = useMemo(() => gTK(), []);
  const tireT = useMemo(() => gTT(), []);
  const susp  = useMemo(() => gSU(), []);

  // ── v4 enhanced data ──────────────────────────────────────────────────────
  const pareto   = useMemo(() => gP4(),  []);
  const conv     = useMemo(() => gCV4(), []);
  const sens     = useMemo(() => gSN4(), []);
  const episodes = useMemo(() => gEpisodes(), []);

  // ── v4 new data ───────────────────────────────────────────────────────────
  const thermal5  = useMemo(() => gThermal5(),    []);
  const energy    = useMemo(() => gEnergyBudget(),[]);
  const alData    = useMemo(() => gALConstraints(),[]);
  const wavelets  = useMemo(() => gWaveletCoeffs(),[]);
  const tubes     = useMemo(() => gTubePoints(track), [track]);
  const landscape = useMemo(() => gHnetLandscape(),[]);
  const rMatrix   = useMemo(() => gRMatrix(),     []);
  const gpEnv     = useMemo(() => gGPEnvelope(),  []);
  const hysteresis= useMemo(() => gHysteresis(),  []);
  const loadSens  = useMemo(() => gLoadSensitivity(),[]);

  // ── Content router ────────────────────────────────────────────────────────
  const renderContent = () => {
    switch (active) {
      case "telemetry":
        return (
          <TelemetryModule
            track={track}
            tireTemps={tireT}
            mode={mode}
            // v4 additions:
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
            // v4 addition:
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

      case "suspension":
        return <SuspensionModule data={susp} />;

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
        <style>{`
          * { margin: 0; padding: 0; box-sizing: border-box; }
          ::-webkit-scrollbar { width: 4px; }
          ::-webkit-scrollbar-track { background: ${C.bg}; }
          ::-webkit-scrollbar-thumb { background: ${C.b2}; border-radius: 2px; }
          @keyframes pulseGlow { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        `}</style>

        {/* ═══ SIDEBAR ═══════════════════════════════════════════════ */}
        <div style={{
          width: 210, minHeight: "100vh",
          background: C.panel,
          backdropFilter: "blur(24px) saturate(1.3)",
          borderRight: `1px solid ${C.glassB}`,
          display: "flex", flexDirection: "column",
          flexShrink: 0, zIndex: 2,
          position: "sticky", top: 0, height: "100vh",
        }}>
          {/* Logo */}
          <div style={{
            padding: "22px 18px 16px",
            borderBottom: `1px solid ${C.b1}`,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <svg width="36" height="36" viewBox="0 0 64 64" style={{ flexShrink: 0 }}>
                <defs>
                  <linearGradient id="logoBg" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#e10600" />
                    <stop offset="100%" stopColor="#ff3d00" />
                  </linearGradient>
                </defs>
                <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#logoBg)" />
                <rect x="4" y="4" width="56" height="56" rx="12" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
                <text x="32" y="44" textAnchor="middle" fontFamily="Arial, Helvetica, sans-serif" fontWeight="900" fontSize="32" fill="white" letterSpacing="-1">GP</text>
                <rect x="14" y="52" width="36" height="2" rx="1" fill="rgba(255,255,255,0.3)" />
              </svg>
              <div>
                <div style={{
                  fontSize: 15, fontWeight: 900, color: C.w,
                  letterSpacing: 2.5, fontFamily: C.hd,
                }}>
                  PROJECT-GP
                </div>
                <div style={{
                  fontSize: 8, color: C.dm, letterSpacing: 2.5,
                  fontFamily: C.dt, textTransform: "uppercase",
                }}>
                  Digital Twin · v4.0
                </div>
              </div>
            </div>
          </div>

          {/* Mode toggle */}
          <div style={{
            padding: "10px 14px",
            borderBottom: `1px solid ${C.b1}`,
          }}>
            <ModeToggle mode={mode} setMode={setMode} />
            <div style={{
              fontSize: 8, color: mode === "LIVE" ? C.gn : C.cy,
              fontFamily: C.dt, fontWeight: 600, marginTop: 6,
            }}>
              {mode === "LIVE"
                ? "● LIVE — Real-time telemetry via WebSocket"
                : "◎ ANALYZE — Post-run data review"
              }
            </div>
          </div>

          {/* Navigation */}
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
            <div style={{
              fontSize: 8, fontWeight: 700, color: C.dm,
              letterSpacing: 3, textTransform: "uppercase",
              padding: "0 10px", marginBottom: 10, fontFamily: C.dt,
            }}>
              Modules
            </div>
            {NAV.map(item => {
              const isA = active === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => setActive(item.key)}
                  style={{
                    width: "100%", display: "flex", alignItems: "center",
                    gap: 10, padding: "11px 12px", marginBottom: 2,
                    borderRadius: 8, border: "none",
                    background: isA ? `${C.red}12` : "transparent",
                    borderLeft: isA
                      ? `2px solid ${C.red}`
                      : "2px solid transparent",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseEnter={e => {
                    if (!isA) e.currentTarget.style.background = C.hover;
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.background = isA
                      ? `${C.red}12`
                      : "transparent";
                  }}
                >
                  <span style={{
                    fontSize: 16,
                    color: isA ? C.red : C.dm,
                    fontFamily: C.hd,
                  }}>
                    {item.icon}
                  </span>
                  <span style={{
                    fontSize: 11,
                    fontWeight: isA ? 700 : 500,
                    color: isA ? C.red : C.md,
                    letterSpacing: 1.5,
                    textTransform: "uppercase",
                    fontFamily: C.hd,
                  }}>
                    {item.label}
                  </span>
                </button>
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
                {NAV.find(n => n.key === active)?.label || "Overview"}
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
            <div style={{
              fontSize: 9, color: C.dm, fontFamily: C.dt,
            }}>
              {time.toLocaleTimeString()}
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