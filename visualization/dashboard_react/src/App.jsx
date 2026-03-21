import React, { useState, useEffect, useMemo } from "react";
import { C, GL, FONTS_URL } from "./theme.js";
import { FadeSlide } from "./components.jsx";
import { gP, gCV, gTK, gTT, gSN, gSU } from "./data.js";
import OverviewModule from "./OverviewModule.jsx";
import SetupModule from "./SetupModule.jsx";
import TelemetryModule from "./TelemetryModule.jsx";
import SuspensionModule from "./SuspensionModule.jsx";

const NAV = [
  { key: "overview", label: "Overview", icon: "⬡" },
  { key: "setup", label: "Setup Opt", icon: "◆" },
  { key: "telemetry", label: "Telemetry", icon: "◇" },
  { key: "suspension", label: "Suspension", icon: "△" },
];

const ModeToggle = ({ mode, setMode }) => (
  <div style={{ display: "flex", borderRadius: 20, border: `1px solid ${C.b2}`, overflow: "hidden" }}>
    {["LIVE", "ANALYZE"].map(m => (
      <button key={m} onClick={() => setMode(m)} style={{
        padding: "5px 16px", border: "none", cursor: "pointer",
        fontSize: 10, fontWeight: 700, letterSpacing: 1.5, fontFamily: C.dt,
        background: mode === m ? (m === "LIVE" ? `${C.gn}20` : `${C.cy}20`) : "transparent",
        color: mode === m ? (m === "LIVE" ? C.gn : C.cy) : C.dm,
        transition: "all 0.2s ease",
      }}>
        {m === "LIVE" && <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: mode === "LIVE" ? C.gn : C.dm, marginRight: 6, boxShadow: mode === "LIVE" ? `0 0 8px ${C.gn}` : "none", animation: mode === "LIVE" ? "pulseGlow 2s infinite" : "none" }} />}
        {m}
      </button>
    ))}
  </div>
);

export default function App() {
  const [active, setActive] = useState("overview");
  const [mode, setMode] = useState("ANALYZE");
  const [time, setTime] = useState(new Date());
  useEffect(() => { const id = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(id); }, []);

  const pareto = useMemo(() => gP(), []);
  const conv = useMemo(() => gCV(), []);
  const sens = useMemo(() => gSN(), []);
  const track = useMemo(() => gTK(), []);
  const tireT = useMemo(() => gTT(), []);
  const susp = useMemo(() => gSU(), []);

  const renderContent = () => {
    switch (active) {
      case "setup": return <SetupModule pareto={pareto} conv={conv} sens={sens} track={track} />;
      case "telemetry": return <TelemetryModule track={track} tireTemps={tireT} />;
      case "suspension": return <SuspensionModule data={susp} />;
      default: return <OverviewModule />;
    }
  };

  return (
    <div style={{ background: C.bg, color: C.br, fontFamily: C.bd, minHeight: "100vh", display: "flex" }}>
      <link href={FONTS_URL} rel="stylesheet" />

      {/* SIDEBAR */}
      <div style={{ width: 210, minHeight: "100vh", background: C.panel, backdropFilter: "blur(24px) saturate(1.3)", borderRight: `1px solid ${C.glassB}`, display: "flex", flexDirection: "column", flexShrink: 0, zIndex: 2 }}>
        <div style={{ padding: "22px 18px 16px", borderBottom: `1px solid ${C.b1}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 8, background: `linear-gradient(135deg, ${C.red}, #ff3d00)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 900, color: "#fff", fontFamily: C.hd, boxShadow: `0 3px 18px ${C.red}35` }}>GP</div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 900, color: C.w, letterSpacing: 2.5, fontFamily: C.hd }}>PROJECT-GP</div>
              <div style={{ fontSize: 8, color: C.dm, letterSpacing: 2.5, fontFamily: C.dt, textTransform: "uppercase" }}>Digital Twin · Ter26</div>
            </div>
          </div>
        </div>
        <div style={{ padding: "14px 10px", flex: 1 }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: C.dm, letterSpacing: 3, textTransform: "uppercase", padding: "0 10px", marginBottom: 10, fontFamily: C.dt }}>Modules</div>
          {NAV.map(item => {
            const isA = active === item.key;
            return (
              <button key={item.key} onClick={() => setActive(item.key)} style={{
                width: "100%", display: "flex", alignItems: "center", gap: 10,
                padding: "11px 12px", marginBottom: 2, borderRadius: 8, border: "none",
                background: isA ? `${C.red}12` : "transparent",
                borderLeft: isA ? `2px solid ${C.red}` : "2px solid transparent",
                cursor: "pointer", transition: "all 0.2s ease",
              }}
                onMouseEnter={e => { if (!isA) e.currentTarget.style.background = C.hover; }}
                onMouseLeave={e => { if (!isA) e.currentTarget.style.background = isA ? `${C.red}12` : "transparent"; }}
              >
                <span style={{ fontSize: 16, color: isA ? C.red : C.dm, fontFamily: C.hd }}>{item.icon}</span>
                <span style={{ fontSize: 11, fontWeight: isA ? 700 : 500, color: isA ? C.red : C.md, letterSpacing: 1.5, textTransform: "uppercase", fontFamily: C.hd }}>{item.label}</span>
              </button>
            );
          })}
        </div>
        {/* Mode info in sidebar */}
        <div style={{ padding: "10px 16px", borderTop: `1px solid ${C.b1}`, borderBottom: `1px solid ${C.b1}` }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: C.dm, letterSpacing: 2, textTransform: "uppercase", marginBottom: 6, fontFamily: C.dt }}>Mode</div>
          <div style={{ fontSize: 9, color: mode === "LIVE" ? C.gn : C.cy, fontFamily: C.dt, fontWeight: 600 }}>
            {mode === "LIVE" ? "● LIVE — Real-time telemetry from physics_server.py via WebSocket" : "◎ ANALYZE — Post-run data review with Diff-WMPC optimal overlay"}
          </div>
        </div>
        <div style={{ padding: "14px 18px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: C.gn, boxShadow: `0 0 10px ${C.gn}`, animation: "pulseGlow 2s ease-in-out infinite" }} />
            <span style={{ fontSize: 9, fontWeight: 700, color: C.gn, letterSpacing: 2, textTransform: "uppercase", fontFamily: C.dt }}>Online</span>
          </div>
          {[["Physics", "JAX NPH"], ["Solver", "Diff-WMPC"], ["Optim", "SB-TRPO"]].map(([k, v]) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 9, fontFamily: C.dt, marginBottom: 3 }}>
              <span style={{ color: C.dm, textTransform: "uppercase", letterSpacing: 1 }}>{k}</span>
              <span style={{ color: C.br, fontWeight: 600 }}>{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* MAIN */}
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", zIndex: 1 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 28px", borderBottom: `1px solid ${C.b1}`, background: C.panel, backdropFilter: "blur(20px)", position: "sticky", top: 0, zIndex: 10 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 900, color: C.w, letterSpacing: 3, textTransform: "uppercase", fontFamily: C.hd }}>
              {NAV.find(n => n.key === active)?.label}
            </h1>
            <div style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, marginTop: 2 }}>Ter26 Formula Student · FSG 2026 · Siemens Digital Twin Award</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <ModeToggle mode={mode} setMode={setMode} />
            <div style={{ width: 1, height: 24, background: C.b1 }} />
            <div style={{ ...GL, padding: "5px 16px", borderRadius: 20, fontSize: 12, fontFamily: C.dt, color: C.w, fontWeight: 600, letterSpacing: 1.5 }}>
              {time.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </div>
            <div style={{ fontSize: 10, color: C.md, fontFamily: C.hd, fontWeight: 500 }}>
              {time.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })}
            </div>
          </div>
        </div>

        {/* LIVE mode banner */}
        {mode === "LIVE" && (
          <div style={{ padding: "8px 28px", background: `${C.gn}08`, borderBottom: `1px solid ${C.gn}20`, display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.gn, boxShadow: `0 0 8px ${C.gn}`, animation: "pulseGlow 1s infinite" }} />
            <span style={{ fontSize: 10, fontFamily: C.dt, color: C.gn, fontWeight: 600, letterSpacing: 1 }}>LIVE TELEMETRY — Waiting for physics_server.py connection on ws://localhost:5001</span>
          </div>
        )}

        <div style={{ padding: "22px 28px", flex: 1 }}>
          <FadeSlide keyVal={active}>{renderContent()}</FadeSlide>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 28px", borderTop: `1px solid ${C.b1}`, fontSize: 9, color: C.dm, fontFamily: C.dt, letterSpacing: 1.2, background: "rgba(5,7,11,0.85)" }}>
          <span>PROJECT-GP v3.0 · 100% JAX/Flax End-to-End Differentiable</span>
          <span>46-state · 28-setup · MF6.2+PINN/GP · Diff-WMPC · MORL-SB-TRPO</span>
        </div>
      </div>

      <style>{`
        @keyframes pulseGlow { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.b2}; border-radius: 3px; }
        button { font-family: inherit; }
        canvas { display: block; }
      `}</style>
    </div>
  );
}
