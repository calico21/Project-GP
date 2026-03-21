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

export default function App() {
  const [active, setActive] = useState("overview");
  const [time, setTime] = useState(new Date());
  useEffect(() => { const id = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(id); }, []);

  const pareto = useMemo(() => gP(), []);
  const conv = useMemo(() => gCV(), []);
  const sens = useMemo(() => gSN(), []);
  const track = useMemo(() => gTK(), []);
  const tireT = useMemo(() => gTT(), []);
  const susp = useMemo(() => gSU(), []);

  const content = () => {
    switch (active) {
      case "setup": return <SetupModule pareto={pareto} conv={conv} sens={sens} />;
      case "telemetry": return <TelemetryModule track={track} tireTemps={tireT} />;
      case "suspension": return <SuspensionModule data={susp} />;
      default: return <OverviewModule />;
    }
  };

  return (
    <div style={{ background: C.bg, color: C.br, fontFamily: C.bd, minHeight: "100vh", display: "flex", position: "relative", overflow: "hidden" }}>
      <link href={FONTS_URL} rel="stylesheet" />

      {/* Ambient glows */}
      <div style={{ position: "fixed", top: -250, right: -200, width: 700, height: 700, borderRadius: "50%", background: `radial-gradient(circle, ${C.red}05, transparent 65%)`, pointerEvents: "none", zIndex: 0 }} />
      <div style={{ position: "fixed", bottom: -350, left: -150, width: 900, height: 900, borderRadius: "50%", background: `radial-gradient(circle, ${C.cy}03, transparent 65%)`, pointerEvents: "none", zIndex: 0 }} />

      {/* ── SIDEBAR ──────────────────────────────────── */}
      <div style={{ width: 210, minHeight: "100vh", background: C.panel, backdropFilter: "blur(24px) saturate(1.3)", borderRight: `1px solid ${C.glassB}`, display: "flex", flexDirection: "column", flexShrink: 0, zIndex: 2 }}>
        {/* Brand */}
        <div style={{ padding: "22px 18px 16px", borderBottom: `1px solid ${C.b1}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 8, background: `linear-gradient(135deg, ${C.red}, #ff3d00)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 900, color: "#fff", fontFamily: C.hd, boxShadow: `0 3px 18px ${C.red}35` }}>GP</div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 900, color: C.w, letterSpacing: 2.5, fontFamily: C.hd }}>PROJECT-GP</div>
              <div style={{ fontSize: 8, color: C.dm, letterSpacing: 2.5, fontFamily: C.dt, textTransform: "uppercase" }}>Digital Twin · Ter26</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <div style={{ padding: "14px 10px", flex: 1 }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: C.dm, letterSpacing: 3, textTransform: "uppercase", padding: "0 10px", marginBottom: 10, fontFamily: C.dt }}>Modules</div>
          {NAV.map(item => {
            const isA = active === item.key;
            return (
              <button key={item.key} onClick={() => setActive(item.key)}
                style={{
                  width: "100%", display: "flex", alignItems: "center", gap: 10,
                  padding: "11px 12px", marginBottom: 2, borderRadius: 8, border: "none",
                  background: isA ? `${C.red}12` : "transparent",
                  borderLeft: isA ? `2px solid ${C.red}` : "2px solid transparent",
                  cursor: "pointer", transition: "all 0.2s cubic-bezier(0.25,0.1,0.25,1)",
                }}
                onMouseEnter={e => { if (!isA) e.currentTarget.style.background = C.hover; }}
                onMouseLeave={e => { if (!isA) e.currentTarget.style.background = isA ? `${C.red}12` : "transparent"; }}
              >
                <span style={{ fontSize: 16, color: isA ? C.red : C.dm, fontFamily: C.hd, transition: "color 0.2s" }}>{item.icon}</span>
                <span style={{ fontSize: 11, fontWeight: isA ? 700 : 500, color: isA ? C.red : C.md, letterSpacing: 1.5, textTransform: "uppercase", fontFamily: C.hd, transition: "all 0.2s" }}>{item.label}</span>
              </button>
            );
          })}
        </div>

        {/* Status */}
        <div style={{ padding: "14px 18px", borderTop: `1px solid ${C.b1}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
            <div className="pulse-dot" style={{ width: 7, height: 7, borderRadius: "50%", background: C.gn, boxShadow: `0 0 10px ${C.gn}` }} />
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

      {/* ── MAIN CONTENT ─────────────────────────────── */}
      <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", zIndex: 1 }}>
        {/* Top bar */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 28px", borderBottom: `1px solid ${C.b1}`, background: C.panel, backdropFilter: "blur(20px)", position: "sticky", top: 0, zIndex: 10 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 900, color: C.w, letterSpacing: 3, textTransform: "uppercase", fontFamily: C.hd }}>{NAV.find(n => n.key === active)?.label}</h1>
            <div style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, marginTop: 2 }}>Ter26 Formula Student · FSG 2026 · Siemens Digital Twin Award</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ ...GL, padding: "5px 16px", borderRadius: 20, fontSize: 12, fontFamily: C.dt, color: C.w, fontWeight: 600, letterSpacing: 1.5 }}>
              {time.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </div>
            <div style={{ fontSize: 10, color: C.md, fontFamily: C.hd, fontWeight: 500 }}>
              {time.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })}
            </div>
          </div>
        </div>

        {/* Content */}
        <div style={{ padding: "22px 28px", flex: 1 }}>
          <FadeSlide keyVal={active}>{content()}</FadeSlide>
        </div>

        {/* Footer */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 28px", borderTop: `1px solid ${C.b1}`, fontSize: 9, color: C.dm, fontFamily: C.dt, letterSpacing: 1.2, background: "rgba(5,7,11,0.85)" }}>
          <span>PROJECT-GP v3.0 · 100% JAX/Flax End-to-End Differentiable</span>
          <span>46-state · 28-setup · MF6.2+PINN/GP · Diff-WMPC · MORL-SB-TRPO</span>
        </div>
      </div>

      <style>{`
        @keyframes pulseGlow { 0%, 100% { opacity: 1; box-shadow: 0 0 10px ${C.gn}; } 50% { opacity: 0.3; box-shadow: 0 0 4px ${C.gn}; } }
        .pulse-dot { animation: pulseGlow 2s ease-in-out infinite; }
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
