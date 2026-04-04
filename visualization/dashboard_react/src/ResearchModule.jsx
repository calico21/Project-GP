// ═══════════════════════════════════════════════════════════════════════════
// src/ResearchModule.jsx — Project-GP Dashboard v4.3
// ═══════════════════════════════════════════════════════════════════════════
// RESEARCH TAB: Parameter freeze/unfreeze controller for suspension geometry
// design exploration. Switch between Ter26 RWD / Ter27 4WD, toggle individual
// parameters on/off, set values via sliders, export config as JSON.
//
// The output JSON maps 1:1 to the Python DesignFreeze class:
//   { car: "ter27", frozen: { castor_f: 5.5, anti_dive_f: 0.35 } }
//
// Integration (3 lines in App.jsx):
//   NAV: { key: "research", label: "Research", icon: "⚙" }
//   Import: import ResearchModule from "./ResearchModule.jsx"
//   Route: case "research": return <ResearchModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";
import ResearchVisualizer from "./ResearchVisualizer.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETER DEFINITIONS — mirrors vehicle_dynamics.py SETUP_NAMES
// ═══════════════════════════════════════════════════════════════════════════
const PARAMS = [
  { idx: 0,  key: "k_f",            label: "Spring rate F",     unit: "N/m",   cat: "springs",   step: 500,   fmt: 0 },
  { idx: 1,  key: "k_r",            label: "Spring rate R",     unit: "N/m",   cat: "springs",   step: 500,   fmt: 0 },
  { idx: 2,  key: "arb_f",          label: "ARB front",         unit: "N·m/rad",cat: "springs",  step: 50,    fmt: 0 },
  { idx: 3,  key: "arb_r",          label: "ARB rear",          unit: "N·m/rad",cat: "springs",  step: 50,    fmt: 0 },
  { idx: 4,  key: "c_low_f",        label: "Damper LS front",   unit: "N·s/m", cat: "dampers",   step: 50,    fmt: 0 },
  { idx: 5,  key: "c_low_r",        label: "Damper LS rear",    unit: "N·s/m", cat: "dampers",   step: 50,    fmt: 0 },
  { idx: 6,  key: "c_high_f",       label: "Damper HS front",   unit: "N·s/m", cat: "dampers",   step: 25,    fmt: 0 },
  { idx: 7,  key: "c_high_r",       label: "Damper HS rear",    unit: "N·s/m", cat: "dampers",   step: 25,    fmt: 0 },
  { idx: 8,  key: "v_knee_f",       label: "Knee vel front",    unit: "m/s",   cat: "dampers",   step: 0.01,  fmt: 3 },
  { idx: 9,  key: "v_knee_r",       label: "Knee vel rear",     unit: "m/s",   cat: "dampers",   step: 0.01,  fmt: 3 },
  { idx: 10, key: "rebound_ratio_f", label: "Rebound ratio F",  unit: "—",     cat: "dampers",   step: 0.05,  fmt: 2 },
  { idx: 11, key: "rebound_ratio_r", label: "Rebound ratio R",  unit: "—",     cat: "dampers",   step: 0.05,  fmt: 2 },
  { idx: 12, key: "h_ride_f",       label: "Ride height F",     unit: "mm",    cat: "geometry",  step: 0.001, fmt: 1, scale: 1000 },
  { idx: 13, key: "h_ride_r",       label: "Ride height R",     unit: "mm",    cat: "geometry",  step: 0.001, fmt: 1, scale: 1000 },
  { idx: 14, key: "camber_f",       label: "Camber front",      unit: "deg",   cat: "alignment", step: 0.1,   fmt: 2 },
  { idx: 15, key: "camber_r",       label: "Camber rear",       unit: "deg",   cat: "alignment", step: 0.1,   fmt: 2 },
  { idx: 16, key: "toe_f",          label: "Toe front",         unit: "deg",   cat: "alignment", step: 0.01,  fmt: 3 },
  { idx: 17, key: "toe_r",          label: "Toe rear",          unit: "deg",   cat: "alignment", step: 0.01,  fmt: 3 },
  { idx: 18, key: "castor_f",       label: "Caster angle",      unit: "deg",   cat: "geometry",  step: 0.1,   fmt: 1 },
  { idx: 19, key: "anti_squat",     label: "Anti-squat %",      unit: "%",     cat: "anti_geom", step: 0.01,  fmt: 0, scale: 100 },
  { idx: 20, key: "anti_dive_f",    label: "Anti-dive F %",     unit: "%",     cat: "anti_geom", step: 0.01,  fmt: 0, scale: 100 },
  { idx: 21, key: "anti_dive_r",    label: "Anti-dive R %",     unit: "%",     cat: "anti_geom", step: 0.01,  fmt: 0, scale: 100 },
  { idx: 22, key: "anti_lift",      label: "Anti-lift %",       unit: "%",     cat: "anti_geom", step: 0.01,  fmt: 0, scale: 100 },
  { idx: 23, key: "diff_lock_ratio", label: "Diff lock ratio",  unit: "—",     cat: "drivetrain",step: 0.05,  fmt: 2 },
  { idx: 24, key: "brake_bias_f",   label: "Brake bias F",      unit: "%",     cat: "drivetrain",step: 0.01,  fmt: 0, scale: 100 },
  { idx: 25, key: "h_cg",           label: "CG height",         unit: "mm",    cat: "geometry",  step: 0.005, fmt: 0, scale: 1000 },
  { idx: 26, key: "bump_steer_f",   label: "Bump steer F",      unit: "rad/m", cat: "geometry",  step: 0.005, fmt: 4 },
  { idx: 27, key: "bump_steer_r",   label: "Bump steer R",      unit: "rad/m", cat: "geometry",  step: 0.005, fmt: 4 },
];

const CATEGORIES = [
  { key: "geometry",  label: "Geometry",       color: C.cy,  icon: "△" },
  { key: "alignment", label: "Alignment",      color: C.am,  icon: "◇" },
  { key: "anti_geom", label: "Anti-Geometry",   color: C.red, icon: "⊕" },
  { key: "springs",   label: "Springs & ARBs", color: C.gn,  icon: "⊗" },
  { key: "dampers",   label: "Dampers",        color: C.pr,  icon: "◈" },
  { key: "drivetrain",label: "Drivetrain",     color: "#ff8c00", icon: "◆" },
];

// ═══════════════════════════════════════════════════════════════════════════
// CAR CONFIGS — bounds + defaults per car (mirrors Python car_config.py)
// ═══════════════════════════════════════════════════════════════════════════
const CARS = {
  ter26: {
    label: "Ter26 RWD", season: 2026, drivetrain: "RWD", mass: 300,
    defaults: [35000,38000,800,600,1800,1800,1200,1200,0.10,0.10,1.50,1.50,0.025,0.022,-2.0,-1.5,-0.10,-0.15,5.0,0.30,0.40,0.10,0.20,0.30,0.60,0.285,0,0],
    lb: [10000,10000,0,0,200,200,50,50,0.03,0.03,1.0,1.0,0.010,0.010,-5.0,-5.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.50,0.18,-0.05,-0.05],
    ub: [120000,120000,5000,5000,8000,8000,4000,4000,0.40,0.40,3.0,3.0,0.060,0.060,0.0,0.0,1.0,1.0,10.0,0.9,0.9,0.9,0.9,1.0,0.80,0.45,0.05,0.05],
  },
  ter27: {
    label: "Ter27 4WD", season: 2027, drivetrain: "4WD", mass: 320,
    defaults: [40000,48000,400,300,2200,1800,900,700,0.10,0.10,1.70,1.55,0.028,0.028,-2.5,-1.8,-0.08,0.00,5.5,0.35,0.35,0.15,0.20,0.0,0.55,0.310,0,0],
    lb: [15000,15000,0,0,200,200,50,50,0.03,0.03,1.0,1.0,0.010,0.010,-6.0,-5.0,-2.0,-2.0,2.0,0.0,0.0,0.0,0.0,0.0,0.40,0.25,-0.08,-0.08],
    ub: [140000,130000,6000,5000,10000,9000,5000,4500,0.40,0.40,3.0,3.0,0.060,0.060,0.0,0.0,2.0,2.0,12.0,0.95,0.95,0.95,0.95,1.0,0.75,0.40,0.08,0.08],
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// PHASE PRESETS
// ═══════════════════════════════════════════════════════════════════════════
const PRESETS = {
  all_free:  { label: "Phase 1 — All Free",        frozen: [] },
  phase2:    { label: "Phase 2 — Hardpoints Locked", frozen: ["castor_f","anti_squat","anti_dive_f","anti_dive_r","anti_lift","bump_steer_f","bump_steer_r","h_cg"] },
  phase3:    { label: "Phase 3 — Track Tuning",      frozen: ["castor_f","anti_squat","anti_dive_f","anti_dive_r","anti_lift","bump_steer_f","bump_steer_r","h_cg","camber_f","camber_r","toe_f","toe_r","diff_lock_ratio"] },
};

// ═══════════════════════════════════════════════════════════════════════════
// STORAGE HELPERS — persist freeze config across sessions
// ═══════════════════════════════════════════════════════════════════════════
const STORAGE_KEY = "research-freeze-config";

async function loadConfig() {
  try {
    const r = await window.storage.get(STORAGE_KEY);
    return r ? JSON.parse(r.value) : null;
  } catch { return null; }
}

async function saveConfig(cfg) {
  try {
    await window.storage.set(STORAGE_KEY, JSON.stringify(cfg));
  } catch (e) { console.warn("Storage save failed:", e); }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════
export default function ResearchModule() {
  const [car, setCar]       = useState("ter27");
  const [frozen, setFrozen] = useState({});  // { paramKey: true/false }
  const [values, setValues] = useState({});  // { paramKey: number }
  const [loaded, setLoaded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState("config");

  const cfg = CARS[car];

  // ── Init values from car defaults ─────────────────────────────────────
  useEffect(() => {
    const v = {};
    PARAMS.forEach((p, i) => { v[p.key] = cfg.defaults[i]; });
    setValues(v);
    setFrozen({});
  }, [car]);

  // ── Load saved config on mount ────────────────────────────────────────
  useEffect(() => {
    (async () => {
      const saved = await loadConfig();
      if (saved) {
        if (saved.car && CARS[saved.car]) setCar(saved.car);
        if (saved.frozen) {
          const fr = {};
          Object.keys(saved.frozen).forEach(k => { fr[k] = true; });
          setFrozen(fr);
          setValues(prev => ({ ...prev, ...saved.frozen }));
        }
      }
      setLoaded(true);
    })();
  }, []);

  // ── Auto-save on change ───────────────────────────────────────────────
  useEffect(() => {
    if (!loaded) return;
    const frozenValues = {};
    Object.keys(frozen).filter(k => frozen[k]).forEach(k => {
      frozenValues[k] = values[k] ?? cfg.defaults[PARAMS.find(p => p.key === k)?.idx ?? 0];
    });
    saveConfig({ car, frozen: frozenValues });
  }, [car, frozen, values, loaded]);

  // ── Derived stats ─────────────────────────────────────────────────────
  const nFrozen = Object.values(frozen).filter(Boolean).length;
  const nFree   = 28 - nFrozen;

  // ── Handlers ──────────────────────────────────────────────────────────
  const toggleFreeze = useCallback((key) => {
    setFrozen(prev => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const setValue = useCallback((key, val) => {
    setValues(prev => ({ ...prev, [key]: val }));
  }, []);

  const applyPreset = useCallback((presetKey) => {
    const preset = PRESETS[presetKey];
    const fr = {};
    preset.frozen.forEach(k => { fr[k] = true; });
    setFrozen(fr);
  }, []);

  const freezeCategory = useCallback((catKey, freeze) => {
    setFrozen(prev => {
      const next = { ...prev };
      PARAMS.filter(p => p.cat === catKey).forEach(p => { next[p.key] = freeze; });
      return next;
    });
  }, []);

  const resetToDefaults = useCallback(() => {
    const v = {};
    PARAMS.forEach((p, i) => { v[p.key] = cfg.defaults[i]; });
    setValues(v);
  }, [cfg]);

  // ── Export JSON ───────────────────────────────────────────────────────
  const exportJSON = useMemo(() => {
    const frozenValues = {};
    Object.keys(frozen).filter(k => frozen[k]).forEach(k => {
      frozenValues[k] = values[k] ?? cfg.defaults[PARAMS.find(p => p.key === k)?.idx ?? 0];
    });
    return JSON.stringify({ car, frozen: frozenValues }, null, 2);
  }, [car, frozen, values, cfg]);

  const handleCopy = () => {
    navigator.clipboard.writeText(exportJSON).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  // ═══════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════
  const TABS = [
    { key: "config", label: "Freeze Config" },
    { key: "visualize", label: "Visualize" },
    { key: "export", label: "Export / CLI" },
  ];

  return (
    <div style={{ padding: "0 4px" }}>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: 3, color: C.w, fontFamily: C.hd }}>RESEARCH</div>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>PARAMETER FREEZE / UNFREEZE CONTROLLER</div>
        </div>
        <div style={{ flex: 1 }} />
        {/* Car selector */}
        <div style={{ display: "flex", borderRadius: 20, border: `1px solid ${C.b2}`, overflow: "hidden" }}>
          {Object.entries(CARS).map(([id, c]) => (
            <button key={id} onClick={() => setCar(id)} style={{
              padding: "6px 18px", border: "none", cursor: "pointer",
              fontSize: 9, fontWeight: 700, letterSpacing: 1.5, fontFamily: C.dt,
              background: car === id ? `${C.cy}20` : "transparent",
              color: car === id ? C.cy : C.dm,
              transition: "all 0.2s ease",
            }}>
              {c.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── KPI strip ──────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Car" value={cfg.label} sub={`${cfg.season}`} sentiment="neutral" delay={0} />
        <KPI label="Drivetrain" value={cfg.drivetrain} sub={`${cfg.mass} kg`} sentiment="neutral" delay={1} />
        <KPI label="Free Params" value={nFree} sub={`of 28`} sentiment={nFree > 20 ? "positive" : nFree > 10 ? "amber" : "negative"} delay={2} />
        <KPI label="Frozen" value={nFrozen} sub={nFrozen === 0 ? "all free" : `${nFrozen} locked`} sentiment={nFrozen === 0 ? "positive" : "amber"} delay={3} />
        <KPI label="Eff. Dim" value={`${nFree}D`} sub={nFree <= 15 ? "focused search" : "broad exploration"} sentiment="neutral" delay={4} />
      </div>

      {/* ── Phase presets ──────────────────────────────────────────────── */}
      <Sec title="Design Phase Presets">
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 4 }}>
          {Object.entries(PRESETS).map(([k, p]) => {
            const active = (() => {
              const frozenKeys = Object.keys(frozen).filter(fk => frozen[fk]);
              return frozenKeys.length === p.frozen.length && p.frozen.every(fk => frozen[fk]);
            })();
            return (
              <Pill key={k} active={active} label={p.label} onClick={() => applyPreset(k)} color={C.cy} />
            );
          })}
          <div style={{ flex: 1 }} />
          <button onClick={resetToDefaults} style={{
            background: "transparent", border: `1px solid ${C.b1}`, borderRadius: 20,
            padding: "6px 14px", cursor: "pointer", fontSize: 9, fontWeight: 600,
            color: C.dm, fontFamily: C.dt, letterSpacing: 1,
          }}>
            ↺ Reset Values
          </button>
        </div>
      </Sec>

      {/* ── Sub-tabs ───────────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 4, marginBottom: 14 }}>
        {TABS.map(t => (
          <Pill key={t.key} active={activeTab === t.key} label={t.label}
            onClick={() => setActiveTab(t.key)} color={C.red} />
        ))}
      </div>

      {activeTab === "config" && (
        <div>
          {CATEGORIES.map(cat => {
            const catParams = PARAMS.filter(p => p.cat === cat.key);
            const catFrozenCount = catParams.filter(p => frozen[p.key]).length;
            return (
              <Sec key={cat.key} title={`${cat.icon}  ${cat.label}`}
                right={
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt }}>
                      {catFrozenCount}/{catParams.length} frozen
                    </span>
                    <button onClick={() => freezeCategory(cat.key, true)} style={catBtnStyle(cat.color)}>
                      FREEZE ALL
                    </button>
                    <button onClick={() => freezeCategory(cat.key, false)} style={catBtnStyle(C.gn)}>
                      FREE ALL
                    </button>
                  </div>
                }
              >
                <GC style={{ padding: "6px 10px" }}>
                  {/* Header row */}
                  <div style={headerRowStyle}>
                    <div style={{ width: 36 }}>Status</div>
                    <div style={{ width: 130 }}>Parameter</div>
                    <div style={{ flex: 1 }}>Value</div>
                    <div style={{ width: 70, textAlign: "right" }}>LB</div>
                    <div style={{ width: 70, textAlign: "right" }}>UB</div>
                    <div style={{ width: 75, textAlign: "right" }}>Current</div>
                  </div>

                  {catParams.map(p => {
                    const isFrozen = !!frozen[p.key];
                    const val = values[p.key] ?? cfg.defaults[p.idx];
                    const lb = cfg.lb[p.idx];
                    const ub = cfg.ub[p.idx];
                    const sc = p.scale || 1;
                    const displayVal = (val * sc).toFixed(p.fmt);
                    const range = ub - lb;
                    const pct = range > 0 ? ((val - lb) / range) * 100 : 50;

                    return (
                      <div key={p.key} style={{
                        display: "flex", alignItems: "center", gap: 8,
                        padding: "5px 0", borderBottom: `1px solid ${C.b1}08`,
                        opacity: isFrozen ? 1 : 0.95,
                      }}>
                        {/* Freeze toggle */}
                        <button onClick={() => toggleFreeze(p.key)} style={{
                          width: 32, height: 20, borderRadius: 10, border: "none",
                          cursor: "pointer", position: "relative",
                          background: isFrozen
                            ? `linear-gradient(90deg, ${C.red}60, ${C.red}90)`
                            : `linear-gradient(90deg, ${C.gn}30, ${C.gn}50)`,
                          transition: "all 0.25s ease",
                        }}>
                          <div style={{
                            width: 14, height: 14, borderRadius: 7,
                            background: isFrozen ? C.red : C.gn,
                            position: "absolute", top: 3,
                            left: isFrozen ? 15 : 3,
                            transition: "all 0.25s ease",
                            boxShadow: `0 0 8px ${isFrozen ? C.red : C.gn}40`,
                          }} />
                        </button>

                        {/* Label */}
                        <div style={{ width: 130 }}>
                          <div style={{ fontSize: 9, fontWeight: 600, color: isFrozen ? C.red : C.br, fontFamily: C.dt }}>
                            {p.label}
                          </div>
                          <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>
                            {p.key} · {p.unit}
                          </div>
                        </div>

                        {/* Slider */}
                        <div style={{ flex: 1, position: "relative" }}>
                          <div style={{
                            height: 4, borderRadius: 2, width: "100%",
                            background: `linear-gradient(90deg, ${cat.color}30 ${pct}%, ${C.b1} ${pct}%)`,
                          }} />
                          <input
                            type="range"
                            min={lb} max={ub} step={p.step}
                            value={val}
                            onChange={(e) => setValue(p.key, parseFloat(e.target.value))}
                            disabled={!isFrozen}
                            style={{
                              position: "absolute", top: -6, left: 0,
                              width: "100%", height: 16,
                              appearance: "none", WebkitAppearance: "none",
                              background: "transparent", cursor: isFrozen ? "pointer" : "default",
                              opacity: isFrozen ? 1 : 0.3,
                            }}
                          />
                        </div>

                        {/* LB */}
                        <div style={{ width: 70, textAlign: "right", fontSize: 8, color: C.dm, fontFamily: C.dt }}>
                          {(lb * sc).toFixed(p.fmt)}
                        </div>

                        {/* UB */}
                        <div style={{ width: 70, textAlign: "right", fontSize: 8, color: C.dm, fontFamily: C.dt }}>
                          {(ub * sc).toFixed(p.fmt)}
                        </div>

                        {/* Current value (editable for frozen) */}
                        <div style={{ width: 75, textAlign: "right" }}>
                          {isFrozen ? (
                            <input
                              type="number"
                              value={displayVal}
                              step={p.step * sc}
                              onChange={(e) => setValue(p.key, parseFloat(e.target.value) / sc)}
                              style={{
                                width: "100%", background: `${C.red}10`,
                                border: `1px solid ${C.red}40`, borderRadius: 4,
                                color: C.red, fontSize: 9, fontWeight: 700,
                                fontFamily: C.dt, textAlign: "right",
                                padding: "2px 6px", outline: "none",
                              }}
                            />
                          ) : (
                            <span style={{ fontSize: 9, color: C.gn, fontWeight: 600, fontFamily: C.dt }}>
                              {displayVal}
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </GC>
              </Sec>
            );
          })}
        </div>
      )}
        {activeTab === "visualize" && (
            <ResearchVisualizer
                values={values}
                frozen={frozen}
                car={car}
                carCfg={cfg}
            />
        )}
      {activeTab === "export" && (
        <div>
          <Sec title="Export Configuration">
            <GC style={{ padding: "14px 16px" }}>
              <div style={{ display: "flex", gap: 10, marginBottom: 14, alignItems: "center" }}>
                <button onClick={handleCopy} style={{
                  background: copied ? `${C.gn}20` : `${C.cy}15`,
                  border: `1px solid ${copied ? C.gn : C.cy}`,
                  borderRadius: 20, padding: "8px 20px", cursor: "pointer",
                  fontSize: 10, fontWeight: 700, color: copied ? C.gn : C.cy,
                  fontFamily: C.dt, letterSpacing: 1,
                  transition: "all 0.2s ease",
                }}>
                  {copied ? "✓ COPIED" : "COPY JSON"}
                </button>
                <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt }}>
                  Paste into DesignFreeze() constructor or save as freeze_config.json
                </span>
              </div>
              <pre style={{
                background: C.bg, border: `1px solid ${C.b1}`, borderRadius: 8,
                padding: 16, fontSize: 9, color: C.br, fontFamily: C.dt,
                lineHeight: 1.8, overflowX: "auto", maxHeight: 400,
              }}>
                {exportJSON}
              </pre>
            </GC>
          </Sec>

          <Sec title="CLI Command">
            <GC style={{ padding: "14px 16px" }}>
              <div style={{ fontSize: 8, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, marginBottom: 8 }}>
                RUN THIS IN YOUR TERMINAL
              </div>
              <pre style={{
                background: C.bg, border: `1px solid ${C.b1}`, borderRadius: 8,
                padding: 14, fontSize: 9, color: C.gn, fontFamily: C.dt,
                lineHeight: 1.8, overflowX: "auto",
              }}>
{`python scripts/run_ter27_design_exploration.py \\
    --car ${car} \\${Object.keys(frozen).filter(k => frozen[k]).length > 0
  ? `\n    --freeze ${Object.keys(frozen).filter(k => frozen[k]).map(k =>
      `${k}=${(values[k] ?? cfg.defaults[PARAMS.find(p => p.key === k)?.idx ?? 0])}`
    ).join(" ")} \\` : ""}
    --quick`}
              </pre>
            </GC>
          </Sec>

          <Sec title="Python Integration">
            <GC style={{ padding: "14px 16px" }}>
              <pre style={{
                background: C.bg, border: `1px solid ${C.b1}`, borderRadius: 8,
                padding: 14, fontSize: 9, color: C.am, fontFamily: C.dt,
                lineHeight: 1.8, overflowX: "auto",
              }}>
{`from data.configs.design_freeze import DesignFreeze

# Paste the JSON from above:
freeze = DesignFreeze(${(() => {
  const fv = {};
  Object.keys(frozen).filter(k => frozen[k]).forEach(k => {
    fv[k] = values[k] ?? cfg.defaults[PARAMS.find(p => p.key === k)?.idx ?? 0];
  });
  return JSON.stringify(fv, null, 4).replace(/"/g, "'");
})()})

print(freeze.summary())
# → ${nFrozen} frozen, ${nFree} free, ${28} total`}
              </pre>
            </GC>
          </Sec>

          <Sec title="Frozen Parameter Summary">
            <GC style={{ padding: "8px 12px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "140px 80px 80px 1fr", gap: 0, fontSize: 8, fontFamily: C.dt }}>
                {["Parameter", "Value", "Unit", "Implementation"].map(h => (
                  <div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 1, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>
                ))}
                {PARAMS.filter(p => frozen[p.key]).map(p => {
                  const val = values[p.key] ?? cfg.defaults[p.idx];
                  const sc = p.scale || 1;
                  const impl = IMPL_NOTES[p.key] || "—";
                  return (
                    <React.Fragment key={p.key}>
                      <div style={{ color: C.red, fontWeight: 600, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>
                        🔒 {p.label}
                      </div>
                      <div style={{ color: C.w, fontWeight: 700, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>
                        {(val * sc).toFixed(p.fmt)}
                      </div>
                      <div style={{ color: C.dm, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>
                        {p.unit}
                      </div>
                      <div style={{ color: C.md, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08`, fontSize: 7 }}>
                        {impl}
                      </div>
                    </React.Fragment>
                  );
                })}
                {nFrozen === 0 && (
                  <div style={{ gridColumn: "1 / -1", color: C.gn, padding: "12px 4px", textAlign: "center" }}>
                    All parameters free — full design exploration mode
                  </div>
                )}
              </div>
            </GC>
          </Sec>
        </div>
      )}

      {/* ── Slider CSS (range thumb styling) ─────────────────────────────── */}
      <style>{`
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none; appearance: none;
          width: 12px; height: 12px; border-radius: 50%;
          background: ${C.cy}; border: 2px solid ${C.bg};
          box-shadow: 0 0 6px ${C.cy}40;
          cursor: pointer;
        }
        input[type="range"]::-moz-range-thumb {
          width: 12px; height: 12px; border-radius: 50%;
          background: ${C.cy}; border: 2px solid ${C.bg};
          box-shadow: 0 0 6px ${C.cy}40;
          cursor: pointer;
        }
        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button {
          opacity: 0.5;
        }
      `}</style>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// IMPLEMENTATION NOTES — how to realise each frozen parameter in CAD
// ═══════════════════════════════════════════════════════════════════════════
const IMPL_NOTES = {
  castor_f:    "UCA fore/aft offset in side view",
  anti_squat:  "Rear lower wishbone pickup height → SVSA angle",
  anti_dive_f: "Front wishbone inclination in side view",
  anti_dive_r: "Rear wishbone inclination under braking loads",
  anti_lift:   "Rear side-view geometry for decel/regen",
  bump_steer_f:"Rack height relative to outer tie rod ball joint arc",
  bump_steer_r:"Rear toe link mount height relative to hub arc",
  h_cg:        "Overall packaging: accumulator, motor, driver position",
  camber_f:    "Upper wishbone shim thickness at inboard pickup",
  camber_r:    "Upper wishbone shim thickness at inboard pickup",
  toe_f:       "Tie rod length — adjusted on alignment rig",
  toe_r:       "Rear toe link length — adjusted on alignment rig",
  h_ride_f:    "Spring preload + pushrod length",
  h_ride_r:    "Spring preload + pushrod length",
  brake_bias_f:"Brake balance bar position",
  diff_lock_ratio: "Electronic torque vectoring — no mechanical diff for 4WD",
};

// ═══════════════════════════════════════════════════════════════════════════
// STYLE HELPERS
// ═══════════════════════════════════════════════════════════════════════════
const headerRowStyle = {
  display: "flex", alignItems: "center", gap: 8,
  padding: "4px 0 6px", borderBottom: `1px solid ${C.b1}`,
  fontSize: 7, fontWeight: 700, color: C.dm, fontFamily: C.dt,
  letterSpacing: 1.5, textTransform: "uppercase",
};

const catBtnStyle = (color) => ({
  background: "transparent", border: `1px solid ${color}30`,
  borderRadius: 10, padding: "2px 8px", cursor: "pointer",
  fontSize: 7, fontWeight: 700, color, fontFamily: C.dt,
  letterSpacing: 0.5,
});