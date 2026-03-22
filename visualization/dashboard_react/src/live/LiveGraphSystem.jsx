// ═══════════════════════════════════════════════════════════════════════════════
// src/live/LiveGraphSystem.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// Configurable time-series graph system for Live telemetry.
//
// Architecture:
//   - CHANNELS: 40+ telemetry channels organized into 8 groups
//   - GraphSlot: individual graph panel with channel picker + remove button
//   - TimeSeriesCanvas: hardware-accelerated Canvas renderer for real-time data
//   - LiveGraphPanel: orchestrator that manages N graph slots
//
// Each GraphSlot renders selected channels from the shared history ring buffer.
// Channel selection is per-graph — different graphs can show different channels.
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useRef, useCallback } from "react";
import { C, GL, GS } from "../theme.js";

// ═════════════════════════════════════════════════════════════════════════════
// CHANNEL REGISTRY — 40+ channels across 8 groups
// ═════════════════════════════════════════════════════════════════════════════

export const GROUPS = [
  "Kinematics", "Thermal", "Slip", "Forces",
  "MPC", "Energy", "Driver", "Twin",
];

export const CHANNELS = [
  // ── Kinematics ─────────────────────────────────────────────────────
  { key: "speed",      label: "Speed",         unit: "m/s",   group: "Kinematics", color: "#00e676", range: [0, 30] },
  { key: "latG",       label: "Lat G",         unit: "G",     group: "Kinematics", color: "#00d2ff", range: [-2, 2] },
  { key: "lonG",       label: "Lon G",         unit: "G",     group: "Kinematics", color: "#ffab00", range: [-2, 2] },
  { key: "combinedG",  label: "Combined G",    unit: "G",     group: "Kinematics", color: "#b388ff", range: [0, 2] },
  { key: "yawRate",    label: "Yaw Rate ψ̇",    unit: "rad/s", group: "Kinematics", color: "#ff80ab", range: [-3, 3] },
  { key: "sideslip",   label: "Sideslip β",    unit: "rad",   group: "Kinematics", color: "#ea80fc", range: [-0.1, 0.1] },
  { key: "steer",      label: "Steer δ",       unit: "°",     group: "Kinematics", color: "#80d8ff", range: [-20, 20] },
  { key: "curvature",  label: "Curvature κ",   unit: "1/m",   group: "Kinematics", color: "#a7ffeb", range: [-0.15, 0.15] },
  { key: "rollRate",   label: "Roll Rate φ̇",   unit: "rad/s", group: "Kinematics", color: "#f48fb1", range: [-2, 2] },
  { key: "pitchRate",  label: "Pitch Rate θ̇",  unit: "rad/s", group: "Kinematics", color: "#ce93d8", range: [-1, 1] },

  // ── Thermal ────────────────────────────────────────────────────────
  { key: "tfl_surf",   label: "T_surf FL",     unit: "°C",    group: "Thermal", color: "#ff5722", range: [20, 160] },
  { key: "tfr_surf",   label: "T_surf FR",     unit: "°C",    group: "Thermal", color: "#ff7043", range: [20, 160] },
  { key: "trl_surf",   label: "T_surf RL",     unit: "°C",    group: "Thermal", color: "#ff8a65", range: [20, 160] },
  { key: "trr_surf",   label: "T_surf RR",     unit: "°C",    group: "Thermal", color: "#ffab91", range: [20, 160] },
  { key: "tfl_flash",  label: "T_flash FL",    unit: "°C",    group: "Thermal", color: "#f44336", range: [20, 170] },
  { key: "tfr_flash",  label: "T_flash FR",    unit: "°C",    group: "Thermal", color: "#e53935", range: [20, 170] },
  { key: "tfl_core",   label: "T_core FL",     unit: "°C",    group: "Thermal", color: "#ff9800", range: [20, 130] },
  { key: "tfr_core",   label: "T_core FR",     unit: "°C",    group: "Thermal", color: "#ffa726", range: [20, 130] },
  { key: "flashDelta", label: "Flash-Surf Δ",  unit: "°C",    group: "Thermal", color: "#d50000", range: [0, 40] },

  // ── Slip ───────────────────────────────────────────────────────────
  { key: "slipAngleF", label: "Slip α Front",  unit: "°",     group: "Slip", color: "#00bcd4", range: [-15, 15] },
  { key: "slipAngleR", label: "Slip α Rear",   unit: "°",     group: "Slip", color: "#26c6da", range: [-15, 15] },
  { key: "slipRatioFL",label: "Slip κ FL",     unit: "",      group: "Slip", color: "#4dd0e1", range: [-0.3, 0.3] },
  { key: "slipRatioFR",label: "Slip κ FR",     unit: "",      group: "Slip", color: "#80deea", range: [-0.3, 0.3] },
  { key: "gripUtil",   label: "Grip Util.",    unit: "%",     group: "Slip", color: "#ffab00", range: [0, 100] },

  // ── Forces ─────────────────────────────────────────────────────────
  { key: "fzFL",       label: "Fz FL",         unit: "N",     group: "Forces", color: "#42a5f5", range: [0, 2000] },
  { key: "fzFR",       label: "Fz FR",         unit: "N",     group: "Forces", color: "#66bb6a", range: [0, 2000] },
  { key: "fzRL",       label: "Fz RL",         unit: "N",     group: "Forces", color: "#ffa726", range: [0, 2000] },
  { key: "fzRR",       label: "Fz RR",         unit: "N",     group: "Forces", color: "#ab47bc", range: [0, 2000] },
  { key: "fxTotal",    label: "Fx Total",      unit: "N",     group: "Forces", color: "#ef5350", range: [-5000, 5000] },
  { key: "fyTotal",    label: "Fy Total",      unit: "N",     group: "Forces", color: "#29b6f6", range: [-5000, 5000] },

  // ── MPC ────────────────────────────────────────────────────────────
  { key: "wmpcSolveMs",label: "WMPC Solve",    unit: "ms",    group: "MPC", color: "#00e676", range: [0, 20] },
  { key: "alIters",    label: "AL Iterations",  unit: "",      group: "MPC", color: "#ffab00", range: [0, 15] },
  { key: "alSlackGrip",label: "AL Slack (μ)",  unit: "",      group: "MPC", color: "#e10600", range: [0, 0.5] },
  { key: "horizonErr", label: "Horizon Error",  unit: "m",     group: "MPC", color: "#b388ff", range: [0, 1] },

  // ── Energy ─────────────────────────────────────────────────────────
  { key: "hamiltonianJ",label:"H(q,p)",        unit: "kJ",    group: "Energy", color: "#ab47bc", range: [3, 6] },
  { key: "dHdt",       label: "dH/dt",         unit: "W",     group: "Energy", color: "#00d2ff", range: [-50, 50] },
  { key: "rDiss",      label: "R_diss",        unit: "W",     group: "Energy", color: "#ef5350", range: [0, 30] },
  { key: "hNetResid",  label: "H_net Resid.",  unit: "J",     group: "Energy", color: "#ce93d8", range: [0, 50] },

  // ── Driver ─────────────────────────────────────────────────────────
  { key: "throttle",   label: "Throttle",      unit: "%",     group: "Driver", color: "#00e676", range: [0, 100] },
  { key: "brake",      label: "Brake",         unit: "%",     group: "Driver", color: "#e10600", range: [0, 100] },
  { key: "steerInput", label: "Steer Input",   unit: "°",     group: "Driver", color: "#00d2ff", range: [-180, 180] },

  // ── Twin Health ────────────────────────────────────────────────────
  { key: "ekfInnov",   label: "EKF Innov.",    unit: "",      group: "Twin", color: "#00d2ff", range: [-0.05, 0.05] },
  { key: "ekfInnovWz", label: "EKF ω_z Innov.",unit: "",      group: "Twin", color: "#ffab00", range: [-0.05, 0.05] },
  { key: "modelConf",  label: "Model Conf.",   unit: "%",     group: "Twin", color: "#00e676", range: [0, 100] },
];

const CHANNEL_MAP = Object.fromEntries(CHANNELS.map(c => [c.key, c]));

// ═════════════════════════════════════════════════════════════════════════════
// CANVAS TIME-SERIES RENDERER
// ═════════════════════════════════════════════════════════════════════════════

function TimeSeriesCanvas({ history, channels, height = 140 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !history.length || !channels.length) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const pad = { t: 10, b: 20, l: 40, r: 10 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    ctx.clearRect(0, 0, W, H);

    // Background grid
    ctx.strokeStyle = "rgba(25,35,55,0.4)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
    }

    // Zero line
    ctx.strokeStyle = "rgba(62,74,100,0.5)";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([3, 3]);
    const zeroY = pad.t + plotH / 2;
    ctx.beginPath();
    ctx.moveTo(pad.l, zeroY);
    ctx.lineTo(pad.l + plotW, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    const n = history.length;

    channels.forEach(chKey => {
      const ch = CHANNEL_MAP[chKey];
      if (!ch) return;

      const [rMin, rMax] = ch.range;
      const rRange = rMax - rMin || 1;

      ctx.strokeStyle = ch.color;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();

      let started = false;
      for (let i = 0; i < n; i++) {
        const val = history[i]?.[chKey];
        if (val === undefined || val === null) continue;

        const x = pad.l + (i / Math.max(1, n - 1)) * plotW;
        const norm = (val - rMin) / rRange;
        const y = pad.t + plotH * (1 - Math.max(0, Math.min(1, norm)));

        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Current value label (right edge)
      const lastVal = history[n - 1]?.[chKey];
      if (lastVal !== undefined) {
        const norm = (lastVal - rMin) / rRange;
        const y = pad.t + plotH * (1 - Math.max(0, Math.min(1, norm)));
        ctx.fillStyle = ch.color;
        ctx.font = `bold ${Math.round(W * 0.018)}px ${C.dt}`;
        ctx.textAlign = "left";
        ctx.fillText(
          `${Number(lastVal).toFixed(ch.unit === "°C" || ch.unit === "N" ? 0 : 2)}${ch.unit}`,
          pad.l + plotW + 2,
          y + 3,
        );
      }
    });

    // Y-axis tick labels (auto-ranged to first selected channel)
    if (channels.length > 0) {
      const ch = CHANNEL_MAP[channels[0]];
      if (ch) {
        ctx.fillStyle = C.dm;
        ctx.font = `${Math.round(W * 0.016)}px ${C.dt}`;
        ctx.textAlign = "right";
        const [rMin, rMax] = ch.range;
        for (let i = 0; i <= 4; i++) {
          const frac = 1 - i / 4;
          const val = rMin + frac * (rMax - rMin);
          const y = pad.t + (plotH / 4) * i;
          ctx.fillText(val.toFixed(1), pad.l - 4, y + 3);
        }
      }
    }

    // Time axis (last N frames)
    ctx.fillStyle = C.dm;
    ctx.font = `${Math.round(W * 0.014)}px ${C.dt}`;
    ctx.textAlign = "center";
    const tStart = history[0]?.t || 0;
    const tEnd = history[n - 1]?.t || 0;
    for (let i = 0; i <= 3; i++) {
      const frac = i / 3;
      const x = pad.l + frac * plotW;
      const t = (+tStart + frac * (+tEnd - +tStart)).toFixed(1);
      ctx.fillText(`${t}s`, x, H - 4);
    }

  }, [history, channels]);

  return (
    <canvas
      ref={canvasRef}
      width={700}
      height={height * 2}
      style={{
        width: "100%", height,
        borderRadius: 6, display: "block",
      }}
    />
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// GRAPH SLOT — individual configurable graph panel
// ═════════════════════════════════════════════════════════════════════════════

function GraphSlot({ id, history, onRemove }) {
  const [selected, setSelected] = useState([]);
  const [showPicker, setShowPicker] = useState(true);
  const [filterGroup, setFilterGroup] = useState(null);

  const toggle = useCallback((key) => {
    setSelected(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  }, []);

  const filtered = filterGroup
    ? CHANNELS.filter(c => c.group === filterGroup)
    : CHANNELS;

  return (
    <div style={{
      ...GL, padding: "8px 10px", marginBottom: 6,
      borderLeft: `2px solid ${selected.length > 0 ? CHANNEL_MAP[selected[0]]?.color || C.cy : C.b2}`,
    }}>
      {/* Header */}
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: 4,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            fontSize: 8, fontWeight: 700, color: C.dm,
            fontFamily: C.dt, letterSpacing: 1.5,
          }}>
            GRAPH {id}
          </span>
          <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>
            {selected.length} ch
          </span>
          <button
            onClick={() => setShowPicker(!showPicker)}
            style={{
              background: showPicker ? `${C.cy}15` : "none",
              border: `1px solid ${showPicker ? C.cy : C.b1}`,
              borderRadius: 8, padding: "1px 8px",
              fontSize: 7, color: showPicker ? C.cy : C.dm,
              fontFamily: C.dt, cursor: "pointer",
            }}
          >
            {showPicker ? "▼ Channels" : "▶ Channels"}
          </button>
        </div>
        <button
          onClick={onRemove}
          style={{
            background: `${C.red}10`, border: `1px solid ${C.red}30`,
            borderRadius: 4, padding: "1px 8px",
            fontSize: 7, color: C.red,
            fontFamily: C.dt, cursor: "pointer",
          }}
        >
          ✕ Remove
        </button>
      </div>

      {/* Channel picker */}
      {showPicker && (
        <div style={{ marginBottom: 6 }}>
          {/* Group filter tabs */}
          <div style={{
            display: "flex", gap: 3, marginBottom: 4, flexWrap: "wrap",
          }}>
            <button
              onClick={() => setFilterGroup(null)}
              style={{
                background: !filterGroup ? `${C.cy}15` : "none",
                border: `1px solid ${!filterGroup ? C.cy : C.b1}`,
                borderRadius: 10, padding: "2px 8px",
                fontSize: 7, color: !filterGroup ? C.cy : C.dm,
                fontFamily: C.dt, cursor: "pointer",
              }}
            >
              ALL
            </button>
            {GROUPS.map(g => (
              <button
                key={g}
                onClick={() => setFilterGroup(g)}
                style={{
                  background: filterGroup === g ? `${C.cy}15` : "none",
                  border: `1px solid ${filterGroup === g ? C.cy : C.b1}`,
                  borderRadius: 10, padding: "2px 8px",
                  fontSize: 7, color: filterGroup === g ? C.cy : C.dm,
                  fontFamily: C.dt, cursor: "pointer",
                }}
              >
                {g}
              </button>
            ))}
          </div>

          {/* Channel badges */}
          <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
            {filtered.map(ch => {
              const isOn = selected.includes(ch.key);
              return (
                <button
                  key={ch.key}
                  onClick={() => toggle(ch.key)}
                  style={{
                    background: isOn ? `${ch.color}18` : "none",
                    border: `1px solid ${isOn ? ch.color : C.b1}`,
                    borderRadius: 10, padding: "2px 8px",
                    fontSize: 7,
                    color: isOn ? ch.color : C.dm,
                    fontFamily: C.dt, cursor: "pointer",
                    transition: "all 0.15s",
                  }}
                >
                  {ch.label}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Time-series canvas or empty state */}
      {selected.length > 0 ? (
        <TimeSeriesCanvas
          history={history}
          channels={selected}
          height={140}
        />
      ) : (
        <div style={{
          height: 50, display: "flex", alignItems: "center",
          justifyContent: "center",
          color: C.dm, fontSize: 9, fontFamily: C.dt,
        }}>
          Select channels above to plot
        </div>
      )}

      {/* Active channel legend */}
      {selected.length > 0 && (
        <div style={{
          display: "flex", gap: 6, padding: "4px 0",
          flexWrap: "wrap",
        }}>
          {selected.map(key => {
            const ch = CHANNEL_MAP[key];
            if (!ch) return null;
            return (
              <div key={key} style={{
                display: "flex", alignItems: "center", gap: 3,
              }}>
                <div style={{
                  width: 8, height: 2, background: ch.color, borderRadius: 1,
                }} />
                <span style={{
                  fontSize: 6, color: ch.color, fontFamily: C.dt,
                }}>
                  {ch.label}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// LIVE GRAPH PANEL — manages N configurable graph slots
// ═════════════════════════════════════════════════════════════════════════════

export default function LiveGraphPanel({ history }) {
  const [graphs, setGraphs] = useState([1]);
  const nextId = useRef(2);

  const addGraph = useCallback(() => {
    if (graphs.length >= 6) return;   // cap at 6 simultaneous graphs
    setGraphs(prev => [...prev, nextId.current++]);
  }, [graphs.length]);

  const removeGraph = useCallback((id) => {
    setGraphs(prev => prev.filter(g => g !== id));
  }, []);

  const histArr = history.current || [];

  return (
    <div>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8, marginBottom: 6,
      }}>
        <span style={{
          fontSize: 8, fontWeight: 700, color: C.dm,
          fontFamily: C.dt, letterSpacing: 1.5, textTransform: "uppercase",
        }}>
          TIME-SERIES GRAPHS
        </span>
        <button
          onClick={addGraph}
          style={{
            background: `${C.gn}15`, border: `1px solid ${C.gn}30`,
            borderRadius: 4, padding: "2px 10px",
            fontSize: 9, color: C.gn,
            fontFamily: C.dt, cursor: "pointer",
          }}
        >
          + Add Graph
        </button>
        <span style={{
          fontSize: 8, color: C.dm, fontFamily: C.dt,
        }}>
          {graphs.length} active · {histArr.length} frames (~{(histArr.length * 0.05).toFixed(0)}s buffer)
        </span>
      </div>

      {/* Graph slots */}
      {graphs.map(id => (
        <GraphSlot
          key={id}
          id={id}
          history={histArr}
          onRemove={() => removeGraph(id)}
        />
      ))}
    </div>
  );
}

export { CHANNELS, CHANNEL_MAP, GROUPS, TimeSeriesCanvas, GraphSlot };