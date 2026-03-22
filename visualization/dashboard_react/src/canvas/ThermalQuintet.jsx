// ═══════════════════════════════════════════════════════════════════════════════
// src/canvas/ThermalQuintet.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// 5-Node Jaeger tire thermal visualization.
//
// Renders 4 radial diagrams (FL, FR, RL, RR) arranged in car plan-view.
// Each diagram shows 5 concentric rings mapping to the thermal stack:
//
//   Ring 0 (outermost): T_flash    — Jaeger transient surface flash temperature
//   Ring 1:             T_surface  — tread surface (primary grip driver)
//   Ring 2:             T_bulk     — tread rubber bulk
//   Ring 3:             T_carcass  — structural carcass (hysteresis heating)
//   Ring 4 (center):    T_gas      — inflation gas temperature
//
// Ring width is proportional to thermal mass (gas thin, carcass thick).
// Color is continuous via thermalHex() interpolation.
//
// The flash-to-surface delta (ΔT = T_flash - T_surface) is the primary
// indicator of transient thermal abuse — displayed as a numeric readout
// below the surface temperature at the diagram center.
//
// Props:
//   data   — Array from gThermal5() 
//   step   — current index into the data array (for Live mode animation)
//   size   — optional, px diameter of each radial diagram (default 100)
//
// ═══════════════════════════════════════════════════════════════════════════════

import React from "react";
import { C, thermalHex, thermalColor } from "../theme.js";

// Ring geometry: [radiusFraction, strokeWidthFraction]
// Outermost = flash (thin, hot spike), innermost = gas (thin, slow)
// Carcass is thickest — largest thermal mass in the real tire.
const RING_CONFIG = [
  { name: "Flash",   rFrac: 0.92, wFrac: 0.13 },
  { name: "Surface", rFrac: 0.76, wFrac: 0.14 },
  { name: "Bulk",    rFrac: 0.60, wFrac: 0.14 },
  { name: "Carcass", rFrac: 0.44, wFrac: 0.16 },
  { name: "Gas",     rFrac: 0.26, wFrac: 0.10 },
];

// ─── Single-corner radial diagram ────────────────────────────────────────────

function ThermalRadial({ temps, label, size = 100 }) {
  if (!temps || temps.length < 5) return null;

  const cx = size / 2;
  const cy = size / 2;
  const surfTemp = temps[1];
  const flashDelta = temps[0] - temps[1];

  return (
    <svg width={size} height={size + 22} viewBox={`0 0 ${size} ${size + 22}`}>
      {/* Concentric thermal rings */}
      {RING_CONFIG.map((cfg, i) => {
        const r = cfg.rFrac * cx;
        const w = cfg.wFrac * cx;
        return (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={thermalHex(temps[i])}
            strokeWidth={w}
            opacity={0.88 + i * 0.025}
          >
            <title>{cfg.name}: {temps[i].toFixed(1)}°C</title>
          </circle>
        );
      })}

      {/* Center readout: surface temp (large) + flash delta (small) */}
      <text
        x={cx}
        y={cy - 5}
        textAnchor="middle"
        dominantBaseline="central"
        fill={C.w}
        fontSize={Math.max(10, size * 0.13)}
        fontWeight={700}
        fontFamily={C.dt}
      >
        {surfTemp.toFixed(0)}°
      </text>
      <text
        x={cx}
        y={cy + size * 0.1}
        textAnchor="middle"
        dominantBaseline="central"
        fill={flashDelta > 15 ? C.th_crit : flashDelta > 8 ? C.th_warm : C.th_nom}
        fontSize={Math.max(7, size * 0.08)}
        fontWeight={600}
        fontFamily={C.dt}
      >
        Δ{flashDelta.toFixed(0)}°
      </text>

      {/* Corner label */}
      <text
        x={cx}
        y={size + 14}
        textAnchor="middle"
        fill={C.dm}
        fontSize={Math.max(7, size * 0.08)}
        fontWeight={700}
        fontFamily={C.dt}
        letterSpacing={1.5}
      >
        {label}
      </text>
    </svg>
  );
}

// ─── Flash-to-surface delta bar ──────────────────────────────────────────────

function FlashDeltaBar({ delta, maxDelta = 30, width = 80 }) {
  const norm = Math.min(1, Math.abs(delta) / maxDelta);
  const color = delta > 15 ? C.th_crit : delta > 8 ? C.th_warm : C.th_nom;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <div style={{
        width, height: 4, background: C.b1, borderRadius: 2, overflow: "hidden",
      }}>
        <div style={{
          width: `${norm * 100}%`, height: "100%",
          background: color, borderRadius: 2,
          transition: "width 0.15s ease-out",
        }} />
      </div>
      <span style={{
        fontSize: 7, fontFamily: C.dt, color, fontWeight: 600, minWidth: 24,
      }}>
        {delta.toFixed(0)}°
      </span>
    </div>
  );
}

// ─── 4-corner plan-view layout ───────────────────────────────────────────────

export default function ThermalQuintet({ data, step, size = 100 }) {
  if (!data || !data.length) return null;

  const frame = data[Math.min(step || 0, data.length - 1)];
  if (!frame) return null;

  const corners = [
    { key: "fl", label: "FL", temps: frame.fl },
    { key: "fr", label: "FR", temps: frame.fr },
    { key: "rl", label: "RL", temps: frame.rl },
    { key: "rr", label: "RR", temps: frame.rr },
  ];

  return (
    <div>
      {/* 2×2 plan-view grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 4,
        justifyItems: "center",
      }}>
        {corners.map(c => (
          <ThermalRadial
            key={c.key}
            temps={c.temps}
            label={c.label}
            size={size}
          />
        ))}
      </div>

      {/* Flash-to-surface delta bars per corner */}
      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr",
        gap: 2, marginTop: 6, padding: "0 8px",
      }}>
        {corners.map(c => (
          <div key={`d-${c.key}`} style={{
            display: "flex", alignItems: "center", gap: 4,
          }}>
            <span style={{
              fontSize: 7, fontFamily: C.dt, color: C.dm,
              width: 16, textAlign: "right",
            }}>
              {c.label}
            </span>
            <FlashDeltaBar delta={c.temps[0] - c.temps[1]} width={60} />
          </div>
        ))}
      </div>

      {/* Legend */}
      <div style={{
        display: "flex", gap: 6, justifyContent: "center",
        marginTop: 6, flexWrap: "wrap",
      }}>
        {RING_CONFIG.map((cfg, i) => (
          <div key={cfg.name} style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <div style={{
              width: 8, height: 3, borderRadius: 1,
              background: thermalHex(corners[0].temps[i]),
            }} />
            <span style={{ fontSize: 6, color: C.dm, fontFamily: C.dt }}>
              {cfg.name}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Named export for individual radial (used by TirePhysicsModule)
export { ThermalRadial, FlashDeltaBar };