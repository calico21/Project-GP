// ═══════════════════════════════════════════════════════════════════════════════
// src/canvas/SensitivityHeatmap.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// ∂Objective / ∂Parameter sensitivity heatmap.
//
// Displays the Jacobian of the surrogate objective function w.r.t. the 28
// setup parameters as a matrix heatmap. Three rows (Grip, Stability, Thermal),
// 28 columns (one per parameter in canonical order).
//
// Color: diverging — positive sensitivity = objective-colored fill,
//        negative = red. Intensity proportional to magnitude.
//
// Interaction:
//   - Hover: highlights row + column crosshair, shows exact value in tooltip
//   - Click: triggers sendPrompt or callback for marginal effect drill-down
//   - hoveredParam: linked to SelectionContext for cross-chart highlighting
//
// Props:
//   sens — Array from gSN4(): { param, unit, dGrip, dStab, dTherm }
//   onCellClick — optional callback(param, objective) for drill-down
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useCallback } from "react";
import { C, GL } from "../theme.js";

const OBJECTIVES = [
  { key: "dGrip",  label: "Grip",      posColor: "0,230,118",  posHex: C.gn },
  { key: "dStab",  label: "Stability",  posColor: "255,171,0",  posHex: C.am },
  { key: "dTherm", label: "Thermal",    posColor: "0,210,255",  posHex: C.cy },
];

const NEG_COLOR = "225,6,0";   // red for negative sensitivity

export default function SensitivityHeatmap({ sens, onCellClick }) {
  const [hover, setHover] = useState(null);   // { pi, oi }

  if (!sens || !sens.length) return null;

  const nParams = sens.length;
  const nObjs = OBJECTIVES.length;

  // Layout
  const cellW = 22;
  const cellH = 28;
  const labelW = 70;      // left label column
  const headerH = 60;     // top header row (rotated param names)
  const svgW = labelW + nParams * cellW + 10;
  const svgH = headerH + nObjs * cellH + 30;

  // Normalization: per-objective max for meaningful color intensity
  const maxPerObj = OBJECTIVES.map(obj =>
    Math.max(...sens.map(s => Math.abs(s[obj.key])), 0.001)
  );

  const handleMouseEnter = useCallback((pi, oi) => setHover({ pi, oi }), []);
  const handleMouseLeave = useCallback(() => setHover(null), []);

  return (
    <div style={{ overflowX: "auto", position: "relative" }}>
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        style={{ display: "block" }}
      >
        {/* ── Objective row labels ─────────────────────────────────── */}
        {OBJECTIVES.map((obj, oi) => (
          <text
            key={obj.key}
            x={labelW - 6}
            y={headerH + oi * cellH + cellH / 2 + 3}
            textAnchor="end"
            fill={hover && hover.oi === oi ? obj.posHex : C.dm}
            fontSize={9}
            fontWeight={hover && hover.oi === oi ? 700 : 500}
            fontFamily={C.dt}
          >
            {obj.label}
          </text>
        ))}

        {/* ── Parameter column headers (rotated) ───────────────────── */}
        {sens.map((s, pi) => (
          <text
            key={`h-${pi}`}
            x={labelW + pi * cellW + cellW / 2}
            y={headerH - 6}
            textAnchor="end"
            fill={hover && hover.pi === pi ? C.cy : C.dm}
            fontSize={6.5}
            fontWeight={hover && hover.pi === pi ? 700 : 400}
            fontFamily={C.dt}
            transform={`rotate(-55, ${labelW + pi * cellW + cellW / 2}, ${headerH - 6})`}
          >
            {s.param}
          </text>
        ))}

        {/* ── Heatmap cells ────────────────────────────────────────── */}
        {sens.map((s, pi) =>
          OBJECTIVES.map((obj, oi) => {
            const val = s[obj.key];
            const norm = val / maxPerObj[oi];
            const absNorm = Math.abs(norm);
            const color = val >= 0 ? obj.posColor : NEG_COLOR;
            const isHov = hover && hover.pi === pi && hover.oi === oi;

            return (
              <rect
                key={`${pi}-${oi}`}
                x={labelW + pi * cellW}
                y={headerH + oi * cellH}
                width={cellW - 1}
                height={cellH - 1}
                rx={3}
                fill={`rgba(${color},${absNorm * 0.75})`}
                stroke={isHov ? C.cy : "rgba(255,255,255,0.02)"}
                strokeWidth={isHov ? 1.5 : 0.3}
                style={{ cursor: onCellClick ? "pointer" : "crosshair" }}
                onMouseEnter={() => handleMouseEnter(pi, oi)}
                onMouseLeave={handleMouseLeave}
                onClick={() => onCellClick?.(s.param, obj.key)}
              >
                <title>
                  {s.param} → {obj.label}: {val.toFixed(3)}
                </title>
              </rect>
            );
          })
        )}

        {/* ── Crosshair highlight ──────────────────────────────────── */}
        {hover && (
          <>
            {/* Vertical column highlight */}
            <rect
              x={labelW + hover.pi * cellW}
              y={headerH}
              width={cellW - 1}
              height={nObjs * cellH}
              fill="none"
              stroke={C.cy}
              strokeWidth={0.5}
              strokeDasharray="2 2"
              pointerEvents="none"
            />
            {/* Horizontal row highlight */}
            <rect
              x={labelW}
              y={headerH + hover.oi * cellH}
              width={nParams * cellW}
              height={cellH - 1}
              fill="none"
              stroke={OBJECTIVES[hover.oi].posHex}
              strokeWidth={0.5}
              strokeDasharray="2 2"
              pointerEvents="none"
            />
          </>
        )}
      </svg>

      {/* ── Hover tooltip ──────────────────────────────────────────── */}
      {hover && (
        <div style={{
          position: "absolute",
          top: headerH + hover.oi * cellH - 36,
          left: Math.min(
            labelW + hover.pi * cellW + cellW + 8,
            svgW - 140,   // prevent overflow
          ),
          ...GL,
          padding: "6px 10px",
          fontSize: 9,
          fontFamily: C.dt,
          color: C.br,
          zIndex: 10,
          pointerEvents: "none",
          whiteSpace: "nowrap",
        }}>
          <span style={{ color: C.cy, fontWeight: 700 }}>
            {sens[hover.pi].param}
          </span>
          <span style={{ color: C.dm }}> → </span>
          <span style={{ color: OBJECTIVES[hover.oi].posHex, fontWeight: 700 }}>
            {OBJECTIVES[hover.oi].label}
          </span>
          <br />
          <span style={{
            color: sens[hover.pi][OBJECTIVES[hover.oi].key] >= 0 ? C.gn : C.red,
            fontWeight: 700, fontSize: 11,
          }}>
            {sens[hover.pi][OBJECTIVES[hover.oi].key] > 0 ? "+" : ""}
            {sens[hover.pi][OBJECTIVES[hover.oi].key].toFixed(3)}
          </span>
          {sens[hover.pi].unit && (
            <span style={{ color: C.dm, marginLeft: 4 }}>
              / {sens[hover.pi].unit}
            </span>
          )}
        </div>
      )}

      {/* ── Legend ──────────────────────────────────────────────────── */}
      <div style={{
        display: "flex", gap: 10, padding: "6px 0",
        justifyContent: "center", flexWrap: "wrap",
      }}>
        {OBJECTIVES.map(obj => (
          <div key={obj.key} style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <div style={{
              width: 10, height: 6, borderRadius: 2,
              background: `rgba(${obj.posColor},0.6)`,
            }} />
            <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>
              +∂{obj.label}
            </span>
          </div>
        ))}
        <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <div style={{
            width: 10, height: 6, borderRadius: 2,
            background: `rgba(${NEG_COLOR},0.6)`,
          }} />
          <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>
            Negative ∂
          </span>
        </div>
      </div>
    </div>
  );
}