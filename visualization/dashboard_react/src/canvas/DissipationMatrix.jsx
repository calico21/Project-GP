// ═══════════════════════════════════════════════════════════════════════════════
// src/canvas/DissipationMatrix.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// SVG heatmap of the 14×14 symmetric PSD dissipation matrix from R_net.
//
// Structure: R = LLᵀ + diag(softplus(d))
//   - Diagonal entries: direct damping per DOF (always positive)
//   - Off-diagonal: cross-coupling between DOFs (can be negative)
//   - Symmetry enforced by construction (Cholesky factor)
//
// Visual encoding:
//   - Cell color: blue = positive coupling, red = negative, intensity = |R_ij|
//   - Diagonal cells: slightly thicker border (these are the "damper" terms)
//   - Hover: tooltip with DOF pair and exact value
//   - Row/column labels: standard vehicle dynamics DOF notation
//
// Props:
//   matrix   — number[][] (14×14) from gRMatrix()
//   labels   — string[] (14) from R_DOF_LABELS export
//   cellSize — px per cell (default 24)
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState } from "react";
import { C, GL } from "../theme.js";

const DEFAULT_LABELS = [
  "x", "y", "z", "φ", "θ", "ψ",
  "z_fl", "z_fr", "z_rl", "z_rr",
  "φ_wfl", "φ_wfr", "φ_wrl", "φ_wrr",
];

export default function DissipationMatrix({
  matrix,
  labels = DEFAULT_LABELS,
  cellSize = 24,
}) {
  const [hover, setHover] = useState(null);   // { i, j, val }

  if (!matrix || !matrix.length) return null;

  const n = matrix.length;
  const pad = 60;   // space for labels
  const svgW = n * cellSize + pad + 12;
  const svgH = n * cellSize + pad + 12;

  // Find absolute max for normalization
  const maxAbs = Math.max(
    ...matrix.flatMap(row => row.map(v => Math.abs(v))),
    0.01,
  );

  function cellColor(val) {
    const norm = Math.abs(val) / maxAbs;
    if (val > 0) return `rgba(0,210,255,${norm * 0.8})`;     // cyan for positive
    if (val < 0) return `rgba(225,6,0,${norm * 0.8})`;       // red for negative
    return "rgba(255,255,255,0.02)";
  }

  return (
    <div style={{ position: "relative" }}>
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        style={{ display: "block" }}
      >
        {/* Column headers */}
        {labels.map((l, j) => (
          <text
            key={`col-${j}`}
            x={pad + j * cellSize + cellSize / 2}
            y={pad - 6}
            textAnchor="middle"
            fill={hover && hover.j === j ? C.cy : C.dm}
            fontSize={7}
            fontFamily={C.dt}
            fontWeight={hover && hover.j === j ? 700 : 400}
          >
            {l}
          </text>
        ))}

        {/* Row headers */}
        {labels.map((l, i) => (
          <text
            key={`row-${i}`}
            x={pad - 5}
            y={pad + i * cellSize + cellSize / 2 + 3}
            textAnchor="end"
            fill={hover && hover.i === i ? C.cy : C.dm}
            fontSize={7}
            fontFamily={C.dt}
            fontWeight={hover && hover.i === i ? 700 : 400}
          >
            {l}
          </text>
        ))}

        {/* Matrix cells */}
        {matrix.map((row, i) =>
          row.map((val, j) => {
            const isDiag = i === j;
            const isHov = hover && hover.i === i && hover.j === j;

            return (
              <rect
                key={`${i}-${j}`}
                x={pad + j * cellSize}
                y={pad + i * cellSize}
                width={cellSize - 1}
                height={cellSize - 1}
                rx={2}
                fill={cellColor(val)}
                stroke={
                  isHov ? C.cy
                    : isDiag ? "rgba(255,255,255,0.12)"
                    : "rgba(255,255,255,0.03)"
                }
                strokeWidth={isHov ? 1.5 : isDiag ? 0.8 : 0.3}
                style={{ cursor: "crosshair" }}
                onMouseEnter={() => setHover({ i, j, val })}
                onMouseLeave={() => setHover(null)}
              />
            );
          })
        )}

        {/* Diagonal highlight label */}
        {matrix.map((row, i) => {
          if (i !== Math.floor(n / 2)) return null;  // label only at midpoint
          return (
            <text
              key="diag-label"
              x={pad + i * cellSize + cellSize / 2}
              y={pad + i * cellSize + cellSize / 2 + 3}
              textAnchor="middle"
              fill="rgba(255,255,255,0.2)"
              fontSize={6}
              fontFamily={C.dt}
            >
              diag
            </text>
          );
        })}
      </svg>

      {/* Hover tooltip */}
      {hover && (
        <div style={{
          position: "absolute",
          top: pad + hover.i * cellSize - 30,
          left: pad + hover.j * cellSize + cellSize + 8,
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
            {labels[hover.i]}×{labels[hover.j]}
          </span>
          {" = "}
          <span style={{
            color: hover.val > 0 ? C.cy : hover.val < 0 ? C.red : C.dm,
            fontWeight: 700,
          }}>
            {hover.val.toFixed(2)}
          </span>
          <span style={{ color: C.dm, marginLeft: 4 }}>
            {hover.i === hover.j ? "Nm·s/rad" : "coupling"}
          </span>
        </div>
      )}

      {/* Legend */}
      <div style={{
        display: "flex", gap: 10, padding: "6px 0",
        justifyContent: "center",
      }}>
        {[
          ["Positive (damping)", "rgba(0,210,255,0.6)"],
          ["Negative (coupling)", "rgba(225,6,0,0.6)"],
          ["Diagonal = direct", "rgba(255,255,255,0.15)"],
        ].map(([l, c]) => (
          <div key={l} style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <div style={{ width: 10, height: 6, background: c, borderRadius: 2 }} />
            <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}