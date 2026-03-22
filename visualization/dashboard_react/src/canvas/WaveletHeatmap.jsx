// ═══════════════════════════════════════════════════════════════════════════════
// src/canvas/WaveletHeatmap.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// Renders the Db4 3-level discrete wavelet transform coefficients for the
// three MPC control channels (steer, throttle, brake) as a heatmap.
//
// Layout:
//   Three horizontal columns (one per control channel)
//   Four vertical rows (cA3 approximation at top, cD1 detail at bottom)
//   Each cell = one wavelet coefficient
//   Color: diverging blue (negative) → transparent (zero) → red (positive)
//
// Physical interpretation:
//   - Energy concentrated in cA3 → solver doing path planning (low-frequency)
//   - Energy in cD1 → high-frequency corrections (chattering risk)
//   - Throttle cD2 active → aggressive traction modulation
//
// Props:
//   data   — Array from gWaveletCoeffs(): { ctl, level, idx, mag }
//   width  — CSS px (default 380)
//   height — CSS px (default 180)
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useRef, useEffect } from "react";
import { C } from "../theme.js";

const CONTROLS = ["steer", "throttle", "brake"];
const CONTROL_COLORS = [C.cy, C.gn, C.am];
const LEVELS = ["cA3", "cD3", "cD2", "cD1"];

export default function WaveletHeatmap({ data, width = 380, height = 180 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !data.length) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const colW = W / 3;                // width per control column
    const headerH = 18;                // space for control labels
    const levelLabelW = 28;            // space for level labels on right
    const plotH = H - headerH;
    const rowH = plotH / LEVELS.length;

    // ── Per-control column ───────────────────────────────────────────
    CONTROLS.forEach((ctl, ci) => {
      const sub = data.filter(d => d.ctl === ctl);

      // Control label header
      ctx.fillStyle = CONTROL_COLORS[ci];
      ctx.font = `bold ${Math.round(W * 0.024)}px ${C.dt}`;
      ctx.fillText(ctl.toUpperCase(), ci * colW + 6, headerH - 4);

      // ── Per-level row ──────────────────────────────────────────────
      LEVELS.forEach((level, li) => {
        const lvData = sub.filter(d => d.level === level);
        if (!lvData.length) return;

        const maxMag = Math.max(...lvData.map(d => Math.abs(d.mag)), 0.001);
        const cellW = (colW - (ci === 2 ? levelLabelW : 0)) / lvData.length;
        const y0 = headerH + li * rowH;

        // Background stripe for alternating levels
        if (li % 2 === 0) {
          ctx.fillStyle = "rgba(255,255,255,0.015)";
          ctx.fillRect(ci * colW, y0, colW, rowH);
        }

        // ── Coefficient cells ────────────────────────────────────────
        lvData.forEach((d, di) => {
          const norm = d.mag / maxMag;   // -1..+1
          const abs = Math.abs(norm);

          if (norm > 0) {
            // Positive → cyan/blue
            ctx.fillStyle = `rgba(0,210,255,${abs * 0.85})`;
          } else {
            // Negative → red
            ctx.fillStyle = `rgba(225,6,0,${abs * 0.85})`;
          }

          ctx.fillRect(
            ci * colW + di * cellW,
            y0 + 1,
            cellW + 0.5,   // +0.5 to avoid sub-pixel gaps
            rowH - 2,
          );
        });
      });

      // Column separator
      if (ci > 0) {
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(ci * colW, headerH);
        ctx.lineTo(ci * colW, H);
        ctx.stroke();
      }
    });

    // ── Level labels (right edge of last column) ─────────────────────
    ctx.fillStyle = C.dm;
    ctx.font = `${Math.round(W * 0.02)}px ${C.dt}`;
    LEVELS.forEach((level, li) => {
      const y = headerH + li * rowH + rowH / 2 + 3;
      ctx.fillText(level, W - levelLabelW + 2, y);
    });

    // ── Horizontal level separators ──────────────────────────────────
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    for (let li = 1; li < LEVELS.length; li++) {
      const y = headerH + li * rowH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

  }, [data, width, height]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={width * 2}
        height={height * 2}
        style={{
          width, height, borderRadius: 6,
          background: "rgba(5,7,11,0.5)",
          display: "block",
        }}
      />
      <div style={{
        fontSize: 8, color: C.dm, fontFamily: C.dt,
        padding: "4px 0", textAlign: "center",
      }}>
        Db4 3-level · Blue = positive coeff · Red = negative · Rows = wavelet level
      </div>
    </div>
  );
}