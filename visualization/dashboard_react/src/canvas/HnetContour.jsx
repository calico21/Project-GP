// ═══════════════════════════════════════════════════════════════════════════════
// src/canvas/HnetContour.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// 2D contour plot of the H_net energy landscape evaluated over a grid of
// front × rear suspension deflections with all other DOFs at equilibrium.
//
// Rendering:
//   - Filled cells: color mapped from energy magnitude (cool blue → warm amber)
//   - Contour lines: iso-energy lines at 10 evenly-spaced thresholds
//   - Passivity violation overlay: red tint where H_net > threshold (phantom energy)
//   - Crosshair: equilibrium point (12.8mm front, 14.2mm rear)
//
// Props:
//   landscape — Array from gHnetLandscape(): { i, j, qf, qr, H }
//   width     — CSS px (default 380)
//   height    — CSS px (default 300)
//   eqFront   — equilibrium front deflection in mm (default 0, center of sweep)
//   eqRear    — equilibrium rear deflection in mm (default 0)
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useRef, useEffect } from "react";
import { C } from "../theme.js";

// ─── Energy value → RGB color ────────────────────────────────────────────────

function energyColor(norm) {
  // norm: 0 (min) → 1 (max)
  // Cool blue → teal → green → amber → warm orange
  const r = Math.round(30 + norm * 195);
  const g = Math.round(60 + (1 - norm) * 140 + norm * 40);
  const b = Math.round(180 * (1 - norm) + 30);
  return { r, g, b };
}

export default function HnetContour({
  landscape, width = 380, height = 300, eqFront = 0, eqRear = 0,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !landscape || !landscape.length) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Compute grid resolution from data
    const res = Math.round(Math.sqrt(landscape.length));
    if (res < 2) return;

    // Padding for axis labels
    const padL = 40, padB = 30, padT = 10, padR = 10;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const cellW = plotW / res;
    const cellH = plotH / res;

    // Energy range
    const allH = landscape.map(d => d.H);
    const minH = Math.min(...allH);
    const maxH = Math.max(...allH);
    const range = maxH - minH || 1;

    // Passivity violation threshold (top 15% of energy)
    const passivityThresh = minH + range * 0.85;

    // ── Filled cells ─────────────────────────────────────────────────
    for (const d of landscape) {
      const norm = (d.H - minH) / range;
      const { r, g, b } = energyColor(norm);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(
        padL + d.i * cellW,
        padT + (res - 1 - d.j) * cellH,   // flip j for screen Y
        cellW + 0.5,
        cellH + 0.5,
      );
    }

    // ── Contour lines (10 iso-levels) ────────────────────────────────
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 0.5;
    const nLevels = 10;
    for (let level = 1; level < nLevels; level++) {
      const thresh = minH + level * range / nLevels;
      const tolerance = range / (nLevels * 5);

      for (const d of landscape) {
        if (Math.abs(d.H - thresh) < tolerance) {
          ctx.beginPath();
          ctx.arc(
            padL + d.i * cellW + cellW / 2,
            padT + (res - 1 - d.j) * cellH + cellH / 2,
            0.8, 0, Math.PI * 2,
          );
          ctx.stroke();
        }
      }
    }

    // ── Passivity violation overlay ──────────────────────────────────
    ctx.fillStyle = "rgba(225,6,0,0.18)";
    for (const d of landscape) {
      if (d.H > passivityThresh) {
        ctx.fillRect(
          padL + d.i * cellW,
          padT + (res - 1 - d.j) * cellH,
          cellW + 0.5,
          cellH + 0.5,
        );
      }
    }

    // ── Equilibrium crosshair ────────────────────────────────────────
    // Map equilibrium coordinates to pixel position
    // qf range: landscape min to max
    const qfMin = landscape[0].qf, qfMax = landscape[res - 1]?.qf ?? 20;
    const qrMin = landscape[0].qr, qrMax = landscape[(res - 1) * res]?.qr ?? landscape[landscape.length - 1]?.qr ?? 20;
    const qfRange = (+qfMax) - (+qfMin) || 1;
    const qrRange = (+qrMax) - (+qrMin) || 1;

    const eqPx = padL + ((eqFront - (+qfMin)) / qfRange) * plotW;
    const eqPy = padT + (1 - (eqRear - (+qrMin)) / qrRange) * plotH;

    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([4, 3]);
    // Vertical
    ctx.beginPath();
    ctx.moveTo(eqPx, padT);
    ctx.lineTo(eqPx, padT + plotH);
    ctx.stroke();
    // Horizontal
    ctx.beginPath();
    ctx.moveTo(padL, eqPy);
    ctx.lineTo(padL + plotW, eqPy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Crosshair dot
    ctx.beginPath();
    ctx.arc(eqPx, eqPy, 3, 0, Math.PI * 2);
    ctx.fillStyle = C.w;
    ctx.fill();

    // ── Axis labels ──────────────────────────────────────────────────
    ctx.fillStyle = C.dm;
    const fontSize = Math.round(W * 0.022);
    ctx.font = `${fontSize}px ${C.dt}`;

    // X axis
    ctx.textAlign = "center";
    ctx.fillText("q_front (mm)", padL + plotW / 2, H - 4);

    // Y axis (rotated)
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("q_rear (mm)", 0, 0);
    ctx.restore();

    // Tick marks
    ctx.font = `${Math.round(fontSize * 0.75)}px ${C.dt}`;
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const qfVal = +qfMin + frac * qfRange;
      const px = padL + frac * plotW;
      ctx.fillText(qfVal.toFixed(0), px, padT + plotH + 14);
    }
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const qrVal = +qrMin + frac * qrRange;
      const py = padT + plotH - frac * plotH;
      ctx.fillText(qrVal.toFixed(0), padL - 4, py + 3);
    }

    // ── Colorbar legend (right edge) ─────────────────────────────────
    const barW = 8, barH = plotH * 0.6;
    const barX = W - padR - barW - 2;
    const barY = padT + (plotH - barH) / 2;
    for (let i = 0; i < barH; i++) {
      const norm = 1 - i / barH;
      const { r, g, b } = energyColor(norm);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(barX, barY + i, barW, 1.5);
    }
    ctx.fillStyle = C.dm;
    ctx.font = `${Math.round(fontSize * 0.7)}px ${C.dt}`;
    ctx.textAlign = "left";
    ctx.fillText(`${maxH.toFixed(0)}J`, barX + barW + 3, barY + 4);
    ctx.fillText(`${minH.toFixed(0)}J`, barX + barW + 3, barY + barH);

  }, [landscape, eqFront, eqRear, width, height]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={width * 2}
        height={height * 2}
        style={{
          width, height, borderRadius: 8,
          background: "rgba(5,7,11,0.3)",
          display: "block",
        }}
      />
      {/* Legend */}
      <div style={{
        display: "flex", gap: 8, padding: "6px 0",
        justifyContent: "center", flexWrap: "wrap",
      }}>
        {[
          ["Low H(q)", "rgba(30,60,180,0.7)"],
          ["High H(q)", "rgba(225,100,30,0.7)"],
          ["Passivity violation", "rgba(225,6,0,0.25)"],
          ["Equilibrium", C.w],
        ].map(([l, c]) => (
          <div key={l} style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <div style={{ width: 8, height: 4, background: c, borderRadius: 1 }} />
            <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}