// ═══════════════════════════════════════════════════════════════════════════════
// src/canvas/FrictionHeatmap.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// Canvas heatmap of the combined slip friction surface:
//   |F|(α, κ) = sqrt(Fy(α,κ)² + Fx(α,κ)²)
//
// Visualizes the friction circle topology — peak force lies along the
// diagonal ridge where combined slip optimally balances lateral and
// longitudinal force generation.
//
// Color: cool blue (low force) → green → amber → warm red (peak force)
// Axes: slip angle α (°) horizontal, slip ratio κ vertical
//
// Props:
//   data   — Array from gFrictionSurface(): { ai, ki, alpha, kappa, Fy, Fx, F }
//   width  — CSS px (default 360)
//   height — CSS px (default 360)
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useRef, useEffect } from "react";
import { C } from "../theme.js";

function forceColor(norm) {
  // norm: 0 (zero force) → 1 (peak force)
  // Blue → teal → green → amber → orange-red
  const r = Math.round(20 + norm * 210);
  const g = Math.round(50 + (norm < 0.5 ? norm * 2 * 160 : (1 - norm) * 2 * 160) + 20);
  const b = Math.round(200 * (1 - norm * norm) + 30);
  return `rgb(${r},${g},${b})`;
}

export default function FrictionHeatmap({ data, width = 360, height = 360 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !data.length) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Grid resolution
    const res = Math.round(Math.sqrt(data.length));
    if (res < 2) return;

    // Layout
    const padL = 44, padB = 34, padT = 10, padR = 14;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const cellW = plotW / res;
    const cellH = plotH / res;

    // Force range
    const maxF = Math.max(...data.map(d => d.F), 1);

    // ── Filled cells ─────────────────────────────────────────────────
    for (const d of data) {
      const norm = d.F / maxF;
      ctx.fillStyle = forceColor(norm);
      // ai maps to X (slip angle), ki maps to Y (slip ratio — flip for screen)
      ctx.fillRect(
        padL + d.ai * cellW,
        padT + (res - 1 - d.ki) * cellH,
        cellW + 0.5,
        cellH + 0.5,
      );
    }

    // ── Friction circle contour (iso-force lines at 25%, 50%, 75%, 95%) ──
    ctx.lineWidth = 0.5;
    for (const pct of [0.25, 0.50, 0.75, 0.95]) {
      const thresh = maxF * pct;
      const tol = maxF * 0.03;
      ctx.strokeStyle = `rgba(255,255,255,${0.1 + pct * 0.15})`;

      for (const d of data) {
        if (Math.abs(d.F - thresh) < tol) {
          ctx.beginPath();
          ctx.arc(
            padL + d.ai * cellW + cellW / 2,
            padT + (res - 1 - d.ki) * cellH + cellH / 2,
            0.8, 0, Math.PI * 2,
          );
          ctx.stroke();
        }
      }
    }

    // ── Origin crosshair (α=0, κ=0) ─────────────────────────────────
    const originX = padL + plotW / 2;
    const originY = padT + plotH / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(originX, padT);
    ctx.lineTo(originX, padT + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padL, originY);
    ctx.lineTo(padL + plotW, originY);
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Axis labels ──────────────────────────────────────────────────
    const fontSize = Math.round(W * 0.022);
    ctx.fillStyle = C.dm;
    ctx.font = `${fontSize}px ${C.dt}`;

    // X axis
    ctx.textAlign = "center";
    ctx.fillText("Slip Angle α (°)", padL + plotW / 2, H - 4);

    // Y axis
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Slip Ratio κ", 0, 0);
    ctx.restore();

    // Tick values
    ctx.font = `${Math.round(fontSize * 0.75)}px ${C.dt}`;
    // X ticks (slip angle)
    const alphaMin = data[0]?.alpha ?? -15;
    const alphaMax = data[res - 1]?.alpha ?? 15;
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const val = +alphaMin + frac * (+alphaMax - +alphaMin);
      ctx.fillText(val.toFixed(0) + "°", padL + frac * plotW, padT + plotH + 14);
    }
    // Y ticks (slip ratio)
    ctx.textAlign = "right";
    const kappaVals = data.filter(d => d.ai === 0).map(d => +d.kappa);
    const kappaMin = Math.min(...kappaVals);
    const kappaMax = Math.max(...kappaVals);
    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const val = kappaMin + frac * (kappaMax - kappaMin);
      ctx.fillText(val.toFixed(2), padL - 4, padT + plotH - frac * plotH + 3);
    }

    // ── Colorbar ─────────────────────────────────────────────────────
    const barW = 8, barH = plotH * 0.5;
    const barX = W - padR - barW;
    const barY = padT + (plotH - barH) / 2;
    for (let i = 0; i < barH; i++) {
      const norm = 1 - i / barH;
      ctx.fillStyle = forceColor(norm);
      ctx.fillRect(barX, barY + i, barW, 1.5);
    }
    ctx.fillStyle = C.dm;
    ctx.font = `${Math.round(fontSize * 0.65)}px ${C.dt}`;
    ctx.textAlign = "left";
    ctx.fillText(`${maxF.toFixed(0)}N`, barX - 4, barY - 4);
    ctx.fillText("0N", barX + 1, barY + barH + 10);

  }, [data, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width * 2}
      height={height * 2}
      style={{
        width, height, borderRadius: 8,
        display: "block",
      }}
    />
  );
}