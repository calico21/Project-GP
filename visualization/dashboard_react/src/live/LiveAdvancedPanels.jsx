// ═══════════════════════════════════════════════════════════════════════════════
// src/live/LiveAdvancedPanels.jsx — v4.5 RESPONSIVE REBUILD
// All Canvas panels fill their container. No fixed pixel widths.
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useRef, useEffect, useState, useCallback } from "react";
import { C } from "../theme.js";

// ── Responsive canvas hook: measures container width AND height from grid ─────
function useResponsiveCanvas() {
  const containerRef = useRef(null);
  const [dims, setDims] = useState({ w: 300, h: 200 });
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      const w = el.clientWidth || 300;
      const h = el.clientHeight || 200;
      setDims({ w, h: Math.max(h, 80) });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { containerRef, ...dims };
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. FOUR-CORNER BREATHING FRICTION ELLIPSES
// ═════════════════════════════════════════════════════════════════════════════
export function FourCornerEllipses({ f }) {
  const { containerRef, w, h } = useResponsiveCanvas();
  const cv = useRef(null);
  useEffect(() => {
    const c = cv.current; if (!c || w < 10) return;
    c.width = w * 2; c.height = h * 2;
    const ctx = c.getContext("2d"); const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const mu = 1.35, maxFz = 2000, maxR = W * 0.18;
    const corners = [
      { label: "FL", fz: +f.FzFL || 700, fx: +f.FxFL || 0, fy: +f.FyFL || 0, cx: W * 0.25, cy: H * 0.28 },
      { label: "FR", fz: +f.FzFR || 700, fx: +f.FxFR || 0, fy: +f.FyFR || 0, cx: W * 0.75, cy: H * 0.28 },
      { label: "RL", fz: +f.FzRL || 600, fx: +f.FxRL || 0, fy: +f.FyRL || 0, cx: W * 0.25, cy: H * 0.72 },
      { label: "RR", fz: +f.FzRR || 600, fx: +f.FxRR || 0, fy: +f.FyRR || 0, cx: W * 0.75, cy: H * 0.72 },
    ];
    corners.forEach(t => {
      const absFz = Math.abs(t.fz);
      const r = Math.max(2, (absFz / maxFz) * maxR);
      const fMax = mu * absFz || 1;
      [1, 0.75, 0.5].forEach(s => { ctx.strokeStyle = `rgba(0,210,255,${0.08 + s * 0.08})`; ctx.lineWidth = 0.5; ctx.beginPath(); ctx.arc(t.cx, t.cy, r * s, 0, Math.PI * 2); ctx.stroke(); });
      ctx.strokeStyle = "rgba(62,74,100,0.15)"; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(t.cx - r, t.cy); ctx.lineTo(t.cx + r, t.cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(t.cx, t.cy - r); ctx.lineTo(t.cx, t.cy + r); ctx.stroke();
      const dotX = t.cx + (t.fy / fMax) * r, dotY = t.cy - (t.fx / fMax) * r;
      const util = Math.sqrt(t.fx * t.fx + t.fy * t.fy) / fMax;
      const col = util > 0.9 ? C.red : util > 0.7 ? C.am : C.gn;
      ctx.beginPath(); ctx.arc(dotX, dotY, Math.max(1, W * 0.008), 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
      ctx.fillStyle = C.dm; ctx.font = `bold ${W * 0.028}px ${C.dt}`; ctx.textAlign = "center";
      ctx.fillText(t.label, t.cx, t.cy - r - W * 0.015);
      ctx.fillStyle = C.br; ctx.font = `${W * 0.022}px ${C.dt}`;
      ctx.fillText(`${absFz.toFixed(0)}N · ${(util * 100).toFixed(0)}%`, t.cx, t.cy + r + W * 0.03);
    });
  }, [f, w, h]);
  return <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative", minHeight: 120 }}><canvas ref={cv} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "block" }} /></div>;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. β vs ψ̇ STABILITY PHASE PLANE
// ═════════════════════════════════════════════════════════════════════════════
export function StabilityPhasePlane({ f }) {
  const { containerRef, w, h } = useResponsiveCanvas();
  const cv = useRef(null);
  const trail = useRef([]);
  useEffect(() => {
    const c = cv.current; if (!c || w < 10) return;
    c.width = w * 2; c.height = h * 2;
    const ctx = c.getContext("2d"); const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, bLim = 0.05, wLim = 2;
    const scX = (W * 0.38) / bLim, scY = (H * 0.38) / wLim;
    // Safe diamond
    ctx.fillStyle = "rgba(0,230,118,0.03)"; ctx.strokeStyle = "rgba(0,230,118,0.25)"; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cx, cy - wLim * scY); ctx.lineTo(cx + bLim * scX, cy); ctx.lineTo(cx, cy + wLim * scY); ctx.lineTo(cx - bLim * scX, cy); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
    // Axes
    ctx.strokeStyle = "rgba(62,74,100,0.25)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(W * 0.08, cy); ctx.lineTo(W * 0.92, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, H * 0.08); ctx.lineTo(cx, H * 0.92); ctx.stroke();
    ctx.fillStyle = C.dm; ctx.font = `${W * 0.025}px ${C.dt}`; ctx.textAlign = "center";
    ctx.fillText("β →", W * 0.85, cy + W * 0.04); ctx.fillText("ψ̇ ↑", cx + W * 0.04, H * 0.1);
    ctx.fillStyle = "rgba(225,6,0,0.2)"; ctx.font = `${W * 0.02}px ${C.dt}`;
    ctx.fillText("SPIN", cx, H * 0.06); ctx.fillText("SPIN", cx, H * 0.96);
    // Trail
    const beta = +f.sideslip || 0, wz = +f.wz || 0;
    trail.current.push({ b: beta, w: wz }); if (trail.current.length > 80) trail.current.shift();
    trail.current.forEach((p, i) => {
      const a = i / trail.current.length;
      const outside = Math.abs(p.b) / bLim + Math.abs(p.w) / wLim > 1;
      ctx.beginPath(); ctx.arc(cx + p.b * scX, cy - p.w * scY, Math.max(1, W * 0.004), 0, Math.PI * 2);
      ctx.fillStyle = outside ? `rgba(225,6,0,${a * 0.5})` : `rgba(0,210,255,${a * 0.35})`; ctx.fill();
    });
    ctx.beginPath(); ctx.arc(cx + beta * scX, cy - wz * scY, Math.max(2, W * 0.01), 0, Math.PI * 2);
    ctx.fillStyle = (Math.abs(beta) / bLim + Math.abs(wz) / wLim > 1) ? C.red : C.gn; ctx.fill();
  }, [f, w, h]);
  return <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative", minHeight: 120 }}><canvas ref={cv} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "block" }} /></div>;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. AERO PLATFORM
// ═════════════════════════════════════════════════════════════════════════════
export function AeroPlatform({ f }) {
  const { containerRef, w, h } = useResponsiveCanvas();
  const cv = useRef(null);
  const trail = useRef([]);
  useEffect(() => {
    const c = cv.current; if (!c || w < 10) return;
    c.width = w * 2; c.height = h * 2;
    const ctx = c.getContext("2d"); const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const pad = W * 0.12, pW = W - 2 * pad, pH = H - 2 * pad;
    const rhFmin = 15, rhFmax = 55, rhRmin = 20, rhRmax = 75;
    const tx = v => pad + ((v - rhFmin) / (rhFmax - rhFmin)) * pW;
    const ty = v => pad + pH - ((v - rhRmin) / (rhRmax - rhRmin)) * pH;
    // Danger zone
    ctx.fillStyle = "rgba(225,6,0,0.05)"; ctx.fillRect(pad, ty(rhRmin + 5), tx(20) - pad, ty(rhRmin) - ty(rhRmin + 5));
    // Axes
    ctx.strokeStyle = "rgba(62,74,100,0.25)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, pad + pH); ctx.lineTo(pad + pW, pad + pH); ctx.stroke();
    ctx.fillStyle = C.dm; ctx.font = `${W * 0.025}px ${C.dt}`; ctx.textAlign = "center";
    ctx.fillText("Front RH (mm) →", pad + pW / 2, H - W * 0.02);
    ctx.save(); ctx.translate(W * 0.04, pad + pH / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Rear RH →", 0, 0); ctx.restore();
    const rhF = +f.rideHF || 30, rhR = +f.rideHR || 45;
    trail.current.push({ f: rhF, r: rhR }); if (trail.current.length > 50) trail.current.shift();
    trail.current.forEach((p, i) => { ctx.beginPath(); ctx.arc(tx(p.f), ty(p.r), Math.max(1, W * 0.004), 0, Math.PI * 2); ctx.fillStyle = `rgba(0,210,255,${(i / trail.current.length) * 0.3})`; ctx.fill(); });
    ctx.beginPath(); ctx.arc(tx(rhF), ty(rhR), Math.max(2, W * 0.012), 0, Math.PI * 2); ctx.fillStyle = rhF < 20 ? C.red : C.gn; ctx.fill();
    ctx.fillStyle = C.br; ctx.font = `bold ${W * 0.028}px ${C.dt}`; ctx.textAlign = "left";
    ctx.fillText(`F:${rhF.toFixed(1)}  R:${rhR.toFixed(1)} mm`, pad + W * 0.01, pad + W * 0.04);
  }, [f, w, h]);
  return <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative", minHeight: 120 }}><canvas ref={cv} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "block" }} /></div>;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. DAMPER SATURATION BARS
// ═════════════════════════════════════════════════════════════════════════════
export function DamperBars({ f }) {
  const { containerRef, w, h } = useResponsiveCanvas();
  const cv = useRef(null);
  useEffect(() => {
    const c = cv.current; if (!c || w < 10) return;
    c.width = w * 2; c.height = h * 2;
    const ctx = c.getContext("2d"); const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const corners = [
      { label: "FL", vel: +f.zdFL || 0, travel: +f.zFL || -12.8, col: "#42a5f5" },
      { label: "FR", vel: +f.zdFR || 0, travel: +f.zFR || -12.8, col: "#66bb6a" },
      { label: "RL", vel: +f.zdRL || 0, travel: +f.zRL || -14.2, col: "#ffa726" },
      { label: "RR", vel: +f.zdRR || 0, travel: +f.zRR || -14.2, col: "#ab47bc" },
    ];
    const barW = W / 6, gap = (W - 4 * barW) / 5, cy = H / 2, maxVel = 0.4, kneeVel = 0.15;
    corners.forEach((t, i) => {
      const x = gap + i * (barW + gap);
      ctx.fillStyle = "rgba(30,40,60,0.35)"; ctx.fillRect(x, cy - H * 0.35, barW, H * 0.7);
      ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(x, cy); ctx.lineTo(x + barW, cy); ctx.stroke();
      const norm = Math.min(1, Math.abs(t.vel) / maxVel), fillH = norm * H * 0.35;
      ctx.fillStyle = `${t.col}70`;
      if (t.vel < 0) ctx.fillRect(x, cy - fillH, barW, fillH); else ctx.fillRect(x, cy, barW, fillH);
      const kneeY = (kneeVel / maxVel) * H * 0.35;
      ctx.strokeStyle = "rgba(255,171,0,0.4)"; ctx.lineWidth = 1; ctx.setLineDash([3, 2]);
      ctx.beginPath(); ctx.moveTo(x, cy - kneeY); ctx.lineTo(x + barW, cy - kneeY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, cy + kneeY); ctx.lineTo(x + barW, cy + kneeY); ctx.stroke();
      ctx.setLineDash([]);
      if (t.travel < -25) { ctx.fillStyle = "rgba(225,6,0,0.2)"; ctx.fillRect(x, cy - H * 0.35, barW, H * 0.7); }
      ctx.fillStyle = t.col; ctx.font = `bold ${W * 0.028}px ${C.dt}`; ctx.textAlign = "center";
      ctx.fillText(t.label, x + barW / 2, H - W * 0.015);
      ctx.fillStyle = C.br; ctx.font = `${W * 0.024}px ${C.dt}`;
      ctx.fillText(`${(t.vel * 1000).toFixed(0)}`, x + barW / 2, W * 0.035);
    });
    ctx.fillStyle = C.dm; ctx.font = `${W * 0.018}px ${C.dt}`; ctx.textAlign = "center";
    ctx.fillText("mm/s · Dashed = knee (LS→HS) · Red = bumpstop", W / 2, H - 2);
  }, [f, w, h]);
  return <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative", minHeight: 120 }}><canvas ref={cv} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "block" }} /></div>;
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. WMPC COST RADAR
// ═════════════════════════════════════════════════════════════════════════════
export function WMPCCostRadar({ f }) {
  const { containerRef, w, h } = useResponsiveCanvas();
  const cv = useRef(null);
  useEffect(() => {
    const c = cv.current; if (!c || w < 10) return;
    c.width = w * 2; c.height = h * 2;
    const ctx = c.getContext("2d"); const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, r = W * 0.32;
    const axes = [
      { label: "Lap Time", val: +f.costLap || 0, max: 50, col: C.red },
      { label: "Tracking", val: +f.costTrack || 0, max: 30, col: C.cy },
      { label: "L1 Wavelet", val: +f.costL1 || 0, max: 15, col: C.pr },
      { label: "μ Slack", val: Math.max(0, 0.15 - (+f.slkGrip || 0.1)) * 100, max: 15, col: C.am },
      { label: "UT Safety", val: Math.max(0, 1.5 - (+f.dLeft || 1.5)) * 10, max: 15, col: C.gn },
    ];
    const n = axes.length;
    [0.25, 0.5, 0.75, 1].forEach(s => {
      ctx.strokeStyle = `rgba(62,74,100,${0.08 + s * 0.08})`; ctx.lineWidth = 0.5;
      ctx.beginPath(); for (let i = 0; i <= n; i++) { const a = (i / n) * Math.PI * 2 - Math.PI / 2; const px = cx + Math.cos(a) * r * s, py = cy + Math.sin(a) * r * s; i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); } ctx.stroke();
    });
    axes.forEach((ax, i) => { const a = (i / n) * Math.PI * 2 - Math.PI / 2; ctx.strokeStyle = "rgba(62,74,100,0.15)"; ctx.lineWidth = 0.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r); ctx.stroke();
      ctx.fillStyle = ax.col; ctx.font = `${W * 0.024}px ${C.dt}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(ax.label, cx + Math.cos(a) * (r + W * 0.06), cy + Math.sin(a) * (r + W * 0.06));
    });
    ctx.beginPath(); axes.forEach((ax, i) => { const a = (i / n) * Math.PI * 2 - Math.PI / 2; const nm = Math.min(1, ax.val / ax.max); i === 0 ? ctx.moveTo(cx + Math.cos(a) * r * nm, cy + Math.sin(a) * r * nm) : ctx.lineTo(cx + Math.cos(a) * r * nm, cy + Math.sin(a) * r * nm); }); ctx.closePath();
    ctx.fillStyle = "rgba(0,210,255,0.08)"; ctx.fill(); ctx.strokeStyle = "rgba(0,210,255,0.5)"; ctx.lineWidth = 1.5; ctx.stroke();
    axes.forEach((ax, i) => { const a = (i / n) * Math.PI * 2 - Math.PI / 2; const nm = Math.min(1, ax.val / ax.max); ctx.beginPath(); ctx.arc(cx + Math.cos(a) * r * nm, cy + Math.sin(a) * r * nm, Math.max(1, W * 0.008), 0, Math.PI * 2); ctx.fillStyle = ax.col; ctx.fill(); });
  }, [f, w, h]);
  return <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative", minHeight: 120 }}><canvas ref={cv} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "block" }} /></div>;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. LIVE PACEJKA CURVE + GP ±2σ UNCERTAINTY BAND
// ═════════════════════════════════════════════════════════════════════════════
export function LivePacejkaCurve({ f }) {
  const { containerRef, w, h } = useResponsiveCanvas();
  const cv = useRef(null);
  useEffect(() => {
    const c = cv.current; if (!c || w < 10) return;
    c.width = w * 2; c.height = h * 2;
    const ctx = c.getContext("2d"); const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const pad = { l: W * 0.07, r: W * 0.03, t: H * 0.08, b: H * 0.14 };
    const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
    const aMax = 15, fMax = 3500, nPts = 80;
    const muT = +f.muTherm || 0.95, muP = +f.muPress || 1.0;
    const dFy = +f.dFyPinn || 0, curAlpha = +f.alphaFL || 0, Fz = Math.abs(+f.FzFL || 700);
    const sigBase = 150, sigSlope = 200;

    const toX = a => pad.l + ((a + aMax) / (2 * aMax)) * pW;
    const toY = fy => pad.t + pH * (1 - (fy + fMax) / (2 * fMax));
    const pacejka = (a, scale) => (Fz / 700) * 3000 * Math.sin(1.4 * Math.atan(12 * (a * Math.PI / 180))) * scale;

    // ── GP ±2σ uncertainty band ──────────────────────────────────────
    ctx.beginPath();
    for (let i = 0; i < nPts; i++) { const a = (i / (nPts - 1) - 0.5) * 2 * aMax; const sig = sigBase + sigSlope * Math.abs(a) / aMax; const fy = pacejka(a, muT * muP) + dFy * Math.sin(a * Math.PI / 180 * 3) + 2 * sig; ctx.lineTo(toX(a), toY(fy)); }
    for (let i = nPts - 1; i >= 0; i--) { const a = (i / (nPts - 1) - 0.5) * 2 * aMax; const sig = sigBase + sigSlope * Math.abs(a) / aMax; const fy = pacejka(a, muT * muP) + dFy * Math.sin(a * Math.PI / 180 * 3) - 2 * sig; ctx.lineTo(toX(a), toY(fy)); }
    ctx.closePath(); ctx.fillStyle = "rgba(179,136,255,0.08)"; ctx.fill();

    // ── ±1σ band ─────────────────────────────────────────────────────
    ctx.beginPath();
    for (let i = 0; i < nPts; i++) { const a = (i / (nPts - 1) - 0.5) * 2 * aMax; const sig = (sigBase + sigSlope * Math.abs(a) / aMax) * 0.5; const fy = pacejka(a, muT * muP) + dFy * Math.sin(a * Math.PI / 180 * 3) + sig; ctx.lineTo(toX(a), toY(fy)); }
    for (let i = nPts - 1; i >= 0; i--) { const a = (i / (nPts - 1) - 0.5) * 2 * aMax; const sig = (sigBase + sigSlope * Math.abs(a) / aMax) * 0.5; const fy = pacejka(a, muT * muP) + dFy * Math.sin(a * Math.PI / 180 * 3) - sig; ctx.lineTo(toX(a), toY(fy)); }
    ctx.closePath(); ctx.fillStyle = "rgba(179,136,255,0.06)"; ctx.fill();

    // ── Cold baseline (dashed white) ─────────────────────────────────
    ctx.beginPath(); ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
    for (let i = 0; i < nPts; i++) { const a = (i / (nPts - 1) - 0.5) * 2 * aMax; i === 0 ? ctx.moveTo(toX(a), toY(pacejka(a, 1))) : ctx.lineTo(toX(a), toY(pacejka(a, 1))); }
    ctx.stroke(); ctx.setLineDash([]);

    // ── Effective curve (solid amber) ────────────────────────────────
    ctx.beginPath(); ctx.strokeStyle = C.am; ctx.lineWidth = 2.5;
    for (let i = 0; i < nPts; i++) { const a = (i / (nPts - 1) - 0.5) * 2 * aMax; const fy = pacejka(a, muT * muP) + dFy * Math.sin(a * Math.PI / 180 * 3); i === 0 ? ctx.moveTo(toX(a), toY(fy)) : ctx.lineTo(toX(a), toY(fy)); }
    ctx.stroke();

    // ── ±2σ boundary lines ───────────────────────────────────────────
    [1, -1].forEach(sign => { ctx.beginPath(); ctx.strokeStyle = "rgba(179,136,255,0.25)"; ctx.lineWidth = 0.5; ctx.setLineDash([3, 3]);
      for (let i = 0; i < nPts; i++) { const a = (i / (nPts - 1) - 0.5) * 2 * aMax; const sig = sigBase + sigSlope * Math.abs(a) / aMax; const fy = pacejka(a, muT * muP) + dFy * Math.sin(a * Math.PI / 180 * 3) + sign * 2 * sig; i === 0 ? ctx.moveTo(toX(a), toY(fy)) : ctx.lineTo(toX(a), toY(fy)); }
      ctx.stroke(); ctx.setLineDash([]);
    });

    // ── Current operating point ──────────────────────────────────────
    const curFy = pacejka(curAlpha, muT * muP) + dFy * Math.sin(curAlpha * Math.PI / 180 * 3);
    ctx.beginPath(); ctx.arc(toX(curAlpha), toY(curFy), Math.max(2, W * 0.015), 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,171,0,0.15)"; ctx.fill();
    ctx.beginPath(); ctx.arc(toX(curAlpha), toY(curFy), Math.max(2, W * 0.008), 0, Math.PI * 2);
    ctx.fillStyle = C.am; ctx.fill();

    // ── Axes ─────────────────────────────────────────────────────────
    ctx.strokeStyle = "rgba(62,74,100,0.25)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t + pH / 2); ctx.lineTo(pad.l + pW, pad.t + pH / 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l + pW / 2, pad.t); ctx.lineTo(pad.l + pW / 2, pad.t + pH); ctx.stroke();
    ctx.fillStyle = C.dm; ctx.font = `${W * 0.022}px ${C.dt}`; ctx.textAlign = "center";
    ctx.fillText("Slip Angle α (°)", pad.l + pW / 2, H - W * 0.01);
    // Legend
    ctx.textAlign = "left"; ctx.fillStyle = C.dm; ctx.font = `${W * 0.02}px ${C.dt}`;
    ctx.fillText(`μ_T=${muT.toFixed(2)} · μ_P=${muP.toFixed(2)}`, pad.l + W * 0.01, pad.t + W * 0.025);
    const peakDelta = ((muT * muP - 1) * 100).toFixed(0);
    ctx.fillStyle = Math.abs(+peakDelta) > 5 ? C.red : C.gn;
    ctx.fillText(`Δpeak=${peakDelta}%`, pad.l + pW * 0.5, pad.t + W * 0.025);
    ctx.fillStyle = C.pr; ctx.fillText("±2σ GP", pad.l + pW * 0.75, pad.t + W * 0.025);
    // Bottom legend
    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = `${W * 0.018}px ${C.dt}`; ctx.textAlign = "right";
    ctx.fillText("--- Cold baseline", pad.l + pW, H - W * 0.06);
    ctx.fillStyle = C.am; ctx.fillText("— Effective", pad.l + pW, H - W * 0.04);
    ctx.fillStyle = C.pr; ctx.fillText("■ GP ±2σ band", pad.l + pW, H - W * 0.02);
  }, [f, w, h]);
  return <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative", minHeight: 120 }}><canvas ref={cv} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "block" }} /></div>;
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. DRIVER INPUTS — 3 STACKED LINE CHARTS (actual vs WMPC optimal)
// ═════════════════════════════════════════════════════════════════════════════
export function DriverInputsPanel({ history, tick }) {
  const { containerRef, w, h } = useResponsiveCanvas();
  const cv = useRef(null);
  useEffect(() => {
    const c = cv.current; if (!c || w < 10) return;
    c.width = w * 2; c.height = h * 2;
    const ctx = c.getContext("2d"); const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const data = (history || []).slice(-80);
    if (data.length < 2) return;
    const n = data.length;
    const charts = [
      { label: "STEER", colA: C.cy, colO: "rgba(0,210,255,0.75)", getA: d => +d.steer || 0, getO: d => +d.ctrlSteer || 0, min: -20, max: 20 },
      { label: "THROTTLE", colA: C.gn, colO: "rgba(0,230,118,0.75)", getA: d => Math.max(0, (+d.ctrlThrot || 0)) * 100, getO: d => Math.max(0, (+d.ax || 0)) * 80, min: 0, max: 100 },
      { label: "BRAKE", colA: C.red, colO: "rgba(225,6,0,0.75)", getA: d => Math.max(0, -(+d.ax || 0)) * 70, getO: d => (+d.ctrlBrake || 0) / 30, min: 0, max: 100 },
    ];
    const nC = charts.length, gapY = H * 0.02, chartH = (H - gapY * (nC + 1)) / nC;
    const padL = W * 0.12, padR = W * 0.16, padT = H * 0.03, padB = H * 0.01;

    charts.forEach((ch, ci) => {
      const y0 = gapY + ci * (chartH + gapY);
      const pW = W - padL - padR, pH = chartH - padT - padB;
      // BG
      ctx.fillStyle = "rgba(10,15,25,0.25)"; ctx.fillRect(0, y0, W, chartH);
      // Label
      ctx.fillStyle = ch.colA; ctx.font = `bold ${W * 0.035}px ${C.dt}`; ctx.textAlign = "left";
      ctx.fillText(ch.label, padL, y0 + padT + W * 0.025);
      // Mid grid line
      ctx.strokeStyle = "rgba(25,35,55,0.35)"; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(padL, y0 + padT + pH / 2); ctx.lineTo(padL + pW, y0 + padT + pH / 2); ctx.stroke();
      // Lines: optimal (dashed, behind) then actual (solid)
      [{ get: ch.getO, col: ch.colO, w: 1.5, dash: [5, 4] }, { get: ch.getA, col: ch.colA, w: 2.5, dash: [] }].forEach(line => {
        ctx.strokeStyle = line.col; ctx.lineWidth = line.w; ctx.setLineDash(line.dash); ctx.beginPath();
        for (let i = 0; i < n; i++) { const val = line.get(data[i]); const norm = (val - ch.min) / (ch.max - ch.min); const x = padL + (i / (n - 1)) * pW; const y = y0 + padT + pH * (1 - Math.max(0, Math.min(1, norm))); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
        ctx.stroke(); ctx.setLineDash([]);
      });
      // Current values (right margin)
      const lastA = ch.getA(data[n - 1]), lastO = ch.getO(data[n - 1]), delta = lastA - lastO;
      ctx.fillStyle = ch.colA; ctx.font = `bold ${W * 0.035}px ${C.dt}`; ctx.textAlign = "left";
      ctx.fillText(lastA.toFixed(1), padL + pW + W * 0.015, y0 + chartH * 0.35);
      ctx.fillStyle = ch.colO; ctx.font = `${W * 0.028}px ${C.dt}`;
      ctx.fillText(lastO.toFixed(1), padL + pW + W * 0.015, y0 + chartH * 0.55);
      if (Math.abs(delta) > (ch.max - ch.min) * 0.05) {
        ctx.fillStyle = Math.abs(delta) > (ch.max - ch.min) * 0.15 ? C.red : C.am;
        ctx.font = `bold ${W * 0.03}px ${C.dt}`;
        ctx.fillText(`Δ${delta > 0 ? "+" : ""}${delta.toFixed(1)}`, padL + pW + W * 0.015, y0 + chartH * 0.78);
      }
      // Y ticks
      ctx.fillStyle = C.dm; ctx.font = `${W * 0.022}px ${C.dt}`; ctx.textAlign = "right";
      ctx.fillText(ch.max.toString(), padL - W * 0.01, y0 + padT + W * 0.015);
      ctx.fillText(ch.min.toString(), padL - W * 0.01, y0 + padT + pH);
    });
    // Legend
    ctx.fillStyle = C.dm; ctx.font = `${W * 0.022}px ${C.dt}`; ctx.textAlign = "center";
    ctx.fillText("Solid = Driver · Dashed = WMPC Optimal", W / 2, H - W * 0.005);
  }, [history, tick, w, h]);
  return <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative", minHeight: 120 }}><canvas ref={cv} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "block" }} /></div>;
}