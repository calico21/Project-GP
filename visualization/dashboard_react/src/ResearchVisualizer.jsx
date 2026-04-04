// ═══════════════════════════════════════════════════════════════════════════
// src/ResearchVisualizer.jsx — Project-GP Dashboard v4.3
// ═══════════════════════════════════════════════════════════════════════════
// Parametric suspension engineering schematics — three canonical views
// (front / side / top) that react in real-time to parameter changes.
//
// Shows: camber, caster, toe, SVSA lines, roll center, anti-geometry,
// natural frequencies, damping ratios, LLTD.
//
// Props:
//   values:  { paramKey: number }  — current parameter values
//   frozen:  { paramKey: boolean } — freeze state
//   car:     "ter26" | "ter27"
//   carCfg:  car config object (defaults, lb, ub, mass, etc.)
// ═══════════════════════════════════════════════════════════════════════════

import React, { useMemo, useState } from "react";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// PHYSICS COMPUTATIONS (all pure functions, mirror vehicle_dynamics.py)
// ═══════════════════════════════════════════════════════════════════════════

function computeDerived(v, carCfg) {
  const mass   = carCfg.mass;
  const mS     = mass * 0.88;   // sprung mass approx
  const mUsF   = carCfg.drivetrain === "4WD" ? 12.5 : 7.74;
  const mUsR   = carCfg.drivetrain === "4WD" ? 8.0 : 7.76;
  const wb     = 1.55;
  const lf     = carCfg.drivetrain === "4WD" ? 0.806 : 0.8525;
  const lr     = wb - lf;
  const tF     = carCfg.drivetrain === "4WD" ? 1.22 : 1.20;
  const tR     = carCfg.drivetrain === "4WD" ? 1.20 : 1.18;
  const Ix     = carCfg.drivetrain === "4WD" ? 52 : 45;
  const Iy     = carCfg.drivetrain === "4WD" ? 92 : 85;
  const mrF    = 1.14, mrR = 1.16;
  const kTire  = 50000;  // tire radial stiffness N/m

  // Spring rates and wheel rates
  const kF = v.k_f || 35000, kR = v.k_r || 38000;
  const cF = v.c_low_f || 1800, cR = v.c_low_r || 1800;
  const arbF = v.arb_f || 800, arbR = v.arb_r || 600;
  const hCG = v.h_cg || 0.33;

  const wrF = kF / (mrF * mrF);
  const wrR = kR / (mrR * mrR);
  const drF = cF / (mrF * mrF);
  const drR = cR / (mrR * mrR);

  // Natural frequencies
  const fnHeave = Math.sqrt((2 * wrF + 2 * wrR) / mS) / (2 * Math.PI);
  const fnRoll  = Math.sqrt((wrF + arbF) * tF * tF * 0.5 + (wrR + arbR) * tR * tR * 0.5) / (Ix > 0 ? Math.sqrt(Ix) : 1) / (2 * Math.PI);
  const fnPitch = Math.sqrt(wrF * lf * lf + wrR * lr * lr) / (Iy > 0 ? Math.sqrt(Iy) : 1) / (2 * Math.PI);
  const fnWheelF = Math.sqrt((wrF + kTire) / mUsF) / (2 * Math.PI);
  const fnWheelR = Math.sqrt((wrR + kTire) / mUsR) / (2 * Math.PI);

  // Damping ratios
  const zetaHeave = (2 * drF + 2 * drR) / (2 * Math.sqrt((2 * wrF + 2 * wrR) * mS) + 1e-3);
  const zetaRoll  = ((drF + drR) * tF * tF * 0.5) / (2 * Math.sqrt(((wrF + arbF) * tF * tF * 0.5 + (wrR + arbR) * tR * tR * 0.5) * Ix) + 1e-3);
  const zetaPitch = (drF * lf * lf + drR * lr * lr) / (2 * Math.sqrt((wrF * lf * lf + wrR * lr * lr) * Iy) + 1e-3);

  // LLTD (Lateral Load Transfer Distribution)
  const kRollF  = (wrF + arbF) * (tF / 2) * (tF / 2);
  const kRollR  = (wrR + arbR) * (tR / 2) * (tR / 2);
  const lltdF   = kRollF / (kRollF + kRollR + 1e-3);
  const lltdR   = 1 - lltdF;
  const underOvr = lltdF > 0.52 ? "US" : lltdF < 0.48 ? "OS" : "Neutral";

  // Anti-geometry SVSA angles
  const asDeg  = Math.atan((v.anti_squat || 0.3) * hCG / wb) * 180 / Math.PI;
  const adfDeg = Math.atan((v.anti_dive_f || 0.4) * hCG / wb) * 180 / Math.PI;
  const adrDeg = Math.atan((v.anti_dive_r || 0.1) * hCG / wb) * 180 / Math.PI;
  const alDeg  = Math.atan((v.anti_lift || 0.2) * hCG / wb) * 180 / Math.PI;

  // Roll center heights (simplified: from anti-geometry)
  const hRCf = 0.040 + (v.h_ride_f || 0.03) * 0.2;
  const hRCr = 0.055 + (v.h_ride_r || 0.03) * 0.3;

  return {
    wrF, wrR, drF, drR,
    fnHeave, fnRoll, fnPitch, fnWheelF, fnWheelR,
    zetaHeave, zetaRoll, zetaPitch,
    lltdF, lltdR, underOvr,
    asDeg, adfDeg, adrDeg, alDeg,
    hRCf, hRCr,
    hCG, wb, lf, lr, tF, tR,
    camberF: v.camber_f ?? -2.0, camberR: v.camber_r ?? -1.5,
    toeF: v.toe_f ?? -0.1, toeR: v.toe_r ?? -0.15,
    casterF: v.castor_f ?? 5.0,
    rideF: (v.h_ride_f ?? 0.03) * 1000,
    rideR: (v.h_ride_r ?? 0.03) * 1000,
    antiSq: (v.anti_squat ?? 0.3) * 100,
    antiDiveF: (v.anti_dive_f ?? 0.4) * 100,
    antiDiveR: (v.anti_dive_r ?? 0.1) * 100,
    antiLift: (v.anti_lift ?? 0.2) * 100,
    brakeBias: (v.brake_bias_f ?? 0.6) * 100,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// SVG HELPERS
// ═══════════════════════════════════════════════════════════════════════════
const S = { w: 520, h: 340 }; // viewport

function DimLine({ x1, y1, x2, y2, label, color = C.dm, offset = 0 }) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={0.5} strokeDasharray="3 2" />
      <line x1={x1} y1={y1 - 3} x2={x1} y2={y1 + 3} stroke={color} strokeWidth={0.5} />
      <line x1={x2} y1={y2 - 3} x2={x2} y2={y2 + 3} stroke={color} strokeWidth={0.5} />
      <text x={mx} y={my + offset - 4} fill={color} fontSize={7} textAnchor="middle" fontFamily={C.dt}>{label}</text>
    </g>
  );
}

function AngleArc({ cx, cy, r, startDeg, endDeg, color, label }) {
  const s = (startDeg * Math.PI) / 180;
  const e = (endDeg * Math.PI) / 180;
  const x1 = cx + r * Math.cos(s), y1 = cy - r * Math.sin(s);
  const x2 = cx + r * Math.cos(e), y2 = cy - r * Math.sin(e);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  const sweep = endDeg > startDeg ? 0 : 1;
  const mid = ((startDeg + endDeg) / 2 * Math.PI) / 180;
  const lx = cx + (r + 10) * Math.cos(mid), ly = cy - (r + 10) * Math.sin(mid);
  return (
    <g>
      <path d={`M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} ${sweep} ${x2} ${y2}`}
        fill="none" stroke={color} strokeWidth={1} />
      <text x={lx} y={ly + 3} fill={color} fontSize={7} textAnchor="middle" fontFamily={C.dt} fontWeight="bold">{label}</text>
    </g>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// FRONT VIEW SCHEMATIC
// ═══════════════════════════════════════════════════════════════════════════
function FrontView({ d, frozen }) {
  const cx = S.w / 2, ground = S.h - 30;
  const scale = 200;  // pixels per metre

  const halfTrackF = (d.tF / 2) * scale;
  const rideH = d.rideF * scale / 1000;
  const wheelR = 0.2032 * scale;
  const wheelW = 16;

  // Chassis body rectangle
  const bodyW = halfTrackF * 0.7, bodyH = 30;
  const bodyY = ground - rideH - wheelR - bodyH;

  // Roll center
  const rcY = ground - d.hRCf * scale;
  const cgY = ground - d.hCG * scale;

  // Wishbone geometry (simplified schematic)
  const lbjY = ground - wheelR + 8;
  const ubjY = ground - wheelR - 35;
  const lca_inZ = halfTrackF * 0.38;
  const uca_inZ = halfTrackF * 0.30;

  // Camber angle visualization
  const camF = d.camberF;
  const camRad = (camF * Math.PI) / 180;

  return (
    <svg width={S.w} height={S.h} style={{ background: C.bg, borderRadius: 8, border: `1px solid ${C.b1}` }}>
      {/* Grid */}
      {Array.from({ length: 11 }, (_, i) => (
        <line key={`gv${i}`} x1={i * S.w / 10} y1={0} x2={i * S.w / 10} y2={S.h} stroke={C.b1} strokeWidth={0.3} />
      ))}
      {Array.from({ length: 8 }, (_, i) => (
        <line key={`gh${i}`} x1={0} y1={i * S.h / 7} x2={S.w} y2={i * S.h / 7} stroke={C.b1} strokeWidth={0.3} />
      ))}

      {/* Ground */}
      <line x1={20} y1={ground} x2={S.w - 20} y2={ground} stroke={C.dm} strokeWidth={1.5} />
      <text x={S.w - 20} y={ground + 12} fill={C.dm} fontSize={7} textAnchor="end" fontFamily={C.dt}>GROUND</text>

      {/* Title */}
      <text x={12} y={16} fill={C.cy} fontSize={9} fontWeight="bold" fontFamily={C.dt}>FRONT VIEW</text>
      <text x={12} y={26} fill={C.dm} fontSize={7} fontFamily={C.dt}>Looking rearward</text>

      {/* Chassis body */}
      <rect x={cx - bodyW} y={bodyY} width={bodyW * 2} height={bodyH}
        rx={4} fill={`${C.cy}08`} stroke={C.cy} strokeWidth={1} />
      <text x={cx} y={bodyY + bodyH / 2 + 3} fill={C.cy} fontSize={7} textAnchor="middle" fontFamily={C.dt}>CHASSIS</text>

      {/* Left and Right suspension */}
      {[1, -1].map(side => {
        const wx = cx + side * halfTrackF;
        const camTilt = side * camRad;
        return (
          <g key={side}>
            {/* Wheel */}
            <g transform={`translate(${wx}, ${ground - wheelR}) rotate(${-camF * side})`}>
              <rect x={-wheelW / 2} y={-wheelR} width={wheelW} height={wheelR * 2}
                rx={4} fill={`${C.am}15`} stroke={C.am} strokeWidth={1.5} />
              <line x1={0} y1={-wheelR + 4} x2={0} y2={wheelR - 4} stroke={C.am} strokeWidth={0.5} strokeDasharray="2 2" />
            </g>

            {/* Camber angle line (vertical reference) */}
            <line x1={wx} y1={ground - wheelR * 2.2} x2={wx} y2={ground}
              stroke={C.dm} strokeWidth={0.3} strokeDasharray="2 3" />

            {/* Camber angle label */}
            {Math.abs(camF) > 0.05 && (
              <AngleArc cx={wx} cy={ground - wheelR} r={wheelR * 0.6}
                startDeg={90} endDeg={90 + camF * side}
                color={frozen?.camber_f ? C.red : C.am}
                label={`${camF.toFixed(1)}°`} />
            )}

            {/* Lower wishbone */}
            <line x1={cx + side * lca_inZ} y1={lbjY + 8} x2={wx} y2={lbjY}
              stroke={C.gn} strokeWidth={2} strokeLinecap="round" />
            {/* Upper wishbone */}
            <line x1={cx + side * uca_inZ} y1={ubjY + 6} x2={wx} y2={ubjY}
              stroke={C.cy} strokeWidth={1.5} strokeLinecap="round" />

            {/* Upright */}
            <line x1={wx} y1={lbjY} x2={wx} y2={ubjY}
              stroke={C.br} strokeWidth={2.5} strokeLinecap="round" />

            {/* Ball joints */}
            <circle cx={wx} cy={lbjY} r={3} fill={C.gn} />
            <circle cx={wx} cy={ubjY} r={2.5} fill={C.cy} />

            {/* Spring (zigzag) */}
            <SpringSVG x1={cx + side * lca_inZ * 0.8} y1={bodyY + bodyH}
              x2={cx + side * lca_inZ * 0.8} y2={ubjY + 15}
              color={C.gn} />
          </g>
        );
      })}

      {/* CG marker */}
      <circle cx={cx} cy={cgY} r={5} fill="none" stroke={C.red} strokeWidth={1.5} />
      <line x1={cx - 4} y1={cgY} x2={cx + 4} y2={cgY} stroke={C.red} strokeWidth={1} />
      <line x1={cx} y1={cgY - 4} x2={cx} y2={cgY + 4} stroke={C.red} strokeWidth={1} />
      <text x={cx + 10} y={cgY + 3} fill={C.red} fontSize={7} fontFamily={C.dt}>CG</text>

      {/* Roll center */}
      <circle cx={cx} cy={rcY} r={4} fill={C.pr} fillOpacity={0.3} stroke={C.pr} strokeWidth={1.5} />
      <text x={cx + 10} y={rcY + 3} fill={C.pr} fontSize={7} fontFamily={C.dt}>RC {(d.hRCf * 1000).toFixed(0)}mm</text>

      {/* Dimension: track width */}
      <DimLine x1={cx - halfTrackF} y1={ground + 16} x2={cx + halfTrackF} y2={ground + 16}
        label={`Track ${(d.tF * 1000).toFixed(0)}mm`} color={C.dm} />

      {/* Dimension: ride height */}
      <DimLine x1={S.w - 35} y1={ground} x2={S.w - 35} y2={ground - rideH}
        label={`${d.rideF.toFixed(1)}mm`} color={C.gn} offset={-10} />
    </svg>
  );
}

// ── Spring SVG helper ───────────────────────────────────────────────────
function SpringSVG({ x1, y1, x2, y2, color, coils = 5 }) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  const ux = dx / len, uy = dy / len;
  const nx = -uy, ny = ux;
  const amp = 5;
  const points = [`${x1},${y1}`];
  for (let i = 1; i <= coils * 2; i++) {
    const t = i / (coils * 2 + 1);
    const px = x1 + dx * t + nx * amp * (i % 2 === 0 ? 1 : -1);
    const py = y1 + dy * t + ny * amp * (i % 2 === 0 ? 1 : -1);
    points.push(`${px},${py}`);
  }
  points.push(`${x2},${y2}`);
  return <polyline points={points.join(" ")} fill="none" stroke={color} strokeWidth={1.2} />;
}

// ═══════════════════════════════════════════════════════════════════════════
// SIDE VIEW SCHEMATIC
// ═══════════════════════════════════════════════════════════════════════════
function SideView({ d, frozen }) {
  const ground = S.h - 30;
  const scale = 160;
  const wheelR = 0.2032 * scale;
  const frontX = 80 + d.lf * scale;
  const rearX = 80;
  const cgX = 80 + d.lf * scale * 0.55;
  const cgY = ground - d.hCG * scale;
  const rideH = d.rideF * scale / 1000;

  return (
    <svg width={S.w} height={S.h} style={{ background: C.bg, borderRadius: 8, border: `1px solid ${C.b1}` }}>
      {/* Grid */}
      {Array.from({ length: 11 }, (_, i) => (
        <line key={`gv${i}`} x1={i * S.w / 10} y1={0} x2={i * S.w / 10} y2={S.h} stroke={C.b1} strokeWidth={0.3} />
      ))}

      {/* Ground */}
      <line x1={20} y1={ground} x2={S.w - 20} y2={ground} stroke={C.dm} strokeWidth={1.5} />

      {/* Title */}
      <text x={12} y={16} fill={C.cy} fontSize={9} fontWeight="bold" fontFamily={C.dt}>SIDE VIEW</text>
      <text x={12} y={26} fill={C.dm} fontSize={7} fontFamily={C.dt}>Looking from left</text>

      {/* Front wheel */}
      <circle cx={frontX} cy={ground - wheelR} r={wheelR} fill={`${C.am}08`} stroke={C.am} strokeWidth={1.5} />
      <text x={frontX} y={ground + 14} fill={C.am} fontSize={7} textAnchor="middle" fontFamily={C.dt}>FRONT</text>

      {/* Rear wheel */}
      <circle cx={rearX} cy={ground - wheelR} r={wheelR} fill={`${C.am}08`} stroke={C.am} strokeWidth={1.5} />
      <text x={rearX} y={ground + 14} fill={C.am} fontSize={7} textAnchor="middle" fontFamily={C.dt}>REAR</text>

      {/* Chassis link */}
      <line x1={rearX} y1={ground - wheelR - rideH - 5}
        x2={frontX} y2={ground - wheelR - rideH - 5}
        stroke={C.cy} strokeWidth={2} strokeLinecap="round" />

      {/* CG */}
      <circle cx={cgX} cy={cgY} r={5} fill="none" stroke={C.red} strokeWidth={1.5} />
      <line x1={cgX - 4} y1={cgY} x2={cgX + 4} y2={cgY} stroke={C.red} strokeWidth={1} />
      <line x1={cgX} y1={cgY - 4} x2={cgX} y2={cgY + 4} stroke={C.red} strokeWidth={1} />
      <text x={cgX + 10} y={cgY + 3} fill={C.red} fontSize={7} fontFamily={C.dt}>CG {(d.hCG * 1000).toFixed(0)}mm</text>

      {/* Caster axis */}
      {(() => {
        const casterRad = (d.casterF * Math.PI) / 180;
        const axLen = wheelR * 1.8;
        const cx2 = frontX + Math.sin(casterRad) * axLen;
        const cy2 = ground - wheelR - Math.cos(casterRad) * axLen;
        const isFz = frozen?.castor_f;
        return (
          <g>
            <line x1={frontX - Math.sin(casterRad) * 10} y1={ground - wheelR + Math.cos(casterRad) * 10}
              x2={cx2} y2={cy2}
              stroke={isFz ? C.red : C.pr} strokeWidth={1.5} strokeDasharray="4 2" />
            <text x={cx2 + 5} y={cy2 + 3} fill={isFz ? C.red : C.pr} fontSize={7} fontFamily={C.dt} fontWeight="bold">
              Caster {d.casterF.toFixed(1)}°
            </text>
          </g>
        );
      })()}

      {/* Anti-dive front SVSA line */}
      {(() => {
        const adfRad = (d.adfDeg * Math.PI) / 180;
        const len = 80;
        const isFz = frozen?.anti_dive_f;
        return (
          <g>
            <line x1={frontX} y1={ground}
              x2={frontX - Math.cos(adfRad) * len} y2={ground - Math.sin(adfRad) * len}
              stroke={isFz ? `${C.red}80` : `${C.gn}80`} strokeWidth={1.5} strokeDasharray="6 3" />
            <text x={frontX - len * 0.5} y={ground - Math.sin(adfRad) * len * 0.5 - 6}
              fill={isFz ? C.red : C.gn} fontSize={7} fontFamily={C.dt} fontWeight="bold">
              Anti-dive F {d.antiDiveF.toFixed(0)}%
            </text>
          </g>
        );
      })()}

      {/* Anti-squat rear SVSA line */}
      {(() => {
        const asRad = (d.asDeg * Math.PI) / 180;
        const len = 80;
        const isFz = frozen?.anti_squat;
        return (
          <g>
            <line x1={rearX} y1={ground}
              x2={rearX + Math.cos(asRad) * len} y2={ground - Math.sin(asRad) * len}
              stroke={isFz ? `${C.red}80` : `${C.am}80`} strokeWidth={1.5} strokeDasharray="6 3" />
            <text x={rearX + len * 0.5} y={ground - Math.sin(asRad) * len * 0.5 - 6}
              fill={isFz ? C.red : C.am} fontSize={7} fontFamily={C.dt} fontWeight="bold">
              Anti-squat {d.antiSq.toFixed(0)}%
            </text>
          </g>
        );
      })()}

      {/* Wheelbase dimension */}
      <DimLine x1={rearX} y1={ground + 18} x2={frontX} y2={ground + 18}
        label={`WB ${(d.wb * 1000).toFixed(0)}mm`} color={C.dm} />
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TOP VIEW SCHEMATIC
// ═══════════════════════════════════════════════════════════════════════════
function TopView({ d, frozen }) {
  const cx = S.w / 2, cy = S.h / 2;
  const scaleX = 130, scaleY = 85;
  const halfTF = (d.tF / 2) * scaleY;
  const halfTR = (d.tR / 2) * scaleY;
  const frontX = cx + d.lf * scaleX * 0.6;
  const rearX  = cx - d.lr * scaleX * 0.6;
  const wheelL = 20, wheelW = 8;

  return (
    <svg width={S.w} height={S.h} style={{ background: C.bg, borderRadius: 8, border: `1px solid ${C.b1}` }}>
      {/* Grid */}
      {Array.from({ length: 11 }, (_, i) => (
        <line key={`gv${i}`} x1={i * S.w / 10} y1={0} x2={i * S.w / 10} y2={S.h} stroke={C.b1} strokeWidth={0.3} />
      ))}

      {/* Title */}
      <text x={12} y={16} fill={C.cy} fontSize={9} fontWeight="bold" fontFamily={C.dt}>TOP VIEW</text>
      <text x={12} y={26} fill={C.dm} fontSize={7} fontFamily={C.dt}>Looking down · front = right</text>

      {/* Chassis outline */}
      <polygon points={`${frontX + 15},${cy} ${frontX},${cy - halfTF * 0.5} ${rearX},${cy - halfTR * 0.5} ${rearX - 10},${cy} ${rearX},${cy + halfTR * 0.5} ${frontX},${cy + halfTF * 0.5}`}
        fill={`${C.cy}06`} stroke={C.cy} strokeWidth={0.8} />

      {/* Centerline */}
      <line x1={rearX - 20} y1={cy} x2={frontX + 30} y2={cy}
        stroke={C.dm} strokeWidth={0.5} strokeDasharray="4 4" />

      {/* Wheels with toe angle */}
      {[
        { x: frontX, y: cy - halfTF, toe: d.toeF, label: "FL", isFz: frozen?.toe_f },
        { x: frontX, y: cy + halfTF, toe: -d.toeF, label: "FR", isFz: frozen?.toe_f },
        { x: rearX,  y: cy - halfTR, toe: d.toeR, label: "RL", isFz: frozen?.toe_r },
        { x: rearX,  y: cy + halfTR, toe: -d.toeR, label: "RR", isFz: frozen?.toe_r },
      ].map(w => {
        const toeRad = (w.toe * Math.PI) / 180;
        return (
          <g key={w.label} transform={`translate(${w.x}, ${w.y}) rotate(${-w.toe})`}>
            <rect x={-wheelL / 2} y={-wheelW / 2} width={wheelL} height={wheelW}
              rx={2} fill={`${C.am}20`} stroke={w.isFz ? C.red : C.am} strokeWidth={1.5} />
            <line x1={0} y1={0} x2={wheelL * 0.8} y2={0}
              stroke={w.isFz ? C.red : C.gn} strokeWidth={0.8} />
            <text x={0} y={wheelW + 10} fill={C.dm} fontSize={7} textAnchor="middle"
              fontFamily={C.dt} transform={`rotate(${w.toe})`}>
              {w.label} {Math.abs(d.toeF).toFixed(2)}°
            </text>
          </g>
        );
      })}

      {/* Toe direction labels */}
      <text x={frontX + 35} y={cy - halfTF} fill={C.am} fontSize={7} fontFamily={C.dt}>
        {d.toeF < 0 ? "toe-in" : d.toeF > 0 ? "toe-out" : "parallel"}
      </text>

      {/* Forward arrow */}
      <polygon points={`${frontX + 40},${cy} ${frontX + 34},${cy - 4} ${frontX + 34},${cy + 4}`}
        fill={C.dm} />
      <text x={frontX + 45} y={cy + 3} fill={C.dm} fontSize={7} fontFamily={C.dt}>FWD</text>
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// DERIVED METRICS PANEL
// ═══════════════════════════════════════════════════════════════════════════
function MetricsPanel({ d }) {
  const zetaColor = (z) => z > 0.5 && z < 0.9 ? C.gn : z > 0.3 ? C.am : C.red;
  const freqColor = (f, target) => Math.abs(f - target) / target < 0.2 ? C.gn : C.am;

  const metrics = [
    { cat: "Natural Frequencies", items: [
      { label: "Heave", val: `${d.fnHeave.toFixed(2)} Hz`, target: "1.5–2.5", color: freqColor(d.fnHeave, 2.0) },
      { label: "Roll", val: `${d.fnRoll.toFixed(2)} Hz`, target: "1.2–2.0", color: freqColor(d.fnRoll, 1.5) },
      { label: "Pitch", val: `${d.fnPitch.toFixed(2)} Hz`, target: "1.5–2.5", color: freqColor(d.fnPitch, 2.0) },
      { label: "Wheel hop F", val: `${d.fnWheelF.toFixed(1)} Hz`, target: "10–15", color: C.br },
      { label: "Wheel hop R", val: `${d.fnWheelR.toFixed(1)} Hz`, target: "10–15", color: C.br },
    ]},
    { cat: "Damping Ratios", items: [
      { label: "ζ heave", val: d.zetaHeave.toFixed(3), target: "0.5–0.7", color: zetaColor(d.zetaHeave) },
      { label: "ζ roll", val: d.zetaRoll.toFixed(3), target: "0.6–0.8", color: zetaColor(d.zetaRoll) },
      { label: "ζ pitch", val: d.zetaPitch.toFixed(3), target: "0.5–0.7", color: zetaColor(d.zetaPitch) },
    ]},
    { cat: "Load Transfer", items: [
      { label: "LLTD front", val: `${(d.lltdF * 100).toFixed(1)}%`, target: "50–55%", color: d.lltdF > 0.48 && d.lltdF < 0.56 ? C.gn : C.red },
      { label: "LLTD rear", val: `${(d.lltdR * 100).toFixed(1)}%`, target: "45–50%", color: C.br },
      { label: "Balance", val: d.underOvr, target: "US preferred", color: d.underOvr === "US" ? C.gn : d.underOvr === "Neutral" ? C.am : C.red },
    ]},
    { cat: "Anti-Geometry SVSA", items: [
      { label: "Anti-squat", val: `${d.antiSq.toFixed(1)}%  (${d.asDeg.toFixed(1)}°)`, target: "20–50%", color: C.am },
      { label: "Anti-dive F", val: `${d.antiDiveF.toFixed(1)}%  (${d.adfDeg.toFixed(1)}°)`, target: "30–60%", color: C.gn },
      { label: "Anti-dive R", val: `${d.antiDiveR.toFixed(1)}%  (${d.adrDeg.toFixed(1)}°)`, target: "10–30%", color: C.br },
      { label: "Anti-lift", val: `${d.antiLift.toFixed(1)}%  (${d.alDeg.toFixed(1)}°)`, target: "15–30%", color: C.br },
    ]},
  ];

  return (
    <GC style={{ padding: "10px 14px" }}>
      <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 2, color: C.dm, fontFamily: C.dt, marginBottom: 8 }}>
        DERIVED QUANTITIES
      </div>
      {metrics.map(m => (
        <div key={m.cat} style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: C.cy, fontFamily: C.dt, letterSpacing: 1.5, marginBottom: 4 }}>
            {m.cat.toUpperCase()}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "120px 1fr 80px", gap: 0, fontSize: 8, fontFamily: C.dt }}>
            {m.items.map(i => (
              <React.Fragment key={i.label}>
                <div style={{ color: C.md, padding: "3px 0" }}>{i.label}</div>
                <div style={{ color: i.color, fontWeight: 700, padding: "3px 0" }}>{i.val}</div>
                <div style={{ color: C.dm, padding: "3px 0", fontSize: 7 }}>target: {i.target}</div>
              </React.Fragment>
            ))}
          </div>
        </div>
      ))}
    </GC>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT — embeds into ResearchModule as "Visualize" sub-tab
// ═══════════════════════════════════════════════════════════════════════════
export default function ResearchVisualizer({ values, frozen, car, carCfg }) {
  const [view, setView] = useState("front");
  const d = useMemo(() => computeDerived(values || {}, carCfg || { mass: 300, drivetrain: "RWD" }), [values, carCfg]);

  const VIEWS = [
    { key: "front", label: "Front View" },
    { key: "side",  label: "Side View" },
    { key: "top",   label: "Top View" },
    { key: "all",   label: "All Views" },
  ];

  return (
    <div>
      <div style={{ display: "flex", gap: 4, marginBottom: 10 }}>
        {VIEWS.map(v => (
          <Pill key={v.key} active={view === v.key} label={v.label}
            onClick={() => setView(v.key)} color={C.cy} />
        ))}
      </div>

      {(view === "front" || view === "all") && (
        <Sec title="Front View — Camber, Roll Center, Track Width">
          <FrontView d={d} frozen={frozen} />
        </Sec>
      )}
      {(view === "side" || view === "all") && (
        <Sec title="Side View — Caster, Anti-Dive, Anti-Squat SVSA">
          <SideView d={d} frozen={frozen} />
        </Sec>
      )}
      {(view === "top" || view === "all") && (
        <Sec title="Top View — Toe Angles, Steering Geometry">
          <TopView d={d} frozen={frozen} />
        </Sec>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 }}>
        <MetricsPanel d={d} />
        <GC style={{ padding: "10px 14px" }}>
          <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 2, color: C.dm, fontFamily: C.dt, marginBottom: 8 }}>
            PARAMETER COLOR KEY
          </div>
          <div style={{ fontSize: 8, fontFamily: C.dt, lineHeight: 2.2 }}>
            <span style={{ color: C.red, fontWeight: 700 }}>● RED</span>
            <span style={{ color: C.md }}> = Frozen (locked in CAD)</span><br />
            <span style={{ color: C.gn, fontWeight: 700 }}>● GREEN</span>
            <span style={{ color: C.md }}> = Free (optimizer will explore)</span><br />
            <span style={{ color: C.am, fontWeight: 700 }}>● AMBER</span>
            <span style={{ color: C.md }}> = Wheel / tire reference</span><br />
            <span style={{ color: C.cy, fontWeight: 700 }}>● CYAN</span>
            <span style={{ color: C.md }}> = Chassis / structural</span><br />
            <span style={{ color: C.pr, fontWeight: 700 }}>● PURPLE</span>
            <span style={{ color: C.md }}> = Roll center / kinematic</span>
          </div>
          <div style={{ marginTop: 12, padding: "8px 10px", background: `${C.cy}08`, borderRadius: 6, borderLeft: `3px solid ${C.cy}` }}>
            <div style={{ fontSize: 7, color: C.cy, fontWeight: 700, fontFamily: C.dt, letterSpacing: 1.5, marginBottom: 4 }}>
              WHAT THE SCHEMATICS SHOW
            </div>
            <div style={{ fontSize: 7, color: C.md, fontFamily: C.dt, lineHeight: 1.8 }}>
              These are quasi-static engineering schematics — they show instantaneous geometry
              at the current parameter values. Frozen parameters appear in red; adjust them
              in the Freeze Config tab and watch the geometry update in real-time.
              For dynamic animation (roll, pitch, maneuvers), use the Suspension 3D module.
            </div>
          </div>
        </GC>
      </div>
    </div>
  );
}