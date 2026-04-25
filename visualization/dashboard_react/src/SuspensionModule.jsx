// ═══════════════════════════════════════════════════════════════════════════
// src/SuspensionModule.jsx — Project-GP Dashboard v6.0
// ═══════════════════════════════════════════════════════════════════════════
// Optimum Kinematics–style suspension analysis workstation.
// SVG-based 2D kinematic schematics with force vector overlay,
// full kinematic sweep charts, live digital twin mode, and
// Bloomberg-terminal data density.
//
// Sub-views:
//   SCHEMATIC  — Front/side 2D SVG kinematic diagrams with force vectors
//   KINEMATICS — Camber, toe, bump steer, RC migration, MR vs heave
//   DYNAMICS   — Load transfer, damper force, spring energy, ARB torque
//   ANTI-GEOM  — Anti-squat/dive/lift pitch centre visualization
//   COMPLIANCE — Compliance steer, bushing deflection, flex analysis
//   MODAL      — Natural frequencies, damping ratios, mode shapes
//   SETUP      — Full 40-dim parameter table with sensitivity ∂lap/∂param
//
// LIVE mode: useLiveTelemetry hook feeds real-time suspension state
// ANALYZE mode: synthetic maneuver profiles (lap, skidpad, brake, accel)
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, ComposedChart, BarChart, Bar,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Legend, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import { C, GL, GS, TT, AX } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const SUS = "#00d4ff"; // suspension accent
const SUS_G = "rgba(0,212,255,0.06)";
const CL = { fl: C.cy, fr: C.gn, rl: C.am, rr: C.red };
const CN = { fl: "FL", fr: "FR", rl: "RL", rr: "RR" };
const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });
const tt = () => ({ contentStyle: { background: C.panel || "#0e1420", border: `1px solid ${C.b1}`, borderRadius: 4, fontSize: 9, fontFamily: C.dt, color: C.br }, itemStyle: { color: C.br, fontSize: 9 } });

// Seeded RNG
function srng(s) { return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; }; }

// ═══════════════════════════════════════════════════════════════════════════
// TER27 HARDPOINTS — Optimum Kinematics coordinate system
// X = longitudinal (fwd+), Y = lateral (left+), Z = vertical (up+)
// All values in millimetres for display, metres internally
// ═══════════════════════════════════════════════════════════════════════════

const HP = {
  front: {
    lca_f: { x: 160, y: 160, z: 110, label: "Lower Control Arm, Front" },
    lca_r: { x: -160, y: 160, z: 130, label: "Lower Control Arm, Rear" },
    uca_f: { x: 120, y: 245, z: 267, label: "Upper Control Arm, Front" },
    uca_r: { x: -120, y: 245, z: 258, label: "Upper Control Arm, Rear" },

    lca_o: { x: 2.27, y: 583.374, z: 122.65, label: "Upright Lower Balljoint" },
    uca_o: { x: -11.496, y: 555.63, z: 280, label: "Upright Upper Balljoint" },

    tie_i: { x: 50, y: 144.78, z: 144.5, label: "Tie Rod Inner" },
    tie_o: { x: 70, y: 571, z: 150, label: "Tie Rod Outer" },

    pushrod: { x: -3.51, y: 514.71, z: 294.18, label: "Pushrod Attachment" },
    chShock: { x: -179.33, y: 150, z: 614.79, label: "Chassis Shock Mount" },
    rockerPivot: { x: 0.67, y: 195.06, z: 575.18, label: "Rocker Pivot" },
    rockerRod: { x: 59.98, y: 201.85, z: 569.21, label: "Rocker Pushrod Mount" },
    rockerShock: { x: 0.67, y: 150, z: 614.79, label: "Rocker Shock Mount" },

    wc: { x: 0, y: 615, z: 228.6, label: "Wheel Centre" },
    cp: { x: 0, y: 615, z: 0, label: "Contact Patch" },
  },

  rear: {
    lca_f: { x: 150, y: 240, z: 126.2, label: "Lower Control Arm, Front" },
    lca_r: { x: -150, y: 240, z: 120, label: "Lower Control Arm, Rear" },
    uca_f: { x: 150, y: 240, z: 282, label: "Upper Control Arm, Front" },
    uca_r: { x: -150, y: 240, z: 250, label: "Upper Control Arm, Rear" },

    lca_o: { x: 0, y: 576.78, z: 112.65, label: "Upright Lower Balljoint" },
    uca_o: { x: 0, y: 520.001, z: 280, label: "Upright Upper Balljoint" },

    tie_i: { x: -95, y: 240, z: 163, label: "Tie Link Inner" },
    tie_o: { x: -80, y: 590, z: 165.8, label: "Tie Link Outer" },

    pushrod: { x: 8.93, y: 497.39, z: 297.58, label: "Pushrod Attachment" },
    chShock: { x: -30, y: 50, z: 430, label: "Chassis Shock Mount" },
    rockerPivot: { x: 107.43, y: 108.26, z: 547.13, label: "Rocker Pivot" },
    rockerRod: { x: 148.42, y: 144.1, z: 572.38, label: "Rocker Pushrod Mount" },
    rockerShock: { x: 97.28, y: 50, z: 557.28, label: "Rocker Shock Mount" },

    wc: { x: 0, y: 615, z: 228.6, label: "Wheel Centre" },
    cp: { x: 0, y: 615, z: 0, label: "Contact Patch" },
  },
};

// Vehicle geometry
const VG = {
  wb: 1550, lf: 852.5, lr: 697.5, tF: 1220, tR: 1180,
  hCG: 280, mass: 300, wR: 228.6,
  // Suspension params
  kF: 35000, kR: 52540, arbF: 200, arbR: 150,
  cLowF: 1800, cLowR: 1600, cHiF: 720, cHiR: 640,
  vKneeF: 0.10, vKneeR: 0.10, rebF: 1.6, rebR: 1.6,
  MR_F: 1.14, MR_R: 1.16,
  // Anti-geometry (Ter27)
  antiSquat: 0.35, antiSquatF: 0.15, antiDiveF: 0.35, antiDiveR: 0.15, antiLift: 0.20,
  // Alignment
  camberF: -2.5, camberR: -1.8, toeF: -0.08, toeR: 0.0, castorF: 5.5,
  camGainF: -0.75, camGainR: -0.60,
  // RC
  hRcF: 45, hRcR: 55, dhRcDzF: 0.22, dhRcDzR: 0.28,
};

// Setup parameter definitions (28-dim canonical + extended 40-dim)
const SETUP_PARAMS = [
  { key: "k_f", label: "Spring Rate Front", unit: "N/m", val: 35000, group: "springs", sens: -0.032 },
  { key: "k_r", label: "Spring Rate Rear", unit: "N/m", val: 52540, group: "springs", sens: -0.028 },
  { key: "arb_f", label: "ARB Front", unit: "N/m", val: 200, group: "springs", sens: -0.019 },
  { key: "arb_r", label: "ARB Rear", unit: "N/m", val: 150, group: "springs", sens: -0.015 },
  { key: "c_low_f", label: "LS Comp Front", unit: "Ns/m", val: 1800, group: "dampers", sens: -0.012 },
  { key: "c_low_r", label: "LS Comp Rear", unit: "Ns/m", val: 1600, group: "dampers", sens: -0.009 },
  { key: "c_hi_f", label: "HS Comp Front", unit: "Ns/m", val: 720, group: "dampers", sens: -0.005 },
  { key: "c_hi_r", label: "HS Comp Rear", unit: "Ns/m", val: 640, group: "dampers", sens: -0.004 },
  { key: "v_knee_f", label: "Knee Vel Front", unit: "m/s", val: 0.10, group: "dampers", sens: -0.003 },
  { key: "v_knee_r", label: "Knee Vel Rear", unit: "m/s", val: 0.10, group: "dampers", sens: -0.002 },
  { key: "reb_f", label: "Rebound Ratio F", unit: "—", val: 1.6, group: "dampers", sens: -0.006 },
  { key: "reb_r", label: "Rebound Ratio R", unit: "—", val: 1.6, group: "dampers", sens: -0.005 },
  { key: "h_ride_f", label: "Ride Height F", unit: "mm", val: 30, group: "geometry", sens: -0.041 },
  { key: "h_ride_r", label: "Ride Height R", unit: "mm", val: 30, group: "geometry", sens: -0.038 },
  { key: "camber_f", label: "Camber Front", unit: "deg", val: -2.5, group: "geometry", sens: -0.025 },
  { key: "camber_r", label: "Camber Rear", unit: "deg", val: -1.8, group: "geometry", sens: -0.018 },
  { key: "toe_f", label: "Toe Front", unit: "deg", val: -0.08, group: "geometry", sens: -0.008 },
  { key: "toe_r", label: "Toe Rear", unit: "deg", val: 0.0, group: "geometry", sens: -0.006 },
  { key: "castor", label: "Castor", unit: "deg", val: 5.5, group: "geometry", sens: -0.003 },
  { key: "anti_sq", label: "Anti-Squat", unit: "%", val: 35, group: "geometry", sens: -0.007 },
  { key: "anti_dive_f", label: "Anti-Dive F", unit: "%", val: 35, group: "geometry", sens: -0.004 },
  { key: "anti_dive_r", label: "Anti-Dive R", unit: "%", val: 15, group: "geometry", sens: -0.002 },
  { key: "anti_lift", label: "Anti-Lift", unit: "%", val: 20, group: "geometry", sens: -0.003 },
  { key: "diff_lock", label: "Diff Lock", unit: "—", val: 0.30, group: "drivetrain", sens: -0.011 },
  { key: "brake_bias", label: "Brake Bias F", unit: "%", val: 58, group: "brakes", sens: -0.022 },
  { key: "h_cg", label: "CG Height", unit: "mm", val: 285, group: "mass", sens: -0.045 },
  { key: "bs_f", label: "Bumpstop F", unit: "N/m", val: 150000, group: "springs", sens: -0.001 },
  { key: "bs_r", label: "Bumpstop R", unit: "N/m", val: 150000, group: "springs", sens: -0.001 },
  // Extended 40-dim (Ter27)
  { key: "k_heave_f", label: "Heave Spring F", unit: "N/mm", val: 0, group: "springs", sens: -0.002 },
  { key: "k_heave_r", label: "Heave Spring R", unit: "N/mm", val: 0, group: "springs", sens: -0.002 },
  { key: "inerter_f", label: "Inerter F", unit: "kg", val: 0, group: "springs", sens: -0.001 },
  { key: "inerter_r", label: "Inerter R", unit: "kg", val: 0, group: "springs", sens: -0.001 },
  { key: "mr_rise_f", label: "MR Rise Rate F", unit: "—", val: 2.8, group: "geometry", sens: -0.004 },
  { key: "mr_rise_r", label: "MR Rise Rate R", unit: "—", val: 2.2, group: "geometry", sens: -0.003 },
  { key: "anti_sq_f", label: "Anti-Squat F", unit: "%", val: 15, group: "geometry", sens: -0.005 },
  { key: "ackermann", label: "Ackermann", unit: "%", val: 0, group: "geometry", sens: -0.009 },
  { key: "c_ls_reb_f", label: "LS Reb Front", unit: "Ns/m", val: 2880, group: "dampers", sens: -0.006 },
  { key: "c_ls_reb_r", label: "LS Reb Rear", unit: "Ns/m", val: 2560, group: "dampers", sens: -0.005 },
  { key: "c_hs_reb_f", label: "HS Reb Front", unit: "Ns/m", val: 1152, group: "dampers", sens: -0.003 },
  { key: "c_hs_reb_r", label: "HS Reb Rear", unit: "Ns/m", val: 1024, group: "dampers", sens: -0.003 },
];

const PARAM_GROUPS = [
  { key: "springs", label: "Springs & ARBs", color: C.am },
  { key: "dampers", label: "4-Way Dampers", color: C.cy },
  { key: "geometry", label: "Geometry", color: C.gn },
  { key: "mass", label: "Mass & CG", color: "#a78bfa" },
  { key: "brakes", label: "Brakes", color: C.red },
  { key: "drivetrain", label: "Drivetrain", color: "#ff6090" },
];

// Sub-tab configuration
const TABS = [
  { key: "schematic", label: "Kinematic Schematic" },
  { key: "kinematics", label: "Kinematic Sweeps" },
  { key: "dynamics", label: "Dynamics & Forces" },
  { key: "antigeom", label: "Anti-Geometry" },
  { key: "compliance", label: "Compliance" },
  { key: "modal", label: "Modal Analysis" },
  { key: "setup", label: "Setup (40-dim)" },
];

// ═══════════════════════════════════════════════════════════════════════════
// KINEMATIC MATH — IC, RC, sweep computations
// ═══════════════════════════════════════════════════════════════════════════

function computeIC(lca_inner_z, lca_outer_z, uca_inner_z, uca_outer_z,
                    lca_inner_y, lca_outer_y, uca_inner_y, uca_outer_y) {
  // Instant centre from intersection of LCA and UCA lines (front view: Y-Z plane)
  const m1 = (lca_outer_z - lca_inner_z) / (lca_outer_y - lca_inner_y || 0.001);
  const b1 = lca_inner_z - m1 * lca_inner_y;
  const m2 = (uca_outer_z - uca_inner_z) / (uca_outer_y - uca_inner_y || 0.001);
  const b2 = uca_inner_z - m2 * uca_inner_y;
  if (Math.abs(m1 - m2) < 0.0001) return { y: 5000, z: lca_inner_z }; // parallel
  const y = (b2 - b1) / (m1 - m2);
  const z = m1 * y + b1;
  return { y, z };
}

function computeRC(icY, icZ, cpY) {
  // Roll centre: line from CP through IC → intersection at centreline (Y=0)
  if (Math.abs(icY - cpY) < 0.001) return { y: 0, z: icZ };
  const slope = (icZ - 0) / (icY - cpY);
  const rcZ = 0 - slope * (0 - cpY); // at Y=0
  return { y: 0, z: icZ + slope * (0 - icY) };
}

function generateKinSweep(axle, nPts = 41) {
  const hp = HP[axle];
  const isF = axle === "front";
  const kW = isF ? VG.kF : VG.kR;
  const mr0 = isF ? VG.MR_F : VG.MR_R;
  const mrRise = isF ? 2.8 : 2.2;
  const camGain = isF ? VG.camGainF : VG.camGainR;
  const camPerM = isF ? -22.0 : -18.0;
  const hRc0 = isF ? VG.hRcF : VG.hRcR;
  const dhRcDz = isF ? VG.dhRcDzF : VG.dhRcDzR;

  const data = [];
  for (let i = 0; i < nPts; i++) {
    const z_mm = -50 + (100 / (nPts - 1)) * i; // -50mm to +50mm
    const z_m = z_mm / 1000;
    const mr = mr0 + mrRise * z_m + 0 * z_m * z_m;
    const camber = (isF ? -2.5 : -1.8) + camPerM * z_m;
    const toe = (isF ? -0.08 : 0.0) + (isF ? 0.05 : 0.03) * z_m * z_m * 1000; // quadratic bump steer
    const rcH = hRc0 + dhRcDz * z_mm;
    const scrub = 5.0 + 0.8 * z_m * 1000; // mm
    const trackChange = -0.12 * z_m * 1000; // mm
    const caster = (isF ? 5.5 : 0) - 0.3 * z_m * 1000;
    const kpi = (isF ? 6.2 : 0) + 0.15 * z_m * 1000;
    const springF = kW * z_m * mr;
    data.push({
      z_mm: +z_mm.toFixed(1), camber: +camber.toFixed(3), toe: +toe.toFixed(4),
      mr: +mr.toFixed(4), rcH: +rcH.toFixed(1), scrub: +scrub.toFixed(2),
      trackChange: +trackChange.toFixed(2), caster: +caster.toFixed(2),
      kpi: +kpi.toFixed(2), springF: +springF.toFixed(0),
    });
  }
  return data;
}

function generateDynamicsData(maneuver = "lap", dur = 8, n = 480) {
  const r = srng(42);
  const data = [];
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * dur;
    const p = t / dur;
    let roll = 0, pitch = 0, ay = 0, ax = 0, spd = 12, steer = 0;
    switch (maneuver) {
      case "lap": {
        const q = p * 4 * Math.PI;
        ay = 1.4 * Math.sin(q * 0.4) + 0.5 * Math.sin(q * 1.1);
        ax = 0.8 * Math.cos(q * 0.6);
        roll = ay * 2.1; pitch = -ax * 1.5;
        steer = 0.26 * Math.sin(q * 0.4) + 0.10 * Math.sin(q * 1.1);
        spd = 14 + 6 * Math.sin(q * 0.3); break;
      }
      case "skidpad": {
        const e = p < 0.12 ? p / 0.12 : p > 0.88 ? (1 - p) / 0.12 : 1;
        ay = 1.5 * e; roll = 3.1 * e; steer = 0.22 * e; spd = 11; break;
      }
      case "brake": {
        const e = Math.sin(p * Math.PI);
        ax = -2.4 * e; pitch = 3.6 * e; spd = 25 * (1 - p * 0.75); break;
      }
      case "accel": {
        const e = p < 0.08 ? p / 0.08 : 1, d = Math.max(0, 1 - p * 0.55);
        ax = 1.8 * e * d; pitch = -2.4 * e * d; spd = 2 + 28 * p; break;
      }
      case "chicane": {
        const f = p * Math.PI * 6;
        ay = 1.8 * Math.sin(f); roll = 3.4 * Math.sin(f); steer = 0.28 * Math.sin(f);
        pitch = 0.3 * Math.sin(f * 2); ax = -0.35 * Math.cos(f * 0.5); spd = 15; break;
      }
    }

    const tAvg = (VG.tF + VG.tR) / 2;
    const latLT = ay * VG.mass * VG.hCG / tAvg * 9.81 / 1000;
    const lonLT = ax * VG.mass * VG.hCG / VG.wb * 9.81 / 1000;
    const Fz0 = VG.mass * 9.81 / 4;
    const Fz_FL = Fz0 - latLT / 2 + lonLT / 2;
    const Fz_FR = Fz0 + latLT / 2 + lonLT / 2;
    const Fz_RL = Fz0 - latLT / 2 - lonLT / 2;
    const Fz_RR = Fz0 + latLT / 2 - lonLT / 2;

    // Wheel travels from load transfer
    const zFL = (Fz_FL - Fz0) / VG.kF * 1000;
    const zFR = (Fz_FR - Fz0) / VG.kF * 1000;
    const zRL = (Fz_RL - Fz0) / VG.kR * 1000;
    const zRR = (Fz_RR - Fz0) / VG.kR * 1000;

    // Damper velocities (finite diff)
    const dt = dur / n;
    const vFL = i > 0 ? (zFL - (data[i - 1]?.zFL || 0)) / dt / 1000 : 0;
    const vFR = i > 0 ? (zFR - (data[i - 1]?.zFR || 0)) / dt / 1000 : 0;
    const vRL = i > 0 ? (zRL - (data[i - 1]?.zRL || 0)) / dt / 1000 : 0;
    const vRR = i > 0 ? (zRR - (data[i - 1]?.zRR || 0)) / dt / 1000 : 0;

    // Damper forces (digressive bilinear)
    const dF = (v, isF) => {
      const cL = isF ? VG.cLowF : VG.cLowR;
      const cH = isF ? VG.cHiF : VG.cHiR;
      const vk = isF ? VG.vKneeF : VG.vKneeR;
      const va = Math.abs(v);
      return Math.sign(v) * (cL * va / (1 + va / vk) + cH * va);
    };
    const Fd_FL = dF(vFL, true), Fd_FR = dF(vFR, true);
    const Fd_RL = dF(vRL, false), Fd_RR = dF(vRR, false);

    // Spring forces
    const Fs_FL = zFL / 1000 * VG.kF, Fs_FR = zFR / 1000 * VG.kF;
    const Fs_RL = zRL / 1000 * VG.kR, Fs_RR = zRR / 1000 * VG.kR;

    // Damper power dissipation
    const Pd_FL = Math.abs(Fd_FL * vFL), Pd_FR = Math.abs(Fd_FR * vFR);
    const Pd_RL = Math.abs(Fd_RL * vRL), Pd_RR = Math.abs(Fd_RR * vRR);

    // ARB torque
    const arbF = roll * Math.PI / 180 * VG.arbF;
    const arbR = roll * Math.PI / 180 * VG.arbR;

    // Camber from roll
    const camFL = VG.camberF + VG.camGainF * roll;
    const camFR = VG.camberF - VG.camGainF * roll;
    const camRL = VG.camberR + VG.camGainR * roll;
    const camRR = VG.camberR - VG.camGainR * roll;

    // LLTD
    const latLT_F = Math.abs(ay) * VG.mass * VG.hCG / VG.tF * 9.81 / 1000 * 0.52;
    const latLT_R = Math.abs(ay) * VG.mass * VG.hCG / VG.tR * 9.81 / 1000 * 0.48;
    const LLTD = Math.abs(latLT_F + latLT_R) > 1 ? (Math.abs(latLT_F) / (Math.abs(latLT_F) + Math.abs(latLT_R))) * 100 : 50;

    // Roll gradient
    const rollGrad = Math.abs(ay) > 0.05 ? Math.abs(roll / ay) : 0;

    // Friction utilisation
    const muPeak = 1.6;
    const frUtil = cn => {
      const Fy = Math.abs(ay) * VG.mass / 4 * 9.81;
      const Fx = Math.abs(ax) * VG.mass / 4 * 9.81;
      const Fz = cn === "fl" ? Fz_FL : cn === "fr" ? Fz_FR : cn === "rl" ? Fz_RL : Fz_RR;
      return Math.min(100, Math.sqrt(Fy * Fy + Fx * Fx) / (Math.max(50, Fz) * muPeak) * 100);
    };

    // Spring energy
    const Es = 0.5 * VG.kF * (zFL / 1000) ** 2 + 0.5 * VG.kF * (zFR / 1000) ** 2 +
               0.5 * VG.kR * (zRL / 1000) ** 2 + 0.5 * VG.kR * (zRR / 1000) ** 2;

    // RC migration (simplified)
    const rcF = VG.hRcF + VG.dhRcDzF * (zFL + zFR) / 2;
    const rcR = VG.hRcR + VG.dhRcDzR * (zRL + zRR) / 2;

    data.push({
      t: +t.toFixed(3), roll: +roll.toFixed(2), pitch: +pitch.toFixed(2),
      ay: +ay.toFixed(3), ax: +ax.toFixed(3), spd: +spd.toFixed(1),
      steer: +steer.toFixed(3),
      zFL: +zFL.toFixed(2), zFR: +zFR.toFixed(2), zRL: +zRL.toFixed(2), zRR: +zRR.toFixed(2),
      vFL: +vFL.toFixed(4), vFR: +vFR.toFixed(4), vRL: +vRL.toFixed(4), vRR: +vRR.toFixed(4),
      Fz_FL: +Fz_FL.toFixed(0), Fz_FR: +Fz_FR.toFixed(0), Fz_RL: +Fz_RL.toFixed(0), Fz_RR: +Fz_RR.toFixed(0),
      Fs_FL: +Fs_FL.toFixed(0), Fs_FR: +Fs_FR.toFixed(0), Fs_RL: +Fs_RL.toFixed(0), Fs_RR: +Fs_RR.toFixed(0),
      Fd_FL: +Fd_FL.toFixed(0), Fd_FR: +Fd_FR.toFixed(0), Fd_RL: +Fd_RL.toFixed(0), Fd_RR: +Fd_RR.toFixed(0),
      Pd_FL: +Pd_FL.toFixed(1), Pd_FR: +Pd_FR.toFixed(1), Pd_RL: +Pd_RL.toFixed(1), Pd_RR: +Pd_RR.toFixed(1),
      arbF: +arbF.toFixed(1), arbR: +arbR.toFixed(1),
      camFL: +camFL.toFixed(2), camFR: +camFR.toFixed(2), camRL: +camRL.toFixed(2), camRR: +camRR.toFixed(2),
      LLTD: +LLTD.toFixed(1), rollGrad: +rollGrad.toFixed(2),
      utilFL: +frUtil("fl").toFixed(1), utilFR: +frUtil("fr").toFixed(1),
      utilRL: +frUtil("rl").toFixed(1), utilRR: +frUtil("rr").toFixed(1),
      Es: +Es.toFixed(2), rcF: +rcF.toFixed(1), rcR: +rcR.toFixed(1),
    });
  }
  return data;
}

// ═══════════════════════════════════════════════════════════════════════════
// SVG KINEMATIC SCHEMATIC — Optimum Kinematics-style front view
// ═══════════════════════════════════════════════════════════════════════════

function KinematicSchematicSVG({ axle, heave, roll, liveForces }) {
  const hp = HP[axle];
  const isF = axle === "front";

  const svgW = 1400;
  const svgH = 420;
  const panelW = svgW / 2;
  const panelH = svgH;
  const scale = 0.42;
  const oY = panelH - 40;

  const Fz = liveForces?.Fz || (VG.mass * 9.81) / 4;
  const Fy = liveForces?.Fy || 0;
  const Fs = liveForces?.Fs || 0;
  const Fd = liveForces?.Fd || 0;
  const fScale = 0.04;

  const lcaC = "#00d4ff";
  const ucaC = "#00ff88";
  const tieC = "#ffaa00";
  const pushC = "#ff6090";
  const springC = "#00d4ff";
  const uprightC = "#8892a8";
  const forceC = "#ff4444";
  const rcColor = "#ff6090";

  function renderPanel(side, offsetX) {
    const sideSign = side === "left" ? 1 : -1;
    const cx = offsetX + panelW / 2;

    // Small visual roll effect so both sides do not look identical when animating.
    const motion = (heave || 0) + sideSign * (roll || 0) * 2.5;

    const toXY = (y_mm, z_mm) => ({
      x: cx - y_mm * scale,
      y: oY - z_mm * scale,
    });

    const toPt = (pt, zExtra = 0, moving = false) =>
      toXY(sideSign * pt.y, pt.z + zExtra + (moving ? motion : 0));

    const lcaI = toPt(hp.lca_f);
    const lcaO = toPt(hp.lca_o, 0, true);
    const ucaI = toPt(hp.uca_f);
    const ucaO = toPt(hp.uca_o, 0, true);

    const tieI = toPt(hp.tie_i);
    const tieO = toPt(hp.tie_o, 0, true);

    const pushA = toPt(hp.pushrod, 0, true);
    const chShock = toPt(hp.chShock);
    const rockerPivot = toPt(hp.rockerPivot);
    const rockerRod = toPt(hp.rockerRod);
    const rockerShock = toPt(hp.rockerShock);

    const wc = toPt(hp.wc, 0, true);
    const cp = toXY(sideSign * hp.cp.y, hp.cp.z);
    const ic = computeIC(
      hp.lca_f.z, hp.lca_o.z,
      hp.uca_f.z, hp.uca_o.z,
      sideSign * hp.lca_f.y, sideSign * hp.lca_o.y,
      sideSign * hp.uca_f.y, sideSign * hp.uca_o.y
    );
    const rc = computeRC(ic.y, ic.z, sideSign * hp.cp.y);
    const icSvg = toXY(ic.y, ic.z);
    const rcSvg = toXY(0, rc.z);

    return (
      <g key={side}>
        <rect
          x={offsetX}
          y={0}
          width={panelW}
          height={panelH}
          fill="transparent"
        />

        <line
          x1={offsetX + 20}
          y1={oY}
          x2={offsetX + panelW - 20}
          y2={oY}
          stroke={C.b2 || "#1a2035"}
          strokeWidth="1"
          strokeDasharray="4 3"
        />
        <line
          x1={cx}
          y1={10}
          x2={cx}
          y2={oY - 5}
          stroke={C.b1 || "#1a2035"}
          strokeWidth="0.5"
          strokeDasharray="2 4"
        />

        <text
          x={offsetX + 12}
          y={16}
          fill={C.w || "#e8eaf6"}
          fontSize="10"
          fontFamily="monospace"
          fontWeight="700"
        >
          {side.toUpperCase()} SIDE — {isF ? "FRONT" : "REAR"} VIEW
        </text>
        <text
          x={offsetX + 12}
          y={28}
          fill={C.dm || "#4a5568"}
          fontSize="7"
          fontFamily="monospace"
        >
          Track: {(isF ? VG.tF : VG.tR)}mm | Heave: {(heave || 0).toFixed(1)}mm | Roll: {(roll || 0).toFixed(2)}°
        </text>

        <line
          x1={lcaI.x}
          y1={lcaI.y}
          x2={lcaO.x}
          y2={lcaO.y}
          stroke={lcaC}
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx={lcaI.x} cy={lcaI.y} r="4" fill={C.bg || "#0a0a0f"} stroke={lcaC} strokeWidth="1.5" />
        <circle cx={lcaO.x} cy={lcaO.y} r="4" fill={C.bg || "#0a0a0f"} stroke={lcaC} strokeWidth="1.5" />

        <line
          x1={ucaI.x}
          y1={ucaI.y}
          x2={ucaO.x}
          y2={ucaO.y}
          stroke={ucaC}
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx={ucaI.x} cy={ucaI.y} r="4" fill={C.bg || "#0a0a0f"} stroke={ucaC} strokeWidth="1.5" />
        <circle cx={ucaO.x} cy={ucaO.y} r="4" fill={C.bg || "#0a0a0f"} stroke={ucaC} strokeWidth="1.5" />

        <line
          x1={tieI.x}
          y1={tieI.y}
          x2={tieO.x}
          y2={tieO.y}
          stroke={tieC}
          strokeWidth="2"
          strokeDasharray="6 3"
        />
        <circle cx={tieI.x} cy={tieI.y} r="3" fill={tieC} opacity="0.7" />
        <circle cx={tieO.x} cy={tieO.y} r="3" fill={tieC} opacity="0.7" />

        <line
          x1={lcaO.x}
          y1={lcaO.y}
          x2={ucaO.x}
          y2={ucaO.y}
          stroke={uprightC}
          strokeWidth="4"
          strokeLinecap="round"
          opacity="0.8"
        />

        <line
          x1={pushA.x}
          y1={pushA.y}
          x2={rockerRod.x}
          y2={rockerRod.y}
          stroke={pushC}
          strokeWidth="2.5"
          strokeLinecap="round"
        />

        <line
          x1={rockerPivot.x}
          y1={rockerPivot.y}
          x2={rockerRod.x}
          y2={rockerRod.y}
          stroke="#a78bfa"
          strokeWidth="2.2"
        />
        <line
          x1={rockerPivot.x}
          y1={rockerPivot.y}
          x2={rockerShock.x}
          y2={rockerShock.y}
          stroke={springC}
          strokeWidth="2.5"
          strokeDasharray="4 2"
        />
        <line
          x1={rockerShock.x}
          y1={rockerShock.y}
          x2={chShock.x}
          y2={chShock.y}
          stroke={springC}
          strokeWidth="2.2"
          strokeDasharray="4 2"
        />
        <circle cx={rockerPivot.x} cy={rockerPivot.y} r="5" fill="#a78bfa" opacity="0.5" />
        <circle cx={rockerRod.x} cy={rockerRod.y} r="4" fill="#a78bfa" opacity="0.5" />
        <rect
          x={rockerShock.x - 6}
          y={rockerShock.y - 3}
          width="12"
          height="6"
          rx="2"
          fill={C.bg || "#0a0a0f"}
          stroke={springC}
          strokeWidth="1"
        />
        <circle cx={chShock.x} cy={chShock.y} r="4" fill={springC} opacity="0.8" />

        <circle
          cx={wc.x}
          cy={wc.y}
          r={VG.wR * scale * 0.9}
          fill="none"
          stroke={C.dm || "#4a5568"}
          strokeWidth="1.5"
          opacity="0.4"
        />
        <circle cx={wc.x} cy={wc.y} r="3" fill={C.dm || "#4a5568"} />
        <rect
          x={cp.x - 12}
          y={cp.y - 2}
          width="24"
          height="4"
          rx="2"
          fill={C.gn || "#00ff88"}
          opacity="0.5"
        />

        <line
          x1={lcaI.x}
          y1={lcaI.y}
          x2={icSvg.x}
          y2={icSvg.y}
          stroke={lcaC}
          strokeWidth="0.5"
          strokeDasharray="3 3"
          opacity="0.35"
        />
        <line
          x1={ucaI.x}
          y1={ucaI.y}
          x2={icSvg.x}
          y2={icSvg.y}
          stroke={ucaC}
          strokeWidth="0.5"
          strokeDasharray="3 3"
          opacity="0.35"
        />

        <circle cx={icSvg.x} cy={icSvg.y} r="5" fill="none" stroke={rcColor} strokeWidth="1.5" />
        <line x1={icSvg.x - 4} y1={icSvg.y} x2={icSvg.x + 4} y2={icSvg.y} stroke={rcColor} strokeWidth="1" />
        <line x1={icSvg.x} y1={icSvg.y - 4} x2={icSvg.x} y2={icSvg.y + 4} stroke={rcColor} strokeWidth="1" />
        <text x={icSvg.x + 8} y={icSvg.y - 4} fill={rcColor} fontSize="7" fontFamily="monospace" fontWeight="700">
          IC
        </text>

        <line
          x1={cp.x}
          y1={cp.y}
          x2={icSvg.x}
          y2={icSvg.y}
          stroke={rcColor}
          strokeWidth="0.7"
          strokeDasharray="4 2"
          opacity="0.5"
        />
        <line
          x1={icSvg.x}
          y1={icSvg.y}
          x2={rcSvg.x}
          y2={rcSvg.y}
          stroke={rcColor}
          strokeWidth="0.7"
          strokeDasharray="4 2"
          opacity="0.5"
        />

        <circle cx={rcSvg.x} cy={rcSvg.y} r="6" fill={rcColor} opacity="0.3" />
        <circle cx={rcSvg.x} cy={rcSvg.y} r="3" fill={rcColor} />
        <text x={rcSvg.x + 10} y={rcSvg.y + 3} fill={rcColor} fontSize="8" fontFamily="monospace" fontWeight="700">
          RC
        </text>

        {Fz > 0 && (
          <line
            x1={cp.x}
            y1={cp.y}
            x2={cp.x}
            y2={cp.y + Math.min(80, Fz * fScale)}
            stroke={forceC}
            strokeWidth="2"
            markerEnd="url(#arrowF)"
            opacity="0.8"
          />
        )}
        {Fz > 0 && (
          <text
            x={cp.x + 6}
            y={cp.y + Math.min(60, Fz * fScale * 0.7)}
            fill={forceC}
            fontSize="7"
            fontFamily="monospace"
          >
            Fz={Fz.toFixed(0)}N
          </text>
        )}

        {Math.abs(Fy) > 10 && (
          <line
            x1={cp.x}
            y1={cp.y - 3}
            x2={cp.x + Math.sign(Fy) * Math.min(60, Math.abs(Fy) * fScale)}
            y2={cp.y - 3}
            stroke={C.am || "#ffaa00"}
            strokeWidth="2"
            markerEnd="url(#arrowF)"
            opacity="0.8"
          />
        )}

        {Math.abs(Fs) > 10 && (
          <line
            x1={rockerShock.x}
            y1={rockerShock.y}
            x2={rockerShock.x}
            y2={rockerShock.y + Math.sign(Fs) * Math.min(40, Math.abs(Fs) * fScale * 0.5)}
            stroke={springC}
            strokeWidth="1.5"
            markerEnd="url(#arrowS)"
            opacity="0.7"
          />
        )}

        <text x={lcaI.x - 2} y={lcaI.y + 12} fill={lcaC} fontSize="6" fontFamily="monospace" textAnchor="end">
          LCA
        </text>
        <text x={ucaI.x - 2} y={ucaI.y - 6} fill={ucaC} fontSize="6" fontFamily="monospace" textAnchor="end">
          UCA
        </text>
        <text x={tieI.x - 2} y={tieI.y + 10} fill={tieC} fontSize="6" fontFamily="monospace" textAnchor="end">
          TIE
        </text>
        <text x={pushA.x + 6} y={pushA.y - 4} fill={pushC} fontSize="6" fontFamily="monospace">
          PUSH
        </text>
        <text x={rockerPivot.x + 6} y={rockerPivot.y - 4} fill="#a78bfa" fontSize="6" fontFamily="monospace">
          ROCKER
        </text>
        <text x={chShock.x + 6} y={chShock.y - 4} fill={springC} fontSize="6" fontFamily="monospace">
          SHOCK
        </text>

        <g transform={`translate(${offsetX + panelW - 130}, 12)`}>
          {[
            { c: lcaC, l: "Lower Control Arm" },
            { c: ucaC, l: "Upper Control Arm" },
            { c: tieC, l: "Tie Rod" },
            { c: pushC, l: "Pushrod" },
            { c: rcColor, l: "IC / Roll Centre" },
            { c: forceC, l: "Force Vectors" },
          ].map((item, i) => (
            <g key={i} transform={`translate(0, ${i * 11})`}>
              <line x1="0" y1="4" x2="12" y2="4" stroke={item.c} strokeWidth="2" />
              <text x="16" y="7" fill={C.dm || "#4a5568"} fontSize="6.5" fontFamily="monospace">
                {item.l}
              </text>
            </g>
          ))}
        </g>
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} style={{ width: "100%", height: "100%", background: "transparent" }}>
      <defs>
        <marker id="arrowF" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" fill={forceC} />
        </marker>
        <marker id="arrowS" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" fill={springC} />
        </marker>
      </defs>

      {renderPanel("left", 0)}
      <line x1={panelW} y1="8" x2={panelW} y2={svgH - 8} stroke={C.b1 || "#1a2035"} strokeWidth="1" opacity="0.7" />
      {renderPanel("right", panelW)}
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// HARDPOINT TABLE — Bloomberg-style data panel
// ═══════════════════════════════════════════════════════════════════════════

function HardpointTable({ axle }) {
  const hp = HP[axle];
  const entries = Object.entries(hp);
  return (
    <div style={{ ...GL, padding: "8px 10px", maxHeight: 280, overflowY: "auto" }}>
      <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6 }}>
        HARDPOINTS — {axle.toUpperCase()} [{entries.length} POINTS]
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 8, fontFamily: C.dt }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${C.b1}` }}>
            <th style={{ textAlign: "left", color: C.dm, padding: "2px 4px", fontWeight: 600 }}>Point</th>
            <th style={{ textAlign: "right", color: C.dm, padding: "2px 4px" }}>X (mm)</th>
            <th style={{ textAlign: "right", color: C.dm, padding: "2px 4px" }}>Y (mm)</th>
            <th style={{ textAlign: "right", color: C.dm, padding: "2px 4px" }}>Z (mm)</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, pt]) => (
            <tr key={key} style={{ borderBottom: `1px solid ${C.b1}22` }}>
              <td style={{ padding: "2px 4px", color: C.br, fontWeight: 500 }}>{pt.label || key}</td>
              <td style={{ textAlign: "right", padding: "2px 4px", color: C.cy }}>{pt.x?.toFixed(1)}</td>
              <td style={{ textAlign: "right", padding: "2px 4px", color: C.gn }}>{pt.y?.toFixed(1)}</td>
              <td style={{ textAlign: "right", padding: "2px 4px", color: C.am }}>{pt.z?.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SCHEMATIC TAB — SVG kinematics + hardpoint table + live KPIs
// ═══════════════════════════════════════════════════════════════════════════

function SchematicTab({ dynamics, frame }) {
  const [axle, setAxle] = useState("front");
  const [animIdx, setAnimIdx] = useState(0);

  useEffect(() => {
    if (!dynamics.length) return;
    const iv = setInterval(() => setAnimIdx(i => (i + 1) % dynamics.length), 33);
    return () => clearInterval(iv);
  }, [dynamics]);

  const d = frame || dynamics[animIdx] || {};
  const heave = axle === "front" ? ((d.zFL || 0) + (d.zFR || 0)) / 2 : ((d.zRL || 0) + (d.zRR || 0)) / 2;
  const Fz = axle === "front" ? ((d.Fz_FL || 735) + (d.Fz_FR || 735)) / 2 : ((d.Fz_RL || 735) + (d.Fz_RR || 735)) / 2;
  const Fy = (d.ay || 0) * VG.mass / 4 * 9.81;
  const Fs = axle === "front" ? ((d.Fs_FL || 0) + (d.Fs_FR || 0)) / 2 : ((d.Fs_RL || 0) + (d.Fs_RR || 0)) / 2;
  const Fd = axle === "front" ? ((d.Fd_FL || 0) + (d.Fd_FR || 0)) / 2 : ((d.Fd_RL || 0) + (d.Fd_RR || 0)) / 2;

  const ic = computeIC(
    HP[axle].lca_f.z, HP[axle].lca_o.z, HP[axle].uca_f.z, HP[axle].uca_o.z,
    HP[axle].lca_f.y, HP[axle].lca_o.y, HP[axle].uca_f.y, HP[axle].uca_o.y
  );
  const rcZ = computeRC(ic.y, ic.z, HP[axle].cp.y).z;

  return (
    <div>
      {/* Axle selector + live KPI strip */}
      <div style={{ display: "flex", gap: 6, marginBottom: 8, alignItems: "center" }}>
        {["front", "rear"].map(a => (
          <button key={a} onClick={() => setAxle(a)} style={{
            background: axle === a ? `${SUS}18` : "transparent",
            border: `1px solid ${axle === a ? `${SUS}55` : C.b1}`,
            color: axle === a ? SUS : C.dm,
            fontSize: 9, fontWeight: 700, letterSpacing: 1.2,
            padding: "4px 14px", borderRadius: 4, cursor: "pointer", fontFamily: C.dt,
          }}>{a.toUpperCase()}</button>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", gap: 12 }}>
          {[
            { l: "IC Height", v: `${ic.z.toFixed(0)}mm`, c: "#ff6090" },
            { l: "RC Height", v: `${rcZ.toFixed(0)}mm`, c: "#ff6090" },
            { l: "MR", v: (axle === "front" ? VG.MR_F : VG.MR_R).toFixed(3), c: C.cy },
            { l: "Camber", v: `${(axle === "front" ? VG.camberF : VG.camberR).toFixed(1)}°`, c: C.gn },
            { l: "Heave", v: `${heave.toFixed(1)}mm`, c: C.am },
            { l: "Roll", v: `${(d.roll || 0).toFixed(2)}°`, c: C.red },
          ].map(k => (
            <div key={k.l} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, letterSpacing: 1 }}>{k.l}</div>
              <div style={{ fontSize: 11, color: k.c, fontFamily: C.dt, fontWeight: 700 }}>{k.v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Main content: SVG + hardpoint table side by side */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 10 }}>
        {/* SVG Kinematic Schematic */}
        <div style={{ ...GL, padding: 6, minHeight: 380 }}>
          <KinematicSchematicSVG axle={axle} heave={heave} roll={d.roll}
            liveForces={{ Fz, Fy, Fs, Fd }} />
        </div>

        {/* Right panel: hardpoints + force summary */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <HardpointTable axle={axle} />

          {/* Force summary panel */}
          <div style={{ ...GL, padding: "8px 10px" }}>
            <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6 }}>
              FORCE SUMMARY
            </div>
            {[
              { l: "Fz (avg)", v: `${Fz.toFixed(0)} N`, c: C.red },
              { l: "Fy (lateral)", v: `${Fy.toFixed(0)} N`, c: C.am },
              { l: "Fs (spring)", v: `${Fs.toFixed(0)} N`, c: C.cy },
              { l: "Fd (damper)", v: `${Fd.toFixed(0)} N`, c: C.gn },
              { l: "Roll Moment", v: `${((d.ay || 0) * VG.mass * 9.81 * VG.hCG / 1000).toFixed(1)} Nm`, c: "#a78bfa" },
              { l: "ARB F", v: `${(d.arbF || 0).toFixed(1)} Nm`, c: C.am },
              { l: "LLTD", v: `${(d.LLTD || 50).toFixed(1)}%`, c: C.cy },
            ].map(f => (
              <div key={f.l} style={{ display: "flex", justifyContent: "space-between", fontSize: 8, fontFamily: C.dt, marginBottom: 2, padding: "1px 0" }}>
                <span style={{ color: C.dm }}>{f.l}</span>
                <span style={{ color: f.c, fontWeight: 600 }}>{f.v}</span>
              </div>
            ))}
          </div>

          {/* Steer axis geometry */}
          <div style={{ ...GL, padding: "8px 10px" }}>
            <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6 }}>
              STEER AXIS GEOMETRY
            </div>
            {[
              { l: "KPI", v: axle === "front" ? "6.2°" : "—" },
              { l: "Castor", v: axle === "front" ? `${VG.castorF}°` : "—" },
              { l: "Mech Trail", v: axle === "front" ? "18.5mm" : "—" },
              { l: "Scrub Radius", v: "5.0mm" },
              { l: "Ackermann", v: `${VG.ackermann || 0}%` },
            ].map(f => (
              <div key={f.l} style={{ display: "flex", justifyContent: "space-between", fontSize: 8, fontFamily: C.dt, marginBottom: 1.5 }}>
                <span style={{ color: C.dm }}>{f.l}</span>
                <span style={{ color: C.br, fontWeight: 500 }}>{f.v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// KINEMATICS TAB — Sweep charts (camber, toe, RC, MR vs heave)
// ═══════════════════════════════════════════════════════════════════════════

function KinematicsTab() {
  const frontSweep = useMemo(() => generateKinSweep("front"), []);
  const rearSweep = useMemo(() => generateKinSweep("rear"), []);

  const CH = ({ title, children }) => (
    <div style={{ ...GL, padding: "10px 8px 6px" }}>
      <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6, paddingLeft: 2 }}>{title}</div>
      {children}
    </div>
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      {/* Camber vs Heave */}
      <CH title="CAMBER vs HEAVE">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="z_mm" type="number" domain={[-50, 50]} {...ax()} label={{ value: "Heave [mm]", fontSize: 7, fill: C.dm, position: "bottom" }} />
            <YAxis {...ax()} label={{ value: "Camber [°]", fontSize: 7, fill: C.dm, angle: -90, position: "insideLeft" }} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line data={frontSweep} dataKey="camber" stroke={C.cy} dot={false} strokeWidth={2} name="Front" />
            <Line data={rearSweep} dataKey="camber" stroke={C.am} dot={false} strokeWidth={2} name="Rear" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      {/* Toe (Bump Steer) vs Heave */}
      <CH title="TOE / BUMP STEER vs HEAVE">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="z_mm" type="number" domain={[-50, 50]} {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line data={frontSweep} dataKey="toe" stroke={C.cy} dot={false} strokeWidth={2} name="Front" />
            <Line data={rearSweep} dataKey="toe" stroke={C.am} dot={false} strokeWidth={2} name="Rear" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      {/* Roll Centre Height vs Heave */}
      <CH title="ROLL CENTRE HEIGHT vs HEAVE">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="z_mm" type="number" domain={[-50, 50]} {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line data={frontSweep} dataKey="rcH" stroke={C.cy} dot={false} strokeWidth={2} name="Front RC" />
            <Line data={rearSweep} dataKey="rcH" stroke={C.am} dot={false} strokeWidth={2} name="Rear RC" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      {/* Motion Ratio vs Heave */}
      <CH title="MOTION RATIO vs HEAVE">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="z_mm" type="number" domain={[-50, 50]} {...ax()} />
            <YAxis domain={["auto", "auto"]} {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line data={frontSweep} dataKey="mr" stroke={C.cy} dot={false} strokeWidth={2} name="Front MR" />
            <Line data={rearSweep} dataKey="mr" stroke={C.am} dot={false} strokeWidth={2} name="Rear MR" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      {/* Caster & KPI */}
      <CH title="CASTER & KPI vs HEAVE (FRONT)">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={frontSweep} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="z_mm" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line dataKey="caster" stroke={C.gn} dot={false} strokeWidth={2} name="Castor [°]" />
            <Line dataKey="kpi" stroke="#a78bfa" dot={false} strokeWidth={2} name="KPI [°]" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      {/* Track Change & Scrub */}
      <CH title="TRACK CHANGE & SCRUB RADIUS vs HEAVE">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="z_mm" type="number" domain={[-50, 50]} {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line data={frontSweep} dataKey="trackChange" stroke={C.cy} dot={false} strokeWidth={2} name="Front ΔTrack" />
            <Line data={rearSweep} dataKey="trackChange" stroke={C.am} dot={false} strokeWidth={2} name="Rear ΔTrack" />
            <Line data={frontSweep} dataKey="scrub" stroke={C.gn} dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="Front Scrub" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// DYNAMICS TAB — Load transfer, damper force, spring energy, wheel travel
// ═══════════════════════════════════════════════════════════════════════════

function DynamicsTab({ dynamics }) {
  const ds = useMemo(() => dynamics.filter((_, i) => i % 3 === 0), [dynamics]);
  const CH = ({ title, children }) => (
    <div style={{ ...GL, padding: "10px 8px 6px" }}>
      <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6, paddingLeft: 2 }}>{title}</div>
      {children}
    </div>
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <CH title="VERTICAL LOAD Fz — PER CORNER">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <Line dataKey="Fz_FL" stroke={CL.fl} dot={false} strokeWidth={1.5} name="FL" />
            <Line dataKey="Fz_FR" stroke={CL.fr} dot={false} strokeWidth={1.5} name="FR" />
            <Line dataKey="Fz_RL" stroke={CL.rl} dot={false} strokeWidth={1.5} name="RL" />
            <Line dataKey="Fz_RR" stroke={CL.rr} dot={false} strokeWidth={1.5} name="RR" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      <CH title="WHEEL TRAVEL — PER CORNER [mm]">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine y={0} stroke={C.b2} />
            <Line dataKey="zFL" stroke={CL.fl} dot={false} strokeWidth={1.5} name="FL" />
            <Line dataKey="zFR" stroke={CL.fr} dot={false} strokeWidth={1.5} name="FR" />
            <Line dataKey="zRL" stroke={CL.rl} dot={false} strokeWidth={1.5} name="RL" />
            <Line dataKey="zRR" stroke={CL.rr} dot={false} strokeWidth={1.5} name="RR" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      <CH title="DAMPER FORCE — PER CORNER [N]">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine y={0} stroke={C.b2} />
            <Line dataKey="Fd_FL" stroke={CL.fl} dot={false} strokeWidth={1.5} name="FL" />
            <Line dataKey="Fd_FR" stroke={CL.fr} dot={false} strokeWidth={1.5} name="FR" />
            <Line dataKey="Fd_RL" stroke={CL.rl} dot={false} strokeWidth={1.5} name="RL" />
            <Line dataKey="Fd_RR" stroke={CL.rr} dot={false} strokeWidth={1.5} name="RR" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      <CH title="DAMPER POWER DISSIPATION [W]">
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <Area dataKey="Pd_FL" stroke={CL.fl} fill={CL.fl} fillOpacity={0.15} strokeWidth={1.5} name="FL" />
            <Area dataKey="Pd_FR" stroke={CL.fr} fill={CL.fr} fillOpacity={0.15} strokeWidth={1.5} name="FR" />
            <Area dataKey="Pd_RL" stroke={CL.rl} fill={CL.rl} fillOpacity={0.15} strokeWidth={1.5} name="RL" />
            <Area dataKey="Pd_RR" stroke={CL.rr} fill={CL.rr} fillOpacity={0.15} strokeWidth={1.5} name="RR" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </AreaChart>
        </ResponsiveContainer>
      </CH>

      <CH title="ROLL & PITCH ANGLE [°]">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine y={0} stroke={C.b2} />
            <Line dataKey="roll" stroke={C.am} dot={false} strokeWidth={2} name="Roll [°]" />
            <Line dataKey="pitch" stroke={C.cy} dot={false} strokeWidth={2} name="Pitch [°]" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      <CH title="LLTD & ROLL GRADIENT">
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis yAxisId="l" {...ax()} />
            <YAxis yAxisId="r" orientation="right" {...ax()} />
            <Tooltip {...tt()} />
            <Line yAxisId="l" dataKey="LLTD" stroke={C.cy} dot={false} strokeWidth={2} name="LLTD [%]" />
            <Line yAxisId="r" dataKey="rollGrad" stroke={C.am} dot={false} strokeWidth={2} name="Roll Grad [°/G]" />
            <ReferenceLine yAxisId="l" y={50} stroke={C.b2} strokeDasharray="3 3" label={{ value: "50%", fontSize: 7, fill: C.dm }} />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </ComposedChart>
        </ResponsiveContainer>
      </CH>

      <CH title="CAMBER — PER CORNER [°]">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <Line dataKey="camFL" stroke={CL.fl} dot={false} strokeWidth={1.5} name="FL" />
            <Line dataKey="camFR" stroke={CL.fr} dot={false} strokeWidth={1.5} name="FR" />
            <Line dataKey="camRL" stroke={CL.rl} dot={false} strokeWidth={1.5} name="RL" />
            <Line dataKey="camRR" stroke={CL.rr} dot={false} strokeWidth={1.5} name="RR" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      <CH title="ROLL CENTRE MIGRATION [mm]">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ds} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="t" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <Line dataKey="rcF" stroke={C.cy} dot={false} strokeWidth={2} name="Front RC [mm]" />
            <Line dataKey="rcR" stroke={C.am} dot={false} strokeWidth={2} name="Rear RC [mm]" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// ANTI-GEOMETRY TAB — Anti-squat/dive/lift pitch centre visualization
// ═══════════════════════════════════════════════════════════════════════════

function AntiGeomTab() {
  const antiData = [
    { param: "Anti-Squat (R)", value: VG.antiSquat * 100, target: 35, color: C.cy },
    { param: "Anti-Squat (F)", value: VG.antiSquatF * 100, target: 15, color: C.gn },
    { param: "Anti-Dive (F)", value: VG.antiDiveF * 100, target: 35, color: C.am },
    { param: "Anti-Dive (R)", value: VG.antiDiveR * 100, target: 15, color: C.red },
    { param: "Anti-Lift", value: VG.antiLift * 100, target: 20, color: "#a78bfa" },
  ];

  // Pitch centre diagram (SVG side view)
  const svgW = 700, svgH = 300;
  const scale = 0.3;
  const oX = 100, oY = svgH - 50;
  const toS = (x, z) => ({ x: oX + x * scale, y: oY - z * scale });

  const fcpF = toS(VG.lf, 0); // front contact patch
  const fcpR = toS(-VG.lr, 0); // rear contact patch
  const cgP = toS(0, VG.hCG);
  const asLine = VG.antiSquat; // fraction
  const adLine = VG.antiDiveF;
  const asZ = VG.hCG * asLine;
  const adZ = VG.hCG * adLine;
  const pcF = toS(VG.lf, adZ); // anti-dive intersection at front axle
  const pcR = toS(-VG.lr, asZ); // anti-squat intersection at rear axle

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 240px", gap: 10 }}>
        {/* Side view pitch centre diagram */}
        <div style={{ ...GL, padding: 8 }}>
          <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6 }}>
            PITCH CENTRE DIAGRAM — SIDE VIEW
          </div>
          <svg viewBox={`0 0 ${svgW} ${svgH}`} style={{ width: "100%", background: "transparent" }}>
            {/* Ground */}
            <line x1="30" y1={oY} x2={svgW - 30} y2={oY} stroke={C.b2} strokeWidth="1" strokeDasharray="4 3" />

            {/* Wheelbase */}
            <line x1={fcpR.x} y1={oY} x2={fcpF.x} y2={oY} stroke={C.dm} strokeWidth="2" />
            <circle cx={fcpF.x} cy={oY} r="18" fill="none" stroke={C.dm} strokeWidth="1.5" opacity="0.4" />
            <circle cx={fcpR.x} cy={oY} r="18" fill="none" stroke={C.dm} strokeWidth="1.5" opacity="0.4" />
            <text x={fcpF.x} y={oY + 28} textAnchor="middle" fill={C.dm} fontSize="7" fontFamily="monospace">FRONT</text>
            <text x={fcpR.x} y={oY + 28} textAnchor="middle" fill={C.dm} fontSize="7" fontFamily="monospace">REAR</text>

            {/* CG */}
            <circle cx={cgP.x} cy={cgP.y} r="5" fill={C.red} opacity="0.6" />
            <text x={cgP.x + 8} y={cgP.y + 3} fill={C.red} fontSize="7" fontFamily="monospace" fontWeight="700">CG</text>

            {/* Anti-dive line (front CP → pcF) */}
            <line x1={fcpF.x} y1={oY} x2={pcF.x} y2={pcF.y} stroke={C.am} strokeWidth="1.5" strokeDasharray="6 3" />
            <text x={pcF.x + 6} y={pcF.y} fill={C.am} fontSize="7" fontFamily="monospace">Anti-Dive {(adLine * 100).toFixed(0)}%</text>

            {/* Anti-squat line (rear CP → pcR) */}
            <line x1={fcpR.x} y1={oY} x2={pcR.x} y2={pcR.y} stroke={C.cy} strokeWidth="1.5" strokeDasharray="6 3" />
            <text x={pcR.x - 80} y={pcR.y} fill={C.cy} fontSize="7" fontFamily="monospace">Anti-Squat {(asLine * 100).toFixed(0)}%</text>

            {/* Pitch centre (intersection) */}
            {(() => {
              // Intersection of the two lines
              const m1 = -adZ / VG.lf; // slope front line (going from front CP upward)
              const m2 = asZ / VG.lr; // slope rear line (going from rear CP upward)
              // Simplified pitch centre at CG x-position
              const pcZ = (adZ + asZ) / 2;
              const pc = toS(0, pcZ);
              return (
                <g>
                  <circle cx={pc.x} cy={pc.y} r="6" fill="none" stroke="#ff6090" strokeWidth="1.5" />
                  <circle cx={pc.x} cy={pc.y} r="2.5" fill="#ff6090" />
                  <text x={pc.x + 10} y={pc.y + 3} fill="#ff6090" fontSize="8" fontFamily="monospace" fontWeight="700">
                    PC h={pcZ.toFixed(0)}mm
                  </text>
                </g>
              );
            })()}

            <text x="12" y="16" fill={C.w} fontSize="9" fontFamily="monospace" fontWeight="700">PITCH GEOMETRY — Ter27 4WD</text>
          </svg>
        </div>

        {/* Anti-geometry bar chart */}
        <div style={{ ...GL, padding: "8px 10px" }}>
          <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 8 }}>
            ANTI-GEOMETRY %
          </div>
          {antiData.map(d => (
            <div key={d.param} style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, fontFamily: C.dt, marginBottom: 2 }}>
                <span style={{ color: C.dm }}>{d.param}</span>
                <span style={{ color: d.color, fontWeight: 700 }}>{d.value.toFixed(0)}%</span>
              </div>
              <div style={{ height: 6, background: `${C.b1}40`, borderRadius: 3, overflow: "hidden" }}>
                <div style={{ width: `${d.value}%`, height: "100%", background: d.color, borderRadius: 3, transition: "width 0.3s" }} />
              </div>
            </div>
          ))}
          <div style={{ borderTop: `1px solid ${C.b1}`, paddingTop: 8, marginTop: 8 }}>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, lineHeight: 1.5 }}>
              4WD NOTE: Front anti-squat is active under acceleration (Ter27-specific).
              Rear anti-squat dominates at 35%. Combined pitch reduction: {((VG.antiSquat + VG.antiSquatF) / 2 * 100).toFixed(0)}%.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLIANCE TAB
// ═══════════════════════════════════════════════════════════════════════════

function ComplianceTab() {
  const compData = useMemo(() => {
    const r = srng(99);
    return Array.from({ length: 21 }, (_, i) => {
      const Fy = -5000 + (10000 / 20) * i;
      return {
        Fy: +Fy.toFixed(0),
        toe_f: +(-0.12 * Fy / 1000 + (r() - 0.5) * 0.005).toFixed(4),
        toe_r: +(-0.10 * Fy / 1000 + (r() - 0.5) * 0.003).toFixed(4),
        camber_f: +(0.03 * Fy / 1000).toFixed(4),
        camber_r: +(0.02 * Fy / 1000).toFixed(4),
      };
    });
  }, []);

  const CH = ({ title, children }) => (
    <div style={{ ...GL, padding: "10px 8px 6px" }}>
      <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6, paddingLeft: 2 }}>{title}</div>
      {children}
    </div>
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <CH title="COMPLIANCE STEER vs LATERAL FORCE">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={compData} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="Fy" {...ax()} label={{ value: "Fy [N]", fontSize: 7, fill: C.dm, position: "bottom" }} />
            <YAxis {...ax()} label={{ value: "Δ Toe [°]", fontSize: 7, fill: C.dm, angle: -90, position: "insideLeft" }} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line dataKey="toe_f" stroke={C.cy} dot={false} strokeWidth={2} name="Front Comp. Steer" />
            <Line dataKey="toe_r" stroke={C.am} dot={false} strokeWidth={2} name="Rear Comp. Steer" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      <CH title="COMPLIANCE CAMBER vs LATERAL FORCE">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={compData} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="Fy" {...ax()} />
            <YAxis {...ax()} />
            <Tooltip {...tt()} />
            <ReferenceLine x={0} stroke={C.b2} strokeDasharray="3 3" />
            <Line dataKey="camber_f" stroke={C.cy} dot={false} strokeWidth={2} name="Front" />
            <Line dataKey="camber_r" stroke={C.am} dot={false} strokeWidth={2} name="Rear" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </CH>

      {/* Compliance coefficients panel */}
      <div style={{ ...GL, padding: "10px 12px", gridColumn: "1 / -1" }}>
        <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 8 }}>
          COMPLIANCE COEFFICIENTS — ANALYTICALLY DERIVED (suspension/compliance.py)
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
          {[
            { l: "Comp. Steer Front", v: "-0.120 deg/kN", note: "toe-in under lat load (stabilizing)", c: C.cy },
            { l: "Comp. Steer Rear", v: "-0.100 deg/kN", note: "toe-in under lat load (stabilizing)", c: C.am },
            { l: "Bushing K (tie rod)", v: "85,000 N/m", note: "radial stiffness", c: C.gn },
            { l: "Bushing K (wishbone)", v: "120,000 N/m", note: "radial stiffness", c: "#a78bfa" },
          ].map(item => (
            <div key={item.l} style={{ borderLeft: `2px solid ${item.c}30`, paddingLeft: 8 }}>
              <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>{item.l}</div>
              <div style={{ fontSize: 12, color: item.c, fontFamily: C.dt, fontWeight: 700, margin: "2px 0" }}>{item.v}</div>
              <div style={{ fontSize: 6.5, color: C.dm, fontFamily: C.dt }}>{item.note}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MODAL ANALYSIS TAB
// ═══════════════════════════════════════════════════════════════════════════

function ModalTab() {
  // Natural frequencies from quarter-car models
  const modes = [
    { mode: "Heave (body)", fn: 2.1, zeta: 0.32, color: C.cy },
    { mode: "Pitch (body)", fn: 2.4, zeta: 0.28, color: C.gn },
    { mode: "Roll (body)", fn: 3.1, zeta: 0.35, color: C.am },
    { mode: "Wheel Hop F", fn: 14.2, zeta: 0.08, color: C.red },
    { mode: "Wheel Hop R", fn: 15.8, zeta: 0.07, color: "#a78bfa" },
  ];

  // Generate frequency response data
  const freqResp = useMemo(() => {
    const data = [];
    for (let i = 0; i < 100; i++) {
      const f = 0.5 + (30 / 99) * i;
      const entry = { f: +f.toFixed(2) };
      modes.forEach(m => {
        const r = f / m.fn;
        const mag = 1 / Math.sqrt((1 - r * r) ** 2 + (2 * m.zeta * r) ** 2);
        entry[m.mode] = +Math.min(20 * Math.log10(mag), 30).toFixed(2);
      });
      data.push(entry);
    }
    return data;
  }, []);

  return (
    <div>
      {/* Mode summary cards */}
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${modes.length}, 1fr)`, gap: 8, marginBottom: 12 }}>
        {modes.map(m => (
          <div key={m.mode} style={{ ...GL, padding: "8px 10px", borderLeft: `3px solid ${m.color}` }}>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, letterSpacing: 1 }}>{m.mode}</div>
            <div style={{ fontSize: 16, color: m.color, fontFamily: C.dt, fontWeight: 700 }}>{m.fn} Hz</div>
            <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt }}>ζ = {m.zeta}</div>
          </div>
        ))}
      </div>

      {/* Bode magnitude plot */}
      <div style={{ ...GL, padding: "10px 8px 6px" }}>
        <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt, marginBottom: 6 }}>
          FREQUENCY RESPONSE — MAGNITUDE [dB]
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={freqResp} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.b1} strokeDasharray="2 4" />
            <XAxis dataKey="f" scale="log" domain={[0.5, 30]} {...ax()} label={{ value: "Frequency [Hz]", fontSize: 7, fill: C.dm, position: "bottom" }} />
            <YAxis domain={[-30, 30]} {...ax()} label={{ value: "Magnitude [dB]", fontSize: 7, fill: C.dm, angle: -90, position: "insideLeft" }} />
            <Tooltip {...tt()} />
            <ReferenceLine y={0} stroke={C.b2} strokeDasharray="3 3" />
            {modes.map(m => (
              <Line key={m.mode} dataKey={m.mode} stroke={m.color} dot={false} strokeWidth={1.5} name={m.mode} />
            ))}
            <Legend wrapperStyle={{ fontSize: 7, fontFamily: C.dt }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SETUP TAB — Full 40-dim parameter table with sensitivity
// ═══════════════════════════════════════════════════════════════════════════

function SetupTab() {
  const [groupFilter, setGroupFilter] = useState("all");
  const [sortBy, setSortBy] = useState("idx"); // idx | sens

  const filtered = useMemo(() => {
    let params = SETUP_PARAMS.map((p, i) => ({ ...p, idx: i }));
    if (groupFilter !== "all") params = params.filter(p => p.group === groupFilter);
    if (sortBy === "sens") params.sort((a, b) => Math.abs(b.sens) - Math.abs(a.sens));
    return params;
  }, [groupFilter, sortBy]);

  const maxSens = Math.max(...SETUP_PARAMS.map(p => Math.abs(p.sens)));

  return (
    <div>
      {/* Controls */}
      <div style={{ display: "flex", gap: 6, marginBottom: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={() => setGroupFilter("all")} style={{
          background: groupFilter === "all" ? `${SUS}18` : "transparent",
          border: `1px solid ${groupFilter === "all" ? `${SUS}55` : C.b1}`,
          color: groupFilter === "all" ? SUS : C.dm,
          fontSize: 8, fontWeight: 700, padding: "3px 10px", borderRadius: 3, cursor: "pointer", fontFamily: C.dt,
        }}>ALL ({SETUP_PARAMS.length})</button>
        {PARAM_GROUPS.map(g => {
          const count = SETUP_PARAMS.filter(p => p.group === g.key).length;
          return (
            <button key={g.key} onClick={() => setGroupFilter(g.key)} style={{
              background: groupFilter === g.key ? `${g.color}18` : "transparent",
              border: `1px solid ${groupFilter === g.key ? `${g.color}55` : C.b1}`,
              color: groupFilter === g.key ? g.color : C.dm,
              fontSize: 8, fontWeight: 700, padding: "3px 10px", borderRadius: 3, cursor: "pointer", fontFamily: C.dt,
            }}>{g.label} ({count})</button>
          );
        })}
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>Sort:</div>
        {["idx", "sens"].map(s => (
          <button key={s} onClick={() => setSortBy(s)} style={{
            background: sortBy === s ? `${SUS}18` : "transparent",
            border: `1px solid ${sortBy === s ? `${SUS}55` : C.b1}`,
            color: sortBy === s ? SUS : C.dm,
            fontSize: 8, fontWeight: 600, padding: "2px 8px", borderRadius: 3, cursor: "pointer", fontFamily: C.dt,
          }}>{s === "idx" ? "INDEX" : "|∂lap/∂p|"}</button>
        ))}
      </div>

      {/* Parameter table */}
      <div style={{ ...GL, padding: "8px 10px", maxHeight: 500, overflowY: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 8, fontFamily: C.dt }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${C.b1}`, position: "sticky", top: 0, background: C.panel || "#0e1420", zIndex: 2 }}>
              <th style={{ textAlign: "left", color: C.dm, padding: "4px 6px", fontWeight: 600, width: 30 }}>#</th>
              <th style={{ textAlign: "left", color: C.dm, padding: "4px 6px", fontWeight: 600 }}>Parameter</th>
              <th style={{ textAlign: "right", color: C.dm, padding: "4px 6px", fontWeight: 600 }}>Value</th>
              <th style={{ textAlign: "left", color: C.dm, padding: "4px 6px", fontWeight: 600 }}>Unit</th>
              <th style={{ textAlign: "left", color: C.dm, padding: "4px 6px", fontWeight: 600, width: 90 }}>Group</th>
              <th style={{ textAlign: "right", color: C.dm, padding: "4px 6px", fontWeight: 600 }}>∂lap/∂p</th>
              <th style={{ textAlign: "left", color: C.dm, padding: "4px 6px", fontWeight: 600, width: 100 }}>Sensitivity</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => {
              const grp = PARAM_GROUPS.find(g => g.key === p.group);
              const sensBar = Math.abs(p.sens) / maxSens * 100;
              return (
                <tr key={p.key} style={{ borderBottom: `1px solid ${C.b1}15` }}>
                  <td style={{ padding: "3px 6px", color: C.dm }}>{p.idx}</td>
                  <td style={{ padding: "3px 6px", color: C.br, fontWeight: 500 }}>{p.label}</td>
                  <td style={{ textAlign: "right", padding: "3px 6px", color: C.w, fontWeight: 700 }}>{typeof p.val === "number" ? p.val.toFixed(p.val >= 100 ? 0 : p.val >= 1 ? 2 : 3) : p.val}</td>
                  <td style={{ padding: "3px 6px", color: C.dm }}>{p.unit}</td>
                  <td style={{ padding: "3px 6px" }}>
                    <span style={{ fontSize: 7, color: grp?.color || C.dm, background: `${grp?.color || C.dm}15`, padding: "1px 5px", borderRadius: 2 }}>
                      {grp?.label || p.group}
                    </span>
                  </td>
                  <td style={{ textAlign: "right", padding: "3px 6px", color: Math.abs(p.sens) > 0.02 ? C.red : C.dm, fontWeight: Math.abs(p.sens) > 0.02 ? 700 : 400 }}>
                    {p.sens.toFixed(4)}
                  </td>
                  <td style={{ padding: "3px 6px" }}>
                    <div style={{ height: 4, background: `${C.b1}40`, borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ width: `${sensBar}%`, height: "100%", background: Math.abs(p.sens) > 0.02 ? C.red : C.cy, borderRadius: 2 }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div style={{ display: "flex", gap: 12, marginTop: 10, fontSize: 8, fontFamily: C.dt, color: C.dm }}>
        <span>Total: {SETUP_PARAMS.length} parameters</span>
        <span>Canonical 28-dim + 12 extended (Ter27 4WD)</span>
        <span>Shown: {filtered.length}</span>
        <span style={{ color: C.red }}>Top sensitivity: CG Height (−0.045 s/mm)</span>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN MODULE EXPORT
// ═══════════════════════════════════════════════════════════════════════════

export default function SuspensionModule({ data, mode }) {
  const [tab, setTab] = useState("schematic");
  const [maneuver, setManeuver] = useState("lap");

  const dynamics = useMemo(() => generateDynamicsData(maneuver), [maneuver]);

  // In LIVE mode, frame would come from useLiveTelemetry
  // For now, use synthetic data
  const frame = mode === "LIVE" ? null : null; // TODO: connect useLiveTelemetry

  // Compute summary KPIs from dynamics
  const kpis = useMemo(() => {
    if (!dynamics.length) return {};
    const last = dynamics[dynamics.length - 1];
    const maxRoll = Math.max(...dynamics.map(d => Math.abs(d.roll)));
    const maxAy = Math.max(...dynamics.map(d => Math.abs(d.ay)));
    const avgRollGrad = dynamics.reduce((s, d) => s + d.rollGrad, 0) / dynamics.length;
    const avgLLTD = dynamics.reduce((s, d) => s + d.LLTD, 0) / dynamics.length;
    const maxFz = Math.max(...dynamics.map(d => Math.max(d.Fz_FL, d.Fz_FR, d.Fz_RL, d.Fz_RR)));
    const maxTravel = Math.max(...dynamics.map(d => Math.max(Math.abs(d.zFL), Math.abs(d.zFR), Math.abs(d.zRL), Math.abs(d.zRR))));
    return { maxRoll, maxAy, avgRollGrad, avgLLTD, maxFz, maxTravel };
  }, [dynamics]);

  return (
    <div>
      {/* KPI header strip */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8, marginBottom: 10 }}>
        <KPI label="Roll Gradient" value={`${(kpis.avgRollGrad || 0).toFixed(2)}`} sub="°/G lateral" sentiment="neutral" delay={0} />
        <KPI label="LLTD" value={`${(kpis.avgLLTD || 50).toFixed(1)}%`} sub="target 52%" sentiment={(kpis.avgLLTD || 50) > 48 && (kpis.avgLLTD || 50) < 56 ? "positive" : "warning"} delay={1} />
        <KPI label="Max Travel" value={`${(kpis.maxTravel || 0).toFixed(1)}`} sub="mm peak" sentiment={(kpis.maxTravel || 0) < 40 ? "positive" : "warning"} delay={2} />
        <KPI label="Peak Fz" value={`${(kpis.maxFz || 0).toFixed(0)}`} sub="N single corner" sentiment="neutral" delay={3} />
        <KPI label="Setup Dim" value="40" sub="28+12 extended" sentiment="positive" delay={4} />
        <KPI label="Hardpoints" value="28" sub="14 per axle" sentiment="positive" delay={5} />
      </div>

      {/* Maneuver selector + sub-tab pills */}
      <div style={{ display: "flex", gap: 6, marginBottom: 10, alignItems: "center", flexWrap: "wrap" }}>
        {/* Maneuver select */}
        <div style={{ display: "flex", gap: 4, marginRight: 10 }}>
          {["lap", "skidpad", "brake", "accel", "chicane"].map(m => (
            <button key={m} onClick={() => setManeuver(m)} style={{
              background: maneuver === m ? `${C.am}18` : "transparent",
              border: `1px solid ${maneuver === m ? `${C.am}55` : C.b1}`,
              color: maneuver === m ? C.am : C.dm,
              fontSize: 8, fontWeight: 600, padding: "3px 8px", borderRadius: 3,
              cursor: "pointer", fontFamily: C.dt, textTransform: "uppercase",
            }}>{m}</button>
          ))}
        </div>
        <div style={{ width: 1, height: 16, background: C.b1 }} />
        {/* Sub-tabs */}
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {TABS.map(t => (
            <Pill key={t.key} active={tab === t.key} label={t.label}
              onClick={() => setTab(t.key)} color={SUS} />
          ))}
        </div>
      </div>

      {/* Tab content */}
      {tab === "schematic" && <SchematicTab dynamics={dynamics} frame={frame} />}
      {tab === "kinematics" && <KinematicsTab />}
      {tab === "dynamics" && <DynamicsTab dynamics={dynamics} />}
      {tab === "antigeom" && <AntiGeomTab />}
      {tab === "compliance" && <ComplianceTab />}
      {tab === "modal" && <ModalTab />}
      {tab === "setup" && <SetupTab />}
    </div>
  );
}