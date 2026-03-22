// ═══════════════════════════════════════════════════════════════════════════════
// src/live/LiveGraphSystem.jsx — v4.3 FULL CHANNEL REGISTRY
// ═══════════════════════════════════════════════════════════════════════════════
// 130+ telemetry channels from the 46-DOF state vector, derived dynamics,
// multi-fidelity tire model, Port-Hamiltonian energy audit, and Diff-WMPC.
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useRef, useCallback, useEffect } from "react";
import { C, GL } from "../theme.js";

// ═════════════════════════════════════════════════════════════════════════════
// 12 PHYSICS GROUPS
// ═════════════════════════════════════════════════════════════════════════════
export const GROUPS = [
  "Chassis",       // Spatial kinematics + velocities + accelerations
  "Suspension",    // Corner heave, velocities, forces
  "Wheels",        // Spin rates, rotation angles
  "Thermal",       // 5-node Jaeger per axle (flash, surf, bulk, carcass, gas)
  "Slip",          // Transient + kinematic slip angles/ratios, friction util
  "Tire Forces",   // Pacejka Fx/Fy/Mz, combined slip modifiers, PINN/GP
  "Aero",          // Downforce, drag, moments, ground clearance
  "Geometry",      // Dynamic camber, toe, turn-slip, compliance steer
  "Energy",        // Port-Hamiltonian: H, T, V, H_res, dH/dt, R_diss
  "WMPC",          // Controls, AL multipliers/slacks, solver telemetry
  "E-Drive",       // Inverter/motor temps, torque vectoring
  "Twin",          // EKF innovations, model confidence, fidelity
];

// ═════════════════════════════════════════════════════════════════════════════
// 130+ CHANNEL DEFINITIONS
// Each: { key, label, unit, group, color, range:[min,max] }
// ═════════════════════════════════════════════════════════════════════════════
export const CHANNELS = [

  // ═══ §1: CHASSIS SPATIAL KINEMATICS & VELOCITIES (14 states) ══════════
  { key:"posX",      label:"X Position",       unit:"m",    group:"Chassis", color:"#80cbc4", range:[-50,50] },
  { key:"posY",      label:"Y Position",       unit:"m",    group:"Chassis", color:"#80deea", range:[-50,50] },
  { key:"posZ",      label:"Z Position",       unit:"m",    group:"Chassis", color:"#a5d6a7", range:[-0.5,0.5] },
  { key:"roll",      label:"Roll φ",           unit:"°",    group:"Chassis", color:"#f48fb1", range:[-5,5] },
  { key:"pitch",     label:"Pitch θ",          unit:"°",    group:"Chassis", color:"#ce93d8", range:[-3,3] },
  { key:"yaw",       label:"Yaw ψ",            unit:"°",    group:"Chassis", color:"#90caf9", range:[-180,180] },
  { key:"speed",     label:"Speed |v|",        unit:"m/s",  group:"Chassis", color:"#00e676", range:[0,30] },
  { key:"vx",        label:"v_x (body)",       unit:"m/s",  group:"Chassis", color:"#69f0ae", range:[0,30] },
  { key:"vy",        label:"v_y (body)",       unit:"m/s",  group:"Chassis", color:"#64ffda", range:[-5,5] },
  { key:"vz",        label:"v_z (body)",       unit:"m/s",  group:"Chassis", color:"#b2ff59", range:[-1,1] },
  { key:"wx",        label:"ω_x (roll rate)",  unit:"rad/s",group:"Chassis", color:"#ff80ab", range:[-3,3] },
  { key:"wy",        label:"ω_y (pitch rate)", unit:"rad/s",group:"Chassis", color:"#ea80fc", range:[-2,2] },
  { key:"wz",        label:"ω_z (yaw rate)",   unit:"rad/s",group:"Chassis", color:"#b388ff", range:[-4,4] },
  // Derived accelerations
  { key:"ax",        label:"a_x (Lon G)",      unit:"G",    group:"Chassis", color:"#ffab00", range:[-2,2] },
  { key:"ay",        label:"a_y (Lat G)",      unit:"G",    group:"Chassis", color:"#00d2ff", range:[-2,2] },
  { key:"az",        label:"a_z (Vert G)",     unit:"G",    group:"Chassis", color:"#b2ff59", range:[-0.5,0.5] },
  { key:"combinedG", label:"Combined G",       unit:"G",    group:"Chassis", color:"#00e676", range:[0,2.5] },
  { key:"sideslip",  label:"Sideslip β",       unit:"rad",  group:"Chassis", color:"#ea80fc", range:[-0.1,0.1] },

  // ═══ §1: SUSPENSION KINEMATICS & FORCES (8 states + derived) ══════════
  { key:"zFL",       label:"z_FL (heave)",     unit:"mm",   group:"Suspension", color:"#42a5f5", range:[-30,10] },
  { key:"zFR",       label:"z_FR (heave)",     unit:"mm",   group:"Suspension", color:"#66bb6a", range:[-30,10] },
  { key:"zRL",       label:"z_RL (heave)",     unit:"mm",   group:"Suspension", color:"#ffa726", range:[-30,10] },
  { key:"zRR",       label:"z_RR (heave)",     unit:"mm",   group:"Suspension", color:"#ab47bc", range:[-30,10] },
  { key:"zdFL",      label:"ż_FL (velocity)",  unit:"m/s",  group:"Suspension", color:"#42a5f5", range:[-0.5,0.5] },
  { key:"zdFR",      label:"ż_FR (velocity)",  unit:"m/s",  group:"Suspension", color:"#66bb6a", range:[-0.5,0.5] },
  { key:"zdRL",      label:"ż_RL (velocity)",  unit:"m/s",  group:"Suspension", color:"#ffa726", range:[-0.5,0.5] },
  { key:"zdRR",      label:"ż_RR (velocity)",  unit:"m/s",  group:"Suspension", color:"#ab47bc", range:[-0.5,0.5] },
  // Forces (derived)
  { key:"FspFL",     label:"F_spring FL",      unit:"N",    group:"Suspension", color:"#29b6f6", range:[-3000,3000] },
  { key:"FspFR",     label:"F_spring FR",      unit:"N",    group:"Suspension", color:"#66bb6a", range:[-3000,3000] },
  { key:"FdmpFL",    label:"F_damper FL",      unit:"N",    group:"Suspension", color:"#4fc3f7", range:[-2000,2000] },
  { key:"FdmpFR",    label:"F_damper FR",      unit:"N",    group:"Suspension", color:"#81c784", range:[-2000,2000] },
  { key:"FbsFL",     label:"F_bumpstop FL",    unit:"N",    group:"Suspension", color:"#e57373", range:[0,5000] },
  { key:"FarbF",     label:"F_arb Front",      unit:"N",    group:"Suspension", color:"#ffb74d", range:[-1500,1500] },
  { key:"FarbR",     label:"F_arb Rear",       unit:"N",    group:"Suspension", color:"#ff8a65", range:[-1500,1500] },
  { key:"FzFL",      label:"Fz FL (contact)",  unit:"N",    group:"Suspension", color:"#42a5f5", range:[0,2500] },
  { key:"FzFR",      label:"Fz FR (contact)",  unit:"N",    group:"Suspension", color:"#66bb6a", range:[0,2500] },
  { key:"FzRL",      label:"Fz RL (contact)",  unit:"N",    group:"Suspension", color:"#ffa726", range:[0,2500] },
  { key:"FzRR",      label:"Fz RR (contact)",  unit:"N",    group:"Suspension", color:"#ab47bc", range:[0,2500] },

  // ═══ §1: WHEEL KINEMATICS (8 states) ══════════════════════════════════
  { key:"whlAngFL",  label:"θ_wheel FL",       unit:"rad",  group:"Wheels", color:"#4dd0e1", range:[0,1000] },
  { key:"whlAngFR",  label:"θ_wheel FR",       unit:"rad",  group:"Wheels", color:"#4db6ac", range:[0,1000] },
  { key:"whlSpdFL",  label:"ω_wheel FL",       unit:"rad/s",group:"Wheels", color:"#4dd0e1", range:[0,150] },
  { key:"whlSpdFR",  label:"ω_wheel FR",       unit:"rad/s",group:"Wheels", color:"#4db6ac", range:[0,150] },
  { key:"whlSpdRL",  label:"ω_wheel RL",       unit:"rad/s",group:"Wheels", color:"#aed581", range:[0,150] },
  { key:"whlSpdRR",  label:"ω_wheel RR",       unit:"rad/s",group:"Wheels", color:"#dce775", range:[0,150] },

  // ═══ §1: THERMODYNAMICS (10 states — 5 nodes × 2 axles) ══════════════
  { key:"TsurfInF",  label:"T_surf_inner F",   unit:"°C",   group:"Thermal", color:"#ff5722", range:[20,170] },
  { key:"TsurfMdF",  label:"T_surf_mid F",     unit:"°C",   group:"Thermal", color:"#ff7043", range:[20,170] },
  { key:"TsurfOtF",  label:"T_surf_outer F",   unit:"°C",   group:"Thermal", color:"#ff8a65", range:[20,170] },
  { key:"TgasF",     label:"T_gas Front",      unit:"°C",   group:"Thermal", color:"#0d47a1", range:[20,80] },
  { key:"TcoreF",    label:"T_core Front",     unit:"°C",   group:"Thermal", color:"#ff9800", range:[20,130] },
  { key:"TsurfInR",  label:"T_surf_inner R",   unit:"°C",   group:"Thermal", color:"#e53935", range:[20,170] },
  { key:"TsurfMdR",  label:"T_surf_mid R",     unit:"°C",   group:"Thermal", color:"#ef5350", range:[20,170] },
  { key:"TsurfOtR",  label:"T_surf_outer R",   unit:"°C",   group:"Thermal", color:"#e57373", range:[20,170] },
  { key:"TgasR",     label:"T_gas Rear",       unit:"°C",   group:"Thermal", color:"#1565c0", range:[20,80] },
  { key:"TcoreR",    label:"T_core Rear",      unit:"°C",   group:"Thermal", color:"#ffa726", range:[20,130] },
  { key:"flashDtF",  label:"ΔT_flash-surf F",  unit:"°C",   group:"Thermal", color:"#d50000", range:[0,40] },
  { key:"flashDtR",  label:"ΔT_flash-surf R",  unit:"°C",   group:"Thermal", color:"#b71c1c", range:[0,40] },
  { key:"muTherm",   label:"μ_thermal",        unit:"",     group:"Thermal", color:"#4caf50", range:[0.5,1.2] },
  { key:"muPress",   label:"μ_pressure",       unit:"",     group:"Thermal", color:"#66bb6a", range:[0.8,1.1] },

  // ═══ §1: TRANSIENT SLIP (8 states) + §3: KINEMATIC SLIP ═══════════════
  { key:"alphaFL",   label:"α_t FL (trans.)",  unit:"°",    group:"Slip", color:"#00bcd4", range:[-15,15] },
  { key:"kappaFL",   label:"κ_t FL (trans.)",  unit:"",     group:"Slip", color:"#26c6da", range:[-0.3,0.3] },
  { key:"alphaFR",   label:"α_t FR",           unit:"°",    group:"Slip", color:"#4dd0e1", range:[-15,15] },
  { key:"kappaFR",   label:"κ_t FR",           unit:"",     group:"Slip", color:"#80deea", range:[-0.3,0.3] },
  { key:"alphaRL",   label:"α_t RL",           unit:"°",    group:"Slip", color:"#009688", range:[-15,15] },
  { key:"kappaRL",   label:"κ_t RL",           unit:"",     group:"Slip", color:"#4db6ac", range:[-0.3,0.3] },
  { key:"alphaRR",   label:"α_t RR",           unit:"°",    group:"Slip", color:"#80cbc4", range:[-15,15] },
  { key:"kappaRR",   label:"κ_t RR",           unit:"",     group:"Slip", color:"#b2dfdb", range:[-0.3,0.3] },
  { key:"aKinF",     label:"α_kin Front",      unit:"°",    group:"Slip", color:"#00acc1", range:[-15,15] },
  { key:"aKinR",     label:"α_kin Rear",       unit:"°",    group:"Slip", color:"#00838f", range:[-15,15] },
  { key:"kKinF",     label:"κ_kin Front",      unit:"",     group:"Slip", color:"#0097a7", range:[-0.3,0.3] },
  { key:"kKinR",     label:"κ_kin Rear",       unit:"",     group:"Slip", color:"#00695c", range:[-0.3,0.3] },
  { key:"fricUtil",  label:"Friction Util.",   unit:"%",    group:"Slip", color:"#ffab00", range:[0,110] },

  // ═══ §3: TIRE FORCES (Pacejka + PINN + GP) ════════════════════════════
  { key:"FxFL",      label:"Fx FL",            unit:"N",    group:"Tire Forces", color:"#ef5350", range:[-4000,4000] },
  { key:"FxFR",      label:"Fx FR",            unit:"N",    group:"Tire Forces", color:"#e57373", range:[-4000,4000] },
  { key:"FxRL",      label:"Fx RL",            unit:"N",    group:"Tire Forces", color:"#ff8a65", range:[-4000,4000] },
  { key:"FxRR",      label:"Fx RR",            unit:"N",    group:"Tire Forces", color:"#ffab91", range:[-4000,4000] },
  { key:"FyFL",      label:"Fy FL",            unit:"N",    group:"Tire Forces", color:"#42a5f5", range:[-4000,4000] },
  { key:"FyFR",      label:"Fy FR",            unit:"N",    group:"Tire Forces", color:"#64b5f6", range:[-4000,4000] },
  { key:"FyRL",      label:"Fy RL",            unit:"N",    group:"Tire Forces", color:"#29b6f6", range:[-4000,4000] },
  { key:"FyRR",      label:"Fy RR",            unit:"N",    group:"Tire Forces", color:"#4fc3f7", range:[-4000,4000] },
  { key:"MzFL",      label:"Mz FL (align.)",   unit:"Nm",   group:"Tire Forces", color:"#7e57c2", range:[-80,80] },
  { key:"MzFR",      label:"Mz FR",            unit:"Nm",   group:"Tire Forces", color:"#9575cd", range:[-80,80] },
  { key:"Gyk",       label:"G_yk (lat red.)",  unit:"",     group:"Tire Forces", color:"#ffca28", range:[0,1] },
  { key:"Gxa",       label:"G_xa (lon red.)",  unit:"",     group:"Tire Forces", color:"#ffd54f", range:[0,1] },
  { key:"dFxPinn",   label:"ΔFx PINN",         unit:"N",    group:"Tire Forces", color:"#ab47bc", range:[-500,500] },
  { key:"dFyPinn",   label:"ΔFy PINN",         unit:"N",    group:"Tire Forces", color:"#ce93d8", range:[-500,500] },
  { key:"sigGP",     label:"σ² GP (epist.)",   unit:"N²",   group:"Tire Forces", color:"#b388ff", range:[0,50000] },
  { key:"lcbPen",    label:"LCB Penalty",      unit:"",     group:"Tire Forces", color:"#d1c4e9", range:[0,0.15] },

  // ═══ §2: AERODYNAMICS ═════════════════════════════════════════════════
  { key:"FzAeroF",   label:"Fz_aero Front",    unit:"N",    group:"Aero", color:"#29b6f6", range:[0,800] },
  { key:"FzAeroR",   label:"Fz_aero Rear",     unit:"N",    group:"Aero", color:"#4fc3f7", range:[0,800] },
  { key:"FxAero",    label:"Fx_aero (drag)",   unit:"N",    group:"Aero", color:"#ef5350", range:[0,400] },
  { key:"MyAero",    label:"M_y aero (pitch)", unit:"Nm",   group:"Aero", color:"#ab47bc", range:[-200,200] },
  { key:"MxAero",    label:"M_x aero (roll)",  unit:"Nm",   group:"Aero", color:"#ce93d8", range:[-100,100] },
  { key:"rideHF",    label:"Ride Height F",    unit:"mm",   group:"Aero", color:"#66bb6a", range:[15,60] },
  { key:"rideHR",    label:"Ride Height R",    unit:"mm",   group:"Aero", color:"#aed581", range:[20,80] },

  // ═══ §2: STEERING & GEOMETRY ══════════════════════════════════════════
  { key:"steer",     label:"Steer δ (rack)",   unit:"°",    group:"Geometry", color:"#00d2ff", range:[-20,20] },
  { key:"camFL",     label:"Camber γ FL",      unit:"°",    group:"Geometry", color:"#42a5f5", range:[-4,0] },
  { key:"camFR",     label:"Camber γ FR",      unit:"°",    group:"Geometry", color:"#66bb6a", range:[-4,0] },
  { key:"camRL",     label:"Camber γ RL",      unit:"°",    group:"Geometry", color:"#ffa726", range:[-3,0] },
  { key:"camRR",     label:"Camber γ RR",      unit:"°",    group:"Geometry", color:"#ab47bc", range:[-3,0] },
  { key:"toeFL",     label:"Toe FL (dyn.)",    unit:"°",    group:"Geometry", color:"#4dd0e1", range:[-1,1] },
  { key:"toeFR",     label:"Toe FR (dyn.)",    unit:"°",    group:"Geometry", color:"#4db6ac", range:[-1,1] },
  { key:"compSteer", label:"Compliance Steer", unit:"°",    group:"Geometry", color:"#ffca28", range:[-0.5,0.5] },
  { key:"turnSlip",  label:"Turn Slip 1/R",    unit:"1/m",  group:"Geometry", color:"#a5d6a7", range:[-0.15,0.15] },

  // ═══ §4: PORT-HAMILTONIAN ENERGY AUDIT ════════════════════════════════
  { key:"Htotal",    label:"H_total",          unit:"J",    group:"Energy", color:"#ab47bc", range:[3000,6000] },
  { key:"Tkin",      label:"T_kinetic (½mv²)", unit:"J",    group:"Energy", color:"#29b6f6", range:[0,5000] },
  { key:"Vstruct",   label:"V_structural",     unit:"J",    group:"Energy", color:"#66bb6a", range:[0,500] },
  { key:"Hres",      label:"H_net residual",   unit:"J",    group:"Energy", color:"#ce93d8", range:[0,100] },
  { key:"dHdt",      label:"dH/dt (power)",    unit:"W",    group:"Energy", color:"#00d2ff", range:[-50,50] },
  { key:"Rdiss",     label:"R_dissipated",     unit:"W",    group:"Energy", color:"#ef5350", range:[0,30] },
  { key:"Qtherm",    label:"Q̇_thermal",        unit:"W",    group:"Energy", color:"#ffa726", range:[0,20] },
  { key:"passViol",  label:"Passivity Viol.",  unit:"W",    group:"Energy", color:"#d50000", range:[0,5] },
  { key:"RnetCond",  label:"R_net κ(L)",       unit:"",     group:"Energy", color:"#7e57c2", range:[1,100] },

  // ═══ §5: DIFF-WMPC CONTROLLER ═════════════════════════════════════════
  { key:"ctrlSteer", label:"u_steer (cmd)",    unit:"°",    group:"WMPC", color:"#00d2ff", range:[-20,20] },
  { key:"ctrlThrot", label:"u_throttle",       unit:"",     group:"WMPC", color:"#00e676", range:[0,1] },
  { key:"ctrlBrake", label:"u_brake",          unit:"N",    group:"WMPC", color:"#e10600", range:[0,3000] },
  { key:"muNmean",   label:"μ_n (tube center)",unit:"m",    group:"WMPC", color:"#80deea", range:[-3,3] },
  { key:"sigNvar",   label:"σ_n (tube width)", unit:"m",    group:"WMPC", color:"#b388ff", range:[0,2] },
  { key:"dLeft",     label:"d_left (boundary)",unit:"m",    group:"WMPC", color:"#a5d6a7", range:[0,5] },
  { key:"dRight",    label:"d_right",          unit:"m",    group:"WMPC", color:"#c5e1a5", range:[0,5] },
  { key:"lamGrip",   label:"λ_grip",           unit:"",     group:"WMPC", color:"#ef5350", range:[0,10] },
  { key:"lamSteer",  label:"λ_steer_rate",     unit:"",     group:"WMPC", color:"#ffab00", range:[0,5] },
  { key:"lamAx",     label:"λ_ax_limit",       unit:"",     group:"WMPC", color:"#42a5f5", range:[0,5] },
  { key:"lamTrack",  label:"λ_track",          unit:"",     group:"WMPC", color:"#66bb6a", range:[0,5] },
  { key:"slkGrip",   label:"Slack_grip",       unit:"",     group:"WMPC", color:"#e10600", range:[0,0.5] },
  { key:"slkSteer",  label:"Slack_steer",      unit:"",     group:"WMPC", color:"#ffab00", range:[0,0.5] },
  { key:"slkTrack",  label:"Slack_track",      unit:"",     group:"WMPC", color:"#66bb6a", range:[0,0.5] },
  { key:"alRho",     label:"ρ (AL penalty)",   unit:"",     group:"WMPC", color:"#7e57c2", range:[0,1000] },
  { key:"lbfgsIter", label:"L-BFGS-B iters",  unit:"",     group:"WMPC", color:"#78909c", range:[0,30] },
  { key:"costLap",   label:"Cost: lap time",   unit:"",     group:"WMPC", color:"#ef5350", range:[0,100] },
  { key:"costTrack", label:"Cost: tracking",   unit:"",     group:"WMPC", color:"#42a5f5", range:[0,50] },
  { key:"costL1",    label:"Cost: L1 wavelet", unit:"",     group:"WMPC", color:"#b388ff", range:[0,20] },
  { key:"jaxFwdMs",  label:"JAX fwd pass",     unit:"ms",   group:"WMPC", color:"#00e676", range:[0,15] },
  { key:"jaxBwdMs",  label:"JAX bwd pass",     unit:"ms",   group:"WMPC", color:"#69f0ae", range:[0,25] },
  { key:"solveMs",   label:"WMPC total solve", unit:"ms",   group:"WMPC", color:"#ffab00", range:[0,20] },

  // ═══ E-DRIVE & TORQUE VECTORING ═══════════════════════════════════════
  { key:"invTmpR",   label:"Inverter Temp R",  unit:"°C",   group:"E-Drive", color:"#42a5f5", range:[20,100] },
  { key:"invTmpL",   label:"Inverter Temp L",  unit:"°C",   group:"E-Drive", color:"#66bb6a", range:[20,100] },
  { key:"motTmpR",   label:"Motor Temp R",     unit:"°C",   group:"E-Drive", color:"#ef5350", range:[20,120] },
  { key:"motTmpL",   label:"Motor Temp L",     unit:"°C",   group:"E-Drive", color:"#ffa726", range:[20,120] },
  { key:"tvTarget",  label:"TV Target Yaw",    unit:"Nm",   group:"E-Drive", color:"#ab47bc", range:[-150,150] },
  { key:"tvActual",  label:"TV Actual Yaw",    unit:"Nm",   group:"E-Drive", color:"#ce93d8", range:[-150,150] },
  { key:"tvError",   label:"TV Error",         unit:"Nm",   group:"E-Drive", color:"#e10600", range:[-30,30] },

  // ═══ DIGITAL TWIN HEALTH ══════════════════════════════════════════════
  { key:"ekfAx",     label:"EKF innov a_x",   unit:"",     group:"Twin", color:"#00d2ff", range:[-0.05,0.05] },
  { key:"ekfWz",     label:"EKF innov ω_z",   unit:"",     group:"Twin", color:"#ffab00", range:[-0.05,0.05] },
  { key:"ekfVy",     label:"EKF innov v_y",   unit:"",     group:"Twin", color:"#b388ff", range:[-0.05,0.05] },
  { key:"modConf",   label:"Model Confidence", unit:"%",    group:"Twin", color:"#00e676", range:[70,100] },
  { key:"twinFid",   label:"Twin Fidelity",    unit:"%",    group:"Twin", color:"#66bb6a", range:[0,100] },
];

export const CHANNEL_MAP = Object.fromEntries(CHANNELS.map(c => [c.key, c]));

// ═════════════════════════════════════════════════════════════════════════════
// CANVAS TIME-SERIES RENDERER — hardware-accelerated, multi-channel
// ═════════════════════════════════════════════════════════════════════════════
function TimeSeriesCanvas({ history, channels, height = 140, tick }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !history.length || !channels.length) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const pad = { t: 8, b: 18, l: 38, r: 50 };
    const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "rgba(25,35,55,0.4)"; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) { const y = pad.t + (pH / 4) * i; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pW, y); ctx.stroke(); }
    // Zero line
    ctx.strokeStyle = "rgba(62,74,100,0.4)"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t + pH / 2); ctx.lineTo(pad.l + pW, pad.t + pH / 2); ctx.stroke(); ctx.setLineDash([]);

    const n = history.length;
    channels.forEach(chKey => {
      const ch = CHANNEL_MAP[chKey]; if (!ch) return;
      const [rMin, rMax] = ch.range; const rR = rMax - rMin || 1;
      ctx.strokeStyle = ch.color; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.85; ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        const val = history[i]?.[chKey]; if (val === undefined || val === null) continue;
        const x = pad.l + (i / Math.max(1, n - 1)) * pW;
        const norm = (val - rMin) / rR;
        const y = pad.t + pH * (1 - Math.max(0, Math.min(1, norm)));
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
      // Current value label
      const last = history[n - 1]?.[chKey];
      if (last !== undefined) {
        const norm = (last - rMin) / rR;
        const y = pad.t + pH * (1 - Math.max(0, Math.min(1, norm)));
        ctx.fillStyle = ch.color; ctx.font = `bold ${Math.round(W * 0.016)}px 'Azeret Mono',monospace`;
        ctx.textAlign = "left"; ctx.fillText(`${Number(last).toFixed(ch.unit === "N" || ch.unit === "°C" ? 0 : 2)}`, pad.l + pW + 3, y + 3);
      }
    });
    // Y ticks (first channel)
    if (channels[0]) { const ch = CHANNEL_MAP[channels[0]]; if (ch) {
      ctx.fillStyle = C.dm; ctx.font = `${Math.round(W * 0.014)}px 'Azeret Mono',monospace`; ctx.textAlign = "right";
      const [rMin, rMax] = ch.range;
      for (let i = 0; i <= 4; i++) { const f = 1 - i / 4; ctx.fillText((rMin + f * (rMax - rMin)).toFixed(1), pad.l - 3, pad.t + (pH / 4) * i + 3); }
    }}
    // Time axis
    ctx.fillStyle = C.dm; ctx.font = `${Math.round(W * 0.012)}px 'Azeret Mono',monospace`; ctx.textAlign = "center";
    const t0 = history[0]?.t || 0, tN = history[n - 1]?.t || 0;
    for (let i = 0; i <= 3; i++) { const f = i / 3; ctx.fillText((+t0 + f * (+tN - +t0)).toFixed(1) + "s", pad.l + f * pW, H - 2); }
  }, [history, channels, tick]);

  return <canvas ref={canvasRef} width={700} height={height * 2} style={{ width: "100%", height, borderRadius: 6, display: "block" }} />;
}

// ═════════════════════════════════════════════════════════════════════════════
// GRAPH SLOT — individual configurable panel
// ═════════════════════════════════════════════════════════════════════════════
function GraphSlot({ id, history, onRemove, compact = false, tick }) {
  const [selected, setSelected] = useState([]);
  const [showPicker, setShowPicker] = useState(!compact);
  const [filterGroup, setFilterGroup] = useState(null);
  const toggle = useCallback(key => setSelected(p => p.includes(key) ? p.filter(k => k !== key) : [...p, key]), []);
  const filtered = filterGroup ? CHANNELS.filter(c => c.group === filterGroup) : CHANNELS;

  return (
    <div style={{ ...GL, padding: "6px 8px", marginBottom: 5, borderLeft: `2px solid ${selected.length > 0 ? CHANNEL_MAP[selected[0]]?.color || C.cy : C.b2}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ fontSize: 7, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>GRAPH {id}</span>
          <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>{selected.length} ch · {CHANNELS.length} avail</span>
          <button onClick={() => setShowPicker(!showPicker)} style={{ background: showPicker ? `${C.cy}15` : "none", border: `1px solid ${showPicker ? C.cy : C.b1}`, borderRadius: 8, padding: "1px 6px", fontSize: 7, color: showPicker ? C.cy : C.dm, fontFamily: C.dt, cursor: "pointer" }}>
            {showPicker ? "▼ Channels" : "▶ Channels"}
          </button>
        </div>
        {onRemove && <button onClick={onRemove} style={{ background: `${C.red}10`, border: `1px solid ${C.red}30`, borderRadius: 4, padding: "1px 6px", fontSize: 7, color: C.red, fontFamily: C.dt, cursor: "pointer" }}>✕</button>}
      </div>
      {showPicker && (<div style={{ marginBottom: 4 }}>
        <div style={{ display: "flex", gap: 2, marginBottom: 3, flexWrap: "wrap" }}>
          <button onClick={() => setFilterGroup(null)} style={{ background: !filterGroup ? `${C.cy}15` : "none", border: `1px solid ${!filterGroup ? C.cy : C.b1}`, borderRadius: 8, padding: "1px 6px", fontSize: 6, color: !filterGroup ? C.cy : C.dm, fontFamily: C.dt, cursor: "pointer" }}>ALL ({CHANNELS.length})</button>
          {GROUPS.map(g => { const cnt = CHANNELS.filter(c => c.group === g).length; return (
            <button key={g} onClick={() => setFilterGroup(g)} style={{ background: filterGroup === g ? `${C.cy}15` : "none", border: `1px solid ${filterGroup === g ? C.cy : C.b1}`, borderRadius: 8, padding: "1px 6px", fontSize: 6, color: filterGroup === g ? C.cy : C.dm, fontFamily: C.dt, cursor: "pointer" }}>{g} ({cnt})</button>
          ); })}
        </div>
        <div style={{ display: "flex", gap: 2, flexWrap: "wrap", maxHeight: 60, overflowY: "auto" }}>
          {filtered.map(ch => { const isOn = selected.includes(ch.key); return (
            <button key={ch.key} onClick={() => toggle(ch.key)} style={{ background: isOn ? `${ch.color}18` : "none", border: `1px solid ${isOn ? ch.color : C.b1}`, borderRadius: 8, padding: "1px 6px", fontSize: 6, color: isOn ? ch.color : C.dm, fontFamily: C.dt, cursor: "pointer", transition: "all 0.1s" }}>{ch.label}</button>
          ); })}
        </div>
      </div>)}
      {selected.length > 0 ? <TimeSeriesCanvas history={history} channels={selected} height={compact ? 100 : 130} tick={tick} /> : (
        <div style={{ height: 36, display: "flex", alignItems: "center", justifyContent: "center", color: C.dm, fontSize: 8, fontFamily: C.dt }}>Select channels to plot</div>
      )}
      {selected.length > 0 && <div style={{ display: "flex", gap: 4, padding: "2px 0", flexWrap: "wrap" }}>
        {selected.map(k => { const ch = CHANNEL_MAP[k]; return ch ? <div key={k} style={{ display: "flex", alignItems: "center", gap: 2 }}><div style={{ width: 6, height: 2, background: ch.color, borderRadius: 1 }} /><span style={{ fontSize: 5, color: ch.color, fontFamily: C.dt }}>{ch.label}</span></div> : null; })}
      </div>}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// LIVE GRAPH PANEL — manages up to 8 graph slots
// ═════════════════════════════════════════════════════════════════════════════
export default function LiveGraphPanel({ history, tick }) {
  const [graphs, setGraphs] = useState([1]);
  const nextId = useRef(2);
  const addGraph = useCallback(() => { if (graphs.length >= 8) return; setGraphs(p => [...p, nextId.current++]); }, [graphs.length]);
  const removeGraph = useCallback(id => setGraphs(p => p.filter(g => g !== id)), []);
  const histArr = history.current || [];

  return (<div>
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
      <span style={{ fontSize: 7, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, textTransform: "uppercase" }}>CONFIGURABLE GRAPHS</span>
      <button onClick={addGraph} style={{ background: `${C.gn}15`, border: `1px solid ${C.gn}30`, borderRadius: 4, padding: "1px 8px", fontSize: 8, color: C.gn, fontFamily: C.dt, cursor: "pointer" }}>+ Add Graph</button>
      <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>{graphs.length}/8 · {CHANNELS.length} channels · {histArr.length} frames</span>
    </div>
    {graphs.map(id => <GraphSlot key={id} id={id} history={histArr} tick={tick} onRemove={graphs.length > 1 ? () => removeGraph(id) : null} />)}
  </div>);
}

export { TimeSeriesCanvas, GraphSlot };