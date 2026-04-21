// ═══════════════════════════════════════════════════════════════════════════
// src/SuspensionModule.jsx — Project-GP Dashboard v4.2
// ═══════════════════════════════════════════════════════════════════════════
// Drop-in replacement for the "Suspension" tab in the main dashboard.
// Uses the existing theme (C, GL from theme.js) and KPI from components.jsx.
// Chart categories are sub-tabs WITHIN this module — the main sidebar is
// untouched (that's App.jsx's responsibility).
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { C, GL } from "./theme.js";
import { KPI } from "./components.jsx";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, ReferenceLine, ReferenceArea,
  Legend, AreaChart, Area, ComposedChart,
} from "recharts";

// ═══════════════════════════════════════════════════════════════════════════
// VEHICLE PARAMETERS (metres)
// ═══════════════════════════════════════════════════════════════════════════
const V = {
  wb: 1.55, lf: 0.8525, lr: 0.6975,
  tF: 1.22, tR: 1.18, hCG: 0.28,
  wR: 0.2286, tW: 0.205, maxSt: 0.42,
  tubFront: 0.65, tubRear: -0.32,
  tubTopHW: 0.28, tubBotHW: 0.28,
  tubFloor: -0.22, tubTop: 0.10,
  noseTip: 1.12,
  cpFront: 0.12, cpRear: -0.25, cpHW: 0.20,
  rhX: -0.26, rhTopY: 0.48, rhBaseHW: 0.25, rhTopHW: 0.13,
  rsfRear: -0.78, rsfHW: 0.22, rsfH: 0.18,
  fwX: 1.14, fwY: -0.24, fwSpan: 1.36, fwChord: 0.26,
  rwX: -0.82, rwMainY: 0.50, rwSpan: 0.88, rwChord: 0.30,
  rwEpBot: 0.10, rwEpTop: 0.56,
  uprH: 0.12, monoH: 0.32,
};

// Suspension model constants for derived calcs
const SP = {
  mass: 300, kF: 35000, kR: 38000,
  kARB_F: 800, kARB_R: 600,
  cLS_F: 2500, cLS_R: 2800, cHS_F: 900, cHS_R: 1000,
  vKnee: 0.15, bsGap: 0.025, bsK: 150000,
  camGainF: 0.65, camGainR: 0.45, bsCoeff: 0.015,
  MR_F: 1.14, MR_R: 1.16, muPeak: 1.6,
};

// ═══════════════════════════════════════════════════════════════════════════
// EXACT TER27 - VELIS HARDPOINTS (Converted to Three.js [X=Long, Y=Vert, Z=Lat])
// ═══════════════════════════════════════════════════════════════════════════
const TER27_HP = {
  front: {
    lca_f: new THREE.Vector3(0.160, 0.110, 0.160),
    lca_r: new THREE.Vector3(-0.160, 0.130, 0.160),
    uca_f: new THREE.Vector3(0.120, 0.267, 0.245),
    uca_r: new THREE.Vector3(-0.120, 0.258, 0.245),
    lca_o: new THREE.Vector3(0.002, 0.122, 0.583),
    uca_o: new THREE.Vector3(-0.011, 0.280, 0.555),
    tie_i: new THREE.Vector3(0.050, 0.144, 0.144),
    tie_o: new THREE.Vector3(0.070, 0.150, 0.571),
  },
  rear: {
    lca_f: new THREE.Vector3(0.150, 0.119, 0.195),
    lca_r: new THREE.Vector3(-0.150, 0.110, 0.195),
    uca_f: new THREE.Vector3(0.150, 0.245, 0.195),
    uca_r: new THREE.Vector3(-0.150, 0.272, 0.195),
    lca_o: new THREE.Vector3(0.0, 0.112, 0.585),
    uca_o: new THREE.Vector3(0.0, 0.290, 0.537),
    tie_i: new THREE.Vector3(-0.080, 0.180, 0.195),
    tie_o: new THREE.Vector3(-0.095, 0.230, 0.530),
  }
};
// ═══════════════════════════════════════════════════════════════════════════
// MANEUVER GENERATORS
// ═══════════════════════════════════════════════════════════════════════════
function genData(type, dur = 8, n = 480) {
  const o = [];
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * dur, p = t / dur;
    let roll = 0, pitch = 0, steer = 0, spd = 12, ay = 0, ax = 0;
    switch (type) {
      case "lap": {
        const q = p * 4 * Math.PI;
        ay = 1.4 * Math.sin(q * 0.4) + 0.5 * Math.sin(q * 1.1);
        ax = 0.8 * Math.cos(q * 0.6);
        roll = ay * 2.1; pitch = -ax * 1.5;
        steer = Math.max(-V.maxSt, Math.min(V.maxSt, 0.26 * Math.sin(q * 0.4) + 0.10 * Math.sin(q * 1.1)));
        spd = 14 + 6 * Math.sin(q * 0.3); break;
      }
      case "skidpad": {
        const e = p < 0.12 ? p / 0.12 : p > 0.88 ? (1 - p) / 0.12 : 1;
        ay = 1.5 * e; roll = 3.1 * e; steer = 0.22 * e; pitch = -0.25 * e; spd = 11; break;
      }
      case "brake": {
        const e = Math.sin(p * Math.PI), ab = 1 + 0.05 * Math.sin(p * 80);
        ax = -2.4 * e * ab; pitch = 3.6 * e; roll = 0.12 * Math.sin(p * 5);
        steer = 0.015 * Math.sin(p * 2); spd = 25 * (1 - p * 0.75); break;
      }
      case "accel": {
        const e = p < 0.08 ? p / 0.08 : 1, d = Math.max(0, 1 - p * 0.55);
        ax = 1.8 * e * d; pitch = -2.4 * e * d; spd = 2 + 28 * p; roll = 0.08 * Math.sin(p * 3); break;
      }
      case "chicane": {
        const f = p * Math.PI * 6;
        ay = 1.8 * Math.sin(f); roll = 3.4 * Math.sin(f);
        steer = Math.max(-V.maxSt, Math.min(V.maxSt, 0.28 * Math.sin(f)));
        pitch = 0.3 * Math.sin(f * 2); ax = -0.35 * Math.cos(f * 0.5); spd = 15; break;
      }
    }
    o.push({ t, roll, pitch, steer, spd, ay, ax });
  }
  return o;
}

// ═══════════════════════════════════════════════════════════════════════════
// THREE.JS HELPERS (unchanged from last iteration)
// ═══════════════════════════════════════════════════════════════════════════
const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
function tube(a, b, r, mat) { const d = v3(0,0,0).subVectors(b, a), l = d.length(); if (l < 1e-5) return new THREE.Group(); const g = new THREE.CylinderGeometry(r, r, l, 6); const m = new THREE.Mesh(g, mat); m.castShadow = true; m.position.lerpVectors(a, b, 0.5); m.quaternion.setFromUnitVectors(v3(0,1,0), d.normalize()); return m; }
function dynTube(a, b, r, mat) { const l = a.distanceTo(b); const g = new THREE.CylinderGeometry(r, r, l, 6); const m = new THREE.Mesh(g, mat); m.castShadow = true; m.userData._nl = l; repos(m, a, b); return m; }
function repos(m, a, b) { const d = v3(0,0,0).subVectors(b, a), l = d.length(); if (l < 1e-5) return; m.position.lerpVectors(a, b, 0.5); m.quaternion.setFromUnitVectors(v3(0,1,0), d.normalize()); m.scale.y = l / (m.userData._nl || l); }
function coilGeo(r, coils, wire) { const pts = [], N = coils * 16; for (let i = 0; i <= N; i++) { const f = i / N, a = f * coils * Math.PI * 2; pts.push(v3(r * Math.cos(a), f - 0.5, r * Math.sin(a))); } return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), coils * 12, wire, 5, false); }
function wingGeo(chord, thick, span) { const s = new THREE.Shape(); s.moveTo(0, 0); s.bezierCurveTo(chord*0.12, thick*0.7, chord*0.28, thick, chord*0.38, thick); s.bezierCurveTo(chord*0.55, thick*0.88, chord*0.85, thick*0.3, chord, thick*0.04); s.lineTo(chord, -thick*0.04); s.bezierCurveTo(chord*0.85, -thick*0.12, chord*0.5, -thick*0.32, chord*0.3, -thick*0.28); s.bezierCurveTo(chord*0.12, -thick*0.18, 0.005, -thick*0.04, 0, 0); const g = new THREE.ExtrudeGeometry(s, { depth: span, bevelEnabled: false }); g.translate(-chord*0.3, 0, -span/2); return g; }
function loftBody(stations, nPts, mat) { const positions = [], indices = [], N = nPts; for (const st of stations) { for (let p = 0; p < N; p++) { const a = (p/N)*Math.PI*2, ca = Math.cos(a), sa = Math.sin(a), exp = 4.0, r = Math.pow(Math.pow(Math.abs(ca), exp) + Math.pow(Math.abs(sa), exp), -1/exp); positions.push(st.x, st.cy + sa*r*st.hh, ca*r*st.hw); } } for (let s = 0; s < stations.length - 1; s++) { for (let p = 0; p < N; p++) { const p2 = (p+1)%N, a = s*N+p, b = s*N+p2, c = (s+1)*N+p, d = (s+1)*N+p2; indices.push(a,c,b,b,c,d); } } const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3)); geo.setIndex(indices); geo.computeVertexNormals(); const m = new THREE.Mesh(geo, mat); m.castShadow = true; return m; }

// ═══════════════════════════════════════════════════════════════════════════
// SCENE BUILDERS (materials, lights, ground, chassis, wings, corners, coilovers)
// Condensed — same logic as standalone SuspensionViz.jsx
// ═══════════════════════════════════════════════════════════════════════════
function buildMats() { const s = (c, met, rgh, o = {}) => new THREE.MeshStandardMaterial({ color: c, metalness: met, roughness: rgh, ...o }); return { body: s(0x1e2430, 0.32, 0.48, { side: THREE.DoubleSide }), nose: s(0x181e2a, 0.25, 0.52, { side: THREE.DoubleSide }), cockpit: s(0x0a0e16, 0.20, 0.70), accent: s(0x00b8e6, 0.55, 0.30, { emissive: 0x004050, emissiveIntensity: 0.25 }), steel: s(0x6a7288, 0.85, 0.20), stLt: s(0x8892a8, 0.70, 0.25), push: s(0xc8b030, 0.70, 0.30), spring: s(0x00d4ff, 0.45, 0.30, { emissive: 0x003848, emissiveIntensity: 0.40 }), dampB: s(0x3e4650, 0.80, 0.20), dampS: s(0x98a0b0, 0.90, 0.10), tire: s(0x161616, 0.00, 0.94), rim: s(0x808890, 0.88, 0.12), wingCF: s(0x141820, 0.25, 0.50), wingEP: s(0x1a2030, 0.30, 0.45), floor: s(0x0c1018, 0.15, 0.70), ground: s(0x080c14, 0.00, 0.95), head: s(0x222830, 0.20, 0.60) }; }
function buildLights(sc) { sc.add(new THREE.AmbientLight(0x354060, 0.50)); const k = new THREE.DirectionalLight(0xdde8ff, 1.8); k.position.set(4, 6, 3); k.castShadow = true; k.shadow.mapSize.set(2048, 2048); Object.assign(k.shadow.camera, { near: 0.5, far: 20, left: -5, right: 5, top: 5, bottom: -5 }); sc.add(k); sc.add(new THREE.DirectionalLight(0x5070a0, 0.40).translateX(-4).translateY(3).translateZ(-3)); }
function buildGround(sc, mat) { const g = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), mat); g.rotation.x = -Math.PI/2; g.receiveShadow = true; sc.add(g); const gr = new THREE.GridHelper(24, 48, 0x182035, 0x182035); gr.material.opacity = 0.15; gr.material.transparent = true; gr.position.y = 0.001; sc.add(gr); const road = new THREE.Group(); const asph = new THREE.Mesh(new THREE.PlaneGeometry(30, 4.0), new THREE.MeshStandardMaterial({ color: 0x0e1220, roughness: 0.92 })); asph.rotation.x = -Math.PI/2; asph.position.y = 0.0015; road.add(asph); const lM = new THREE.MeshBasicMaterial({ color: 0x2a3550, transparent: true, opacity: 0.5 }); for (const s of [-1, 1]) { for (let i = 0; i < 30; i++) { const d = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.004, 0.06), lM); d.position.set(-15+i*1.1, 0.003, s*1.8); road.add(d); } } sc.add(road); return { grid: gr, road }; }

function buildChassis(body, M) {
  const fl = V.tubFloor, tp = V.tubTop, cy = (fl+tp)/2, hh = (tp-fl)/2;
  const stations = [
    { x: V.noseTip, hw: 0.03, hh: 0.04, cy: cy-0.06 },
    { x: V.noseTip-0.08, hw: 0.06, hh: 0.07, cy: cy-0.04 },
    { x: V.noseTip-0.20, hw: 0.12, hh: 0.10, cy: cy-0.02 },
    { x: V.tubFront+0.15, hw: 0.22, hh: 0.13, cy },
    { x: V.tubFront+0.05, hw: V.tubTopHW, hh: hh*0.95, cy },
    { x: V.tubFront, hw: V.tubTopHW, hh, cy },
    { x: V.cpFront+0.05, hw: V.tubTopHW, hh, cy },
    { x: V.cpRear-0.05, hw: V.tubTopHW, hh, cy },
    { x: V.tubRear, hw: V.tubTopHW*0.95, hh: hh*0.95, cy },
  ];
  body.add(loftBody(stations, 20, M.body));
  const cpLen = V.cpFront-V.cpRear;
  body.add((() => { const m = new THREE.Mesh(new THREE.BoxGeometry(cpLen, 0.07, V.cpHW*2), M.cockpit); m.position.set((V.cpFront+V.cpRear)/2, tp-0.015, 0); return m; })());
  body.add((() => { const m = new THREE.Mesh(new THREE.BoxGeometry(cpLen+0.04, 0.018, V.cpHW*2+0.03), M.body); m.position.set((V.cpFront+V.cpRear)/2, tp+0.005, 0); return m; })());
  body.add((() => { const m = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.11, 0.17), M.head); m.position.set(V.cpRear-0.05, tp+0.055, 0); return m; })());
  const tubLen = V.tubFront-V.tubRear;
  for (const s of [-1, 1]) { const m = new THREE.Mesh(new THREE.BoxGeometry(tubLen*0.4, 0.025, 0.010), M.accent); m.position.set(V.tubFront-tubLen*0.35, cy, s*(V.tubTopHW+0.006)); body.add(m); }
  body.add((() => { const m = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.005, V.tubTopHW*1.8), M.floor); m.position.set(0.15, fl-0.003, 0); return m; })());
  const rhBase = tp, rhTop = V.rhTopY;
  for (const s of [-1, 1]) body.add(tube(v3(V.rhX, rhBase, s*V.rhBaseHW), v3(V.rhX, rhTop, s*V.rhTopHW), 0.013, M.steel));
  body.add(tube(v3(V.rhX, rhTop, -V.rhTopHW), v3(V.rhX, rhTop, V.rhTopHW), 0.013, M.steel));
  for (const s of [-1, 1]) { body.add(tube(v3(V.rhX, rhTop*0.88, s*V.rhTopHW), v3(V.rhX-0.25, rhBase*0.85, s*V.rhBaseHW*0.65), 0.009, M.steel)); body.add(tube(v3(V.rhX, rhTop, s*V.rhTopHW), v3(V.rhX-0.32, rhBase, s*V.rhBaseHW*0.4), 0.007, M.steel)); }
  const rsF = V.tubRear, rsR = V.rsfRear, sw = V.rsfHW, sh = V.rsfH;
  for (const s of [-1, 1]) { body.add(tube(v3(rsF, fl, s*sw), v3(rsR, fl, s*sw), 0.008, M.steel)); body.add(tube(v3(rsF, fl+sh, s*sw*0.82), v3(rsR, fl+sh*0.72, s*sw*0.65), 0.007, M.steel)); for (const f of [0.3, 0.7]) { const x = rsF+(rsR-rsF)*f; body.add(tube(v3(x, fl, s*sw), v3(x, fl+sh*(0.88-f*0.18), s*sw*(0.88-f*0.10)), 0.006, M.steel)); } }
  for (const f of [0, 0.5, 1]) { const x = rsF+(rsR-rsF)*f; body.add(tube(v3(x, fl, -sw), v3(x, fl, sw), 0.006, M.steel)); }
}
function buildFW(body, M) { const { fwSpan: sp, fwChord: ch } = V, wx = V.fwX, wy = V.fwY; const mg = wingGeo(ch, 0.020, sp); mg.rotateY(Math.PI); const mm = new THREE.Mesh(mg, M.wingCF); mm.position.set(wx, wy, 0); mm.rotation.z = 0.08; mm.castShadow = true; body.add(mm); const fg = wingGeo(ch*0.35, 0.013, sp*0.93); fg.rotateY(Math.PI); const fm = new THREE.Mesh(fg, M.wingCF); fm.position.set(wx-ch*0.44, wy+0.032, 0); fm.rotation.z = 0.20; fm.castShadow = true; body.add(fm); for (const s of [-1, 1]) { const m = new THREE.Mesh(new THREE.BoxGeometry(ch*1.3, 0.075, 0.004), M.wingEP); m.position.set(wx-ch*0.2, wy+0.015, s*sp*0.505); body.add(m); } for (const s of [-1, 1]) body.add(tube(v3(wx-ch*0.15, wy+0.04, s*sp*0.15), v3(V.noseTip-0.22, V.tubFloor+0.10, s*0.04), 0.005, M.body)); }
function buildRW(body, M) { const { rwSpan: sp, rwChord: ch } = V, wx = V.rwX, wy = V.rwMainY; const mg = wingGeo(ch, 0.024, sp); mg.rotateY(Math.PI); body.add((() => { const m = new THREE.Mesh(mg, M.wingCF); m.position.set(wx, wy, 0); m.rotation.z = 0.14; m.castShadow = true; return m; })()); const fg = wingGeo(ch*0.30, 0.015, sp*0.96); fg.rotateY(Math.PI); body.add((() => { const m = new THREE.Mesh(fg, M.wingCF); m.position.set(wx-ch*0.30, wy+0.055, 0); m.rotation.z = 0.30; m.castShadow = true; return m; })()); for (const s of [-1, 1]) { const epH = V.rwEpTop-V.rwEpBot; const m = new THREE.Mesh(new THREE.BoxGeometry(ch*1.5, epH, 0.005), M.wingEP); m.position.set(wx-ch*0.05, (V.rwEpBot+V.rwEpTop)/2, s*sp*0.505); m.castShadow = true; body.add(m); } for (const s of [-1, 1]) body.add(tube(v3(wx+ch*0.35, V.rwEpBot+0.02, s*sp*0.5), v3(V.rsfRear+0.08, V.tubFloor+V.rsfH*0.3, s*V.rsfHW*0.8), 0.007, M.steel)); }

function buildCorner(sc, body, M, name, isF, yS) {
  const hp = isF ? TER27_HP.front : TER27_HP.rear;
  
  // Wheel Group Center
  const wc = new THREE.Vector3(0, V.wR, yS * 0.615); 
  const sg = new THREE.Group(); 
  sg.position.set(0, 0, 0); // Absolute positioning based on HP now
  sc.add(sg);

  // Wheel meshes (Tire & Rim)
  const wg = new THREE.Group(); wg.position.copy(wc); sg.add(wg);
  wg.add(new THREE.Mesh(new THREE.CylinderGeometry(V.wR*0.58, V.wR*0.58, V.tW*0.22, 20).rotateX(Math.PI/2), M.rim));
  const tireM = new THREE.Mesh(new THREE.TorusGeometry(V.wR, V.tW*0.38, 14, 32), M.tire); 
  tireM.castShadow = true; wg.add(tireM);

  // Upright Group
  const ug = new THREE.Group(); 
  const uprCenter = hp.lca_o.clone().lerp(hp.uca_o, 0.5);
  uprCenter.z *= yS; // mirror for right side
  ug.position.copy(uprCenter);
  ug.add(new THREE.Mesh(new THREE.BoxGeometry(0.032, V.uprH, 0.040), M.stLt)); 
  sg.add(ug);

  // Chassis Inboard Points (mirrored by yS)
  const lInLocal = [
    new THREE.Vector3(hp.lca_f.x, hp.lca_f.y, hp.lca_f.z * yS),
    new THREE.Vector3(hp.lca_r.x, hp.lca_r.y, hp.lca_r.z * yS)
  ];
  const uInLocal = [
    new THREE.Vector3(hp.uca_f.x, hp.uca_f.y, hp.uca_f.z * yS),
    new THREE.Vector3(hp.uca_r.x, hp.uca_r.y, hp.uca_r.z * yS)
  ];

  // Upright Outboard Points
  const lO0 = new THREE.Vector3(hp.lca_o.x, hp.lca_o.y, hp.lca_o.z * yS);
  const uO0 = new THREE.Vector3(hp.uca_o.x, hp.uca_o.y, hp.uca_o.z * yS);

  // Tie Rod Points
  const trIn = new THREE.Vector3(hp.tie_i.x, hp.tie_i.y, hp.tie_i.z * yS);
  const trOut = new THREE.Vector3(hp.tie_o.x, hp.tie_o.y, hp.tie_o.z * yS);

  // Generate Tubes (A-Arms)
  const lA = lInLocal.map((ip) => { 
    const m = dynTube(ip, lO0, 0.007, M.steel); sc.add(m); 
    return { m, ip: ip.clone(), local: ip.clone() }; 
  });
  const uA = uInLocal.map((ip) => { 
    const m = dynTube(ip, uO0, 0.006, M.stLt); sc.add(m); 
    return { m, ip: ip.clone(), local: ip.clone() }; 
  });

  // Tie Rod Tube
  const tieRod = dynTube(trIn, trOut, 0.005, M.accent); sc.add(tieRod);

  // Pushrod (connecting to LCA for front, UCA for rear as per your CSV)
  const pB0 = isF ? lO0.clone().lerp(lInLocal[0], 0.25) : uO0.clone().lerp(uInLocal[0], 0.25);
  const bcLocal = new THREE.Vector3(isF ? 0.0 : -V.wb, V.tubTop + 0.05, 0.15 * yS); // Approx rocker location
  const pM = dynTube(pB0, bcLocal, 0.005, M.push); sc.add(pM);

  return { name, isF, yS, sg, wg, ug, lA, uA, tieRod, trIn, trOut, lO0, uO0, pM, pB0, bc0: bcLocal, bcLocal, ws: 0 };
}

function buildCoilover(body, M, c) {
  const springLen = 0.16, tubTop = V.tubTop;
  const rockerX = c.isF ? Math.min(c.xA, V.tubFront-0.05) : Math.max(c.xA, V.tubRear+0.05);
  const rockerY = tubTop+0.015, tiltRad = -5*Math.PI/180;
  let rockerPt, anchorPt, dOffVec;
  if (c.isF) { const rZ = c.yS*(V.tubTopHW-0.03); rockerPt = v3(rockerX, rockerY, rZ); anchorPt = v3(rockerX, rockerY+springLen*Math.sin(tiltRad), rZ-c.yS*springLen*Math.cos(tiltRad)); dOffVec = v3(0.020, 0, 0); }
  else { const rZ = c.yS*(V.rsfHW-0.03); rockerPt = v3(rockerX, rockerY, rZ); anchorPt = v3(rockerX+springLen*Math.cos(tiltRad), rockerY+springLen*Math.sin(tiltRad), rZ); dOffVec = v3(0, 0, c.yS*0.020); }
  const nomD = rockerPt.distanceTo(anchorPt), axis = v3(0,0,0).subVectors(anchorPt, rockerPt).normalize();
  body.add((() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.009, 8, 6), M.stLt); m.position.copy(rockerPt); return m; })());
  body.add((() => { const m = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.014, 0.018), M.steel); m.position.copy(anchorPt); return m; })());
  const sGeo = coilGeo(0.012, 8, 0.0022); const sM = new THREE.Mesh(sGeo, M.spring); sM.castShadow = true;
  sM.position.lerpVectors(rockerPt, anchorPt, 0.5); sM.quaternion.setFromUnitVectors(v3(0,1,0), axis); sM.scale.set(1, nomD, 1); body.add(sM);
  const dR = rockerPt.clone().add(dOffVec), dA = anchorPt.clone().add(dOffVec), dLen = nomD*0.48;
  const dBM = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, dLen, 8), M.dampB); dBM.castShadow = true;
  const dSM = new THREE.Mesh(new THREE.CylinderGeometry(0.0045, 0.0045, nomD*0.42, 8), M.dampS);
  const dGr = new THREE.Group(); dBM.position.y = dLen*0.26; dSM.position.y = -dLen*0.20; dGr.add(dBM); dGr.add(dSM);
  dGr.position.lerpVectors(dR, dA, 0.5); dGr.quaternion.setFromUnitVectors(v3(0,1,0), axis); body.add(dGr);
  const bcL = v3(c.bc0.x, c.bc0.y-V.hCG, c.bc0.z); body.add(tube(bcL, rockerPt, 0.004, M.push));
  return { sM, dGr, dSM, base: rockerPt, top: anchorPt, axis, nomD, dOffVec: dOffVec.clone() };
}

function tick(bg, corners, coils, fr, dt) {
  const rR = fr.roll*Math.PI/180, pR = fr.pitch*Math.PI/180;
  bg.rotation.z = rR; bg.rotation.x = pR; bg.position.y = V.hCG-0.0015*(Math.abs(fr.roll)+Math.abs(fr.pitch));
  bg.updateMatrixWorld(true);
  const tvl = {};
  for (const c of Object.values(corners)) {
    const tw = c.isF ? V.tF : V.tR, raw = -(tw/2)*c.yS*rR+c.xA*pR, vis = raw*4.0; tvl[c.name] = raw*1000;
    if (c.isF) c.sg.rotation.y = fr.steer; c.ws += (fr.spd/V.wR)*dt; c.wg.rotation.z = c.ws;
    c.ug.position.y = V.wR+vis*0.12;
    const lo = c.lO0.clone(); lo.y += vis*0.12; const uo = c.uO0.clone(); uo.y += vis*0.12;
    c.lA.forEach(a => { repos(a.m, bg.localToWorld(a.local.clone()), lo); });
    c.uA.forEach(a => { repos(a.m, bg.localToWorld(a.local.clone()), uo); });
    repos(c.pM, lo.clone().lerp(uo, 0.25), bg.localToWorld(c.bcLocal.clone()));
    const ib = coils[c.name]; if (ib) { const nd = Math.max(ib.nomD*0.6, Math.min(ib.nomD*1.18, ib.nomD-vis)); const newTop = ib.base.clone().add(ib.axis.clone().multiplyScalar(nd)); ib.sM.position.lerpVectors(ib.base, newTop, 0.5); ib.sM.scale.set(1, nd, 1); ib.dGr.position.lerpVectors(ib.base.clone().add(ib.dOffVec), newTop.clone().add(ib.dOffVec), 0.5); ib.dSM.scale.y = nd/ib.nomD; }
  }
  return tvl;
}

// ═══════════════════════════════════════════════════════════════════════════
// CHART SECTION — sub-tabs within the Suspension module
// ═══════════════════════════════════════════════════════════════════════════
const CHART_CATS = [
  { key: "kinematics", label: "Kinematics", icon: "△" },
  { key: "dynamics", label: "Dynamics", icon: "◆" },
  { key: "vehicle", label: "Vehicle", icon: "⬡" },
  { key: "tire", label: "Tire", icon: "⊗" },
  { key: "energy", label: "Energy", icon: "⊕" },
  { key: "modal", label: "Modal", icon: "∿" },
];

const CA = "#2a3550";
const CG = "rgba(0,192,242,0.06)";
const TT = { contentStyle: { background: "#0e1420", border: `1px solid ${C.b1}`, borderRadius: 6, fontSize: 10, fontFamily: C.dt, color: C.br }, itemStyle: { color: C.br, fontSize: 10 } };
const CL = { fl: C.cy, fr: C.gn, rl: C.am, rr: C.red };

const CC = ({ title, children }) => (
  <div style={{ ...GL, padding: "12px 10px 6px" }}>
    <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.8, color: C.dm, fontFamily: C.dt, marginBottom: 8, paddingLeft: 4 }}>{title}</div>
    {children}
  </div>
);

function computeDerived(history) {
  return history.map((h, i) => {
    const prev = i > 0 ? history[i-1] : h, dt = h.t-prev.t || 0.017;
    const vFL = (h.zFL-prev.zFL)/dt, vFR = (h.zFR-prev.zFR)/dt, vRL = (h.zRL-prev.zRL)/dt, vRR = (h.zRR-prev.zRR)/dt;
    const tAvg = (V.tF+V.tR)/2, latLT = h.ay*SP.mass*SP.hCG||V.hCG/tAvg*9.81, lonLT = h.ax*SP.mass*(SP.hCG||V.hCG)/V.wb*9.81;
    const Fz0 = SP.mass*9.81/4;
    const Fz_FL = Fz0-latLT/2+lonLT/2, Fz_FR = Fz0+latLT/2+lonLT/2, Fz_RL = Fz0-latLT/2-lonLT/2, Fz_RR = Fz0+latLT/2-lonLT/2;
    const camFL = -2.0+SP.camGainF*h.roll, camFR = -2.0-SP.camGainF*h.roll, camRL = -1.5+SP.camGainR*h.roll, camRR = -1.5-SP.camGainR*h.roll;
    const bsFL = h.zFL*SP.bsCoeff, bsFR = -h.zFR*SP.bsCoeff, bsRL = h.zRL*SP.bsCoeff*0.3, bsRR = -h.zRR*SP.bsCoeff*0.3;
    const latLT_F = h.ay*SP.mass*V.hCG/V.tF*9.81*0.52, latLT_R = h.ay*SP.mass*V.hCG/V.tR*9.81*0.48;
    const LLTD = Math.abs(latLT_F+latLT_R) > 1 ? (Math.abs(latLT_F)/(Math.abs(latLT_F)+Math.abs(latLT_R)))*100 : 50;
    const rollGrad = Math.abs(h.ay) > 0.05 ? Math.abs(h.roll/h.ay) : 0;
    const steerDeg = (h.steer||0)*180/Math.PI;
    const Fs_FL = (h.zFL/1000)*SP.kF, Fs_FR = (h.zFR/1000)*SP.kF, Fs_RL = (h.zRL/1000)*SP.kR, Fs_RR = (h.zRR/1000)*SP.kR;
    const dF = (v, isF) => { const va = Math.abs(v)/1000; const cL = isF ? SP.cLS_F : SP.cLS_R, cH = isF ? SP.cHS_F : SP.cHS_R; return va < SP.vKnee ? cL*va : cL*SP.vKnee+cH*(va-SP.vKnee); };
    const Fd_FL = dF(vFL, true), Fd_FR = dF(vFR, true), Fd_RL = dF(vRL, false), Fd_RR = dF(vRR, false);
    const Pd_FL = Fd_FL*Math.abs(vFL)/1000, Pd_FR = Fd_FR*Math.abs(vFR)/1000, Pd_RL = Fd_RL*Math.abs(vRL)/1000, Pd_RR = Fd_RR*Math.abs(vRR)/1000;
    const Es = 0.5*SP.kF*(h.zFL/1000)**2+0.5*SP.kF*(h.zFR/1000)**2+0.5*SP.kR*(h.zRL/1000)**2+0.5*SP.kR*(h.zRR/1000)**2;
    const fU = Fz => { const Fy = Math.abs(h.ay)*SP.mass/4*9.81, Fx = Math.abs(h.ax)*SP.mass/4*9.81; return Math.min(100, Math.sqrt(Fy*Fy+Fx*Fx)/(Math.max(50, Fz)*SP.muPeak)*100); };
    const arbF = h.roll*SP.kARB_F, arbR = h.roll*SP.kARB_R;
    return { ...h, t: +h.t.toFixed(2), vFL: +vFL.toFixed(1), vFR: +vFR.toFixed(1), vRL: +vRL.toFixed(1), vRR: +vRR.toFixed(1), Fz_FL: +Fz_FL.toFixed(0), Fz_FR: +Fz_FR.toFixed(0), Fz_RL: +Fz_RL.toFixed(0), Fz_RR: +Fz_RR.toFixed(0), camFL: +camFL.toFixed(2), camFR: +camFR.toFixed(2), camRL: +camRL.toFixed(2), camRR: +camRR.toFixed(2), bsFL: +bsFL.toFixed(3), bsFR: +bsFR.toFixed(3), bsRL: +bsRL.toFixed(3), bsRR: +bsRR.toFixed(3), LLTD: +LLTD.toFixed(1), rollGrad: +rollGrad.toFixed(2), steerDeg: +steerDeg.toFixed(1), Fs_FL: +Fs_FL.toFixed(0), Fs_FR: +Fs_FR.toFixed(0), Fs_RL: +Fs_RL.toFixed(0), Fs_RR: +Fs_RR.toFixed(0), Fd_FL: +Fd_FL.toFixed(1), Fd_FR: +Fd_FR.toFixed(1), Fd_RL: +Fd_RL.toFixed(1), Fd_RR: +Fd_RR.toFixed(1), Pd_FL: +Pd_FL.toFixed(2), Pd_FR: +Pd_FR.toFixed(2), Pd_RL: +Pd_RL.toFixed(2), Pd_RR: +Pd_RR.toFixed(2), Es_tot: +Es.toFixed(2), Pd_tot: +(Pd_FL+Pd_FR+Pd_RL+Pd_RR).toFixed(2), muFL: +fU(Fz_FL).toFixed(1), muFR: +fU(Fz_FR).toFixed(1), muRL: +fU(Fz_RL).toFixed(1), muRR: +fU(Fz_RR).toFixed(1), arbF: +arbF.toFixed(1), arbR: +arbR.toFixed(1), heave: +((h.zFL+h.zFR+h.zRL+h.zRR)/4).toFixed(1), rollMode: +((h.zFL-h.zFR+h.zRL-h.zRR)/4).toFixed(1), pitchMode: +((h.zFL+h.zFR-h.zRL-h.zRR)/4).toFixed(1), warp: +((h.zFL-h.zFR-h.zRL+h.zRR)/4).toFixed(1), compSteerF: +((h.ay || 0) * 0.12).toFixed(3),compSteerR: +((h.ay || 0) * 0.28).toFixed(3),totalSteerF: +((h.steer || 0) * 180 / Math.PI + (h.ay || 0) * 0.12).toFixed(2),totalSteerR: +((h.ay || 0) * 0.28).toFixed(2), };
  });
}

function SuspCharts({ history, activeCat }) {
  if (!history || history.length < 10) return <div style={{ padding: 20, color: C.dm, fontFamily: C.dt, fontSize: 10 }}>Accumulating telemetry data…</div>;
  const derived = useMemo(() => computeDerived(history), [history]);
  const ggData = useMemo(() => history.filter((_, i) => i % 3 === 0).map(h => ({ ax: +h.ax.toFixed(2), ay: +h.ay.toFixed(2) })), [history]);
  const cp = { syncId: "susp", data: derived };
  const xP = { dataKey: "t", tick: { fontSize: 8, fill: CA }, stroke: CA, tickLine: false };
  const yP = l => ({ tick: { fontSize: 8, fill: CA }, stroke: CA, tickLine: false, label: { value: l, angle: -90, position: "insideLeft", style: { fontSize: 8, fill: C.dm, fontFamily: C.dt } } });
  const gP = { stroke: CG, strokeDasharray: "3 3" };
  const leg = { iconSize: 8, wrapperStyle: { fontSize: 9, fontFamily: C.dt, color: C.dm } };
  const L = (dk, c, n, w = 1.5) => <Line key={dk} dataKey={dk} stroke={c} dot={false} strokeWidth={w} name={n} />;

  // Build F-v operating cloud data (scatter of velocity vs force per corner)
  const fvCloud = useMemo(() => {
    const pts = [];
    for (let i = 2; i < derived.length; i += 3) {
      const d = derived[i];
      pts.push({ v: d.vFL, F: d.Fd_FL, corner: "FL" });
      pts.push({ v: d.vFR, F: d.Fd_FR, corner: "FR" });
      pts.push({ v: d.vRL, F: d.Fd_RL, corner: "RL" });
      pts.push({ v: d.vRR, F: d.Fd_RR, corner: "RR" });
    }
    return pts;
  }, [derived]);

  // Build digressive characteristic curve for overlay
  const fvCurve = useMemo(() => {
    const pts = [];
    for (let v = 0; v <= 500; v += 5) {
      const vm = v / 1000;
      const Ff = vm < SP.vKnee ? SP.cLS_F * vm : SP.cLS_F * SP.vKnee + SP.cHS_F * (vm - SP.vKnee);
      const Fr = vm < SP.vKnee ? SP.cLS_R * vm : SP.cLS_R * SP.vKnee + SP.cHS_R * (vm - SP.vKnee);
      pts.push({ v, Ff: +Ff.toFixed(1), Fr: +Fr.toFixed(1) });
    }
    return pts;
  }, []);

  // Motion ratio curve data
  const mrCurve = useMemo(() => {
    const pts = [];
    for (let z = -30; z <= 30; z += 1) {
      const zm = z / 1000;
      // Polynomial MR: MR(z) = MR0 + MR1*z + MR2*z^2
      const MR_f = SP.MR_F + 0.8 * zm - 12 * zm * zm;
      const MR_r = SP.MR_R + 0.6 * zm - 10 * zm * zm;
      const wheelRate_f = SP.kF * MR_f * MR_f;
      const wheelRate_r = SP.kR * MR_r * MR_r;
      pts.push({ z, MR_f: +MR_f.toFixed(3), MR_r: +MR_r.toFixed(3), WR_f: +(wheelRate_f / 1000).toFixed(1), WR_r: +(wheelRate_r / 1000).toFixed(1) });
    }
    return pts;
  }, []);

  const bsGapMM = SP.bsGap * 1000;  // 25mm

  const charts = {
    kinematics: [
      <CC key="wt" title="WHEEL TRAVEL [mm]  (+ve = BUMP)"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("mm")}/>
        <ReferenceArea y1={bsGapMM} y2={50} fill={C.red} fillOpacity={0.04} label={{ value: "BUMPSTOP", fill: C.red, fontSize: 7, position: "insideTopRight" }}/>
        <ReferenceArea y1={-50} y2={-bsGapMM} fill={C.red} fillOpacity={0.04} label={{ value: "BUMPSTOP", fill: C.red, fontSize: 7, position: "insideBottomRight" }}/>
        <Tooltip {...TT}/>{L("zFL",CL.fl,"FL")}{L("zFR",CL.fr,"FR")}{L("zRL",CL.rl,"RL")}{L("zRR",CL.rr,"RR")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="cam" title="DYNAMIC CAMBER [deg]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("deg")}/><Tooltip {...TT}/>{L("camFL",CL.fl,"FL")}{L("camFR",CL.fr,"FR")}{L("camRL",CL.rl,"RL")}{L("camRR",CL.rr,"RR")}<Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="bs" title="BUMP STEER (TOE CHANGE) [deg]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("deg")}/><Tooltip {...TT}/>{L("bsFL",CL.fl,"FL")}{L("bsFR",CL.fr,"FR")}{L("bsRL",CL.rl,"RL")}{L("bsRR",CL.rr,"RR")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="st" title="STEERING ANGLE [deg]"><ResponsiveContainer width="100%" height={180}><AreaChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("deg")}/><Tooltip {...TT}/><Area dataKey="steerDeg" stroke="#e879f9" fill="rgba(232,121,249,0.06)" strokeWidth={1.5} name="δ_sw" dot={false}/><ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/></AreaChart></ResponsiveContainer></CC>,
      <CC key="mr" title="MOTION RATIO & WHEEL RATE"><ResponsiveContainer width="100%" height={180}><ComposedChart data={mrCurve}><CartesianGrid {...gP}/><XAxis dataKey="z" tick={{ fontSize: 8, fill: CA }} stroke={CA} label={{ value: "Travel [mm]", position: "bottom", style: { fontSize: 8, fill: CA } }}/><YAxis yAxisId="mr" tick={{ fontSize: 8, fill: CA }} stroke={CA} label={{ value: "MR", angle: -90, position: "insideLeft", style: { fontSize: 8, fill: CA } }}/><YAxis yAxisId="wr" orientation="right" tick={{ fontSize: 8, fill: CA }} stroke={CA} label={{ value: "kN/m", angle: 90, position: "insideRight", style: { fontSize: 8, fill: CA } }}/><Tooltip {...TT}/><Line yAxisId="mr" dataKey="MR_f" stroke={C.cy} dot={false} strokeWidth={1.5} name="MR Front"/><Line yAxisId="mr" dataKey="MR_r" stroke={C.am} dot={false} strokeWidth={1.5} name="MR Rear"/><Line yAxisId="wr" dataKey="WR_f" stroke={C.cy} dot={false} strokeWidth={1} strokeDasharray="4 2" name="Wheel Rate F [kN/m]"/><Line yAxisId="wr" dataKey="WR_r" stroke={C.am} dot={false} strokeWidth={1} strokeDasharray="4 2" name="Wheel Rate R [kN/m]"/><ReferenceLine yAxisId="mr" x={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></ComposedChart></ResponsiveContainer></CC>,
      <CC key="compSteer" title="COMPLIANCE STEER FROM LATERAL LOAD [deg]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("deg")}/><Tooltip {...TT}/>{L("compSteerF","#22d3ee","Front Compliance")}{L("compSteerR","#f472b6","Rear Compliance")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
    ],
    dynamics: [
      <CC key="fz" title="CORNER LOADS Fz [N]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("N")}/><Tooltip {...TT}/>{L("Fz_FL",CL.fl,"FL")}{L("Fz_FR",CL.fr,"FR")}{L("Fz_RL",CL.rl,"RL")}{L("Fz_RR",CL.rr,"RR")}<Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="fvcloud" title="DAMPER F-v OPERATING CLOUD"><ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={fvCurve} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
          <CartesianGrid {...gP}/>
          <XAxis dataKey="v" tick={{ fontSize: 8, fill: CA }} stroke={CA} label={{ value: "Velocity [mm/s]", position: "bottom", style: { fontSize: 8, fill: CA } }}/>
          <YAxis tick={{ fontSize: 8, fill: CA }} stroke={CA} label={{ value: "Force [N]", angle: -90, position: "insideLeft", style: { fontSize: 8, fill: CA } }}/>
          <Tooltip {...TT}/>
          <Line dataKey="Ff" stroke={C.cy} strokeWidth={2} dot={false} name="Front characteristic"/>
          <Line dataKey="Fr" stroke={C.am} strokeWidth={2} dot={false} name="Rear characteristic"/>
          <ReferenceLine x={SP.vKnee * 1000} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" label={{ value: `v_knee=${SP.vKnee*1000}mm/s`, fill: CA, fontSize: 7 }}/>
          {/* Operating points overlaid — front corners cyan/green, rear amber/red */}
          {[{ data: fvCloud.filter(p => p.corner === "FL"), fill: CL.fl, name: "FL op" },
            { data: fvCloud.filter(p => p.corner === "FR"), fill: CL.fr, name: "FR op" },
            { data: fvCloud.filter(p => p.corner === "RL"), fill: CL.rl, name: "RL op" },
            { data: fvCloud.filter(p => p.corner === "RR"), fill: CL.rr, name: "RR op" }].map(s => (
            <Scatter key={s.name} data={s.data.map(p => ({ v: Math.abs(p.v), F: p.F }))} fill={s.fill} fillOpacity={0.4} r={2} name={s.name} legendType="circle"/>
          ))}
          <Legend {...leg}/>
        </ComposedChart>
      </ResponsiveContainer></CC>,
      <CC key="dv" title="DAMPER VELOCITY [mm/s]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("mm/s")}/><Tooltip {...TT}/>{L("vFL",CL.fl,"FL",1.2)}{L("vFR",CL.fr,"FR",1.2)}{L("vRL",CL.rl,"RL",1.2)}{L("vRR",CL.rr,"RR",1.2)}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="sf" title="SPRING FORCE [N]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("N")}/><Tooltip {...TT}/>{L("Fs_FL",CL.fl,"FL")}{L("Fs_FR",CL.fr,"FR")}{L("Fs_RL",CL.rl,"RL")}{L("Fs_RR",CL.rr,"RR")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="arb" title="ARB COUPLING TORQUE [Nm]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("Nm")}/><Tooltip {...TT}/>{L("arbF","#f472b6","Front ARB")}{L("arbR","#a78bfa","Rear ARB")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
    ],
    vehicle: [
      <CC key="rp" title="ROLL & PITCH [deg]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("deg")}/><Tooltip {...TT}/>{L("roll","#f472b6","Roll")}{L("pitch","#a78bfa","Pitch")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="acc" title="ACCELERATIONS [G]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("G")}/><Tooltip {...TT}/>{L("ay",C.cy,"Ay (lat)")}{L("ax",C.am,"Ax (lon)")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="spd" title="SPEED [m/s]"><ResponsiveContainer width="100%" height={180}><AreaChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("m/s")} domain={[0, 32]}/><Tooltip {...TT}/><Area dataKey="spd" stroke={C.gn} fill={`${C.gn}14`} strokeWidth={1.5} name="Speed" dot={false}/></AreaChart></ResponsiveContainer></CC>,
      <CC key="lltd" title="LLTD — LAT LOAD TRANSFER DISTRIBUTION [%]"><ResponsiveContainer width="100%" height={180}><AreaChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("%")} domain={[30, 70]}/><Tooltip {...TT}/><Area dataKey="LLTD" stroke="#e879f9" fill="rgba(232,121,249,0.08)" strokeWidth={1.5} name="LLTD Front %" dot={false}/><ReferenceLine y={50} stroke="rgba(255,255,255,0.12)" strokeDasharray="4 4"/></AreaChart></ResponsiveContainer></CC>,
      <CC key="rg" title="ROLL GRADIENT [deg/G]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("deg/G")} domain={[0, 4]}/><Tooltip {...TT}/>{L("rollGrad","#22d3ee","Roll Grad")}</LineChart></ResponsiveContainer></CC>,
      <CC key="gg" title="G-G DIAGRAM"><ResponsiveContainer width="100%" height={180}><ScatterChart><CartesianGrid {...gP}/><XAxis type="number" dataKey="ay" name="Ay" tick={{ fontSize: 8, fill: CA }} stroke={CA} domain={[-2.5, 2.5]}/><YAxis type="number" dataKey="ax" name="Ax" tick={{ fontSize: 8, fill: CA }} stroke={CA} domain={[-3, 2.5]}/><Tooltip {...TT}/><ReferenceLine x={0} stroke="rgba(255,255,255,0.06)"/><ReferenceLine y={0} stroke="rgba(255,255,255,0.06)"/><Scatter data={ggData} fill={C.cy} fillOpacity={0.5} r={2}/></ScatterChart></ResponsiveContainer></CC>,
    ],
    tire: [
      <CC key="mu" title="FRICTION UTILIZATION [%]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("%")} domain={[0, 110]}/><Tooltip {...TT}/>{L("muFL",CL.fl,"FL")}{L("muFR",CL.fr,"FR")}{L("muRL",CL.rl,"RL")}{L("muRR",CL.rr,"RR")}<ReferenceLine y={100} stroke="rgba(255,75,75,0.25)" strokeDasharray="4 4"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="fz2" title="VERTICAL LOAD [N]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("N")}/><Tooltip {...TT}/>{L("Fz_FL",CL.fl,"FL")}{L("Fz_FR",CL.fr,"FR")}{L("Fz_RL",CL.rl,"RL")}{L("Fz_RR",CL.rr,"RR")}<Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
    ],
    energy: [
      <CC key="pd" title="DAMPER POWER DISSIPATION [W]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("W")}/><Tooltip {...TT}/>{L("Pd_FL",CL.fl,"FL")}{L("Pd_FR",CL.fr,"FR")}{L("Pd_RL",CL.rl,"RL")}{L("Pd_RR",CL.rr,"RR")}<Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="rpse" title="TOTAL RIDE POWER & SPRING ENERGY"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("W / J")}/><Tooltip {...TT}/>{L("Pd_tot",C.red,"Damper Power [W]")}{L("Es_tot",C.cy,"Spring Energy [J]")}<Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="df" title="DAMPER FORCE [N]"><ResponsiveContainer width="100%" height={180}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("N")}/><Tooltip {...TT}/>{L("Fd_FL",CL.fl,"FL")}{L("Fd_FR",CL.fr,"FR")}{L("Fd_RL",CL.rl,"RL")}{L("Fd_RR",CL.rr,"RR")}<Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
    ],
    modal: [
      <CC key="modes" title="MODAL DECOMPOSITION [mm]"><ResponsiveContainer width="100%" height={200}><LineChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("mm")}/><Tooltip {...TT}/>{L("heave","#22d3ee","Heave")}{L("rollMode","#f472b6","Roll")}{L("pitchMode","#a78bfa","Pitch")}{L("warp","#fbbf24","Warp")}<ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/><Legend {...leg}/></LineChart></ResponsiveContainer></CC>,
      <CC key="heave" title="HEAVE MODE (BODY BOUNCE) [mm]"><ResponsiveContainer width="100%" height={180}><AreaChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("mm")}/><Tooltip {...TT}/><Area dataKey="heave" stroke="#22d3ee" fill="rgba(34,211,238,0.08)" strokeWidth={2} dot={false} name="Heave"/><ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/></AreaChart></ResponsiveContainer></CC>,
      <CC key="rollm" title="ROLL MODE [mm]"><ResponsiveContainer width="100%" height={180}><AreaChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("mm")}/><Tooltip {...TT}/><Area dataKey="rollMode" stroke="#f472b6" fill="rgba(244,114,182,0.08)" strokeWidth={2} dot={false} name="Roll Mode"/><ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/></AreaChart></ResponsiveContainer></CC>,
      <CC key="warpmode" title="WARP MODE (CHASSIS TWIST) [mm]"><ResponsiveContainer width="100%" height={180}><AreaChart {...cp}><CartesianGrid {...gP}/><XAxis {...xP}/><YAxis {...yP("mm")}/><Tooltip {...TT}/><Area dataKey="warp" stroke="#fbbf24" fill="rgba(251,191,36,0.08)" strokeWidth={2} dot={false} name="Warp"/><ReferenceLine y={0} stroke="rgba(255,255,255,0.08)"/></AreaChart></ResponsiveContainer></CC>,
    ],
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      {(charts[activeCat] || charts.kinematics)}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MANEUVER PILL + CHART CATEGORY PILL (using dashboard theme)
// ═══════════════════════════════════════════════════════════════════════════
const Pill = ({ active, label, onClick, color = C.cy }) => (
  <button onClick={onClick} style={{
    background: active ? `${color}18` : "transparent",
    border: `1px solid ${active ? `${color}55` : C.b1}`,
    color: active ? color : C.dm,
    fontSize: 9, fontWeight: 700, letterSpacing: 1.2,
    padding: "5px 12px", borderRadius: 4, cursor: "pointer",
    fontFamily: C.dt, transition: "all 0.15s",
  }}>
    {label}
  </button>
);

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT — drops into App.jsx case "suspension":
// ═══════════════════════════════════════════════════════════════════════════
export default function SuspensionModule({ data }) {
  const ref = useRef(null), st = useRef({});
  const [mn, setMn] = useState("lap");
  const [ps, setPs] = useState(false);
  const [hd, setHd] = useState({ roll: 0, pitch: 0, ay: 0, ax: 0, spd: 0, zFL: 0, zFR: 0, zRL: 0, zRR: 0 });
  const [history, setHistory] = useState([]);
  const [chartCat, setChartCat] = useState("kinematics");
  const histRef = useRef([]);
  const psR = useRef(false); psR.current = ps;

  useEffect(() => { st.current.data = genData(mn); st.current.idx = 0; histRef.current = []; setHistory([]); }, [mn]);

  // ── Three.js scene ──────────────────────────────────────────────────
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const sc = new THREE.Scene(); sc.background = new THREE.Color(0x060a14); sc.fog = new THREE.FogExp2(0x060a14, 0.04);
    const cam = new THREE.PerspectiveCamera(34, el.clientWidth/el.clientHeight, 0.05, 80);
    const ren = new THREE.WebGLRenderer({ antialias: true }); ren.setPixelRatio(Math.min(window.devicePixelRatio, 2)); ren.setSize(el.clientWidth, el.clientHeight);
    ren.shadowMap.enabled = true; ren.shadowMap.type = THREE.PCFSoftShadowMap; ren.toneMapping = THREE.ACESFilmicToneMapping; ren.toneMappingExposure = 1.1;
    el.appendChild(ren.domElement);
    const M = buildMats(); buildLights(sc); const groundRefs = buildGround(sc, M.ground); let distAccum = 0;
    const bg = new THREE.Group(); bg.position.set(0, V.hCG, 0); sc.add(bg);
    buildChassis(bg, M); buildFW(bg, M); buildRW(bg, M);
    const corners = { 
      fl: buildCorner(sc, bg, M, "fl", true, 1), 
      fr: buildCorner(sc, bg, M, "fr", true, -1), 
      rl: buildCorner(sc, bg, M, "rl", false, 1), 
      rr: buildCorner(sc, bg, M, "rr", false, -1) 
    };
    const coils = {}; for (const [k, c] of Object.entries(corners)) coils[k] = buildCoilover(bg, M, c);
    let cT = Math.PI*0.22, cP = Math.PI*0.19, cR = 3.3; const cTg = v3(0, 0.15, 0);
    let drg = false, pn = false, lm = { x: 0, y: 0 };
    const uC = () => { cam.position.set(cR*Math.sin(cP)*Math.sin(cT)+cTg.x, cR*Math.cos(cP)+cTg.y, cR*Math.sin(cP)*Math.cos(cT)+cTg.z); cam.lookAt(cTg); }; uC();
    const oD = e => { if (e.shiftKey) pn = true; else drg = true; lm = { x: e.clientX??e.touches?.[0]?.clientX??0, y: e.clientY??e.touches?.[0]?.clientY??0 }; };
    const oM = e => { const cx = e.clientX??e.touches?.[0]?.clientX??0, cy = e.clientY??e.touches?.[0]?.clientY??0, dx = cx-lm.x, dy = cy-lm.y; lm = { x: cx, y: cy }; if (drg) { cT -= dx*0.005; cP = Math.max(0.06, Math.min(Math.PI*0.47, cP-dy*0.005)); uC(); } else if (pn) { cTg.addScaledVector(v3(0,0,0).setFromMatrixColumn(cam.matrixWorld, 0), -dx*0.003).addScaledVector(v3(0,0,0).setFromMatrixColumn(cam.matrixWorld, 1), dy*0.003); uC(); } };
    const oU = () => { drg = false; pn = false; };
    const oW = e => { cR = Math.max(1.2, Math.min(10, cR+e.deltaY*0.003)); uC(); };
    const cv = ren.domElement;
    cv.addEventListener("mousedown", oD); cv.addEventListener("mousemove", oM); cv.addEventListener("mouseup", oU); cv.addEventListener("mouseleave", oU);
    cv.addEventListener("wheel", oW, { passive: true }); cv.addEventListener("touchstart", oD, { passive: true }); cv.addEventListener("touchmove", oM, { passive: true }); cv.addEventListener("touchend", oU);
    st.current.data = genData("lap"); st.current.idx = 0; const clk = new THREE.Clock(); let hC = 0, aId;
    const loop = () => {
      aId = requestAnimationFrame(loop); const dt = Math.min(clk.getDelta(), 0.05);
      if (!psR.current && st.current.data) {
        const d = st.current.data, i = st.current.idx%d.length, f = d[i];
        const tvl = tick(bg, corners, coils, f, dt);
        const frame = { t: f.t, roll: +f.roll.toFixed(2), pitch: +f.pitch.toFixed(2), ay: +f.ay.toFixed(2), ax: +f.ax.toFixed(2), spd: +f.spd.toFixed(1), steer: f.steer, zFL: +(tvl.fl??0).toFixed(1), zFR: +(tvl.fr??0).toFixed(1), zRL: +(tvl.rl??0).toFixed(1), zRR: +(tvl.rr??0).toFixed(1) };
        histRef.current.push(frame); if (histRef.current.length > 480) histRef.current.shift();
        if (++hC%4 === 0) setHd(frame);
        if (hC%12 === 0) setHistory([...histRef.current]);
        st.current.idx = i+1;
        distAccum += f.spd*dt; groundRefs.road.position.x = -(distAccum%(1.1*30)); groundRefs.road.position.z += f.ay*0.003; groundRefs.road.rotation.y = f.steer*0.15; groundRefs.grid.position.x = -(distAccum%0.5);
      }
      if (!drg && !pn) { cT += 0.0006; uC(); } ren.render(sc, cam);
    }; loop();
    const oR = () => { cam.aspect = el.clientWidth/el.clientHeight; cam.updateProjectionMatrix(); ren.setSize(el.clientWidth, el.clientHeight); };
    window.addEventListener("resize", oR);
    return () => { cancelAnimationFrame(aId); window.removeEventListener("resize", oR); cv.removeEventListener("mousedown", oD); cv.removeEventListener("mousemove", oM); cv.removeEventListener("mouseup", oU); cv.removeEventListener("mouseleave", oU); cv.removeEventListener("wheel", oW); cv.removeEventListener("touchstart", oD); cv.removeEventListener("touchmove", oM); cv.removeEventListener("touchend", oU); ren.dispose(); if (el.contains(ren.domElement)) el.removeChild(ren.domElement); };
  }, []);

  return (
    <div>
      {/* ── KPI Row 1: Live state ──────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 10 }}>
        <KPI label="Roll" value={`${hd.roll.toFixed(2)}°`} sub="body attitude" sentiment="neutral" delay={0} />
        <KPI label="Pitch" value={`${hd.pitch.toFixed(2)}°`} sub="body attitude" sentiment="neutral" delay={1} />
        <KPI label="Ay" value={`${hd.ay.toFixed(2)} G`} sub="lateral accel" sentiment="neutral" delay={2} />
        <KPI label="Ax" value={`${hd.ax.toFixed(2)} G`} sub="longitudinal" sentiment="neutral" delay={3} />
        <KPI label="Speed" value={`${hd.spd.toFixed(1)} m/s`} sub={`${(hd.spd * 3.6).toFixed(0)} km/h`} sentiment="positive" delay={4} />
      </div>

      {/* ── KPI Row 2: Suspension engineering metrics ─────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="f_n Front" value={`${((1/(2*Math.PI))*Math.sqrt(SP.kF*SP.MR_F*SP.MR_F/(SP.mass/4))).toFixed(1)} Hz`} sub="ride frequency" sentiment={((1/(2*Math.PI))*Math.sqrt(SP.kF*SP.MR_F*SP.MR_F/(SP.mass/4))) > 3.5 ? "positive" : "amber"} delay={0} />
        <KPI label="f_n Rear" value={`${((1/(2*Math.PI))*Math.sqrt(SP.kR*SP.MR_R*SP.MR_R/(SP.mass/4))).toFixed(1)} Hz`} sub="ride frequency" sentiment={((1/(2*Math.PI))*Math.sqrt(SP.kR*SP.MR_R*SP.MR_R/(SP.mass/4))) > 4.0 ? "positive" : "amber"} delay={1} />
        <KPI label="ζ Front" value={`${(SP.cLS_F/(2*Math.sqrt(SP.kF*(SP.mass/4)))).toFixed(2)}`} sub={SP.cLS_F/(2*Math.sqrt(SP.kF*(SP.mass/4))) < 1 ? "underdamped" : "overdamped"} sentiment={SP.cLS_F/(2*Math.sqrt(SP.kF*(SP.mass/4))) > 0.5 && SP.cLS_F/(2*Math.sqrt(SP.kF*(SP.mass/4))) < 1.0 ? "positive" : "amber"} delay={2} />
        <KPI label="ζ Rear" value={`${(SP.cLS_R/(2*Math.sqrt(SP.kR*(SP.mass/4)))).toFixed(2)}`} sub={SP.cLS_R/(2*Math.sqrt(SP.kR*(SP.mass/4))) < 1 ? "underdamped" : "overdamped"} sentiment={SP.cLS_R/(2*Math.sqrt(SP.kR*(SP.mass/4))) > 0.5 && SP.cLS_R/(2*Math.sqrt(SP.kR*(SP.mass/4))) < 1.0 ? "positive" : "amber"} delay={3} />
        <KPI label="Anti-dive" value={`${(hd.ax < -0.3 ? 40 : 0).toFixed(0)}%`} sub={hd.ax < -0.3 ? "active (braking)" : "inactive"} sentiment={hd.ax < -0.3 ? "positive" : "neutral"} delay={4} />
        <KPI label="Anti-squat" value={`${(hd.ax > 0.3 ? 30 : 0).toFixed(0)}%`} sub={hd.ax > 0.3 ? "active (accel)" : "inactive"} sentiment={hd.ax > 0.3 ? "positive" : "neutral"} delay={5} />
      </div>

      {/* ── Maneuver + Play/Pause Row ────────────────────────────────── */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {[{ k: "lap", l: "COMBINED LAP" }, { k: "skidpad", l: "SKIDPAD 1.5G" }, { k: "brake", l: "BRAKING 2.5G" }, { k: "accel", l: "ACCELERATION" }, { k: "chicane", l: "CHICANE S-S-S" }].map(b =>
          <Pill key={b.k} active={mn === b.k} label={b.l} onClick={() => setMn(b.k)} />
        )}
        <Pill active={false} label={ps ? "▶ PLAY" : "⏸ PAUSE"} onClick={() => setPs(p => !p)} color={ps ? C.red : C.gn} />
      </div>

      {/* ── 3D Viewport ──────────────────────────────────────────────── */}
      <div style={{ ...GL, padding: 0, overflow: "hidden", marginBottom: 14 }}>
        <div ref={ref} style={{ width: "100%", height: 480 }} />
        {/* Corner travel strip */}
        <div style={{ display: "flex", gap: 8, padding: "8px 12px", borderTop: `1px solid ${C.b1}` }}>
          {[{ l: "Z_FL", v: hd.zFL, c: CL.fl }, { l: "Z_FR", v: hd.zFR, c: CL.fr }, { l: "Z_RL", v: hd.zRL, c: CL.rl }, { l: "Z_RR", v: hd.zRR, c: CL.rr }].map(k => (
            <div key={k.l} style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 7, letterSpacing: 1.5, color: C.dm, fontFamily: C.dt }}>{k.l}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: k.c, fontFamily: C.dt }}>{k.v.toFixed(1)} <span style={{ fontSize: 8, color: C.dm }}>mm</span></div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Chart Category Sub-Tabs ──────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: 2, color: C.cy, fontFamily: C.dt }}>TELEMETRY</span>
        <div style={{ width: 1, height: 16, background: C.b1 }} />
        {CHART_CATS.map(cat => (
          <Pill key={cat.key} active={chartCat === cat.key} label={`${cat.icon} ${cat.label}`} onClick={() => setChartCat(cat.key)} />
        ))}
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt }}>{history.length} samples</span>
      </div>

      {/* ── Charts ───────────────────────────────────────────────────── */}
      <SuspCharts history={history} activeCat={chartCat} />
    </div>
  );
}