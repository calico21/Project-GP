import { useState, useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ReferenceLine, Legend, AreaChart, Area } from "recharts";

/* ═══════════════════════════════════════════════════════════════════════════
 * PROJECT-GP  ·  TER26 Formula Student Suspension Dynamics  ·  v4
 *
 * COORDINATE SYSTEM (all metres):
 *   X = forward, Y = up, Z = right
 *   bodyGroup origin at (0, hCG, 0) world — local y=0 = CG height
 *
 * WORLD-SPACE KEY HEIGHTS:
 *   Ground        y = 0
 *   Tub floor     y = 0.06   (60mm clearance)
 *   Wheel centre  y = 0.229
 *   CG / body ref y = 0.28
 *   Tub top       y = 0.38
 *   Roll hoop top y = 0.76
 *   Rear wing     y = 0.78
 * ═══════════════════════════════════════════════════════════════════════════ */

const V = {
  wb: 1.55, lf: 0.8525, lr: 0.6975,
  tF: 1.22, tR: 1.18,
  hCG: 0.28,
  wR: 0.2286, tW: 0.205,
  maxSt: 0.42,
  // Monocoque (body-local y: floor=-0.22, top=+0.10)
  tubFront: 0.65,    // front bulkhead x
  tubRear: -0.32,    // rear bulkhead x
  tubTopHW: 0.28,    // half-width at top
  tubBotHW: 0.28,    // half-width at bottom (widened to match top — rectangular section)
  tubFloor: -0.22,   // body-local y of floor
  tubTop: 0.10,      // body-local y of top
  // Nose
  noseTip: 1.12,
  // Cockpit
  cpFront: 0.12, cpRear: -0.25, cpHW: 0.20,
  // Roll hoop
  rhX: -0.26, rhTopY: 0.48, rhBaseHW: 0.25, rhTopHW: 0.13,
  // Rear subframe
  rsfRear: -0.78, rsfHW: 0.22, rsfH: 0.18,
  // Front wing
  fwX: 1.14, fwY: -0.24, fwSpan: 1.36, fwChord: 0.26,
  // Rear wing
  rwX: -0.82, rwMainY: 0.50, rwSpan: 0.88, rwChord: 0.30,
  rwEpBot: 0.10, rwEpTop: 0.56,
  // Suspension
  uprH: 0.12,
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
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════
const v3 = (x, y, z) => new THREE.Vector3(x, y, z);

function tube(a, b, r, mat) {
  const d = v3(0,0,0).subVectors(b, a), l = d.length();
  if (l < 1e-5) return new THREE.Group();
  const g = new THREE.CylinderGeometry(r, r, l, 6);
  const m = new THREE.Mesh(g, mat); m.castShadow = true;
  m.position.lerpVectors(a, b, 0.5);
  m.quaternion.setFromUnitVectors(v3(0,1,0), d.normalize());
  return m;
}

function dynTube(a, b, r, mat) {
  const l = a.distanceTo(b);
  const g = new THREE.CylinderGeometry(r, r, l, 6);
  const m = new THREE.Mesh(g, mat); m.castShadow = true; m.userData._nl = l;
  repos(m, a, b); return m;
}

function repos(m, a, b) {
  const d = v3(0,0,0).subVectors(b, a), l = d.length();
  if (l < 1e-5) return;
  m.position.lerpVectors(a, b, 0.5);
  m.quaternion.setFromUnitVectors(v3(0,1,0), d.normalize());
  m.scale.y = l / (m.userData._nl || l);
}

function coilGeo(r, coils, wire) {
  const pts = [], N = coils * 16;
  for (let i = 0; i <= N; i++) {
    const f = i / N, a = f * coils * Math.PI * 2;
    pts.push(v3(r * Math.cos(a), f - 0.5, r * Math.sin(a)));
  }
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), coils * 12, wire, 5, false);
}

function wingGeo(chord, thick, span) {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.bezierCurveTo(chord * 0.12, thick * 0.7, chord * 0.28, thick, chord * 0.38, thick);
  s.bezierCurveTo(chord * 0.55, thick * 0.88, chord * 0.85, thick * 0.3, chord, thick * 0.04);
  s.lineTo(chord, -thick * 0.04);
  s.bezierCurveTo(chord * 0.85, -thick * 0.12, chord * 0.5, -thick * 0.32, chord * 0.3, -thick * 0.28);
  s.bezierCurveTo(chord * 0.12, -thick * 0.18, 0.005, -thick * 0.04, 0, 0);
  const g = new THREE.ExtrudeGeometry(s, { depth: span, bevelEnabled: false });
  g.translate(-chord * 0.3, 0, -span / 2);
  return g;
}

/** Smooth lofted body — tapered oval cross-sections along X */
function loftBody(stations, nPts, mat) {
  // stations: [{x, hw, hh, cy}]  hw=half-width, hh=half-height, cy=centre-y
  const positions = [], indices = [];
  const N = nPts;
  for (const st of stations) {
    for (let p = 0; p < N; p++) {
      const a = (p / N) * Math.PI * 2;
      // Superellipse exponent ~4 for nearly rectangular with slight rounding
      const ca = Math.cos(a), sa = Math.sin(a);
      const exp = 4.0;
      const r = Math.pow(Math.pow(Math.abs(ca), exp) + Math.pow(Math.abs(sa), exp), -1 / exp);
      positions.push(st.x, st.cy + sa * r * st.hh, ca * r * st.hw);
    }
  }
  for (let s = 0; s < stations.length - 1; s++) {
    for (let p = 0; p < N; p++) {
      const p2 = (p + 1) % N;
      const a = s * N + p, b = s * N + p2, c = (s + 1) * N + p, d = (s + 1) * N + p2;
      indices.push(a, c, b, b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, mat); m.castShadow = true;
  return m;
}

// ═══════════════════════════════════════════════════════════════════════════
// MATERIALS
// ═══════════════════════════════════════════════════════════════════════════
function buildMats() {
  const s = (c, met, rgh, o = {}) => new THREE.MeshStandardMaterial({ color: c, metalness: met, roughness: rgh, ...o });
  return {
    body:    s(0x1e2430, 0.32, 0.48, { side: THREE.DoubleSide }),
    nose:    s(0x181e2a, 0.25, 0.52, { side: THREE.DoubleSide }),
    cockpit: s(0x0a0e16, 0.20, 0.70),
    accent:  s(0x00b8e6, 0.55, 0.30, { emissive: 0x004050, emissiveIntensity: 0.25 }),
    steel:   s(0x6a7288, 0.85, 0.20),
    stLt:    s(0x8892a8, 0.70, 0.25),
    push:    s(0xc8b030, 0.70, 0.30),
    spring:  s(0x00d4ff, 0.45, 0.30, { emissive: 0x003848, emissiveIntensity: 0.40 }),
    dampB:   s(0x3e4650, 0.80, 0.20),
    dampS:   s(0x98a0b0, 0.90, 0.10),
    tire:    s(0x161616, 0.00, 0.94),
    rim:     s(0x808890, 0.88, 0.12),
    wingCF:  s(0x141820, 0.25, 0.50),
    wingEP:  s(0x1a2030, 0.30, 0.45),
    floor:   s(0x0c1018, 0.15, 0.70),
    ground:  s(0x080c14, 0.00, 0.95),
    head:    s(0x222830, 0.20, 0.60),
  };
}

function buildLights(sc) {
  sc.add(new THREE.AmbientLight(0x354060, 0.50));
  const k = new THREE.DirectionalLight(0xdde8ff, 1.8);
  k.position.set(4, 6, 3); k.castShadow = true;
  k.shadow.mapSize.set(2048, 2048);
  Object.assign(k.shadow.camera, { near: 0.5, far: 20, left: -5, right: 5, top: 5, bottom: -5 });
  sc.add(k);
  sc.add(new THREE.DirectionalLight(0x5070a0, 0.40).translateX(-4).translateY(3).translateZ(-3));
  sc.add((() => { const p = new THREE.PointLight(0x00b8e6, 0.20, 10); p.position.set(-1, 2, -4); return p; })());
}

function buildGround(sc, mat) {
  const g = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), mat);
  g.rotation.x = -Math.PI / 2; g.receiveShadow = true; sc.add(g);
  const gr = new THREE.GridHelper(24, 48, 0x182035, 0x182035);
  gr.material.opacity = 0.15; gr.material.transparent = true;
  gr.position.y = 0.001; sc.add(gr);

  // Road surface group — scrolls for speed feeling
  const road = new THREE.Group();

  // Asphalt strip (slightly lighter than ground)
  const asphalt = new THREE.Mesh(
    new THREE.PlaneGeometry(30, 4.0),
    new THREE.MeshStandardMaterial({ color: 0x0e1220, roughness: 0.92, metalness: 0.0 })
  );
  asphalt.rotation.x = -Math.PI / 2;
  asphalt.position.y = 0.0015;
  road.add(asphalt);

  // White edge lines
  const lineMat = new THREE.MeshBasicMaterial({ color: 0x2a3550, transparent: true, opacity: 0.5 });
  for (const side of [-1, 1]) {
    for (let i = 0; i < 30; i++) {
      const dash = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.004, 0.06), lineMat);
      dash.position.set(-15 + i * 1.1, 0.003, side * 1.8);
      road.add(dash);
    }
  }

  // Centre dashes (yellow)
  const centreMat = new THREE.MeshBasicMaterial({ color: 0x3a3520, transparent: true, opacity: 0.4 });
  for (let i = 0; i < 30; i++) {
    const dash = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.004, 0.04), centreMat);
    dash.position.set(-15 + i * 1.1, 0.003, 0);
    road.add(dash);
  }

  // Kerb strips (red-white alternating)
  const kerbR = new THREE.MeshBasicMaterial({ color: 0x401515, transparent: true, opacity: 0.35 });
  const kerbW = new THREE.MeshBasicMaterial({ color: 0x252530, transparent: true, opacity: 0.35 });
  for (const side of [-1, 1]) {
    for (let i = 0; i < 40; i++) {
      const kMat = i % 2 === 0 ? kerbR : kerbW;
      const kerb = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.008, 0.15), kMat);
      kerb.position.set(-15 + i * 0.75, 0.004, side * 2.1);
      road.add(kerb);
    }
  }

  sc.add(road);
  return { grid: gr, road };
}

// ═══════════════════════════════════════════════════════════════════════════
// MONOCOQUE — smooth lofted body with nose taper
// ═══════════════════════════════════════════════════════════════════════════
function buildChassis(body, M) {
  const fl = V.tubFloor, tp = V.tubTop;
  const cy = (fl + tp) / 2;       // cross-section centre y
  const hh = (tp - fl) / 2;       // half-height

  // ── Smooth lofted nose + tub in one piece ───────────────────────────
  // Define cross-section stations from nose tip to rear bulkhead
  const stations = [
    { x: V.noseTip,        hw: 0.03,        hh: 0.04,     cy: cy - 0.06 },  // nose tip
    { x: V.noseTip - 0.08, hw: 0.06,        hh: 0.07,     cy: cy - 0.04 },
    { x: V.noseTip - 0.20, hw: 0.12,        hh: 0.10,     cy: cy - 0.02 },
    { x: V.tubFront + 0.15, hw: 0.22,       hh: 0.13,     cy: cy },         // wider transition
    { x: V.tubFront + 0.05, hw: V.tubTopHW, hh: hh * 0.95, cy: cy },       // nearly full width before bulkhead
    { x: V.tubFront,       hw: V.tubTopHW,  hh: hh,       cy: cy },         // front bulkhead — full width
    { x: V.cpFront + 0.05, hw: V.tubTopHW,  hh: hh,       cy: cy },         // cockpit start
    { x: V.cpRear - 0.05,  hw: V.tubTopHW,  hh: hh,       cy: cy },         // cockpit end
    { x: V.tubRear,        hw: V.tubTopHW * 0.95, hh: hh * 0.95, cy: cy },  // rear bulkhead
  ];

  body.add(loftBody(stations, 20, M.body));

  // ── Cockpit opening (dark recessed pocket) ──────────────────────────
  const cpLen = V.cpFront - V.cpRear;
  const cpGeo = new THREE.BoxGeometry(cpLen, 0.07, V.cpHW * 2);
  const cpM = new THREE.Mesh(cpGeo, M.cockpit);
  cpM.position.set((V.cpFront + V.cpRear) / 2, tp - 0.015, 0);
  body.add(cpM);

  // Cockpit rim
  const crGeo = new THREE.BoxGeometry(cpLen + 0.04, 0.018, V.cpHW * 2 + 0.03);
  const crM = new THREE.Mesh(crGeo, M.body);
  crM.position.set((V.cpFront + V.cpRear) / 2, tp + 0.005, 0);
  body.add(crM);

  // ── Headrest (behind cockpit, protruding above tub) ─────────────────
  const hrGeo = new THREE.BoxGeometry(0.14, 0.11, 0.17);
  const hrM = new THREE.Mesh(hrGeo, M.head);
  hrM.position.set(V.cpRear - 0.05, tp + 0.055, 0);
  body.add(hrM);

  // ── SIS accent strips ──────────────────────────────────────────────
  const tubLen = V.tubFront - V.tubRear;
  for (const s of [-1, 1]) {
    const sGeo = new THREE.BoxGeometry(tubLen * 0.4, 0.025, 0.010);
    const sM = new THREE.Mesh(sGeo, M.accent); sM.castShadow = true;
    sM.position.set(V.tubFront - tubLen * 0.35, cy, s * (V.tubTopHW + 0.006));
    body.add(sM);
  }

  // ── Flat undertray (subtle, dark, hidden under body) ─────────────────
  const flrGeo = new THREE.BoxGeometry(1.6, 0.005, V.tubTopHW * 1.8);
  const flrM = new THREE.Mesh(flrGeo, M.floor);
  flrM.position.set(0.15, fl - 0.003, 0);
  body.add(flrM);

  // ── Roll hoop ──────────────────────────────────────────────────────
  const rhBase = tp;
  const rhTop = V.rhTopY;
  for (const s of [-1, 1]) {
    body.add(tube(v3(V.rhX, rhBase, s * V.rhBaseHW), v3(V.rhX, rhTop, s * V.rhTopHW), 0.013, M.steel));
  }
  body.add(tube(v3(V.rhX, rhTop, -V.rhTopHW), v3(V.rhX, rhTop, V.rhTopHW), 0.013, M.steel));
  // Rear bracing
  for (const s of [-1, 1]) {
    body.add(tube(v3(V.rhX, rhTop * 0.88, s * V.rhTopHW), v3(V.rhX - 0.25, rhBase * 0.85, s * V.rhBaseHW * 0.65), 0.009, M.steel));
    body.add(tube(v3(V.rhX, rhTop, s * V.rhTopHW), v3(V.rhX - 0.32, rhBase, s * V.rhBaseHW * 0.4), 0.007, M.steel));
  }

  // ── Rear subframe ──────────────────────────────────────────────────
  const rsF = V.tubRear, rsR = V.rsfRear;
  const sw = V.rsfHW, sh = V.rsfH;
  for (const s of [-1, 1]) {
    body.add(tube(v3(rsF, fl, s * sw), v3(rsR, fl, s * sw), 0.008, M.steel));
    body.add(tube(v3(rsF, fl + sh, s * sw * 0.82), v3(rsR, fl + sh * 0.72, s * sw * 0.65), 0.007, M.steel));
    for (const f of [0.3, 0.7]) {
      const x = rsF + (rsR - rsF) * f;
      body.add(tube(v3(x, fl, s * sw), v3(x, fl + sh * (0.88 - f * 0.18), s * sw * (0.88 - f * 0.10)), 0.006, M.steel));
    }
  }
  for (const f of [0, 0.5, 1]) {
    const x = rsF + (rsR - rsF) * f;
    body.add(tube(v3(x, fl, -sw), v3(x, fl, sw), 0.006, M.steel));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// FRONT WING — big, at ground level
// ═══════════════════════════════════════════════════════════════════════════
function buildFW(body, M) {
  const wx = V.fwX, wy = V.fwY, sp = V.fwSpan, ch = V.fwChord;
  // Main element
  const mg = wingGeo(ch, 0.020, sp); mg.rotateY(Math.PI);
  const mm = new THREE.Mesh(mg, M.wingCF);
  mm.position.set(wx, wy, 0); mm.rotation.z = 0.08; mm.castShadow = true; body.add(mm);
  // Flap
  const fg = wingGeo(ch * 0.35, 0.013, sp * 0.93); fg.rotateY(Math.PI);
  const fm = new THREE.Mesh(fg, M.wingCF);
  fm.position.set(wx - ch * 0.44, wy + 0.032, 0); fm.rotation.z = 0.20; fm.castShadow = true; body.add(fm);
  // Gurney
  body.add((() => { const m = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.016, sp * 0.85), M.accent); m.position.set(wx - ch * 0.56, wy + 0.008, 0); return m; })());
  // Endplates
  for (const s of [-1, 1]) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(ch * 1.3, 0.075, 0.004), M.wingEP);
    m.position.set(wx - ch * 0.2, wy + 0.015, s * sp * 0.505); m.castShadow = true; body.add(m);
  }
  // Mount pylons
  for (const s of [-1, 1])
    body.add(tube(v3(wx - ch * 0.15, wy + 0.04, s * sp * 0.15), v3(V.noseTip - 0.22, V.tubFloor + 0.10, s * 0.04), 0.005, M.body));
}

// ═══════════════════════════════════════════════════════════════════════════
// REAR WING — large, on endplate stands
// ═══════════════════════════════════════════════════════════════════════════
function buildRW(body, M) {
  const wx = V.rwX, wy = V.rwMainY, sp = V.rwSpan, ch = V.rwChord;
  // Main element
  const mg = wingGeo(ch, 0.024, sp); mg.rotateY(Math.PI);
  const mm = new THREE.Mesh(mg, M.wingCF);
  mm.position.set(wx, wy, 0); mm.rotation.z = 0.14; mm.castShadow = true; body.add(mm);
  // Flap
  const fg = wingGeo(ch * 0.30, 0.015, sp * 0.96); fg.rotateY(Math.PI);
  const fm = new THREE.Mesh(fg, M.wingCF);
  fm.position.set(wx - ch * 0.30, wy + 0.055, 0); fm.rotation.z = 0.30; fm.castShadow = true; body.add(fm);
  // Tall endplates (stand on rear structure)
  for (const s of [-1, 1]) {
    const epH = V.rwEpTop - V.rwEpBot;
    const m = new THREE.Mesh(new THREE.BoxGeometry(ch * 1.5, epH, 0.005), M.wingEP);
    m.position.set(wx - ch * 0.05, (V.rwEpBot + V.rwEpTop) / 2, s * sp * 0.505);
    m.castShadow = true; body.add(m);
  }
  // Braces to rear subframe
  for (const s of [-1, 1])
    body.add(tube(v3(wx + ch * 0.35, V.rwEpBot + 0.02, s * sp * 0.5), v3(V.rsfRear + 0.08, V.tubFloor + V.rsfH * 0.3, s * V.rsfHW * 0.8), 0.007, M.steel));
}

// ═══════════════════════════════════════════════════════════════════════════
// CORNER — wheel + upright + double wishbone + pushrod
// ═══════════════════════════════════════════════════════════════════════════
function buildCorner(sc, body, M, name, xA, yS, tW, isF) {
  const yP = yS * tW / 2;  // wheel-centre lateral position

  // ── Steer group (world space) ───────────────────────────────────────
  const sg = new THREE.Group(); sg.position.set(xA, 0, yP); sc.add(sg);

  // ── Wheel group (spins inside steer group) ──────────────────────────
  const wg = new THREE.Group(); wg.position.y = V.wR; sg.add(wg);

  // Rim face
  const rimFace = new THREE.Mesh(
    new THREE.CylinderGeometry(V.wR * 0.6, V.wR * 0.6, V.tW * 0.22, 20).rotateX(Math.PI / 2),
    M.rim
  );
  rimFace.castShadow = true; wg.add(rimFace);

  // 5 spokes (flat bars, visible when spinning)
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    const sp = new THREE.Mesh(new THREE.BoxGeometry(0.028, V.wR * 0.42, V.tW * 0.18), M.rim);
    sp.position.set(Math.cos(a) * V.wR * 0.28, Math.sin(a) * V.wR * 0.28, 0);
    sp.rotation.z = a; wg.add(sp);
  }

  // Tire (torus)
  const tireM = new THREE.Mesh(
    new THREE.TorusGeometry(V.wR, V.tW * 0.38, 14, 32),
    M.tire
  );
  tireM.castShadow = true; wg.add(tireM);

  // ── Upright (in steer group, inboard of wheel) ─────────────────────
  const uprOff = -yS * V.tW * 0.34;  // inboard offset
  const ug = new THREE.Group();
  ug.position.set(0, V.wR, uprOff);
  const uprMesh = new THREE.Mesh(new THREE.BoxGeometry(0.032, V.uprH, 0.040), M.stLt);
  uprMesh.castShadow = true; ug.add(uprMesh);
  sg.add(ug);

  // ── Wishbone geometry ────────────────────────────────────────────────
  // Inner pickups stored as BODY-LOCAL coords (will be transformed to world each frame)
  // CRITICAL: inner X must be within the tub, not on the nose
  const chassisHW = isF ? V.tubBotHW : V.rsfHW;
  const iZ = yS * chassisHW * 0.92;  // inset to sit flush on body

  const bodyLowY = 0.10 - V.hCG;   // -0.18 body-local
  const bodyHiY = 0.32 - V.hCG;    // +0.04 body-local

  // For front: inner pickups on the front bulkhead face (x ≈ tubFront)
  // For rear: inner pickups on the rear bulkhead / subframe area
  const bulkX = isF ? V.tubFront : V.tubRear;
  const spread = 0.10;  // fore-aft spread of A-arm pickup pairs on bulkhead

  // Body-local inner points (these ride with the chassis)
  const lInLocal = [
    v3(bulkX + spread, bodyLowY, iZ),
    v3(bulkX - spread, bodyLowY, iZ),
  ];
  const uInLocal = [
    v3(bulkX + spread * 0.85, bodyHiY, iZ * 0.95),
    v3(bulkX - spread * 0.85, bodyHiY, iZ * 0.95),
  ];

  // Initial world-space (before any rotation, body at y=hCG)
  const toWorld = (bl) => v3(bl.x, bl.y + V.hCG, bl.z);
  const lIn = lInLocal.map(toWorld);
  const uIn = uInLocal.map(toWorld);

  // Outer ball joints on upright (world space, move with suspension travel)
  const outerZ = yP + uprOff;
  const lO0 = v3(xA, V.wR - V.uprH * 0.35, outerZ);
  const uO0 = v3(xA, V.wR + V.uprH * 0.33, outerZ);

  const lA = lIn.map((ip, i) => { const m = dynTube(ip, lO0, 0.007, M.steel); sc.add(m); return { m, ip: ip.clone(), local: lInLocal[i].clone() }; });
  const uA = uIn.map((ip, i) => { const m = dynTube(ip, uO0, 0.006, M.stLt); sc.add(m); return { m, ip: ip.clone(), local: uInLocal[i].clone() }; });

  // ── Wishbone mounting flanges on bodyGroup ──────────────────────────
  for (const bl of [...lInLocal, ...uInLocal]) {
    const bracket = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.028, 0.028), M.steel);
    bracket.position.copy(bl);
    bracket.castShadow = true;
    body.add(bracket);
  }

  // ── Pushrod (lower wishbone area → bellcrank at chassis) ────────────
  const pB0 = lO0.clone().lerp(uO0, 0.25);
  const bcLocalY = (bodyLowY + bodyHiY) / 2 + 0.04;
  const bcLocal = v3(bulkX, bcLocalY, iZ * 0.85);  // body-local, on bulkhead
  const bc0 = v3(xA, bcLocalY + V.hCG, iZ * 0.85);  // world initial
  const pM = dynTube(pB0, bc0, 0.005, M.push); sc.add(pM);

  return {
    name, xA, yS, tW, isF, yP,
    sg, wg, ug, lA, uA,
    lO0: lO0.clone(), uO0: uO0.clone(),
    pM, pB0: pB0.clone(), bc0: bc0.clone(),
    bcLocal: bcLocal.clone(),
    ws: 0, iZ, chassisHW,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// COILOVER — Front: TRANSVERSE (lateral toward centreline) on tub top
//            Rear: FORE-AFT (parallel to wheels) on chassis top
// ═══════════════════════════════════════════════════════════════════════════
function buildCoilover(body, M, c) {
  const springLen = 0.16;
  const tubTop = V.tubTop;

  const rockerX = c.isF
    ? Math.min(c.xA, V.tubFront - 0.05)
    : Math.max(c.xA, V.tubRear + 0.05);
  const rockerY = tubTop + 0.015;
  const tiltRad = -5 * Math.PI / 180;

  let rockerPt, anchorPt, dOffVec;

  if (c.isF) {
    // FRONT: transverse — rocker outboard, spring runs inboard
    const rockerZ = c.yS * (V.tubTopHW - 0.03);
    rockerPt = v3(rockerX, rockerY, rockerZ);
    anchorPt = v3(
      rockerX,
      rockerY + springLen * Math.sin(tiltRad),
      rockerZ - c.yS * springLen * Math.cos(tiltRad)
    );
    dOffVec = v3(0.020, 0, 0);  // damper offset fore-aft
  } else {
    // REAR: fore-aft — rocker at chassis top, spring runs forward
    const rockerZ = c.yS * (V.rsfHW - 0.03);
    rockerPt = v3(rockerX, rockerY, rockerZ);
    anchorPt = v3(
      rockerX + springLen * Math.cos(tiltRad),  // forward
      rockerY + springLen * Math.sin(tiltRad),
      rockerZ
    );
    dOffVec = v3(0, 0, c.yS * 0.020);  // damper offset lateral
  }

  const nomD = rockerPt.distanceTo(anchorPt);
  const axis = v3(0,0,0).subVectors(anchorPt, rockerPt).normalize();

  // Rocker pivot marker
  body.add((() => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.009, 8, 6), M.stLt);
    m.position.copy(rockerPt); return m;
  })());

  // Anchor bracket
  body.add((() => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.014, 0.018), M.steel);
    m.position.copy(anchorPt); return m;
  })());

  // Spring coil
  const sGeo = coilGeo(0.012, 8, 0.0022);
  const sM = new THREE.Mesh(sGeo, M.spring); sM.castShadow = true;
  sM.position.lerpVectors(rockerPt, anchorPt, 0.5);
  sM.quaternion.setFromUnitVectors(v3(0,1,0), axis);
  sM.scale.set(1, nomD, 1);
  body.add(sM);

  // Damper — offset alongside spring (direction depends on front/rear)
  const dRocker = rockerPt.clone().add(dOffVec);
  const dAnchor = anchorPt.clone().add(dOffVec);
  const dLen = nomD * 0.48;

  const dBG = new THREE.CylinderGeometry(0.009, 0.009, dLen, 8);
  const dBM = new THREE.Mesh(dBG, M.dampB); dBM.castShadow = true;
  const dSG = new THREE.CylinderGeometry(0.0045, 0.0045, nomD * 0.42, 8);
  const dSM = new THREE.Mesh(dSG, M.dampS);

  const dGr = new THREE.Group();
  dBM.position.y = dLen * 0.26; dSM.position.y = -dLen * 0.20;
  dGr.add(dBM); dGr.add(dSM);
  dGr.position.lerpVectors(dRocker, dAnchor, 0.5);
  dGr.quaternion.setFromUnitVectors(v3(0,1,0), axis);
  body.add(dGr);

  // Actuation link: pushrod bellcrank → rocker
  const bcLocal = v3(c.bc0.x, c.bc0.y - V.hCG, c.bc0.z);
  body.add(tube(bcLocal, rockerPt, 0.004, M.push));

  return { sM, dGr, dSM, base: rockerPt, top: anchorPt, axis, nomD, dOffVec: dOffVec.clone() };
}

// ═══════════════════════════════════════════════════════════════════════════
// PER-FRAME UPDATE
// ═══════════════════════════════════════════════════════════════════════════
function tick(bg, corners, coils, fr, dt) {
  const rR = fr.roll * Math.PI / 180, pR = fr.pitch * Math.PI / 180;
  bg.rotation.z = rR; bg.rotation.x = pR;
  bg.position.y = V.hCG - 0.0015 * (Math.abs(fr.roll) + Math.abs(fr.pitch));

  // Force world matrix update so localToWorld is accurate this frame
  bg.updateMatrixWorld(true);

  const tvl = {};
  for (const c of Object.values(corners)) {
    const tw = c.isF ? V.tF : V.tR;
    const raw = -(tw / 2) * c.yS * rR + c.xA * pR;
    const vis = raw * 4.0;
    tvl[c.name] = raw * 1000;

    if (c.isF) c.sg.rotation.y = fr.steer;
    c.ws += (fr.spd / V.wR) * dt;
    c.wg.rotation.z = c.ws;

    // Upright travel
    c.ug.position.y = V.wR + vis * 0.12;
    const lo = c.lO0.clone(); lo.y += vis * 0.12;
    const uo = c.uO0.clone(); uo.y += vis * 0.12;

    // Transform body-local inner points to world space (tracks roll/pitch)
    c.lA.forEach(a => {
      const wp = bg.localToWorld(a.local.clone());
      repos(a.m, wp, lo);
    });
    c.uA.forEach(a => {
      const wp = bg.localToWorld(a.local.clone());
      repos(a.m, wp, uo);
    });

    // Pushrod: outer end tracks wishbone, inner end (bellcrank) tracks body
    const bcWorld = bg.localToWorld(c.bcLocal.clone());
    repos(c.pM, lo.clone().lerp(uo, 0.25), bcWorld);

    // Coilover
    const ib = coils[c.name];
    if (ib) {
      const nd = Math.max(ib.nomD * 0.6, Math.min(ib.nomD * 1.18, ib.nomD - vis));
      const newTop = ib.base.clone().add(ib.axis.clone().multiplyScalar(nd));
      ib.sM.position.lerpVectors(ib.base, newTop, 0.5);
      ib.sM.scale.set(1, nd, 1);
      const dB = ib.base.clone().add(ib.dOffVec);
      const dT = newTop.clone().add(ib.dOffVec);
      ib.dGr.position.lerpVectors(dB, dT, 0.5);
      ib.dSM.scale.y = nd / ib.nomD;
    }
  }
  return tvl;
}

// ═══════════════════════════════════════════════════════════════════════════
// HUD
// ═══════════════════════════════════════════════════════════════════════════
const FN = "'JetBrains Mono','Fira Code','Courier New',monospace";
const KB = ({ l, v, c }) => (
  <div style={{ background: "rgba(6,10,20,0.88)", border: "1px solid rgba(0,184,230,0.14)", borderRadius: 6, padding: "5px 10px", minWidth: 66, textAlign: "center", backdropFilter: "blur(10px)" }}>
    <div style={{ fontSize: 7, letterSpacing: 1.6, color: "#384560", fontFamily: FN }}>{l}</div>
    <div style={{ fontSize: 14, fontWeight: 700, color: c, fontFamily: FN }}>{v}</div>
  </div>
);

const HUD = ({ d, mn, setMn, ps, setPs }) => {
  const bs = [{ k: "lap", l: "COMBINED LAP" }, { k: "skidpad", l: "SKIDPAD 1.5G" }, { k: "brake", l: "BRAKING 2.5G" }, { k: "accel", l: "ACCELERATION" }, { k: "chicane", l: "CHICANE S-S-S" }];
  const bS = a => ({ background: a ? "rgba(0,184,230,0.14)" : "rgba(255,255,255,0.02)", border: `1px solid ${a ? "rgba(0,184,230,0.45)" : "rgba(0,184,230,0.08)"}`, color: a ? "#00d4ff" : "#384560", fontSize: 8, fontWeight: 700, letterSpacing: 1.1, padding: "4px 9px", borderRadius: 4, cursor: "pointer", fontFamily: FN, transition: "all 0.15s" });
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", fontFamily: FN, color: "#b8c0d4" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 16px", background: "rgba(6,10,20,0.55)", borderBottom: "1px solid rgba(0,184,230,0.08)" }}>
        <div><span style={{ fontSize: 13, fontWeight: 800, letterSpacing: 2, color: "#00b8e6" }}>PROJECT-GP</span><span style={{ color: "#384560", marginLeft: 10, fontSize: 10, fontWeight: 400 }}>SUSPENSION DYNAMICS · TER26</span></div>
        <div style={{ display: "flex", gap: 5, pointerEvents: "auto" }}>
          {bs.map(b => <button key={b.k} onClick={() => setMn(b.k)} style={bS(mn === b.k)}>{b.l}</button>)}
          <button onClick={() => setPs(p => !p)} style={{ ...bS(false), marginLeft: 6, background: ps ? "rgba(255,56,56,0.10)" : "rgba(35,209,96,0.10)", border: `1px solid ${ps ? "#ff3838" : "#23d160"}`, color: ps ? "#ff3838" : "#23d160" }}>{ps ? "▶ PLAY" : "⏸ PAUSE"}</button>
        </div>
      </div>
      <div style={{ position: "absolute", top: 54, left: 16, display: "flex", gap: 7 }}>
        <KB l="ROLL" v={`${(d.roll ?? 0).toFixed(2)}°`} c="#f472b6" />
        <KB l="PITCH" v={`${(d.pitch ?? 0).toFixed(2)}°`} c="#a78bfa" />
        <KB l="Ay" v={`${(d.ay ?? 0).toFixed(2)} G`} c="#00b8e6" />
        <KB l="Ax" v={`${(d.ax ?? 0).toFixed(2)} G`} c="#ffb020" />
        <KB l="SPEED" v={`${(d.spd ?? 0).toFixed(1)} m/s`} c="#23d160" />
      </div>
      <div style={{ position: "absolute", bottom: 14, left: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>
        <KB l="Z_FL" v={`${(d.zFL ?? 0).toFixed(1)} mm`} c="#00b8e6" />
        <KB l="Z_FR" v={`${(d.zFR ?? 0).toFixed(1)} mm`} c="#23d160" />
        <KB l="Z_RL" v={`${(d.zRL ?? 0).toFixed(1)} mm`} c="#ffb020" />
        <KB l="Z_RR" v={`${(d.zRR ?? 0).toFixed(1)} mm`} c="#ff3838" />
      </div>
      <div style={{ position: "absolute", bottom: 14, right: 16, fontSize: 8, color: "#1e2a3e", textAlign: "right", lineHeight: 1.7 }}>DRAG · orbit &nbsp;|&nbsp; SCROLL · zoom &nbsp;|&nbsp; SHIFT+DRAG · pan</div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════
// TELEMETRY CHARTS — Recharts-based graphs beneath the 3D view
// ═══════════════════════════════════════════════════════════════════════════

const CHART_BG = "#0a0e1a";
const CHART_GRID = "rgba(0,184,230,0.06)";
const CHART_AXIS = "#2a3550";
const CHART_TT = { contentStyle: { background: "#0e1420", border: "1px solid rgba(0,184,230,0.2)", borderRadius: 6, fontSize: 10, fontFamily: FN, color: "#b0b8cc" }, itemStyle: { color: "#b0b8cc", fontSize: 10 } };
const CL = { fl: "#00b8e6", fr: "#23d160", rl: "#ffb020", rr: "#ff3838" };

const ChartCard = ({ title, children, span = 1 }) => (
  <div style={{
    background: "rgba(10,14,26,0.92)", border: "1px solid rgba(0,184,230,0.10)",
    borderRadius: 8, padding: "12px 10px 6px", gridColumn: span > 1 ? `span ${span}` : undefined,
  }}>
    <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.8, color: "#384560", fontFamily: FN, marginBottom: 8, paddingLeft: 4 }}>
      {title}
    </div>
    {children}
  </div>
);

function TelemetryCharts({ history, maneuver }) {
  if (!history || history.length < 10) return null;

  // Compute derived channels
  const derived = useMemo(() => {
    const d = history.map((h, i) => {
      const prev = i > 0 ? history[i - 1] : h;
      const dt = h.t - prev.t || 0.017;
      // Damper velocities (mm/s)
      const vFL = (h.zFL - prev.zFL) / dt;
      const vFR = (h.zFR - prev.zFR) / dt;
      const vRL = (h.zRL - prev.zRL) / dt;
      const vRR = (h.zRR - prev.zRR) / dt;
      // Load transfer estimates (simplified)
      const latLT = h.ay * 300 * 0.28 / ((V.tF + V.tR) / 2);  // ΔFz from lateral
      const lonLT = h.ax * 300 * 0.28 / V.wb;                    // ΔFz from longitudinal
      const Fz_FL = 735 - latLT / 2 + lonLT / 2;
      const Fz_FR = 735 + latLT / 2 + lonLT / 2;
      const Fz_RL = 735 - latLT / 2 - lonLT / 2;
      const Fz_RR = 735 + latLT / 2 - lonLT / 2;
      // Camber (static + gain * roll)
      const camFL = -2.0 + 0.65 * h.roll;
      const camFR = -2.0 - 0.65 * h.roll;
      const camRL = -1.5 + 0.45 * h.roll;
      const camRR = -1.5 - 0.45 * h.roll;
      return {
        ...h,
        t: +(h.t).toFixed(2),
        vFL: +vFL.toFixed(1), vFR: +vFR.toFixed(1), vRL: +vRL.toFixed(1), vRR: +vRR.toFixed(1),
        Fz_FL: +Fz_FL.toFixed(0), Fz_FR: +Fz_FR.toFixed(0), Fz_RL: +Fz_RL.toFixed(0), Fz_RR: +Fz_RR.toFixed(0),
        camFL: +camFL.toFixed(2), camFR: +camFR.toFixed(2), camRL: +camRL.toFixed(2), camRR: +camRR.toFixed(2),
      };
    });
    return d;
  }, [history]);

  // G-G scatter data
  const ggData = useMemo(() =>
    history.filter((_, i) => i % 3 === 0).map(h => ({ ax: +h.ax.toFixed(2), ay: +h.ay.toFixed(2) })),
    [history]
  );

  const commonProps = { syncId: "susp", data: derived };
  const xProps = { dataKey: "t", tick: { fontSize: 8, fill: CHART_AXIS }, stroke: CHART_AXIS, tickLine: false };
  const yProps = (label) => ({ tick: { fontSize: 8, fill: CHART_AXIS }, stroke: CHART_AXIS, tickLine: false, label: { value: label, angle: -90, position: "insideLeft", style: { fontSize: 8, fill: "#384560", fontFamily: FN } } });
  const gridProps = { stroke: CHART_GRID, strokeDasharray: "3 3" };

  return (
    <div style={{
      display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12,
      padding: "16px 16px 24px", background: "#060a14",
    }}>

      {/* 1. Wheel Travel */}
      <ChartCard title="WHEEL TRAVEL [mm]  (+ve = BUMP)">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xProps} />
            <YAxis {...yProps("mm")} />
            <Tooltip {...CHART_TT} />
            <Line dataKey="zFL" stroke={CL.fl} dot={false} strokeWidth={1.5} name="FL" />
            <Line dataKey="zFR" stroke={CL.fr} dot={false} strokeWidth={1.5} name="FR" />
            <Line dataKey="zRL" stroke={CL.rl} dot={false} strokeWidth={1.5} name="RL" />
            <Line dataKey="zRR" stroke={CL.rr} dot={false} strokeWidth={1.5} name="RR" />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 9, fontFamily: FN, color: "#506080" }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 2. Roll & Pitch */}
      <ChartCard title="ROLL & PITCH [deg]">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xProps} />
            <YAxis {...yProps("deg")} />
            <Tooltip {...CHART_TT} />
            <Line dataKey="roll" stroke="#f472b6" dot={false} strokeWidth={1.5} name="Roll" />
            <Line dataKey="pitch" stroke="#a78bfa" dot={false} strokeWidth={1.5} name="Pitch" />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 9, fontFamily: FN }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 3. Lateral & Longitudinal G */}
      <ChartCard title="ACCELERATIONS [G]">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xProps} />
            <YAxis {...yProps("G")} />
            <Tooltip {...CHART_TT} />
            <Line dataKey="ay" stroke="#00b8e6" dot={false} strokeWidth={1.5} name="Ay (lat)" />
            <Line dataKey="ax" stroke="#ffb020" dot={false} strokeWidth={1.5} name="Ax (lon)" />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 9, fontFamily: FN }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 4. Speed Profile */}
      <ChartCard title="SPEED [m/s]">
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xProps} />
            <YAxis {...yProps("m/s")} domain={[0, 32]} />
            <Tooltip {...CHART_TT} />
            <Area dataKey="spd" stroke="#23d160" fill="rgba(35,209,96,0.08)" strokeWidth={1.5} name="Speed" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 5. Corner Vertical Loads */}
      <ChartCard title="CORNER LOADS Fz [N]">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xProps} />
            <YAxis {...yProps("N")} />
            <Tooltip {...CHART_TT} />
            <Line dataKey="Fz_FL" stroke={CL.fl} dot={false} strokeWidth={1.5} name="FL" />
            <Line dataKey="Fz_FR" stroke={CL.fr} dot={false} strokeWidth={1.5} name="FR" />
            <Line dataKey="Fz_RL" stroke={CL.rl} dot={false} strokeWidth={1.5} name="RL" />
            <Line dataKey="Fz_RR" stroke={CL.rr} dot={false} strokeWidth={1.5} name="RR" />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 9, fontFamily: FN }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 6. Camber Evolution */}
      <ChartCard title="DYNAMIC CAMBER [deg]">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xProps} />
            <YAxis {...yProps("deg")} />
            <Tooltip {...CHART_TT} />
            <Line dataKey="camFL" stroke={CL.fl} dot={false} strokeWidth={1.5} name="FL" />
            <Line dataKey="camFR" stroke={CL.fr} dot={false} strokeWidth={1.5} name="FR" />
            <Line dataKey="camRL" stroke={CL.rl} dot={false} strokeWidth={1.5} name="RL" />
            <Line dataKey="camRR" stroke={CL.rr} dot={false} strokeWidth={1.5} name="RR" />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 9, fontFamily: FN }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 7. Damper Velocity */}
      <ChartCard title="DAMPER VELOCITY [mm/s]">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xProps} />
            <YAxis {...yProps("mm/s")} />
            <Tooltip {...CHART_TT} />
            <Line dataKey="vFL" stroke={CL.fl} dot={false} strokeWidth={1.2} name="FL" />
            <Line dataKey="vFR" stroke={CL.fr} dot={false} strokeWidth={1.2} name="FR" />
            <Line dataKey="vRL" stroke={CL.rl} dot={false} strokeWidth={1.2} name="RL" />
            <Line dataKey="vRR" stroke={CL.rr} dot={false} strokeWidth={1.2} name="RR" />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 9, fontFamily: FN }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 8. G-G Diagram */}
      <ChartCard title="G-G DIAGRAM">
        <ResponsiveContainer width="100%" height={180}>
          <ScatterChart>
            <CartesianGrid {...gridProps} />
            <XAxis type="number" dataKey="ay" name="Ay" tick={{ fontSize: 8, fill: CHART_AXIS }} stroke={CHART_AXIS}
              label={{ value: "Ay [G]", position: "bottom", style: { fontSize: 8, fill: "#384560" } }} domain={[-2.5, 2.5]} />
            <YAxis type="number" dataKey="ax" name="Ax" tick={{ fontSize: 8, fill: CHART_AXIS }} stroke={CHART_AXIS}
              label={{ value: "Ax [G]", angle: -90, position: "insideLeft", style: { fontSize: 8, fill: "#384560" } }} domain={[-3, 2.5]} />
            <Tooltip {...CHART_TT} />
            <ReferenceLine x={0} stroke="rgba(255,255,255,0.08)" />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" />
            <Scatter data={ggData} fill="#00b8e6" fillOpacity={0.5} r={2} />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
export default function SuspensionViz() {
  const ref = useRef(null), st = useRef({});
  const [mn, setMn] = useState("lap");
  const [ps, setPs] = useState(false);
  const [hd, setHd] = useState({ roll: 0, pitch: 0, ay: 0, ax: 0, spd: 0, zFL: 0, zFR: 0, zRL: 0, zRR: 0 });
  const [history, setHistory] = useState([]);
  const histRef = useRef([]);
  const psR = useRef(false); psR.current = ps;

  // Reset history when maneuver changes
  useEffect(() => {
    st.current.data = genData(mn);
    st.current.idx = 0;
    histRef.current = [];
    setHistory([]);
  }, [mn]);

  useEffect(() => {
    const el = ref.current; if (!el) return;
    const sc = new THREE.Scene();
    sc.background = new THREE.Color(0x060a14);
    sc.fog = new THREE.FogExp2(0x060a14, 0.04);
    const cam = new THREE.PerspectiveCamera(34, el.clientWidth / el.clientHeight, 0.05, 80);
    const ren = new THREE.WebGLRenderer({ antialias: true });
    ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    ren.setSize(el.clientWidth, el.clientHeight);
    ren.shadowMap.enabled = true; ren.shadowMap.type = THREE.PCFSoftShadowMap;
    ren.toneMapping = THREE.ACESFilmicToneMapping; ren.toneMappingExposure = 1.1;
    el.appendChild(ren.domElement);

    const M = buildMats(); buildLights(sc);
    const groundRefs = buildGround(sc, M.ground);
    let distAccum = 0;

    const bg = new THREE.Group(); bg.position.set(0, V.hCG, 0); sc.add(bg);
    buildChassis(bg, M); buildFW(bg, M); buildRW(bg, M);

    const corners = {
      fl: buildCorner(sc, bg, M, "fl", V.lf, 1, V.tF, true),
      fr: buildCorner(sc, bg, M, "fr", V.lf, -1, V.tF, true),
      rl: buildCorner(sc, bg, M, "rl", -V.lr, 1, V.tR, false),
      rr: buildCorner(sc, bg, M, "rr", -V.lr, -1, V.tR, false),
    };
    const coils = {};
    for (const [k, c] of Object.entries(corners)) coils[k] = buildCoilover(bg, M, c);

    let cT = Math.PI * 0.22, cP = Math.PI * 0.19, cR = 3.3;
    const cTg = v3(0, 0.15, 0);
    let drg = false, pn = false, lm = { x: 0, y: 0 };
    const uC = () => { cam.position.set(cR * Math.sin(cP) * Math.sin(cT) + cTg.x, cR * Math.cos(cP) + cTg.y, cR * Math.sin(cP) * Math.cos(cT) + cTg.z); cam.lookAt(cTg); };
    uC();

    const oD = e => { if (e.shiftKey) pn = true; else drg = true; lm = { x: e.clientX ?? e.touches?.[0]?.clientX ?? 0, y: e.clientY ?? e.touches?.[0]?.clientY ?? 0 }; };
    const oM = e => { const cx = e.clientX ?? e.touches?.[0]?.clientX ?? 0, cy = e.clientY ?? e.touches?.[0]?.clientY ?? 0, dx = cx - lm.x, dy = cy - lm.y; lm = { x: cx, y: cy }; if (drg) { cT -= dx * 0.005; cP = Math.max(0.06, Math.min(Math.PI * 0.47, cP - dy * 0.005)); uC(); } else if (pn) { cTg.addScaledVector(v3(0,0,0).setFromMatrixColumn(cam.matrixWorld, 0), -dx * 0.003).addScaledVector(v3(0,0,0).setFromMatrixColumn(cam.matrixWorld, 1), dy * 0.003); uC(); } };
    const oU = () => { drg = false; pn = false; };
    const oW = e => { cR = Math.max(1.2, Math.min(10, cR + e.deltaY * 0.003)); uC(); };

    const cv = ren.domElement;
    cv.addEventListener("mousedown", oD); cv.addEventListener("mousemove", oM);
    cv.addEventListener("mouseup", oU); cv.addEventListener("mouseleave", oU);
    cv.addEventListener("wheel", oW, { passive: true });
    cv.addEventListener("touchstart", oD, { passive: true }); cv.addEventListener("touchmove", oM, { passive: true }); cv.addEventListener("touchend", oU);

    st.current.data = genData("lap"); st.current.idx = 0;
    const clk = new THREE.Clock(); let hC = 0, aId;

    const loop = () => {
      aId = requestAnimationFrame(loop);
      const dt = Math.min(clk.getDelta(), 0.05);
      if (!psR.current && st.current.data) {
        const d = st.current.data, i = st.current.idx % d.length, f = d[i];
        const tvl = tick(bg, corners, coils, f, dt);

        const frame = {
          t: f.t, roll: +f.roll.toFixed(2), pitch: +f.pitch.toFixed(2),
          ay: +f.ay.toFixed(2), ax: +f.ax.toFixed(2), spd: +f.spd.toFixed(1),
          zFL: +(tvl.fl ?? 0).toFixed(1), zFR: +(tvl.fr ?? 0).toFixed(1),
          zRL: +(tvl.rl ?? 0).toFixed(1), zRR: +(tvl.rr ?? 0).toFixed(1),
        };

        // Accumulate history (keep all frames for current maneuver cycle)
        histRef.current.push(frame);
        if (histRef.current.length > 480) histRef.current.shift();

        if (++hC % 4 === 0) {
          setHd(frame);
        }
        // Update charts every ~12 frames (~5 fps) for performance
        if (hC % 12 === 0) {
          setHistory([...histRef.current]);
        }

        st.current.idx = i + 1;

        distAccum += f.spd * dt;
        const roadWrap = 1.1 * 30;
        groundRefs.road.position.x = -(distAccum % roadWrap);
        groundRefs.road.position.z += f.ay * 0.003;
        groundRefs.road.rotation.y = f.steer * 0.15;
        groundRefs.grid.position.x = -(distAccum % 0.5);
      }
      if (!drg && !pn) { cT += 0.0006; uC(); }
      ren.render(sc, cam);
    };
    loop();

    const oR = () => { cam.aspect = el.clientWidth / el.clientHeight; cam.updateProjectionMatrix(); ren.setSize(el.clientWidth, el.clientHeight); };
    window.addEventListener("resize", oR);
    return () => {
      cancelAnimationFrame(aId); window.removeEventListener("resize", oR);
      cv.removeEventListener("mousedown", oD); cv.removeEventListener("mousemove", oM);
      cv.removeEventListener("mouseup", oU); cv.removeEventListener("mouseleave", oU); cv.removeEventListener("wheel", oW);
      cv.removeEventListener("touchstart", oD); cv.removeEventListener("touchmove", oM); cv.removeEventListener("touchend", oU);
      ren.dispose(); if (el.contains(ren.domElement)) el.removeChild(ren.domElement);
    };
  }, []);

  return (
    <div style={{ width: "100%", minHeight: "100vh", background: "#060a14", overflow: "auto" }}>
      {/* 3D Viewport */}
      <div style={{ width: "100%", height: "65vh", position: "relative", overflow: "hidden" }}>
        <div ref={ref} style={{ width: "100%", height: "100%" }} />
        <HUD d={hd} mn={mn} setMn={setMn} ps={ps} setPs={setPs} />
      </div>

      {/* Section divider */}
      <div style={{
        padding: "14px 20px 8px", display: "flex", alignItems: "center", gap: 12,
        borderTop: "1px solid rgba(0,184,230,0.10)",
      }}>
        <span style={{ fontSize: 11, fontWeight: 800, letterSpacing: 2, color: "#00b8e6", fontFamily: FN }}>
          TELEMETRY & KINEMATIC MAPS
        </span>
        <div style={{ flex: 1, height: 1, background: "rgba(0,184,230,0.10)" }} />
        <span style={{ fontSize: 9, color: "#2a3550", fontFamily: FN }}>
          {history.length} SAMPLES · {mn.toUpperCase()}
        </span>
      </div>

      {/* Charts */}
      <TelemetryCharts history={history} maneuver={mn} />
    </div>
  );
}