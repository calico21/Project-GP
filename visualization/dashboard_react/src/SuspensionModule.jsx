import React, { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

const VP = { wheelbase: 1.55, trackF: 1.20, trackR: 1.18, lf: 0.8525, lr: 0.6975, hCG: 0.33, wheelR: 0.2032, tireW: 0.205, uprightH: 0.258, mrF: 1.14, mrR: 1.16 };

function Susp3D({ maneuver }) {
  const ref = useRef(null);
  const anim = useRef(null);
  const fr = useRef(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const W = el.clientWidth, H = el.clientHeight || 440;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050710, 0.5);
    const cam = new THREE.PerspectiveCamera(38, W / H, 0.01, 50);
    cam.position.set(2.4, 1.15, 2.5);
    cam.lookAt(0, 0.18, 0);

    const ren = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    ren.setSize(W, H);
    ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    ren.shadowMap.enabled = true;
    ren.setClearColor(0x000000, 0);
    el.appendChild(ren.domElement);

    scene.add(new THREE.AmbientLight(0x4060a0, 0.5));
    const dl = new THREE.DirectionalLight(0xffffff, 1.1);
    dl.position.set(3, 5, 4); dl.castShadow = true; scene.add(dl);
    const p1 = new THREE.PointLight(0x00d2ff, 0.3, 8); p1.position.set(-2, 2, 0); scene.add(p1);
    const p2 = new THREE.PointLight(0xe10600, 0.15, 6); p2.position.set(2, 1, -2); scene.add(p2);

    const gnd = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: 0x080c12, roughness: 0.95, metalness: 0.1 }));
    gnd.rotation.x = -Math.PI / 2; gnd.receiveShadow = true; scene.add(gnd);
    const grid = new THREE.GridHelper(4, 40, 0x182040, 0x0c1420); grid.position.y = 0.001; scene.add(grid);

    const Mm = new THREE.MeshStandardMaterial({ color: 0x14181f, roughness: 0.25, metalness: 0.85 });
    const Mwl = new THREE.MeshStandardMaterial({ color: 0x2266dd, roughness: 0.35, metalness: 0.65 });
    const Mwu = new THREE.MeshStandardMaterial({ color: 0x7744cc, roughness: 0.35, metalness: 0.65 });
    const Mpr = new THREE.MeshStandardMaterial({ color: 0xe10600, roughness: 0.3, metalness: 0.75 });
    const Msp = new THREE.MeshStandardMaterial({ color: 0x00e676, roughness: 0.35, metalness: 0.55 });
    const Mwh = new THREE.MeshStandardMaterial({ color: 0x1a1e24, roughness: 0.15, metalness: 0.92 });
    const Mti = new THREE.MeshStandardMaterial({ color: 0x0e1014, roughness: 0.88, metalness: 0.03 });
    const Mup = new THREE.MeshStandardMaterial({ color: 0x3a4860, roughness: 0.3, metalness: 0.7 });

    // Monocoque
    const body = new THREE.Group(); scene.add(body);
    const ms = new THREE.Shape();
    ms.moveTo(-VP.lr - 0.05, 0.22); ms.lineTo(VP.lf + 0.1, 0.16); ms.lineTo(VP.lf + 0.35, 0.05);
    ms.lineTo(VP.lf + 0.1, -0.16); ms.lineTo(-VP.lr - 0.05, -0.22); ms.closePath();
    const mg = new THREE.ExtrudeGeometry(ms, { depth: 0.13, bevelEnabled: true, bevelThickness: 0.015, bevelSize: 0.01, bevelSegments: 3 });
    mg.rotateX(Math.PI / 2); mg.translate(0, VP.hCG - 0.065, 0);
    body.add(new THREE.Mesh(mg, Mm));
    const rh = new THREE.TorusGeometry(0.08, 0.008, 8, 12, Math.PI);
    rh.rotateY(Math.PI / 2); rh.translate(0.1, VP.hCG + 0.12, 0);
    body.add(new THREE.Mesh(rh, Mpr));

    const mkT = (a, b, r, mt) => {
      const d = new THREE.Vector3().subVectors(b, a), l = d.length();
      const g = new THREE.CylinderGeometry(r, r, l, 6);
      g.translate(0, l / 2, 0); g.rotateX(Math.PI / 2);
      const m = new THREE.Mesh(g, mt); m.position.copy(a); m.lookAt(b); scene.add(m);
    };

    const corners = {};
    [{ n: "fl", xa: VP.lf, ys: 1, tr: VP.trackF, f: true },
     { n: "fr", xa: VP.lf, ys: -1, tr: VP.trackF, f: true },
     { n: "rl", xa: -VP.lr, ys: 1, tr: VP.trackR, f: false },
     { n: "rr", xa: -VP.lr, ys: -1, tr: VP.trackR, f: false }].forEach(({ n, xa, ys, tr, f }) => {
      const ya = ys * tr / 2;
      const wg = new THREE.Group();
      wg.add(new THREE.Mesh(new THREE.CylinderGeometry(VP.wheelR * 0.62, VP.wheelR * 0.62, 0.155, 16).rotateX(Math.PI / 2), Mwh));
      wg.add(new THREE.Mesh(new THREE.TorusGeometry(VP.wheelR, VP.tireW * 0.42, 12, 24), Mti));
      wg.position.set(xa, VP.wheelR, ya); scene.add(wg);
      const um = new THREE.Mesh(new THREE.BoxGeometry(0.025, VP.uprightH, 0.035), Mup);
      um.position.set(xa, VP.wheelR, ya - ys * VP.tireW * 0.35); scene.add(um);
      const li = [new THREE.Vector3(xa + 0.09, VP.hCG - 0.11, ys * 0.16), new THREE.Vector3(xa - 0.09, VP.hCG - 0.11, ys * 0.16)];
      const ui = [new THREE.Vector3(xa + 0.07, VP.hCG + 0.02, ys * 0.14), new THREE.Vector3(xa - 0.07, VP.hCG + 0.02, ys * 0.14)];
      const lo = new THREE.Vector3(xa, VP.wheelR - VP.uprightH * 0.4, ya - ys * VP.tireW * 0.35);
      const uo = new THREE.Vector3(xa, VP.wheelR + VP.uprightH * 0.35, ya - ys * VP.tireW * 0.35);
      mkT(li[0], lo, 0.006, Mwl); mkT(li[1], lo, 0.006, Mwl);
      mkT(ui[0], uo, 0.005, Mwu); mkT(ui[1], uo, 0.005, Mwu);
      const bc = new THREE.Vector3(xa, VP.hCG - 0.02, ys * 0.13);
      mkT(lo.clone().lerp(uo, 0.25), bc, 0.0045, Mpr);
      const sm = new THREE.Vector3(xa, VP.hCG + 0.06, ys * 0.10);
      const spr = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.10, 8), Msp);
      spr.position.copy(bc).lerp(sm, 0.5); spr.lookAt(sm); scene.add(spr);
      corners[n] = { wg, um, f };
    });

    let theta = 0.4, drag = false, px = 0;
    const oD = e => { drag = true; px = e.clientX || e.touches?.[0]?.clientX || 0; };
    const oM = e => { if (!drag) return; const cx = e.clientX || e.touches?.[0]?.clientX || 0; theta -= (cx - px) * 0.005; px = cx; };
    const oU = () => { drag = false; };
    el.addEventListener("mousedown", oD); el.addEventListener("mousemove", oM); el.addEventListener("mouseup", oU);
    el.addEventListener("touchstart", oD, { passive: true }); el.addEventListener("touchmove", oM, { passive: true }); el.addEventListener("touchend", oU);

    const loop = () => {
      anim.current = requestAnimationFrame(loop);
      fr.current++;
      const t = fr.current * 0.015;
      let roll = 0, pitch = 0, steer = 0;
      const m = maneuver || "lap";
      if (m === "lap") { steer = 0.18 * Math.sin(t * 0.8); roll = steer * 13; pitch = 1.5 * Math.sin(t * 1.6); }
      else if (m === "skidpad") { steer = 0.25; roll = 2.8; pitch = -0.3; }
      else if (m === "brake") { steer = 0.02 * Math.sin(t * 0.3); roll = 0.2; pitch = -3 + 5 * Math.exp(-t * 0.4) * Math.sin(t * 3); }
      else if (m === "chicane") { steer = 0.3 * Math.sin(t * 1.8); roll = steer * 16; pitch = 0.5 * Math.sin(t * 3.6); }
      body.rotation.z = roll * Math.PI / 180;
      body.rotation.x = pitch * Math.PI / 180;
      body.position.y = VP.hCG;
      Object.values(corners).forEach(c => { c.wg.position.y = VP.wheelR; if (c.f) { c.wg.rotation.y = steer; c.um.rotation.y = steer; } });
      if (!drag) theta += 0.0018;
      cam.position.set(2.8 * Math.sin(theta), 1 + 0.15 * Math.sin(t * 0.08), 2.8 * Math.cos(theta));
      cam.lookAt(0, 0.2, 0);
      ren.render(scene, cam);
    };
    loop();

    return () => {
      cancelAnimationFrame(anim.current);
      el.removeEventListener("mousedown", oD); el.removeEventListener("mousemove", oM); el.removeEventListener("mouseup", oU);
      el.removeEventListener("touchstart", oD); el.removeEventListener("touchmove", oM); el.removeEventListener("touchend", oU);
      ren.dispose();
      if (el.contains(ren.domElement)) el.removeChild(ren.domElement);
    };
  }, [maneuver]);

  return <div ref={ref} style={{ width: "100%", height: 440, borderRadius: 12, overflow: "hidden" }} />;
}

export default function SuspensionModule({ data }) {
  const [mn, setMn] = useState("lap");
  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 18 }}>
        <KPI label="Eq. Defl. F" value="−12.8 mm" sub="front static" delay={0} />
        <KPI label="Eq. Defl. R" value="−14.2 mm" sub="rear static" delay={1} />
        <KPI label="Roll Gradient" value="0.68°/G" sub="computed" sentiment="positive" delay={2} />
        <KPI label="Motion Ratio" value={`${VP.mrF} / ${VP.mrR}`} sub="F / R" delay={3} />
      </div>

      <Sec title="Interactive 3D — Ter26 Pushrod-Bellcrank" right={
        <div style={{ display: "flex", gap: 4 }}>
          {["lap", "skidpad", "brake", "chicane"].map(m => (
            <Pill key={m} active={mn === m} label={m} onClick={() => setMn(m)} color={C.red} />
          ))}
        </div>
      }>
        <div style={{ ...GL, padding: 0, overflow: "hidden" }}>
          <Susp3D maneuver={mn} />
          <div style={{ display: "flex", justifyContent: "center", gap: 20, padding: "10px 0 12px", background: "rgba(0,0,0,0.25)", backdropFilter: "blur(10px)" }}>
            {[["Lower Wishbone", "#2266dd"], ["Upper Wishbone", "#7744cc"], ["Pushrod", "#e10600"], ["Spring", "#00e676"], ["Chassis", "#2a3040"]].map(([l, c]) => (
              <div key={l} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 9, fontFamily: C.dt, color: C.md }}>
                <div style={{ width: 12, height: 3, borderRadius: 2, background: c }} />{l}
              </div>
            ))}
          </div>
        </div>
      </Sec>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Sec title="Steering & Roll">
          <GC>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="t" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                <YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                <Tooltip contentStyle={TT} />
                <Line type="monotone" dataKey="steer" stroke={C.pr} strokeWidth={1.5} dot={false} name="δ [deg]" />
                <Line type="monotone" dataKey="roll" stroke={C.am} strokeWidth={1.5} dot={false} name="Roll [deg]" />
                <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd, fontWeight: 600 }} />
              </LineChart>
            </ResponsiveContainer>
          </GC>
        </Sec>
        <Sec title="Suspension Heave">
          <GC>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="t" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                <YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} domain={[-20, 0]} />
                <Tooltip contentStyle={TT} />
                <Area type="monotone" dataKey="heave_f" stroke={C.cy} fill={`${C.cy}06`} strokeWidth={1.5} dot={false} name="Front [mm]" />
                <Area type="monotone" dataKey="heave_r" stroke={C.gn} fill={`${C.gn}06`} strokeWidth={1.5} dot={false} name="Rear [mm]" />
                <ReferenceLine y={-12.8} stroke={C.cy} strokeDasharray="4 3" strokeOpacity={0.35} />
                <ReferenceLine y={-14.2} stroke={C.gn} strokeDasharray="4 3" strokeOpacity={0.35} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd, fontWeight: 600 }} />
              </AreaChart>
            </ResponsiveContainer>
          </GC>
        </Sec>
      </div>
    </>
  );
}
