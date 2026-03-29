// ═══════════════════════════════════════════════════════════════════════════
// src/AerodynamicsModule.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════
// Dedicated aerodynamics department hub. Consolidates all aero data that was
// previously scattered across ∇ Insights (AeroTab), LiveMode (AeroPlatform),
// Overview (Pillar 05), and LiveGraphSystem (Aero group).
//
// Sub-tabs (8):
//   1. Aero Maps       — 5D surrogate C_L/C_D/CoP contour exploration
//   2. Force Balance   — Component-level downforce/drag decomposition
//   3. CoP Tracker     — Center-of-pressure migration & stability
//   4. Platform        — Ride height, pitch, roll attitude analysis
//   5. Sensitivity     — ∂F_aero/∂setup tornado charts
//   6. Fidelity        — Surrogate-vs-CFD parity & training coverage
//   7. Yaw & Crosswind — Side force, yaw moment, stability derivatives
//   8. Aero × Grip     — Downforce-to-tire-grip coupling analysis
//
// Integration (3 lines in App.jsx):
//   NAV: { key: “aero”, label: “Aerodynamics”, icon: “▽” }
//   Import: import AerodynamicsModule from “./AerodynamicsModule.jsx”
//   Route: case “aero”: return <AerodynamicsModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo, useCallback } from “react”;
import {
LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from “recharts”;
import { C, GL, GS, TT, AX } from “./theme.js”;
import { KPI, Sec, GC, Pill } from “./components.jsx”;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const AERO = “#ff6090”;
const AERO_G = “rgba(255,96,144,0.10)”;
const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });

const TABS = [
{ key: “maps”,        label: “Aero Maps” },
{ key: “forces”,      label: “Force Balance” },
{ key: “cop”,         label: “CoP Tracker” },
{ key: “platform”,    label: “Platform” },
{ key: “sensitivity”, label: “Sensitivity” },
{ key: “fidelity”,    label: “Fidelity” },
{ key: “yaw”,         label: “Yaw & Crosswind” },
{ key: “coupling”,    label: “Aero × Grip” },
];

// ═══════════════════════════════════════════════════════════════════════════
// SEEDED RNG
// ═══════════════════════════════════════════════════════════════════════════
function srng(seed) {
let s = seed;
return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; };
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

// 1. Full 5D aero map (pitch × rh slices, parameterized by roll/yaw)
function gAeroMap(res = 30) {
const data = [];
for (let pi = 0; pi < res; pi++) {
for (let ri = 0; ri < res; ri++) {
const pitch = (pi / (res - 1)) * 4 - 2;       // -2° to +2°
const rh = (ri / (res - 1)) * 50 + 10;         // 10–60 mm
const rhOpt = 30;
// Physics-plausible: C_L peaks at moderate nose-down, degrades at extreme low rh (stall)
const stallPenalty = rh < 18 ? 0.3 * Math.pow((18 - rh) / 8, 2) : 0;
const CL = 3.05 + 0.38 * pitch - 0.015 * (rh - rhOpt) * (rh - rhOpt) / 100
+ 0.08 * Math.sin(pitch * 0.7) - stallPenalty;
const CD = 1.15 + 0.055 * Math.abs(pitch) + 0.0008 * (rh - 25) * (rh - 25)
+ (stallPenalty > 0 ? 0.1 * stallPenalty : 0);
const CoPx = 0.475 + 0.018 * pitch - 0.0008 * (rh - 30);
const CoPy = 0.50 + 0.003 * Math.sin(pitch * rh * 0.005);
data.push({
pitch: +pitch.toFixed(2), rh: +rh.toFixed(1),
CL: +CL.toFixed(4), CD: +CD.toFixed(4),
CoPx: +CoPx.toFixed(4), CoPy: +CoPy.toFixed(4),
LD: +(CL / CD).toFixed(3),
ClA: +(CL * 4.554).toFixed(2),
});
}
}
return data;
}

// 2. Force decomposition over a simulated lap
function gForceDecomp(n = 200) {
const R = srng(3001);
const data = [];
for (let i = 0; i < n; i++) {
const s = i * (1370 / n); // lap distance in metres
const speed = 10 + 12 * Math.sin(i * 0.03) + 4 * Math.sin(i * 0.08) + 2 * R();
const q = 0.5 * 1.225 * speed * speed; // dynamic pressure
const fw = q * 1.35 * (0.40 + 0.02 * Math.sin(i * 0.05));   // front wing ~40%
const rw = q * 0.75 * (0.22 + 0.01 * Math.sin(i * 0.07));   // rear wing ~22%
const floor = q * 1.10 * (0.33 + 0.015 * Math.sin(i * 0.04)); // floor ~33%
const body = q * 0.18 * (0.05 + 0.005 * R());                // body ~5%
const drag_fw = q * 0.25; const drag_rw = q * 0.35;
const drag_floor = q * 0.15; const drag_body = q * 0.40;
const total = fw + rw + floor + body;
data.push({
s: +s.toFixed(0), speed: +speed.toFixed(1),
fw: +fw.toFixed(0), rw: +rw.toFixed(0), floor: +floor.toFixed(0), body: +body.toFixed(0),
total: +total.toFixed(0),
drag_total: +(drag_fw + drag_rw + drag_floor + drag_body).toFixed(0),
frontPct: +((fw + floor * 0.55) / total * 100).toFixed(1),
LD: +(total / (drag_fw + drag_rw + drag_floor + drag_body)).toFixed(2),
});
}
return data;
}

// 3. CoP trajectory over a lap
function gCoPTrace(n = 200) {
const R = srng(3101);
return Array.from({ length: n }, (_, i) => {
const s = i * (1370 / n);
const braking = Math.sin(i * 0.04) < -0.3;
const cornering = Math.abs(Math.sin(i * 0.025)) > 0.5;
const latG = 1.5 * Math.sin(i * 0.025) + 0.2 * R();
const lonG = -0.8 * Math.cos(i * 0.04) + 0.15 * R();
const speed = 10 + 12 * Math.sin(i * 0.03) + 2 * R();
// CoP shifts forward under braking (nose dive), backward under accel
const CoPx = 0.480 + (braking ? 0.035 : -0.015) + 0.005 * R();
// CoP shifts laterally with roll
const CoPy = 0.500 + 0.008 * Math.sin(i * 0.025) + 0.003 * R();
return {
s: +s.toFixed(0), CoPx: +CoPx.toFixed(4), CoPy: +CoPy.toFixed(4),
latG: +latG.toFixed(3), lonG: +lonG.toFixed(3), speed: +speed.toFixed(1),
zone: braking ? “Braking” : cornering ? “Cornering” : “Straight”,
};
});
}

// 4. Platform attitude trace
function gPlatform(n = 300) {
const R = srng(3201);
return Array.from({ length: n }, (_, i) => {
const t = i * 0.05;
const speed = 10 + 12 * Math.sin(i * 0.02) + 2 * R();
const rh_f = 28 - 0.15 * speed + 2 * R() + 3 * Math.sin(i * 0.03);
const rh_r = 32 - 0.12 * speed + 1.5 * R() + 2 * Math.sin(i * 0.025);
const pitch = (rh_f - rh_r) / 155 * (180 / Math.PI); // approximate
const roll = 1.8 * Math.sin(i * 0.018) + 0.3 * R();
const bsF = rh_f < 12;
const bsR = rh_r < 14;
const stallRisk = rh_f < 15 || rh_r < 15;
return {
t: +t.toFixed(2), speed: +speed.toFixed(1),
rh_f: +Math.max(5, rh_f).toFixed(1), rh_r: +Math.max(5, rh_r).toFixed(1),
pitch: +pitch.toFixed(3), roll: +roll.toFixed(3),
bsF, bsR, stallRisk,
};
});
}

// 5. Sensitivity: ∂C_L/∂param for 28 setup parameters
function gAeroSens() {
const R = srng(3301);
const PARAMS = [
“k_f”,“k_r”,“arb_f”,“arb_r”,“c_low_f”,“c_low_r”,“c_hi_f”,“c_hi_r”,
“v_knee_f”,“v_knee_r”,“reb_f”,“reb_r”,“h_ride_f”,“h_ride_r”,
“camber_f”,“camber_r”,“toe_f”,“toe_r”,“castor”,“anti_sq”,
“anti_dive_f”,“anti_dive_r”,“anti_lift”,“diff_lock”,“brake_bias”,
“h_cg”,“bs_f”,“bs_r”,
];
const sens = PARAMS.map(p => {
// Ride height and spring rates dominate aero sensitivity
let base = 0.001 + R() * 0.005;
if (p.includes(“ride”)) base = 0.08 + R() * 0.04;
else if (p.includes(“k_”)) base = 0.03 + R() * 0.02;
else if (p.includes(“arb”)) base = 0.015 + R() * 0.01;
else if (p.includes(“h_cg”)) base = 0.025 + R() * 0.015;
else if (p.includes(“bs_”)) base = 0.02 + R() * 0.01;
const sign = p.includes(“ride_f”) ? 1 : (R() > 0.5 ? 1 : -1);
return {
param: p,
dCL: +(sign * base).toFixed(4),
dCD: +(sign * base * 0.3 * (0.5 + R())).toFixed(4),
dLD: +(sign * base * 2 * (R() - 0.3)).toFixed(4),
dCoP: +(sign * base * 0.5 * (R() - 0.4)).toFixed(4),
absCL: +Math.abs(sign * base).toFixed(4),
};
});
return sens.sort((a, b) => b.absCL - a.absCL);
}

// 6. Surrogate fidelity — simulated CFD parity data
function gFidelity(n = 120) {
const R = srng(3401);
const points = [];
for (let i = 0; i < n; i++) {
const actualCL = 2.2 + R() * 1.2;
const residual = (R() - 0.5) * 0.12 + (R() - 0.5) * 0.04;
const predCL = actualCL + residual;
const inHull = R() > 0.12; // 88% coverage
points.push({
actual: +actualCL.toFixed(4), pred: +predCL.toFixed(4),
residual: +residual.toFixed(4), absResidual: +Math.abs(residual).toFixed(4),
inHull,
pitch: +((R() - 0.5) * 4).toFixed(2),
rh: +(10 + R() * 50).toFixed(1),
});
}
return points;
}

// 7. Crosswind & yaw stability
function gCrosswind(nYaw = 25) {
const R = srng(3501);
return Array.from({ length: nYaw }, (_, i) => {
const yaw = (i / (nYaw - 1)) * 20 - 10; // -10° to +10°
// Side force and yaw moment increase roughly linearly with yaw
const CY = 0.012 * yaw + 0.003 * yaw * Math.abs(yaw) / 10 + (R() - 0.5) * 0.01;
const CN = 0.008 * yaw + 0.002 * yaw * Math.abs(yaw) / 10 + (R() - 0.5) * 0.005;
const CL_loss = -0.002 * yaw * yaw; // downforce loss in crosswind
const CD_increase = 0.001 * yaw * yaw;
return {
yaw: +yaw.toFixed(1),
CY: +CY.toFixed(4), CN: +CN.toFixed(5),
dCL: +CL_loss.toFixed(4), dCD: +CD_increase.toFixed(4),
dCY_dyaw: +(0.012 + 0.006 * Math.abs(yaw) / 10).toFixed(5),
dCN_dyaw: +(0.008 + 0.004 * Math.abs(yaw) / 10).toFixed(5),
};
});
}

// 8. Aero × grip coupling
function gAeroGrip(n = 100) {
const R = srng(3601);
return Array.from({ length: n }, (_, i) => {
const CL = 2.0 + i * 0.015 + (R() - 0.5) * 0.1;
const downforce = CL * 0.5 * 1.225 * 15 * 15 * 4.554; // at ~15 m/s reference
const Fz_base = 300 * 9.81 / 4; // per-wheel static
const Fz_total = Fz_base + downforce / 4;
// Tire grip shows load sensitivity — diminishing returns
const mu = 1.65 - 0.08 * Math.log(Fz_total / 735);
const gripUtil = 82 + 12 * (CL - 2.0) / 1.5 - 3 * Math.pow((CL - 2.8) / 0.8, 2) + R() * 4;
const lapDelta = -0.04 * (CL - 2.0) + 0.02 * Math.pow(CL - 3.2, 2);
return {
CL: +CL.toFixed(3), downforce: +downforce.toFixed(0),
Fz: +Fz_total.toFixed(0), mu: +mu.toFixed(4),
gripUtil: +Math.min(100, Math.max(50, gripUtil)).toFixed(1),
lapDelta: +lapDelta.toFixed(3),
LD: +(CL / (1.15 + 0.05 * CL)).toFixed(3),
};
});
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 1: AERO MAPS
// ═══════════════════════════════════════════════════════════════════════════
function MapsTab() {
const aeroData = useMemo(() => gAeroMap(), []);
const [sliceRh, setSliceRh] = useState(30);
const [slicePitch, setSlicePitch] = useState(0);

const maxCL = Math.max(…aeroData.map(d => d.CL));
const bestPt = aeroData.find(d => d.CL >= maxCL - 0.0001);
const maxLD = Math.max(…aeroData.map(d => d.LD));

const pitchSlice = useMemo(() => aeroData.filter(d => Math.abs(d.rh - sliceRh) < 1), [aeroData, sliceRh]);
const rhSlice = useMemo(() => aeroData.filter(d => Math.abs(d.pitch - slicePitch) < 0.15), [aeroData, slicePitch]);

// Heatmap data: unique pitch values, for each -> CL at varying rh
const pitchVals = useMemo(() => […new Set(aeroData.map(d => d.pitch))], [aeroData]);
const rhVals = useMemo(() => […new Set(aeroData.map(d => d.rh))], [aeroData]);
const heatmapCL = useMemo(() => {
const map = {};
aeroData.forEach(d => { map[`${d.pitch}_${d.rh}`] = d.CL; });
return map;
}, [aeroData]);

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(5, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Peak C_L” value={maxCL.toFixed(3)} sub={`pitch=${bestPt?.pitch}°, rh=${bestPt?.rh}mm`} sentiment=“positive” delay={0} />
<KPI label="Peak L/D" value={maxLD.toFixed(2)} sub="aerodynamic efficiency" sentiment="positive" delay={1} />
<KPI label=“Peak ClA” value={`${(maxCL * 4.554).toFixed(1)} m²`} sub=“downforce area” sentiment=“positive” delay={2} />
<KPI label=“CoP at Peak” value={`${((bestPt?.CoPx || 0.48) * 100).toFixed(1)}%`} sub=“from front axle” sentiment=“neutral” delay={3} />
<KPI label=“Operating Range” value={`${pitchVals.length}×${rhVals.length}`} sub=“pitch × rh grid” sentiment=“neutral” delay={4} />
</div>

```
  {/* Heatmap canvas */}
  <Sec title="C_L Contour — Pitch × Ride Height">
    <GC style={{ padding: 8 }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <div style={{ position: "relative" }}>
          <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, textAlign: "center", marginBottom: 4 }}>
            RIDE HEIGHT [mm] →
          </div>
          <div style={{ display: "flex" }}>
            <div style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: 7, color: C.dm, fontFamily: C.dt, textAlign: "center", marginRight: 4 }}>
              ← PITCH [°]
            </div>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${rhVals.length}, 1fr)`, gap: 1 }}>
              {pitchVals.map((p, pi) =>
                rhVals.map((r, ri) => {
                  const cl = heatmapCL[`${p}_${r}`] || 0;
                  const norm = (cl - 2.0) / (maxCL - 2.0);
                  const hue = 240 - norm * 240; // blue→red
                  const isSlice = Math.abs(r - sliceRh) < 1 || Math.abs(p - slicePitch) < 0.15;
                  return (
                    <div
                      key={`${pi}-${ri}`}
                      onClick={() => { setSliceRh(r); setSlicePitch(p); }}
                      style={{
                        width: 14, height: 14,
                        background: `hsla(${hue}, 80%, ${40 + norm * 20}%, 0.85)`,
                        border: isSlice ? `1px solid ${C.w}` : "none",
                        borderRadius: 1, cursor: "crosshair",
                      }}
                      title={`pitch=${p}°, rh=${r}mm → CL=${cl.toFixed(3)}`}
                    />
                  );
                })
              )}
            </div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, color: C.dm, fontFamily: C.dt, marginTop: 4, marginLeft: 16 }}>
            <span>{rhVals[0]}mm</span><span>{rhVals[rhVals.length - 1]}mm</span>
          </div>
        </div>
        {/* Legend */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 7, fontFamily: C.dt }}>
          <div style={{ color: C.dm, fontWeight: 700, letterSpacing: 1 }}>C_L</div>
          {[1.0, 0.75, 0.5, 0.25, 0].map(n => (
            <div key={n} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 10, height: 10, background: `hsla(${240 - n * 240}, 80%, ${40 + n * 20}%, 0.85)`, borderRadius: 1 }} />
              <span style={{ color: C.dm }}>{(2.0 + n * (maxCL - 2.0)).toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title={`C_L vs Pitch — rh=${sliceRh}mm slice`}>
      <GC><ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={pitchSlice} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="pitch" {...ax()} label={{ value: "Pitch [°]", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="CL" stroke={AERO} strokeWidth={2} dot={{ r: 2, fill: AERO }} name="C_L" />
          <Line type="monotone" dataKey="LD" stroke={C.cy} strokeWidth={1.5} dot={false} name="L/D" yAxisId="right" />
          <YAxis yAxisId="right" orientation="right" {...ax()} />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title={`C_L vs Ride Height — pitch=${slicePitch}° slice`}>
      <GC><ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={rhSlice} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="rh" {...ax()} label={{ value: "Ride Height [mm]", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea x1={10} x2={18} fill={C.red} fillOpacity={0.06} label={{ value: "STALL", fill: C.red, fontSize: 7 }} />
          <Line type="monotone" dataKey="CL" stroke={AERO} strokeWidth={2} dot={{ r: 2, fill: AERO }} name="C_L" />
          <Line type="monotone" dataKey="CD" stroke={C.am} strokeWidth={1.5} dot={false} name="C_D" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  {/* CoP map */}
  <Sec title="CoP_x vs Pitch (center of pressure fraction from front)" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={200}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="pitch" type="number" {...ax()} name="Pitch" />
        <YAxis dataKey="CoPx" type="number" {...ax()} domain={[0.42, 0.55]} name="CoP_x" />
        <Tooltip contentStyle={TT} />
        <ReferenceLine y={0.48} stroke={C.gn} strokeDasharray="4 4" label={{ value: "Target", fill: C.gn, fontSize: 7 }} />
        <Scatter data={pitchSlice} fill={AERO} fillOpacity={0.6} r={3} name="CoP_x" />
      </ScatterChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 2: FORCE BALANCE
// ═══════════════════════════════════════════════════════════════════════════
function ForcesTab() {
const forces = useMemo(() => gForceDecomp(), []);
const maxDF = Math.max(…forces.map(f => f.total));
const avgLD = forces.reduce((a, f) => a + f.LD, 0) / forces.length;
const avgFrontPct = forces.reduce((a, f) => a + f.frontPct, 0) / forces.length;
const maxDrag = Math.max(…forces.map(f => f.drag_total));

// Drag breakdown averages
const avgDrag = forces.reduce((a, f) => a + f.drag_total, 0) / forces.length;
const dragBreakdown = [
{ component: “Body/Wheels”, drag: +(avgDrag * 0.40).toFixed(0), color: C.dm },
{ component: “Rear Wing”, drag: +(avgDrag * 0.30).toFixed(0), color: C.am },
{ component: “Front Wing”, drag: +(avgDrag * 0.18).toFixed(0), color: AERO },
{ component: “Floor”, drag: +(avgDrag * 0.12).toFixed(0), color: C.gn },
];

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(5, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Peak Downforce” value={`${maxDF} N`} sub=“at max speed” sentiment=“positive” delay={0} />
<KPI label="Avg L/D" value={avgLD.toFixed(2)} sub="lap average" sentiment={avgLD > 2.5 ? “positive” : “amber”} delay={1} />
<KPI label=“F/R Balance” value={`${avgFrontPct.toFixed(1)}%`} sub=“front downforce” sentiment={Math.abs(avgFrontPct - 48) < 4 ? “positive” : “amber”} delay={2} />
<KPI label=“Peak Drag” value={`${maxDrag} N`} sub=“total aero drag” sentiment=“neutral” delay={3} />
<KPI label=“Drag @ 80 kph” value={`${(0.5 * 1.225 * 22.2 * 22.2 * 1.15 * 4.554).toFixed(0)} N`} sub=“estimated” sentiment=“neutral” delay={4} />
</div>

```
  <Sec title="Downforce Decomposition Over Lap [N]">
    <GC><ResponsiveContainer width="100%" height={260}>
      <AreaChart data={forces} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="s" {...ax()} label={{ value: "Distance [m]", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
        <YAxis {...ax()} />
        <Tooltip contentStyle={TT} />
        <Area type="monotone" dataKey="fw" stackId="1" fill={AERO} fillOpacity={0.35} stroke={AERO} strokeWidth={1} name="Front Wing" />
        <Area type="monotone" dataKey="floor" stackId="1" fill={C.gn} fillOpacity={0.25} stroke={C.gn} strokeWidth={1} name="Floor" />
        <Area type="monotone" dataKey="rw" stackId="1" fill={C.am} fillOpacity={0.25} stroke={C.am} strokeWidth={1} name="Rear Wing" />
        <Area type="monotone" dataKey="body" stackId="1" fill={C.pr} fillOpacity={0.2} stroke={C.pr} strokeWidth={1} name="Body" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </AreaChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Front/Rear Balance [%]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <LineChart data={forces.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="s" {...ax()} />
          <YAxis {...ax()} domain={[35, 65]} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={48} stroke={C.gn} strokeDasharray="4 4" />
          <Line type="monotone" dataKey="frontPct" stroke={AERO} strokeWidth={1.5} dot={false} name="Front %" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="L/D Over Lap">
      <GC><ResponsiveContainer width="100%" height={200}>
        <LineChart data={forces.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="s" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="LD" stroke={C.cy} strokeWidth={1.5} dot={false} name="L/D" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Drag Breakdown [N avg]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <BarChart data={dragBreakdown} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 70 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
          <XAxis type="number" {...ax()} />
          <YAxis dataKey="component" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={65} />
          <Tooltip contentStyle={TT} />
          <Bar dataKey="drag" barSize={14} radius={[0, 4, 4, 0]}>
            {dragBreakdown.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 3: CoP TRACKER
// ═══════════════════════════════════════════════════════════════════════════
function CoPTab() {
const cop = useMemo(() => gCoPTrace(), []);
const meanCoP = cop.reduce((a, c) => a + c.CoPx, 0) / cop.length;
const copRange = Math.max(…cop.map(c => c.CoPx)) - Math.min(…cop.map(c => c.CoPx));
const targetCoP = 0.480;

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Mean CoP_x” value={`${(meanCoP * 100).toFixed(1)}%`} sub=“from front axle” sentiment={Math.abs(meanCoP - targetCoP) < 0.02 ? “positive” : “amber”} delay={0} />
<KPI label=“CoP Migration” value={`${(copRange * 1550).toFixed(1)} mm`} sub=“total range over lap” sentiment={copRange < 0.05 ? “positive” : “amber”} delay={1} />
<KPI label=“Target CoP” value={`${(targetCoP * 100).toFixed(1)}%`} sub=“design target” sentiment=“neutral” delay={2} />
<KPI label=“Deviation” value={`${((meanCoP - targetCoP) * 1550).toFixed(1)} mm`} sub=“from target” sentiment={Math.abs(meanCoP - targetCoP) < 0.015 ? “positive” : “amber”} delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="CoP_x Over Lap Distance">
      <GC><ResponsiveContainer width="100%" height={220}>
        <AreaChart data={cop} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="s" {...ax()} />
          <YAxis {...ax()} domain={[0.43, 0.55]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
          <Tooltip contentStyle={TT} formatter={v => `${(v * 100).toFixed(2)}%`} />
          <ReferenceLine y={targetCoP} stroke={C.gn} strokeDasharray="4 4" label={{ value: "Target", fill: C.gn, fontSize: 7 }} />
          <Area type="monotone" dataKey="CoPx" stroke={AERO} fill={AERO_G} strokeWidth={1.5} dot={false} name="CoP_x" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="CoP_x vs Lateral G — colored by zone">
      <GC><ResponsiveContainer width="100%" height={220}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="latG" type="number" {...ax()} name="Lat G" />
          <YAxis dataKey="CoPx" type="number" {...ax()} domain={[0.43, 0.55]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
          <Tooltip contentStyle={TT} />
          <Scatter data={cop.filter(c => c.zone === "Braking")} fill={C.red} fillOpacity={0.6} r={2} name="Braking" />
          <Scatter data={cop.filter(c => c.zone === "Cornering")} fill={C.am} fillOpacity={0.6} r={2} name="Cornering" />
          <Scatter data={cop.filter(c => c.zone === "Straight")} fill={C.gn} fillOpacity={0.4} r={2} name="Straight" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="CoP_x vs Speed">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="speed" type="number" {...ax()} name="Speed [m/s]" />
          <YAxis dataKey="CoPx" type="number" {...ax()} domain={[0.43, 0.55]} />
          <Tooltip contentStyle={TT} />
          <Scatter data={cop} fill={C.cy} fillOpacity={0.5} r={2} />
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="CoP_y (Lateral Shift) vs Roll">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="latG" type="number" {...ax()} name="Lateral G" />
          <YAxis dataKey="CoPy" type="number" {...ax()} domain={[0.48, 0.52]} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0.5} stroke={C.dm} strokeDasharray="3 3" />
          <Scatter data={cop} fill={C.pr} fillOpacity={0.5} r={2} />
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 4: PLATFORM ATTITUDE
// ═══════════════════════════════════════════════════════════════════════════
function PlatformTab() {
const platform = useMemo(() => gPlatform(), []);
const meanRhF = platform.reduce((a, p) => a + p.rh_f, 0) / platform.length;
const meanRhR = platform.reduce((a, p) => a + p.rh_r, 0) / platform.length;
const maxPitch = Math.max(…platform.map(p => Math.abs(p.pitch)));
const stallEvents = platform.filter(p => p.stallRisk).length;
const bsEventsF = platform.filter(p => p.bsF).length;

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(5, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Mean rh_f” value={`${meanRhF.toFixed(1)} mm`} sub=“front ride height” sentiment={meanRhF > 20 ? “positive” : “amber”} delay={0} />
<KPI label=“Mean rh_r” value={`${meanRhR.toFixed(1)} mm`} sub=“rear ride height” sentiment={meanRhR > 22 ? “positive” : “amber”} delay={1} />
<KPI label=“Max |Pitch|” value={`${maxPitch.toFixed(2)}°`} sub=“peak pitch angle” sentiment={maxPitch < 1.5 ? “positive” : “amber”} delay={2} />
<KPI label=“Stall Risk” value={stallEvents} sub=“timesteps below 15mm” sentiment={stallEvents === 0 ? “positive” : “negative”} delay={3} />
<KPI label=“Bump Stop F” value={bsEventsF} sub=“front engagements” sentiment={bsEventsF < 5 ? “positive” : “amber”} delay={4} />
</div>

```
  <Sec title="Ride Height — Front & Rear [mm]">
    <GC><ResponsiveContainer width="100%" height={240}>
      <ComposedChart data={platform.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="t" {...ax()} />
        <YAxis {...ax()} domain={[0, 55]} />
        <Tooltip contentStyle={TT} />
        <ReferenceArea y1={0} y2={15} fill={C.red} fillOpacity={0.05} label={{ value: "STALL ZONE", fill: C.red, fontSize: 7 }} />
        <ReferenceArea y1={22} y2={38} fill={C.gn} fillOpacity={0.03} label={{ value: "OPTIMAL", fill: C.gn, fontSize: 6, position: "insideTopRight" }} />
        <Line type="monotone" dataKey="rh_f" stroke={AERO} strokeWidth={1.5} dot={false} name="Front rh" />
        <Line type="monotone" dataKey="rh_r" stroke={C.cy} strokeWidth={1.5} dot={false} name="Rear rh" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </ComposedChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Pitch Angle [°]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={platform.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
          <Area type="monotone" dataKey="pitch" stroke={C.am} fill={`${C.am}10`} strokeWidth={1.5} dot={false} name="Pitch" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Roll Angle [°]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={platform.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
          <Area type="monotone" dataKey="roll" stroke={C.pr} fill={`${C.pr}10`} strokeWidth={1.5} dot={false} name="Roll" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 5: SENSITIVITY
// ═══════════════════════════════════════════════════════════════════════════
function SensitivityTab() {
const sens = useMemo(() => gAeroSens(), []);
const top15 = sens.slice(0, 15);
const maxSens = Math.max(…top15.map(s => Math.abs(s.dCL)));
const mostSens = sens[0];

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Most Sensitive” value={mostSens.param} sub={`∂C_L = ${mostSens.dCL}`} sentiment=“neutral” delay={0} />
<KPI label="Max |∂C_L/∂p|" value={maxSens.toFixed(4)} sub="sensitivity magnitude" sentiment="neutral" delay={1} />
<KPI label=“Active Params” value={sens.filter(s => Math.abs(s.dCL) > 0.01).length} sub=“above threshold” sentiment=“positive” delay={2} />
<KPI label=“Setup Leverage” value={`${(maxSens * 28 * 100).toFixed(0)}%`} sub=“total C_L range” sentiment=“positive” delay={3} />
</div>

```
  <Sec title="∂C_L / ∂(setup param) — Tornado Chart (Top 15)">
    <GC><ResponsiveContainer width="100%" height={380}>
      <BarChart data={top15} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 90 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} vertical={true} horizontal={false} />
        <XAxis type="number" {...ax()} domain={[-maxSens * 1.1, maxSens * 1.1]} />
        <YAxis dataKey="param" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={85} />
        <Tooltip contentStyle={TT} />
        <ReferenceLine x={0} stroke={C.dm} />
        <Bar dataKey="dCL" barSize={12} radius={[4, 4, 4, 4]} name="∂C_L">
          {top15.map((e, i) => <Cell key={i} fill={e.dCL > 0 ? C.gn : C.red} fillOpacity={0.7} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="∂(L/D) / ∂(setup param)">
      <GC><ResponsiveContainer width="100%" height={300}>
        <BarChart data={top15} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 90 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
          <XAxis type="number" {...ax()} />
          <YAxis dataKey="param" type="category" tick={{ fontSize: 7, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} width={85} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine x={0} stroke={C.dm} />
          <Bar dataKey="dLD" barSize={10} radius={[4, 4, 4, 4]} name="∂(L/D)">
            {top15.map((e, i) => <Cell key={i} fill={e.dLD > 0 ? C.cy : C.am} fillOpacity={0.6} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="∂CoP_x / ∂(setup param)">
      <GC><ResponsiveContainer width="100%" height={300}>
        <BarChart data={top15} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 90 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
          <XAxis type="number" {...ax()} />
          <YAxis dataKey="param" type="category" tick={{ fontSize: 7, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} width={85} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine x={0} stroke={C.dm} />
          <Bar dataKey="dCoP" barSize={10} radius={[4, 4, 4, 4]} name="∂CoP_x">
            {top15.map((e, i) => <Cell key={i} fill={e.dCoP > 0 ? C.gn : AERO} fillOpacity={0.6} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 6: SURROGATE FIDELITY
// ═══════════════════════════════════════════════════════════════════════════
function FidelityTab() {
const data = useMemo(() => gFidelity(), []);
const residuals = data.map(d => d.residual);
const rmse = Math.sqrt(residuals.reduce((a, r) => a + r * r, 0) / residuals.length);
const r2 = 1 - residuals.reduce((a, r) => a + r * r, 0) / data.reduce((a, d) => a + Math.pow(d.actual - data.reduce((s, dd) => s + dd.actual, 0) / data.length, 2), 0);
const coverage = data.filter(d => d.inHull).length / data.length * 100;
const maxRes = Math.max(…data.map(d => d.absResidual));

// Residual histogram
const bins = 20;
const hist = Array.from({ length: bins }, (_, i) => {
const lo = -0.15 + (i / bins) * 0.30;
const hi = lo + 0.30 / bins;
return { bin: +((lo + hi) / 2).toFixed(3), count: data.filter(d => d.residual >= lo && d.residual < hi).length };
});

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label="R²" value={r2.toFixed(4)} sub="coefficient of determination" sentiment={r2 > 0.97 ? “positive” : r2 > 0.93 ? “amber” : “negative”} delay={0} />
<KPI label=“RMSE” value={rmse.toFixed(4)} sub=“root mean square error” sentiment={rmse < 0.05 ? “positive” : “amber”} delay={1} />
<KPI label=“Coverage” value={`${coverage.toFixed(1)}%`} sub=“training hull” sentiment={coverage > 85 ? “positive” : “amber”} delay={2} />
<KPI label=“Max |Residual|” value={maxRes.toFixed(4)} sub=“worst-case error” sentiment={maxRes < 0.1 ? “positive” : “amber”} delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Parity Plot — Surrogate vs CFD C_L">
      <GC><ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="actual" type="number" {...ax()} name="CFD C_L" label={{ value: "CFD (actual)", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis dataKey="pred" type="number" {...ax()} name="Surrogate C_L" label={{ value: "Surrogate (pred)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 7 }} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine segment={[{ x: 2.0, y: 2.0 }, { x: 3.5, y: 3.5 }]} stroke={C.gn} strokeDasharray="4 4" />
          <Scatter data={data.filter(d => d.inHull)} fill={AERO} fillOpacity={0.6} r={3} name="In Hull" />
          <Scatter data={data.filter(d => !d.inHull)} fill={C.red} fillOpacity={0.8} r={4} name="Extrapolation" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Residual Distribution">
      <GC><ResponsiveContainer width="100%" height={280}>
        <BarChart data={hist} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="bin" {...ax()} label={{ value: "Residual (pred−actual)", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine x={0} stroke={C.gn} strokeDasharray="4 4" />
          <Bar dataKey="count" barSize={12} radius={[4, 4, 0, 0]}>
            {hist.map((h, i) => <Cell key={i} fill={Math.abs(h.bin) < 0.04 ? C.gn : Math.abs(h.bin) < 0.08 ? C.am : C.red} fillOpacity={0.7} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <Sec title="Training Coverage — Pitch × Ride Height Density" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={220}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="pitch" type="number" {...ax()} name="Pitch [°]" label={{ value: "Pitch [°]", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
        <YAxis dataKey="rh" type="number" {...ax()} name="Ride Height [mm]" />
        <Tooltip contentStyle={TT} />
        <Scatter data={data.filter(d => d.inHull)} fill={C.gn} fillOpacity={0.4} r={4} name="In Hull" />
        <Scatter data={data.filter(d => !d.inHull)} fill={C.red} fillOpacity={0.8} r={5} name="Extrapolation Risk" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </ScatterChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 7: YAW & CROSSWIND
// ═══════════════════════════════════════════════════════════════════════════
function YawTab() {
const cw = useMemo(() => gCrosswind(), []);
const maxCY = Math.max(…cw.map(c => Math.abs(c.CY)));
const maxCN = Math.max(…cw.map(c => Math.abs(c.CN)));
// Stability derivatives at yaw=0
const atZero = cw.find(c => Math.abs(c.yaw) < 0.5) || cw[12];
const dCYdyaw = atZero.dCY_dyaw;
const dCNdyaw = atZero.dCN_dyaw;
const maxCLloss = Math.min(…cw.map(c => c.dCL));

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(5, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label="∂C_Y/∂β" value={dCYdyaw.toFixed(4)} sub="side force derivative" sentiment="neutral" delay={0} />
<KPI label="∂C_N/∂β" value={dCNdyaw.toFixed(5)} sub="yaw moment derivative" sentiment={dCNdyaw > 0 ? “positive” : “negative”} delay={1} />
<KPI label="Max |C_Y|" value={maxCY.toFixed(3)} sub="at ±10° yaw" sentiment="neutral" delay={2} />
<KPI label="Max |C_N|" value={maxCN.toFixed(4)} sub="yaw moment coefficient" sentiment="neutral" delay={3} />
<KPI label=“C_L Loss @10°” value={`${(maxCLloss * 100).toFixed(1)}%`} sub=“downforce degradation” sentiment={maxCLloss > -0.2 ? “positive” : “amber”} delay={4} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Side Force Coefficient C_Y vs Yaw Angle">
      <GC><ResponsiveContainer width="100%" height={240}>
        <LineChart data={cw} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="yaw" {...ax()} label={{ value: "Yaw [°]", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3" />
          <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
          <Line type="monotone" dataKey="CY" stroke={C.cy} strokeWidth={2} dot={{ r: 2, fill: C.cy }} name="C_Y" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Yaw Moment Coefficient C_N vs Yaw Angle">
      <GC><ResponsiveContainer width="100%" height={240}>
        <LineChart data={cw} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="yaw" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3" />
          <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
          <Line type="monotone" dataKey="CN" stroke={C.am} strokeWidth={2} dot={{ r: 2, fill: C.am }} name="C_N" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Downforce Loss in Crosswind (ΔC_L)">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={cw} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="yaw" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Area type="monotone" dataKey="dCL" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.5} dot={false} name="ΔC_L" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Drag Increase in Crosswind (ΔC_D)">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={cw} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="yaw" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Area type="monotone" dataKey="dCD" stroke={C.am} fill={`${C.am}10`} strokeWidth={1.5} dot={false} name="ΔC_D" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 8: AERO × GRIP COUPLING
// ═══════════════════════════════════════════════════════════════════════════
function CouplingTab() {
const ag = useMemo(() => gAeroGrip(), []);
const optCL = ag.reduce((best, c) => c.gripUtil > best.gripUtil ? c : best, ag[0]);
const dimReturns = ag.find(c => c.CL > 2.8 && c.gripUtil < ag.find(d => d.CL > 2.4)?.gripUtil);

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Optimal C_L” value={optCL.CL.toFixed(3)} sub={`max grip util: ${optCL.gripUtil}%`} sentiment=“positive” delay={0} />
<KPI label=“Grip Gain” value={`+${(optCL.gripUtil - ag[0].gripUtil).toFixed(1)}%`} sub=“from baseline to optimal” sentiment=“positive” delay={1} />
<KPI label=“Diminishing Returns” value={dimReturns ? `CL>${dimReturns.CL.toFixed(2)}` : “none”} sub=“grip starts declining” sentiment=“neutral” delay={2} />
<KPI label="Optimal L/D" value={optCL.LD} sub="at peak grip utility" sentiment="neutral" delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Grip Utilization [%] vs Downforce Coefficient">
      <GC><ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="CL" type="number" {...ax()} name="C_L" label={{ value: "C_L", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis dataKey="gripUtil" type="number" {...ax()} domain={[50, 100]} name="Grip Util %" />
          <Tooltip contentStyle={TT} />
          <Scatter data={ag} r={3} name="Grip vs CL">
            {ag.map((e, i) => <Cell key={i} fill={e.gripUtil > 90 ? C.gn : e.gripUtil > 80 ? C.am : C.red} fillOpacity={0.7} />)}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Tire μ_eff vs Normal Load (load sensitivity)">
      <GC><ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="Fz" type="number" {...ax()} name="Fz [N]" label={{ value: "Fz [N]", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis dataKey="mu" type="number" {...ax()} domain={[1.3, 1.7]} name="μ_eff" />
          <Tooltip contentStyle={TT} />
          <Scatter data={ag} fill={C.cy} fillOpacity={0.5} r={3} name="μ vs Fz" />
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Lap Time Sensitivity to C_L [s]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <LineChart data={ag} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="CL" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
          <Line type="monotone" dataKey="lapDelta" stroke={AERO} strokeWidth={2} dot={false} name="Δt_lap [s]" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="L/D Efficiency vs C_L">
      <GC><ResponsiveContainer width="100%" height={200}>
        <LineChart data={ag} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="CL" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="LD" stroke={C.gn} strokeWidth={2} dot={false} name="L/D" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function AerodynamicsModule() {
const [tab, setTab] = useState(“maps”);

return (
<div>
{/* Header banner */}
<div style={{
…GL, padding: “12px 16px”, marginBottom: 14,
borderLeft: `3px solid ${AERO}`,
background: `linear-gradient(90deg, ${AERO}08, transparent)`,
}}>
<div style={{ display: “flex”, alignItems: “center”, gap: 10 }}>
<span style={{ fontSize: 20, color: AERO }}>▽</span>
<div>
<span style={{ fontSize: 12, fontWeight: 800, color: AERO, fontFamily: C.dt, letterSpacing: 2 }}>
AERODYNAMICS
</span>
<span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 12 }}>
5D neural aero surrogate — differentiable C_L, C_D, CoP fields over full operating envelope
</span>
</div>
</div>
</div>

```
  <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
    {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={AERO} />)}
  </div>

  {tab === "maps" && <MapsTab />}
  {tab === "forces" && <ForcesTab />}
  {tab === "cop" && <CoPTab />}
  {tab === "platform" && <PlatformTab />}
  {tab === "sensitivity" && <SensitivityTab />}
  {tab === "fidelity" && <FidelityTab />}
  {tab === "yaw" && <YawTab />}
  {tab === "coupling" && <CouplingTab />}
</div>
```

);
}
