// ═══════════════════════════════════════════════════════════════════════════
// src/DriverCoachingModule.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════
// Driver-in-the-loop coaching system. Compares real driver inputs against
// the MPC-optimal trajectory to produce actionable feedback.
//
// v5.0 CHANGES:
//   - Expanded from 5 → 9 tabs
//   - NEW: g-g Diagram — friction circle utilization analysis
//   - NEW: Cornering — technique breakdown per corner type
//   - NEW: Scoring — per-lap pedal application quality metrics
//   - NEW: Report Card — session summary with letter grades
//
// Sub-tabs (9):
//   1. Driver Inputs   — Steering, throttle, brake overlaid vs optimal
//   2. Sectors         — Per-sector time analysis & breakdown table
//   3. Consistency     — Lap-to-lap variation, sigma metric
//   4. Braking         — Brake point accuracy, trail braking quality
//   5. g-g Diagram     — Friction circle utilization & dead zones
//   6. Cornering       — Entry/mid/exit speed analysis by corner type
//   7. Throttle/Brake  — Pedal application smoothness scoring
//   8. Racing Line     — Line deviation from optimal with distance overlay
//   9. Report Card     — Session summary with grades per skill
//
// Integration:
//   NAV: { key: “coaching”, label: “Coaching”, icon: “◈” }
//   Import: import DriverCoachingModule from “./DriverCoachingModule.jsx”
//   Route: case “coaching”: return <DriverCoachingModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from “react”;
import {
LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from “recharts”;
import { C, GL, GS, TT } from “./theme.js”;
import { KPI, Sec, GC, Pill } from “./components.jsx”;

// ═══════════════════════════════════════════════════════════════════════════
// SEEDED RNG
// ═══════════════════════════════════════════════════════════════════════════
function srng(seed) {
let s = seed;
return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; };
}
const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });

const TABS = [
{ key: “inputs”,     label: “Driver Inputs” },
{ key: “sectors”,    label: “Sectors” },
{ key: “consistency”,label: “Consistency” },
{ key: “braking”,    label: “Braking” },
{ key: “gg”,         label: “g-g Diagram” },
{ key: “cornering”,  label: “Cornering” },
{ key: “pedals”,     label: “Pedal Score” },
{ key: “line”,       label: “Racing Line” },
{ key: “report”,     label: “Report Card” },
];

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

function gDriverLap(n = 300) {
const R = srng(6001);
return Array.from({ length: n }, (_, i) => {
const s = i * (1370 / n);
const t = i * 0.05;
const curvature = 0.08 * Math.sin(i * 0.02) + 0.04 * Math.sin(i * 0.07);
const speed = 10 + 12 * Math.sin(i * 0.015) + 4 * Math.sin(i * 0.04) + R() * 1;
const optSteer = curvature * 100;
const actSteer = optSteer + (R() - 0.5) * 5 + 1.5 * Math.sin(i * 0.08);
const optThrottle = speed > 15 ? 0.7 + 0.2 * R() : 0.3 + 0.15 * R();
const actThrottle = Math.max(0, Math.min(1, optThrottle + (R() - 0.5) * 0.15));
const braking = Math.cos(i * 0.03) < -0.4;
const optBrake = braking ? 1500 + R() * 800 : 0;
const actBrake = braking ? optBrake * (0.85 + R() * 0.25) : (R() < 0.05 ? R() * 200 : 0);
const latG = speed * speed * curvature / 9.81;
const lonG = braking ? -1.5 + R() * 0.5 : 0.4 + R() * 0.3;
const lineDeviation = (R() - 0.5) * 0.4 + 0.1 * Math.sin(i * 0.05);
return {
s: +s.toFixed(0), t: +t.toFixed(2), speed: +speed.toFixed(1),
optSteer: +optSteer.toFixed(2), actSteer: +actSteer.toFixed(2),
steerError: +(actSteer - optSteer).toFixed(2),
optThrottle: +optThrottle.toFixed(3), actThrottle: +actThrottle.toFixed(3),
optBrake: +optBrake.toFixed(0), actBrake: +actBrake.toFixed(0),
latG: +latG.toFixed(3), lonG: +lonG.toFixed(3),
combinedG: +Math.sqrt(latG * latG + lonG * lonG).toFixed(3),
timeDelta: +((R() - 0.45) * 0.03).toFixed(4),
lineDeviation: +lineDeviation.toFixed(3), curvature: +curvature.toFixed(4),
};
});
}

function gSectorAnalysis(lap) {
const sectorDefs = [
{ sector: “S1”, sectorName: “Accel Zone”, start: 0, end: 170 },
{ sector: “S2”, sectorName: “Hairpin Entry”, start: 170, end: 340 },
{ sector: “S3”, sectorName: “Chicane”, start: 340, end: 520 },
{ sector: “S4”, sectorName: “Fast Sweep”, start: 520, end: 700 },
{ sector: “S5”, sectorName: “Slalom”, start: 700, end: 870 },
{ sector: “S6”, sectorName: “Final Corner”, start: 870, end: 1050 },
{ sector: “S7”, sectorName: “Accel Out”, start: 1050, end: 1200 },
{ sector: “S8”, sectorName: “Last Chicane”, start: 1200, end: 1370 },
];
const R = srng(6101);
return sectorDefs.map(sd => {
const pts = lap.filter(p => p.s >= sd.start && p.s < sd.end);
const optTime = pts.length * 0.05;
const actTime = optTime + (R() - 0.3) * 0.3;
const avgSpeed = pts.reduce((a, p) => a + p.speed, 0) / (pts.length || 1);
const peakAy = Math.max(…pts.map(p => Math.abs(p.latG)));
return {
…sd, optTime: +optTime.toFixed(2), actTime: +actTime.toFixed(2),
delta: +(actTime - optTime).toFixed(2),
avgSpeedAct: +avgSpeed.toFixed(1), peakAy: +peakAy.toFixed(3),
consistencyPct: +(90 + R() * 8).toFixed(1),
};
});
}

function gConsistency(nLaps = 12) {
const R = srng(6201);
return Array.from({ length: nLaps }, (_, i) => ({
lap: i + 1,
time: +(62.5 + R() * 1.2 + (i === 0 ? 0.8 : 0) + (i > 8 ? 0.3 : 0)).toFixed(2),
s1: +(7.8 + R() * 0.4).toFixed(2), s2: +(8.1 + R() * 0.5).toFixed(2),
s3: +(7.5 + R() * 0.3).toFixed(2), s4: +(8.8 + R() * 0.6).toFixed(2),
tireTemp: +(85 + i * 1.5 + R() * 3).toFixed(1),
fuelEffect: +(0 - i * 0.02).toFixed(2),
}));
}

function gBrakingPoints(n = 15) {
const R = srng(6301);
return Array.from({ length: n }, (_, i) => {
const optDist = 15 + R() * 10;
const actDist = optDist + (R() - 0.4) * 5;
const optPressure = 2000 + R() * 800;
const actPressure = optPressure * (0.82 + R() * 0.3);
const trailScore = 65 + R() * 30;
return {
zone: `BZ${i + 1}`, optDist: +optDist.toFixed(1), actDist: +actDist.toFixed(1),
distError: +(actDist - optDist).toFixed(1),
optPressure: +optPressure.toFixed(0), actPressure: +actPressure.toFixed(0),
pressureError: +((actPressure / optPressure - 1) * 100).toFixed(1),
trailScore: +trailScore.toFixed(0), decel: +(1.2 + R() * 1.0).toFixed(2),
speed: +(18 + R() * 10).toFixed(1),
};
});
}

function gGGDiagram(n = 500) {
const R = srng(6401);
return Array.from({ length: n }, (_, i) => {
const theta = R() * 2 * Math.PI;
const radius = (1.0 + R() * 0.4) * (0.7 + 0.3 * Math.abs(Math.sin(theta)));
const latG = radius * Math.cos(theta) + (R() - 0.5) * 0.15;
const lonG = radius * Math.sin(theta) * 0.85 + (R() - 0.5) * 0.1;
const utilized = Math.sqrt(latG * latG + lonG * lonG);
const maxAvailable = 1.55;
return {
latG: +latG.toFixed(3), lonG: +lonG.toFixed(3),
utilized: +utilized.toFixed(3),
utilPct: +(utilized / maxAvailable * 100).toFixed(1),
quadrant: lonG > 0 ? (latG > 0 ? “Accel+Right” : “Accel+Left”) : (latG > 0 ? “Brake+Right” : “Brake+Left”),
};
});
}

function gCorneringData() {
const R = srng(6501);
const types = [“Hairpin”, “Medium”, “Fast Sweep”, “Chicane L”, “Chicane R”, “Slalom”];
return types.map(type => {
const entrySpeedOpt = type === “Hairpin” ? 8 : type.includes(“Fast”) ? 18 : 12 + R() * 4;
const midSpeedOpt = entrySpeedOpt * (type === “Hairpin” ? 0.7 : 0.85);
const exitSpeedOpt = entrySpeedOpt * (type === “Hairpin” ? 1.1 : 1.05);
return {
type,
entrySpeedOpt: +entrySpeedOpt.toFixed(1),
entrySpeedAct: +(entrySpeedOpt + (R() - 0.4) * 2).toFixed(1),
midSpeedOpt: +midSpeedOpt.toFixed(1),
midSpeedAct: +(midSpeedOpt + (R() - 0.5) * 1.5).toFixed(1),
exitSpeedOpt: +exitSpeedOpt.toFixed(1),
exitSpeedAct: +(exitSpeedOpt + (R() - 0.45) * 2).toFixed(1),
apexHitPct: +(75 + R() * 22).toFixed(0),
trailBrakePct: +(60 + R() * 35).toFixed(0),
timeLost: +(R() * 0.15).toFixed(3),
understeerEvents: Math.floor(R() * 3),
peakLatG: +(1.1 + R() * 0.4).toFixed(2),
};
});
}

function gPedalScoring(nLaps = 10) {
const R = srng(6601);
return Array.from({ length: nLaps }, (_, i) => ({
lap: i + 1,
throttleSmoothness: +(70 + R() * 25 + (i > 2 ? 5 : 0)).toFixed(0),
brakeModulation: +(65 + R() * 30).toFixed(0),
trailBraking: +(55 + R() * 35 + (i > 4 ? 8 : 0)).toFixed(0),
steeringSmooth: +(72 + R() * 23).toFixed(0),
liftAndCoast: +(80 + R() * 15).toFixed(0),
overall: 0,
})).map(l => ({ …l, overall: +((+l.throttleSmoothness + +l.brakeModulation + +l.trailBraking + +l.steeringSmooth + +l.liftAndCoast) / 5).toFixed(0) }));
}

function gLineDeviation(n = 200) {
const R = srng(6701);
return Array.from({ length: n }, (_, i) => {
const s = i * (1370 / n);
const optimal = 0;
const actual = (R() - 0.5) * 0.6 + 0.15 * Math.sin(i * 0.04);
const width = 1.5 + 0.5 * Math.sin(i * 0.01);
return {
s: +s.toFixed(0), deviation: +actual.toFixed(3),
trackWidthL: +width.toFixed(2), trackWidthR: +(-width).toFixed(2),
absDeviation: +Math.abs(actual).toFixed(3),
zone: Math.abs(actual) < 0.15 ? “optimal” : Math.abs(actual) < 0.35 ? “acceptable” : “off-line”,
};
});
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 1: DRIVER INPUTS
// ═══════════════════════════════════════════════════════════════════════════
function InputsTab({ lap }) {
const sparse = useMemo(() => lap.filter((_, i) => i % 2 === 0), [lap]);
return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “1fr 1fr 1fr”, gap: 10 }}>
<Sec title="Steering — Actual vs Optimal [°]">
<GC><ResponsiveContainer width="100%" height={200}>
<LineChart data={sparse} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} />
<XAxis dataKey=“s” {…ax()} />
<YAxis {…ax()} />
<Tooltip contentStyle={TT} />
<Line type="monotone" dataKey="optSteer" stroke={C.gn} strokeWidth={2} dot={false} name="MPC Optimal" />
<Line type="monotone" dataKey="actSteer" stroke={C.cy} strokeWidth={1.5} dot={false} name="Driver" strokeDasharray="4 2" />
<Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
</LineChart>
</ResponsiveContainer></GC>
</Sec>

```
    <Sec title="Throttle Application [0–1]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={sparse} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="s" {...ax()} />
          <YAxis {...ax()} domain={[0, 1]} />
          <Tooltip contentStyle={TT} />
          <Area type="monotone" dataKey="optThrottle" stroke={C.gn} fill={`${C.gn}08`} strokeWidth={1.5} dot={false} name="Optimal" />
          <Area type="monotone" dataKey="actThrottle" stroke={C.cy} fill={`${C.cy}06`} strokeWidth={1.2} dot={false} name="Driver" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Brake Pressure [N]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={sparse} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="s" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Area type="monotone" dataKey="optBrake" stroke={C.gn} fill={`${C.gn}08`} strokeWidth={1.5} dot={false} name="Optimal" />
          <Area type="monotone" dataKey="actBrake" stroke={C.red} fill={`${C.red}06`} strokeWidth={1.2} dot={false} name="Driver" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <Sec title="Steering Error [°]" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={160}>
      <AreaChart data={sparse} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="s" {...ax()} />
        <YAxis {...ax()} />
        <Tooltip contentStyle={TT} />
        <ReferenceLine y={0} stroke={C.dm} />
        <ReferenceArea y1={-3} y2={3} fill={C.gn} fillOpacity={0.03} />
        <Area type="monotone" dataKey="steerError" stroke={C.am} fill={`${C.am}08`} strokeWidth={1.2} dot={false} name="Steer Error" />
      </AreaChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 2: SECTORS
// ═══════════════════════════════════════════════════════════════════════════
function SectorsTab({ sectors }) {
return (
<div>
<Sec title="Sector Time Delta [s] — Actual vs Optimal">
<GC><ResponsiveContainer width="100%" height={220}>
<BarChart data={sectors} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
<CartesianGrid strokeDasharray="3 3" stroke={GS} />
<XAxis dataKey=“sector” {…ax()} />
<YAxis {…ax()} />
<Tooltip contentStyle={TT} />
<ReferenceLine y={0} stroke={C.dm} />
<Bar dataKey=“delta” barSize={20} radius={[4, 4, 0, 0]} name=“Δ Time [s]”>
{sectors.map((s, i) => <Cell key={i} fill={+s.delta <= 0 ? C.gn : C.red} fillOpacity={0.7} />)}
</Bar>
</BarChart>
</ResponsiveContainer></GC>
</Sec>

```
  <Sec title="Sector Breakdown" style={{ marginTop: 10 }}>
    <GC style={{ padding: 10 }}>
      <div style={{ display: "grid", gridTemplateColumns: "60px 100px 65px 65px 55px 65px 65px 65px", gap: 0, fontSize: 8, fontFamily: C.dt }}>
        {["Sector", "Name", "Opt [s]", "Act [s]", "Δ", "Avg Spd", "Peak Ay", "Consist"].map(h => (
          <div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 1, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>
        ))}
        {sectors.map(s => (
          <React.Fragment key={s.sector}>
            <div style={{ color: C.cy, fontWeight: 700, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.sector}</div>
            <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.sectorName}</div>
            <div style={{ color: C.dm, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.optTime}</div>
            <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.actTime}</div>
            <div style={{ color: +s.delta <= 0 ? C.gn : C.red, fontWeight: 700, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{+s.delta > 0 ? "+" : ""}{s.delta}</div>
            <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.avgSpeedAct} m/s</div>
            <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.peakAy}G</div>
            <div style={{ color: +s.consistencyPct > 93 ? C.gn : C.am, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.consistencyPct}%</div>
          </React.Fragment>
        ))}
      </div>
    </GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 3: CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════
function ConsistencyTab({ laps }) {
const bestLap = Math.min(…laps.map(l => +l.time));
const avgLap = laps.reduce((a, l) => a + +l.time, 0) / laps.length;
const stdDev = Math.sqrt(laps.reduce((a, l) => a + (+l.time - avgLap) ** 2, 0) / laps.length);

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Best Lap” value={`${bestLap}s`} sub=“session best” sentiment=“positive” delay={0} />
<KPI label=“Average” value={`${avgLap.toFixed(2)}s`} sub={`Δ +${(avgLap - bestLap).toFixed(2)}s`} sentiment=“neutral” delay={1} />
<KPI label=“Std Dev” value={`${stdDev.toFixed(3)}s`} sub={stdDev < 0.3 ? “excellent” : “needs work”} sentiment={stdDev < 0.3 ? “positive” : “amber”} delay={2} />
<KPI label=“Consistency” value={`${(100 - (stdDev / avgLap) * 100).toFixed(1)}%`} sub=“σ/μ metric” sentiment=“positive” delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Lap Times">
      <GC><ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={laps} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={[bestLap - 0.5, bestLap + 2.5]} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={bestLap} stroke={C.gn} strokeDasharray="4 2" label={{ value: "Best", fill: C.gn, fontSize: 7 }} />
          <ReferenceLine y={avgLap} stroke={C.am} strokeDasharray="4 2" label={{ value: "Avg", fill: C.am, fontSize: 7 }} />
          <Bar dataKey="time" radius={[4, 4, 0, 0]} barSize={24}>
            {laps.map((l, i) => <Cell key={i} fill={+l.time <= bestLap + 0.3 ? C.gn : +l.time <= avgLap ? C.cy : C.am} fillOpacity={0.7} />)}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Tire Temp & Fuel Effect">
      <GC><ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={laps} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis yAxisId="t" {...ax()} />
          <YAxis yAxisId="f" orientation="right" {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line yAxisId="t" dataKey="tireTemp" stroke={C.am} strokeWidth={1.5} dot={false} name="Tire Temp °C" />
          <Line yAxisId="f" dataKey="fuelEffect" stroke={C.gn} strokeWidth={1.5} dot={false} name="Fuel Effect [s]" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 4: BRAKING
// ═══════════════════════════════════════════════════════════════════════════
function BrakingTab({ brakingPts }) {
const avgDistErr = brakingPts.reduce((a, b) => a + Math.abs(+b.distError), 0) / brakingPts.length;
const avgTrail = brakingPts.reduce((a, b) => a + +b.trailScore, 0) / brakingPts.length;

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Avg Brake Pt Error” value={`${avgDistErr.toFixed(1)}m`} sub=“distance from optimal” sentiment={avgDistErr < 2 ? “positive” : “amber”} delay={0} />
<KPI label=“Trail Brake Score” value={`${avgTrail.toFixed(0)}/100`} sub=“modulation quality” sentiment={avgTrail > 80 ? “positive” : “amber”} delay={1} />
<KPI label=“Late Brakers” value={brakingPts.filter(b => +b.distError > 2).length.toString()} sub=“of 15 zones” sentiment=“neutral” delay={2} />
<KPI label=“Avg Decel” value={`${(brakingPts.reduce((a, b) => a + +b.decel, 0) / brakingPts.length).toFixed(2)}G`} sub=“braking force” sentiment=“neutral” delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Brake Point Accuracy [m error]">
      <GC><ResponsiveContainer width="100%" height={240}>
        <BarChart data={brakingPts} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="zone" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0} stroke={C.dm} />
          <Bar dataKey="distError" barSize={16} radius={[4, 4, 4, 4]} name="Distance Error [m]">
            {brakingPts.map((b, i) => <Cell key={i} fill={Math.abs(+b.distError) < 2 ? C.gn : +b.distError > 0 ? C.am : C.cy} fillOpacity={0.7} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Trail Braking Score [/100]">
      <GC><ResponsiveContainer width="100%" height={240}>
        <BarChart data={brakingPts} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="zone" {...ax()} />
          <YAxis {...ax()} domain={[0, 100]} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={80} stroke={C.gn} strokeDasharray="4 2" />
          <Bar dataKey="trailScore" barSize={16} radius={[4, 4, 0, 0]} name="Trail Score">
            {brakingPts.map((b, i) => <Cell key={i} fill={+b.trailScore > 80 ? C.gn : +b.trailScore > 60 ? C.am : C.red} fillOpacity={0.7} />)}
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
// TAB 5: g-g DIAGRAM — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function GGTab() {
const gg = useMemo(() => gGGDiagram(), []);
const avgUtil = gg.reduce((a, g) => a + +g.utilPct, 0) / gg.length;
const highUtil = gg.filter(g => +g.utilPct > 85).length / gg.length * 100;
const deadZone = gg.filter(g => +g.utilPct < 30).length / gg.length * 100;
const quadrants = [“Accel+Right”, “Accel+Left”, “Brake+Right”, “Brake+Left”];
const qData = quadrants.map(q => ({ q, count: gg.filter(g => g.quadrant === q).length, avgUtil: +(gg.filter(g => g.quadrant === q).reduce((a, g) => a + +g.utilPct, 0) / (gg.filter(g => g.quadrant === q).length || 1)).toFixed(1) }));

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Avg Utilization” value={`${avgUtil.toFixed(1)}%`} sub=“of friction circle” sentiment={avgUtil > 70 ? “positive” : “amber”} delay={0} />
<KPI label=“High Util (>85%)” value={`${highUtil.toFixed(0)}%`} sub=“of data points” sentiment={highUtil > 30 ? “positive” : “amber”} delay={1} />
<KPI label=“Dead Zone (<30%)” value={`${deadZone.toFixed(0)}%`} sub=“wasted capacity” sentiment={deadZone < 15 ? “positive” : “amber”} delay={2} />
<KPI label=“Max Combined G” value={`${Math.max(...gg.map(g => +g.utilized)).toFixed(2)}G`} sub=“peak grip use” sentiment=“positive” delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 10 }}>
    <Sec title="g-g Diagram — Friction Circle Utilization">
      <GC><ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="latG" type="number" {...ax()} domain={[-1.8, 1.8]} name="Lat G" label={{ value: "Lateral G", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis dataKey="lonG" type="number" {...ax()} domain={[-2, 1.5]} name="Lon G" label={{ value: "Longitudinal G", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 7 }} />
          <Tooltip contentStyle={TT} />
          <Scatter data={gg} r={2}>
            {gg.map((g, i) => <Cell key={i} fill={+g.utilPct > 85 ? C.gn : +g.utilPct > 60 ? C.cy : +g.utilPct > 30 ? C.am : C.dm} fillOpacity={0.5} />)}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Quadrant Analysis">
      <GC><ResponsiveContainer width="100%" height={340}>
        <BarChart data={qData} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
          <XAxis type="number" {...ax()} domain={[0, 100]} />
          <YAxis dataKey="q" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={75} />
          <Tooltip contentStyle={TT} />
          <Bar dataKey="avgUtil" barSize={16} radius={[0, 4, 4, 0]} name="Avg Util %">
            {qData.map((q, i) => <Cell key={i} fill={+q.avgUtil > 70 ? C.gn : C.am} fillOpacity={0.7} />)}
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
// TAB 6: CORNERING TECHNIQUE — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function CorneringTab() {
const corners = useMemo(() => gCorneringData(), []);

return (
<div>
<Sec title="Cornering Technique — Entry / Mid / Exit Speed Analysis">
<GC style={{ padding: 10 }}>
<div style={{ display: “grid”, gridTemplateColumns: “90px 70px 70px 70px 70px 70px 70px 55px 55px 55px”, gap: 0, fontSize: 8, fontFamily: C.dt }}>
{[“Type”, “Entry Opt”, “Entry Act”, “Mid Opt”, “Mid Act”, “Exit Opt”, “Exit Act”, “Apex %”, “Trail %”, “Δt [s]”].map(h => (
<div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 0.5, padding: “6px 3px”, borderBottom: `1px solid ${C.b1}` }}>{h}</div>
))}
{corners.map(c => (
<React.Fragment key={c.type}>
<div style={{ color: C.cy, fontWeight: 600, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.type}</div>
<div style={{ color: C.gn, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.entrySpeedOpt}</div>
<div style={{ color: +c.entrySpeedAct > +c.entrySpeedOpt + 1 ? C.am : C.br, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.entrySpeedAct}</div>
<div style={{ color: C.gn, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.midSpeedOpt}</div>
<div style={{ color: C.br, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.midSpeedAct}</div>
<div style={{ color: C.gn, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.exitSpeedOpt}</div>
<div style={{ color: C.br, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.exitSpeedAct}</div>
<div style={{ color: +c.apexHitPct > 85 ? C.gn : C.am, fontWeight: 600, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.apexHitPct}%</div>
<div style={{ color: +c.trailBrakePct > 75 ? C.gn : C.am, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>{c.trailBrakePct}%</div>
<div style={{ color: C.red, padding: “5px 3px”, borderBottom: `1px solid ${C.b1}08` }}>+{c.timeLost}</div>
</React.Fragment>
))}
</div>
</GC>
</Sec>

```
  <Sec title="Speed Through Corners — Optimal vs Actual" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={240}>
      <BarChart data={corners} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="type" {...ax()} />
        <YAxis {...ax()} />
        <Tooltip contentStyle={TT} />
        <Bar dataKey="entrySpeedOpt" fill={C.gn} fillOpacity={0.4} barSize={8} name="Entry Opt" />
        <Bar dataKey="entrySpeedAct" fill={C.cy} fillOpacity={0.7} barSize={8} name="Entry Act" />
        <Bar dataKey="exitSpeedOpt" fill={C.gn} fillOpacity={0.4} barSize={8} name="Exit Opt" />
        <Bar dataKey="exitSpeedAct" fill={C.am} fillOpacity={0.7} barSize={8} name="Exit Act" />
        <Legend wrapperStyle={{ fontSize: 7, fontFamily: C.hd }} />
      </BarChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 7: PEDAL SCORING — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function PedalsTab() {
const scores = useMemo(() => gPedalScoring(), []);
const avgOverall = scores.reduce((a, s) => a + +s.overall, 0) / scores.length;

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(5, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Overall” value={`${avgOverall.toFixed(0)}/100`} sub=“average score” sentiment={avgOverall > 80 ? “positive” : avgOverall > 65 ? “amber” : “negative”} delay={0} />
<KPI label=“Throttle” value={`${(scores.reduce((a, s) => a + +s.throttleSmoothness, 0) / scores.length).toFixed(0)}`} sub=“smoothness” sentiment=“neutral” delay={1} />
<KPI label=“Brake Mod” value={`${(scores.reduce((a, s) => a + +s.brakeModulation, 0) / scores.length).toFixed(0)}`} sub=“modulation” sentiment=“neutral” delay={2} />
<KPI label=“Trail Brake” value={`${(scores.reduce((a, s) => a + +s.trailBraking, 0) / scores.length).toFixed(0)}`} sub=“technique” sentiment=“neutral” delay={3} />
<KPI label=“Trend” value={+scores[scores.length - 1].overall > +scores[0].overall ? “IMPROVING” : “DECLINING”} sub=“first → last” sentiment={+scores[scores.length - 1].overall > +scores[0].overall ? “positive” : “amber”} delay={4} />
</div>

```
  <Sec title="Pedal Skill Scores Per Lap [/100]">
    <GC><ResponsiveContainer width="100%" height={280}>
      <LineChart data={scores} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="lap" {...ax()} />
        <YAxis {...ax()} domain={[40, 100]} />
        <Tooltip contentStyle={TT} />
        <Line type="monotone" dataKey="throttleSmoothness" stroke={C.gn} strokeWidth={1.5} dot={{ r: 2 }} name="Throttle" />
        <Line type="monotone" dataKey="brakeModulation" stroke={C.red} strokeWidth={1.5} dot={{ r: 2 }} name="Brake" />
        <Line type="monotone" dataKey="trailBraking" stroke={C.am} strokeWidth={1.5} dot={{ r: 2 }} name="Trail" />
        <Line type="monotone" dataKey="steeringSmooth" stroke={C.cy} strokeWidth={1.5} dot={{ r: 2 }} name="Steering" />
        <Line type="monotone" dataKey="overall" stroke={C.w} strokeWidth={2.5} dot={false} name="Overall" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </LineChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 8: RACING LINE — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function LineTab() {
const lineData = useMemo(() => gLineDeviation(), []);
const avgDev = lineData.reduce((a, l) => a + +l.absDeviation, 0) / lineData.length;
const offLine = lineData.filter(l => l.zone === “off-line”).length / lineData.length * 100;

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(3, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Avg Deviation” value={`${(avgDev * 100).toFixed(0)} cm`} sub=“from optimal line” sentiment={avgDev < 0.2 ? “positive” : “amber”} delay={0} />
<KPI label=“Off-Line %” value={`${offLine.toFixed(0)}%`} sub=”>35cm deviation” sentiment={offLine < 10 ? “positive” : “amber”} delay={1} />
<KPI label=“Max Deviation” value={`${(Math.max(...lineData.map(l => +l.absDeviation)) * 100).toFixed(0)} cm`} sub=“worst point” sentiment=“neutral” delay={2} />
</div>

```
  <Sec title="Racing Line Deviation [m] — with Track Width">
    <GC><ResponsiveContainer width="100%" height={260}>
      <AreaChart data={lineData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="s" {...ax()} />
        <YAxis {...ax()} domain={[-2, 2]} />
        <Tooltip contentStyle={TT} />
        <Area type="monotone" dataKey="trackWidthL" stroke={C.dm} fill="none" strokeWidth={1} strokeDasharray="4 2" name="Track L" />
        <Area type="monotone" dataKey="trackWidthR" stroke={C.dm} fill="none" strokeWidth={1} strokeDasharray="4 2" name="Track R" />
        <ReferenceLine y={0} stroke={C.gn} strokeDasharray="4 4" />
        <Area type="monotone" dataKey="deviation" stroke={C.cy} fill={`${C.cy}10`} strokeWidth={2} dot={false} name="Line Deviation" />
      </AreaChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 9: SESSION REPORT CARD — NEW v5.0
// ═══════════════════════════════════════════════════════════════════════════
function ReportTab({ lap, sectors, brakingPts, laps }) {
const steerRMS = Math.sqrt(lap.reduce((a, d) => a + (+d.steerError) ** 2, 0) / lap.length);
const avgTrail = brakingPts.reduce((a, b) => a + +b.trailScore, 0) / brakingPts.length;
const bestLap = Math.min(…laps.map(l => +l.time));
const stdDev = Math.sqrt(laps.reduce((a, l) => a + (+l.time - laps.reduce((aa, ll) => aa + +ll.time, 0) / laps.length) ** 2, 0) / laps.length);

const grade = (v, thresholds) => {
if (v >= thresholds[0]) return { letter: “A”, color: C.gn };
if (v >= thresholds[1]) return { letter: “B”, color: C.cy };
if (v >= thresholds[2]) return { letter: “C”, color: C.am };
return { letter: “D”, color: C.red };
};

const skills = [
{ skill: “Steering Precision”, value: Math.max(0, 100 - steerRMS * 10), …grade(Math.max(0, 100 - steerRMS * 10), [85, 70, 55]), advice: steerRMS < 3 ? “Excellent — minimal correction” : “Reduce steering corrections; aim for smoother input” },
{ skill: “Braking Accuracy”, value: Math.max(0, 100 - brakingPts.reduce((a, b) => a + Math.abs(+b.distError), 0) / brakingPts.length * 15), …grade(Math.max(0, 100 - brakingPts.reduce((a, b) => a + Math.abs(+b.distError), 0) / brakingPts.length * 15), [80, 65, 50]), advice: “Work on hitting brake markers consistently” },
{ skill: “Trail Braking”, value: avgTrail, …grade(avgTrail, [85, 70, 55]), advice: avgTrail > 80 ? “Strong trail braking technique” : “Practice releasing brake pressure gradually into the corner” },
{ skill: “Consistency”, value: Math.max(0, 100 - stdDev * 100), …grade(Math.max(0, 100 - stdDev * 100), [85, 70, 55]), advice: stdDev < 0.3 ? “Very consistent driver” : “Focus on repeating the same line and inputs each lap” },
{ skill: “Smoothness”, value: 78, …grade(78, [85, 70, 55]), advice: “Good overall smoothness — continue practicing weight transfer management” },
{ skill: “Grip Utilization”, value: 72, …grade(72, [85, 70, 55]), advice: “Explore the friction circle more in braking+cornering transitions” },
];

const overallScore = skills.reduce((a, s) => a + s.value, 0) / skills.length;
const overallGrade = grade(overallScore, [85, 70, 55]);

const radarData = skills.map(s => ({ skill: s.skill.split(” “)[0], score: +s.value.toFixed(0) }));

return (
<div>
<div style={{ display: “grid”, gridTemplateColumns: “repeat(4, 1fr)”, gap: 10, marginBottom: 14 }}>
<KPI label=“Overall Grade” value={overallGrade.letter} sub={`${overallScore.toFixed(0)}/100`} sentiment={overallGrade.letter === “A” ? “positive” : overallGrade.letter === “B” ? “positive” : “amber”} delay={0} />
<KPI label=“Best Lap” value={`${bestLap}s`} sub=“session fastest” sentiment=“positive” delay={1} />
<KPI label=“Weakest Skill” value={skills.sort((a, b) => a.value - b.value)[0].skill} sub={`${skills.sort((a, b) => a.value - b.value)[0].value.toFixed(0)}/100`} sentiment=“amber” delay={2} />
<KPI label=“Strongest” value={skills.sort((a, b) => b.value - a.value)[0].skill} sub={`${skills.sort((a, b) => b.value - a.value)[0].value.toFixed(0)}/100`} sentiment=“positive” delay={3} />
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Skill Radar">
      <GC><ResponsiveContainer width="100%" height={300}>
        <RadarChart data={radarData} outerRadius={100}>
          <PolarGrid stroke={GS} />
          <PolarAngleAxis dataKey="skill" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 7, fill: C.dm }} />
          <Radar dataKey="score" stroke={C.cy} fill={C.cy} fillOpacity={0.15} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Skill Breakdown">
      <GC style={{ padding: 10 }}>
        {skills.map(s => (
          <div key={s.skill} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: `1px solid ${C.b1}08` }}>
            <div style={{ width: 36, height: 36, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", background: `${s.color}15`, border: `2px solid ${s.color}`, fontSize: 16, fontWeight: 800, color: s.color, fontFamily: C.dt, flexShrink: 0 }}>
              {s.letter}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: C.br, fontFamily: C.dt }}>{s.skill}</div>
              <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, marginTop: 2 }}>{s.advice}</div>
            </div>
            <div style={{ fontSize: 14, fontWeight: 800, color: s.color, fontFamily: C.dt }}>{s.value.toFixed(0)}</div>
          </div>
        ))}
      </GC>
    </Sec>
  </div>
</div>
```

);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function DriverCoachingModule() {
const [tab, setTab] = useState(“inputs”);

const lap = useMemo(() => gDriverLap(), []);
const sectors = useMemo(() => gSectorAnalysis(lap), [lap]);
const laps = useMemo(() => gConsistency(), []);
const brakingPts = useMemo(() => gBrakingPoints(), []);

const totalDelta = lap.reduce((a, d) => a + +d.timeDelta, 0);
const steerRMS = Math.sqrt(lap.reduce((a, d) => a + (+d.steerError) ** 2, 0) / lap.length);

return (
<div>
{/* Header banner */}
<div style={{
…GL, padding: “12px 16px”, marginBottom: 14,
borderLeft: `3px solid ${C.cy}`,
background: `linear-gradient(90deg, ${C.cy}08, transparent)`,
}}>
<div style={{ display: “flex”, alignItems: “center”, gap: 10 }}>
<span style={{ fontSize: 20, color: C.cy }}>◈</span>
<div>
<span style={{ fontSize: 12, fontWeight: 800, color: C.cy, fontFamily: C.dt, letterSpacing: 2 }}>DRIVER COACHING</span>
<span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 12 }}>
MPC-optimal vs actual — actionable feedback for driver development
</span>
</div>
</div>
</div>

```
  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
    <KPI label="Time Delta" value={`${totalDelta > 0 ? "+" : ""}${(totalDelta * 10).toFixed(2)}s`} sub="vs MPC optimal" sentiment={totalDelta <= 0 ? "positive" : "negative"} delay={0} />
    <KPI label="Steer RMS" value={`${steerRMS.toFixed(1)}°`} sub="error vs optimal" sentiment={steerRMS < 5 ? "positive" : "amber"} delay={1} />
    <KPI label="Best Sector" value={sectors.reduce((b, s) => +s.delta < +b.delta ? s : b, sectors[0]).sector} sub="closest to MPC" sentiment="positive" delay={2} />
    <KPI label="Worst Sector" value={sectors.reduce((w, s) => +s.delta > +w.delta ? s : w, sectors[0]).sector} sub="most time lost" sentiment="negative" delay={3} />
    <KPI label="Reaction Lag" value="~60 ms" sub="estimated input delay" sentiment="amber" delay={4} />
  </div>

  <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
    {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.cy} />)}
  </div>

  {tab === "inputs" && <InputsTab lap={lap} />}
  {tab === "sectors" && <SectorsTab sectors={sectors} />}
  {tab === "consistency" && <ConsistencyTab laps={laps} />}
  {tab === "braking" && <BrakingTab brakingPts={brakingPts} />}
  {tab === "gg" && <GGTab />}
  {tab === "cornering" && <CorneringTab />}
  {tab === "pedals" && <PedalsTab />}
  {tab === "line" && <LineTab />}
  {tab === "report" && <ReportTab lap={lap} sectors={sectors} brakingPts={brakingPts} laps={laps} />}
</div>
```

);
}