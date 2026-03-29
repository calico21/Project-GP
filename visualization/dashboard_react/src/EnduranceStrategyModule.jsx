// ═══════════════════════════════════════════════════════════════════════════
// src/EnduranceStrategyModule.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════
// Endurance event race strategy module. FSE = 22 km, ~16 laps of ~1.37 km.
// Answers: "Will we finish? At what pace? What are the risks?"
//
// v5.0 CHANGES:
//   - Battery/motor/inverter deep-dives → cross-link to Electronics module
//   - Energy tab retained as race-strategy summary (not component diagnostic)
//   - NEW: Monte Carlo race simulation (probabilistic finish prediction)
//   - NEW: Stint planner (lap-by-lap target pace with thermal/energy gates)
//   - NEW: Pace optimizer (optimal speed profile balancing time vs limits)
//   - Brake thermal & tire degradation retained (mechanical, not electronics)
//   - Regen tab enhanced with efficiency map overlay
//
// Sub-tabs (8):
//   1. Race Overview  — GO/CAUTION banner, all subsystem summaries, finish prob
//   2. Energy Budget  — SoC projection, energy per lap, regen balance
//   3. Stint Planner  — Lap-by-lap pace targets, energy/thermal gates
//   4. Monte Carlo    — 500-run probabilistic finish simulation
//   5. Brake Thermal  — 4-corner rotor temps, fade risk analysis
//   6. Tire Deg       — Grip evolution, lap time impact, wear prediction
//   7. Pace Strategy  — Optimal speed vs thermal headroom trade-off
//   8. Conditions     — Ambient temp, track evolution, density altitude effects
//
// Integration:
//   NAV: { key: "endurance", label: "Endurance", icon: "⏱" }
//   Import: import EnduranceStrategyModule from "./EnduranceStrategyModule.jsx"
//   Route: case "endurance": return <EnduranceStrategyModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
LineChart, Line, AreaChart, Area, ComposedChart, BarChart, Bar,
ScatterChart, Scatter,
XAxis, YAxis, CartesianGrid, Tooltip,
ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// SEEDED RNG & CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
function srng(seed) {
let s = seed;
return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; };
}
const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });
const CL = { fl: C.cy, fr: C.gn, rl: C.am, rr: C.red };
const ELEC = "#7c3aed";

const TABS = [
{ key: "overview",  label: "Race Overview" },
{ key: "energy",    label: "Energy Budget" },
{ key: "stint",     label: "Stint Planner" },
{ key: "montecarlo",label: "Monte Carlo" },
{ key: "brakes",    label: "Brake Thermal" },
{ key: "tires",     label: "Tire Deg" },
{ key: "pace",      label: "Pace Strategy" },
{ key: "conditions",label: "Conditions" },
];

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

function gBatteryTrace(nLaps = 16) {
const R = srng(201);
const data = [];
let soc = 100, energy = 6.5;
for (let lap = 0; lap <= nLaps; lap++) {
const lapEnergy = 0.35 + R() * 0.08;
const regenRecov = 0.04 + R() * 0.02;
energy = Math.max(0, energy - lapEnergy + regenRecov);
soc = (energy / 6.5) * 100;
const voltage = 540 + soc * 0.48 - (lap > 10 ? (lap - 10) * 3 : 0);
const cellTemp = 32 + lap * 2.1 + R() * 3;
data.push({
lap, soc: +soc.toFixed(1), voltage: +voltage.toFixed(0),
energy: +energy.toFixed(2), cellTemp: +cellTemp.toFixed(1),
lapEnergy: +lapEnergy.toFixed(3), regenRecov: +regenRecov.toFixed(3),
netEnergy: +(lapEnergy - regenRecov).toFixed(3),
});
}
return data;
}

function gBrakeThermal(nLaps = 16) {
const R = srng(401);
const data = [];
let temps = [180, 185, 160, 165];
for (let lap = 0; lap <= nLaps; lap++) {
const brakeLoad = 0.6 + 0.4 * Math.sin(lap * 0.7);
for (let c = 0; c < 4; c++) {
const heat = (c < 2 ? 22 : 14) * brakeLoad + R() * 8;
const cool = (temps[c] - 30) * 0.08;
temps[c] = Math.max(30, Math.min(500, temps[c] + heat - cool));
}
data.push({
lap, FL: +temps[0].toFixed(0), FR: +temps[1].toFixed(0),
RL: +temps[2].toFixed(0), RR: +temps[3].toFixed(0),
maxTemp: +Math.max(…temps).toFixed(0),
avgTemp: +(temps.reduce((a, t) => a + t, 0) / 4).toFixed(0),
});
}
return data;
}

function gTireDeg(nLaps = 16) {
const R = srng(501);
const data = [];
let grip = [100, 100, 100, 100];
for (let lap = 0; lap <= nLaps; lap++) {
for (let c = 0; c < 4; c++) {
const wear = (c < 2 ? 0.55 : 0.45) + R() * 0.15;
grip[c] = Math.max(75, grip[c] - wear);
}
const avgGrip = grip.reduce((a, g) => a + g, 0) / 4;
const lapTime = 62.5 + (100 - avgGrip) * 0.15 + R() * 0.3;
data.push({
lap, FL: +grip[0].toFixed(1), FR: +grip[1].toFixed(1),
RL: +grip[2].toFixed(1), RR: +grip[3].toFixed(1),
avgGrip: +avgGrip.toFixed(1), lapTime: +lapTime.toFixed(2),
wearRate: +(lap > 0 ? (data[lap - 1]?.avgGrip || 100) - avgGrip : 0).toFixed(2),
});
}
return data;
}

function gStintPlan(nLaps = 16) {
const R = srng(701);
return Array.from({ length: nLaps }, (_, lap) => {
const targetPace = 63.0 + (lap > 12 ? (lap - 12) * 0.3 : 0); // lift and coast late
const socGate = 100 - lap * 5.5 - R() * 2; // SoC must be above this
const thermalGate = 130 - lap * 0.5; // motor temp must be below
const brakeGate = 380 - lap * 2; // brake temp must be below
const actualPace = targetPace + (R() - 0.5) * 0.8;
const energyUsed = 0.35 + R() * 0.06 + (actualPace < 63 ? 0.03 : 0);
const liftCoast = lap > 12;
return {
lap: lap + 1, targetPace: +targetPace.toFixed(2), actualPace: +actualPace.toFixed(2),
delta: +(actualPace - targetPace).toFixed(2),
socGate: +Math.max(5, socGate).toFixed(0), thermalGate: +thermalGate.toFixed(0),
brakeGate: +brakeGate.toFixed(0), energyUsed: +energyUsed.toFixed(3),
liftCoast, cumEnergy: +(energyUsed * (lap + 1)).toFixed(2),
};
});
}

function gMonteCarlo(nRuns = 500, nLaps = 16) {
const results = [];
for (let run = 0; run < nRuns; run++) {
const R = srng(10000 + run);
let soc = 100, energy = 6.5, motorT = 55, finished = true, dnfLap = -1;
let totalTime = 0;
for (let lap = 0; lap < nLaps; lap++) {
const consumption = 0.33 + R() * 0.12;
const regen = 0.03 + R() * 0.03;
energy = Math.max(0, energy - consumption + regen);
soc = (energy / 6.5) * 100;
motorT = Math.min(165, motorT + 3.5 + R() * 2 - 2.0);
const derate = motorT > 130 ? (motorT - 130) * 0.02 : 0;
const lapTime = 62.5 + R() * 1.5 + derate * 2 + (soc < 15 ? 1.5 : 0);
totalTime += lapTime;
if (soc <= 0 || motorT >= 160) { finished = false; dnfLap = lap + 1; break; }
}
results.push({
run, finished, totalTime: +totalTime.toFixed(2), finalSoC: +soc.toFixed(1),
finalMotorT: +motorT.toFixed(0), dnfLap,
});
}
// Histogram of finish times
const finishers = results.filter(r => r.finished);
const minTime = Math.min(…finishers.map(r => r.totalTime));
const maxTime = Math.max(…finishers.map(r => r.totalTime));
const bins = 25;
const timeHist = Array.from({ length: bins }, (*, i) => {
const lo = minTime + (i / bins) * (maxTime - minTime);
const hi = lo + (maxTime - minTime) / bins;
return {
bin: +((lo + hi) / 2).toFixed(1),
count: finishers.filter(r => r.totalTime >= lo && r.totalTime < hi).length,
};
});
// DNF histogram by lap
const dnfByLap = Array.from({ length: nLaps }, (*, lap) => ({
lap: lap + 1,
dnfCount: results.filter(r => !r.finished && r.dnfLap === lap + 1).length,
}));
const finishProb = (finishers.length / nRuns * 100);
const medianTime = finishers.sort((a, b) => a.totalTime - b.totalTime)[Math.floor(finishers.length / 2)]?.totalTime || 0;
const p5Time = finishers[Math.floor(finishers.length * 0.05)]?.totalTime || 0;
const p95Time = finishers[Math.floor(finishers.length * 0.95)]?.totalTime || 0;
return { results, timeHist, dnfByLap, finishProb, medianTime, p5Time, p95Time, nRuns };
}

function gPaceStrategy(nLaps = 16) {
const R = srng(801);
// Three strategies: aggressive, balanced, conservative
const strategies = ["Aggressive", "Balanced", "Conservative"];
return strategies.map((name, si) => {
const paceOffset = si * 0.6; // faster strategies use more energy
const laps = Array.from({ length: nLaps }, (_, lap) => {
const basePace = 62.0 + paceOffset + (si === 0 && lap > 10 ? (lap - 10) * 0.8 : 0); // aggressive fades
const energy = 6.5 - lap * (0.38 - si * 0.02) + (si > 0 ? lap * 0.01 : 0);
const motorT = 55 + lap * (4.5 - si * 0.8) + R() * 2;
const risk = si === 0 ? (lap > 10 ? "HIGH" : "MED") : si === 1 ? "LOW" : "NONE";
return {
lap: lap + 1, pace: +basePace.toFixed(2), energy: +Math.max(0, energy).toFixed(2),
motorT: +Math.min(160, motorT).toFixed(0), risk,
cumTime: 0, // filled below
};
});
let cum = 0;
laps.forEach(l => { cum += l.pace; l.cumTime = +cum.toFixed(2); });
return { name, laps, totalTime: +cum.toFixed(2), finishEnergy: +laps[laps.length - 1].energy, peakMotorT: +Math.max(…laps.map(l => +l.motorT)) };
});
}

function gConditions() {
const R = srng(901);
const temps = Array.from({ length: 24 }, (*, h) => ({
hour: h, ambient: +(15 + 8 * Math.sin((h - 6) * Math.PI / 12) + R() * 2).toFixed(1),
trackSurface: +(20 + 15 * Math.sin((h - 6) * Math.PI / 12) + R() * 3).toFixed(1),
humidity: +(60 - 15 * Math.sin((h - 6) * Math.PI / 12) + R() * 5).toFixed(0),
airDensity: +(1.225 - 0.004 * (15 + 8 * Math.sin((h - 6) * Math.PI / 12) - 15)).toFixed(4),
}));
// Track evolution: grip improves as rubber is laid
const trackEvo = Array.from({ length: 40 }, (*, session) => ({
session: session + 1, gripMult: +(0.92 + 0.08 * (1 - Math.exp(-session / 8)) + R() * 0.01).toFixed(3),
rubberLevel: +Math.min(100, session * 3.5 + R() * 5).toFixed(0),
}));
// Density altitude effect on aero
const altEffect = Array.from({ length: 20 }, (_, i) => {
const alt = i * 100; // 0–2000m
const densityRatio = Math.exp(-alt / 8500);
return {
altitude: alt, densityRatio: +densityRatio.toFixed(4),
downforcePct: +(densityRatio * 100).toFixed(1),
enginePct: +(densityRatio * 100).toFixed(1), // NA engine; EV = 100% always
evAdvantage: +((1 - densityRatio) * 100).toFixed(1),
};
});
return { temps, trackEvo, altEffect };
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 1: RACE OVERVIEW
// ═══════════════════════════════════════════════════════════════════════════
function OverviewTab({ battery, brakes, tireDeg, mc }) {
const finalSoC = battery[battery.length - 1]?.soc || 0;
const maxBrake = Math.max(…brakes.map(b => b.maxTemp));
const finalGrip = tireDeg[tireDeg.length - 1]?.avgGrip || 100;

const subsystems = [
{ name: "Battery SoC", value: `${finalSoC.toFixed(0)}%`, limit: ">10%", ok: finalSoC > 10, color: ELEC },
{ name: "Motor Temp", value: "→ Electronics", limit: "<150°C", ok: true, color: ELEC, crossLink: true },
{ name: "Inverter", value: "→ Electronics", limit: "<110°C", ok: true, color: ELEC, crossLink: true },
{ name: "Brake Temp", value: `${maxBrake}°C`, limit: "<380°C", ok: maxBrake < 380, color: C.am },
{ name: "Tire Grip", value: `${finalGrip.toFixed(0)}%`, limit: ">85%", ok: finalGrip > 85, color: C.gn },
{ name: "Finish Prob", value: `${mc.finishProb.toFixed(0)}%`, limit: ">95%", ok: mc.finishProb > 95, color: C.cy },
];
const allOk = subsystems.filter(s => !s.crossLink).every(s => s.ok);
const status = allOk && mc.finishProb > 90 ? "GO" : "CAUTION";

return (
<div>
{/* Status cards */}
<div style={{ display: "grid", gridTemplateColumns: `repeat(${subsystems.length}, 1fr)`, gap: 8, marginBottom: 14 }}>
{subsystems.map(s => (
<div key={s.name} style={{
…GL, padding: "10px 12px",
borderTop: `2px solid ${s.crossLink ? ELEC : s.ok ? C.gn : C.red}`,
}}>
<div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, letterSpacing: 1, marginBottom: 4 }}>{s.name}</div>
<div style={{ fontSize: 16, fontWeight: 800, color: s.crossLink ? ELEC : s.ok ? C.gn : C.red, fontFamily: C.dt }}>
{s.value}
</div>
<div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, marginTop: 2 }}>Limit: {s.limit}</div>
</div>
))}
</div>


  {/* Cross-link to Electronics */}
  <div style={{
    ...GL, padding: "8px 14px", marginBottom: 14,
    borderLeft: `2px solid ${ELEC}`, display: "flex", alignItems: "center", gap: 8,
    fontSize: 9, fontFamily: C.dt,
  }}>
    <span style={{ color: ELEC }}>⚡</span>
    <span style={{ color: C.dm }}>Detailed motor, inverter, accumulator, and torque vectoring diagnostics are in the</span>
    <span style={{ color: ELEC, fontWeight: 700 }}>Electronics</span>
    <span style={{ color: C.dm }}>module — 9 sub-tabs covering the full HV system.</span>
  </div>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
    <Sec title="SoC Projection">
      <GC><ResponsiveContainer width="100%" height={180}>
        <AreaChart data={battery} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={[0, 100]} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={0} y2={10} fill={C.red} fillOpacity={0.06} />
          <Area type="monotone" dataKey="soc" stroke={ELEC} fill={`${ELEC}12`} strokeWidth={2} dot={{ r: 2, fill: ELEC }} name="SoC %" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Brake Temps (max)">
      <GC><ResponsiveContainer width="100%" height={180}>
        <LineChart data={brakes} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={[100, 500]} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={380} stroke={C.red} strokeDasharray="4 2" />
          <Line type="monotone" dataKey="maxTemp" stroke={C.am} strokeWidth={2} dot={{ r: 2, fill: C.am }} name="Max Rotor °C" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Tire Grip Remaining">
      <GC><ResponsiveContainer width="100%" height={180}>
        <LineChart data={tireDeg} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={[75, 100]} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="avgGrip" stroke={C.gn} strokeWidth={2} dot={{ r: 2, fill: C.gn }} name="Avg Grip %" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 2: ENERGY BUDGET
// ═══════════════════════════════════════════════════════════════════════════
function EnergyTab({ battery }) {
const finalSoC = battery[battery.length - 1]?.soc || 0;
const totalRegen = battery.reduce((a, b) => a + b.regenRecov, 0);
const totalConsumed = battery.reduce((a, b) => a + b.lapEnergy, 0);

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Final SoC" value={`${finalSoC.toFixed(1)}%`} sub="end of endurance" sentiment={finalSoC > 15 ? "positive" : finalSoC > 5 ? "amber" : "negative"} delay={0} />
<KPI label="Total Consumed" value={`${totalConsumed.toFixed(2)} kWh`} sub="gross consumption" sentiment="neutral" delay={1} />
<KPI label="Total Regen" value={`${totalRegen.toFixed(2)} kWh`} sub="energy recovered" sentiment="positive" delay={2} />
<KPI label="Regen Ratio" value={`${(totalRegen / totalConsumed * 100).toFixed(1)}%`} sub="recovery efficiency" sentiment={totalRegen / totalConsumed > 0.08 ? "positive" : "amber"} delay={3} />
<KPI label="Avg Per Lap" value={`${(totalConsumed / 16).toFixed(3)} kWh`} sub="net consumption" sentiment="neutral" delay={4} />
</div>


  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="SoC & Voltage vs Lap">
      <GC><ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={battery} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis yAxisId="s" {...ax()} domain={[0, 100]} />
          <YAxis yAxisId="v" orientation="right" {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea yAxisId="s" y1={0} y2={10} fill={C.red} fillOpacity={0.06} label={{ value: "CRITICAL", fill: C.red, fontSize: 7 }} />
          <Area yAxisId="s" type="monotone" dataKey="soc" stroke={ELEC} fill={`${ELEC}10`} strokeWidth={2} dot={false} name="SoC [%]" />
          <Line yAxisId="v" type="monotone" dataKey="voltage" stroke={C.cy} strokeWidth={1.5} dot={false} name="Voltage [V]" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Per-Lap Energy [kWh]">
      <GC><ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={battery.slice(1)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Bar dataKey="lapEnergy" fill={C.red} fillOpacity={0.5} barSize={12} name="Consumed" />
          <Bar dataKey="regenRecov" fill={C.gn} fillOpacity={0.6} barSize={12} name="Recovered" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <Sec title="Cell Temperature [°C]" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={180}>
      <AreaChart data={battery} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="lap" {...ax()} />
        <YAxis {...ax()} domain={[25, 70]} />
        <Tooltip contentStyle={TT} />
        <ReferenceArea y1={55} y2={70} fill={C.red} fillOpacity={0.05} label={{ value: "DERATE", fill: C.red, fontSize: 7 }} />
        <Area type="monotone" dataKey="cellTemp" stroke={C.am} fill={`${C.am}12`} strokeWidth={1.5} dot={{ r: 2, fill: C.am }} name="Cell Temp °C" />
      </AreaChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 3: STINT PLANNER
// ═══════════════════════════════════════════════════════════════════════════
function StintTab({ stintPlan }) {
const onTarget = stintPlan.filter(s => Math.abs(s.delta) < 0.3).length;
const totalEnergy = stintPlan.reduce((a, s) => a + s.energyUsed, 0);
const liftCoastLaps = stintPlan.filter(s => s.liftCoast).length;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="On Target" value={`${onTarget}/${stintPlan.length}`} sub="within ±0.3s" sentiment={onTarget > 12 ? "positive" : "amber"} delay={0} />
<KPI label="Total Energy" value={`${totalEnergy.toFixed(2)} kWh`} sub="planned consumption" sentiment={totalEnergy < 6.3 ? "positive" : "amber"} delay={1} />
<KPI label="Lift & Coast" value={`${liftCoastLaps} laps`} sub="energy saving laps" sentiment="neutral" delay={2} />
<KPI label="Target Avg" value={`${(stintPlan.reduce((a, s) => a + s.targetPace, 0) / stintPlan.length).toFixed(2)}s`} sub="mean lap target" sentiment="neutral" delay={3} />
</div>


  {/* Stint plan table */}
  <Sec title="Lap-by-Lap Race Plan">
    <GC style={{ padding: 10 }}>
      <div style={{ display: "grid", gridTemplateColumns: "50px 80px 80px 60px 60px 60px 60px 60px", gap: 0, fontSize: 8, fontFamily: C.dt }}>
        {["Lap", "Target [s]", "Actual [s]", "Δ", "Energy", "SoC Gate", "Mode", "Risk"].map(h => (
          <div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 1, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>
        ))}
        {stintPlan.map(s => (
          <React.Fragment key={s.lap}>
            <div style={{ color: C.cy, fontWeight: 700, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.lap}</div>
            <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.targetPace}</div>
            <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.actualPace}</div>
            <div style={{ color: Math.abs(s.delta) < 0.3 ? C.gn : C.am, fontWeight: 700, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.delta > 0 ? "+" : ""}{s.delta}</div>
            <div style={{ color: C.br, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.energyUsed}</div>
            <div style={{ color: C.dm, padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.socGate}%</div>
            <div style={{ padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>
              <span style={{ fontSize: 7, color: s.liftCoast ? C.am : C.gn, background: s.liftCoast ? `${C.am}15` : `${C.gn}15`, padding: "1px 5px", borderRadius: 4 }}>
                {s.liftCoast ? "L&C" : "PUSH"}
              </span>
            </div>
            <div style={{ padding: "5px 4px", borderBottom: `1px solid ${C.b1}08` }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: s.lap > 14 ? C.am : C.gn }} />
            </div>
          </React.Fragment>
        ))}
      </div>
    </GC>
  </Sec>

  <Sec title="Target vs Actual Pace" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={stintPlan} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="lap" {...ax()} />
        <YAxis {...ax()} domain={[61, 65]} />
        <Tooltip contentStyle={TT} />
        <Line type="monotone" dataKey="targetPace" stroke={C.gn} strokeWidth={2} dot={false} name="Target" strokeDasharray="6 3" />
        <Line type="monotone" dataKey="actualPace" stroke={C.cy} strokeWidth={2} dot={{ r: 2, fill: C.cy }} name="Actual" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </ComposedChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 4: MONTE CARLO
// ═══════════════════════════════════════════════════════════════════════════
function MonteCarloTab({ mc }) {
return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Finish Prob" value={`${mc.finishProb.toFixed(1)}%`} sub={`${mc.nRuns} simulations`} sentiment={mc.finishProb > 95 ? "positive" : mc.finishProb > 85 ? "amber" : "negative"} delay={0} />
<KPI label="Median Time" value={`${mc.medianTime.toFixed(1)}s`} sub="P50 total race time" sentiment="neutral" delay={1} />
<KPI label="P5 (Best)" value={`${mc.p5Time.toFixed(1)}s`} sub="optimistic bound" sentiment="positive" delay={2} />
<KPI label="P95 (Worst)" value={`${mc.p95Time.toFixed(1)}s`} sub="pessimistic bound" sentiment="neutral" delay={3} />
<KPI label="Spread" value={`${(mc.p95Time - mc.p5Time).toFixed(1)}s`} sub="P95 − P5 range" sentiment={(mc.p95Time - mc.p5Time) < 30 ? "positive" : "amber"} delay={4} />
</div>


  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Race Time Distribution (finishers)">
      <GC><ResponsiveContainer width="100%" height={260}>
        <BarChart data={mc.timeHist} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="bin" {...ax()} label={{ value: "Total Race Time [s]", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine x={mc.medianTime} stroke={C.cy} strokeDasharray="4 4" label={{ value: "P50", fill: C.cy, fontSize: 7 }} />
          <Bar dataKey="count" barSize={12} radius={[4, 4, 0, 0]}>
            {mc.timeHist.map((h, i) => <Cell key={i} fill={h.bin < mc.medianTime ? C.gn : C.am} fillOpacity={0.6} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="DNF Events by Lap">
      <GC><ResponsiveContainer width="100%" height={260}>
        <BarChart data={mc.dnfByLap} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} label={{ value: "Lap", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Bar dataKey="dnfCount" fill={C.red} fillOpacity={0.6} barSize={16} radius={[4, 4, 0, 0]} name="DNF count" />
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <div style={{ ...GL, padding: "10px 14px", marginTop: 10, fontSize: 9, color: C.dm, fontFamily: C.dt, lineHeight: 1.7 }}>
    Monte Carlo simulation runs {mc.nRuns} stochastic endurance events with randomized energy consumption, regen efficiency, motor thermal rise, and driver pace variation.
    Each run uses a unique random seed — the distribution captures the inherent uncertainty in race outcomes.
    <span style={{ color: mc.finishProb > 95 ? C.gn : C.am, fontWeight: 600 }}> {mc.finishProb.toFixed(1)}% of simulations completed all 16 laps.</span>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 5: BRAKE THERMAL
// ═══════════════════════════════════════════════════════════════════════════
function BrakeTab({ brakes }) {
const maxBrake = Math.max(…brakes.map(b => b.maxTemp));
const fadeRisk = maxBrake > 400 ? "HIGH" : maxBrake > 350 ? "MODERATE" : "LOW";

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Peak Rotor" value={`${maxBrake}°C`} sub="worst corner" sentiment={maxBrake < 380 ? "positive" : maxBrake < 430 ? "amber" : "negative"} delay={0} />
<KPI label="Fade Risk" value={fadeRisk} sub="onset ~380°C" sentiment={fadeRisk === "LOW" ? "positive" : "negative"} delay={1} />
<KPI label="F/R Bias" value={`${((+brakes[brakes.length - 1]?.FL + +brakes[brakes.length - 1]?.FR) / (+brakes[brakes.length - 1]?.RL + +brakes[brakes.length - 1]?.RR)).toFixed(2)}`} sub="front/rear temp ratio" sentiment="neutral" delay={2} />
<KPI label="Cooling Rate" value={`~8%/lap`} sub="Newton’s law fit" sentiment="neutral" delay={3} />
</div>


  <Sec title="Brake Rotor Temperature Per Corner [°C]">
    <GC><ResponsiveContainer width="100%" height={280}>
      <LineChart data={brakes} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="lap" {...ax()} />
        <YAxis {...ax()} domain={[100, 500]} />
        <Tooltip contentStyle={TT} />
        <ReferenceArea y1={380} y2={500} fill={C.red} fillOpacity={0.05} label={{ value: "FADE ZONE", fill: C.red, fontSize: 7 }} />
        <ReferenceLine y={380} stroke={C.red} strokeDasharray="4 2" />
        <Line dataKey="FL" stroke={CL.fl} strokeWidth={1.5} dot={{ r: 2 }} name="FL" />
        <Line dataKey="FR" stroke={CL.fr} strokeWidth={1.5} dot={{ r: 2 }} name="FR" />
        <Line dataKey="RL" stroke={CL.rl} strokeWidth={1.5} dot={{ r: 2 }} name="RL" />
        <Line dataKey="RR" stroke={CL.rr} strokeWidth={1.5} dot={{ r: 2 }} name="RR" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </LineChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 6: TIRE DEGRADATION
// ═══════════════════════════════════════════════════════════════════════════
function TireTab({ tireDeg }) {
const finalGrip = tireDeg[tireDeg.length - 1]?.avgGrip || 100;
const lapDelta = (tireDeg[tireDeg.length - 1]?.lapTime || 62.5) - (tireDeg[0]?.lapTime || 62.5);
const peakWear = Math.max(…tireDeg.filter(t => t.wearRate > 0).map(t => +t.wearRate));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Final Grip" value={`${finalGrip.toFixed(1)}%`} sub="average remaining" sentiment={finalGrip > 90 ? "positive" : "amber"} delay={0} />
<KPI label="Grip Lost" value={`${(100 - finalGrip).toFixed(1)}%`} sub="total degradation" sentiment={100 - finalGrip < 10 ? "positive" : "amber"} delay={1} />
<KPI label="Lap Time Δ" value={`+${lapDelta.toFixed(2)}s`} sub="deg cost" sentiment={lapDelta < 0.8 ? "positive" : "amber"} delay={2} />
<KPI label="Peak Wear" value={`${peakWear}%/lap`} sub="highest rate" sentiment={peakWear < 0.7 ? "positive" : "amber"} delay={3} />
</div>


  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Grip Remaining Per Corner [%]">
      <GC><ResponsiveContainer width="100%" height={240}>
        <LineChart data={tireDeg} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={[75, 100]} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={75} y2={85} fill={C.am} fillOpacity={0.04} />
          <Line dataKey="FL" stroke={CL.fl} strokeWidth={1.5} dot={{ r: 2 }} name="FL" />
          <Line dataKey="FR" stroke={CL.fr} strokeWidth={1.5} dot={{ r: 2 }} name="FR" />
          <Line dataKey="RL" stroke={CL.rl} strokeWidth={1.5} dot={{ r: 2 }} name="RL" />
          <Line dataKey="RR" stroke={CL.rr} strokeWidth={1.5} dot={{ r: 2 }} name="RR" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Predicted Lap Time vs Grip">
      <GC><ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="avgGrip" type="number" {...ax()} name="Grip %" label={{ value: "Grip %", position: "insideBottom", offset: -8, fill: C.dm, fontSize: 7 }} />
          <YAxis dataKey="lapTime" type="number" {...ax()} domain={[62, 65]} />
          <Tooltip contentStyle={TT} />
          <Scatter data={tireDeg.slice(1)} fill={C.gn} fillOpacity={0.6} r={4} name="Lap Time vs Grip" />
        </ScatterChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 7: PACE STRATEGY
// ═══════════════════════════════════════════════════════════════════════════
function PaceTab({ strategies }) {
return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
{strategies.map((s, i) => (
<div key={s.name} style={{ …GL, padding: "12px 14px", borderTop: `2px solid ${i === 0 ? C.red : i === 1 ? C.gn : C.cy}` }}>
<div style={{ fontSize: 12, fontWeight: 800, color: i === 0 ? C.red : i === 1 ? C.gn : C.cy, fontFamily: C.dt }}>{s.name}</div>
<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, marginTop: 8, fontSize: 8, fontFamily: C.dt }}>
<div><span style={{ color: C.dm }}>Total: </span><span style={{ color: C.br }}>{s.totalTime.toFixed(1)}s</span></div>
<div><span style={{ color: C.dm }}>End E: </span><span style={{ color: s.finishEnergy > 0 ? C.gn : C.red }}>{s.finishEnergy.toFixed(1)} kWh</span></div>
<div><span style={{ color: C.dm }}>Peak T: </span><span style={{ color: s.peakMotorT < 140 ? C.gn : C.am }}>{s.peakMotorT}°C</span></div>
<div><span style={{ color: C.dm }}>Risk: </span><span style={{ color: i === 0 ? C.red : C.gn }}>{i === 0 ? "HIGH" : i === 1 ? "LOW" : "NONE"}</span></div>
</div>
</div>
))}
</div>


  <Sec title="Pace Comparison — Lap Time [s]">
    <GC><ResponsiveContainer width="100%" height={260}>
      <LineChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="lap" type="number" {...ax()} domain={[1, 16]} />
        <YAxis {...ax()} domain={[61, 66]} />
        <Tooltip contentStyle={TT} />
        {strategies.map((s, i) => (
          <Line key={s.name} data={s.laps} dataKey="pace" stroke={i === 0 ? C.red : i === 1 ? C.gn : C.cy}
            strokeWidth={2} dot={false} name={s.name} strokeDasharray={i === 2 ? "6 3" : undefined} />
        ))}
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </LineChart>
    </ResponsiveContainer></GC>
  </Sec>

  <Sec title="Energy Remaining by Strategy [kWh]" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={200}>
      <LineChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="lap" type="number" {...ax()} domain={[1, 16]} />
        <YAxis {...ax()} domain={[0, 7]} />
        <Tooltip contentStyle={TT} />
        <ReferenceLine y={0} stroke={C.red} strokeWidth={2} label={{ value: "EMPTY", fill: C.red, fontSize: 7 }} />
        {strategies.map((s, i) => (
          <Line key={s.name} data={s.laps} dataKey="energy" stroke={i === 0 ? C.red : i === 1 ? C.gn : C.cy}
            strokeWidth={1.5} dot={false} name={s.name} />
        ))}
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </LineChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 8: CONDITIONS
// ═══════════════════════════════════════════════════════════════════════════
function ConditionsTab({ conditions }) {
const { temps, trackEvo, altEffect } = conditions;
const currentHour = 14; // assume 2pm race
const currentTemp = temps.find(t => t.hour === currentHour);

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Ambient" value={`${currentTemp?.ambient || 23}°C`} sub="at race time" sentiment="neutral" delay={0} />
<KPI label="Track Surface" value={`${currentTemp?.trackSurface || 35}°C`} sub="asphalt temp" sentiment="neutral" delay={1} />
<KPI label="Humidity" value={`${currentTemp?.humidity || 55}%`} sub="relative" sentiment="neutral" delay={2} />
<KPI label="Air Density" value={`${currentTemp?.airDensity || 1.22} kg/m³`} sub="for aero calc" sentiment="neutral" delay={3} />
</div>


  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Ambient & Track Temperature [°C]">
      <GC><ResponsiveContainer width="100%" height={220}>
        <LineChart data={temps} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="hour" {...ax()} label={{ value: "Hour", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine x={currentHour} stroke={C.gn} strokeDasharray="4 4" label={{ value: "RACE", fill: C.gn, fontSize: 7 }} />
          <Line type="monotone" dataKey="ambient" stroke={C.cy} strokeWidth={1.5} dot={false} name="Ambient" />
          <Line type="monotone" dataKey="trackSurface" stroke={C.am} strokeWidth={1.5} dot={false} name="Track Surface" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Track Grip Evolution (rubber build-up)">
      <GC><ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={trackEvo} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="session" {...ax()} />
          <YAxis yAxisId="g" {...ax()} domain={[0.9, 1.05]} />
          <YAxis yAxisId="r" orientation="right" {...ax()} domain={[0, 100]} />
          <Tooltip contentStyle={TT} />
          <Line yAxisId="g" type="monotone" dataKey="gripMult" stroke={C.gn} strokeWidth={2} dot={false} name="Grip Multiplier" />
          <Area yAxisId="r" type="monotone" dataKey="rubberLevel" stroke={C.am} fill={`${C.am}08`} strokeWidth={1} dot={false} name="Rubber Level %" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <Sec title="Density Altitude Effect on Aerodynamics & Power" style={{ marginTop: 10 }}>
    <GC><ResponsiveContainer width="100%" height={200}>
      <LineChart data={altEffect} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="altitude" {...ax()} label={{ value: "Altitude [m]", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
        <YAxis {...ax()} domain={[70, 105]} />
        <Tooltip contentStyle={TT} />
        <Line type="monotone" dataKey="downforcePct" stroke={C.cy} strokeWidth={2} dot={false} name="Downforce %" />
        <Line type="monotone" dataKey="evAdvantage" stroke={C.gn} strokeWidth={1.5} dot={false} name="EV Advantage %" strokeDasharray="6 3" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </LineChart>
    </ResponsiveContainer></GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function EnduranceStrategyModule() {
const [tab, setTab] = useState("overview");

const battery = useMemo(() => gBatteryTrace(), []);
const brakes = useMemo(() => gBrakeThermal(), []);
const tireDeg = useMemo(() => gTireDeg(), []);
const stintPlan = useMemo(() => gStintPlan(), []);
const mc = useMemo(() => gMonteCarlo(), []);
const strategies = useMemo(() => gPaceStrategy(), []);
const conditions = useMemo(() => gConditions(), []);

const finalSoC = battery[battery.length - 1]?.soc || 0;
const maxBrake = Math.max(…brakes.map(b => b.maxTemp));
const finalGrip = tireDeg[tireDeg.length - 1]?.avgGrip || 100;
const status = finalSoC > 10 && maxBrake < 450 && mc.finishProb > 85 ? "GO" : "CAUTION";

return (
<div>
{/* Status banner */}
<div style={{
…GL, padding: "12px 16px", marginBottom: 14,
borderLeft: `3px solid ${status === "GO" ? C.gn : C.am}`,
display: "flex", alignItems: "center", gap: 16,
}}>
<div style={{
width: 10, height: 10, borderRadius: 5,
background: status === "GO" ? C.gn : C.am,
boxShadow: `0 0 10px ${status === "GO" ? C.gn : C.am}`,
animation: "pulseGlow 2s infinite",
}} />
<div>
<span style={{ fontSize: 12, fontWeight: 800, color: status === "GO" ? C.gn : C.am, fontFamily: C.dt, letterSpacing: 2 }}>
ENDURANCE: {status}
</span>
<span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 16 }}>
22km · 16 laps · P(finish)={mc.finishProb.toFixed(0)}% · SoC→{finalSoC.toFixed(0)}% · Brake {maxBrake}°C · Grip {finalGrip.toFixed(0)}%
</span>
</div>
</div>


  <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
    {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.gn} />)}
  </div>

  {tab === "overview" && <OverviewTab battery={battery} brakes={brakes} tireDeg={tireDeg} mc={mc} />}
  {tab === "energy" && <EnergyTab battery={battery} />}
  {tab === "stint" && <StintTab stintPlan={stintPlan} />}
  {tab === "montecarlo" && <MonteCarloTab mc={mc} />}
  {tab === "brakes" && <BrakeTab brakes={brakes} />}
  {tab === "tires" && <TireTab tireDeg={tireDeg} />}
  {tab === "pace" && <PaceTab strategies={strategies} />}
  {tab === "conditions" && <ConditionsTab conditions={conditions} />}
</div>


);
}