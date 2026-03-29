// ═══════════════════════════════════════════════════════════════════════════
// src/ElectronicsModule.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════
// Dedicated electronics & powertrain department hub. Consolidates all EE
// data previously scattered across Endurance (battery, thermal, TV), Live
// Mode (E-Drive panel), Weight & CG (mass only), and Compliance (checklist).
//
// Sub-tabs (9):
//   1. Accumulator    — Cell-level monitoring, SoC, voltage sag, impedance
//   2. Inverter/Motor — Efficiency maps, thermal, derating, back-EMF
//   3. Torque Vector  — Yaw moment allocation, slip optimization, TV error
//   4. Regen Strategy — Recovery curves, C-rate limits, energy balance
//   5. Safety Circuits— TSAL, IMD, BSPD, AMS, HVD live status
//   6. CAN Bus Health — Message rates, bus load, latency, error frames
//   7. Sensor Fusion  — IMU calibration, EKF innovations, fault detection
//   8. Thermal Mgmt   — Coolant loop, pump duty, thermal margin
//   9. Power Budget   — Efficiency chain, Sankey breakdown, LV system
//
// Integration (3 lines in App.jsx):
//   NAV: { key: "electronics", label: "Electronics", icon: "⚡" }
//   Import: import ElectronicsModule from "./ElectronicsModule.jsx"
//   Route: case "electronics": return <ElectronicsModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT, AX } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const ELEC = "#7c3aed";
const ELEC_G = "rgba(124,58,237,0.10)";
const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });

const TABS = [
{ key: "accumulator", label: "Accumulator" },
{ key: "inverter",    label: "Inverter / Motor" },
{ key: "tv",          label: "Torque Vectoring" },
{ key: "regen",       label: "Regen Strategy" },
{ key: "safety",      label: "Safety Circuits" },
{ key: "can",         label: "CAN Bus" },
{ key: "sensors",     label: "Sensor Fusion" },
{ key: "thermal",     label: "Thermal Mgmt" },
{ key: "power",       label: "Power Budget" },
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

// 1. Accumulator — 16 laps, per-segment voltages, cell temps, SoC, impedance
function gAccumulator(nLaps = 16) {
const R = srng(5001);
const nSegments = 8;
const data = [];
let soc = 100, packVoltage = 588, energy = 6.5;
for (let lap = 0; lap <= nLaps; lap++) {
const lapEnergy = 0.35 + R() * 0.08;
const regenRecov = 0.04 + R() * 0.02;
energy = Math.max(0, energy - lapEnergy + regenRecov);
soc = (energy / 6.5) * 100;
packVoltage = 540 + soc * 0.48 - (lap > 10 ? (lap - 10) * 3 : 0);
const current = 80 + R() * 40 + (soc < 20 ? 30 : 0);
const cellTemp = 28 + lap * 1.8 + R() * 3 + (soc < 30 ? 5 : 0);
const Ri = 0.008 + 0.002 * (1 - soc / 100) + 0.001 * Math.max(0, cellTemp - 40) / 20;
const segments = Array.from({ length: nSegments }, (_, si) => ({
seg: si + 1,
voltage: +(packVoltage / nSegments + (R() - 0.5) * 2).toFixed(2),
temp: +(cellTemp + (R() - 0.5) * 4 + (si === 3 || si === 4 ? 2 : 0)).toFixed(1),
}));
const maxSegT = Math.max(…segments.map(s => s.temp));
const minSegV = Math.min(…segments.map(s => s.voltage));
data.push({
lap, soc: +soc.toFixed(1), packVoltage: +packVoltage.toFixed(1),
current: +current.toFixed(1), cellTemp: +cellTemp.toFixed(1),
energy: +energy.toFixed(3), lapEnergy: +lapEnergy.toFixed(3),
regenRecov: +regenRecov.toFixed(3), Ri: +Ri.toFixed(4),
segments, maxSegT: +maxSegT.toFixed(1), minSegV: +minSegV.toFixed(2),
powerLimit: +(80 * Math.min(1, soc / 15) * Math.min(1, (60 - cellTemp) / 10)).toFixed(1),
});
}
return data;
}

// 2. Motor efficiency map (torque × speed grid)
function gMotorEffMap(res = 20) {
const data = [];
for (let ti = 0; ti < res; ti++) {
for (let si = 0; si < res; si++) {
const torque = (ti / (res - 1)) * 450;           // 0–450 Nm
const speed = (si / (res - 1)) * 6000;            // 0–6000 rpm
const powerMech = torque * speed * Math.PI / 30;
// Efficiency peaks ~93% at moderate torque/speed, drops at extremes
const effPeak = 0.935;
const tNorm = torque / 450, sNorm = speed / 6000;
const eff = effPeak - 0.15 * Math.pow(tNorm - 0.6, 2) - 0.10 * Math.pow(sNorm - 0.5, 2)
- (speed < 300 ? 0.15 : 0) - (torque < 20 ? 0.1 : 0);
const clampEff = Math.max(0.30, Math.min(0.96, eff));
data.push({
torque: +torque.toFixed(0), speed: +speed.toFixed(0),
eff: +clampEff.toFixed(3), power: +(powerMech / 1000).toFixed(1),
});
}
}
return data;
}

// 3. Inverter & motor thermal trace
function gInvMotorThermal(n = 200) {
const R = srng(5101);
return Array.from({ length: n }, (_, i) => {
const t = i * 0.5;
const load = 0.5 + 0.4 * Math.sin(i * 0.03) + 0.1 * R();
const motL = 40 + i * 0.35 * load + 3 * R();
const motR = 42 + i * 0.33 * load + 2.5 * R();
const invL = 35 + i * 0.20 * load + 2 * R();
const invR = 36 + i * 0.19 * load + 1.8 * R();
const derateMotor = Math.max(0, (Math.max(motL, motR) - 130) / 20 * 100);
const derateInv = Math.max(0, (Math.max(invL, invR) - 95) / 15 * 100);
return {
t: +t.toFixed(1), motL: +motL.toFixed(1), motR: +motR.toFixed(1),
invL: +invL.toFixed(1), invR: +invR.toFixed(1),
derating: +Math.max(derateMotor, derateInv).toFixed(1),
backEMF: +(300 + i * 0.8 + R() * 10).toFixed(0),
};
});
}

// 4. Torque vectoring allocation over time
function gTVAlloc(n = 200) {
const R = srng(5201);
return Array.from({ length: n }, (_, i) => {
const t = i * 0.05;
const yawDemand = 80 * Math.sin(i * 0.04) + 20 * Math.sin(i * 0.11) + 5 * R();
const yawActual = yawDemand * (0.92 + 0.06 * R()) + (R() - 0.5) * 3;
const error = yawActual - yawDemand;
const tRL = 120 + 60 * Math.sin(i * 0.04) + 10 * R();
const tRR = 120 - 60 * Math.sin(i * 0.04) + 10 * R();
return {
t: +t.toFixed(2), yawDemand: +yawDemand.toFixed(1), yawActual: +yawActual.toFixed(1),
tvError: +error.toFixed(2), tRL: +tRL.toFixed(0), tRR: +tRR.toFixed(0),
slipRL: +(0.03 + 0.02 * R() + 0.01 * Math.sin(i * 0.04)).toFixed(4),
slipRR: +(0.03 + 0.02 * R() - 0.01 * Math.sin(i * 0.04)).toFixed(4),
};
});
}

// 5. Regen strategy data
function gRegenData(n = 150) {
const R = srng(5301);
let cumRecovered = 0, cumConsumed = 0;
return Array.from({ length: n }, (_, i) => {
const t = i * 0.1;
const braking = Math.sin(i * 0.06) < -0.2;
const speed = 10 + 12 * Math.sin(i * 0.025) + 2 * R();
const motorSpeed = speed / 0.2032 * 60 / (2 * Math.PI); // rpm from wheel
const maxRegenTorque = Math.min(450, motorSpeed > 500 ? 450 * (1 - (motorSpeed - 500) / 5500) : 450);
const regenPower = braking ? Math.min(25, maxRegenTorque * motorSpeed * Math.PI / 30 / 1000 * 0.85) : 0;
const consumePower = !braking ? 15 + R() * 30 : 0;
cumRecovered += regenPower * 0.1 / 3600;
cumConsumed += consumePower * 0.1 / 3600;
const cRate = braking ? regenPower / 6.5 : -consumePower / 6.5;
return {
t: +t.toFixed(1), speed: +speed.toFixed(1), braking,
regenPower: +regenPower.toFixed(1), consumePower: +consumePower.toFixed(1),
maxRegenTorque: +maxRegenTorque.toFixed(0),
cRate: +cRate.toFixed(2), motorSpeed: +motorSpeed.toFixed(0),
cumRecovered: +cumRecovered.toFixed(4), cumConsumed: +cumConsumed.toFixed(4),
netBalance: +(cumRecovered - cumConsumed).toFixed(4),
};
});
}

// 6. Safety circuit states
function gSafetyState() {
const R = srng(5401);
const circuits = [
{ id: "TSAL", name: "Tractive System Active Light", rule: "EV 6.1", threshold: "HV>60V", reading: +(580 + R() * 10).toFixed(0), unit: "V", state: "ACTIVE", faults: 0, lastFault: null },
{ id: "IMD", name: "Insulation Monitoring Device", rule: "EV 7.1", threshold: "R>500Ω/V", reading: +(620 + R() * 80).toFixed(0), unit: "Ω/V", state: "OK", faults: 1, lastFault: "Pre-event test" },
{ id: "BSPD", name: "Brake System Plausibility", rule: "EV 7.2", threshold: "5kW+brake", reading: "0", unit: "trips", state: "OK", faults: 0, lastFault: null },
{ id: "AMS", name: "Accumulator Management System", rule: "EV 6.3", threshold: "V/T limits", reading: +(588).toFixed(0), unit: "V", state: "OK", faults: 0, lastFault: null },
{ id: "HVD", name: "High Voltage Disconnect", rule: "EV 6.4", threshold: "Manual pull", reading: "CLOSED", unit: "", state: "ARMED", faults: 0, lastFault: null },
{ id: "BOTS", name: "Brake Over-Travel Switch", rule: "T 6.1", threshold: "Pedal overtravel", reading: "0", unit: "trips", state: "OK", faults: 0, lastFault: null },
];
// Fault timeline
const faultTimeline = [
{ t: -120, circuit: "IMD", event: "Pre-event self-test trip", severity: "info", resolved: true },
{ t: -60, circuit: "AMS", event: "Cell #3 temp spike 58°C", severity: "warning", resolved: true },
{ t: 0, circuit: "TSAL", event: "System armed — all clear", severity: "info", resolved: true },
];
return { circuits, faultTimeline };
}

// 7. CAN bus health
function gCANHealth(n = 100) {
const R = srng(5501);
const nodes = ["ECU", "BMS", "IMD", "INV_L", "INV_R", "DASH", "DL", "GPS"];
const timeline = Array.from({ length: n }, (_, i) => {
const t = i * 0.5;
const busLoad = 35 + 15 * Math.sin(i * 0.03) + 5 * R();
const errorFrames = R() > 0.97 ? Math.floor(1 + R() * 3) : 0;
const msgRate = 850 + 150 * Math.sin(i * 0.02) + 30 * R();
return {
t: +t.toFixed(1), busLoad: +busLoad.toFixed(1),
errorFrames, msgRate: +msgRate.toFixed(0),
latencyP50: +(0.8 + R() * 0.4).toFixed(2),
latencyP99: +(2.5 + R() * 1.5 + (errorFrames > 0 ? 3 : 0)).toFixed(2),
};
});
const nodeHealth = nodes.map(n => ({
node: n, expectedHz: n === "BMS" ? 10 : n.startsWith("INV") ? 100 : 50,
actualHz: +(n === "BMS" ? 10 + (R() - 0.5) : n.startsWith("INV") ? 98 + R() * 4 : 49 + R() * 2).toFixed(1),
timeouts: Math.floor(R() > 0.8 ? R() * 3 : 0),
status: R() > 0.1 ? "OK" : "TIMEOUT",
}));
return { timeline, nodeHealth };
}

// 8. Sensor fusion / EKF health
function gSensorHealth(n = 150) {
const R = srng(5601);
return Array.from({ length: n }, (_, i) => {
const t = i * 0.1;
const innovAx = (R() - 0.5) * 0.06;
const innovWz = (R() - 0.5) * 0.04;
const innovVy = (R() - 0.5) * 0.05;
const imuBiasAx = 0.012 + 0.002 * Math.sin(i * 0.005);
const imuBiasGz = -0.003 + 0.001 * Math.sin(i * 0.003);
const wheelNoise = 0.5 + R() * 0.3;
const gpsHDOP = 1.2 + R() * 0.8 + (R() > 0.95 ? 3 : 0); // occasional GPS degradation
const confidence = Math.max(0, Math.min(100, 95 - Math.abs(innovAx) * 500 - Math.abs(innovWz) * 800 - (gpsHDOP > 2.5 ? 10 : 0)));
const sensorFault = R() > 0.98;
return {
t: +t.toFixed(1), innovAx: +innovAx.toFixed(4), innovWz: +innovWz.toFixed(4),
innovVy: +innovVy.toFixed(4), imuBiasAx: +imuBiasAx.toFixed(4),
imuBiasGz: +imuBiasGz.toFixed(5), wheelNoise: +wheelNoise.toFixed(2),
gpsHDOP: +gpsHDOP.toFixed(2), confidence: +confidence.toFixed(1), sensorFault,
};
});
}

// 9. Coolant loop thermal
function gCoolantLoop(n = 100) {
const R = srng(5701);
return Array.from({ length: n }, (_, i) => {
const t = i * 1.0; // seconds
const ambientT = 25 + 3 * R();
const radIn = ambientT + 8 + i * 0.15 + R() * 2;
const radOut = radIn - 8 - R() * 3;
const motJacket = radIn + 5 + i * 0.1 + R() * 3;
const invPlate = radIn + 2 + i * 0.08 + R() * 2;
const battPlate = radIn + 1 + i * 0.05 + R() * 1.5;
const pumpDuty = Math.min(100, 30 + i * 0.5 + R() * 5);
const fanSpeed = pumpDuty > 60 ? 2000 + (pumpDuty - 60) * 50 + R() * 200 : R() * 500;
const flowRate = 5 + pumpDuty * 0.1; // L/min
const thermalMargin = Math.max(0, 140 - Math.max(motJacket, invPlate, battPlate));
return {
t: +t.toFixed(0), radIn: +radIn.toFixed(1), radOut: +radOut.toFixed(1),
motJacket: +motJacket.toFixed(1), invPlate: +invPlate.toFixed(1),
battPlate: +battPlate.toFixed(1), pumpDuty: +pumpDuty.toFixed(0),
fanSpeed: +fanSpeed.toFixed(0), flowRate: +flowRate.toFixed(1),
thermalMargin: +thermalMargin.toFixed(1), ambientT: +ambientT.toFixed(1),
};
});
}

// 10. Power budget
function gPowerBudget(n = 100) {
const R = srng(5801);
return Array.from({ length: n }, (_, i) => {
const t = i * 0.15;
const battOut = 25 + 20 * Math.sin(i * 0.04) + 5 * R();
const invLoss = battOut * 0.035 + R() * 0.5;
const motorCopper = (battOut - invLoss) * 0.04 + R() * 0.3;
const motorIron = (battOut - invLoss) * 0.015 + R() * 0.2;
const mechOut = battOut - invLoss - motorCopper - motorIron;
const tireLoss = mechOut * 0.08;
const aeroLoss = 0.5 * 1.225 * Math.pow(12 + 8 * Math.sin(i * 0.03), 3) * 1.15 * 4.554 / 1000;
const lvDraw = 0.18 + R() * 0.05; // kW low-voltage system
const totalEff = mechOut / battOut;
return {
t: +t.toFixed(2), battOut: +battOut.toFixed(1), invLoss: +invLoss.toFixed(2),
motorCopper: +motorCopper.toFixed(2), motorIron: +motorIron.toFixed(2),
mechOut: +mechOut.toFixed(1), tireLoss: +tireLoss.toFixed(2),
aeroLoss: +aeroLoss.toFixed(2), lvDraw: +lvDraw.toFixed(2),
totalEff: +totalEff.toFixed(3),
};
});
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 1: ACCUMULATOR
// ═══════════════════════════════════════════════════════════════════════════
function AccumulatorTab() {
const acc = useMemo(() => gAccumulator(), []);
const finalSoC = acc[acc.length - 1].soc;
const maxTemp = Math.max(…acc.map(a => a.cellTemp));
const minVoltage = Math.min(…acc.map(a => a.packVoltage));
const totalRegen = acc.reduce((a, d) => a + d.regenRecov, 0);

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Final SoC" value={`${finalSoC.toFixed(1)}%`} sub="end of endurance" sentiment={finalSoC > 15 ? "positive" : finalSoC > 5 ? "amber" : "negative"} delay={0} />
<KPI label="Peak Cell Temp" value={`${maxTemp.toFixed(0)}°C`} sub="worst cell" sentiment={maxTemp < 55 ? "positive" : maxTemp < 60 ? "amber" : "negative"} delay={1} />
<KPI label="Min Pack V" value={`${minVoltage.toFixed(0)} V`} sub="voltage floor" sentiment={minVoltage > 480 ? "positive" : "amber"} delay={2} />
<KPI label="Total Regen" value={`${totalRegen.toFixed(2)} kWh`} sub="energy recovered" sentiment="positive" delay={3} />
<KPI label="R_internal" value={`${acc[acc.length - 1].Ri.toFixed(3)} Ω`} sub="at end state" sentiment={acc[acc.length - 1].Ri < 0.012 ? "positive" : "amber"} delay={4} />
</div>


  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="State of Charge [%]">
      <GC><ResponsiveContainer width="100%" height={220}>
        <AreaChart data={acc} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={[0, 100]} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={0} y2={10} fill={C.red} fillOpacity={0.06} label={{ value: "CRITICAL", fill: C.red, fontSize: 7 }} />
          <Area type="monotone" dataKey="soc" stroke={ELEC} fill={ELEC_G} strokeWidth={2} dot={{ r: 2, fill: ELEC }} name="SoC %" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Pack Voltage & Current">
      <GC><ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={acc} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis yAxisId="v" {...ax()} label={{ value: "V", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }} />
          <YAxis yAxisId="i" orientation="right" {...ax()} label={{ value: "A", angle: 90, position: "insideRight", fill: C.dm, fontSize: 8 }} />
          <Tooltip contentStyle={TT} />
          <Line yAxisId="v" dataKey="packVoltage" stroke={C.cy} strokeWidth={1.5} dot={false} name="Voltage [V]" />
          <Line yAxisId="i" dataKey="current" stroke={C.am} strokeWidth={1.2} dot={false} name="Current [A]" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  {/* Segment-level cell matrix */}
  <Sec title="Cell Segment Temperature Matrix" style={{ marginTop: 10 }}>
    <GC style={{ padding: 10 }}>
      <div style={{ display: "flex", gap: 16, overflowX: "auto" }}>
        {acc.filter((_, i) => i % 2 === 0).map(lap => (
          <div key={lap.lap} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, marginBottom: 4 }}>Lap {lap.lap}</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 2 }}>
              {lap.segments.map(seg => {
                const norm = Math.max(0, Math.min(1, (seg.temp - 25) / 40));
                const hue = 120 - norm * 120; // green→red
                return (
                  <div key={seg.seg} style={{
                    width: 16, height: 16, borderRadius: 2,
                    background: `hsla(${hue}, 70%, 45%, 0.8)`,
                    fontSize: 5, color: C.w, display: "flex", alignItems: "center", justifyContent: "center",
                    fontFamily: C.dt,
                  }} title={`Seg${seg.seg}: ${seg.temp}°C / ${seg.voltage}V`}>
                    {seg.seg}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Internal Resistance vs Lap">
      <GC><ResponsiveContainer width="100%" height={180}>
        <LineChart data={acc} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={["auto", "auto"]} tickFormatter={v => `${(v * 1000).toFixed(0)}mΩ`} />
          <Tooltip contentStyle={TT} formatter={v => `${(v * 1000).toFixed(1)} mΩ`} />
          <Line type="monotone" dataKey="Ri" stroke={C.am} strokeWidth={1.5} dot={{ r: 2, fill: C.am }} name="R_i" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Power Limit Envelope [kW]">
      <GC><ResponsiveContainer width="100%" height={180}>
        <AreaChart data={acc} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="lap" {...ax()} />
          <YAxis {...ax()} domain={[0, 85]} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={80} stroke={C.dm} strokeDasharray="4 4" label={{ value: "80kW FSG", fill: C.dm, fontSize: 7 }} />
          <Area type="monotone" dataKey="powerLimit" stroke={ELEC} fill={ELEC_G} strokeWidth={1.5} dot={false} name="P_max [kW]" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 2: INVERTER & MOTOR
// ═══════════════════════════════════════════════════════════════════════════
function InverterTab() {
const effMap = useMemo(() => gMotorEffMap(), []);
const thermal = useMemo(() => gInvMotorThermal(), []);
const maxMotT = Math.max(…thermal.map(t => Math.max(t.motL, t.motR)));
const maxInvT = Math.max(…thermal.map(t => Math.max(t.invL, t.invR)));
const peakDerate = Math.max(…thermal.map(t => t.derating));
const peakEff = Math.max(…effMap.map(e => e.eff));

// Operating trace — simulated lap
const R = srng(5150);
const opTrace = Array.from({ length: 60 }, (_, i) => ({
torque: +(100 + 200 * Math.sin(i * 0.1) + 50 * R()).toFixed(0),
speed: +(1500 + 2000 * Math.sin(i * 0.08) + 500 * R()).toFixed(0),
eff: +(0.85 + 0.08 * R()).toFixed(3),
}));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Peak Motor T" value={`${maxMotT.toFixed(0)}°C`} sub="winding temp" sentiment={maxMotT < 130 ? "positive" : maxMotT < 145 ? "amber" : "negative"} delay={0} />
<KPI label="Peak Inv T" value={`${maxInvT.toFixed(0)}°C`} sub="IGBT junction" sentiment={maxInvT < 100 ? "positive" : "amber"} delay={1} />
<KPI label="Peak Derate" value={`${peakDerate.toFixed(0)}%`} sub="power reduction" sentiment={peakDerate < 5 ? "positive" : "amber"} delay={2} />
<KPI label="Peak η" value={`${(peakEff * 100).toFixed(1)}%`} sub="motor efficiency" sentiment="positive" delay={3} />
<KPI label="Peak Power" value="80 kW" sub="FSG limit" sentiment="neutral" delay={4} />
</div>


  {/* Efficiency map as scatter heatmap */}
  <Sec title="Motor Efficiency Map — Torque × Speed [%]">
    <GC style={{ padding: 8 }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <div>
          <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, textAlign: "center", marginBottom: 4 }}>RPM →</div>
          <div style={{ display: "flex" }}>
            <div style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: 7, color: C.dm, fontFamily: C.dt, marginRight: 4 }}>← Torque [Nm]</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(20, 1fr)", gap: 1 }}>
              {effMap.map((e, i) => {
                const norm = (e.eff - 0.3) / 0.66;
                const hue = norm * 120; // red→green
                const isOp = opTrace.some(o => Math.abs(o.torque - e.torque) < 25 && Math.abs(o.speed - e.speed) < 200);
                return (
                  <div key={i} style={{
                    width: 14, height: 14, borderRadius: 1,
                    background: e.power > 80 ? `rgba(50,50,50,0.3)` : `hsla(${hue}, 70%, 40%, 0.8)`,
                    border: isOp ? `1px solid ${C.w}` : "none",
                  }} title={`T=${e.torque}Nm, N=${e.speed}rpm → η=${(e.eff * 100).toFixed(1)}%`} />
                );
              })}
            </div>
          </div>
        </div>
        <div style={{ fontSize: 7, fontFamily: C.dt, color: C.dm }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>η [%]</div>
          {[95, 90, 80, 70, 50].map(v => (
            <div key={v} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
              <div style={{ width: 10, height: 10, borderRadius: 1, background: `hsla(${((v - 30) / 66) * 120}, 70%, 40%, 0.8)` }} />
              <span>{v}%</span>
            </div>
          ))}
          <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 10, height: 10, borderRadius: 1, background: "rgba(50,50,50,0.3)" }} />
            <span>&gt;80kW</span>
          </div>
        </div>
      </div>
    </GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Motor Temperature [°C]">
      <GC><ResponsiveContainer width="100%" height={220}>
        <LineChart data={thermal.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={130} y2={160} fill={C.red} fillOpacity={0.05} label={{ value: "DERATE", fill: C.red, fontSize: 7 }} />
          <Line type="monotone" dataKey="motL" stroke={C.cy} strokeWidth={1.5} dot={false} name="Motor L" />
          <Line type="monotone" dataKey="motR" stroke={C.am} strokeWidth={1.5} dot={false} name="Motor R" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Inverter Temperature [°C]">
      <GC><ResponsiveContainer width="100%" height={220}>
        <LineChart data={thermal.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={95} y2={120} fill={C.am} fillOpacity={0.05} label={{ value: "LIMIT", fill: C.am, fontSize: 7 }} />
          <Line type="monotone" dataKey="invL" stroke={C.gn} strokeWidth={1.5} dot={false} name="Inv L" />
          <Line type="monotone" dataKey="invR" stroke={ELEC} strokeWidth={1.5} dot={false} name="Inv R" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 3: TORQUE VECTORING
// ═══════════════════════════════════════════════════════════════════════════
function TVTab() {
const tv = useMemo(() => gTVAlloc(), []);
const rmseTV = Math.sqrt(tv.reduce((a, t) => a + t.tvError * t.tvError, 0) / tv.length);
const maxErr = Math.max(…tv.map(t => Math.abs(t.tvError)));
const trackingRate = tv.filter(t => Math.abs(t.tvError) < 5).length / tv.length * 100;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="TV RMSE" value={`${rmseTV.toFixed(1)} Nm`} sub="yaw moment error" sentiment={rmseTV < 5 ? "positive" : "amber"} delay={0} />
<KPI label="Max |Error|" value={`${maxErr.toFixed(1)} Nm`} sub="worst-case" sentiment={maxErr < 15 ? "positive" : "amber"} delay={1} />
<KPI label="Tracking" value={`${trackingRate.toFixed(0)}%`} sub="within ±5 Nm" sentiment={trackingRate > 90 ? "positive" : "amber"} delay={2} />
<KPI label="Allocation" value="Convex" sub="global optimum guarantee" sentiment="positive" delay={3} />
</div>


  <Sec title="Yaw Moment — Demand vs Actual [Nm]">
    <GC><ResponsiveContainer width="100%" height={240}>
      <LineChart data={tv.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="t" {...ax()} />
        <YAxis {...ax()} />
        <Tooltip contentStyle={TT} />
        <Line type="monotone" dataKey="yawDemand" stroke={ELEC} strokeWidth={2} dot={false} name="Demand" />
        <Line type="monotone" dataKey="yawActual" stroke={C.cy} strokeWidth={1.5} dot={false} name="Actual" strokeDasharray="4 2" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </LineChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="TV Error [Nm]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={tv.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0} stroke={C.dm} />
          <ReferenceArea y1={-5} y2={5} fill={C.gn} fillOpacity={0.04} />
          <Area type="monotone" dataKey="tvError" stroke={C.red} fill={`${C.red}08`} strokeWidth={1.5} dot={false} name="Error" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Per-Wheel Torque [Nm]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <LineChart data={tv.filter((_, i) => i % 3 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="tRL" stroke={C.am} strokeWidth={1.5} dot={false} name="T_RL" />
          <Line type="monotone" dataKey="tRR" stroke={C.cy} strokeWidth={1.5} dot={false} name="T_RR" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 4: REGEN STRATEGY
// ═══════════════════════════════════════════════════════════════════════════
function RegenTab() {
const regen = useMemo(() => gRegenData(), []);
const totalRecov = regen[regen.length - 1].cumRecovered;
const totalConsumed = regen[regen.length - 1].cumConsumed;
const peakRegen = Math.max(…regen.map(r => r.regenPower));
const maxCRate = Math.max(…regen.map(r => Math.abs(r.cRate)));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Total Recovered" value={`${(totalRecov * 1000).toFixed(0)} Wh`} sub="cumulative regen" sentiment="positive" delay={0} />
<KPI label="Recovery Rate" value={`${(totalRecov / (totalConsumed || 1) * 100).toFixed(1)}%`} sub="vs consumption" sentiment={totalRecov / totalConsumed > 0.08 ? "positive" : "amber"} delay={1} />
<KPI label="Peak Regen" value={`${peakRegen.toFixed(1)} kW`} sub="instantaneous" sentiment="neutral" delay={2} />
<KPI label="Max C-Rate" value={`${maxCRate.toFixed(2)}C`} sub="charge rate" sentiment={maxCRate < 3 ? "positive" : "amber"} delay={3} />
</div>


  <Sec title="Power Flow — Consumption vs Regen [kW]">
    <GC><ResponsiveContainer width="100%" height={240}>
      <ComposedChart data={regen.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="t" {...ax()} />
        <YAxis {...ax()} />
        <Tooltip contentStyle={TT} />
        <ReferenceLine y={0} stroke={C.dm} />
        <Area type="monotone" dataKey="consumePower" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.2} dot={false} name="Consumed [kW]" />
        <Area type="monotone" dataKey="regenPower" stroke={C.gn} fill={`${C.gn}10`} strokeWidth={1.5} dot={false} name="Regen [kW]" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </ComposedChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Cumulative Energy Balance [kWh]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <LineChart data={regen.filter((_, i) => i % 3 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="cumRecovered" stroke={C.gn} strokeWidth={1.5} dot={false} name="Recovered" />
          <Line type="monotone" dataKey="cumConsumed" stroke={C.red} strokeWidth={1.5} dot={false} name="Consumed" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="C-Rate During Regen">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={regen.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={3} y2={5} fill={C.red} fillOpacity={0.05} label={{ value: "C-RATE LIMIT", fill: C.red, fontSize: 7 }} />
          <Area type="monotone" dataKey="cRate" stroke={C.am} fill={`${C.am}08`} strokeWidth={1.2} dot={false} name="C-Rate" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 5: SAFETY CIRCUITS
// ═══════════════════════════════════════════════════════════════════════════
function SafetyTab() {
const { circuits, faultTimeline } = useMemo(() => gSafetyState(), []);

const stateColor = (s) => s === "OK" || s === "ACTIVE" || s === "ARMED" ? C.gn : s === "FAULT" ? C.red : C.am;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Shutdown Loop" value="CLOSED" sub="all interlocks OK" sentiment="positive" delay={0} />
<KPI label="Active Faults" value="0" sub="no active faults" sentiment="positive" delay={1} />
<KPI label="Total Trips" value={circuits.reduce((a, c) => a + c.faults, 0).toString()} sub="lifetime" sentiment="neutral" delay={2} />
</div>


  {/* Circuit status cards */}
  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
    {circuits.map(c => (
      <div key={c.id} style={{ ...GL, padding: "12px 14px", borderTop: `2px solid ${stateColor(c.state)}` }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 800, color: stateColor(c.state), fontFamily: C.dt }}>{c.id}</div>
            <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt }}>{c.name}</div>
          </div>
          <div style={{
            width: 10, height: 10, borderRadius: "50%",
            background: stateColor(c.state),
            boxShadow: `0 0 8px ${stateColor(c.state)}`,
            animation: "pulseGlow 2s infinite",
          }} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 8, fontFamily: C.dt }}>
          <div><span style={{ color: C.dm }}>Rule: </span><span style={{ color: C.br }}>{c.rule}</span></div>
          <div><span style={{ color: C.dm }}>State: </span><span style={{ color: stateColor(c.state), fontWeight: 700 }}>{c.state}</span></div>
          <div><span style={{ color: C.dm }}>Threshold: </span><span style={{ color: C.br }}>{c.threshold}</span></div>
          <div><span style={{ color: C.dm }}>Reading: </span><span style={{ color: C.br }}>{c.reading} {c.unit}</span></div>
          <div><span style={{ color: C.dm }}>Faults: </span><span style={{ color: c.faults > 0 ? C.am : C.gn }}>{c.faults}</span></div>
          <div><span style={{ color: C.dm }}>Last: </span><span style={{ color: C.dm }}>{c.lastFault || "—"}</span></div>
        </div>
      </div>
    ))}
  </div>

  {/* Fault timeline */}
  <Sec title="Fault / Event Timeline">
    <GC style={{ padding: 10 }}>
      {faultTimeline.map((f, i) => (
        <div key={i} style={{ display: "flex", gap: 10, alignItems: "center", padding: "6px 0", borderBottom: `1px solid ${C.b1}08` }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: f.severity === "warning" ? C.am : C.gn, flexShrink: 0 }} />
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, width: 50 }}>{f.t >= 0 ? `+${f.t}s` : `${f.t}s`}</div>
          <div style={{ fontSize: 9, color: ELEC, fontFamily: C.dt, fontWeight: 700, width: 50 }}>{f.circuit}</div>
          <div style={{ fontSize: 8, color: C.br, fontFamily: C.dt, flex: 1 }}>{f.event}</div>
          <div style={{ fontSize: 7, color: f.resolved ? C.gn : C.red, fontFamily: C.dt }}>{f.resolved ? "RESOLVED" : "ACTIVE"}</div>
        </div>
      ))}
    </GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 6: CAN BUS HEALTH
// ═══════════════════════════════════════════════════════════════════════════
function CANTab() {
const { timeline, nodeHealth } = useMemo(() => gCANHealth(), []);
const avgBusLoad = timeline.reduce((a, t) => a + t.busLoad, 0) / timeline.length;
const totalErrors = timeline.reduce((a, t) => a + t.errorFrames, 0);
const avgLatP99 = timeline.reduce((a, t) => a + t.latencyP99, 0) / timeline.length;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Avg Bus Load" value={`${avgBusLoad.toFixed(1)}%`} sub="CAN utilization" sentiment={avgBusLoad < 60 ? "positive" : "amber"} delay={0} />
<KPI label="Error Frames" value={totalErrors.toString()} sub="total in session" sentiment={totalErrors < 5 ? "positive" : "amber"} delay={1} />
<KPI label="Avg P99 Latency" value={`${avgLatP99.toFixed(1)} ms`} sub="99th percentile" sentiment={avgLatP99 < 5 ? "positive" : "amber"} delay={2} />
<KPI label="Active Nodes" value={nodeHealth.filter(n => n.status === "OK").length + "/" + nodeHealth.length} sub="heartbeat OK" sentiment="positive" delay={3} />
</div>


  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="Bus Load Over Time [%]">
      <GC><ResponsiveContainer width="100%" height={220}>
        <AreaChart data={timeline.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} domain={[0, 100]} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={70} y2={100} fill={C.red} fillOpacity={0.05} label={{ value: "HIGH LOAD", fill: C.red, fontSize: 7 }} />
          <Area type="monotone" dataKey="busLoad" stroke={ELEC} fill={ELEC_G} strokeWidth={1.5} dot={false} name="Bus Load %" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Latency — P50 & P99 [ms]">
      <GC><ResponsiveContainer width="100%" height={220}>
        <LineChart data={timeline.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="latencyP50" stroke={C.gn} strokeWidth={1.5} dot={false} name="P50" />
          <Line type="monotone" dataKey="latencyP99" stroke={C.red} strokeWidth={1.5} dot={false} name="P99" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  {/* Node heartbeat matrix */}
  <Sec title="Node Heartbeat Matrix" style={{ marginTop: 10 }}>
    <GC style={{ padding: 10 }}>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${nodeHealth.length}, 1fr)`, gap: 8 }}>
        {nodeHealth.map(n => (
          <div key={n.node} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: n.status === "OK" ? C.gn : C.red, fontFamily: C.dt }}>{n.node}</div>
            <div style={{
              width: 32, height: 32, borderRadius: "50%", margin: "6px auto",
              background: n.status === "OK" ? `${C.gn}20` : `${C.red}20`,
              border: `2px solid ${n.status === "OK" ? C.gn : C.red}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 8, color: n.status === "OK" ? C.gn : C.red, fontFamily: C.dt, fontWeight: 700,
            }}>
              {n.status === "OK" ? "✓" : "✕"}
            </div>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>{n.actualHz} Hz</div>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>exp: {n.expectedHz}</div>
            {n.timeouts > 0 && <div style={{ fontSize: 7, color: C.am, fontFamily: C.dt }}>{n.timeouts} timeouts</div>}
          </div>
        ))}
      </div>
    </GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 7: SENSOR FUSION
// ═══════════════════════════════════════════════════════════════════════════
function SensorsTab() {
const sensors = useMemo(() => gSensorHealth(), []);
const avgConf = sensors.reduce((a, s) => a + s.confidence, 0) / sensors.length;
const faults = sensors.filter(s => s.sensorFault).length;
const maxInnov = Math.max(…sensors.map(s => Math.abs(s.innovAx)));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Avg Confidence" value={`${avgConf.toFixed(1)}%`} sub="EKF state estimation" sentiment={avgConf > 90 ? "positive" : "amber"} delay={0} />
<KPI label="Sensor Faults" value={faults.toString()} sub="detected anomalies" sentiment={faults < 3 ? "positive" : "amber"} delay={1} />
<KPI label="Max Innovation" value={maxInnov.toFixed(3)} sub="EKF a_x innovation" sentiment={maxInnov < 0.05 ? "positive" : "amber"} delay={2} />
<KPI label="GPS HDOP" value={sensors[sensors.length - 1].gpsHDOP.toFixed(1)} sub="satellite geometry" sentiment={sensors[sensors.length - 1].gpsHDOP < 2 ? "positive" : "amber"} delay={3} />
</div>


  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    <Sec title="EKF Innovations — a_x, ω_z, v_y">
      <GC><ResponsiveContainer width="100%" height={220}>
        <LineChart data={sensors.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceLine y={0} stroke={C.dm} />
          <Line type="monotone" dataKey="innovAx" stroke={C.cy} strokeWidth={1.2} dot={false} name="innov a_x" />
          <Line type="monotone" dataKey="innovWz" stroke={C.am} strokeWidth={1.2} dot={false} name="innov ω_z" />
          <Line type="monotone" dataKey="innovVy" stroke={C.gn} strokeWidth={1.2} dot={false} name="innov v_y" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="State Estimation Confidence [%]">
      <GC><ResponsiveContainer width="100%" height={220}>
        <AreaChart data={sensors.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} domain={[70, 100]} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={70} y2={85} fill={C.am} fillOpacity={0.04} />
          <Area type="monotone" dataKey="confidence" stroke={ELEC} fill={ELEC_G} strokeWidth={1.5} dot={false} name="Confidence %" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="IMU Bias Drift — a_x, ω_z">
      <GC><ResponsiveContainer width="100%" height={180}>
        <LineChart data={sensors.filter((_, i) => i % 3 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Line type="monotone" dataKey="imuBiasAx" stroke={C.cy} strokeWidth={1.2} dot={false} name="Bias a_x" />
          <Line type="monotone" dataKey="imuBiasGz" stroke={C.am} strokeWidth={1.2} dot={false} name="Bias ω_z" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="GPS HDOP (satellite quality)">
      <GC><ResponsiveContainer width="100%" height={180}>
        <AreaChart data={sensors.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={2.5} y2={6} fill={C.red} fillOpacity={0.05} label={{ value: "DEGRADED", fill: C.red, fontSize: 7 }} />
          <Area type="monotone" dataKey="gpsHDOP" stroke={C.gn} fill={`${C.gn}08`} strokeWidth={1.2} dot={false} name="HDOP" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 8: THERMAL MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════
function ThermalMgmtTab() {
const coolant = useMemo(() => gCoolantLoop(), []);
const minMargin = Math.min(…coolant.map(c => c.thermalMargin));
const maxMotJ = Math.max(…coolant.map(c => c.motJacket));
const maxPump = Math.max(…coolant.map(c => c.pumpDuty));

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Thermal Margin" value={`${minMargin.toFixed(0)}°C`} sub="time to derate" sentiment={minMargin > 20 ? "positive" : minMargin > 5 ? "amber" : "negative"} delay={0} />
<KPI label="Peak Motor Jacket" value={`${maxMotJ.toFixed(0)}°C`} sub="coolant node" sentiment={maxMotJ < 80 ? "positive" : "amber"} delay={1} />
<KPI label="Max Pump Duty" value={`${maxPump.toFixed(0)}%`} sub="coolant pump" sentiment={maxPump < 90 ? "positive" : "amber"} delay={2} />
<KPI label="ΔT Radiator" value={`${(coolant[coolant.length - 1].radIn - coolant[coolant.length - 1].radOut).toFixed(1)}°C`} sub="cooling capacity" sentiment="neutral" delay={3} />
</div>


  <Sec title="Coolant Loop Temperatures [°C]">
    <GC><ResponsiveContainer width="100%" height={260}>
      <LineChart data={coolant.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GS} />
        <XAxis dataKey="t" {...ax()} label={{ value: "Time [s]", position: "insideBottom", offset: -2, fill: C.dm, fontSize: 7 }} />
        <YAxis {...ax()} />
        <Tooltip contentStyle={TT} />
        <Line type="monotone" dataKey="radIn" stroke={C.red} strokeWidth={1.5} dot={false} name="Rad In (hot)" />
        <Line type="monotone" dataKey="radOut" stroke={C.cy} strokeWidth={1.5} dot={false} name="Rad Out (cool)" />
        <Line type="monotone" dataKey="motJacket" stroke={C.am} strokeWidth={1.5} dot={false} name="Motor Jacket" />
        <Line type="monotone" dataKey="invPlate" stroke={ELEC} strokeWidth={1.5} dot={false} name="Inv Plate" />
        <Line type="monotone" dataKey="battPlate" stroke={C.gn} strokeWidth={1.5} dot={false} name="Batt Plate" />
        <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
      </LineChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Pump Duty & Fan Speed">
      <GC><ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={coolant.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis yAxisId="d" {...ax()} domain={[0, 100]} />
          <YAxis yAxisId="f" orientation="right" {...ax()} />
          <Tooltip contentStyle={TT} />
          <Area yAxisId="d" type="monotone" dataKey="pumpDuty" stroke={C.cy} fill={`${C.cy}08`} strokeWidth={1.5} dot={false} name="Pump [%]" />
          <Line yAxisId="f" type="monotone" dataKey="fanSpeed" stroke={C.am} strokeWidth={1.2} dot={false} name="Fan [RPM]" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </ComposedChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Thermal Margin to Derate [°C]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={coolant.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} domain={[0, "auto"]} />
          <Tooltip contentStyle={TT} />
          <ReferenceArea y1={0} y2={10} fill={C.red} fillOpacity={0.06} label={{ value: "CRITICAL", fill: C.red, fontSize: 7 }} />
          <Area type="monotone" dataKey="thermalMargin" stroke={C.gn} fill={`${C.gn}08`} strokeWidth={1.5} dot={false} name="Margin [°C]" />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 9: POWER BUDGET
// ═══════════════════════════════════════════════════════════════════════════
function PowerBudgetTab() {
const pb = useMemo(() => gPowerBudget(), []);
const avgEff = pb.reduce((a, p) => a + p.totalEff, 0) / pb.length;
const avgInvLoss = pb.reduce((a, p) => a + p.invLoss, 0) / pb.length;
const avgMotCu = pb.reduce((a, p) => a + p.motorCopper, 0) / pb.length;
const avgMotFe = pb.reduce((a, p) => a + p.motorIron, 0) / pb.length;

// Sankey-style breakdown (average values)
const avgBatt = pb.reduce((a, p) => a + p.battOut, 0) / pb.length;
const avgMech = pb.reduce((a, p) => a + p.mechOut, 0) / pb.length;
const avgLV = pb.reduce((a, p) => a + p.lvDraw, 0) / pb.length;
const sankeyData = [
{ stage: "Battery Out", value: +avgBatt.toFixed(1), color: ELEC },
{ stage: "Inverter Loss", value: +avgInvLoss.toFixed(2), color: C.red },
{ stage: "Motor Cu Loss", value: +avgMotCu.toFixed(2), color: C.am },
{ stage: "Motor Fe Loss", value: +avgMotFe.toFixed(2), color: "#ff8c00" },
{ stage: "LV System", value: +avgLV.toFixed(2), color: C.dm },
{ stage: "Mech Output", value: +avgMech.toFixed(1), color: C.gn },
];

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Avg η_total" value={`${(avgEff * 100).toFixed(1)}%`} sub="battery→wheel" sentiment={avgEff > 0.88 ? "positive" : "amber"} delay={0} />
<KPI label="Inv Loss" value={`${avgInvLoss.toFixed(1)} kW avg`} sub="switching + conduction" sentiment="neutral" delay={1} />
<KPI label="Motor Loss" value={`${(avgMotCu + avgMotFe).toFixed(1)} kW avg`} sub="copper + iron" sentiment="neutral" delay={2} />
<KPI label="LV Draw" value={`${(avgLV * 1000).toFixed(0)} W`} sub="12V subsystem" sentiment={avgLV < 0.25 ? "positive" : "amber"} delay={3} />
</div>


  {/* Stacked loss waterfall */}
  <Sec title="Power Flow Breakdown [kW avg]">
    <GC><ResponsiveContainer width="100%" height={220}>
      <BarChart data={sankeyData} margin={{ top: 8, right: 16, bottom: 8, left: 70 }} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
        <XAxis type="number" {...ax()} />
        <YAxis dataKey="stage" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={65} />
        <Tooltip contentStyle={TT} />
        <Bar dataKey="value" barSize={14} radius={[0, 4, 4, 0]}>
          {sankeyData.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer></GC>
  </Sec>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
    <Sec title="Efficiency Chain Over Time">
      <GC><ResponsiveContainer width="100%" height={200}>
        <LineChart data={pb.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} domain={[0.7, 1.0]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
          <Tooltip contentStyle={TT} formatter={v => `${(v * 100).toFixed(1)}%`} />
          <Line type="monotone" dataKey="totalEff" stroke={ELEC} strokeWidth={2} dot={false} name="η_total" />
        </LineChart>
      </ResponsiveContainer></GC>
    </Sec>

    <Sec title="Loss Breakdown Over Time [kW]">
      <GC><ResponsiveContainer width="100%" height={200}>
        <AreaChart data={pb.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} />
          <XAxis dataKey="t" {...ax()} />
          <YAxis {...ax()} />
          <Tooltip contentStyle={TT} />
          <Area type="monotone" dataKey="invLoss" stackId="1" fill={C.red} fillOpacity={0.3} stroke={C.red} strokeWidth={1} name="Inv Loss" />
          <Area type="monotone" dataKey="motorCopper" stackId="1" fill={C.am} fillOpacity={0.25} stroke={C.am} strokeWidth={1} name="Cu Loss" />
          <Area type="monotone" dataKey="motorIron" stackId="1" fill="#ff8c00" fillOpacity={0.2} stroke="#ff8c00" strokeWidth={1} name="Fe Loss" />
          <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
        </AreaChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function ElectronicsModule() {
const [tab, setTab] = useState("accumulator");

return (
<div>
{/* Header banner */}
<div style={{
…GL, padding: "12px 16px", marginBottom: 14,
borderLeft: `3px solid ${ELEC}`,
background: `linear-gradient(90deg, ${ELEC}08, transparent)`,
}}>
<div style={{ display: "flex", alignItems: "center", gap: 10 }}>
<span style={{ fontSize: 20, color: ELEC }}>⚡</span>
<div>
<span style={{ fontSize: 12, fontWeight: 800, color: ELEC, fontFamily: C.dt, letterSpacing: 2 }}>
ELECTRONICS & POWERTRAIN
</span>
<span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 12 }}>
HV system monitoring — accumulator, inverters, motors, torque vectoring, safety circuits
</span>
</div>
</div>
</div>


  <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
    {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={ELEC} />)}
  </div>

  {tab === "accumulator" && <AccumulatorTab />}
  {tab === "inverter" && <InverterTab />}
  {tab === "tv" && <TVTab />}
  {tab === "regen" && <RegenTab />}
  {tab === "safety" && <SafetyTab />}
  {tab === "can" && <CANTab />}
  {tab === "sensors" && <SensorsTab />}
  {tab === "thermal" && <ThermalMgmtTab />}
  {tab === "power" && <PowerBudgetTab />}
</div>


);
}