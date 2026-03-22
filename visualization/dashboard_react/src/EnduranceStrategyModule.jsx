// ═══════════════════════════════════════════════════════════════════════════
// src/EnduranceStrategyModule.jsx — Project-GP Dashboard v4.2
// ═══════════════════════════════════════════════════════════════════════════
// Endurance event strategy: energy management, thermal limits, tire wear.
// FSE = 22 km, ~16 laps of ~1.37 km. This module answers: "Will we finish?"
//
// Integration (3 lines in App.jsx):
//   NAV: { key: "endurance", label: "Endurance", icon: "⏱" }
//   Import: import EnduranceStrategyModule from "./EnduranceStrategyModule.jsx"
//   Route: case "endurance": return <EnduranceStrategyModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, ComposedChart, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATORS
// ═══════════════════════════════════════════════════════════════════════════
function seededRng(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; };
}

// Battery / accumulator model: 6.5 kWh pack, 600V nominal
function gBatteryTrace(nLaps = 16) {
  const R = seededRng(201);
  const data = [];
  let soc = 100, voltage = 588, energy = 6.5;
  const lapDist = 1.37; // km
  for (let lap = 0; lap <= nLaps; lap++) {
    const dist = lap * lapDist;
    const lapEnergy = 0.35 + R() * 0.08; // kWh per lap
    const regenRecov = 0.04 + R() * 0.02;
    const netEnergy = lapEnergy - regenRecov;
    energy = Math.max(0, energy - netEnergy);
    soc = (energy / 6.5) * 100;
    voltage = 540 + soc * 0.48 - (lap > 10 ? (lap - 10) * 3 : 0); // voltage sag
    const current = 80 + R() * 40 + (soc < 20 ? 20 : 0);
    const cellTemp = 32 + lap * 2.1 + R() * 3;
    data.push({
      lap, dist: +dist.toFixed(2), soc: +soc.toFixed(1),
      voltage: +voltage.toFixed(0), current: +current.toFixed(0),
      energy: +energy.toFixed(2), cellTemp: +cellTemp.toFixed(1),
      lapEnergy: +lapEnergy.toFixed(3), regenRecov: +regenRecov.toFixed(3),
      netEnergy: +netEnergy.toFixed(3),
    });
  }
  return data;
}

// Motor + inverter thermal model
function gPowertrainThermal(nLaps = 16) {
  const R = seededRng(301);
  const data = [];
  let motorL = 55, motorR = 54, invL = 42, invR = 43;
  for (let lap = 0; lap <= nLaps; lap++) {
    const intensity = 0.7 + 0.3 * Math.sin(lap * 0.5) + R() * 0.15;
    motorL = Math.min(160, motorL + intensity * 4.5 - 2.8 + R() * 1.5);
    motorR = Math.min(160, motorR + intensity * 4.2 - 2.6 + R() * 1.5);
    invL = Math.min(120, invL + intensity * 2.8 - 2.0 + R() * 1.0);
    invR = Math.min(120, invR + intensity * 2.6 - 1.8 + R() * 1.0);
    const derating = motorL > 130 || motorR > 130 ? Math.min(40, (Math.max(motorL, motorR) - 130) * 2.5) : 0;
    data.push({
      lap, motorL: +motorL.toFixed(1), motorR: +motorR.toFixed(1),
      invL: +invL.toFixed(1), invR: +invR.toFixed(1),
      derating: +derating.toFixed(1),
    });
  }
  return data;
}

// Brake thermal model (4 corners)
function gBrakeThermal(nLaps = 16) {
  const R = seededRng(401);
  const data = [];
  let temps = [180, 185, 160, 165]; // FL, FR, RL, RR
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
      maxTemp: +Math.max(...temps).toFixed(0),
    });
  }
  return data;
}

// Tire degradation model
function gTireDeg(nLaps = 16) {
  const R = seededRng(501);
  const data = [];
  let grip = [100, 100, 100, 100];
  for (let lap = 0; lap <= nLaps; lap++) {
    for (let c = 0; c < 4; c++) {
      const wear = (c < 2 ? 0.55 : 0.45) + R() * 0.15; // front wears faster
      grip[c] = Math.max(75, grip[c] - wear);
    }
    const avgGrip = grip.reduce((a, g) => a + g, 0) / 4;
    const lapTime = 62.5 + (100 - avgGrip) * 0.15 + R() * 0.3;
    data.push({
      lap, FL: +grip[0].toFixed(1), FR: +grip[1].toFixed(1),
      RL: +grip[2].toFixed(1), RR: +grip[3].toFixed(1),
      avgGrip: +avgGrip.toFixed(1), lapTime: +lapTime.toFixed(2),
    });
  }
  return data;
}

// Regen efficiency curve
function gRegenCurve(n = 50) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const speed = i * 0.6; // 0-30 m/s
    const maxRegen = Math.min(20, speed * 1.2); // kW, limited by motor speed
    const efficiency = speed < 3 ? 0 : Math.min(92, 60 + 25 * (1 - Math.exp(-speed / 8)));
    data.push({
      speed: +speed.toFixed(1),
      maxRegen: +maxRegen.toFixed(1),
      efficiency: +efficiency.toFixed(1),
    });
  }
  return data;
}

// Torque vectoring map
function gTVMap(nLaps = 16) {
  const R = seededRng(601);
  const data = [];
  for (let lap = 0; lap <= nLaps; lap++) {
    for (let i = 0; i < 20; i++) {
      const t = lap + i / 20;
      const ay = 1.5 * Math.sin(t * 3.5) + R() * 0.3;
      const yawTarget = ay * 85; // Nm target yaw moment
      const yawActual = yawTarget * (0.88 + R() * 0.15);
      data.push({
        t: +t.toFixed(2), ay: +ay.toFixed(2),
        yawTarget: +yawTarget.toFixed(0), yawActual: +yawActual.toFixed(0),
        tvError: +(yawActual - yawTarget).toFixed(1),
      });
    }
  }
  return data;
}

// ═══════════════════════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════════════════════
const TABS = [
  { key: "energy", label: "Energy Budget" },
  { key: "thermal", label: "Thermal Limits" },
  { key: "brakes", label: "Brake Thermal" },
  { key: "tires", label: "Tire Degradation" },
  { key: "regen", label: "Regen & TV" },
];

const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });
const CL = { fl: C.cy, fr: C.gn, rl: C.am, rr: C.red };

// ═══════════════════════════════════════════════════════════════════════════
// ENERGY BUDGET TAB
// ═══════════════════════════════════════════════════════════════════════════
function EnergyTab({ battery }) {
  const finalSoC = battery[battery.length - 1]?.soc || 0;
  const totalRegen = battery.reduce((a, b) => a + b.regenRecov, 0);
  const willFinish = finalSoC > 5;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Final SoC" value={`${finalSoC.toFixed(1)}%`} sub="end of endurance" sentiment={finalSoC > 15 ? "positive" : finalSoC > 5 ? "amber" : "negative"} delay={0} />
        <KPI label="Status" value={willFinish ? "WILL FINISH" : "DNF RISK"} sub={willFinish ? "energy sufficient" : "reduce pace!"} sentiment={willFinish ? "positive" : "negative"} delay={1} />
        <KPI label="Total Regen" value={`${totalRegen.toFixed(2)} kWh`} sub={`${(totalRegen / 6.5 * 100).toFixed(1)}% of pack`} sentiment="positive" delay={2} />
        <KPI label="Avg Consumption" value={`${(battery[1]?.netEnergy || 0.3).toFixed(2)} kWh/lap`} sub={`${((battery[1]?.netEnergy || 0.3) / 1.37 * 100).toFixed(0)} Wh/km`} sentiment="neutral" delay={3} />
        <KPI label="Min Voltage" value={`${Math.min(...battery.map(b => b.voltage))} V`} sub="cell sag under load" sentiment={Math.min(...battery.map(b => b.voltage)) > 480 ? "positive" : "amber"} delay={4} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="State of Charge Over Endurance [%]">
          <GC><ResponsiveContainer width="100%" height={220}>
            <AreaChart data={battery}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="dist" {...ax()} label={{ value: "Distance [km]", position: "bottom", fill: C.dm, fontSize: 8 }}/>
              <YAxis {...ax()} domain={[0, 105]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceArea y1={0} y2={10} fill={C.red} fillOpacity={0.06} label={{ value: "CRITICAL", fill: C.red, fontSize: 7, position: "insideTop" }}/>
              <ReferenceLine y={20} stroke={C.am} strokeDasharray="4 2"/>
              <Area dataKey="soc" stroke={C.gn} fill={`${C.gn}15`} strokeWidth={2} dot={false} name="SoC [%]"/>
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Pack Voltage & Current">
          <GC><ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={battery}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis yAxisId="v" {...ax()} label={{ value: "V", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }}/>
              <YAxis yAxisId="i" orientation="right" {...ax()} label={{ value: "A", angle: 90, position: "insideRight", fill: C.dm, fontSize: 8 }}/>
              <Tooltip contentStyle={TT}/>
              <Line yAxisId="v" dataKey="voltage" stroke={C.cy} strokeWidth={1.5} dot={false} name="Voltage [V]"/>
              <Line yAxisId="i" dataKey="current" stroke={C.am} strokeWidth={1.2} dot={false} name="Current [A]"/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </ComposedChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Energy Consumption Per Lap [kWh]">
          <GC><ResponsiveContainer width="100%" height={180}>
            <ComposedChart data={battery.slice(1)}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()}/>
              <Tooltip contentStyle={TT}/>
              <Bar dataKey="lapEnergy" fill={C.red} fillOpacity={0.5} name="Consumed" barSize={12}/>
              <Bar dataKey="regenRecov" fill={C.gn} fillOpacity={0.6} name="Regen Recovered" barSize={12}/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </ComposedChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Cell Temperature [°C]">
          <GC><ResponsiveContainer width="100%" height={180}>
            <AreaChart data={battery}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[25, 70]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceArea y1={55} y2={70} fill={C.red} fillOpacity={0.05} label={{ value: "DERATE", fill: C.red, fontSize: 7 }}/>
              <Area dataKey="cellTemp" stroke={C.am} fill={`${C.am}12`} strokeWidth={1.5} dot={{ r: 2, fill: C.am }} name="Cell Temp"/>
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// THERMAL LIMITS TAB — Motor + Inverter
// ═══════════════════════════════════════════════════════════════════════════
function ThermalTab({ thermal }) {
  const maxMotor = Math.max(...thermal.map(t => Math.max(t.motorL, t.motorR)));
  const maxInv = Math.max(...thermal.map(t => Math.max(t.invL, t.invR)));
  const peakDerate = Math.max(...thermal.map(t => t.derating));
  const deratingLap = thermal.findIndex(t => t.derating > 0);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Peak Motor" value={`${maxMotor.toFixed(0)}°C`} sub="winding temp" sentiment={maxMotor < 130 ? "positive" : maxMotor < 145 ? "amber" : "negative"} delay={0} />
        <KPI label="Peak Inverter" value={`${maxInv.toFixed(0)}°C`} sub="IGBT junction" sentiment={maxInv < 100 ? "positive" : maxInv < 110 ? "amber" : "negative"} delay={1} />
        <KPI label="Max Derating" value={`${peakDerate.toFixed(0)}%`} sub={peakDerate > 0 ? `from lap ${deratingLap}` : "none"} sentiment={peakDerate === 0 ? "positive" : "negative"} delay={2} />
        <KPI label="Thermal Margin" value={`${Math.max(0, 150 - maxMotor).toFixed(0)}°C`} sub="to hard cutoff" sentiment={150 - maxMotor > 20 ? "positive" : "negative"} delay={3} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Motor Winding Temperature [°C]">
          <GC><ResponsiveContainer width="100%" height={220}>
            <LineChart data={thermal}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[40, 165]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceArea y1={130} y2={165} fill={C.red} fillOpacity={0.06} label={{ value: "DERATE ZONE", fill: C.red, fontSize: 7, position: "insideTop" }}/>
              <ReferenceLine y={150} stroke={C.red} strokeDasharray="3 3" label={{ value: "CUTOFF", fill: C.red, fontSize: 7 }}/>
              <Line dataKey="motorL" stroke={C.cy} strokeWidth={1.5} dot={{ r: 2 }} name="Motor L"/>
              <Line dataKey="motorR" stroke={C.gn} strokeWidth={1.5} dot={{ r: 2 }} name="Motor R"/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </LineChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Inverter Temperature [°C]">
          <GC><ResponsiveContainer width="100%" height={220}>
            <LineChart data={thermal}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[30, 125]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceArea y1={100} y2={125} fill={C.am} fillOpacity={0.06}/>
              <ReferenceLine y={110} stroke={C.red} strokeDasharray="3 3"/>
              <Line dataKey="invL" stroke={C.cy} strokeWidth={1.5} dot={{ r: 2 }} name="Inv L"/>
              <Line dataKey="invR" stroke={C.gn} strokeWidth={1.5} dot={{ r: 2 }} name="Inv R"/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </LineChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Power Derating [%]">
          <GC><ResponsiveContainer width="100%" height={180}>
            <AreaChart data={thermal}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[0, 50]}/>
              <Tooltip contentStyle={TT}/>
              <Area dataKey="derating" stroke={C.red} fill={`${C.red}15`} strokeWidth={2} dot={false} name="Derating %"/>
              <ReferenceLine y={0} stroke={C.dm}/>
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// BRAKE THERMAL TAB
// ═══════════════════════════════════════════════════════════════════════════
function BrakeTab({ brakes }) {
  const maxBrake = Math.max(...brakes.map(b => b.maxTemp));
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Peak Rotor Temp" value={`${maxBrake}°C`} sub="max across corners" sentiment={maxBrake < 380 ? "positive" : maxBrake < 430 ? "amber" : "negative"} delay={0} />
        <KPI label="Fade Risk" value={maxBrake > 400 ? "HIGH" : maxBrake > 350 ? "MODERATE" : "LOW"} sub={`fade onset ~380°C`} sentiment={maxBrake < 380 ? "positive" : "negative"} delay={1} />
        <KPI label="Front Bias" value={`${((brakes[brakes.length - 1]?.FL || 0) > (brakes[brakes.length - 1]?.RL || 0)) ? "FRONT HOT" : "BALANCED"}`} sub="front vs rear loading" sentiment="neutral" delay={2} />
      </div>
      <Sec title="Brake Rotor Temperature Per Corner [°C]">
        <GC><ResponsiveContainer width="100%" height={260}>
          <LineChart data={brakes}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="lap" {...ax()} label={{ value: "Lap", position: "bottom", fill: C.dm, fontSize: 8 }}/>
            <YAxis {...ax()} domain={[100, 500]}/>
            <Tooltip contentStyle={TT}/>
            <ReferenceArea y1={380} y2={500} fill={C.red} fillOpacity={0.05} label={{ value: "FADE ZONE", fill: C.red, fontSize: 7, position: "insideTop" }}/>
            <ReferenceLine y={380} stroke={C.red} strokeDasharray="4 2"/>
            <Line dataKey="FL" stroke={CL.fl} strokeWidth={1.5} dot={{ r: 2 }} name="FL"/>
            <Line dataKey="FR" stroke={CL.fr} strokeWidth={1.5} dot={{ r: 2 }} name="FR"/>
            <Line dataKey="RL" stroke={CL.rl} strokeWidth={1.5} dot={{ r: 2 }} name="RL"/>
            <Line dataKey="RR" stroke={CL.rr} strokeWidth={1.5} dot={{ r: 2 }} name="RR"/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </LineChart>
        </ResponsiveContainer></GC>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TIRE DEGRADATION TAB
// ═══════════════════════════════════════════════════════════════════════════
function TireTab({ tireDeg }) {
  const finalGrip = tireDeg[tireDeg.length - 1]?.avgGrip || 100;
  const lapTimeDelta = (tireDeg[tireDeg.length - 1]?.lapTime || 62.5) - (tireDeg[0]?.lapTime || 62.5);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Final Grip" value={`${finalGrip.toFixed(1)}%`} sub="end of endurance" sentiment={finalGrip > 90 ? "positive" : finalGrip > 85 ? "amber" : "negative"} delay={0} />
        <KPI label="Grip Lost" value={`${(100 - finalGrip).toFixed(1)}%`} sub="total degradation" sentiment={100 - finalGrip < 10 ? "positive" : "amber"} delay={1} />
        <KPI label="Lap Time Impact" value={`+${lapTimeDelta.toFixed(2)}s`} sub="degradation cost" sentiment={lapTimeDelta < 0.8 ? "positive" : "amber"} delay={2} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Per-Corner Grip Level [%]">
          <GC><ResponsiveContainer width="100%" height={220}>
            <LineChart data={tireDeg}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[70, 102]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceLine y={90} stroke={C.am} strokeDasharray="4 2" label={{ value: "90% threshold", fill: C.am, fontSize: 7 }}/>
              <Line dataKey="FL" stroke={CL.fl} strokeWidth={1.5} dot={{ r: 2 }} name="FL"/>
              <Line dataKey="FR" stroke={CL.fr} strokeWidth={1.5} dot={{ r: 2 }} name="FR"/>
              <Line dataKey="RL" stroke={CL.rl} strokeWidth={1.5} dot={{ r: 2 }} name="RL"/>
              <Line dataKey="RR" stroke={CL.rr} strokeWidth={1.5} dot={{ r: 2 }} name="RR"/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </LineChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Projected Lap Time [s]">
          <GC><ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={tireDeg}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[61.5, 65]}/>
              <Tooltip contentStyle={TT}/>
              <Area dataKey="lapTime" stroke={C.am} fill={`${C.am}12`} strokeWidth={1.5} dot={{ r: 2 }} name="Lap Time [s]"/>
              <ReferenceLine y={tireDeg[0]?.lapTime || 62.5} stroke={C.gn} strokeDasharray="4 2" label={{ value: "Fresh tire", fill: C.gn, fontSize: 7 }}/>
            </ComposedChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// REGEN & TORQUE VECTORING TAB
// ═══════════════════════════════════════════════════════════════════════════
function RegenTVTab({ regenCurve, tvMap }) {
  const tvSparse = useMemo(() => tvMap.filter((_, i) => i % 4 === 0), [tvMap]);
  const tvError = useMemo(() => {
    const rms = Math.sqrt(tvMap.reduce((a, p) => a + p.tvError ** 2, 0) / tvMap.length);
    return rms;
  }, [tvMap]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <Sec title="Regenerative Braking Efficiency [%] vs Speed">
        <GC><ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={regenCurve}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="speed" {...ax()} label={{ value: "Speed [m/s]", position: "bottom", fill: C.dm, fontSize: 8 }}/>
            <YAxis yAxisId="eff" {...ax()} domain={[0, 100]} label={{ value: "%", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }}/>
            <YAxis yAxisId="pwr" orientation="right" {...ax()} label={{ value: "kW", angle: 90, position: "insideRight", fill: C.dm, fontSize: 8 }}/>
            <Tooltip contentStyle={TT}/>
            <Area yAxisId="eff" dataKey="efficiency" stroke={C.gn} fill={`${C.gn}12`} strokeWidth={1.5} dot={false} name="Efficiency [%]"/>
            <Line yAxisId="pwr" dataKey="maxRegen" stroke={C.cy} strokeWidth={1.5} dot={false} name="Max Regen [kW]"/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </ComposedChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="Torque Vectoring — Target vs Actual Yaw Moment [Nm]">
        <GC><ResponsiveContainer width="100%" height={220}>
          <LineChart data={tvSparse}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="t" {...ax()}/>
            <YAxis {...ax()}/>
            <Tooltip contentStyle={TT}/>
            <Line dataKey="yawTarget" stroke={C.gn} strokeWidth={1.5} dot={false} name="TV Target" opacity={0.7}/>
            <Line dataKey="yawActual" stroke={C.cy} strokeWidth={1.2} dot={false} name="TV Actual"/>
            <ReferenceLine y={0} stroke={C.dm}/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </LineChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="TV Tracking Error [Nm]">
        <GC>
          <div style={{ display: "flex", gap: 10, marginBottom: 8 }}>
            <KPI label="TV RMS Error" value={`${tvError.toFixed(1)} Nm`} sub="tracking quality" sentiment={tvError < 8 ? "positive" : "amber"} delay={0} />
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={tvSparse}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="t" {...ax()}/>
              <YAxis {...ax()}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceLine y={0} stroke={C.dm}/>
              <Area dataKey="tvError" stroke="#e879f9" fill="rgba(232,121,249,0.08)" strokeWidth={1.2} dot={false} name="ΔM_z"/>
            </AreaChart>
          </ResponsiveContainer>
        </GC>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function EnduranceStrategyModule() {
  const [tab, setTab] = useState("energy");

  const battery = useMemo(() => gBatteryTrace(), []);
  const thermal = useMemo(() => gPowertrainThermal(), []);
  const brakes = useMemo(() => gBrakeThermal(), []);
  const tireDeg = useMemo(() => gTireDeg(), []);
  const regenCurve = useMemo(() => gRegenCurve(), []);
  const tvMap = useMemo(() => gTVMap(), []);

  // Top-level endurance status
  const finalSoC = battery[battery.length - 1]?.soc || 0;
  const maxMotor = Math.max(...thermal.map(t => Math.max(t.motorL, t.motorR)));
  const maxBrake = Math.max(...brakes.map(b => b.maxTemp));
  const finalGrip = tireDeg[tireDeg.length - 1]?.avgGrip || 100;
  const status = finalSoC > 10 && maxMotor < 150 && maxBrake < 450 ? "GO" : "CAUTION";

  return (
    <div>
      {/* Status banner */}
      <div style={{
        ...GL, padding: "12px 16px", marginBottom: 14,
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
            ENDURANCE STATUS: {status}
          </span>
          <span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 16 }}>
            22 km · 16 laps · SoC {finalSoC.toFixed(0)}% · Motor {maxMotor.toFixed(0)}°C · Brake {maxBrake.toFixed(0)}°C · Grip {finalGrip.toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Tab Switcher */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {TABS.map(t => (
          <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.gn} />
        ))}
      </div>

      {tab === "energy" && <EnergyTab battery={battery} />}
      {tab === "thermal" && <ThermalTab thermal={thermal} />}
      {tab === "brakes" && <BrakeTab brakes={brakes} />}
      {tab === "tires" && <TireTab tireDeg={tireDeg} />}
      {tab === "regen" && <RegenTVTab regenCurve={regenCurve} tvMap={tvMap} />}
    </div>
  );
}