// ═══════════════════════════════════════════════════════════════════════════
// src/DriverCoachingModule.jsx — Project-GP Dashboard v4.2
// ═══════════════════════════════════════════════════════════════════════════
// New module: Driver performance analysis and coaching.
// Compares driver inputs to MPC-optimal, sector breakdown, consistency.
//
// Integration:
//   1. Add to App.jsx NAV:  { key: "coaching", label: "Coaching", icon: "◈" }
//   2. Add to App.jsx routing: case "coaching": return <DriverCoachingModule />
//   3. Import: import DriverCoachingModule from "./DriverCoachingModule.jsx"
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
  ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATORS — synthetic until real telemetry is piped in
// ═══════════════════════════════════════════════════════════════════════════

function seededRng(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; };
}

function gDriverLap(n = 300) {
  const R = seededRng(42);
  const data = [];
  for (let i = 0; i < n; i++) {
    const t = i * 0.1;       // 30s lap at 100Hz → 0.1s steps
    const dist = t * 14;     // ~14 m/s average → ~420m lap
    const phase = (i / n) * Math.PI * 4;

    // MPC-optimal reference
    const optSteer = 85 * Math.sin(phase * 0.6) + 25 * Math.sin(phase * 1.3);
    const optThrottle = Math.max(0, Math.min(100, 60 + 35 * Math.cos(phase * 0.5)));
    const optBrake = Math.max(0, Math.min(100, -60 * Math.cos(phase * 0.5) + 15 * Math.sin(phase * 1.8)));
    const optSpeed = 10 + 12 * Math.sin(phase * 0.3) + 4 * Math.cos(phase * 0.7);

    // Actual driver (delayed, smoothed, with error)
    const lag = 0.06;  // 60ms reaction delay
    const jPrev = Math.max(0, i - Math.round(lag / 0.1));
    const prevPhase = (jPrev / n) * Math.PI * 4;
    const actSteer = 85 * Math.sin(prevPhase * 0.6) + 25 * Math.sin(prevPhase * 1.3) + (R() - 0.5) * 12;
    const actThrottle = Math.max(0, Math.min(100, 55 + 35 * Math.cos(prevPhase * 0.5) + (R() - 0.5) * 8));
    const actBrake = Math.max(0, Math.min(100, -55 * Math.cos(prevPhase * 0.5) + 15 * Math.sin(prevPhase * 1.8) + (R() - 0.5) * 6));
    const actSpeed = optSpeed - 0.3 + (R() - 0.5) * 1.2;

    // Delta
    const timeDelta = (actSpeed - optSpeed) * -0.008;

    // Sector assignment (4 sectors)
    const sector = Math.floor((i / n) * 4);

    data.push({
      t: +t.toFixed(1), dist: +dist.toFixed(0), sector,
      optSteer: +optSteer.toFixed(1), actSteer: +actSteer.toFixed(1),
      optThrottle: +optThrottle.toFixed(1), actThrottle: +actThrottle.toFixed(1),
      optBrake: +optBrake.toFixed(1), actBrake: +actBrake.toFixed(1),
      optSpeed: +optSpeed.toFixed(1), actSpeed: +actSpeed.toFixed(1),
      timeDelta: +timeDelta.toFixed(3),
      steerError: +(actSteer - optSteer).toFixed(1),
      trailBrake: +(Math.max(0, actBrake) * Math.abs(actSteer) / 100).toFixed(1),
    });
  }
  return data;
}

function gSectorAnalysis(lapData) {
  const sectors = [0, 1, 2, 3].map(s => {
    const pts = lapData.filter(d => d.sector === s);
    const n = pts.length || 1;
    return {
      sector: `S${s + 1}`,
      sectorName: ["Accel Zone", "Turn 1-2", "Chicane", "Final Corner"][s],
      optTime: +(pts.reduce((a, p) => a + 0.1, 0)).toFixed(1),
      actTime: +(pts.reduce((a, p) => a + 0.1 + Math.abs(p.timeDelta), 0)).toFixed(1),
      delta: +(pts.reduce((a, p) => a + p.timeDelta, 0) * 10).toFixed(2),
      avgSpeedOpt: +(pts.reduce((a, p) => a + p.optSpeed, 0) / n).toFixed(1),
      avgSpeedAct: +(pts.reduce((a, p) => a + p.actSpeed, 0) / n).toFixed(1),
      peakAy: +(Math.max(...pts.map(p => Math.abs(p.actSteer))) / 60).toFixed(2),
      brakingScore: +(90 + (Math.random() - 0.5) * 15).toFixed(0),
      consistencyPct: +(88 + Math.random() * 10).toFixed(1),
    };
  });
  return sectors;
}

function gConsistency(nLaps = 8) {
  const R = seededRng(77);
  return Array.from({ length: nLaps }, (_, i) => {
    const baseLap = 62.5;
    return {
      lap: i + 1,
      time: +(baseLap + (R() - 0.4) * 1.8 - i * 0.08).toFixed(2),
      s1: +(15.0 + (R() - 0.5) * 0.5).toFixed(2),
      s2: +(16.5 + (R() - 0.5) * 0.6).toFixed(2),
      s3: +(15.8 + (R() - 0.5) * 0.4).toFixed(2),
      s4: +(15.2 + (R() - 0.5) * 0.35).toFixed(2),
      brakeTemp: +(280 + i * 12 + R() * 20).toFixed(0),
      tireGrip: +(100 - i * 0.8 - R() * 2).toFixed(1),
    };
  });
}

function gBrakingPoints(n = 12) {
  const R = seededRng(99);
  return Array.from({ length: n }, (_, i) => ({
    corner: `T${i + 1}`,
    optBrakeDist: +(28 + R() * 15).toFixed(1),
    actBrakeDist: +(30 + R() * 18).toFixed(1),
    entrySpeed: +(18 + R() * 8).toFixed(1),
    optEntrySpeed: +(19 + R() * 7).toFixed(1),
    apexSpeed: +(11 + R() * 5).toFixed(1),
    optApexSpeed: +(12 + R() * 4).toFixed(1),
  }));
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-TAB CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════
const TABS = [
  { key: "inputs", label: "Input Overlay" },
  { key: "sectors", label: "Sector Analysis" },
  { key: "consistency", label: "Consistency" },
  { key: "braking", label: "Braking & Corners" },
  { key: "trail", label: "Trail Braking" },
];

const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });

// ═══════════════════════════════════════════════════════════════════════════
// INPUT OVERLAY TAB
// ═══════════════════════════════════════════════════════════════════════════
function InputsTab({ lap }) {
  const sparse = useMemo(() => lap.filter((_, i) => i % 2 === 0), [lap]);
  const cumDelta = useMemo(() => {
    let cum = 0;
    return sparse.map(d => { cum += d.timeDelta; return { ...d, cumDelta: +cum.toFixed(3) }; });
  }, [sparse]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <Sec title="Steering — Optimal vs Actual [deg]">
        <GC><ResponsiveContainer width="100%" height={200}>
          <LineChart data={sparse} syncId="coach"><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="dist" {...ax()} label={{ value: "Distance [m]", position: "bottom", fill: C.dm, fontSize: 8 }}/>
            <YAxis {...ax()}/>
            <Tooltip contentStyle={TT}/>
            <Line dataKey="optSteer" stroke={C.gn} strokeWidth={2} dot={false} name="MPC Optimal" opacity={0.8}/>
            <Line dataKey="actSteer" stroke={C.cy} strokeWidth={1.5} dot={false} name="Driver Actual"/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </LineChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="Throttle & Brake [%]">
        <GC><ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={sparse} syncId="coach"><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="dist" {...ax()}/>
            <YAxis {...ax()} domain={[0, 100]}/>
            <Tooltip contentStyle={TT}/>
            <Area dataKey="optThrottle" stroke={C.gn} fill={`${C.gn}10`} strokeWidth={1.5} dot={false} name="Opt Throttle" opacity={0.6}/>
            <Line dataKey="actThrottle" stroke={C.gn} strokeWidth={1} dot={false} name="Act Throttle" strokeDasharray="3 2"/>
            <Area dataKey="optBrake" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.5} dot={false} name="Opt Brake" opacity={0.6}/>
            <Line dataKey="actBrake" stroke={C.red} strokeWidth={1} dot={false} name="Act Brake" strokeDasharray="3 2"/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </ComposedChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="Speed Profile [m/s]">
        <GC><ResponsiveContainer width="100%" height={200}>
          <LineChart data={sparse} syncId="coach"><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="dist" {...ax()}/>
            <YAxis {...ax()}/>
            <Tooltip contentStyle={TT}/>
            <Line dataKey="optSpeed" stroke={C.gn} strokeWidth={2} dot={false} name="Optimal" opacity={0.8}/>
            <Line dataKey="actSpeed" stroke={C.cy} strokeWidth={1.5} dot={false} name="Actual"/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </LineChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="Cumulative Time Delta [s]">
        <GC><ResponsiveContainer width="100%" height={200}>
          <AreaChart data={cumDelta} syncId="coach"><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="dist" {...ax()}/>
            <YAxis {...ax()}/>
            <Tooltip contentStyle={TT}/>
            <ReferenceArea y1={0} y2={1} fill={C.red} fillOpacity={0.03}/>
            <ReferenceArea y1={-1} y2={0} fill={C.gn} fillOpacity={0.03}/>
            <ReferenceLine y={0} stroke={C.dm}/>
            <Area dataKey="cumDelta" stroke={C.am} fill={`${C.am}15`} strokeWidth={1.5} dot={false} name="ΔTime"/>
          </AreaChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="Steering Error [deg]">
        <GC><ResponsiveContainer width="100%" height={180}>
          <AreaChart data={sparse} syncId="coach"><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="dist" {...ax()}/>
            <YAxis {...ax()}/>
            <Tooltip contentStyle={TT}/>
            <ReferenceLine y={0} stroke={C.dm}/>
            <Area dataKey="steerError" stroke="#e879f9" fill="rgba(232,121,249,0.08)" strokeWidth={1.2} dot={false} name="δ_error"/>
          </AreaChart>
        </ResponsiveContainer></GC>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTOR ANALYSIS TAB
// ═══════════════════════════════════════════════════════════════════════════
function SectorsTab({ sectors }) {
  return (
    <div>
      {/* Sector delta bar chart */}
      <Sec title="Sector Time Delta [s] — Green = faster than optimal estimate">
        <GC><ResponsiveContainer width="100%" height={200}>
          <BarChart data={sectors} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false}/>
            <XAxis dataKey="sector" {...ax()}/>
            <YAxis {...ax()}/>
            <Tooltip contentStyle={TT}/>
            <ReferenceLine y={0} stroke={C.dm}/>
            <Bar dataKey="delta" radius={[4, 4, 0, 0]} barSize={32}>
              {sectors.map((s, i) => <Cell key={i} fill={s.delta <= 0 ? C.gn : C.red} fillOpacity={0.7}/>)}
            </Bar>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>

      {/* Sector detail table */}
      <div style={{ ...GL, padding: "12px 14px", marginTop: 10 }}>
        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.8, color: C.dm, fontFamily: C.dt, marginBottom: 10 }}>
          SECTOR BREAKDOWN
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "80px repeat(7, 1fr)", gap: 0, fontSize: 8, fontFamily: C.dt }}>
          {/* Header */}
          {["Sector", "Name", "Opt Time", "Act Time", "Δ [s]", "Avg Speed", "Peak Ay [G]", "Consistency"].map(h => (
            <div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 1, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>
          ))}
          {/* Rows */}
          {sectors.map(s => (
            <React.Fragment key={s.sector}>
              <div style={{ color: C.cy, fontWeight: 700, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.sector}</div>
              <div style={{ color: C.br, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.sectorName}</div>
              <div style={{ color: C.dm, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.optTime}s</div>
              <div style={{ color: C.br, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.actTime}s</div>
              <div style={{ color: s.delta <= 0 ? C.gn : C.red, fontWeight: 700, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.delta > 0 ? "+" : ""}{s.delta}</div>
              <div style={{ color: C.br, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.avgSpeedAct} m/s</div>
              <div style={{ color: C.br, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.peakAy}</div>
              <div style={{ color: parseFloat(s.consistencyPct) > 93 ? C.gn : C.am, padding: "6px 4px", borderBottom: `1px solid ${C.b1}08` }}>{s.consistencyPct}%</div>
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSISTENCY TAB — lap-to-lap variation
// ═══════════════════════════════════════════════════════════════════════════
function ConsistencyTab({ laps }) {
  const bestLap = Math.min(...laps.map(l => l.time));
  const avgLap = laps.reduce((a, l) => a + l.time, 0) / laps.length;
  const stdDev = Math.sqrt(laps.reduce((a, l) => a + (l.time - avgLap) ** 2, 0) / laps.length);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Best Lap" value={`${bestLap.toFixed(2)}s`} sub="session best" sentiment="positive" delay={0} />
        <KPI label="Average" value={`${avgLap.toFixed(2)}s`} sub={`Δ best: +${(avgLap - bestLap).toFixed(2)}s`} sentiment="neutral" delay={1} />
        <KPI label="Std Dev" value={`${stdDev.toFixed(3)}s`} sub={stdDev < 0.3 ? "excellent" : stdDev < 0.6 ? "good" : "inconsistent"} sentiment={stdDev < 0.3 ? "positive" : stdDev < 0.6 ? "amber" : "negative"} delay={2} />
        <KPI label="Consistency" value={`${(100 - (stdDev / avgLap) * 100).toFixed(1)}%`} sub="σ/μ metric" sentiment="positive" delay={3} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Lap Times">
          <GC><ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={laps} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()} label={{ value: "Lap", position: "bottom", fill: C.dm, fontSize: 8 }}/>
              <YAxis {...ax()} domain={[bestLap - 0.5, bestLap + 2.5]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceLine y={bestLap} stroke={C.gn} strokeDasharray="4 2" label={{ value: "Best", fill: C.gn, fontSize: 7 }}/>
              <ReferenceLine y={avgLap} stroke={C.am} strokeDasharray="4 2" label={{ value: "Avg", fill: C.am, fontSize: 7 }}/>
              <Bar dataKey="time" radius={[4, 4, 0, 0]} barSize={24}>
                {laps.map((l, i) => <Cell key={i} fill={l.time <= bestLap + 0.3 ? C.gn : l.time <= avgLap ? C.cy : C.am} fillOpacity={0.7}/>)}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Sector Split Stacked">
          <GC><ResponsiveContainer width="100%" height={200}>
            <BarChart data={laps} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()}/>
              <Tooltip contentStyle={TT}/>
              <Bar dataKey="s1" stackId="s" fill={C.cy} fillOpacity={0.6} name="S1"/>
              <Bar dataKey="s2" stackId="s" fill={C.gn} fillOpacity={0.6} name="S2"/>
              <Bar dataKey="s3" stackId="s" fill={C.am} fillOpacity={0.6} name="S3"/>
              <Bar dataKey="s4" stackId="s" fill="#e879f9" fillOpacity={0.6} name="S4"/>
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
            </BarChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Brake Temperature Rise [°C]">
          <GC><ResponsiveContainer width="100%" height={180}>
            <AreaChart data={laps}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[200, 450]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceArea y1={380} y2={450} fill={C.red} fillOpacity={0.05} label={{ value: "FADE ZONE", fill: C.red, fontSize: 7, position: "insideTop" }}/>
              <Area dataKey="brakeTemp" stroke={C.red} fill={`${C.red}12`} strokeWidth={1.5} dot={{ r: 3, fill: C.red }} name="Rotor Temp [°C]"/>
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Tire Grip Degradation [%]">
          <GC><ResponsiveContainer width="100%" height={180}>
            <AreaChart data={laps}><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="lap" {...ax()}/>
              <YAxis {...ax()} domain={[85, 102]}/>
              <Tooltip contentStyle={TT}/>
              <ReferenceLine y={95} stroke={C.am} strokeDasharray="4 2"/>
              <Area dataKey="tireGrip" stroke={C.am} fill={`${C.am}12`} strokeWidth={1.5} dot={{ r: 3, fill: C.am }} name="Relative Grip [%]"/>
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// BRAKING & CORNERS TAB
// ═══════════════════════════════════════════════════════════════════════════
function BrakingTab({ brakingPts }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <Sec title="Braking Distance — Optimal vs Actual [m]">
        <GC><ResponsiveContainer width="100%" height={220}>
          <BarChart data={brakingPts} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="corner" {...ax()}/>
            <YAxis {...ax()} label={{ value: "m", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }}/>
            <Tooltip contentStyle={TT}/>
            <Bar dataKey="optBrakeDist" fill={C.gn} fillOpacity={0.5} name="Optimal" barSize={10}/>
            <Bar dataKey="actBrakeDist" fill={C.cy} fillOpacity={0.7} name="Actual" barSize={10}/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="Corner Entry Speed [m/s]">
        <GC><ResponsiveContainer width="100%" height={220}>
          <BarChart data={brakingPts} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="corner" {...ax()}/>
            <YAxis {...ax()}/>
            <Tooltip contentStyle={TT}/>
            <Bar dataKey="optEntrySpeed" fill={C.gn} fillOpacity={0.5} name="Opt Entry" barSize={10}/>
            <Bar dataKey="entrySpeed" fill={C.cy} fillOpacity={0.7} name="Act Entry" barSize={10}/>
            <Bar dataKey="optApexSpeed" fill={C.am} fillOpacity={0.5} name="Opt Apex" barSize={10}/>
            <Bar dataKey="apexSpeed" fill="#e879f9" fillOpacity={0.7} name="Act Apex" barSize={10}/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </BarChart>
        </ResponsiveContainer></GC>
      </Sec>

      {/* Braking point scatter */}
      <Sec title="Late Braking Analysis (entry speed vs brake distance)">
        <GC><ResponsiveContainer width="100%" height={200}>
          <ScatterChart margin={{ top: 12, right: 16, bottom: 20, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis type="number" dataKey="actBrakeDist" name="Brake Dist" {...ax()} label={{ value: "Brake Dist [m]", position: "bottom", fill: C.dm, fontSize: 8 }}/>
            <YAxis type="number" dataKey="entrySpeed" name="Entry Speed" {...ax()} label={{ value: "Entry Speed [m/s]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }}/>
            <Tooltip contentStyle={TT}/>
            <Scatter data={brakingPts} fill={C.cy} fillOpacity={0.7} r={5} name="Actual"/>
            <Scatter data={brakingPts.map(p => ({ actBrakeDist: p.optBrakeDist, entrySpeed: p.optEntrySpeed }))} fill={C.gn} fillOpacity={0.5} r={5} name="Optimal" legendType="diamond"/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </ScatterChart>
        </ResponsiveContainer></GC>
      </Sec>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAIL BRAKING TAB
// ═══════════════════════════════════════════════════════════════════════════
function TrailTab({ lap }) {
  // Trail braking = simultaneous brake + steering
  const trailData = useMemo(() =>
    lap.filter((_, i) => i % 2 === 0).map(d => ({
      ...d,
      brakeSteerOverlap: +(Math.min(d.actBrake, Math.abs(d.actSteer) * 0.8)).toFixed(1),
    })),
    [lap]
  );

  // Quality metric: smoothness of brake release during turn-in
  const trailScore = useMemo(() => {
    let score = 0, n = 0;
    for (let i = 1; i < trailData.length; i++) {
      if (trailData[i].actBrake > 5 && Math.abs(trailData[i].actSteer) > 10) {
        const smoothness = 1 - Math.min(1, Math.abs(trailData[i].actBrake - trailData[i - 1].actBrake) / 15);
        score += smoothness; n++;
      }
    }
    return n > 0 ? (score / n * 100) : 0;
  }, [trailData]);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Trail Score" value={`${trailScore.toFixed(0)}%`} sub="brake release smoothness" sentiment={trailScore > 75 ? "positive" : trailScore > 50 ? "amber" : "negative"} delay={0} />
        <KPI label="Overlap Time" value={`${(trailData.filter(d => d.brakeSteerOverlap > 5).length * 0.2).toFixed(1)}s`} sub="brake+steer combined" sentiment="neutral" delay={1} />
        <KPI label="Peak Overlap" value={`${Math.max(...trailData.map(d => d.brakeSteerOverlap)).toFixed(0)}%`} sub="max combined input" sentiment="neutral" delay={2} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Trail Braking Intensity (brake × |steer|)">
          <GC><ResponsiveContainer width="100%" height={200}>
            <AreaChart data={trailData} syncId="trail"><CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis dataKey="dist" {...ax()}/>
              <YAxis {...ax()} domain={[0, 80]}/>
              <Tooltip contentStyle={TT}/>
              <Area dataKey="brakeSteerOverlap" stroke={C.am} fill={`${C.am}15`} strokeWidth={1.5} dot={false} name="Trail Brake"/>
              <Area dataKey="actBrake" stroke={C.red} fill={`${C.red}08`} strokeWidth={1} dot={false} name="Brake %"/>
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title="Brake vs Steer Phase (should be smooth handoff)">
          <GC><ResponsiveContainer width="100%" height={200}>
            <ScatterChart margin={{ top: 12, right: 16, bottom: 20, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
              <XAxis type="number" dataKey="actBrake" name="Brake %" {...ax()} domain={[0, 100]}
                label={{ value: "Brake %", position: "bottom", fill: C.dm, fontSize: 8 }}/>
              <YAxis type="number" dataKey="x" name="|Steer|" {...ax()} domain={[0, 120]}
                label={{ value: "|Steer| deg", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 8 }}/>
              <Tooltip contentStyle={TT}/>
              <Scatter data={trailData.map(d => ({ actBrake: d.actBrake, x: Math.abs(d.actSteer) }))} name="Phase">
                {trailData.map((d, i) => <Cell key={i} fill={C.cy} opacity={0.1 + 0.9 * (i / trailData.length)}/>)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN MODULE EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function DriverCoachingModule() {
  const [tab, setTab] = useState("inputs");

  // Generate data (would come from real telemetry in production)
  const lap = useMemo(() => gDriverLap(), []);
  const sectors = useMemo(() => gSectorAnalysis(lap), [lap]);
  const consistency = useMemo(() => gConsistency(), []);
  const brakingPts = useMemo(() => gBrakingPoints(), []);

  // Summary KPIs
  const totalDelta = useMemo(() => lap.reduce((a, d) => a + d.timeDelta, 0), [lap]);
  const avgSteerErr = useMemo(() => Math.sqrt(lap.reduce((a, d) => a + d.steerError ** 2, 0) / lap.length), [lap]);

  return (
    <div>
      {/* KPI Summary */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Time Delta" value={`${totalDelta > 0 ? "+" : ""}${(totalDelta * 10).toFixed(2)}s`} sub="vs MPC optimal" sentiment={totalDelta <= 0 ? "positive" : "negative"} delay={0} />
        <KPI label="Steer RMS Error" value={`${avgSteerErr.toFixed(1)}°`} sub="vs optimal path" sentiment={avgSteerErr < 5 ? "positive" : avgSteerErr < 10 ? "amber" : "negative"} delay={1} />
        <KPI label="Best Sector" value={sectors.reduce((best, s) => parseFloat(s.delta) < parseFloat(best.delta) ? s : best, sectors[0]).sector} sub="closest to optimal" sentiment="positive" delay={2} />
        <KPI label="Worst Sector" value={sectors.reduce((worst, s) => parseFloat(s.delta) > parseFloat(worst.delta) ? s : worst, sectors[0]).sector} sub="most time lost" sentiment="negative" delay={3} />
        <KPI label="Reaction Lag" value="~60 ms" sub="estimated from input delay" sentiment="amber" delay={4} />
      </div>

      {/* Tab Switcher */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {TABS.map(t => (
          <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.cy} />
        ))}
      </div>

      {/* Tab Content */}
      {tab === "inputs" && <InputsTab lap={lap} />}
      {tab === "sectors" && <SectorsTab sectors={sectors} />}
      {tab === "consistency" && <ConsistencyTab laps={consistency} />}
      {tab === "braking" && <BrakingTab brakingPts={brakingPts} />}
      {tab === "trail" && <TrailTab lap={lap} />}
    </div>
  );
}