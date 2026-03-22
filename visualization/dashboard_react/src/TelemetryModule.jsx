// ═══════════════════════════════════════════════════════════════════════════════
// src/TelemetryModule.jsx — Project-GP Dashboard v4.0
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHANGES FROM v3.0:
// ─────────────────────────────────────────────────────────────────────────────
// LIVE MODE:
//   - Track map replaced with TubeTrackMap (stochastic safety tubes)
//   - IR strips replaced with ThermalQuintet (5-node radial diagrams)
//   - NEW: WaveletHeatmap panel (Db4 3-level coefficient visualization)
//   - NEW: AL Constraint Monitor (stacked area with slack/λ toggle)
//   - NEW: Energy Flow Strip (Port-Hamiltonian stacked area with dH/dt readout)
//   - KPI bar expanded to 7 columns
//   - Top row: 3-column grid (track+tubes | wavelets | thermal)
//   - Bottom row: 2-column grid (AL constraints | energy flow)
//   - Configurable time-series graphs preserved below
//
// ANALYZE MODE:
//   - Track map now shows stochastic tube overlay
//   - NEW: Thermal Evolution chart (5-node per corner, with grip band annotation)
//   - Existing workspace tabs (overlay, fidelity, frequency, spatial, ai) preserved
//
// NEW PROPS (from App.jsx):
//   thermal5  — Array from gThermal5()
//   tubes     — Array from gTubePoints()
//   wavelets  — Array from gWaveletCoeffs()
//   alData    — Array from gALConstraints()
//   energy    — Array from gEnergyBudget()
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Legend,
} from "recharts";
import { C, GL, GS, TT, AX, thermalColor } from "./theme.js";
import { KPI, Sec, GC, Pill, FadeSlide } from "./components.jsx";

// v3 data generators still used by AnalyzeMode
import {
  gTK, gTT as gTireT, gLapDelta, gSlipEnergy, gFrictionSatHist,
  gFreqResponse, gDamperHist, gRideHeightHist, gRollCenterMig, gPSDOverlay,
  gEKFInnovation, gPacejkaDrift, gGripUtil, gLoadTransfer,
} from "./data.js";

// v4 canvas components
import { TubeTrackMap, ThermalQuintet, WaveletHeatmap } from "./canvas";


// ═════════════════════════════════════════════════════════════════════════════
// SHARED MICRO-COMPONENTS
// ═════════════════════════════════════════════════════════════════════════════

const Lbl = ({ children, color }) => (
  <span style={{
    fontSize: 8, fontWeight: 700, color: color || C.dm,
    fontFamily: C.dt, letterSpacing: 1.5, textTransform: "uppercase",
  }}>
    {children}
  </span>
);

const Val = ({ children, color, big }) => (
  <span style={{
    fontSize: big ? 18 : 12, fontWeight: 700,
    color: color || C.w, fontFamily: C.dt,
  }}>
    {children}
  </span>
);

const Vu = ({ children }) => (
  <span style={{ fontSize: 7, color: C.dm, fontFamily: C.dt, marginLeft: 2 }}>
    {children}
  </span>
);

const tc = (v, lo, hi) => v < lo ? C.gn : v < hi ? C.am : C.red;


// ═════════════════════════════════════════════════════════════════════════════
// LIVE DATA HOOK (preserved from v3, drives the step counter)
// ═════════════════════════════════════════════════════════════════════════════

const HIST_LEN = 200;

function useLiveData(active) {
  const [frame, setFrame] = useState(null);
  const history = useRef([]);
  const step = useRef(0);

  useEffect(() => {
    if (!active) return;

    const id = setInterval(() => {
      step.current += 1;
      const t = step.current * 0.05;
      const steer = 12 * Math.sin(t * 1.5);
      const speed = 14 + 4 * Math.sin(t * 0.3) + Math.random();
      const kappa = 0.08 * Math.sin(t / 15) + 0.04 * Math.sin(t / 7);
      const latG = speed * speed * kappa / 9.81;
      const lonG = (Math.random() - 0.5) * 0.3;
      const combinedG = Math.sqrt(latG * latG + lonG * lonG);

      const f = {
        step: step.current,
        t: +t.toFixed(2),
        speed: +speed.toFixed(1),
        steer: +steer.toFixed(1),
        latG: +latG.toFixed(3),
        lonG: +lonG.toFixed(3),
        combinedG: +combinedG.toFixed(3),
        curvature: +kappa.toFixed(5),
        yawRate: +(latG * 0.6 + Math.random() * 0.1).toFixed(3),
        sideslip: +(latG * 0.02 - 0.01 + Math.random() * 0.005).toFixed(4),
        wmpcSolveMs: +(6 + 5 * Math.random()).toFixed(1),
        hamiltonianJ: +(4.5 + 0.5 * Math.sin(t * 0.2)).toFixed(2),
        ekfInnov: +(Math.random() * 0.04 - 0.02).toFixed(3),
      };

      history.current.push(f);
      if (history.current.length > HIST_LEN) history.current.shift();
      setFrame(f);
    }, 50);

    return () => clearInterval(id);
  }, [active]);

  return { frame, history, stepIdx: step };
}


// ═════════════════════════════════════════════════════════════════════════════
// LIVE MODE
// ═════════════════════════════════════════════════════════════════════════════

function LiveMode({ mode, thermal5, tubes, wavelets, alData, energy, track }) {
  const { frame, history, stepIdx } = useLiveData(mode === "LIVE");
  const [alView, setAlView] = useState("slack");
  const step = stepIdx.current || 0;

  const f = frame || {
    speed: 0, steer: 0, latG: 0, lonG: 0, combinedG: 0,
    curvature: 0, yawRate: 0, wmpcSolveMs: 0, hamiltonianJ: 0,
  };

  // Window into rolling data for charts
  const alWindow = useMemo(
    () => alData.slice(Math.max(0, (step % alData.length) - 80), (step % alData.length) + 1),
    [alData, step],
  );
  const energyWindow = useMemo(
    () => energy.slice(Math.max(0, (step % energy.length) - 100), (step % energy.length) + 1),
    [energy, step],
  );
  const energyFrame = energy[step % energy.length] || {};
  const alFrame = alData[step % alData.length] || {};

  return (
    <div>
      {/* ═══ KPI BAR ═══════════════════════════════════════════════════ */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(7, 1fr)",
        gap: 6, marginBottom: 10,
      }}>
        {[
          ["Speed",    f.speed,     C.w,  "m/s"],
          ["Lat G",    f.latG,      C.cy, "G"],
          ["Lon G",    f.lonG,      C.am, "G"],
          ["Combined", f.combinedG, C.gn, "G"],
          ["Yaw ψ̇",   f.yawRate,   C.pr, "rad/s"],
          ["WMPC",     f.wmpcSolveMs, tc(+f.wmpcSolveMs, 8, 10), "ms"],
          ["H(q,p)",   f.hamiltonianJ, C.pr, "kJ"],
        ].map(([label, value, color, unit], i) => (
          <div key={i} style={{
            ...GL, padding: "6px 8px", textAlign: "center",
            borderTop: `2px solid ${color}`,
          }}>
            <div style={{
              fontSize: 7, color: C.dm, fontFamily: C.dt,
              letterSpacing: 1.5, textTransform: "uppercase",
            }}>
              {label}
            </div>
            <div style={{
              fontSize: 16, fontWeight: 800,
              color, fontFamily: C.dt,
            }}>
              {value}
            </div>
            <div style={{ fontSize: 7, color: C.dm, fontFamily: C.dt }}>
              {unit}
            </div>
          </div>
        ))}
      </div>

      {/* ═══ ROW 1: TRACK + WAVELETS + THERMAL ════════════════════════ */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1.3fr 1fr 0.8fr",
        gap: 10, marginBottom: 10,
      }}>
        {/* Track Map with Stochastic Safety Tubes */}
        <Sec title="Track + Safety Tubes">
          <GC style={{ padding: 6 }}>
            <TubeTrackMap
              track={track}
              tubes={tubes}
              step={step % track.length}
              width={380}
              height={260}
            />
          </GC>
        </Sec>

        {/* Wavelet Coefficient Heatmap */}
        <Sec title="MPC Wavelet Decomposition (Db4)">
          <GC style={{ padding: 6 }}>
            <WaveletHeatmap data={wavelets} width={320} height={200} />
            <div style={{
              fontSize: 7, color: C.dm, fontFamily: C.dt,
              padding: "4px", textAlign: "center", lineHeight: 1.5,
            }}>
              Energy in cA3 = path planning · Energy in cD1 = high-freq corrections (chattering risk)
            </div>
          </GC>
        </Sec>

        {/* 5-Node Tire Thermal Diagrams */}
        <Sec title="Tire Thermal (5-Node Jaeger)">
          <GC style={{ padding: 6 }}>
            <ThermalQuintet
              data={thermal5}
              step={step % thermal5.length}
              size={88}
            />
          </GC>
        </Sec>
      </div>

      {/* ═══ ROW 2: AL CONSTRAINTS + ENERGY FLOW ═════════════════════ */}
      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr",
        gap: 10, marginBottom: 10,
      }}>
        {/* AL Constraint Monitor */}
        <Sec
          title="AL Constraint Monitor"
          right={
            <div style={{ display: "flex", gap: 4 }}>
              <Pill active={alView === "slack"} label="Slack" onClick={() => setAlView("slack")} color={C.cy} />
              <Pill active={alView === "lambda"} label="λ" onClick={() => setAlView("lambda")} color={C.am} />
            </div>
          }
        >
          <GC>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={alWindow} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="solve" {...AX} />
                <YAxis {...AX} />
                <Tooltip contentStyle={TT} />
                {alView === "slack" ? (
                  <>
                    {[
                      ["grip",  C.red],
                      ["steer", C.am],
                      ["ax",    C.cy],
                      ["track", C.gn],
                      ["vx",    C.pr],
                    ].map(([key, color]) => (
                      <Area
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stackId="1"
                        fill={color}
                        fillOpacity={0.25}
                        stroke={color}
                        strokeWidth={1}
                        name={key}
                      />
                    ))}
                  </>
                ) : (
                  <Line
                    type="monotone"
                    dataKey="lambda_grip"
                    stroke={C.am}
                    strokeWidth={2}
                    dot={false}
                    name="λ_grip"
                  />
                )}
                <Legend wrapperStyle={{ fontSize: 7, fontFamily: C.hd }} />
              </AreaChart>
            </ResponsiveContainer>
            {/* Status bar */}
            <div style={{
              display: "flex", gap: 10, padding: "4px 8px",
              borderTop: `1px solid ${C.b1}`,
            }}>
              <Lbl color={alFrame.grip < 0.05 ? C.red : C.gn}>
                Binding: {
                  ["grip", "steer", "ax", "track", "vx"]
                    .filter(k => (alFrame[k] || 1) < 0.05).length
                }/5
              </Lbl>
              <Lbl color={C.cy}>AL iters: {alFrame.iters || "—"}</Lbl>
              <Lbl color={C.am}>λ_grip: {alFrame.lambda_grip || "—"}</Lbl>
            </div>
          </GC>
        </Sec>

        {/* Energy Flow Strip (Port-Hamiltonian) */}
        <Sec title="Energy Flow (Port-Hamiltonian)">
          <GC>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={energyWindow} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="t" {...AX} />
                <YAxis {...AX} />
                <Tooltip contentStyle={TT} />
                <Area type="monotone" dataKey="ke"     stackId="1" fill={C.e_ke}   fillOpacity={0.25} stroke={C.e_ke}   strokeWidth={1}   name="KE" />
                <Area type="monotone" dataKey="pe_s"   stackId="1" fill={C.e_spe}  fillOpacity={0.25} stroke={C.e_spe}  strokeWidth={1}   name="PE_spring" />
                <Area type="monotone" dataKey="pe_arb" stackId="1" fill={C.e_arb}  fillOpacity={0.2}  stroke={C.e_arb}  strokeWidth={1}   name="PE_ARB" />
                <Area type="monotone" dataKey="h_net"  stackId="1" fill={C.e_hnet} fillOpacity={0.35} stroke={C.e_hnet} strokeWidth={1.5} name="H_net" />
                <Legend wrapperStyle={{ fontSize: 7, fontFamily: C.hd }} />
              </AreaChart>
            </ResponsiveContainer>
            {/* Live readout bar */}
            <div style={{
              display: "flex", justifyContent: "space-between",
              padding: "4px 8px",
              borderTop: `1px solid ${C.b1}`,
            }}>
              <div>
                <Lbl>H(q,p) = </Lbl>
                <Val color={C.cy}>{energyFrame.H || "—"}</Val>
                <Vu>J</Vu>
              </div>
              <div>
                <Lbl>dH/dt = </Lbl>
                <Val color={+energyFrame.dH > 0 ? C.red : C.gn}>
                  {energyFrame.dH || "—"}
                </Val>
                <Vu>W</Vu>
              </div>
              <div>
                <Lbl>R_diss = </Lbl>
                <Val color={C.e_diss}>{energyFrame.r_diss || "—"}</Val>
                <Vu>W</Vu>
              </div>
            </div>
          </GC>
        </Sec>
      </div>

      {/* ═══ ROW 3: TIME-SERIES HISTORY (from useLiveData) ════════════ */}
      <Sec title="Live Time-Series (200-frame buffer)">
        <GC>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart
              data={history.current.slice(-100)}
              margin={{ top: 8, right: 12, bottom: 8, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="t" {...AX} />
              <YAxis {...AX} />
              <Tooltip contentStyle={TT} />
              <Line type="monotone" dataKey="latG"   stroke={C.cy} strokeWidth={1.5} dot={false} name="Lat G" />
              <Line type="monotone" dataKey="lonG"   stroke={C.am} strokeWidth={1}   dot={false} name="Lon G" />
              <Line type="monotone" dataKey="speed"  stroke={C.gn} strokeWidth={1}   dot={false} name="Speed (m/s)" />
              <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
              <Legend wrapperStyle={{ fontSize: 7, fontFamily: C.hd }} />
            </LineChart>
          </ResponsiveContainer>
        </GC>
      </Sec>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// ANALYZE MODE
// ═════════════════════════════════════════════════════════════════════════════

const AZ_TABS = [
  { key: "overview",  label: "Overview" },
  { key: "tubes",     label: "Tubes & Thermal" },
  { key: "frequency", label: "Frequency" },
  { key: "fidelity",  label: "Fidelity" },
];

function AnalyzeMode({ track, tireTemps, thermal5, tubes, energy }) {
  const [ws, setWs] = useState("overview");

  const lapDelta  = useMemo(() => track ? gLapDelta?.(track) || [] : [],  [track]);
  const gripUtil  = useMemo(() => track ? gGripUtil?.(track) || [] : [],  [track]);
  const ekf       = useMemo(() => gEKFInnovation?.() || [],               []);
  const pacDrift  = useMemo(() => gPacejkaDrift?.() || [],                []);
  const freqResp  = useMemo(() => gFreqResponse?.() || [],                []);
  const damperH   = useMemo(() => gDamperHist?.() || [],                  []);
  const psd       = useMemo(() => gPSDOverlay?.() || [],                  []);
  const loadTrans = useMemo(() => gLoadTransfer?.() || [],                []);

  return (
    <>
      {/* Tab bar */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {AZ_TABS.map(t => (
          <Pill
            key={t.key}
            active={ws === t.key}
            label={t.label}
            onClick={() => setWs(t.key)}
            color={C.cy}
          />
        ))}
      </div>

      <FadeSlide keyVal={ws}>
        {/* ── OVERVIEW TAB ──────────────────────────────────────────── */}
        {ws === "overview" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <Sec title="Speed Profile">
              <GC>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={track} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="s" {...AX} />
                    <YAxis {...AX} />
                    <Tooltip contentStyle={TT} />
                    <Area type="monotone" dataKey="speed" stroke={C.cy} fill={C.cy} fillOpacity={0.1} strokeWidth={1.5} name="Speed (m/s)" />
                  </AreaChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
            <Sec title="Lateral G">
              <GC>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={track} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="s" {...AX} />
                    <YAxis {...AX} />
                    <Tooltip contentStyle={TT} />
                    <Area type="monotone" dataKey="lat_g" stroke={C.am} fill={C.am} fillOpacity={0.1} strokeWidth={1.5} name="Lat G" />
                    <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
                  </AreaChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
            <Sec title="Curvature">
              <GC>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={track} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="s" {...AX} />
                    <YAxis {...AX} />
                    <Tooltip contentStyle={TT} />
                    <Line type="monotone" dataKey="curvature" stroke={C.pr} strokeWidth={1.5} dot={false} name="κ (1/m)" />
                  </LineChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
          </div>
        )}

        {/* ── TUBES & THERMAL TAB (NEW) ─────────────────────────────── */}
        {ws === "tubes" && (
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 12 }}>
            {/* Track with tube overlay */}
            <Sec title="Track + Stochastic Safety Tubes (Post-Run)">
              <GC style={{ padding: 8 }}>
                <TubeTrackMap
                  track={track}
                  tubes={tubes}
                  step={track.length - 1}
                  width={480}
                  height={340}
                />
              </GC>
            </Sec>

            {/* 5-node thermal evolution */}
            <Sec title="Thermal Evolution — 5-Node (FL)">
              <GC>
                <ResponsiveContainer width="100%" height={340}>
                  <LineChart
                    data={thermal5.filter((_, i) => i % 3 === 0)}
                    margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="s" {...AX} />
                    <YAxis {...AX} domain={[20, 165]} />
                    <Tooltip contentStyle={TT} />

                    <ReferenceArea
                      y1={85} y2={105} fill={C.gn} fillOpacity={0.04}
                      label={{
                        value: "Optimal grip band",
                        fill: C.gn, fontSize: 7, fontFamily: C.dt,
                        position: "insideTopRight",
                      }}
                    />
                    <ReferenceLine
                      y={140} stroke={C.red} strokeDasharray="4 2"
                      label={{
                        value: "Degradation",
                        fill: C.red, fontSize: 7, fontFamily: C.dt,
                      }}
                    />

                    {[
                      ["Flash",   0, C.th_crit, "4 2"],
                      ["Surface", 1, C.th_hot,  ""],
                      ["Bulk",    2, C.th_warm, ""],
                      ["Carcass", 3, C.th_cool, ""],
                      ["Gas",     4, C.th_cold, ""],
                    ].map(([name, idx, color, dash]) => (
                      <Line
                        key={name}
                        type="monotone"
                        dataKey={d => d.fl?.[idx]}
                        stroke={color}
                        strokeWidth={idx === 1 ? 2.5 : 1.5}
                        dot={false}
                        name={name}
                        strokeDasharray={dash}
                      />
                    ))}
                    <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
                  </LineChart>
                </ResponsiveContainer>
              </GC>
            </Sec>

            {/* Energy dH/dt trace */}
            <Sec title="dH/dt — Energy Balance">
              <GC>
                <ResponsiveContainer width="100%" height={180}>
                  <ComposedChart
                    data={energy.filter((_, i) => i % 3 === 0)}
                    margin={{ top: 8, right: 12, bottom: 8, left: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="t" {...AX} />
                    <YAxis {...AX} />
                    <Tooltip contentStyle={TT} />
                    <ReferenceArea y1={0} y2={50} fill={C.red} fillOpacity={0.03} />
                    <ReferenceArea y1={-50} y2={0} fill={C.gn} fillOpacity={0.03} />
                    <ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="dH" stroke={C.cy} strokeWidth={1.5} dot={false} name="dH/dt (W)" />
                    <Line type="monotone" dataKey="r_diss" stroke={C.e_diss} strokeWidth={1} dot={false} name="R_diss (W)" />
                    <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
                  </ComposedChart>
                </ResponsiveContainer>
              </GC>
            </Sec>

            {/* AL constraint convergence over full run */}
            <Sec title="AL Constraint Convergence">
              <GC>
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={alData} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="solve" {...AX} />
                    <YAxis {...AX} />
                    <Tooltip contentStyle={TT} />
                    {[
                      ["grip",  C.red],
                      ["steer", C.am],
                      ["ax",    C.cy],
                      ["track", C.gn],
                      ["vx",    C.pr],
                    ].map(([k, c]) => (
                      <Area key={k} type="monotone" dataKey={k} stackId="1" fill={c} fillOpacity={0.2} stroke={c} strokeWidth={1} name={k} />
                    ))}
                    <Legend wrapperStyle={{ fontSize: 7, fontFamily: C.hd }} />
                  </AreaChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
          </div>
        )}

        {/* ── FREQUENCY TAB (preserved from v3) ─────────────────────── */}
        {ws === "frequency" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Sec title="Frequency Response (Heave)">
              <GC>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={freqResp} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="freq" {...AX} label={{ value: "Frequency (Hz)", position: "bottom", fill: C.dm, fontSize: 9 }} />
                    <YAxis {...AX} label={{ value: "Gain (dB)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
                    <Tooltip contentStyle={TT} />
                    <Line type="monotone" dataKey="front_dB" stroke={C.cy} strokeWidth={2} dot={false} name="Front" />
                    <Line type="monotone" dataKey="rear_dB" stroke={C.am} strokeWidth={2} dot={false} name="Rear" />
                    <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
                    <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "4px 8px" }}>
                  Peaks at ~1.8 Hz (front) and ~2.1 Hz (rear) = undamped natural frequencies.
                  ζ ≈ 0.35 — underdamped. Damper tuning target: ζ = 0.4–0.6.
                </div>
              </GC>
            </Sec>
            <Sec title="PSD Overlay (Real vs Sim)">
              <GC>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={psd} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="freq" {...AX} label={{ value: "Frequency (Hz)", position: "bottom", fill: C.dm, fontSize: 9 }} />
                    <YAxis {...AX} />
                    <Tooltip contentStyle={TT} />
                    <Line type="monotone" dataKey="real" stroke={C.am} strokeWidth={1.5} dot={false} name="Real" />
                    <Line type="monotone" dataKey="sim" stroke={C.cy} strokeWidth={1.5} dot={false} name="Sim" strokeDasharray="4 2" />
                    <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
                  </LineChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
          </div>
        )}

        {/* ── FIDELITY TAB (preserved from v3) ──────────────────────── */}
        {ws === "fidelity" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Sec title="EKF Innovation Residuals">
              <GC>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart
                    data={ekf.filter((_, i) => i % 2 === 0)}
                    margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="t" {...AX} />
                    <YAxis {...AX} />
                    <Tooltip contentStyle={TT} />
                    <ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" />
                    <ReferenceArea y1={-0.02} y2={0.02} fill={C.gn} fillOpacity={0.04} />
                    <Line type="monotone" dataKey="innov_ax" stroke={C.cy} strokeWidth={1} dot={false} name="a_x" />
                    <Line type="monotone" dataKey="innov_wz" stroke={C.am} strokeWidth={1} dot={false} name="ω_z" />
                    <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
                  </LineChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
            <Sec title="Pacejka Drift (μ_y % per Lap)">
              <GC>
                <ResponsiveContainer width="100%" height={260}>
                  <ComposedChart data={pacDrift} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="lap" {...AX} />
                    <YAxis yAxisId="l" {...AX} />
                    <YAxis yAxisId="r" orientation="right" {...AX} />
                    <Tooltip contentStyle={TT} />
                    <Line yAxisId="l" type="monotone" dataKey="muY_pct" stroke={C.am} strokeWidth={2} dot={{ r: 3, fill: C.am }} name="μ_y %" />
                    <Line yAxisId="r" type="monotone" dataKey="stiffness" stroke={C.cy} strokeWidth={1.5} dot={false} name="C_α" />
                    <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
                  </ComposedChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
          </div>
        )}
      </FadeSlide>
    </>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═════════════════════════════════════════════════════════════════════════════

export default function TelemetryModule({
  track, tireTemps, mode,
  // v4 additions:
  thermal5 = [], tubes = [], wavelets = [],
  alData = [], energy = [],
}) {
  if (mode === "LIVE") {
    return (
      <LiveMode
        mode={mode}
        thermal5={thermal5}
        tubes={tubes}
        wavelets={wavelets}
        alData={alData}
        energy={energy}
        track={track}
      />
    );
  }

  return (
    <AnalyzeMode
      track={track}
      tireTemps={tireTemps}
      thermal5={thermal5}
      tubes={tubes}
      energy={energy}
      alData={alData}
    />
  );
}