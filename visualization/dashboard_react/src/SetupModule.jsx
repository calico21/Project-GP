import React, { useState, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter, BarChart, Bar,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill, FadeSlide, Eq, Pt, Hl } from "./components.jsx";
import {
  PN, PU, gLoadTransfer, gDamperHist, gFreqResponse, gLapDelta,
  gSlipEnergy, gFrictionSatHist, gUndersteerGrad,
  gTrustRegion, gEpistemicConf,
  gYawPhaseLag, gRideHeightHist, gRollCenterMig,
  gHamiltonianEnergy, gDissipationBreakdown, gRegenEnvelope, gTorqueVectoring,
  gHorizonTraj, gWaveletCoeffs, gALSlack, gControlEffort,
  gEKFInnovation, gPacejkaDrift, gPSDOverlay, gFidelitySpider,
} from "./data.js";

// ── Shared sub-components ───────────────────────────────────────────
const Note = ({ children }) => <div style={{ marginTop: 10, ...GL, padding: "10px 14px", fontSize: 10, color: C.md, fontFamily: C.dt, lineHeight: 1.8 }}>{children}</div>;
const ChartTitle = ({ text }) => <div style={{ fontSize: 9, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 2, textTransform: "uppercase", marginBottom: 6, paddingLeft: 4 }}>{text}</div>;

// ── Parallel Coordinates ────────────────────────────────────────────
function ParCoords({ pareto }) {
  const W = 820, H = 340, pd = { t: 38, b: 38, l: 20, r: 20 };
  const axes = PN.slice(0, 16), nA = axes.length, xS = (W - pd.l - pd.r) / (nA - 1);
  const yT = pd.t, yB = H - pd.b, bG = Math.max(...pareto.map(p => p.grip));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%" }}>
      <defs>
        <linearGradient id="pcG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor={C.cy} stopOpacity=".5" /><stop offset="100%" stopColor={C.am} stopOpacity=".5" /></linearGradient>
        <linearGradient id="pcB" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor={C.red} /><stop offset="100%" stopColor="#ff6040" /></linearGradient>
      </defs>
      {axes.map((a, i) => { const x = pd.l + i * xS; return <g key={a}><line x1={x} y1={yT} x2={x} y2={yB} stroke={C.b1} /><text x={x} y={yT - 8} textAnchor="middle" fill={C.md} fontSize={7} fontFamily="Outfit" fontWeight={600} transform={`rotate(-40 ${x} ${yT - 8})`}>{a}</text></g>; })}
      {pareto.map((s, si) => { const best = s.grip >= bG - 0.01; return <polyline key={si} points={axes.map((_, i) => `${pd.l + i * xS},${yT + (1 - (s.params[i] || 0)) * (yB - yT)}`).join(" ")} fill="none" stroke={best ? "url(#pcB)" : "url(#pcG)"} strokeWidth={best ? 2.5 : 1} strokeOpacity={best ? 1 : 0.14} />; })}
    </svg>
  );
}

// ── Confidence Gauge ────────────────────────────────────────────────
function ConfGauge({ score }) {
  const r = 70, cx = 90, cy = 85;
  const startAngle = Math.PI, endAngle = 0;
  const angle = startAngle - (score / 100) * Math.PI;
  const x = cx + r * Math.cos(angle), y = cy - r * Math.sin(angle);
  const arcPath = `M ${cx - r} ${cy} A ${r} ${r} 0 ${score > 50 ? 1 : 0} 1 ${x} ${y}`;
  const ac = score > 80 ? C.gn : score > 60 ? C.am : C.red;
  return (
    <svg viewBox="0 0 180 110" style={{ width: 180, height: 110 }}>
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`} fill="none" stroke={C.b1} strokeWidth={8} strokeLinecap="round" />
      <path d={arcPath} fill="none" stroke={ac} strokeWidth={8} strokeLinecap="round" />
      <text x={cx} y={cy - 10} textAnchor="middle" fill={ac} fontSize={28} fontWeight={800} fontFamily="Outfit">{score}%</text>
      <text x={cx} y={cy + 10} textAnchor="middle" fill={C.dm} fontSize={8} fontFamily="Azeret Mono">CONFIDENCE</text>
    </svg>
  );
}

// ── GRID DEFINITIONS ────────────────────────────────────────────────
const GRIDS = [
  { key: "g1", label: "Optimizer Core", icon: "◆" },
  { key: "g2", label: "Kinematic & Aero", icon: "△" },
  { key: "g3", label: "Contact Patch", icon: "◉" },
  { key: "g4", label: "Energy Flow", icon: "⬡" },
  { key: "g5", label: "WMPC Horizon", icon: "◈" },
  { key: "g6", label: "Twin Fidelity", icon: "⬢" },
];

export default function SetupModule({ pareto, conv, sens, track }) {
  const [grid, setGrid] = useState("g1");

  // ── Memoized derived data ─────────────────────────────────────────
  const trustRegion = useMemo(() => gTrustRegion(), []);
  const epistemic = useMemo(() => gEpistemicConf(pareto), [pareto]);
  const freqResp = useMemo(() => gFreqResponse(), []);
  const yawLag = useMemo(() => track ? gYawPhaseLag(track) : [], [track]);
  const rideHist = useMemo(() => gRideHeightHist(), []);
  const rcMig = useMemo(() => gRollCenterMig(), []);
  const lapDelta = useMemo(() => track ? gLapDelta(track) : [], [track]);
  const loads = useMemo(() => track ? gLoadTransfer(track) : [], [track]);
  const slipE = useMemo(() => track ? gSlipEnergy(track) : [], [track]);
  const fricSat = useMemo(() => track ? gFrictionSatHist(track) : [], [track]);
  const Ku = useMemo(() => track ? gUndersteerGrad(track) : [], [track]);
  const hamiltonian = useMemo(() => track ? gHamiltonianEnergy(track) : [], [track]);
  const dissipation = useMemo(() => gDissipationBreakdown(), []);
  const regen = useMemo(() => gRegenEnvelope(), []);
  const torqueVec = useMemo(() => track ? gTorqueVectoring(track) : [], [track]);
  const horizon = useMemo(() => gHorizonTraj(), []);
  const wavelets = useMemo(() => gWaveletCoeffs(), []);
  const alSlack = useMemo(() => track ? gALSlack(track) : [], [track]);
  const ctrlEffort = useMemo(() => track ? gControlEffort(track) : [], [track]);
  const ekf = useMemo(() => gEKFInnovation(), []);
  const pacDrift = useMemo(() => gPacejkaDrift(), []);
  const psd = useMemo(() => gPSDOverlay(), []);
  const fidelity = useMemo(() => gFidelitySpider(), []);
  const damperHist = useMemo(() => gDamperHist(), []);

  const bG = Math.max(...pareto.map(p => p.grip));
  const bS = Math.min(...pareto.filter(p => p.grip > bG - 0.05).map(p => p.stability));
  const fidelityAvg = +(fidelity.reduce((a, f) => a + f.score, 0) / fidelity.length).toFixed(1);

  // ── Chart shorthand ───────────────────────────────────────────────
  const ax = (props) => ({ tick: { fontSize: 9, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, ...props });

  return (
    <>
      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 16 }}>
        <KPI label="Peak Grip" value={`${bG.toFixed(3)}G`} sub="Pareto Max" sentiment="positive" delay={0} />
        <KPI label="Stability" value={bS.toFixed(2)} sub="rad/s" sentiment="amber" delay={1} />
        <KPI label="Pareto" value={pareto.length} sub="setups" delay={2} />
        <KPI label="Confidence" value={`${epistemic.score}%`} sub="epistemic" sentiment={epistemic.score > 80 ? "positive" : "amber"} delay={3} />
        <KPI label="Twin R²" value={`${fidelityAvg}%`} sub="fidelity" sentiment="positive" delay={4} />
        <KPI label="Converged" value={conv[conv.length - 1].bestGrip.toFixed(3)} sub={`${conv.length} iter`} sentiment="positive" delay={5} />
      </div>

      {/* Grid selector */}
      <div style={{ display: "flex", gap: 5, marginBottom: 16, flexWrap: "wrap" }}>
        {GRIDS.map(g => <Pill key={g.key} active={grid === g.key} label={`${g.icon} ${g.label}`} onClick={() => setGrid(g.key)} color={C.red} />)}
      </div>

      <FadeSlide keyVal={grid}>
        {/* ═══════════════════════════════════════════════════════════════
            GRID 1: OPTIMIZER CORE
            ═══════════════════════════════════════════════════════════════ */}
        {grid === "g1" && (<>
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
            <Sec title="Pareto Front — Grip vs Stability"><GC><ResponsiveContainer width="100%" height={320}><ScatterChart margin={{ top: 10, right: 20, bottom: 26, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="grip" type="number" domain={["auto","auto"]} {...ax()} label={{ value: "Lat Grip [G]", position: "bottom", offset: 8, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis dataKey="stability" type="number" {...ax()} label={{ value: "Stab [rad/s]", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><Tooltip contentStyle={TT} /><ReferenceArea y1={0} y2={5} fill={C.gn} fillOpacity={.03} /><ReferenceLine y={5} stroke={C.red} strokeDasharray="6 3" /><Scatter data={pareto} fillOpacity={.85} r={4}>{pareto.map((e, i) => <Cell key={i} fill={e.grip >= bG - .01 ? C.gn : C.cy} />)}</Scatter></ScatterChart></ResponsiveContainer></GC></Sec>
            <div>
              <Sec title="Epistemic Confidence">
                <GC style={{ textAlign: "center", padding: "20px 14px" }}>
                  <ConfGauge score={epistemic.score} />
                  <div style={{ marginTop: 10 }}>
                    {epistemic.breakdown.map(b => (
                      <div key={b.axis} style={{ display: "flex", justifyContent: "space-between", fontSize: 9, fontFamily: C.dt, marginBottom: 3, padding: "0 10px" }}>
                        <span style={{ color: C.md }}>{b.axis}</span>
                        <span style={{ color: b.score > 80 ? C.gn : b.score > 60 ? C.am : C.red, fontWeight: 600 }}>{b.score}%</span>
                      </div>
                    ))}
                  </div>
                </GC>
              </Sec>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Convergence History"><GC><ResponsiveContainer width="100%" height={220}><AreaChart data={conv} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="iter" {...ax()} /><YAxis {...ax()} domain={["auto","auto"]} /><Tooltip contentStyle={TT} /><Area type="monotone" dataKey="bestGrip" stroke={C.red} fill={`${C.red}0c`} strokeWidth={2} dot={false} /></AreaChart></ResponsiveContainer></GC></Sec>
            <Sec title="Trust Region — Fisher Damping"><GC><ResponsiveContainer width="100%" height={220}><ComposedChart data={trustRegion.filter((_, i) => i % 2 === 0)} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="iter" {...ax()} /><YAxis yAxisId="l" {...ax()} /><YAxis yAxisId="r" orientation="right" {...ax()} /><Tooltip contentStyle={TT} /><Line yAxisId="l" type="monotone" dataKey="fishNorm" stroke={C.am} strokeWidth={1.5} dot={false} name="‖F‖ Fisher" /><Area yAxisId="r" type="monotone" dataKey="dampFactor" stroke={C.pr} fill={`${C.pr}08`} strokeWidth={1} dot={false} name="Damp Factor" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></ComposedChart></ResponsiveContainer></GC></Sec>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Sensitivity — ∂Grip / ∂Parameter"><GC><ResponsiveContainer width="100%" height={340}><BarChart data={sens} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 55 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} /><XAxis type="number" {...ax()} /><YAxis dataKey="param" type="category" tick={{ fontSize: 9, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={50} /><Tooltip contentStyle={TT} /><ReferenceLine x={0} stroke={C.dm} /><Bar dataKey="dGrip" radius={[0,4,4,0]} barSize={11}>{sens.map((e, i) => <Cell key={i} fill={e.dGrip > 0 ? C.gn : C.red} fillOpacity={.8} />)}</Bar></BarChart></ResponsiveContainer></GC></Sec>
            <Sec title="Parallel Coordinates — 28D Setup Space"><GC style={{ padding: "12px" }}><div style={{ width: "100%", height: 340 }}><ParCoords pareto={pareto} /></div></GC></Sec>
          </div>
        </>)}

        {/* ═══════════════════════════════════════════════════════════════
            GRID 2: KINEMATIC & AERO
            ═══════════════════════════════════════════════════════════════ */}
        {grid === "g2" && (<>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Damper Frequency Response (Bode)"><GC><ResponsiveContainer width="100%" height={280}><LineChart data={freqResp} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="freq" {...ax()} label={{ value: "Hz", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis {...ax()} label={{ value: "dB", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md } }} /><Tooltip contentStyle={TT} /><ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" /><ReferenceArea x1={1} x2={3} fill={C.gn} fillOpacity={.04} /><ReferenceArea x1={8} x2={15} fill={C.am} fillOpacity={.04} /><Line type="monotone" dataKey="front_dB" stroke={C.cy} strokeWidth={2} dot={false} name="Front" /><Line type="monotone" dataKey="rear_dB" stroke={C.am} strokeWidth={2} dot={false} name="Rear" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></LineChart></ResponsiveContainer></GC><Note>ωn_f = 1.8 Hz, ωn_r = 2.1 Hz, ζ = 0.35. Green = ride band, amber = handling band.</Note></Sec>
            <Sec title="Lap Delta — Actual vs Optimal"><GC><ResponsiveContainer width="100%" height={280}><ComposedChart data={lapDelta.filter((_,i) => i % 2 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis yAxisId="l" {...ax()} /><YAxis yAxisId="r" orientation="right" {...ax()} /><Tooltip contentStyle={TT} /><ReferenceLine yAxisId="l" y={0} stroke={C.gn} strokeDasharray="3 3" /><Area yAxisId="l" type="monotone" dataKey="delta" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.8} dot={false} name="Δt [s]" /><Line yAxisId="r" type="monotone" dataKey="vOptimal" stroke={C.gn} strokeWidth={1} dot={false} name="V_opt" strokeDasharray="4 2" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></ComposedChart></ResponsiveContainer></GC></Sec>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Transient Yaw Phase Lag"><GC><ResponsiveContainer width="100%" height={250}><LineChart data={yawLag} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis {...ax()} domain={[0, 50]} label={{ value: "ms", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md } }} /><Tooltip contentStyle={TT} /><ReferenceArea y1={0} y2={25} fill={C.gn} fillOpacity={.04} /><ReferenceArea y1={35} y2={50} fill={C.red} fillOpacity={.04} /><Line type="monotone" dataKey="lagMs" stroke={C.pr} strokeWidth={1.5} dot={false} name="Yaw Lag [ms]" /></LineChart></ResponsiveContainer></GC><Note>Target: &lt;25ms for sharp turn-in. Influenced by H_net torsional flex, ARB coupling, and tire carcass lag τ.</Note></Sec>
            <Sec title="Dynamic Ride Height Distribution"><GC><ResponsiveContainer width="100%" height={250}><BarChart data={rideHist.filter(d => d.front > 5 || d.rear > 5)} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="height" {...ax()} label={{ value: "Ride Height [mm]", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis {...ax()} /><Tooltip contentStyle={TT} /><Bar dataKey="front" fill={C.cy} fillOpacity={.5} name="Front" radius={[2,2,0,0]} /><Bar dataKey="rear" fill={C.am} fillOpacity={.5} name="Rear" radius={[2,2,0,0]} /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></BarChart></ResponsiveContainer></GC></Sec>
          </div>

          <Sec title="Roll Center Migration — Lateral G Loading"><GC><ResponsiveContainer width="100%" height={260}><ScatterChart margin={{ top: 10, right: 20, bottom: 24, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="rcXf" type="number" {...ax()} label={{ value: "RC Height [mm]", position: "bottom", offset: 6, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis dataKey="rcYf" type="number" {...ax()} label={{ value: "RC Lateral [mm]", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md } }} /><Tooltip contentStyle={TT} /><Scatter data={rcMig} fill={C.cy} fillOpacity={.5} r={3} name="Front RC" /><Scatter data={rcMig.map(d => ({ ...d, rcXf: d.rcXr, rcYf: d.rcYr }))} fill={C.am} fillOpacity={.5} r={3} name="Rear RC" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></ScatterChart></ResponsiveContainer></GC><Note>Minimal migration = stable load transfer characteristic. Front RC at ~40mm, rear ~60mm. Lateral migration should stay &lt;20mm under peak G.</Note></Sec>
        </>)}

        {/* ═══════════════════════════════════════════════════════════════
            GRID 3: CONTACT PATCH & ENERGY
            ═══════════════════════════════════════════════════════════════ */}
        {grid === "g3" && (<>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Dynamic Fz — 4-Corner Load Distribution"><GC><ResponsiveContainer width="100%" height={280}><AreaChart data={loads.filter((_,i) => i % 3 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis {...ax()} label={{ value: "Fz [N]", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md } }} /><Tooltip contentStyle={TT} /><Area type="monotone" dataKey="Fz_fl" stroke={C.cy} fill={`${C.cy}06`} strokeWidth={1.2} dot={false} name="FL" /><Area type="monotone" dataKey="Fz_fr" stroke={C.gn} fill={`${C.gn}06`} strokeWidth={1.2} dot={false} name="FR" /><Area type="monotone" dataKey="Fz_rl" stroke={C.am} fill={`${C.am}06`} strokeWidth={1.2} dot={false} name="RL" /><Area type="monotone" dataKey="Fz_rr" stroke={C.red} fill={`${C.red}06`} strokeWidth={1.2} dot={false} name="RR" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></AreaChart></ResponsiveContainer></GC></Sec>
            <Sec title="Friction Circle Saturation"><GC><ResponsiveContainer width="100%" height={280}><BarChart data={fricSat} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="bin" {...ax()} label={{ value: "µ Utilisation", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis {...ax()} /><Tooltip contentStyle={TT} /><ReferenceArea x1={0.85} x2={1.0} fill={C.red} fillOpacity={.06} /><Bar dataKey="count" radius={[2,2,0,0]} barSize={14}>{fricSat.map((e, i) => <Cell key={i} fill={e.bin >= .85 ? C.red : e.bin >= .6 ? C.am : C.cy} fillOpacity={.7} />)}</Bar></BarChart></ResponsiveContainer></GC><Note>{(() => { const gt90 = fricSat.filter(b => b.bin >= 0.9).reduce((a, b) => a + b.count, 0); const total = fricSat.reduce((a, b) => a + b.count, 0); return `${((gt90 / total) * 100).toFixed(1)}% of samples above 90% µ saturation — aggressive but sustainable.`; })()}</Note></Sec>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Slip Energy Dissipation [kJ/m]"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={slipE.filter((_,i) => i % 2 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis {...ax()} /><Tooltip contentStyle={TT} /><Area type="monotone" dataKey="energy" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.5} dot={false} name="Slip Energy [J]" /></AreaChart></ResponsiveContainer></GC><Note>Peak energy = corner exit (combined longitudinal + lateral slip). Sustained &gt;40 J/m risks thermal degradation on Hoosier R20 beyond optimal 60-95°C window.</Note></Sec>
            <Sec title="Understeer Gradient (Ku) Trace"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={Ku} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis domain={[-5, 5]} {...ax()} /><Tooltip contentStyle={TT} /><ReferenceArea y1={-1} y2={1} fill={C.gn} fillOpacity={.04} /><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" /><Area type="monotone" dataKey="Ku" stroke={C.pr} fill={`${C.pr}08`} strokeWidth={1.5} dot={false} name="Ku [deg/G]" /></AreaChart></ResponsiveContainer></GC><Note>Ku &gt; 0 = understeer, Ku &lt; 0 = oversteer. Green band = neutral ±1 deg/G. Snap oversteer = Ku discontinuity below -3.</Note></Sec>
          </div>
        </>)}

        {/* ═══════════════════════════════════════════════════════════════
            GRID 4: PORT-HAMILTONIAN ENERGY
            ═══════════════════════════════════════════════════════════════ */}
        {grid === "g4" && (<>
          <Sec title="Hamiltonian Energy State H(q,p) [kJ]"><GC><ResponsiveContainer width="100%" height={300}><AreaChart data={hamiltonian} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis {...ax()} /><Tooltip contentStyle={TT} /><Area type="monotone" dataKey="kinetic" stackId="1" stroke={C.cy} fill={`${C.cy}15`} strokeWidth={1.2} name="T (kinetic)" /><Area type="monotone" dataKey="potential" stackId="1" stroke={C.gn} fill={`${C.gn}15`} strokeWidth={1.2} name="V (potential)" /><Line type="monotone" dataKey="dissipated" stroke={C.red} strokeWidth={1.5} dot={false} name="Σ Dissipated" strokeDasharray="4 2" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></AreaChart></ResponsiveContainer></GC><Note>H = T + V + H_res. Dissipation monotonically increasing confirms R = LLᵀ passivity — no phantom energy injection from R_net.</Note></Sec>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Dissipation Network Breakdown"><GC style={{ padding: "16px" }}>
              {dissipation.map(d => {const maxE = Math.max(...dissipation.map(x => x.energy)); return (
                <div key={d.component} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <div style={{ width: 100, fontSize: 9, color: C.md, fontFamily: C.dt, textAlign: "right" }}>{d.component}</div>
                  <div style={{ flex: 1, height: 16, background: C.b1, borderRadius: 4, overflow: "hidden" }}>
                    <div style={{ width: `${(d.energy / maxE) * 100}%`, height: "100%", background: d.color, borderRadius: 4, transition: "width 0.5s ease" }} />
                  </div>
                  <div style={{ width: 50, fontSize: 10, color: C.br, fontFamily: C.dt, fontWeight: 600 }}>{d.energy} J</div>
                </div>
              );})}
              <Note>Total dissipation: {dissipation.reduce((a, d) => a + d.energy, 0)} J. Tire slip dominates at {((dissipation[0].energy + dissipation[1].energy) / dissipation.reduce((a, d) => a + d.energy, 0) * 100).toFixed(0)}% — grip limited, not drag limited.</Note>
            </GC></Sec>
            <Sec title="Regen Braking Envelope (EV)"><GC><ResponsiveContainer width="100%" height={260}><AreaChart data={regen} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="speed" {...ax()} label={{ value: "Speed [m/s]", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis {...ax()} label={{ value: "Regen Power [kW]", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md } }} /><Tooltip contentStyle={TT} /><Area type="monotone" dataKey="motorMax" stroke={C.cy} fill={`${C.cy}06`} strokeWidth={1} dot={false} name="Motor Limit" strokeDasharray="4 2" /><Area type="monotone" dataKey="batteryLimit" stroke={C.am} fill={`${C.am}06`} strokeWidth={1} dot={false} name="Battery Limit" strokeDasharray="4 2" /><Line type="monotone" dataKey="actual" stroke={C.gn} strokeWidth={2} dot={false} name="Actual Regen" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></AreaChart></ResponsiveContainer></GC></Sec>
          </div>

          <Sec title="Torque Vectoring Yaw Moment"><GC><ResponsiveContainer width="100%" height={220}><BarChart data={torqueVec.filter((_,i) => i % 2 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis {...ax()} /><Tooltip contentStyle={TT} /><Bar dataKey="mechYaw" stackId="a" fill={C.cy} fillOpacity={.5} name="Mechanical [Nm]" /><Bar dataKey="activeYaw" stackId="a" fill={C.gn} fillOpacity={.6} name="Active Diff [Nm]" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></BarChart></ResponsiveContainer></GC></Sec>
        </>)}

        {/* ═══════════════════════════════════════════════════════════════
            GRID 5: DIFF-WMPC HORIZON
            ═══════════════════════════════════════════════════════════════ */}
        {grid === "g5" && (<>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Receding Horizon — 64-step Prediction"><GC><ResponsiveContainer width="100%" height={300}><ComposedChart data={horizon} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="xPred" type="number" {...ax()} label={{ value: "X [m]", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis dataKey="yPred" type="number" {...ax()} label={{ value: "Y [m]", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md } }} /><Tooltip contentStyle={TT} /><Line type="monotone" dataKey="yBoundL" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3" name="Track L" /><Line type="monotone" dataKey="yBoundR" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3" name="Track R" /><Line type="monotone" dataKey="yPred" stroke={C.cy} strokeWidth={2} dot={false} name="Predicted" /></ComposedChart></ResponsiveContainer></GC></Sec>
            <Sec title="Wavelet Coefficient Sparsity"><GC style={{ padding: "16px" }}>
              {["cA3","cD3","cD2","cD1"].map(lv => {const coeffs = wavelets.filter(w => w.level === lv); const activeN = coeffs.filter(w => w.active).length; return (
                <div key={lv} style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 10, fontFamily: C.dt, color: lv === "cA3" ? C.cy : C.am, fontWeight: 700 }}>{lv}</span>
                    <span style={{ fontSize: 9, fontFamily: C.dt, color: C.dm }}>{activeN}/{coeffs.length} active</span>
                  </div>
                  <div style={{ display: "flex", gap: 2 }}>
                    {coeffs.map((w, j) => <div key={j} style={{ flex: 1, height: 20, borderRadius: 2, background: w.active ? (lv === "cA3" ? C.cy : C.am) : C.b1, opacity: 0.3 + w.mag * 0.7, transition: "all 0.3s" }} />)}
                  </div>
                </div>
              );})}
              <Note>Db4 3-level DWT compresses 128 time steps → {wavelets.filter(w => w.active).length} active coefficients. Sparsity ratio: {((1 - wavelets.filter(w => w.active).length / wavelets.length) * 100).toFixed(0)}% — proving smooth human-drivable control trajectories.</Note>
            </GC></Sec>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="AL Constraint Slack — Grip & Steering"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={alSlack} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis {...ax()} /><Tooltip contentStyle={TT} /><ReferenceArea y1={0} y2={0.1} fill={C.red} fillOpacity={.06} /><Area type="monotone" dataKey="slackGrip" stroke={C.red} fill={`${C.red}08`} strokeWidth={1.5} dot={false} name="Grip Slack [G]" /><Area type="monotone" dataKey="slackSteer" stroke={C.am} fill={`${C.am}06`} strokeWidth={1.2} dot={false} name="Steer Slack [rad]" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></AreaChart></ResponsiveContainer></GC><Note>Red zone = slack &lt; 0.1G → AL penalty activates quadratically. Zero slack = riding the friction circle limit.</Note></Sec>
            <Sec title="Control Effort Saturation"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={ctrlEffort} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax()} /><YAxis domain={[0, 1]} {...ax()} /><Tooltip contentStyle={TT} /><ReferenceArea y1={0.9} y2={1} fill={C.red} fillOpacity={.06} /><Line type="monotone" dataKey="steerUtil" stroke={C.pr} strokeWidth={1.5} dot={false} name="Steer %" /><Line type="monotone" dataKey="brakeUtil" stroke={C.red} strokeWidth={1.5} dot={false} name="Brake %" /><Line type="monotone" dataKey="throttleUtil" stroke={C.gn} strokeWidth={1.5} dot={false} name="Throttle %" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></AreaChart></ResponsiveContainer></GC><Note>&gt;90% sustained = actuator-limited. If steering saturation correlates with laptime loss, increase rack ratio or widen steering authority bounds.</Note></Sec>
          </div>
        </>)}

        {/* ═══════════════════════════════════════════════════════════════
            GRID 6: TWIN FIDELITY & EKF
            ═══════════════════════════════════════════════════════════════ */}
        {grid === "g6" && (<>
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
            <Sec title="EKF Innovation Sequence (ỹ = z − Hx)"><GC><ResponsiveContainer width="100%" height={280}><LineChart data={ekf.filter((_,i) => i % 2 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="t" {...ax()} /><YAxis {...ax()} /><Tooltip contentStyle={TT} /><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" /><ReferenceArea y1={-0.02} y2={0.02} fill={C.gn} fillOpacity={.04} /><Line type="monotone" dataKey="innov_ax" stroke={C.cy} strokeWidth={1} dot={false} name="a_x innov" /><Line type="monotone" dataKey="innov_wz" stroke={C.am} strokeWidth={1} dot={false} name="ω_z innov" /><Line type="monotone" dataKey="innov_vy" stroke={C.pr} strokeWidth={1} dot={false} name="v_y innov" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></LineChart></ResponsiveContainer></GC><Note>Innovation → 0 = perfect twin accuracy. Green band = ±0.02 (2σ). Persistent bias → model misspecification in H_net or tire model.</Note></Sec>
            <Sec title="Twin Fidelity Breakdown">
              <GC style={{ padding: "14px", height: "100%" }}>
                <ResponsiveContainer width="100%" height={240}>
                  <RadarChart data={fidelity} cx="50%" cy="50%" outerRadius="72%">
                    <PolarGrid stroke={C.b1} />
                    <PolarAngleAxis dataKey="axis" tick={{ fontSize: 8, fill: C.md, fontFamily: C.dt }} />
                    <PolarRadiusAxis angle={90} domain={[70, 100]} tick={{ fontSize: 8, fill: C.dm }} />
                    <Radar dataKey="score" stroke={C.cy} fill={C.cy} fillOpacity={0.15} strokeWidth={2} />
                  </RadarChart>
                </ResponsiveContainer>
                <div style={{ textAlign: "center", marginTop: 4 }}>
                  <span style={{ fontSize: 22, fontWeight: 800, color: C.gn, fontFamily: C.hd }}>{fidelityAvg}%</span>
                  <span style={{ fontSize: 10, color: C.dm, fontFamily: C.dt, marginLeft: 8 }}>aggregate</span>
                </div>
              </GC>
            </Sec>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="Pacejka Coefficient Drift (Online Learning)"><GC><ResponsiveContainer width="100%" height={250}><ComposedChart data={pacDrift} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="lap" {...ax()} label={{ value: "Lap #", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis yAxisId="l" {...ax()} domain={["auto","auto"]} /><YAxis yAxisId="r" orientation="right" {...ax()} /><Tooltip contentStyle={TT} /><Line yAxisId="l" type="monotone" dataKey="muY_pct" stroke={C.am} strokeWidth={2} dot={{ r: 3, fill: C.am }} name="µ_y [% of fresh]" /><Bar yAxisId="r" dataKey="stiffness" fill={C.cy} fillOpacity={.3} name="C_α [N/rad]" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></ComposedChart></ResponsiveContainer></GC><Note>jax.grad updates λ_µy at 100 Hz. µ_y degradation of {(100 - pacDrift[pacDrift.length - 1].muY_pct).toFixed(1)}% over {pacDrift.length} laps matches TTC Hoosier R20 thermal wear profile.</Note></Sec>
            <Sec title="Real vs Simulated PSD"><GC><ResponsiveContainer width="100%" height={250}><LineChart data={psd} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="freq" {...ax()} label={{ value: "Frequency [Hz]", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis {...ax()} label={{ value: "PSD [dB]", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md } }} /><Tooltip contentStyle={TT} /><Line type="monotone" dataKey="real" stroke={C.am} strokeWidth={1.5} dot={false} name="Real Damper" /><Line type="monotone" dataKey="sim" stroke={C.cy} strokeWidth={1.5} dot={false} name="Sim (H_net)" strokeDasharray="4 2" /><Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} /></LineChart></ResponsiveContainer></GC><Note>PSD overlap validates H_net frequency content matches real suspension travel. Divergence above 20 Hz = unmodeled high-frequency dynamics (tyre cavity resonance).</Note></Sec>
          </div>
        </>)}
      </FadeSlide>
    </>
  );
}
