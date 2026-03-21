import React, { useState, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
  ComposedChart,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill, FadeSlide } from "./components.jsx";
import { PN, PU, gLoadTransfer, gDamperHist, gFreqResponse, gLapDelta, gGripUtil } from "./data.js";

// ── Parallel Coordinates SVG ────────────────────────────────────────
function ParCoords({ pareto }) {
  const W = 820, H = 360, pd = { t: 40, b: 40, l: 20, r: 20 };
  const axes = PN.slice(0, 16), nA = axes.length, xS = (W - pd.l - pd.r) / (nA - 1);
  const yT = pd.t, yB = H - pd.b, bG = Math.max(...pareto.map(p => p.grip));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%" }}>
      <defs>
        <linearGradient id="pcG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor={C.cy} stopOpacity="0.5" /><stop offset="50%" stopColor={C.gn} stopOpacity="0.5" /><stop offset="100%" stopColor={C.am} stopOpacity="0.5" /></linearGradient>
        <linearGradient id="pcB" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor={C.red} /><stop offset="100%" stopColor="#ff6040" /></linearGradient>
      </defs>
      {axes.map((a, i) => { const x = pd.l + i * xS; return <g key={a}><line x1={x} y1={yT} x2={x} y2={yB} stroke={C.b1} /><text x={x} y={yT - 10} textAnchor="middle" fill={C.md} fontSize={8} fontFamily="Outfit" fontWeight={600} transform={`rotate(-40 ${x} ${yT - 10})`}>{a}</text><text x={x} y={yB + 14} textAnchor="middle" fill={C.dm} fontSize={7} fontFamily="Azeret Mono">{PU[i]}</text></g>; })}
      {pareto.map((s, si) => { const best = s.grip >= bG - 0.01; return <polyline key={si} points={axes.map((_, i) => `${pd.l + i * xS},${yT + (1 - (s.params[i] || 0)) * (yB - yT)}`).join(" ")} fill="none" stroke={best ? "url(#pcB)" : "url(#pcG)"} strokeWidth={best ? 2.5 : 1} strokeOpacity={best ? 1 : 0.16} />; })}
    </svg>
  );
}

// ── Sub-tab definitions ─────────────────────────────────────────────
const TABS = [
  { key: "pareto", label: "Pareto" },
  { key: "convergence", label: "Convergence" },
  { key: "sensitivity", label: "Sensitivity" },
  { key: "parallel", label: "Parallel" },
  { key: "loads", label: "Fz Loads" },
  { key: "damper", label: "Damper" },
  { key: "freq", label: "Freq Resp" },
  { key: "delta", label: "Lap Delta" },
];

export default function SetupModule({ pareto, conv, sens, track }) {
  const [sub, setSub] = useState("pareto");
  const bG = Math.max(...pareto.map(p => p.grip));
  const bS = Math.min(...pareto.filter(p => p.grip > bG - 0.05).map(p => p.stability));

  // Derived data — only computed when track exists and tab is active
  const loads = useMemo(() => track ? gLoadTransfer(track) : [], [track]);
  const damperHist = useMemo(() => gDamperHist(), []);
  const freqResp = useMemo(() => gFreqResponse(), []);
  const lapDelta = useMemo(() => track ? gLapDelta(track) : [], [track]);
  const gripUtil = useMemo(() => track ? gGripUtil(track) : [], [track]);

  return (
    <>
      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 18 }}>
        <KPI label="Peak Grip" value={`${bG.toFixed(3)}G`} sub="Pareto Max" sentiment="positive" delay={0} />
        <KPI label="Stability" value={bS.toFixed(2)} sub="rad/s @ peak" sentiment="amber" delay={1} />
        <KPI label="Pareto" value={pareto.length} sub="setups" delay={2} />
        <KPI label="Converged" value={conv[conv.length - 1].bestGrip.toFixed(3)} sub={`${conv.length} iter`} sentiment="positive" delay={3} />
        <KPI label="Grip Util" value={gripUtil.length > 0 ? `${(gripUtil.reduce((a, g) => a + g.utilisation, 0) / gripUtil.length * 100).toFixed(0)}%` : "—"} sub="mean µ usage" sentiment="amber" delay={4} />
      </div>

      {/* Tab pills */}
      <div style={{ display: "flex", gap: 5, marginBottom: 16, flexWrap: "wrap" }}>
        {TABS.map(t => <Pill key={t.key} active={sub === t.key} label={t.label} onClick={() => setSub(t.key)} color={C.red} />)}
      </div>

      <FadeSlide keyVal={sub}>
        {/* ── PARETO ──────────────────────────────────── */}
        {sub === "pareto" && (
          <Sec title="Pareto Front — Grip vs Stability Overshoot">
            <GC><ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 28, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="grip" type="number" domain={["auto", "auto"]} tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Lateral Grip [G]", position: "bottom", offset: 10, style: { fontSize: 10, fill: C.md, fontFamily: C.hd, fontWeight: 600 } }} />
                <YAxis dataKey="stability" type="number" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Stability [rad/s]", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 10, fill: C.md, fontFamily: C.hd, fontWeight: 600 } }} />
                <Tooltip contentStyle={TT} />
                <ReferenceArea y1={0} y2={5} fill={C.gn} fillOpacity={0.03} />
                <ReferenceLine y={5} stroke={C.red} strokeDasharray="6 3" label={{ value: "5.0 rad/s cap", fill: C.red, fontSize: 9, fontFamily: C.dt }} />
                <Scatter data={pareto} fillOpacity={0.85} r={5}>{pareto.map((e, i) => <Cell key={i} fill={e.grip >= bG - 0.01 ? C.gn : C.cy} />)}</Scatter>
              </ScatterChart>
            </ResponsiveContainer></GC>
          </Sec>
        )}

        {/* ── CONVERGENCE ─────────────────────────────── */}
        {sub === "convergence" && (<>
          <Sec title="Best Grip Convergence">
            <GC><ResponsiveContainer width="100%" height={280}>
              <AreaChart data={conv} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="iter" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><YAxis tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} domain={["auto", "auto"]} /><Tooltip contentStyle={TT} />
                <Area type="monotone" dataKey="bestGrip" stroke={C.red} fill={`${C.red}0c`} strokeWidth={2} dot={false} name="Best Grip [G]" />
              </AreaChart>
            </ResponsiveContainer></GC>
          </Sec>
          <Sec title="Trust Region: D_KL & Entropy">
            <GC><ResponsiveContainer width="100%" height={180}>
              <LineChart data={conv} margin={{ top: 8, right: 20, bottom: 4, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="iter" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><Tooltip contentStyle={TT} />
                <Line type="monotone" dataKey="kl" stroke={C.am} strokeWidth={1.5} dot={false} name="D_KL" />
                <Line type="monotone" dataKey="entropy" stroke={C.pr} strokeWidth={1.5} dot={false} name="Σlog(σ)" />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd }} />
              </LineChart>
            </ResponsiveContainer></GC>
          </Sec>
        </>)}

        {/* ── SENSITIVITY ─────────────────────────────── */}
        {sub === "sensitivity" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="∂Grip / ∂Parameter"><GC><ResponsiveContainer width="100%" height={380}>
              <BarChart data={sens} layout="vertical" margin={{ top: 10, right: 20, bottom: 10, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} /><XAxis type="number" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><YAxis dataKey="param" type="category" tick={{ fontSize: 10, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={55} /><Tooltip contentStyle={TT} /><ReferenceLine x={0} stroke={C.dm} />
                <Bar dataKey="dGrip" name="∂Grip" radius={[0, 4, 4, 0]} barSize={12}>{sens.map((e, i) => <Cell key={i} fill={e.dGrip > 0 ? C.gn : C.red} fillOpacity={0.8} />)}</Bar>
              </BarChart>
            </ResponsiveContainer></GC></Sec>
            <Sec title="∂Stability / ∂Parameter"><GC><ResponsiveContainer width="100%" height={380}>
              <BarChart data={sens} layout="vertical" margin={{ top: 10, right: 20, bottom: 10, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} /><XAxis type="number" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><YAxis dataKey="param" type="category" tick={{ fontSize: 10, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={55} /><Tooltip contentStyle={TT} /><ReferenceLine x={0} stroke={C.dm} />
                <Bar dataKey="dStab" name="∂Stab" radius={[0, 4, 4, 0]} barSize={12}>{sens.map((e, i) => <Cell key={i} fill={e.dStab > 0 ? C.am : C.pr} fillOpacity={0.8} />)}</Bar>
              </BarChart>
            </ResponsiveContainer></GC></Sec>
          </div>
        )}

        {/* ── PARALLEL COORDS ─────────────────────────── */}
        {sub === "parallel" && (
          <Sec title="Parallel Coordinates — 28D Setup (16 Shown)">
            <GC style={{ padding: "16px 14px" }}>
              <div style={{ width: "100%", height: 360 }}><ParCoords pareto={pareto} /></div>
              <div style={{ display: "flex", justifyContent: "center", gap: 20, marginTop: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, fontFamily: C.dt, color: C.md }}><div style={{ width: 24, height: 2, background: `linear-gradient(90deg,${C.cy},${C.gn},${C.am})`, borderRadius: 2 }} />Population</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, fontFamily: C.dt, color: C.red }}><div style={{ width: 24, height: 2.5, background: `linear-gradient(90deg,${C.red},#ff6040)`, borderRadius: 2 }} />Best Grip</div>
              </div>
            </GC>
          </Sec>
        )}

        {/* ── Fz LOAD TRANSFER ────────────────────────── */}
        {sub === "loads" && (
          <Sec title="Vertical Load Distribution — Fz per Corner">
            <GC><ResponsiveContainer width="100%" height={320}>
              <AreaChart data={loads.filter((_, i) => i % 3 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Distance [m]", position: "bottom", offset: 2, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Fz [N]", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><Tooltip contentStyle={TT} />
                <Area type="monotone" dataKey="Fz_fl" stroke={C.cy} fill={`${C.cy}08`} strokeWidth={1.3} dot={false} name="FL" />
                <Area type="monotone" dataKey="Fz_fr" stroke={C.gn} fill={`${C.gn}08`} strokeWidth={1.3} dot={false} name="FR" />
                <Area type="monotone" dataKey="Fz_rl" stroke={C.am} fill={`${C.am}08`} strokeWidth={1.3} dot={false} name="RL" />
                <Area type="monotone" dataKey="Fz_rr" stroke={C.red} fill={`${C.red}08`} strokeWidth={1.3} dot={false} name="RR" />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd, fontWeight: 600 }} />
              </AreaChart>
            </ResponsiveContainer></GC>
            <div style={{ marginTop: 12, ...GL, padding: "12px 16px", fontSize: 11, color: C.md, fontFamily: C.dt, lineHeight: 1.7 }}>
              Load transfer model: lateral dFz = m·g·|a_lat|·h_cg/tw, longitudinal dFz = m·g·a_lon·h_cg/L. Static weight split: {((0.6975 / 1.55) * 100).toFixed(0)}% front / {((0.8525 / 1.55) * 100).toFixed(0)}% rear (rear-biased by {((0.8525 - 0.6975) / 1.55 * 100).toFixed(0)}%).
            </div>
          </Sec>
        )}

        {/* ── DAMPER HISTOGRAM ────────────────────────── */}
        {sub === "damper" && (
          <Sec title="Damper Velocity Histogram — Piston Speed Distribution">
            <GC><ResponsiveContainer width="100%" height={320}>
              <BarChart data={damperHist} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="vel" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Piston Velocity [m/s]", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Sample Count", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><Tooltip contentStyle={TT} />
                <Bar dataKey="front" fill={C.cy} fillOpacity={0.6} name="Front Axle" radius={[2, 2, 0, 0]} />
                <Bar dataKey="rear" fill={C.am} fillOpacity={0.6} name="Rear Axle" radius={[2, 2, 0, 0]} />
                <ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3" />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd, fontWeight: 600 }} />
              </BarChart>
            </ResponsiveContainer></GC>
            <div style={{ marginTop: 12, ...GL, padding: "12px 16px", fontSize: 11, color: C.md, fontFamily: C.dt, lineHeight: 1.7 }}>
              Low-speed threshold at ±0.05 m/s defines the crossover between low-speed and high-speed damper maps. ~72% of operation occurs within this region — low-speed damping is the primary handling tuning parameter. Negative velocity = rebound, positive = compression.
            </div>
          </Sec>
        )}

        {/* ── FREQUENCY RESPONSE ──────────────────────── */}
        {sub === "freq" && (
          <Sec title="Heave Frequency Response — Ride/Handling Bode Plot">
            <GC><ResponsiveContainer width="100%" height={320}>
              <LineChart data={freqResp} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="freq" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Frequency [Hz]", position: "bottom", offset: 4, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Magnitude [dB]", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} /><Tooltip contentStyle={TT} />
                <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
                <ReferenceArea x1={1} x2={3} fill={C.gn} fillOpacity={0.04} label={{ value: "RIDE", fill: `${C.gn}50`, fontSize: 8, fontFamily: C.dt }} />
                <ReferenceArea x1={8} x2={15} fill={C.am} fillOpacity={0.04} label={{ value: "HANDLING", fill: `${C.am}50`, fontSize: 8, fontFamily: C.dt }} />
                <Line type="monotone" dataKey="front_dB" stroke={C.cy} strokeWidth={2} dot={false} name="Front Axle" />
                <Line type="monotone" dataKey="rear_dB" stroke={C.am} strokeWidth={2} dot={false} name="Rear Axle" />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd, fontWeight: 600 }} />
              </LineChart>
            </ResponsiveContainer></GC>
            <div style={{ marginTop: 12, ...GL, padding: "12px 16px", fontSize: 11, color: C.md, fontFamily: C.dt, lineHeight: 1.7 }}>
              Front natural frequency: 1.8 Hz (ride comfort), rear: 2.1 Hz (handling response). Peak at resonance = 1/(2ζ) gain. Damping ratio ζ ≈ 0.35 — underdamped for maximum tyre contact patch compliance. The 8–15 Hz handling band must be flat (≤ 0 dB) to avoid roll oscillation amplification.
            </div>
          </Sec>
        )}

        {/* ── LAP DELTA ───────────────────────────────── */}
        {sub === "delta" && (
          <>
            <Sec title="Lap Time Delta — Actual vs Optimal">
              <GC><ResponsiveContainer width="100%" height={200}>
                <AreaChart data={lapDelta.filter((_, i) => i % 2 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><Tooltip contentStyle={TT} />
                  <ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" label={{ value: "Optimal", fill: C.gn, fontSize: 8, fontFamily: C.dt }} />
                  <Area type="monotone" dataKey="delta" stroke={C.red} fill={`${C.red}12`} strokeWidth={1.8} dot={false} name="Cum. Delta [s]" />
                </AreaChart>
              </ResponsiveContainer></GC>
            </Sec>
            <Sec title="Speed Comparison — Actual (cyan) vs Optimal (green)">
              <GC><ResponsiveContainer width="100%" height={220}>
                <LineChart data={lapDelta.filter((_, i) => i % 2 === 0)} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><Tooltip contentStyle={TT} />
                  <Line type="monotone" dataKey="vActual" stroke={C.cy} strokeWidth={1.3} dot={false} name="Actual [km/h]" />
                  <Line type="monotone" dataKey="vOptimal" stroke={C.gn} strokeWidth={1.3} dot={false} name="Optimal [km/h]" strokeDasharray="5 3" />
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd, fontWeight: 600 }} />
                </LineChart>
              </ResponsiveContainer></GC>
            </Sec>
            <Sec title="Grip Utilisation — Combined µ Usage">
              <GC><ResponsiveContainer width="100%" height={180}>
                <AreaChart data={gripUtil} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} /><Tooltip contentStyle={TT} />
                  <ReferenceArea y1={0.85} y2={1.0} fill={C.red} fillOpacity={0.06} label={{ value: "LIMIT", fill: `${C.red}50`, fontSize: 8, fontFamily: C.dt }} />
                  <Area type="monotone" dataKey="utilisation" stroke={C.am} fill={`${C.am}0c`} strokeWidth={1.5} dot={false} name="µ Utilisation" />
                </AreaChart>
              </ResponsiveContainer></GC>
            </Sec>
          </>
        )}
      </FadeSlide>
    </>
  );
}
