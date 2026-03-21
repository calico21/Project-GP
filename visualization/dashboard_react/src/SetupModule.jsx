import React, { useState } from "react";
import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill, FadeSlide } from "./components.jsx";
import { PN, PU } from "./data.js";

function ParCoords({ pareto }) {
  const W = 820, H = 360, pd = { t: 40, b: 40, l: 20, r: 20 };
  const axes = PN.slice(0, 16);
  const nA = axes.length;
  const xS = (W - pd.l - pd.r) / (nA - 1);
  const yT = pd.t, yB = H - pd.b;
  const bG = Math.max(...pareto.map(p => p.grip));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%" }}>
      <defs>
        <linearGradient id="pcGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={C.cy} stopOpacity="0.5" />
          <stop offset="50%" stopColor={C.gn} stopOpacity="0.5" />
          <stop offset="100%" stopColor={C.am} stopOpacity="0.5" />
        </linearGradient>
        <linearGradient id="pcBest" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={C.red} />
          <stop offset="100%" stopColor="#ff6040" />
        </linearGradient>
      </defs>
      {axes.map((a, i) => {
        const x = pd.l + i * xS;
        return (
          <g key={a}>
            <line x1={x} y1={yT} x2={x} y2={yB} stroke={C.b1} strokeWidth={1} />
            <text x={x} y={yT - 10} textAnchor="middle" fill={C.md} fontSize={8} fontFamily="Outfit" fontWeight={600} transform={`rotate(-40 ${x} ${yT - 10})`}>{a}</text>
            <text x={x} y={yB + 14} textAnchor="middle" fill={C.dm} fontSize={7} fontFamily="Azeret Mono">{PU[i]}</text>
          </g>
        );
      })}
      {pareto.map((s, si) => {
        const best = s.grip >= bG - 0.01;
        const pts = axes.map((_, i) => `${pd.l + i * xS},${yT + (1 - (s.params[i] || 0)) * (yB - yT)}`).join(" ");
        return <polyline key={si} points={pts} fill="none" stroke={best ? "url(#pcBest)" : "url(#pcGrad)"} strokeWidth={best ? 2.5 : 1} strokeOpacity={best ? 1 : 0.16} />;
      })}
    </svg>
  );
}

export default function SetupModule({ pareto, conv, sens }) {
  const [sub, setSub] = useState("pareto");
  const bG = Math.max(...pareto.map(p => p.grip));
  const bS = Math.min(...pareto.filter(p => p.grip > bG - 0.05).map(p => p.stability));

  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 18 }}>
        <KPI label="Peak Grip" value={`${bG.toFixed(3)}G`} sub="Pareto Max" sentiment="positive" delay={0} />
        <KPI label="Stability" value={bS.toFixed(2)} sub="rad/s @ peak" sentiment="amber" delay={1} />
        <KPI label="Pareto" value={pareto.length} sub="setups" delay={2} />
        <KPI label="Converged" value={conv[conv.length - 1].bestGrip.toFixed(3)} sub={`${conv.length} iter`} sentiment="positive" delay={3} />
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        {["pareto", "convergence", "sensitivity", "parallel"].map(t => (
          <Pill key={t} active={sub === t} label={t === "parallel" ? "Parallel Coords" : t} onClick={() => setSub(t)} color={C.red} />
        ))}
      </div>

      <FadeSlide keyVal={sub}>
        {sub === "pareto" && (
          <Sec title="Pareto Front — Grip vs Stability">
            <GC>
              <ResponsiveContainer width="100%" height={340}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 28, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                  <XAxis dataKey="grip" type="number" domain={["auto", "auto"]} tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Lateral Grip [G]", position: "bottom", offset: 10, style: { fontSize: 10, fill: C.md, fontFamily: C.hd, fontWeight: 600 } }} />
                  <YAxis dataKey="stability" type="number" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Stability [rad/s]", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 10, fill: C.md, fontFamily: C.hd, fontWeight: 600 } }} />
                  <Tooltip contentStyle={TT} />
                  <ReferenceArea y1={0} y2={5} fill={C.gn} fillOpacity={0.03} />
                  <ReferenceLine y={5} stroke={C.red} strokeDasharray="6 3" label={{ value: "5.0 rad/s cap", fill: C.red, fontSize: 9, fontFamily: C.dt }} />
                  <Scatter data={pareto} fillOpacity={0.85} r={5}>
                    {pareto.map((e, i) => <Cell key={i} fill={e.grip >= bG - 0.01 ? C.gn : C.cy} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </GC>
          </Sec>
        )}

        {sub === "convergence" && (
          <>
            <Sec title="Best Grip Convergence">
              <GC>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={conv} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="iter" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                    <YAxis tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={TT} />
                    <Area type="monotone" dataKey="bestGrip" stroke={C.red} fill={`${C.red}0c`} strokeWidth={2} dot={false} name="Best Grip [G]" />
                  </AreaChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
            <Sec title="KL Divergence & Entropy">
              <GC>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={conv} margin={{ top: 8, right: 20, bottom: 4, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                    <XAxis dataKey="iter" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                    <YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                    <Tooltip contentStyle={TT} />
                    <Line type="monotone" dataKey="kl" stroke={C.am} strokeWidth={1.5} dot={false} name="D_KL" />
                    <Line type="monotone" dataKey="entropy" stroke={C.pr} strokeWidth={1.5} dot={false} name="Σlog(σ)" />
                    <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd }} />
                  </LineChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
          </>
        )}

        {sub === "sensitivity" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Sec title="∂Grip / ∂Parameter">
              <GC>
                <ResponsiveContainer width="100%" height={380}>
                  <BarChart data={sens} layout="vertical" margin={{ top: 10, right: 20, bottom: 10, left: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                    <YAxis dataKey="param" type="category" tick={{ fontSize: 10, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={55} />
                    <Tooltip contentStyle={TT} />
                    <ReferenceLine x={0} stroke={C.dm} />
                    <Bar dataKey="dGrip" name="∂Grip" radius={[0, 4, 4, 0]} barSize={12}>
                      {sens.map((e, i) => <Cell key={i} fill={e.dGrip > 0 ? C.gn : C.red} fillOpacity={0.8} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
            <Sec title="∂Stability / ∂Parameter">
              <GC>
                <ResponsiveContainer width="100%" height={380}>
                  <BarChart data={sens} layout="vertical" margin={{ top: 10, right: 20, bottom: 10, left: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 10, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                    <YAxis dataKey="param" type="category" tick={{ fontSize: 10, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={55} />
                    <Tooltip contentStyle={TT} />
                    <ReferenceLine x={0} stroke={C.dm} />
                    <Bar dataKey="dStab" name="∂Stab" radius={[0, 4, 4, 0]} barSize={12}>
                      {sens.map((e, i) => <Cell key={i} fill={e.dStab > 0 ? C.am : C.pr} fillOpacity={0.8} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </GC>
            </Sec>
          </div>
        )}

        {sub === "parallel" && (
          <Sec title="Parallel Coordinates — 28D Setup Space (16 Shown)">
            <GC style={{ padding: "16px 14px" }}>
              <div style={{ width: "100%", height: 360 }}><ParCoords pareto={pareto} /></div>
              <div style={{ display: "flex", justifyContent: "center", gap: 20, marginTop: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, fontFamily: C.dt, color: C.md }}>
                  <div style={{ width: 24, height: 2, background: `linear-gradient(90deg,${C.cy},${C.gn},${C.am})`, borderRadius: 2 }} />Population
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, fontFamily: C.dt, color: C.red }}>
                  <div style={{ width: 24, height: 2.5, background: `linear-gradient(90deg,${C.red},#ff6040)`, borderRadius: 2 }} />Best Grip
                </div>
              </div>
            </GC>
          </Sec>
        )}
      </FadeSlide>
    </>
  );
}
