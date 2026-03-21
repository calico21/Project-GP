import React, { useState, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill, FadeSlide } from "./components.jsx";

// Pure SVG track map — avoids Recharts Scatter+Cell crash entirely
function TrackMapSVG({ track }) {
  const xs = track.map(p => p.x), ys = track.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const W = 640, H = 280, pad = 30;
  const sc = Math.min((W - 2 * pad) / (xMax - xMin || 1), (H - 2 * pad) / (yMax - yMin || 1));
  const ox = pad + (W - 2 * pad - (xMax - xMin) * sc) / 2;
  const oy = pad + (H - 2 * pad - (yMax - yMin) * sc) / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%" }}>
      {track.map((p, i) => {
        if (i === 0) return null;
        const prev = track[i - 1];
        const x1 = ox + (prev.x - xMin) * sc, y1 = H - (oy + (prev.y - yMin) * sc);
        const x2 = ox + (p.x - xMin) * sc, y2 = H - (oy + (p.y - yMin) * sc);
        const ratio = Math.min(1, Math.max(0, (p.speed - 8) / 16));
        const r = Math.round(225 * (1 - ratio));
        const g = Math.round(40 + 190 * ratio);
        const b = Math.round(60 + 140 * (1 - Math.abs(ratio - 0.5) * 2));
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={`rgb(${r},${g},${b})`} strokeWidth={2.5} strokeLinecap="round" />;
      })}
      <circle cx={ox + (track[0].x - xMin) * sc} cy={H - (oy + (track[0].y - yMin) * sc)} r={5} fill={C.gn} stroke="#fff" strokeWidth={1.5} />
    </svg>
  );
}

export default function TelemetryModule({ track, tireTemps }) {
  const [ch, setCh] = useState("speed");
  const ggData = useMemo(() => track.filter((_, i) => i % 3 === 0).map(p => ({ lat: Number(p.lat_g), lon: Number(p.lon_g) })), [track]);
  const vMax = useMemo(() => Math.max(...track.map(p => p.speed)), [track]);
  const latMax = useMemo(() => Math.max(...track.map(p => Math.abs(p.lat_g))), [track]);
  const tMax = useMemo(() => Math.max(...tireTemps.map(t => Math.max(t.tfl, t.tfr, t.trl, t.trr))), [tireTemps]);

  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 18 }}>
        <KPI label="V_max" value={vMax.toFixed(1)} sub="km/h" sentiment="positive" delay={0} />
        <KPI label="Peak Lat G" value={latMax.toFixed(2)} sub="G lateral" sentiment="amber" delay={1} />
        <KPI label="Track" value={`${track[track.length - 1].s}m`} sub="total length" delay={2} />
        <KPI label="T_max Tyre" value={`${tMax.toFixed(0)}°C`} sub="surface peak" sentiment={tMax > 95 ? "negative" : "positive"} delay={3} />
      </div>

      <Sec title="Racing Line — Speed Coloured">
        <GC style={{ padding: "18px 14px" }}>
          <div style={{ width: "100%", height: 280 }}><TrackMapSVG track={track} /></div>
          <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 8 }}>
            <span style={{ fontSize: 9, fontWeight: 700, color: C.red, fontFamily: C.dt }}>◀ SLOW</span>
            <div style={{ width: 100, height: 5, borderRadius: 3, background: `linear-gradient(90deg, ${C.red}, ${C.am}, ${C.gn})` }} />
            <span style={{ fontSize: 9, fontWeight: 700, color: C.gn, fontFamily: C.dt }}>FAST ▶</span>
          </div>
        </GC>
      </Sec>

      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {[["speed", "Speed"], ["lat_g", "Lat G"], ["curvature", "κ Curvature"]].map(([k, l]) => (
          <Pill key={k} active={ch === k} label={l} onClick={() => setCh(k)} color={C.cy} />
        ))}
      </div>

      <FadeSlide keyVal={ch}>
        <Sec title={`Channel: ${ch.replace("_", " ").toUpperCase()}`}>
          <GC>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={track} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="s" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                <YAxis tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                <Tooltip contentStyle={TT} />
                <Area type="monotone" dataKey={ch} stroke={ch === "speed" ? C.cy : ch === "lat_g" ? C.am : C.pr} fill={ch === "speed" ? `${C.cy}08` : ch === "lat_g" ? `${C.am}08` : `${C.pr}08`} strokeWidth={1.5} dot={false} />
                {ch === "lat_g" && <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />}
              </AreaChart>
            </ResponsiveContainer>
          </GC>
        </Sec>
      </FadeSlide>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Sec title="G-G Diagram">
          <GC>
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 10, right: 10, bottom: 24, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="lat" type="number" domain={[-2, 2]} tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Lateral [G]", position: "bottom", offset: 6, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} />
                <YAxis dataKey="lon" type="number" domain={[-2, 2]} tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} label={{ value: "Lon [G]", angle: -90, position: "insideLeft", offset: 8, style: { fontSize: 9, fill: C.md, fontFamily: C.hd } }} />
                <ReferenceLine x={0} stroke={C.dm} strokeWidth={0.5} />
                <ReferenceLine y={0} stroke={C.dm} strokeWidth={0.5} />
                <Scatter data={ggData} fill={C.red} fillOpacity={0.4} r={2.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </GC>
        </Sec>
        <Sec title="Tyre Thermals — 60–95°C Optimal">
          <GC>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={tireTemps} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GS} />
                <XAxis dataKey="s" tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                <YAxis domain={[20, 135]} tick={{ fontSize: 9, fill: C.dm, fontFamily: C.dt }} stroke={C.b1} />
                <Tooltip contentStyle={TT} />
                <ReferenceArea y1={60} y2={95} fill={C.gn} fillOpacity={0.04} label={{ value: "OPTIMAL", fill: `${C.gn}50`, fontSize: 8, fontFamily: C.dt }} />
                <Line type="monotone" dataKey="tfl" stroke={C.cy} strokeWidth={1.3} dot={false} name="FL" />
                <Line type="monotone" dataKey="tfr" stroke={C.gn} strokeWidth={1.3} dot={false} name="FR" />
                <Line type="monotone" dataKey="trl" stroke={C.am} strokeWidth={1.3} dot={false} name="RL" />
                <Line type="monotone" dataKey="trr" stroke={C.red} strokeWidth={1.3} dot={false} name="RR" />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: C.hd, fontWeight: 600 }} />
              </LineChart>
            </ResponsiveContainer>
          </GC>
        </Sec>
      </div>
    </>
  );
}
