import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
  BarChart, Bar, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill, FadeSlide } from "./components.jsx";
import {
  gTK, gTT as gTireT, gLoadTransfer, gLapDelta, gSlipEnergy, gFrictionSatHist,
  gFreqResponse, gDamperHist, gRideHeightHist, gRollCenterMig, gPSDOverlay,
  gEKFInnovation, gPacejkaDrift, gFidelitySpider, gHorizonTraj, gWaveletCoeffs,
  gALSlack, gControlEffort, gGripUtil,
} from "./data.js";

// ═══════════════════════════════════════════════════════════════════════
// SHARED HELPERS
// ═══════════════════════════════════════════════════════════════════════
const Lbl = ({ children, color }) => <span style={{ fontSize: 9, fontWeight: 700, color: color || C.dm, fontFamily: C.dt, letterSpacing: 2, textTransform: "uppercase" }}>{children}</span>;
const Val = ({ children, color, big }) => <span style={{ fontSize: big ? 22 : 14, fontWeight: 700, color: color || C.w, fontFamily: C.dt }}>{children}</span>;
const Bar_ = ({ value, max = 1, color, h = 6 }) => <div style={{ width: "100%", height: h, background: C.b1, borderRadius: 3, overflow: "hidden" }}><div style={{ width: `${Math.min(100, (value / max) * 100)}%`, height: "100%", background: color || C.cy, borderRadius: 3, transition: "width 0.15s" }} /></div>;
const StatusDot = ({ ok }) => <div style={{ width: 6, height: 6, borderRadius: "50%", background: ok ? C.gn : C.red, boxShadow: `0 0 6px ${ok ? C.gn : C.red}` }} />;
const Note = ({ children }) => <div style={{ ...GL, padding: "10px 14px", fontSize: 10, color: C.md, fontFamily: C.dt, lineHeight: 1.8, marginTop: 10 }}>{children}</div>;

// Threshold color
const tc = (v, lo, hi) => v < lo ? C.gn : v < hi ? C.am : C.red;

// ═══════════════════════════════════════════════════════════════════════
// LIVE DATA HOOK — simulates 30Hz telemetry stream
// ═══════════════════════════════════════════════════════════════════════
function useLiveData(active) {
  const [frame, setFrame] = useState(null);
  const ref = useRef(0);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      ref.current++;
      const t = ref.current * 0.033;
      const steer = 18 * Math.sin(t * 0.8);
      const throttle = Math.max(0, 0.7 + 0.3 * Math.cos(t * 1.2));
      const brake = Math.max(0, -0.5 * Math.cos(t * 1.2) + 0.1);
      const speed = 12 + 10 * Math.sin(t * 0.3) + 3 * Math.random();
      const latG = speed * speed * 0.003 * Math.sin(t * 0.8) / 9.81;
      const lonG = (throttle - brake) * 1.2;
      const yawRate = latG * 2.5;
      const psi = t * 0.4;
      const x = 50 * Math.cos(psi * 0.3);
      const y = 30 * Math.sin(psi * 0.3);
      setFrame({
        t: +(t % 120).toFixed(2), speed: +speed.toFixed(1), steer: +steer.toFixed(1),
        throttle: +Math.min(1, throttle).toFixed(3), brake: +Math.min(1, brake).toFixed(3),
        latG: +latG.toFixed(3), lonG: +lonG.toFixed(3), yawRate: +yawRate.toFixed(2),
        x: +x.toFixed(1), y: +y.toFixed(1), gear: Math.min(4, Math.floor(speed / 8) + 1),
        // 4-corner data
        Fz_fl: +(750 - latG * 200 - lonG * 100 + 30 * Math.random()).toFixed(0),
        Fz_fr: +(750 + latG * 200 - lonG * 100 + 30 * Math.random()).toFixed(0),
        Fz_rl: +(680 - latG * 180 + lonG * 120 + 25 * Math.random()).toFixed(0),
        Fz_rr: +(680 + latG * 180 + lonG * 120 + 25 * Math.random()).toFixed(0),
        T_fl: +(65 + 20 * Math.abs(latG) + 5 * Math.random()).toFixed(1),
        T_fr: +(68 + 22 * Math.abs(latG) + 5 * Math.random()).toFixed(1),
        T_rl: +(60 + 15 * Math.abs(latG) + 4 * Math.random()).toFixed(1),
        T_rr: +(62 + 16 * Math.abs(latG) + 4 * Math.random()).toFixed(1),
        slip_fl: +(latG * 3 + 0.5 * Math.random()).toFixed(2),
        slip_fr: +(-latG * 3 + 0.5 * Math.random()).toFixed(2),
        slip_rl: +(latG * 2.5 + 0.4 * Math.random()).toFixed(2),
        slip_rr: +(-latG * 2.5 + 0.4 * Math.random()).toFixed(2),
        damp_fl: +(0.05 * Math.sin(t * 5) + 0.02 * Math.random()).toFixed(3),
        damp_fr: +(0.04 * Math.sin(t * 5 + 1) + 0.02 * Math.random()).toFixed(3),
        damp_rl: +(0.03 * Math.sin(t * 5 + 2) + 0.015 * Math.random()).toFixed(3),
        damp_rr: +(0.035 * Math.sin(t * 5 + 3) + 0.015 * Math.random()).toFixed(3),
        // EV
        soc: +(85 - t * 0.08).toFixed(1), power: +(speed * throttle * 0.8).toFixed(1),
        regen: +(brake > 0.3 ? brake * speed * 0.3 : 0).toFixed(1),
        cellTempMax: +(38 + t * 0.01 + 2 * Math.random()).toFixed(1),
        cellTempMin: +(35 + t * 0.005 + Math.random()).toFixed(1),
        invTempL: +(52 + 10 * throttle + 3 * Math.random()).toFixed(1),
        invTempR: +(50 + 10 * throttle + 3 * Math.random()).toFixed(1),
        motorTempL: +(65 + 15 * throttle + 4 * Math.random()).toFixed(1),
        motorTempR: +(63 + 15 * throttle + 4 * Math.random()).toFixed(1),
        tvDelta: +((latG * 120) * (0.9 + 0.2 * Math.random())).toFixed(0),
        tvTarget: +(latG * 120).toFixed(0),
        // Twin health
        ekfInnov: +(0.015 * Math.sin(t * 8) + 0.008 * Math.random()).toFixed(4),
        lambda_mu: +(1.35 - t * 0.0003).toFixed(4),
        wmpcSolveMs: +(4 + 3 * Math.random()).toFixed(1),
        hamiltonianJ: +(0.5 * 300 * speed * speed / 1000).toFixed(1),
        constraint: t % 15 < 0.1 ? "WARN: Thermal Derating 85°C" : t % 22 < 0.1 ? "WARN: Max Slip Angle" : null,
      });
    }, 33);
    return () => clearInterval(id);
  }, [active]);
  return frame;
}

// ═══════════════════════════════════════════════════════════════════════
// LIVE MODE — Canvas Track Map
// ═══════════════════════════════════════════════════════════════════════
function LiveTrackMap({ frame, history }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const cv = canvasRef.current; if (!cv) return;
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    // Track outline (static ellipse for demo)
    ctx.strokeStyle = C.b2; ctx.lineWidth = 2; ctx.beginPath();
    ctx.ellipse(W / 2, H / 2, W * 0.38, H * 0.38, 0, 0, Math.PI * 2); ctx.stroke();
    // History trail
    if (history.length > 1) {
      for (let i = 1; i < history.length; i++) {
        const alpha = i / history.length;
        const h = history[i];
        const combinedG = Math.sqrt(h.latG * h.latG + h.lonG * h.lonG);
        const r = Math.min(1, combinedG / 1.5);
        ctx.strokeStyle = `rgba(${Math.round(225 * r)}, ${Math.round(230 * (1 - r))}, ${Math.round(80 + 140 * (1 - r))}, ${alpha * 0.6})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(W / 2 + history[i - 1].x * 2.5, H / 2 - history[i - 1].y * 2.5);
        ctx.lineTo(W / 2 + h.x * 2.5, H / 2 - h.y * 2.5);
        ctx.stroke();
      }
    }
    // Car dot
    if (frame) {
      const cx = W / 2 + frame.x * 2.5, cy = H / 2 - frame.y * 2.5;
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = C.gn; ctx.fill();
      ctx.beginPath(); ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.strokeStyle = `${C.gn}60`; ctx.lineWidth = 2; ctx.stroke();
    }
  }, [frame, history]);
  return <canvas ref={canvasRef} width={360} height={260} style={{ width: "100%", height: 260, borderRadius: 8 }} />;
}

// ═══════════════════════════════════════════════════════════════════════
// LIVE MODE — 4-Corner Cell
// ═══════════════════════════════════════════════════════════════════════
function CornerCell({ label, fz, temp, slip, dampV, color }) {
  return (
    <div style={{ ...GL, padding: "10px 12px", borderTop: `2px solid ${color}` }}>
      <div style={{ fontSize: 10, fontWeight: 700, color, fontFamily: C.dt, letterSpacing: 2, marginBottom: 8 }}>{label}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        <div><Lbl>Fz</Lbl><div><Val>{fz}</Val> <Lbl>N</Lbl></div></div>
        <div><Lbl>Temp</Lbl><div><Val color={tc(temp, 80, 100)}>{temp}</Val> <Lbl>°C</Lbl></div></div>
        <div><Lbl>Slip α</Lbl><div><Val>{slip}</Val> <Lbl>deg</Lbl></div></div>
        <div><Lbl>Damp V</Lbl><div><Val>{dampV}</Val> <Lbl>m/s</Lbl></div></div>
      </div>
      <div style={{ marginTop: 6 }}><Lbl>Grip Usage</Lbl><Bar_ value={Math.min(1, Math.abs(slip) / 8 + Math.abs(dampV) * 3)} color={tc(Math.abs(slip), 4, 7)} h={4} /></div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// LIVE MODE LAYOUT
// ═══════════════════════════════════════════════════════════════════════
function LiveMode({ mode }) {
  const frame = useLiveData(mode === "LIVE");
  const histRef = useRef([]);
  const [constraints, setConstraints] = useState([]);
  const [innovHist, setInnovHist] = useState([]);

  useEffect(() => {
    if (!frame) return;
    histRef.current = [...histRef.current.slice(-90), frame];
    setInnovHist(prev => [...prev.slice(-60), { t: frame.t, v: Number(frame.ekfInnov) }]);
    if (frame.constraint) setConstraints(prev => [{ t: frame.t, msg: frame.constraint }, ...prev.slice(0, 6)]);
  }, [frame]);

  if (!frame) return <div style={{ padding: 40, color: C.md, fontFamily: C.dt, textAlign: "center" }}>Waiting for telemetry stream…<br /><span style={{ fontSize: 10, color: C.dm }}>Connect physics_server.py on ws://localhost:5001</span></div>;

  const f = frame;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gridTemplateRows: "auto 1fr auto", gap: 12, minHeight: "calc(100vh - 200px)" }}>
      {/* ── P2: DRIVER INPUTS STRIP (top, full width) ─────────── */}
      <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10 }}>
        {[
          { lbl: "SPEED", val: f.speed, unit: "km/h", color: C.cy, max: 30 },
          { lbl: "GEAR", val: f.gear, unit: "", color: C.am },
          { lbl: "THROTTLE", val: (f.throttle * 100).toFixed(0), unit: "%", color: C.gn, bar: f.throttle },
          { lbl: "BRAKE", val: (f.brake * 100).toFixed(0), unit: "%", color: C.red, bar: f.brake },
          { lbl: "STEER", val: f.steer, unit: "°", color: C.pr },
          { lbl: "LAT G", val: f.latG, unit: "G", color: C.am },
          { lbl: "LON G", val: f.lonG, unit: "G", color: C.cy },
          { lbl: "YAW", val: f.yawRate, unit: "°/s", color: C.pr },
        ].map(d => (
          <div key={d.lbl} style={{ flex: 1, ...GL, padding: "8px 10px", borderTop: `2px solid ${d.color}`, textAlign: "center" }}>
            <Lbl color={d.color}>{d.lbl}</Lbl>
            <div style={{ margin: "4px 0" }}><Val big>{d.val}</Val>{d.unit && <span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 3 }}>{d.unit}</span>}</div>
            {d.bar !== undefined && <Bar_ value={d.bar} color={d.color} />}
          </div>
        ))}
      </div>

      {/* ── P1+P3: CENTER (track map + 4-corner) ──────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* Track map + FPV placeholder */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <GC style={{ padding: "12px", position: "relative" }}>
            <Lbl color={C.cy}>LIVE TRACK MAP</Lbl>
            <LiveTrackMap frame={f} history={histRef.current} />
            <div style={{ position: "absolute", bottom: 16, left: 16, display: "flex", gap: 8 }}>
              <div style={{ ...GL, padding: "3px 8px", fontSize: 9, fontFamily: C.dt, color: C.gn }}>EKF: LOCKED</div>
              <div style={{ ...GL, padding: "3px 8px", fontSize: 9, fontFamily: C.dt, color: C.cy }}>Lap {Math.floor(f.t / 30) + 1}</div>
            </div>
          </GC>
          <GC style={{ padding: "12px", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", background: "rgba(5,7,11,0.8)" }}>
            <Lbl color={C.dm}>FPV CAMERA FEED</Lbl>
            <div style={{ margin: "20px 0", fontSize: 11, color: C.dm, fontFamily: C.dt, textAlign: "center", lineHeight: 1.8 }}>
              RTSP stream from roll-hoop camera<br />
              <span style={{ color: C.am }}>Connect: rtsp://192.168.1.10:8554/fpv</span><br />
              <span style={{ fontSize: 9 }}>HUD overlay: Speed + WMPC trajectory</span>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <div style={{ ...GL, padding: "4px 10px", fontSize: 10, fontFamily: C.dt, color: C.red }}>● REC</div>
              <div style={{ ...GL, padding: "4px 10px", fontSize: 10, fontFamily: C.dt, color: C.dm }}>NO SIGNAL</div>
            </div>
          </GC>
        </div>

        {/* 4-corner matrix */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <CornerCell label="FRONT LEFT" fz={f.Fz_fl} temp={f.T_fl} slip={f.slip_fl} dampV={f.damp_fl} color={C.cy} />
          <CornerCell label="FRONT RIGHT" fz={f.Fz_fr} temp={f.T_fr} slip={f.slip_fr} dampV={f.damp_fr} color={C.gn} />
          <CornerCell label="REAR LEFT" fz={f.Fz_rl} temp={f.T_rl} slip={f.slip_rl} dampV={f.damp_rl} color={C.am} />
          <CornerCell label="REAR RIGHT" fz={f.Fz_rr} temp={f.T_rr} slip={f.slip_rr} dampV={f.damp_rr} color={C.red} />
        </div>
      </div>

      {/* ── P4: EV POWERTRAIN SIDEBAR ─────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <GC style={{ padding: "12px" }}>
          <Lbl color={C.gn}>BATTERY</Lbl>
          <div style={{ margin: "8px 0" }}><Val big color={tc(100 - f.soc, 30, 60)}>{f.soc}%</Val> <Lbl>SoC</Lbl></div>
          <Bar_ value={f.soc / 100} color={tc(100 - f.soc, 30, 60)} h={8} />
        </GC>
        <GC style={{ padding: "10px 12px" }}>
          <Lbl color={C.am}>POWER</Lbl>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
            <div><Val color={C.red}>{f.power}</Val> <Lbl>kW out</Lbl></div>
            <div><Val color={C.gn}>{f.regen}</Val> <Lbl>kW regen</Lbl></div>
          </div>
        </GC>
        {[
          { lbl: "CELL TEMP", vals: [["Max", f.cellTempMax, "°C"], ["Min", f.cellTempMin, "°C"], ["Δ", (f.cellTempMax - f.cellTempMin).toFixed(1), "°C"]], th: [45, 55] },
          { lbl: "INVERTER", vals: [["Left", f.invTempL, "°C"], ["Right", f.invTempR, "°C"]], th: [70, 85] },
          { lbl: "MOTOR", vals: [["Left", f.motorTempL, "°C"], ["Right", f.motorTempR, "°C"]], th: [80, 100] },
        ].map(g => (
          <GC key={g.lbl} style={{ padding: "8px 12px" }}>
            <Lbl>{g.lbl}</Lbl>
            {g.vals.map(([k, v, u]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
                <Lbl>{k}</Lbl><div><Val color={tc(v, g.th[0], g.th[1])}>{v}</Val><Lbl> {u}</Lbl></div>
              </div>
            ))}
          </GC>
        ))}
        <GC style={{ padding: "8px 12px" }}>
          <Lbl color={C.pr}>TORQUE VECTORING</Lbl>
          <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between" }}>
            <div><Lbl>Target</Lbl><div><Val>{f.tvTarget}</Val> <Lbl>Nm</Lbl></div></div>
            <div><Lbl>Actual</Lbl><div><Val color={Math.abs(f.tvDelta - f.tvTarget) < 20 ? C.gn : C.am}>{f.tvDelta}</Val> <Lbl>Nm</Lbl></div></div>
          </div>
        </GC>
      </div>

      {/* ── P5: TWIN HEALTH STRIP (bottom, full width) ────────── */}
      <div style={{ gridColumn: "1 / -1", display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 2fr", gap: 10 }}>
        {/* EKF Innovation */}
        <GC style={{ padding: "8px 12px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Lbl color={C.cy}>EKF INNOVATION</Lbl>
            <div><StatusDot ok={Math.abs(f.ekfInnov) < 0.03} /><Lbl> {Math.abs(f.ekfInnov) < 0.03 ? "NOMINAL" : "DRIFT"}</Lbl></div>
          </div>
          <div style={{ height: 40, marginTop: 4 }}>
            <svg viewBox={`0 0 ${innovHist.length} 40`} style={{ width: "100%", height: 40 }} preserveAspectRatio="none">
              <line x1={0} y1={20} x2={innovHist.length} y2={20} stroke={C.gn} strokeOpacity={0.15} strokeWidth={0.5} />
              {innovHist.length > 1 && <polyline points={innovHist.map((d, i) => `${i},${20 - d.v * 600}`).join(" ")} fill="none" stroke={C.cy} strokeWidth={1.2} />}
            </svg>
          </div>
        </GC>
        {/* Lambda */}
        <GC style={{ padding: "8px 12px", textAlign: "center" }}>
          <Lbl color={C.am}>λ_μ FRICTION</Lbl>
          <div style={{ margin: "6px 0" }}><Val big color={C.am}>{f.lambda_mu}</Val></div>
          <Bar_ value={(f.lambda_mu - 1.0) / 0.5} color={C.am} h={4} />
        </GC>
        {/* Solve time */}
        <GC style={{ padding: "8px 12px", textAlign: "center" }}>
          <Lbl color={tc(f.wmpcSolveMs, 8, 10)}>WMPC SOLVE</Lbl>
          <div style={{ margin: "6px 0" }}><Val big color={tc(f.wmpcSolveMs, 8, 10)}>{f.wmpcSolveMs}</Val><Lbl> ms</Lbl></div>
          <Bar_ value={f.wmpcSolveMs / 12} color={tc(f.wmpcSolveMs, 8, 10)} h={4} />
        </GC>
        {/* Hamiltonian */}
        <GC style={{ padding: "8px 12px", textAlign: "center" }}>
          <Lbl color={C.pr}>H(q,p)</Lbl>
          <div style={{ margin: "6px 0" }}><Val big color={C.pr}>{f.hamiltonianJ}</Val><Lbl> kJ</Lbl></div>
        </GC>
        {/* Constraints log */}
        <GC style={{ padding: "8px 12px" }}>
          <Lbl color={C.red}>ACTIVE CONSTRAINTS</Lbl>
          <div style={{ marginTop: 4, maxHeight: 50, overflowY: "auto" }}>
            {constraints.length === 0 && <div style={{ fontSize: 9, color: C.gn, fontFamily: C.dt }}>✓ All clear — no active penalties</div>}
            {constraints.map((c, i) => (
              <div key={i} style={{ fontSize: 9, fontFamily: C.dt, color: C.am, marginBottom: 2, display: "flex", gap: 6 }}>
                <span style={{ color: C.dm }}>{c.t.toFixed(1)}s</span>{c.msg}
              </div>
            ))}
          </div>
        </GC>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// ANALYZE MODE — Sub-navigation workspaces
// ═══════════════════════════════════════════════════════════════════════
const AZ_TABS = [
  { key: "overlay", label: "Overlay & Delta" },
  { key: "fidelity", label: "Fidelity" },
  { key: "frequency", label: "Frequency" },
  { key: "spatial", label: "Spatial" },
  { key: "ai", label: "AI Diagnostics" },
];

function AnalyzeMode({ track, tireTemps }) {
  const [ws, setWs] = useState("overlay");
  const ax = { tick: { fontSize: 9, fill: C.dm, fontFamily: C.dt }, stroke: C.b1 };

  const loads = useMemo(() => track ? gLoadTransfer(track) : [], [track]);
  const lapDelta = useMemo(() => track ? gLapDelta(track) : [], [track]);
  const slipE = useMemo(() => track ? gSlipEnergy(track) : [], [track]);
  const fricSat = useMemo(() => track ? gFrictionSatHist(track) : [], [track]);
  const gripUtil = useMemo(() => track ? gGripUtil(track) : [], [track]);
  const freqResp = useMemo(() => gFreqResponse(), []);
  const damperH = useMemo(() => gDamperHist(), []);
  const rideH = useMemo(() => gRideHeightHist(), []);
  const rcMig = useMemo(() => gRollCenterMig(), []);
  const psd = useMemo(() => gPSDOverlay(), []);
  const ekf = useMemo(() => gEKFInnovation(), []);
  const pacDrift = useMemo(() => gPacejkaDrift(), []);
  const horizon = useMemo(() => gHorizonTraj(), []);
  const wavelets = useMemo(() => gWaveletCoeffs(), []);
  const alSlack = useMemo(() => track ? gALSlack(track) : [], [track]);
  const ctrlEff = useMemo(() => track ? gControlEffort(track) : [], [track]);

  const ggData = useMemo(() => track ? track.filter((_, i) => i % 3 === 0).map(p => ({ lat: Number(p.lat_g), lon: Number(p.lon_g) })) : [], [track]);

  return (
    <>
      {/* Workspace pills */}
      <div style={{ display: "flex", gap: 5, marginBottom: 16, flexWrap: "wrap" }}>
        {AZ_TABS.map(t => <Pill key={t.key} active={ws === t.key} label={t.label} onClick={() => setWs(t.key)} color={C.cy} />)}
      </div>

      <FadeSlide keyVal={ws}>
        {/* ── WS1: OVERLAY & DELTA ───────────────────────── */}
        {ws === "overlay" && (<>
          <div style={{ ...GL, padding: "10px 14px", marginBottom: 14, display: "flex", alignItems: "center", gap: 10, fontSize: 10, fontFamily: C.dt }}>
            <StatusDot ok /><span style={{ color: C.gn }}>Loaded:</span><span style={{ color: C.br }}>Physical_Stint_1_Lap_4</span>
            <span style={{ color: C.dm }}>vs</span><span style={{ color: C.cy }}>Sim_WMPC_Baseline</span>
            <div style={{ marginLeft: "auto", color: C.dm }}>X-axis: Track Distance [m]</div>
          </div>
          <Sec title="Cumulative Time Delta"><GC><ResponsiveContainer width="100%" height={200}><AreaChart data={lapDelta.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" /><Area type="monotone" dataKey="delta" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.8} dot={false} name="Δt [s]" /></AreaChart></ResponsiveContainer></GC></Sec>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <Sec title="Speed — Actual vs Optimal"><GC><ResponsiveContainer width="100%" height={200}><LineChart data={lapDelta.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><Line type="monotone" dataKey="vActual" stroke={C.cy} strokeWidth={1.3} dot={false} name="Actual" /><Line type="monotone" dataKey="vOptimal" stroke={C.gn} strokeWidth={1.3} dot={false} name="WMPC Optimal" strokeDasharray="4 2" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></LineChart></ResponsiveContainer></GC></Sec>
            <Sec title="Chassis Attitude (Lat G / Curvature)"><GC><ResponsiveContainer width="100%" height={200}><LineChart data={track.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><Line type="monotone" dataKey="lat_g" stroke={C.am} strokeWidth={1.3} dot={false} name="Lat G" /><Line type="monotone" dataKey="curvature" stroke={C.pr} strokeWidth={1} dot={false} name="κ [1/m]" strokeDasharray="3 2" /><ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></LineChart></ResponsiveContainer></GC></Sec>
          </div>
          <Sec title="Grip Utilisation"><GC><ResponsiveContainer width="100%" height={160}><AreaChart data={gripUtil} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax} /><YAxis domain={[0,1]} {...ax} /><Tooltip contentStyle={TT} /><ReferenceArea y1={.85} y2={1} fill={C.red} fillOpacity={.05} /><Area type="monotone" dataKey="utilisation" stroke={C.am} fill={`${C.am}0c`} strokeWidth={1.5} dot={false} /></AreaChart></ResponsiveContainer></GC></Sec>
        </>)}

        {/* ── WS2: FIDELITY ──────────────────────────────── */}
        {ws === "fidelity" && (<>
          <Sec title="EKF Residuals (ỹ = z − Hx)"><GC><ResponsiveContainer width="100%" height={260}><LineChart data={ekf.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="t" {...ax} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3" /><ReferenceArea y1={-.02} y2={.02} fill={C.gn} fillOpacity={.04} /><Line type="monotone" dataKey="innov_ax" stroke={C.cy} strokeWidth={1} dot={false} name="a_x" /><Line type="monotone" dataKey="innov_wz" stroke={C.am} strokeWidth={1} dot={false} name="ω_z" /><Line type="monotone" dataKey="innov_vy" stroke={C.pr} strokeWidth={1} dot={false} name="v_y" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></LineChart></ResponsiveContainer></GC><Note>ỹ → 0 = perfect twin. Green band = ±0.02 (2σ). Persistent bias → model misspecification in H_net or Pacejka.</Note></Sec>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <Sec title="Pacejka λ_μy Drift Over Stint"><GC><ResponsiveContainer width="100%" height={240}><ComposedChart data={pacDrift} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="lap" {...ax} label={{value:"Lap #",position:"bottom",offset:4,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}} /><YAxis yAxisId="l" {...ax} domain={["auto","auto"]} /><YAxis yAxisId="r" orientation="right" {...ax} /><Tooltip contentStyle={TT} /><Line yAxisId="l" type="monotone" dataKey="muY_pct" stroke={C.am} strokeWidth={2} dot={{r:3,fill:C.am}} name="µ_y [% fresh]" /><Bar yAxisId="r" dataKey="stiffness" fill={C.cy} fillOpacity={.25} name="C_α [N/rad]" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></ComposedChart></ResponsiveContainer></GC></Sec>
            <Sec title="Energy Dissipation: Sim vs Physical"><GC><ResponsiveContainer width="100%" height={240}><BarChart data={[{cat:"Aero Drag",sim:3200,real:3350},{cat:"Damper",sim:1890,real:1920},{cat:"Tire Slip",sim:7190,real:7400},{cat:"Drivetrain",sim:420,real:480}]} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="cat" {...ax} /><YAxis {...ax} label={{value:"Joules",angle:-90,position:"insideLeft",offset:8,style:{fontSize:9,fill:C.md}}} /><Tooltip contentStyle={TT} /><Bar dataKey="sim" fill={C.cy} fillOpacity={.6} name="Simulated" /><Bar dataKey="real" fill={C.am} fillOpacity={.6} name="Physical" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></BarChart></ResponsiveContainer></GC></Sec>
          </div>
        </>)}

        {/* ── WS3: FREQUENCY ─────────────────────────────── */}
        {ws === "frequency" && (<>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <Sec title="Damper Velocity Histogram"><GC><ResponsiveContainer width="100%" height={280}><BarChart data={damperH} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="vel" {...ax} label={{value:"Piston Vel [m/s]",position:"bottom",offset:4,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><Bar dataKey="front" fill={C.cy} fillOpacity={.5} name="Front" radius={[2,2,0,0]} /><Bar dataKey="rear" fill={C.am} fillOpacity={.5} name="Rear" radius={[2,2,0,0]} /><ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></BarChart></ResponsiveContainer></GC><Note>Low-speed threshold ±0.05 m/s. ~72% operation in low-speed region. Asymmetry indicates bump/rebound bias.</Note></Sec>
            <Sec title="PSD — Real vs Simulated Suspension"><GC><ResponsiveContainer width="100%" height={280}><LineChart data={psd} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="freq" {...ax} label={{value:"Hz",position:"bottom",offset:4,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}} /><YAxis {...ax} label={{value:"PSD [dB]",angle:-90,position:"insideLeft",offset:8,style:{fontSize:9,fill:C.md}}} /><Tooltip contentStyle={TT} /><Line type="monotone" dataKey="real" stroke={C.am} strokeWidth={1.5} dot={false} name="Real Damper" /><Line type="monotone" dataKey="sim" stroke={C.cy} strokeWidth={1.5} dot={false} name="Sim (H_net)" strokeDasharray="4 2" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></LineChart></ResponsiveContainer></GC><Note>Overlap validates H_net frequency content. Divergence &gt;20 Hz = unmodeled tyre cavity resonance.</Note></Sec>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <Sec title="Frequency Response (Bode)"><GC><ResponsiveContainer width="100%" height={260}><LineChart data={freqResp} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="freq" {...ax} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" /><ReferenceArea x1={1} x2={3} fill={C.gn} fillOpacity={.04} /><ReferenceArea x1={8} x2={15} fill={C.am} fillOpacity={.04} /><Line type="monotone" dataKey="front_dB" stroke={C.cy} strokeWidth={2} dot={false} name="Front" /><Line type="monotone" dataKey="rear_dB" stroke={C.am} strokeWidth={2} dot={false} name="Rear" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></LineChart></ResponsiveContainer></GC></Sec>
            <Sec title="Roll Center Migration"><GC><ResponsiveContainer width="100%" height={260}><ScatterChart margin={{top:10,right:20,bottom:24,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="rcXf" type="number" {...ax} label={{value:"RC Height [mm]",position:"bottom",offset:6,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}} /><YAxis dataKey="rcYf" type="number" {...ax} label={{value:"RC Lateral [mm]",angle:-90,position:"insideLeft",offset:8,style:{fontSize:9,fill:C.md}}} /><Tooltip contentStyle={TT} /><Scatter data={rcMig} fill={C.cy} fillOpacity={.4} r={3} name="Front" /><Scatter data={rcMig.map(d=>({...d,rcXf:d.rcXr,rcYf:d.rcYr}))} fill={C.am} fillOpacity={.4} r={3} name="Rear" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></ScatterChart></ResponsiveContainer></GC></Sec>
          </div>
        </>)}

        {/* ── WS4: SPATIAL ────────────────────────────────── */}
        {ws === "spatial" && (<>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <Sec title="Slip Energy Dissipation [J]"><GC style={{padding:"16px 14px"}}>
              {/* SVG track coloured by slip energy */}
              {(() => {
                const xs = slipE.map(p => Number(p.x)), ys = slipE.map(p => Number(p.y));
                const xn = Math.min(...xs), xx = Math.max(...xs), yn = Math.min(...ys), yx = Math.max(...ys);
                const W = 360, H = 260, pd = 20;
                const sc = Math.min((W - 2 * pd) / (xx - xn || 1), (H - 2 * pd) / (yx - yn || 1));
                const ox = pd + (W - 2 * pd - (xx - xn) * sc) / 2, oy = pd + (H - 2 * pd - (yx - yn) * sc) / 2;
                const eMax = Math.max(...slipE.map(d => d.energy), 1);
                return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 260 }}>
                  {slipE.map((p, i) => { if (!i) return null; const prev = slipE[i - 1]; const r = Math.min(1, p.energy / eMax);
                    return <line key={i} x1={ox + (Number(prev.x) - xn) * sc} y1={H - (oy + (Number(prev.y) - yn) * sc)} x2={ox + (Number(p.x) - xn) * sc} y2={H - (oy + (Number(p.y) - yn) * sc)} stroke={`rgb(${Math.round(225 * r)}, ${Math.round(80 * (1 - r))}, ${Math.round(40 + 60 * (1 - r))})`} strokeWidth={3} strokeLinecap="round" />; })}
                </svg>;
              })()}
              <div style={{display:"flex",justifyContent:"center",gap:16,marginTop:6}}>
                <Lbl color={C.gn}>Low Energy</Lbl>
                <div style={{width:80,height:4,borderRadius:2,background:`linear-gradient(90deg,${C.gn},${C.am},${C.red})`}} />
                <Lbl color={C.red}>High Energy</Lbl>
              </div>
            </GC></Sec>
            <Sec title="G-G Friction Circle (Bounded)"><GC><ResponsiveContainer width="100%" height={280}><ScatterChart margin={{top:10,right:10,bottom:24,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="lat" type="number" domain={[-2,2]} {...ax} label={{value:"Lat [G]",position:"bottom",offset:6,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}} /><YAxis dataKey="lon" type="number" domain={[-2,2]} {...ax} label={{value:"Lon [G]",angle:-90,position:"insideLeft",offset:8,style:{fontSize:9,fill:C.md}}} /><ReferenceLine x={0} stroke={C.dm} strokeWidth={.5} /><ReferenceLine y={0} stroke={C.dm} strokeWidth={.5} /><Scatter data={ggData} fill={C.red} fillOpacity={.35} r={2} name="Actual" /></ScatterChart></ResponsiveContainer></GC><Note>Pacejka limit ring at µ=1.35 (theoretical). Points outside = tyre saturation. Gap = unused grip potential.</Note></Sec>
          </div>
          <Sec title="Friction Circle Saturation Distribution"><GC><ResponsiveContainer width="100%" height={200}><BarChart data={fricSat} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="bin" {...ax} label={{value:"µ Utilisation",position:"bottom",offset:4,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><ReferenceArea x1={.85} x2={1} fill={C.red} fillOpacity={.05} /><Bar dataKey="count" radius={[2,2,0,0]} barSize={12}>{fricSat.map((e,i)=><Cell key={i} fill={e.bin>=.85?C.red:e.bin>=.6?C.am:C.cy} fillOpacity={.7} />)}</Bar></BarChart></ResponsiveContainer></GC></Sec>
        </>)}

        {/* ── WS5: AI DIAGNOSTICS ─────────────────────────── */}
        {ws === "ai" && (<>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <Sec title="Receding Horizon — 64-step Prediction"><GC><ResponsiveContainer width="100%" height={280}><ComposedChart data={horizon} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="xPred" type="number" {...ax} label={{value:"X [m]",position:"bottom",offset:4,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}} /><YAxis dataKey="yPred" type="number" {...ax} /><Tooltip contentStyle={TT} /><Line type="monotone" dataKey="yBoundL" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3" name="Track L" /><Line type="monotone" dataKey="yBoundR" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3" name="Track R" /><Line type="monotone" dataKey="yPred" stroke={C.cy} strokeWidth={2} dot={false} name="Predicted" /></ComposedChart></ResponsiveContainer></GC></Sec>
            <Sec title="Wavelet Coefficient Sparsity"><GC style={{padding:"14px"}}>
              {["cA3","cD3","cD2","cD1"].map(lv => {const co=wavelets.filter(w=>w.level===lv);const actN=co.filter(w=>w.active).length;return(
                <div key={lv} style={{marginBottom:10}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                    <span style={{fontSize:10,fontFamily:C.dt,color:lv==="cA3"?C.cy:C.am,fontWeight:700}}>{lv}</span>
                    <span style={{fontSize:9,fontFamily:C.dt,color:C.dm}}>{actN}/{co.length} active</span>
                  </div>
                  <div style={{display:"flex",gap:2}}>
                    {co.map((w,j)=><div key={j} style={{flex:1,height:18,borderRadius:2,background:w.active?(lv==="cA3"?C.cy:C.am):C.b1,opacity:.3+w.mag*.7}} />)}
                  </div>
                </div>
              );})}
              <Note>Db4 3-level DWT: {wavelets.filter(w=>w.active).length}/{wavelets.length} active. Sparsity: {((1-wavelets.filter(w=>w.active).length/wavelets.length)*100).toFixed(0)}%</Note>
            </GC></Sec>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <Sec title="AL Constraint Slack"><GC><ResponsiveContainer width="100%" height={220}><AreaChart data={alSlack} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax} /><YAxis {...ax} /><Tooltip contentStyle={TT} /><ReferenceArea y1={0} y2={.1} fill={C.red} fillOpacity={.05} /><Area type="monotone" dataKey="slackGrip" stroke={C.red} fill={`${C.red}08`} strokeWidth={1.5} dot={false} name="Grip Slack [G]" /><Area type="monotone" dataKey="slackSteer" stroke={C.am} fill={`${C.am}06`} strokeWidth={1.2} dot={false} name="Steer Slack [rad]" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></AreaChart></ResponsiveContainer></GC></Sec>
            <Sec title="Control Effort Saturation"><GC><ResponsiveContainer width="100%" height={220}><AreaChart data={ctrlEff} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS} /><XAxis dataKey="s" {...ax} /><YAxis domain={[0,1]} {...ax} /><Tooltip contentStyle={TT} /><ReferenceArea y1={.9} y2={1} fill={C.red} fillOpacity={.05} /><Line type="monotone" dataKey="steerUtil" stroke={C.pr} strokeWidth={1.5} dot={false} name="Steer %" /><Line type="monotone" dataKey="brakeUtil" stroke={C.red} strokeWidth={1.5} dot={false} name="Brake %" /><Line type="monotone" dataKey="throttleUtil" stroke={C.gn} strokeWidth={1.5} dot={false} name="Throttle %" /><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}} /></AreaChart></ResponsiveContainer></GC></Sec>
          </div>
        </>)}
      </FadeSlide>
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN EXPORT — switches between LIVE and ANALYZE based on mode prop
// ═══════════════════════════════════════════════════════════════════════
export default function TelemetryModule({ track, tireTemps, mode }) {
  if (mode === "LIVE") return <LiveMode mode={mode} />;
  return <AnalyzeMode track={track} tireTemps={tireTemps} />;
}
