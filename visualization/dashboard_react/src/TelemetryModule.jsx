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
  gEKFInnovation, gPacejkaDrift, gHorizonTraj, gWaveletCoeffs,
  gALSlack, gControlEffort, gGripUtil, gDriverInputs,
} from "./data.js";

// ═══════════════════════════════════════════════════════════════════════
// MICRO-COMPONENTS
// ═══════════════════════════════════════════════════════════════════════
const Lbl=({children,color})=><span style={{fontSize:8,fontWeight:700,color:color||C.dm,fontFamily:C.dt,letterSpacing:1.5,textTransform:"uppercase"}}>{children}</span>;
const Val=({children,color,big})=><span style={{fontSize:big?18:12,fontWeight:700,color:color||C.w,fontFamily:C.dt}}>{children}</span>;
const Vu=({children})=><span style={{fontSize:7,color:C.dm,fontFamily:C.dt,marginLeft:2}}>{children}</span>;
const Bar_=({value,max=1,color,h=4})=><div style={{width:"100%",height:h,background:C.b1,borderRadius:2,overflow:"hidden"}}><div style={{width:`${Math.min(100,(value/max)*100)}%`,height:"100%",background:color||C.cy,borderRadius:2,transition:"width 0.1s"}}/></div>;
const StatusDot=({ok})=><div style={{width:5,height:5,borderRadius:"50%",background:ok?C.gn:C.red,boxShadow:`0 0 4px ${ok?C.gn:C.red}`,display:"inline-block"}}/>;
const tc=(v,lo,hi)=>v<lo?C.gn:v<hi?C.am:C.red;
const MRow=({label,value,unit,color,warn})=>(
  <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"1px 0",borderBottom:`1px solid ${C.b1}15`}}>
    <Lbl>{label}</Lbl>
    <div><Val color={color}>{value}</Val>{unit&&<Vu>{unit}</Vu>}{warn&&<span style={{fontSize:7,color:C.red,marginLeft:3}}>⚠</span>}</div>
  </div>
);
const Panel=({title,color,children,style})=>(
  <div style={{...GL,padding:"6px 8px",borderTop:`2px solid ${color||C.cy}`,...style}}>
    <div style={{marginBottom:4}}><Lbl color={color}>{title}</Lbl></div>
    {children}
  </div>
);
const IRStrip=({zones})=>(
  <div style={{display:"flex",gap:1,marginTop:1}}>
    {zones.map((z,i)=>{const r=Math.min(1,Math.max(0,(z-50)/60));return<div key={i} style={{flex:1,height:10,borderRadius:1,background:`rgb(${Math.round(40+185*r)},${Math.round(200*(1-r))},${Math.round(60*(1-r))})`}}><span style={{fontSize:6,color:C.w,fontFamily:C.dt,display:"block",textAlign:"center",lineHeight:"10px"}}>{z}</span></div>;})}
  </div>
);

// ═══════════════════════════════════════════════════════════════════════
// CHANNEL REGISTRY — every plottable parameter
// ═══════════════════════════════════════════════════════════════════════
const CHANNELS = [
  // Driver
  {key:"speed",label:"Speed",unit:"km/h",color:"#00d2ff",group:"Driver"},
  {key:"steer",label:"Steering",unit:"°",color:"#b388ff",group:"Driver"},
  {key:"throttle",label:"Throttle",unit:"%",color:"#00e676",group:"Driver",scale:100},
  {key:"brake",label:"Brake",unit:"%",color:"#e10600",group:"Driver",scale:100},
  {key:"latG",label:"Lat G",unit:"G",color:"#ffab00",group:"Driver"},
  {key:"lonG",label:"Lon G",unit:"G",color:"#00d2ff",group:"Driver"},
  {key:"combinedG",label:"Combined G",unit:"G",color:"#ff6090",group:"Driver"},
  {key:"yawRate",label:"Yaw Rate",unit:"°/s",color:"#b388ff",group:"Driver"},
  // Inertial
  {key:"beta",label:"Sideslip β",unit:"°",color:"#ff6090",group:"Inertial"},
  {key:"betaDot",label:"β̇ Rate",unit:"°/s",color:"#e10600",group:"Inertial"},
  {key:"yawAccel",label:"Yaw Accel ψ̈",unit:"°/s²",color:"#ffab00",group:"Inertial"},
  {key:"jerkLat",label:"Jerk (lat)",unit:"m/s³",color:"#b388ff",group:"Inertial"},
  {key:"jerkLon",label:"Jerk (lon)",unit:"m/s³",color:"#00d2ff",group:"Inertial"},
  {key:"rollAccelIMU",label:"Roll Accel",unit:"°/s²",color:"#00e676",group:"Inertial"},
  {key:"pitchAccelIMU",label:"Pitch Accel",unit:"°/s²",color:"#ffab00",group:"Inertial"},
  // Loads
  {key:"Fz_fl",label:"Fz FL",unit:"N",color:"#00d2ff",group:"Loads"},
  {key:"Fz_fr",label:"Fz FR",unit:"N",color:"#00e676",group:"Loads"},
  {key:"Fz_rl",label:"Fz RL",unit:"N",color:"#ffab00",group:"Loads"},
  {key:"Fz_rr",label:"Fz RR",unit:"N",color:"#e10600",group:"Loads"},
  // Temps
  {key:"T_fl",label:"Temp FL",unit:"°C",color:"#00d2ff",group:"Thermal"},
  {key:"T_fr",label:"Temp FR",unit:"°C",color:"#00e676",group:"Thermal"},
  {key:"T_rl",label:"Temp RL",unit:"°C",color:"#ffab00",group:"Thermal"},
  {key:"T_rr",label:"Temp RR",unit:"°C",color:"#e10600",group:"Thermal"},
  {key:"gasTemp_fl",label:"Gas FL",unit:"°C",color:"#40e0d0",group:"Thermal"},
  {key:"gasTemp_fr",label:"Gas FR",unit:"°C",color:"#40e080",group:"Thermal"},
  // Strain
  {key:"pushrod_fl",label:"Pushrod FL",unit:"N",color:"#00d2ff",group:"Strain"},
  {key:"pushrod_fr",label:"Pushrod FR",unit:"N",color:"#00e676",group:"Strain"},
  {key:"pushrod_rl",label:"Pushrod RL",unit:"N",color:"#ffab00",group:"Strain"},
  {key:"pushrod_rr",label:"Pushrod RR",unit:"N",color:"#e10600",group:"Strain"},
  // Geometry
  {key:"rideH_fl",label:"Ride H FL",unit:"mm",color:"#00d2ff",group:"Geometry"},
  {key:"rideH_fr",label:"Ride H FR",unit:"mm",color:"#00e676",group:"Geometry"},
  {key:"chassisTwist",label:"Chassis Twist",unit:"°",color:"#b388ff",group:"Geometry"},
  // Slip
  {key:"slip_fl",label:"Slip FL",unit:"°",color:"#00d2ff",group:"Slip"},
  {key:"slip_fr",label:"Slip FR",unit:"°",color:"#00e676",group:"Slip"},
  {key:"slip_rl",label:"Slip RL",unit:"°",color:"#ffab00",group:"Slip"},
  {key:"slip_rr",label:"Slip RR",unit:"°",color:"#e10600",group:"Slip"},
  // Damper
  {key:"damp_fl",label:"Damp V FL",unit:"m/s",color:"#00d2ff",group:"Damper"},
  {key:"damp_fr",label:"Damp V FR",unit:"m/s",color:"#00e676",group:"Damper"},
  {key:"damp_rl",label:"Damp V RL",unit:"m/s",color:"#ffab00",group:"Damper"},
  {key:"damp_rr",label:"Damp V RR",unit:"m/s",color:"#e10600",group:"Damper"},
  // Aero
  {key:"airspeed",label:"Airspeed",unit:"m/s",color:"#00d2ff",group:"Aero"},
  {key:"aeroFzF",label:"Aero Fz F",unit:"N",color:"#00d2ff",group:"Aero"},
  {key:"aeroFzR",label:"Aero Fz R",unit:"N",color:"#ffab00",group:"Aero"},
  {key:"windAngle",label:"Wind Angle",unit:"°",color:"#b388ff",group:"Aero"},
  // Brakes
  {key:"brakePressF",label:"Brake P Front",unit:"bar",color:"#e10600",group:"Brakes"},
  {key:"brakePressR",label:"Brake P Rear",unit:"bar",color:"#ffab00",group:"Brakes"},
  {key:"steerTorque",label:"Steer Torque",unit:"Nm",color:"#b388ff",group:"Brakes"},
  {key:"dynBrakeBias",label:"Dyn Bias",unit:"%F",color:"#ff6090",group:"Brakes"},
  // EV
  {key:"soc",label:"Battery SoC",unit:"%",color:"#00e676",group:"EV"},
  {key:"power",label:"Power Out",unit:"kW",color:"#e10600",group:"EV"},
  {key:"regen",label:"Regen In",unit:"kW",color:"#00e676",group:"EV"},
  {key:"cellTempMax",label:"Cell Temp Max",unit:"°C",color:"#ffab00",group:"EV"},
  {key:"invTempL",label:"Inv Temp L",unit:"°C",color:"#b388ff",group:"EV"},
  {key:"motorTempL",label:"Motor Temp L",unit:"°C",color:"#ff6090",group:"EV"},
  // Twin
  {key:"ekfInnov",label:"EKF Innovation",unit:"",color:"#00d2ff",group:"Twin"},
  {key:"lambda_mu",label:"λ_μ Friction",unit:"",color:"#ffab00",group:"Twin"},
  {key:"wmpcSolveMs",label:"WMPC Solve",unit:"ms",color:"#00e676",group:"Twin"},
  {key:"hamiltonianJ",label:"H(q,p)",unit:"kJ",color:"#b388ff",group:"Twin"},
];

const GROUPS = [...new Set(CHANNELS.map(c => c.group))];
const COLORS_POOL = ["#00d2ff","#00e676","#ffab00","#e10600","#b388ff","#ff6090","#40e0d0","#ff8c00","#8855dd","#3399ff"];

// ═══════════════════════════════════════════════════════════════════════
// LIVE DATA HOOK + HISTORY BUFFER
// ═══════════════════════════════════════════════════════════════════════
const HIST_LEN = 300; // ~10 sec at 30Hz

function useLiveData(active) {
  const [frame, setFrame] = useState(null);
  const historyRef = useRef([]);
  const tickRef = useRef(0);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      tickRef.current++;
      const t = tickRef.current * 0.033;
      const rn = () => Math.random();
      const steer=18*Math.sin(t*0.8),throttle=Math.max(0,.7+.3*Math.cos(t*1.2)),brake=Math.max(0,-.5*Math.cos(t*1.2)+.1);
      const speed=12+10*Math.sin(t*0.3)+3*rn(),latG=speed*speed*0.003*Math.sin(t*0.8)/9.81,lonG=(throttle-brake)*1.2;
      const yawRate=latG*2.5;
      const beta=Math.atan2(latG*0.3,Math.max(speed,1)*0.06)*180/Math.PI;
      const mkS=(b,g)=>+(b+Math.abs(latG)*g+Math.abs(lonG)*g*0.5+rn()*50).toFixed(0);
      const mkIR=(b)=>[0,1,2,3].map(z=>+(b+z*3+rn()*4-latG*(z-1.5)*8).toFixed(1));
      const air=speed+2*Math.sin(t*0.1)+rn();
      const aFzF=0.5*1.225*air*air*1.1*1.6*0.55;const aFzR=0.5*1.225*air*air*1.1*1.6*0.45;

      const f = {
        t:+(t%120).toFixed(2),speed:+speed.toFixed(1),steer:+steer.toFixed(1),
        throttle:+Math.min(1,throttle).toFixed(3),brake:+Math.min(1,brake).toFixed(3),
        latG:+latG.toFixed(3),lonG:+lonG.toFixed(3),yawRate:+yawRate.toFixed(2),
        combinedG:+Math.sqrt(latG*latG+lonG*lonG).toFixed(3),motorRPM:+((speed/0.2032)*4.5*60/(2*Math.PI)).toFixed(0),
        x:+(50*Math.cos(t*0.12)).toFixed(1),y:+(30*Math.sin(t*0.12)).toFixed(1),
        Fz_fl:+(750-latG*200-lonG*100+30*rn()).toFixed(0),Fz_fr:+(750+latG*200-lonG*100+30*rn()).toFixed(0),
        Fz_rl:+(680-latG*180+lonG*120+25*rn()).toFixed(0),Fz_rr:+(680+latG*180+lonG*120+25*rn()).toFixed(0),
        T_fl:+(65+20*Math.abs(latG)+5*rn()).toFixed(1),T_fr:+(68+22*Math.abs(latG)+5*rn()).toFixed(1),
        T_rl:+(60+15*Math.abs(latG)+4*rn()).toFixed(1),T_rr:+(62+16*Math.abs(latG)+4*rn()).toFixed(1),
        slip_fl:+(latG*3+.5*rn()).toFixed(2),slip_fr:+(-latG*3+.5*rn()).toFixed(2),
        slip_rl:+(latG*2.5+.4*rn()).toFixed(2),slip_rr:+(-latG*2.5+.4*rn()).toFixed(2),
        damp_fl:+(.05*Math.sin(t*5)+.02*rn()).toFixed(3),damp_fr:+(.04*Math.sin(t*5+1)+.02*rn()).toFixed(3),
        damp_rl:+(.03*Math.sin(t*5+2)+.015*rn()).toFixed(3),damp_rr:+(.035*Math.sin(t*5+3)+.015*rn()).toFixed(3),
        pushrod_fl:mkS(1200,400),pushrod_fr:mkS(1250,420),pushrod_rl:mkS(1100,350),pushrod_rr:mkS(1150,370),
        rideH_fl:+(35-Math.abs(latG)*8-lonG*3+rn()*2).toFixed(1),rideH_fr:+(36+Math.abs(latG)*8-lonG*3+rn()*2).toFixed(1),
        chassisTwist:+(latG*0.8+rn()*0.15).toFixed(3),
        beta:+beta.toFixed(2),betaDot:+(latG*15+5*rn()).toFixed(1),yawAccel:+(yawRate*2*Math.cos(t*1.6)+rn()*3).toFixed(1),
        rollAccelIMU:+(latG*18+rn()*3).toFixed(1),pitchAccelIMU:+(lonG*12+rn()*2).toFixed(1),
        jerkLat:+(latG*30*Math.cos(t*1.2)+rn()*5).toFixed(1),jerkLon:+(lonG*20*Math.sin(t*1.6)+rn()*3).toFixed(1),
        gasTemp_fl:+(45+15*Math.abs(latG)+3*rn()).toFixed(1),gasTemp_fr:+(47+16*Math.abs(latG)+3*rn()).toFixed(1),
        airspeed:+air.toFixed(1),windAngle:+(5*Math.sin(t*0.05)+3*rn()).toFixed(1),
        aeroFzF:+aFzF.toFixed(0),aeroFzR:+aFzR.toFixed(0),
        steerTorque:+(steer*0.8+latG*12+rn()*3).toFixed(1),
        brakePressF:+(brake*80+rn()*3).toFixed(1),brakePressR:+(brake*55+rn()*2).toFixed(1),
        dynBrakeBias:+(55+lonG*3+rn()*2).toFixed(1),
        soc:+(85-t*0.08).toFixed(1),power:+(speed*throttle*0.8).toFixed(1),regen:+(brake>0.3?brake*speed*0.3:0).toFixed(1),
        cellTempMax:+(38+t*0.01+2*rn()).toFixed(1),invTempL:+(52+10*throttle+3*rn()).toFixed(1),motorTempL:+(65+15*throttle+4*rn()).toFixed(1),
        ekfInnov:+(0.015*Math.sin(t*8)+0.008*rn()).toFixed(4),lambda_mu:+(1.35-t*0.0003).toFixed(4),
        wmpcSolveMs:+(4+3*rn()).toFixed(1),hamiltonianJ:+(0.5*300*speed*speed/1000).toFixed(1),
        irFL:mkIR(Number((65+20*Math.abs(latG)).toFixed(0))),
        constraint:t%15<0.1?"WARN: Thermal Derating":t%22<0.1?"WARN: Max Slip":null,
      };
      historyRef.current = [...historyRef.current.slice(-(HIST_LEN - 1)), f];
      setFrame(f);
    }, 33);
    return () => clearInterval(id);
  }, [active]);

  return { frame, history: historyRef };
}

// ═══════════════════════════════════════════════════════════════════════
// CANVAS TIME-SERIES GRAPH — renders selected channels from history
// ═══════════════════════════════════════════════════════════════════════
function TimeSeriesCanvas({ history, channels, height = 160 }) {
  const cvRef = useRef(null);
  const frameRef = useRef(0);

  useEffect(() => {
    const cv = cvRef.current; if (!cv || !channels.length) return;
    let running = true;
    const draw = () => {
      if (!running) return;
      const ctx = cv.getContext("2d");
      const W = cv.width, H = cv.height;
      const data = history.current;
      if (!data.length) { requestAnimationFrame(draw); return; }

      ctx.clearRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = "rgba(30,40,60,0.4)"; ctx.lineWidth = 0.5;
      for (let i = 0; i < 5; i++) { const y = H * i / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

      // Per channel
      channels.forEach(ch => {
        const chDef = CHANNELS.find(c => c.key === ch);
        if (!chDef) return;
        const vals = data.map(d => { let v = Number(d[ch]) || 0; if (chDef.scale) v *= chDef.scale; return v; });
        if (!vals.length) return;

        const vMin = Math.min(...vals), vMax = Math.max(...vals);
        const range = vMax - vMin || 1;
        const pad = range * 0.1;

        ctx.strokeStyle = chDef.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        vals.forEach((v, i) => {
          const x = (i / (HIST_LEN - 1)) * W;
          const y = H - ((v - vMin + pad) / (range + 2 * pad)) * H;
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Latest value label
        const last = vals[vals.length - 1];
        const ly = H - ((last - vMin + pad) / (range + 2 * pad)) * H;
        ctx.fillStyle = chDef.color;
        ctx.font = "bold 10px 'Azeret Mono', monospace";
        ctx.fillText(`${last.toFixed(1)} ${chDef.unit}`, W - 80, Math.max(12, Math.min(H - 4, ly - 4)));
      });

      requestAnimationFrame(draw);
    };
    draw();
    return () => { running = false; };
  }, [channels, history]);

  return <canvas ref={cvRef} width={800} height={height} style={{ width: "100%", height, borderRadius: 6, background: "rgba(6,8,12,0.6)" }} />;
}

// ═══════════════════════════════════════════════════════════════════════
// GRAPH SLOT — one chart panel with channel picker
// ═══════════════════════════════════════════════════════════════════════
function GraphSlot({ id, history, onRemove }) {
  const [selected, setSelected] = useState([]);
  const [filterGroup, setFilterGroup] = useState(null);
  const [showPicker, setShowPicker] = useState(true);

  const filtered = filterGroup ? CHANNELS.filter(c => c.group === filterGroup) : CHANNELS;
  const toggle = (key) => setSelected(prev => prev.includes(key) ? prev.filter(k => k !== key) : prev.length < 8 ? [...prev, key] : prev);

  return (
    <div style={{ ...GL, padding: "8px 10px", marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <Lbl color={C.cy}>GRAPH {id}</Lbl>
        <div style={{ flex: 1 }} />
        {selected.map(k => { const ch = CHANNELS.find(c => c.key === k); return ch ? <span key={k} onClick={() => toggle(k)} style={{ fontSize: 8, fontFamily: C.dt, color: ch.color, background: `${ch.color}15`, border: `1px solid ${ch.color}30`, padding: "1px 6px", borderRadius: 10, cursor: "pointer" }}>{ch.label} ×</span> : null; })}
        <button onClick={() => setShowPicker(!showPicker)} style={{ background: "none", border: `1px solid ${C.b1}`, borderRadius: 4, padding: "2px 8px", fontSize: 8, color: C.md, fontFamily: C.dt, cursor: "pointer" }}>{showPicker ? "▲ Hide" : "▼ Channels"}</button>
        <button onClick={onRemove} style={{ background: "none", border: `1px solid ${C.red}30`, borderRadius: 4, padding: "2px 6px", fontSize: 8, color: C.red, fontFamily: C.dt, cursor: "pointer" }}>✕</button>
      </div>

      {showPicker && (
        <div style={{ marginBottom: 6 }}>
          <div style={{ display: "flex", gap: 3, marginBottom: 4, flexWrap: "wrap" }}>
            <button onClick={() => setFilterGroup(null)} style={{ background: !filterGroup ? `${C.cy}15` : "none", border: `1px solid ${!filterGroup ? C.cy : C.b1}`, borderRadius: 10, padding: "2px 8px", fontSize: 7, color: !filterGroup ? C.cy : C.dm, fontFamily: C.dt, cursor: "pointer" }}>ALL</button>
            {GROUPS.map(g => <button key={g} onClick={() => setFilterGroup(g)} style={{ background: filterGroup === g ? `${C.cy}15` : "none", border: `1px solid ${filterGroup === g ? C.cy : C.b1}`, borderRadius: 10, padding: "2px 8px", fontSize: 7, color: filterGroup === g ? C.cy : C.dm, fontFamily: C.dt, cursor: "pointer" }}>{g}</button>)}
          </div>
          <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
            {filtered.map(ch => {
              const isOn = selected.includes(ch.key);
              return <button key={ch.key} onClick={() => toggle(ch.key)} style={{ background: isOn ? `${ch.color}18` : "none", border: `1px solid ${isOn ? ch.color : C.b1}`, borderRadius: 10, padding: "2px 8px", fontSize: 7, color: isOn ? ch.color : C.dm, fontFamily: C.dt, cursor: "pointer", transition: "all 0.15s" }}>{ch.label}</button>;
            })}
          </div>
        </div>
      )}

      {selected.length > 0 ? (
        <TimeSeriesCanvas history={history} channels={selected} height={140} />
      ) : (
        <div style={{ height: 60, display: "flex", alignItems: "center", justifyContent: "center", color: C.dm, fontSize: 10, fontFamily: C.dt }}>Select channels to plot</div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// LIVE TRACK MAP (canvas)
// ═══════════════════════════════════════════════════════════════════════
function LiveTrackMap({ frame, hist }) {
  const cv = useRef(null);
  useEffect(() => {
    const c = cv.current; if (!c || !frame) return;
    const ctx = c.getContext("2d"), W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = C.b2; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.ellipse(W / 2, H / 2, W * .38, H * .38, 0, 0, Math.PI * 2); ctx.stroke();
    const data = hist.current;
    if (data.length > 1) for (let i = 1; i < data.length; i++) {
      const a = i / data.length, h = data[i], g = Math.sqrt(h.latG * h.latG + h.lonG * h.lonG), r = Math.min(1, g / 1.5);
      ctx.strokeStyle = `rgba(${Math.round(225 * r)},${Math.round(230 * (1 - r))},${Math.round(80 + 140 * (1 - r))},${a * .5})`;
      ctx.lineWidth = 1.5; ctx.beginPath();
      ctx.moveTo(W / 2 + data[i - 1].x * 2, H / 2 - data[i - 1].y * 2);
      ctx.lineTo(W / 2 + h.x * 2, H / 2 - h.y * 2); ctx.stroke();
    }
    const cx = W / 2 + frame.x * 2, cy = H / 2 - frame.y * 2;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fillStyle = C.gn; ctx.fill();
  }, [frame, hist]);
  return <canvas ref={cv} width={280} height={180} style={{ width: "100%", height: 180, borderRadius: 6 }} />;
}

// ═══════════════════════════════════════════════════════════════════════
// LIVE MODE — FULL LAYOUT
// ═══════════════════════════════════════════════════════════════════════
function LiveMode({ mode }) {
  const { frame, history } = useLiveData(mode === "LIVE");
  const [graphs, setGraphs] = useState([1]);
  const [constraints, setConstraints] = useState([]);
  const nextId = useRef(2);

  const addGraph = () => { setGraphs(prev => [...prev, nextId.current++]); };
  const removeGraph = (id) => { setGraphs(prev => prev.filter(g => g !== id)); };

  useEffect(() => {
    if (frame?.constraint) setConstraints(prev => [{ t: frame.t, msg: frame.constraint }, ...prev.slice(0, 8)]);
  }, [frame]);

  if (!frame) return <div style={{ padding: 40, color: C.md, fontFamily: C.dt, textAlign: "center" }}>Waiting for telemetry…<br /><span style={{ fontSize: 10, color: C.dm }}>ws://localhost:5001</span></div>;
  const f = frame;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {/* ═══ ROW 1: DRIVER STRIP ════════════════════════════════════ */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(8,1fr)", gap: 4 }}>
        {[{l:"SPD",v:f.speed,u:"km/h",c:C.cy},{l:"RPM",v:f.motorRPM,u:"rpm",c:C.am},{l:"THR",v:(f.throttle*100).toFixed(0),u:"%",c:C.gn,b:f.throttle},
          {l:"BRK",v:(f.brake*100).toFixed(0),u:"%",c:C.red,b:f.brake},{l:"STR",v:f.steer,u:"°",c:C.pr},{l:"LAT",v:f.latG,u:"G",c:C.am},
          {l:"LON",v:f.lonG,u:"G",c:C.cy},{l:"CMB",v:f.combinedG,u:"G",c:C.red}
        ].map(d=><Panel key={d.l} title={d.l} color={d.c} style={{textAlign:"center",padding:"4px 4px"}}>
          <Val big>{d.v}</Val><Vu>{d.u}</Vu>{d.b!==undefined&&<div style={{marginTop:2}}><Bar_ value={d.b} color={d.c}/></div>}
        </Panel>)}
      </div>

      {/* ═══ ROW 2: GRAPHS + TRACK MAP ══════════════════════════════ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 8 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <Lbl color={C.cy}>TIME-SERIES GRAPHS</Lbl>
            <button onClick={addGraph} style={{ background: `${C.gn}15`, border: `1px solid ${C.gn}30`, borderRadius: 4, padding: "2px 10px", fontSize: 9, color: C.gn, fontFamily: C.dt, cursor: "pointer" }}>+ Add Graph</button>
            <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt }}>{graphs.length} active · {HIST_LEN} frame buffer (~10s)</span>
          </div>
          {graphs.map(id => <GraphSlot key={id} id={id} history={history} onRemove={() => removeGraph(id)} />)}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <Panel title="TRACK MAP" color={C.cy}><LiveTrackMap frame={f} hist={history} /></Panel>
          <Panel title="FPV CAMERA + HUD" color={C.dm} style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",background:"rgba(5,7,11,0.8)",padding:"12px 8px",minHeight:100}}>
            <Lbl color={C.dm}>RTSP ROLL-HOOP CAMERA</Lbl>
            <div style={{margin:"6px 0",fontSize:9,color:C.dm,fontFamily:C.dt,textAlign:"center",lineHeight:1.7}}>
              HUD overlay: Speed + WMPC trajectory<br/>
              <span style={{color:C.am}}>rtsp://192.168.1.10:8554/fpv</span>
            </div>
            <div style={{display:"flex",gap:4}}>
              <div style={{...GL,padding:"2px 6px",fontSize:8,fontFamily:C.dt,color:C.red}}>● REC</div>
              <div style={{...GL,padding:"2px 6px",fontSize:8,fontFamily:C.dt,color:C.dm}}>NO SIGNAL</div>
            </div>
          </Panel>
          <Panel title="6-AXIS INERTIAL" color={C.pr}>
            <MRow label="Sideslip β" value={f.beta} unit="°" color={Math.abs(f.beta)>4?C.red:C.w}/>
            <MRow label="β̇ rate" value={f.betaDot} unit="°/s" color={Math.abs(f.betaDot)>30?C.red:C.w} warn={Math.abs(f.betaDot)>40}/>
            <MRow label="ψ̈ Yaw" value={f.yawAccel} unit="°/s²"/>
            <MRow label="Jerk lat" value={f.jerkLat} unit="m/s³"/>
            <MRow label="Jerk lon" value={f.jerkLon} unit="m/s³"/>
          </Panel>
        </div>
      </div>

      {/* ═══ ROW 3: 4-CORNER COMPACT ════════════════════════════════ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 6 }}>
        {[{l:"FL",c:C.cy,fz:f.Fz_fl,t:f.T_fl,sl:f.slip_fl,dv:f.damp_fl,pr:f.pushrod_fl,rh:f.rideH_fl,ir:f.irFL},
          {l:"FR",c:C.gn,fz:f.Fz_fr,t:f.T_fr,sl:f.slip_fr,dv:f.damp_fr,pr:f.pushrod_fr,rh:f.rideH_fr,ir:f.irFL},
          {l:"RL",c:C.am,fz:f.Fz_rl,t:f.T_rl,sl:f.slip_rl,dv:f.damp_rl,pr:f.pushrod_rl,rh:f.rideH_fl,ir:f.irFL},
          {l:"RR",c:C.red,fz:f.Fz_rr,t:f.T_rr,sl:f.slip_rr,dv:f.damp_rr,pr:f.pushrod_rr,rh:f.rideH_fr,ir:f.irFL}
        ].map(cn=><Panel key={cn.l} title={cn.l} color={cn.c}>
          <MRow label="Fz" value={cn.fz} unit="N"/><MRow label="Temp" value={cn.t} unit="°C" color={tc(cn.t,80,100)}/>
          <MRow label="Slip α" value={cn.sl} unit="°"/><MRow label="Damp V" value={cn.dv} unit="m/s"/>
          <MRow label="Pushrod" value={cn.pr} unit="N" color={cn.pr>1800?C.am:C.w}/>
          <MRow label="Ride H" value={cn.rh} unit="mm" color={cn.rh<25?C.red:cn.rh<30?C.am:C.w}/>
          <div style={{marginTop:2}}><Lbl color={cn.c}>IR (IN→OUT)</Lbl><IRStrip zones={cn.ir||[65,68,70,66]}/></div>
        </Panel>)}
      </div>

      {/* ═══ ROW 4: AERO + SUSP MODAL + HYDRAULICS ══════════════════ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
        <Panel title="AERODYNAMIC STATE" color="#ff6090">
          <MRow label="Airspeed" value={f.airspeed} unit="m/s"/>
          <MRow label="Wind Angle" value={f.windAngle} unit="°"/>
          <MRow label="Fz Front" value={f.aeroFzF} unit="N" color={C.cy}/>
          <MRow label="Fz Rear" value={f.aeroFzR} unit="N" color={C.am}/>
          <MRow label="Balance" value={((Number(f.aeroFzF)/(Number(f.aeroFzF)+Number(f.aeroFzR)))*100).toFixed(1)} unit="%F"/>
        </Panel>
        <Panel title="CHASSIS MODAL" color={C.pr}>
          <MRow label="Twist Δ" value={f.chassisTwist} unit="°" color={Math.abs(f.chassisTwist)>0.6?C.am:C.w} warn={Math.abs(f.chassisTwist)>0.8}/>
          <MRow label="Roll φ̈" value={f.rollAccelIMU} unit="°/s²"/>
          <MRow label="Pitch θ̈" value={f.pitchAccelIMU} unit="°/s²"/>
        </Panel>
        <Panel title="STEERING & BRAKES" color={C.am}>
          <MRow label="Steer Torque" value={f.steerTorque} unit="Nm" color={Math.abs(f.steerTorque)>25?C.am:C.w}/>
          <MRow label="Press F" value={f.brakePressF} unit="bar" color={f.brakePressF>60?C.red:C.w}/>
          <MRow label="Press R" value={f.brakePressR} unit="bar"/>
          <MRow label="Dyn Bias" value={f.dynBrakeBias} unit="%F"/>
        </Panel>
      </div>

      {/* ═══ ROW 5: EV POWERTRAIN ═══════════════════════════════════ */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 6 }}>
        <Panel title="BATTERY" color={C.gn} style={{textAlign:"center"}}><Val big color={tc(100-f.soc,30,60)}>{f.soc}%</Val><div style={{marginTop:2}}><Bar_ value={f.soc/100} color={tc(100-f.soc,30,60)} h={5}/></div></Panel>
        <Panel title="POWER" color={C.am}><MRow label="Out" value={f.power} unit="kW" color={C.red}/><MRow label="Regen" value={f.regen} unit="kW" color={C.gn}/></Panel>
        <Panel title="CELLS" color={tc(f.cellTempMax,45,55)}><MRow label="Max" value={f.cellTempMax} unit="°C" color={tc(f.cellTempMax,45,55)}/></Panel>
        <Panel title="DRIVE" color={C.pr}><MRow label="Inv L" value={f.invTempL} unit="°C" color={tc(f.invTempL,70,85)}/><MRow label="Motor L" value={f.motorTempL} unit="°C" color={tc(f.motorTempL,80,100)}/></Panel>
        <Panel title="TV" color={C.cy}><MRow label="Target" value={(f.latG*120).toFixed(0)} unit="Nm"/><MRow label="Error" value={((f.latG*120)*(0.9+0.2*Math.random())-(f.latG*120)).toFixed(0)} unit="Nm"/></Panel>
      </div>

      {/* ═══ ROW 6: TWIN HEALTH ═════════════════════════════════════ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr 1.5fr", gap: 6 }}>
        <Panel title="EKF" color={C.cy}><div style={{display:"flex",gap:4,alignItems:"center"}}><StatusDot ok={Math.abs(f.ekfInnov)<0.03}/><Val color={C.cy}>{f.ekfInnov}</Val></div></Panel>
        <Panel title="λ_μ" color={C.am} style={{textAlign:"center"}}><Val big color={C.am}>{f.lambda_mu}</Val></Panel>
        <Panel title="WMPC" color={tc(f.wmpcSolveMs,8,10)} style={{textAlign:"center"}}><Val big color={tc(f.wmpcSolveMs,8,10)}>{f.wmpcSolveMs}</Val><Vu>ms</Vu></Panel>
        <Panel title="H(q,p)" color={C.pr} style={{textAlign:"center"}}><Val big color={C.pr}>{f.hamiltonianJ}</Val><Vu>kJ</Vu></Panel>
        <Panel title="CONSTRAINTS" color={C.red}>
          <div style={{maxHeight:30,overflowY:"auto"}}>
            {constraints.length===0?<span style={{fontSize:7,color:C.gn,fontFamily:C.dt}}>✓ Clear</span>:
            constraints.slice(0,3).map((c,i)=><div key={i} style={{fontSize:7,fontFamily:C.dt,color:C.am}}>{c.msg}</div>)}
          </div>
        </Panel>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// ANALYZE MODE (5 workspaces — carried forward)
// ═══════════════════════════════════════════════════════════════════════
const AZ_TABS=[{key:"overlay",label:"Overlay & Delta"},{key:"fidelity",label:"Fidelity"},{key:"frequency",label:"Frequency"},{key:"spatial",label:"Spatial"},{key:"ai",label:"AI Diagnostics"}];

function AnalyzeMode({track,tireTemps}){
  const[ws,setWs]=useState("overlay");
  const ax={tick:{fontSize:9,fill:C.dm,fontFamily:C.dt},stroke:C.b1};
  const lapDelta=useMemo(()=>track?gLapDelta(track):[],[track]);
  const gripUtil=useMemo(()=>track?gGripUtil(track):[],[track]);
  const ekf=useMemo(()=>gEKFInnovation(),[]);const pacDrift=useMemo(()=>gPacejkaDrift(),[]);
  const freqResp=useMemo(()=>gFreqResponse(),[]);const damperH=useMemo(()=>gDamperHist(),[]);
  const rcMig=useMemo(()=>gRollCenterMig(),[]);const psd=useMemo(()=>gPSDOverlay(),[]);
  const slipE=useMemo(()=>track?gSlipEnergy(track):[],[track]);
  const fricSat=useMemo(()=>track?gFrictionSatHist(track):[],[track]);
  const ggData=useMemo(()=>track?track.filter((_,i)=>i%3===0).map(p=>({lat:Number(p.lat_g),lon:Number(p.lon_g)})):[],[track]);
  const horizon=useMemo(()=>gHorizonTraj(),[]);const wavelets=useMemo(()=>gWaveletCoeffs(),[]);
  const alSlack=useMemo(()=>track?gALSlack(track):[],[track]);const ctrlEff=useMemo(()=>track?gControlEffort(track):[],[track]);
  const drvInputs=useMemo(()=>track?gDriverInputs(track):[],[track]);

  return(<>
    <div style={{display:"flex",gap:5,marginBottom:14,flexWrap:"wrap"}}>{AZ_TABS.map(t=><Pill key={t.key} active={ws===t.key} label={t.label} onClick={()=>setWs(t.key)} color={C.cy}/>)}</div>
    <FadeSlide keyVal={ws}>
      {ws==="overlay"&&(<><Sec title="Time Delta"><GC><ResponsiveContainer width="100%" height={200}><AreaChart data={lapDelta.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3"/><Area type="monotone" dataKey="delta" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.8} dot={false}/></AreaChart></ResponsiveContainer></GC></Sec>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}><Sec title="Speed Comparison"><GC><ResponsiveContainer width="100%" height={200}><LineChart data={lapDelta.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="vActual" stroke={C.cy} strokeWidth={1.3} dot={false} name="Actual"/><Line type="monotone" dataKey="vOptimal" stroke={C.gn} strokeWidth={1.3} dot={false} name="Optimal" strokeDasharray="4 2"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></LineChart></ResponsiveContainer></GC></Sec>
        <Sec title="Grip Utilisation"><GC><ResponsiveContainer width="100%" height={200}><AreaChart data={gripUtil} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis domain={[0,1]} {...ax}/><Tooltip contentStyle={TT}/><ReferenceArea y1={.85} y2={1} fill={C.red} fillOpacity={.05}/><Area type="monotone" dataKey="utilisation" stroke={C.am} fill={`${C.am}0c`} strokeWidth={1.5} dot={false}/></AreaChart></ResponsiveContainer></GC></Sec></div>

        {/* Driver Inputs: Actual vs WMPC Optimal */}
        <div style={{...GL,padding:"8px 12px",marginTop:6,marginBottom:10,display:"flex",alignItems:"center",gap:8,fontSize:9,fontFamily:C.dt}}>
          <Lbl color={C.pr}>DRIVER INPUT COMPARISON</Lbl>
          <span style={{color:C.dm}}>—</span>
          <span style={{color:C.cy}}>Solid = Actual driver input</span>
          <span style={{color:C.dm}}>·</span>
          <span style={{color:C.gn}}>Dashed = WMPC optimal input</span>
        </div>
        <Sec title="Steering — Actual vs WMPC Optimal [°]"><GC><ResponsiveContainer width="100%" height={180}><LineChart data={drvInputs} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/>
          <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3"/>
          <Line type="monotone" dataKey="steerAct" stroke={C.cy} strokeWidth={1.5} dot={false} name="Actual δ"/>
          <Line type="monotone" dataKey="steerOpt" stroke={C.gn} strokeWidth={1.5} dot={false} name="WMPC δ" strokeDasharray="5 3"/>
          <Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/>
        </LineChart></ResponsiveContainer></GC></Sec>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Throttle — Actual vs Optimal [%]"><GC><ResponsiveContainer width="100%" height={180}><AreaChart data={drvInputs} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis domain={[0,100]} {...ax}/><Tooltip contentStyle={TT}/>
            <Area type="monotone" dataKey="thrAct" stroke={C.gn} fill={`${C.gn}08`} strokeWidth={1.5} dot={false} name="Actual %"/>
            <Line type="monotone" dataKey="thrOpt" stroke="#80ff80" strokeWidth={1.3} dot={false} name="WMPC %" strokeDasharray="5 3"/>
            <Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/>
          </AreaChart></ResponsiveContainer></GC></Sec>
          <Sec title="Brake — Actual vs Optimal [%]"><GC><ResponsiveContainer width="100%" height={180}><AreaChart data={drvInputs} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis domain={[0,100]} {...ax}/><Tooltip contentStyle={TT}/>
            <Area type="monotone" dataKey="brkAct" stroke={C.red} fill={`${C.red}08`} strokeWidth={1.5} dot={false} name="Actual %"/>
            <Line type="monotone" dataKey="brkOpt" stroke="#ff8080" strokeWidth={1.3} dot={false} name="WMPC %" strokeDasharray="5 3"/>
            <Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/>
          </AreaChart></ResponsiveContainer></GC></Sec>
        </div>
      </>)}
      {ws==="fidelity"&&(<><Sec title="EKF Residuals"><GC><ResponsiveContainer width="100%" height={250}><LineChart data={ekf.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="t" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3"/><ReferenceArea y1={-.02} y2={.02} fill={C.gn} fillOpacity={.04}/><Line type="monotone" dataKey="innov_ax" stroke={C.cy} strokeWidth={1} dot={false} name="a_x"/><Line type="monotone" dataKey="innov_wz" stroke={C.am} strokeWidth={1} dot={false} name="ω_z"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></LineChart></ResponsiveContainer></GC></Sec>
        <Sec title="Pacejka Drift"><GC><ResponsiveContainer width="100%" height={220}><ComposedChart data={pacDrift} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="lap" {...ax}/><YAxis yAxisId="l" {...ax}/><YAxis yAxisId="r" orientation="right" {...ax}/><Tooltip contentStyle={TT}/><Line yAxisId="l" type="monotone" dataKey="muY_pct" stroke={C.am} strokeWidth={2} dot={{r:3,fill:C.am}} name="µ_y%"/><Bar yAxisId="r" dataKey="stiffness" fill={C.cy} fillOpacity={.25} name="C_α"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></ComposedChart></ResponsiveContainer></GC></Sec>
      </>)}
      {ws==="frequency"&&(<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <Sec title="Damper Histogram"><GC><ResponsiveContainer width="100%" height={260}><BarChart data={damperH} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="vel" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><Bar dataKey="front" fill={C.cy} fillOpacity={.5} name="Front" radius={[2,2,0,0]}/><Bar dataKey="rear" fill={C.am} fillOpacity={.5} name="Rear" radius={[2,2,0,0]}/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></BarChart></ResponsiveContainer></GC></Sec>
        <Sec title="PSD Overlay"><GC><ResponsiveContainer width="100%" height={260}><LineChart data={psd} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="freq" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="real" stroke={C.am} strokeWidth={1.5} dot={false} name="Real"/><Line type="monotone" dataKey="sim" stroke={C.cy} strokeWidth={1.5} dot={false} name="Sim" strokeDasharray="4 2"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></LineChart></ResponsiveContainer></GC></Sec>
        <Sec title="Bode Response"><GC><ResponsiveContainer width="100%" height={240}><LineChart data={freqResp} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="freq" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3"/><Line type="monotone" dataKey="front_dB" stroke={C.cy} strokeWidth={2} dot={false} name="Front"/><Line type="monotone" dataKey="rear_dB" stroke={C.am} strokeWidth={2} dot={false} name="Rear"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></LineChart></ResponsiveContainer></GC></Sec>
        <Sec title="Roll Center Migration"><GC><ResponsiveContainer width="100%" height={240}><ScatterChart margin={{top:10,right:20,bottom:24,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="rcXf" type="number" {...ax}/><YAxis dataKey="rcYf" type="number" {...ax}/><Tooltip contentStyle={TT}/><Scatter data={rcMig} fill={C.cy} fillOpacity={.4} r={3} name="Front"/><Scatter data={rcMig.map(d=>({...d,rcXf:d.rcXr,rcYf:d.rcYr}))} fill={C.am} fillOpacity={.4} r={3} name="Rear"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></ScatterChart></ResponsiveContainer></GC></Sec>
      </div>)}
      {ws==="spatial"&&(<><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <Sec title="Slip Energy Map"><GC style={{padding:"14px"}}>{(()=>{const xs=slipE.map(p=>Number(p.x)),ys=slipE.map(p=>Number(p.y));const xn=Math.min(...xs),xx=Math.max(...xs),yn=Math.min(...ys),yx=Math.max(...ys),W=360,H=260,pd=20;const sc=Math.min((W-2*pd)/(xx-xn||1),(H-2*pd)/(yx-yn||1)),ox=pd+(W-2*pd-(xx-xn)*sc)/2,oy=pd+(H-2*pd-(yx-yn)*sc)/2,eM=Math.max(...slipE.map(d=>d.energy),1);return<svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:260}}>{slipE.map((p,i)=>{if(!i)return null;const pv=slipE[i-1],r=Math.min(1,p.energy/eM);return<line key={i} x1={ox+(Number(pv.x)-xn)*sc} y1={H-(oy+(Number(pv.y)-yn)*sc)} x2={ox+(Number(p.x)-xn)*sc} y2={H-(oy+(Number(p.y)-yn)*sc)} stroke={`rgb(${Math.round(225*r)},${Math.round(80*(1-r))},${Math.round(40+60*(1-r))})`} strokeWidth={3} strokeLinecap="round"/>;})}</svg>;})()}</GC></Sec>
        <Sec title="G-G Friction Circle"><GC><ResponsiveContainer width="100%" height={260}><ScatterChart margin={{top:10,right:10,bottom:24,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="lat" type="number" domain={[-2,2]} {...ax}/><YAxis dataKey="lon" type="number" domain={[-2,2]} {...ax}/><ReferenceLine x={0} stroke={C.dm} strokeWidth={.5}/><ReferenceLine y={0} stroke={C.dm} strokeWidth={.5}/><Scatter data={ggData} fill={C.red} fillOpacity={.35} r={2}/></ScatterChart></ResponsiveContainer></GC></Sec>
      </div><Sec title="Friction Saturation"><GC><ResponsiveContainer width="100%" height={160}><BarChart data={fricSat} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="bin" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><Bar dataKey="count" radius={[2,2,0,0]} barSize={12}>{fricSat.map((e,i)=><Cell key={i} fill={e.bin>=.85?C.red:e.bin>=.6?C.am:C.cy} fillOpacity={.7}/>)}</Bar></BarChart></ResponsiveContainer></GC></Sec></>)}
      {ws==="ai"&&(<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <Sec title="Receding Horizon"><GC><ResponsiveContainer width="100%" height={260}><ComposedChart data={horizon} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="xPred" type="number" {...ax}/><YAxis dataKey="yPred" type="number" {...ax}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="yBoundL" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3"/><Line type="monotone" dataKey="yBoundR" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3"/><Line type="monotone" dataKey="yPred" stroke={C.cy} strokeWidth={2} dot={false}/></ComposedChart></ResponsiveContainer></GC></Sec>
        <Sec title="Wavelet Sparsity"><GC style={{padding:"12px"}}>{["cA3","cD3","cD2","cD1"].map(lv=>{const co=wavelets.filter(w=>w.level===lv);return<div key={lv} style={{marginBottom:6}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}><span style={{fontSize:9,fontFamily:C.dt,color:lv==="cA3"?C.cy:C.am,fontWeight:700}}>{lv}</span><span style={{fontSize:7,fontFamily:C.dt,color:C.dm}}>{co.filter(w=>w.active).length}/{co.length}</span></div><div style={{display:"flex",gap:1}}>{co.map((w,j)=><div key={j} style={{flex:1,height:14,borderRadius:2,background:w.active?(lv==="cA3"?C.cy:C.am):C.b1,opacity:.3+w.mag*.7}}/>)}</div></div>;})}</GC></Sec>
        <Sec title="AL Slack"><GC><ResponsiveContainer width="100%" height={200}><AreaChart data={alSlack} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis {...ax}/><Tooltip contentStyle={TT}/><Area type="monotone" dataKey="slackGrip" stroke={C.red} fill={`${C.red}08`} strokeWidth={1.5} dot={false} name="Grip"/><Area type="monotone" dataKey="slackSteer" stroke={C.am} fill={`${C.am}06`} strokeWidth={1.2} dot={false} name="Steer"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></AreaChart></ResponsiveContainer></GC></Sec>
        <Sec title="Control Effort"><GC><ResponsiveContainer width="100%" height={200}><AreaChart data={ctrlEff} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax}/><YAxis domain={[0,1]} {...ax}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="steerUtil" stroke={C.pr} strokeWidth={1.5} dot={false} name="Steer"/><Line type="monotone" dataKey="brakeUtil" stroke={C.red} strokeWidth={1.5} dot={false} name="Brake"/><Line type="monotone" dataKey="throttleUtil" stroke={C.gn} strokeWidth={1.5} dot={false} name="Throttle"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></AreaChart></ResponsiveContainer></GC></Sec>
      </div>)}
    </FadeSlide>
  </>);
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════
export default function TelemetryModule({ track, tireTemps, mode }) {
  if (mode === "LIVE") return <LiveMode mode={mode} />;
  return <AnalyzeMode track={track} tireTemps={tireTemps} />;
}