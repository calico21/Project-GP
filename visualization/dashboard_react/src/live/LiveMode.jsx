// ═══════════════════════════════════════════════════════════════════════════════
// src/live/LiveMode.jsx — Live Telemetry Orchestrator v4.2
// ═══════════════════════════════════════════════════════════════════════════════
//
// Layout:
//   ROW 0: 10-KPI numeric bar with color-coded thresholds
//   ROW 1: Track+Tubes (Canvas) | FPV Camera (expandable) | 5-Node Thermal
//   ROW 2: Wavelet Heatmap | G-Force Diagram (Canvas) | E-Drive + TV Panel
//   ROW 3: AL Constraint Monitor | Energy Flow (Port-Hamiltonian)
//   ROW 4: Twin Health status row (EKF, λ_μ, WMPC, H(q,p), Constraints)
//   ROW 5: Configurable time-series graphs (add/remove, 40+ channels)
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import { C, GL, GS, TT, AX } from "../theme.js";
import { Sec, GC, Pill } from "../components.jsx";
import { TubeTrackMap, ThermalQuintet, WaveletHeatmap } from "../canvas";
import LiveGraphPanel from "./LiveGraphSystem.jsx";

// ═════════════════════════════════════════════════════════════════════════════
// MICRO-COMPONENTS
// ═════════════════════════════════════════════════════════════════════════════
const Lbl=({children,color})=><span style={{fontSize:8,fontWeight:700,color:color||C.dm,fontFamily:C.dt,letterSpacing:1.5,textTransform:"uppercase"}}>{children}</span>;
const Val=({children,color,big})=><span style={{fontSize:big?18:12,fontWeight:700,color:color||C.w,fontFamily:C.dt}}>{children}</span>;
const Vu=({children})=><span style={{fontSize:7,color:C.dm,fontFamily:C.dt,marginLeft:2}}>{children}</span>;
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

// ═════════════════════════════════════════════════════════════════════════════
// LIVE DATA HOOK — expanded with 40+ synthetic channels
// ═════════════════════════════════════════════════════════════════════════════
const HIST_LEN=200;

function useLiveData(active){
  const[frame,setFrame]=useState(null);const history=useRef([]);const step=useRef(0);
  useEffect(()=>{if(!active)return;const id=setInterval(()=>{
    step.current+=1;const t=step.current*0.05;const s=12*Math.sin(t*1.5);
    const spd=14+4*Math.sin(t*0.3)+Math.random();
    const k=0.08*Math.sin(t/15)+0.04*Math.sin(t/7);
    const latG=spd*spd*k/9.81;const lonG=(Math.random()-0.5)*0.3;
    const cG=Math.sqrt(latG*latG+lonG*lonG);
    const yr=latG*0.6+Math.random()*0.1;const bs=latG*0.02-0.01+Math.random()*0.005;
    const ms=6+5*Math.random();const hj=4.5+0.5*Math.sin(t*0.2);
    const ei=Math.random()*0.04-0.02;
    // Thermal (synthetic 5-node)
    const tBase=80+15*Math.sin(t*0.1)+10*cG;
    // Forces
    const m=300,hcg=0.33,tw=1.2,L=1.55;
    const dLat=m*9.81*Math.abs(latG)*hcg/tw;const dLon=m*9.81*lonG*hcg/L;
    const sf=m*9.81*0.45,sr=m*9.81*0.55;
    const f={
      step:step.current,t:+t.toFixed(2),speed:+spd.toFixed(1),steer:+s.toFixed(1),
      latG:+latG.toFixed(3),lonG:+lonG.toFixed(3),combinedG:+cG.toFixed(3),
      curvature:+k.toFixed(5),yawRate:+yr.toFixed(3),sideslip:+bs.toFixed(4),
      rollRate:+(yr*0.3+Math.random()*0.1).toFixed(3),
      pitchRate:+(lonG*0.2+Math.random()*0.05).toFixed(3),
      wmpcSolveMs:+ms.toFixed(1),hamiltonianJ:+hj.toFixed(2),ekfInnov:+ei.toFixed(3),
      ekfInnovWz:+(Math.random()*0.03-0.015).toFixed(3),
      modelConf:+(92+5*Math.random()).toFixed(0),
      // Thermal
      tfl_surf:+(tBase+Math.random()*3).toFixed(1),tfr_surf:+(tBase+2+Math.random()*3).toFixed(1),
      trl_surf:+(tBase-5+Math.random()*3).toFixed(1),trr_surf:+(tBase-3+Math.random()*3).toFixed(1),
      tfl_flash:+(tBase+8+Math.random()*5).toFixed(1),tfr_flash:+(tBase+10+Math.random()*5).toFixed(1),
      tfl_core:+(tBase-15+Math.random()*2).toFixed(1),tfr_core:+(tBase-13+Math.random()*2).toFixed(1),
      flashDelta:+(8+5*cG+Math.random()*3).toFixed(1),
      // Slip
      slipAngleF:+(latG*4+Math.random()).toFixed(2),slipAngleR:+(latG*3.5+Math.random()).toFixed(2),
      slipRatioFL:+(lonG*0.05+Math.random()*0.02).toFixed(3),slipRatioFR:+(lonG*0.05+Math.random()*0.02).toFixed(3),
      gripUtil:+(Math.min(100,cG/1.35*100)).toFixed(0),
      // Forces
      fzFL:+(sf-dLon/2-dLat/2).toFixed(0),fzFR:+(sf-dLon/2+dLat/2).toFixed(0),
      fzRL:+(sr+dLon/2-dLat/2).toFixed(0),fzRR:+(sr+dLon/2+dLat/2).toFixed(0),
      fxTotal:+(m*9.81*lonG).toFixed(0),fyTotal:+(m*9.81*latG).toFixed(0),
      // MPC
      alIters:Math.round(3+5*Math.random()),alSlackGrip:+(0.05+0.15*Math.random()).toFixed(3),
      horizonErr:+(0.1+0.3*Math.random()).toFixed(2),
      // Energy
      dHdt:+(-10+20*Math.random()).toFixed(1),rDiss:+(8+12*Math.random()).toFixed(1),
      hNetResid:+(15+2*Math.sin(t*0.05)).toFixed(1),
      // Driver
      throttle:+(Math.max(0,lonG)*80+Math.random()*10).toFixed(0),
      brake:+(Math.max(0,-lonG)*70+Math.random()*5).toFixed(0),
      steerInput:+(s*14).toFixed(0),
      // Inverter & motor (e-drive)
      invTempR:+(42+8*cG+Math.random()*3).toFixed(0),invTempL:+(41+8*cG+Math.random()*3).toFixed(0),
      motorTempR:+(55+12*cG+Math.random()*4).toFixed(0),motorTempL:+(54+12*cG+Math.random()*4).toFixed(0),
      lambda_mu:+(1+3*(1-Math.exp(-step.current/200))+0.3*Math.random()).toFixed(2),
      x:+(Math.cos(t*0.3)*20).toFixed(2),y:+(Math.sin(t*0.3)*15).toFixed(2),
    };
    history.current.push(f);if(history.current.length>HIST_LEN)history.current.shift();setFrame(f);
  },50);return()=>clearInterval(id);},[active]);
  return{frame,history,stepIdx:step};
}

// ═════════════════════════════════════════════════════════════════════════════
// G-FORCE FRICTION CIRCLE (Canvas)
// ═════════════════════════════════════════════════════════════════════════════
function GForceDiagram({latG,lonG,size=160}){
  const cv=useRef(null);const trail=useRef([]);
  useEffect(()=>{
    const c=cv.current;if(!c)return;const ctx=c.getContext("2d");
    const W=c.width,H=c.height,cx=W/2,cy=H/2,r=W*0.38;
    ctx.clearRect(0,0,W,H);
    // Friction circle boundary
    [1.0,0.75,0.5,0.25].forEach(f=>{ctx.strokeStyle=`rgba(62,74,100,${0.15+f*0.1})`;ctx.lineWidth=0.5;ctx.beginPath();ctx.arc(cx,cy,r*f,0,Math.PI*2);ctx.stroke();});
    // Crosshair
    ctx.strokeStyle="rgba(62,74,100,0.3)";ctx.lineWidth=0.5;
    ctx.beginPath();ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);ctx.stroke();
    ctx.beginPath();ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.stroke();
    // Labels
    ctx.fillStyle=C.dm;ctx.font=`8px ${C.dt}`;ctx.textAlign="center";
    ctx.fillText("1.35G",cx,cy-r-4);ctx.fillText("Lat →",cx+r+2,cy+12);
    // Trail
    trail.current.push({x:+latG,y:+lonG});if(trail.current.length>60)trail.current.shift();
    trail.current.forEach((p,i)=>{
      const a=i/trail.current.length;
      const px=cx+(p.x/1.5)*r,py=cy-(p.y/1.5)*r;
      const g=Math.sqrt(p.x*p.x+p.y*p.y);const norm=Math.min(1,g/1.35);
      ctx.beginPath();ctx.arc(px,py,1.5,0,Math.PI*2);
      ctx.fillStyle=`rgba(${Math.round(225*norm)},${Math.round(230*(1-norm))},${Math.round(80+140*(1-norm))},${a*0.6})`;
      ctx.fill();
    });
    // Current dot
    const gx=cx+(+latG/1.5)*r,gy=cy-(+lonG/1.5)*r;
    ctx.beginPath();ctx.arc(gx,gy,4,0,Math.PI*2);ctx.fillStyle=C.gn;ctx.fill();
    ctx.beginPath();ctx.arc(gx,gy,7,0,Math.PI*2);ctx.strokeStyle=C.gn;ctx.lineWidth=1;ctx.globalAlpha=0.3;ctx.stroke();ctx.globalAlpha=1;
    // Utilisation readout
    const util=Math.sqrt(latG*latG+lonG*lonG)/1.35*100;
    ctx.fillStyle=util>90?C.red:util>70?C.am:C.gn;ctx.font=`bold 14px ${C.dt}`;ctx.textAlign="center";
    ctx.fillText(`${util.toFixed(0)}%`,cx,cy+r+16);
  },[latG,lonG]);
  return<canvas ref={cv} width={size*2} height={(size+24)*2} style={{width:size,height:size+24,display:"block"}}/>;
}

// ═════════════════════════════════════════════════════════════════════════════
// FPV CAMERA PANEL (expandable with graph overlay)
// ═════════════════════════════════════════════════════════════════════════════
function FPVCamera({history,expanded,setExpanded}){
  return(
    <Panel title="FPV CAMERA" color={expanded?C.cy:C.dm}>
      {!expanded?(
        <div onClick={()=>setExpanded(true)} style={{cursor:"pointer",textAlign:"center",padding:"12px 0"}}>
          <div style={{width:"100%",height:100,borderRadius:6,background:`linear-gradient(135deg,${C.bg2},rgba(14,20,32,0.8))`,display:"flex",alignItems:"center",justifyContent:"center",border:`1px dashed ${C.b2}`}}>
            <div>
              <div style={{fontSize:20,color:C.dm,marginBottom:4}}>📹</div>
              <div style={{fontSize:8,color:C.dm,fontFamily:C.dt}}>Click to expand FPV</div>
            </div>
          </div>
          <div style={{display:"flex",justifyContent:"center",gap:4,marginTop:4}}>
            <StatusDot ok={true}/><span style={{fontSize:7,fontFamily:C.dt,color:C.gn}}>● TRACKING</span>
          </div>
        </div>
      ):(
        <div>
          {/* Expanded camera view with telemetry overlay */}
          <div style={{width:"100%",height:200,borderRadius:6,background:`linear-gradient(135deg,${C.bg2},rgba(14,20,32,0.9))`,position:"relative",overflow:"hidden",border:`1px solid ${C.b2}`,marginBottom:6}}>
            {/* Simulated camera feed (gradient + grid overlay) */}
            <div style={{position:"absolute",inset:0,background:`repeating-linear-gradient(0deg,transparent,transparent 19px,rgba(0,210,255,0.03) 19px,rgba(0,210,255,0.03) 20px),repeating-linear-gradient(90deg,transparent,transparent 19px,rgba(0,210,255,0.03) 19px,rgba(0,210,255,0.03) 20px)`}}/>
            <div style={{position:"absolute",top:8,left:8,display:"flex",alignItems:"center",gap:4}}>
              <div style={{width:6,height:6,borderRadius:"50%",background:C.red,animation:"pulseGlow 1s infinite"}}/>
              <span style={{fontSize:8,fontFamily:C.dt,color:C.red,fontWeight:700}}>REC</span>
            </div>
            <div style={{position:"absolute",top:8,right:8,fontSize:7,fontFamily:C.dt,color:C.dm}}>1920×1080 @ 60fps</div>
            <div style={{position:"absolute",bottom:8,left:8,right:8,display:"flex",justifyContent:"space-between"}}>
              <span style={{fontSize:7,fontFamily:C.dt,color:C.gn}}>● TRACKING</span>
              <span style={{fontSize:7,fontFamily:C.dt,color:C.dm}}>CAM-01 · FRONT</span>
            </div>
          </div>
          <button onClick={()=>setExpanded(false)} style={{width:"100%",padding:"4px",background:`${C.red}10`,border:`1px solid ${C.red}30`,borderRadius:4,fontSize:8,color:C.red,fontFamily:C.dt,cursor:"pointer",fontWeight:700}}>✕ Close FPV</button>
        </div>
      )}
    </Panel>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// E-DRIVE & TORQUE VECTORING PANEL
// ═════════════════════════════════════════════════════════════════════════════
function EDrivePanel({f}){
  return(
    <div style={{display:"flex",flexDirection:"column",gap:4}}>
      <Panel title="INVERTER" color={C.cy}>
        <MRow label="Temp R" value={f.invTempR} unit="°C" color={tc(+f.invTempR,70,85)}/>
        <MRow label="Temp L" value={f.invTempL} unit="°C" color={tc(+f.invTempL,70,85)}/>
      </Panel>
      <Panel title="MOTOR" color={C.am}>
        <MRow label="Temp R" value={f.motorTempR} unit="°C" color={tc(+f.motorTempR,80,100)}/>
        <MRow label="Temp L" value={f.motorTempL} unit="°C" color={tc(+f.motorTempL,80,100)}/>
      </Panel>
      <Panel title="TORQUE VECTOR" color={C.pr}>
        <MRow label="Target" value={(+f.latG*120).toFixed(0)} unit="Nm"/>
        <MRow label="Split" value={+f.latG>0?"R bias":"L bias"} color={C.pr}/>
      </Panel>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// CONSTRAINT STATUS BAR
// ═════════════════════════════════════════════════════════════════════════════
function ConstraintBar({alFrame}){
  const constraints=[];
  if((+alFrame.grip||1)<0.05)constraints.push({msg:"μ-circle binding",color:C.red});
  if((+alFrame.alSlackGrip||1)<0.03)constraints.push({msg:"Grip slack < 3%",color:C.am});
  return(
    <div style={{maxHeight:28,overflowY:"auto"}}>
      {constraints.length===0?<span style={{fontSize:7,color:C.gn,fontFamily:C.dt}}>✓ All constraints clear</span>:
        constraints.map((c,i)=><div key={i} style={{fontSize:7,fontFamily:C.dt,color:c.color}}>{c.msg}</div>)}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN LIVE MODE EXPORT
// ═════════════════════════════════════════════════════════════════════════════
export default function LiveMode({mode,thermal5,tubes,wavelets,alData,energy,track}){
  const{frame,history,stepIdx}=useLiveData(mode==="LIVE");
  const[alView,setAlView]=useState("slack");
  const[fpvExpanded,setFpvExpanded]=useState(false);
  const step=stepIdx.current||0;
  const f=frame||{speed:0,steer:0,latG:0,lonG:0,combinedG:0,curvature:0,yawRate:0,sideslip:0,wmpcSolveMs:0,hamiltonianJ:0,ekfInnov:0,ekfInnovWz:0,modelConf:0,invTempR:0,invTempL:0,motorTempR:0,motorTempL:0,lambda_mu:0,throttle:0,brake:0,gripUtil:0};
  const sl=a=>Math.max(1,a.length);
  const alW=useMemo(()=>alData.length?alData.slice(Math.max(0,(step%sl(alData))-80),(step%sl(alData))+1):[],[alData,step]);
  const enW=useMemo(()=>energy.length?energy.slice(Math.max(0,(step%sl(energy))-100),(step%sl(energy))+1):[],[energy,step]);
  const ef=energy.length?energy[step%energy.length]:{};
  const af=alData.length?alData[step%alData.length]:{};

  return(<div>
    {/* ═══ ROW 0: KPI BAR ════════════════════════════════════════════ */}
    <div style={{display:"grid",gridTemplateColumns:"repeat(10,1fr)",gap:4,marginBottom:8}}>
      {[["Speed",f.speed,C.w,"m/s"],["Lat G",f.latG,C.cy,"G"],["Lon G",f.lonG,C.am,"G"],["Comb G",f.combinedG,C.gn,"G"],["ψ̇",f.yawRate,C.pr,"rad/s"],["β",f.sideslip,C.pr,"rad"],["Grip",f.gripUtil,+f.gripUtil>85?C.am:C.gn,"%"],["WMPC",f.wmpcSolveMs,tc(+f.wmpcSolveMs,8,10),"ms"],["H(q,p)",f.hamiltonianJ,C.e_hnet,"kJ"],["λ_μ",f.lambda_mu,C.am,""]].map(([l,v,c,u],i)=>
        <div key={i} style={{...GL,padding:"4px 5px",textAlign:"center",borderTop:`2px solid ${c}`}}>
          <div style={{fontSize:6,color:C.dm,fontFamily:C.dt,letterSpacing:1,textTransform:"uppercase"}}>{l}</div>
          <div style={{fontSize:14,fontWeight:800,color:c,fontFamily:C.dt}}>{v}</div>
          {u&&<div style={{fontSize:6,color:C.dm,fontFamily:C.dt}}>{u}</div>}
        </div>)}
    </div>

    {/* ═══ ROW 1: TRACK + FPV + THERMAL ══════════════════════════════ */}
    <div style={{display:"grid",gridTemplateColumns:fpvExpanded?"1fr 1.5fr 0.7fr":"1.3fr 0.7fr 0.8fr",gap:8,marginBottom:8}}>
      <Sec title="Track + Safety Tubes"><GC style={{padding:6}}>
        <TubeTrackMap track={track} tubes={tubes} step={step%Math.max(1,track.length)} width={fpvExpanded?320:380} height={fpvExpanded?220:260}/>
      </GC></Sec>
      <div style={{display:"flex",flexDirection:"column",gap:6}}>
        <FPVCamera history={history} expanded={fpvExpanded} setExpanded={setFpvExpanded}/>
      </div>
      <Sec title="Tire Thermal (5-Node)"><GC style={{padding:6}}>
        <ThermalQuintet data={thermal5} step={step%Math.max(1,thermal5.length)} size={fpvExpanded?72:88}/>
      </GC></Sec>
    </div>

    {/* ═══ ROW 2: WAVELET + G-FORCE + E-DRIVE ════════════════════════ */}
    <div style={{display:"grid",gridTemplateColumns:"1.2fr 0.6fr 0.6fr",gap:8,marginBottom:8}}>
      <Sec title="MPC Wavelet (Db4 3-Level)"><GC style={{padding:6}}>
        <WaveletHeatmap data={wavelets} width={340} height={160}/>
      </GC></Sec>
      <Sec title="Friction Circle"><GC style={{padding:4,display:"flex",justifyContent:"center"}}>
        <GForceDiagram latG={+f.latG} lonG={+f.lonG} size={140}/>
      </GC></Sec>
      <Sec title="E-Drive"><EDrivePanel f={f}/></Sec>
    </div>

    {/* ═══ ROW 3: AL CONSTRAINTS + ENERGY FLOW ═══════════════════════ */}
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:8}}>
      <Sec title="AL Constraint Monitor" right={<div style={{display:"flex",gap:4}}><Pill active={alView==="slack"} label="Slack" onClick={()=>setAlView("slack")} color={C.cy}/><Pill active={alView==="lambda"} label="λ" onClick={()=>setAlView("lambda")} color={C.am}/></div>}>
        <GC><ResponsiveContainer width="100%" height={160}><AreaChart data={alW} margin={{top:8,right:12,bottom:8,left:8}}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="solve" {...AX}/><YAxis {...AX}/><Tooltip contentStyle={TT}/>
          {alView==="slack"?[["grip",C.red],["steer",C.am],["ax",C.cy],["track",C.gn],["vx",C.pr]].map(([k,c])=><Area key={k} type="monotone" dataKey={k} stackId="1" fill={c} fillOpacity={.25} stroke={c} strokeWidth={1} name={k}/>)
            :<Line type="monotone" dataKey="lambda_grip" stroke={C.am} strokeWidth={2} dot={false} name="λ_grip"/>}
          <Legend wrapperStyle={{fontSize:7,fontFamily:C.hd}}/>
        </AreaChart></ResponsiveContainer>
        <div style={{display:"flex",gap:8,padding:"3px 8px",borderTop:`1px solid ${C.b1}`}}>
          <Lbl color={(af.grip||1)<.05?C.red:C.gn}>Binding: {["grip","steer","ax","track","vx"].filter(k=>(af[k]||1)<.05).length}/5</Lbl>
          <Lbl color={C.cy}>Iters: {af.iters||"—"}</Lbl><Lbl color={C.am}>λ: {af.lambda_grip||"—"}</Lbl>
        </div></GC>
      </Sec>
      <Sec title="Energy Flow (Port-Hamiltonian)"><GC>
        <ResponsiveContainer width="100%" height={160}><AreaChart data={enW} margin={{top:8,right:12,bottom:8,left:8}}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="t" {...AX}/><YAxis {...AX}/><Tooltip contentStyle={TT}/>
          <Area type="monotone" dataKey="ke" stackId="1" fill={C.e_ke} fillOpacity={.25} stroke={C.e_ke} strokeWidth={1} name="KE"/>
          <Area type="monotone" dataKey="pe_s" stackId="1" fill={C.e_spe} fillOpacity={.2} stroke={C.e_spe} strokeWidth={1} name="PE_s"/>
          <Area type="monotone" dataKey="pe_arb" stackId="1" fill={C.e_arb} fillOpacity={.15} stroke={C.e_arb} strokeWidth={1} name="PE_arb"/>
          <Area type="monotone" dataKey="h_net" stackId="1" fill={C.e_hnet} fillOpacity={.3} stroke={C.e_hnet} strokeWidth={1.5} name="H_net"/>
          <Legend wrapperStyle={{fontSize:7,fontFamily:C.hd}}/>
        </AreaChart></ResponsiveContainer>
        <div style={{display:"flex",justifyContent:"space-between",padding:"3px 8px",borderTop:`1px solid ${C.b1}`}}>
          <div><Lbl>H=</Lbl><Val color={C.cy}>{ef.H||"—"}</Val><Vu>J</Vu></div>
          <div><Lbl>dH/dt=</Lbl><Val color={+ef.dH>0?C.red:C.gn}>{ef.dH||"—"}</Val><Vu>W</Vu></div>
          <div><Lbl>R=</Lbl><Val color={C.e_diss}>{ef.r_diss||"—"}</Val><Vu>W</Vu></div>
        </div>
      </GC></Sec>
    </div>

    {/* ═══ ROW 4: TWIN HEALTH ════════════════════════════════════════ */}
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr 1.5fr",gap:5,marginBottom:8}}>
      <Panel title="EKF" color={C.cy}><div style={{display:"flex",gap:4,alignItems:"center"}}><StatusDot ok={Math.abs(+f.ekfInnov)<0.03}/><Val color={C.cy}>{f.ekfInnov}</Val></div></Panel>
      <Panel title="λ_μ" color={C.am} style={{textAlign:"center"}}><Val big color={C.am}>{f.lambda_mu}</Val></Panel>
      <Panel title="WMPC" color={tc(+f.wmpcSolveMs,8,10)} style={{textAlign:"center"}}><Val big color={tc(+f.wmpcSolveMs,8,10)}>{f.wmpcSolveMs}</Val><Vu>ms</Vu></Panel>
      <Panel title="H(q,p)" color={C.e_hnet} style={{textAlign:"center"}}><Val big color={C.e_hnet}>{f.hamiltonianJ}</Val><Vu>kJ</Vu></Panel>
      <Panel title="CONSTRAINTS" color={C.red}><ConstraintBar alFrame={af}/></Panel>
    </div>

    {/* ═══ ROW 5: CONFIGURABLE TIME-SERIES GRAPHS ════════════════════ */}
    <Sec title="Configurable Telemetry Graphs">
      <LiveGraphPanel history={history}/>
    </Sec>
  </div>);
}