// src/live/LiveMode.jsx — v4.5

import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { C, GL, GS, TT, AX } from "../theme.js";
import { Sec, GC, Pill } from "../components.jsx";
import { TubeTrackMap, ThermalQuintet, WaveletHeatmap } from "../canvas";
import LiveGraphPanel, { GraphSlot } from "./LiveGraphSystem.jsx";
import {
  FourCornerEllipses, StabilityPhasePlane, AeroPlatform,
  DamperBars, WMPCCostRadar, LivePacejkaCurve, DriverInputsPanel,
} from "./LiveAdvancedPanels.jsx";

const Lbl=({children,color})=><span style={{fontSize:9,fontWeight:700,color:color||C.dm,fontFamily:C.dt,letterSpacing:1.5,textTransform:"uppercase"}}>{children}</span>;
const Val=({children,color,big})=><span style={{fontSize:big?20:14,fontWeight:700,color:color||C.w,fontFamily:C.dt}}>{children}</span>;
const Vu=({children})=><span style={{fontSize:8,color:C.dm,fontFamily:C.dt,marginLeft:2}}>{children}</span>;
const StatusDot=({ok})=><div style={{width:6,height:6,borderRadius:"50%",background:ok?C.gn:C.red,boxShadow:`0 0 5px ${ok?C.gn:C.red}`,display:"inline-block"}}/>;
const tc=(v,lo,hi)=>v<lo?C.gn:v<hi?C.am:C.red;
const MRow=({label,value,unit,color})=>(<div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"2px 0",borderBottom:`1px solid ${C.b1}15`}}><Lbl>{label}</Lbl><div><Val color={color}>{value}</Val>{unit&&<Vu>{unit}</Vu>}</div></div>);
const Panel=({title,color,children,style})=>(<div style={{...GL,padding:"8px 10px",borderTop:`2px solid ${color||C.cy}`,...style}}><div style={{marginBottom:5}}><Lbl color={color}>{title}</Lbl></div>{children}</div>);

const HIST_LEN=200;
function useLiveData(active){
  const[frame,setFrame]=useState(null);const history=useRef([]);const step=useRef(0);
  useEffect(()=>{if(!active)return;const id=setInterval(()=>{
    step.current+=1;const t=step.current*0.05;const R=Math.random;
    const s=12*Math.sin(t*1.5),spd=14+4*Math.sin(t*0.3)+R();
    const k=0.08*Math.sin(t/15)+0.04*Math.sin(t/7);
    const ay=spd*spd*k/9.81,ax=(R()-0.5)*0.3,cG=Math.sqrt(ax*ax+ay*ay);
    const wz=ay*0.6+R()*0.1,bs=ay*0.02-0.01+R()*0.005;
    const zEqF=-12.8,zEqR=-14.2;
    const zFL=zEqF+ay*3+R()*0.5,zFR=zEqF-ay*3+R()*0.5,zRL=zEqR+ay*2.5+R()*0.4,zRR=zEqR-ay*2.5+R()*0.4;
    const m=300,hcg=0.33,tw=1.2,L=1.55,lf=0.8525,lr=0.6975;
    const dLat=m*9.81*Math.abs(ay)*hcg/tw,dLon=m*9.81*ax*hcg/L;
    const sf=m*9.81*lr/L/2,sr=m*9.81*lf/L/2;
    const tBase=82+15*Math.sin(t*0.08)+12*cG;
    const ei=()=>(R()-0.5)*0.04;
    // Derived intermediates for force/geometry calculations
    const FzFL=sf-dLon/2-dLat/2, FzFR=sf-dLon/2+dLat/2;
    const FzRL=sr+dLon/2-dLat/2, FzRR_=sr+dLon/2+dLat/2;
    const whlSpd=spd/0.2032; // wheel angular velocity
    const tBaseR=tBase-5;
    const FxFL_=m*9.81*ax*0.3, FxFR_=m*9.81*ax*0.3, FxRL_=m*9.81*ax*0.2, FxRR_=m*9.81*ax*0.2;
    const FyFL_=m*9.81*ay*0.28, FyFR_=m*9.81*ay*0.28, FyRL_=m*9.81*ay*0.22, FyRR_=m*9.81*ay*0.22;
    const FzAeroF_=0.5*1.225*1.2*spd*spd*0.45, FzAeroR_=0.5*1.225*1.2*spd*spd*0.55;
    const muT=0.95+0.05*Math.sin(t*0.1)-0.1*Math.max(0,tBase-110)/50;
    const muP=1+0.02*Math.sin(t*0.05);
    const solveMs=6+5*R(),Htotal=4500+500*Math.sin(t*0.2);

    const f={step:step.current,t:+t.toFixed(2),
      // ═══ CHASSIS (18) ═══
      posX:+(Math.cos(t*0.3)*20).toFixed(2), posY:+(Math.sin(t*0.3)*15).toFixed(2), posZ:+(0.33+0.005*Math.sin(t*3)).toFixed(4),
      roll:+(ay*1.2+R()*0.1).toFixed(2), pitch:+(ax*0.8+R()*0.05).toFixed(2), yaw:+(t*20%360-180).toFixed(1),
      speed:+spd.toFixed(1), vx:+spd.toFixed(1), vy:+(spd*bs).toFixed(3), vz:+(R()*0.02-0.01).toFixed(4),
      wx:+(ay*0.3+R()*0.05).toFixed(3), wy:+(ax*0.2+R()*0.03).toFixed(3), wz:+wz.toFixed(3),
      ax:+ax.toFixed(3), ay:+ay.toFixed(3), az:+(R()*0.02-0.01).toFixed(3),
      combinedG:+cG.toFixed(3), sideslip:+bs.toFixed(4),
      // ═══ SUSPENSION (20) ═══
      zFL:+zFL.toFixed(1), zFR:+zFR.toFixed(1), zRL:+zRL.toFixed(1), zRR:+zRR.toFixed(1),
      zdFL:+(R()*0.2-0.1).toFixed(3), zdFR:+(R()*0.2-0.1).toFixed(3), zdRL:+(R()*0.15-0.075).toFixed(3), zdRR:+(R()*0.15-0.075).toFixed(3),
      FspFL:+(35000*(zFL-zEqF)/1000).toFixed(0), FspFR:+(35000*(zFR-zEqF)/1000).toFixed(0),
      FdmpFL:+(2500*(R()-0.5)*0.3).toFixed(0), FdmpFR:+(2500*(R()-0.5)*0.3).toFixed(0),
      FbsFL:+(Math.max(0,Math.abs(zFL)-25)*500).toFixed(0),
      FarbF:+(1200*(zFL-zFR)/1000).toFixed(0), FarbR:+(1000*(zRL-zRR)/1000).toFixed(0),
      FzFL:+FzFL.toFixed(0), FzFR:+FzFR.toFixed(0), FzRL:+FzRL.toFixed(0), FzRR:+FzRR_.toFixed(0),
      // ═══ WHEELS (6) ═══
      whlAngFL:+(step.current*whlSpd*0.05).toFixed(0), whlAngFR:+(step.current*whlSpd*0.05).toFixed(0),
      whlSpdFL:+(whlSpd+R()*2).toFixed(1), whlSpdFR:+(whlSpd+R()*2).toFixed(1),
      whlSpdRL:+(whlSpd+R()*1.5).toFixed(1), whlSpdRR:+(whlSpd+R()*1.5).toFixed(1),
      // ═══ THERMAL (14) ═══
      TsurfInF:+(tBase+R()*3).toFixed(1), TsurfMdF:+(tBase+2+R()*2).toFixed(1), TsurfOtF:+(tBase+1+R()*3).toFixed(1),
      TgasF:+(35+5*Math.sin(t*0.02)).toFixed(1), TcoreF:+(tBase-20+R()*2).toFixed(1),
      TsurfInR:+(tBaseR+R()*3).toFixed(1), TsurfMdR:+(tBaseR+2+R()*2).toFixed(1), TsurfOtR:+(tBaseR+1+R()*3).toFixed(1),
      TgasR:+(34+4*Math.sin(t*0.02)).toFixed(1), TcoreR:+(tBaseR-18+R()*2).toFixed(1),
      flashDtF:+(8+5*cG+R()*3).toFixed(1), flashDtR:+(6+4*cG+R()*3).toFixed(1),
      muTherm:+muT.toFixed(3), muPress:+muP.toFixed(3),
      // ═══ SLIP (13) ═══
      alphaFL:+(ay*4.2+R()).toFixed(2), kappaFL:+(ax*0.05+R()*0.02).toFixed(3),
      alphaFR:+(ay*3.8+R()).toFixed(2), kappaFR:+(ax*0.05+R()*0.02).toFixed(3),
      alphaRL:+(ay*3.5+R()).toFixed(2), kappaRL:+(ax*0.04+R()*0.015).toFixed(3),
      alphaRR:+(ay*3.2+R()).toFixed(2), kappaRR:+(ax*0.04+R()*0.015).toFixed(3),
      aKinF:+(ay*4+R()*0.5).toFixed(2), aKinR:+(ay*3.3+R()*0.5).toFixed(2),
      kKinF:+(ax*0.055+R()*0.01).toFixed(3), kKinR:+(ax*0.045+R()*0.01).toFixed(3),
      fricUtil:+(Math.min(100,cG/1.35*100)).toFixed(0),
      // ═══ TIRE FORCES (16) ═══
      FxFL:+FxFL_.toFixed(0), FxFR:+FxFR_.toFixed(0), FxRL:+FxRL_.toFixed(0), FxRR:+FxRR_.toFixed(0),
      FyFL:+FyFL_.toFixed(0), FyFR:+FyFR_.toFixed(0), FyRL:+FyRL_.toFixed(0), FyRR:+FyRR_.toFixed(0),
      MzFL:+(ay*12+R()*3).toFixed(1), MzFR:+(ay*11+R()*3).toFixed(1),
      Gyk:+(1-0.3*Math.abs(ax*0.05)).toFixed(3), Gxa:+(1-0.4*Math.abs(ay*0.05)).toFixed(3),
      dFxPinn:+(50*(R()-0.5)).toFixed(0), dFyPinn:+(80*(R()-0.5)).toFixed(0),
      sigGP:+(5000+10000*cG+R()*3000).toFixed(0), lcbPen:+(0.03+0.05*cG).toFixed(3),
      // ═══ AERO (7) ═══
      FzAeroF:+FzAeroF_.toFixed(0), FzAeroR:+FzAeroR_.toFixed(0),
      FxAero:+(0.5*1.225*0.8*spd*spd).toFixed(0),
      MyAero:+(0.5*1.225*0.1*spd*spd).toFixed(0), MxAero:+(ay*5).toFixed(1),
      rideHF:+(30-0.02*spd*spd+R()*2).toFixed(1), rideHR:+(45-0.015*spd*spd+R()*2).toFixed(1),
      // ═══ GEOMETRY (9) ═══
      steer:+s.toFixed(1),
      camFL:+(-2.5+ay*0.3+zFL*0.01).toFixed(2), camFR:+(-2.5-ay*0.3+zFR*0.01).toFixed(2),
      camRL:+(-1.8+ay*0.2+zRL*0.008).toFixed(2), camRR:+(-1.8-ay*0.2+zRR*0.008).toFixed(2),
      toeFL:+(0.1+zFL*0.002).toFixed(3), toeFR:+(-0.1+zFR*0.002).toFixed(3),
      compSteer:+(ay*0.12+R()*0.03).toFixed(3), turnSlip:+k.toFixed(5),
      // ═══ ENERGY (9) ═══
      Htotal:+Htotal.toFixed(0), Tkin:+(0.5*m*spd*spd).toFixed(0),
      Vstruct:+(0.5*35000*((zFL-zEqF)**2+(zFR-zEqF)**2+(zRL-zEqR)**2+(zRR-zEqR)**2)/1e6).toFixed(1),
      Hres:+(15+2*Math.sin(t*0.05)).toFixed(1),
      dHdt:+(-10+20*R()).toFixed(1), Rdiss:+(8+12*R()).toFixed(1),
      Qtherm:+(5+3*cG+R()*2).toFixed(1),
      passViol:+(Math.max(0,-10+20*R())*0.3).toFixed(2), RnetCond:+(12+5*R()).toFixed(1),
      // ═══ WMPC (19) ═══
      ctrlSteer:+(s*(0.9+0.2*R())).toFixed(1), ctrlThrot:+(Math.max(0,ax)*0.8+R()*0.1).toFixed(3),
      ctrlBrake:+(Math.max(0,-ax)*2000+R()*100).toFixed(0),
      muNmean:+(R()*0.4-0.2).toFixed(3), sigNvar:+(0.3+0.5*R()).toFixed(3),
      dLeft:+(1.5+R()).toFixed(2), dRight:+(1.5+R()).toFixed(2),
      lamGrip:+(1+3*(1-Math.exp(-step.current/200))+R()*0.3).toFixed(2),
      lamSteer:+(0.5+R()*0.3).toFixed(2), lamAx:+(0.3+R()*0.2).toFixed(2), lamTrack:+(0.8+R()*0.4).toFixed(2),
      slkGrip:+(0.15*Math.exp(-step.current/100)+0.03*R()).toFixed(3),
      slkSteer:+(0.12+0.04*R()).toFixed(3), slkTrack:+(0.2+0.05*R()).toFixed(3),
      alRho:+(50+200*(1-Math.exp(-step.current/150))).toFixed(0),
      lbfgsIter:Math.round(5+8*R()),
      costLap:+(20+15*R()).toFixed(1), costTrack:+(5+8*R()).toFixed(1), costL1:+(2+4*R()).toFixed(1),
      jaxFwdMs:+(2+3*R()).toFixed(1), jaxBwdMs:+(4+6*R()).toFixed(1), solveMs:+solveMs.toFixed(1),
      // ═══ E-DRIVE (7) ═══
      invTmpR:+(42+8*cG+R()*3).toFixed(0), invTmpL:+(41+8*cG+R()*3).toFixed(0),
      motTmpR:+(55+12*cG+R()*4).toFixed(0), motTmpL:+(54+12*cG+R()*4).toFixed(0),
      tvTarget:+(ay*120).toFixed(0), tvActual:+(ay*120*(0.9+0.2*R())).toFixed(0),
      tvError:+(ay*12*(R()-0.5)).toFixed(0),
      // ═══ TWIN (5) ═══
      ekfAx:+ei().toFixed(3), ekfWz:+ei().toFixed(3), ekfVy:+ei().toFixed(3),
      modConf:+(92+5*R()).toFixed(0), twinFid:+(88+8*R()).toFixed(0),
    };
    history.current.push(f);if(history.current.length>HIST_LEN)history.current.shift();setFrame(f);
  },50);return()=>clearInterval(id);},[active]);
  return{frame,history,stepIdx:step};
}

// Row wrapper — forces all children to same height via stretch + child height
const Row = ({ cols, gap = 8, mb = 10, minH, children }) => (
  <div style={{
    display: "grid",
    gridTemplateColumns: cols,
    gap,
    marginBottom: mb,
    alignItems: "stretch",
    ...(minH ? { minHeight: minH } : {}),
  }}>
    {React.Children.map(children, child => (
      <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        {child}
      </div>
    ))}
  </div>
);

// Box wrapper — stretches canvas content to fill parent grid cell
const Box = ({ title, children, noPad }) => (
  <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
    <div style={{ fontSize: 9, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5, textTransform: "uppercase", borderLeft: `2px solid ${C.red}`, paddingLeft: 8, marginBottom: 6 }}>{title}</div>
    <div style={{ ...GL, padding: noPad ? 0 : 6, flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
        {children}
      </div>
    </div>
  </div>
);

export default function LiveMode({mode,thermal5,tubes,wavelets,alData,energy,track}){
  const{frame,history,stepIdx}=useLiveData(mode==="LIVE");
  const[alView,setAlView]=useState("slack");
  const[fpvExpanded,setFpvExpanded]=useState(true);
  const step=stepIdx.current||0;
  const f=frame||{};
  const sl=a=>Math.max(1,(a||[]).length);
  const alW=useMemo(()=>(alData||[]).length?alData.slice(Math.max(0,(step%sl(alData))-80),(step%sl(alData))+1):[],[alData,step]);
  const enW=useMemo(()=>(energy||[]).length?energy.slice(Math.max(0,(step%sl(energy))-100),(step%sl(energy))+1):[],[energy,step]);
  const ef=(energy||[]).length?energy[step%energy.length]:{};
  const af=(alData||[]).length?alData[step%alData.length]:{};

  return(<div>
    {/* ═══ ROW 0: KPI BAR ════════════════════════════════════════════ */}
    <Row cols="repeat(8,1fr)" gap={5} mb={10}>
      {[["SPEED",f.speed,C.w,"m/s"],["LAT G",f.ay,C.cy,"G"],["LON G",f.ax,C.am,"G"],["COMB G",f.combinedG,C.gn,"G"],["GRIP",f.fricUtil,+f.fricUtil>85?C.red:C.gn,"%"],["WMPC",f.solveMs,tc(+f.solveMs,8,10),"ms"],["H(q,p)",f.Htotal,C.e_hnet,"J"],["λ_grip",f.lamGrip,C.am,""]].map(([l,v,c,u],i)=>
        <div key={i} style={{...GL,padding:"6px 8px",textAlign:"center",borderTop:`2px solid ${c}`}}>
          <div style={{fontSize:8,color:C.dm,fontFamily:C.dt,letterSpacing:1.5}}>{l}</div>
          <div style={{fontSize:18,fontWeight:800,color:c,fontFamily:C.dt}}>{v||"—"}</div>
          <div style={{fontSize:8,color:C.dm,fontFamily:C.dt}}>{u}</div></div>)}
    </Row>

    {/* ═══ ROW 1: FPV + DRIVER INPUTS + TRACK ════════════════════════ */}
    <Row cols="1.4fr 0.6fr 1fr" minH={380}>
      {/* FPV Camera */}
      <div style={{...GL,padding:0,overflow:"hidden",borderTop:`2px solid ${C.cy}`,display:"flex",flexDirection:"column"}}>
        <div style={{flex:1,minHeight:200,background:`linear-gradient(135deg,${C.bg2},rgba(14,20,32,0.9))`,position:"relative",overflow:"hidden"}}>
          <div style={{position:"absolute",inset:0,background:`repeating-linear-gradient(0deg,transparent,transparent 19px,rgba(0,210,255,0.03) 19px,rgba(0,210,255,0.03) 20px),repeating-linear-gradient(90deg,transparent,transparent 19px,rgba(0,210,255,0.03) 19px,rgba(0,210,255,0.03) 20px)`}}/>
          <div style={{position:"absolute",top:8,left:10,display:"flex",alignItems:"center",gap:4}}>
            <div style={{width:7,height:7,borderRadius:"50%",background:C.red,animation:"pulseGlow 1s infinite"}}/><span style={{fontSize:10,fontFamily:C.dt,color:C.red,fontWeight:700}}>REC</span>
          </div>
          <div style={{position:"absolute",top:8,right:10,fontSize:9,fontFamily:C.dt,color:C.dm}}>1920×1080 @ 60fps</div>
          <div style={{position:"absolute",bottom:8,left:10,display:"flex",gap:14}}>
            {[["SPD",f.speed,"m/s",C.gn],["LAT",f.ay,"G",C.cy],["LON",f.ax,"G",C.am],["μ%",f.fricUtil,"",+f.fricUtil>85?C.red:C.gn]].map(([l,v,u,c])=>(
              <div key={l}><div style={{fontSize:7,color:C.dm,fontFamily:C.dt}}>{l}</div><div style={{fontSize:16,fontWeight:800,color:c,fontFamily:C.dt}}>{v||"—"}</div></div>))}
          </div>
          <div style={{position:"absolute",bottom:8,right:10}}><StatusDot ok={true}/><span style={{fontSize:8,fontFamily:C.dt,color:C.gn,marginLeft:3}}>TRACKING</span></div>
        </div>
        <div style={{padding:"6px 8px",background:"rgba(5,7,11,0.9)",borderTop:`1px solid ${C.b2}`}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:3}}>
            <Lbl color={C.cy}>FPV OVERLAY GRAPH</Lbl>
            <button onClick={()=>setFpvExpanded(!fpvExpanded)} style={{background:`${fpvExpanded?C.red:C.cy}15`,border:`1px solid ${fpvExpanded?C.red:C.cy}30`,borderRadius:4,padding:"2px 10px",fontSize:9,color:fpvExpanded?C.red:C.cy,fontFamily:C.dt,cursor:"pointer",fontWeight:700}}>{fpvExpanded?"▲ Compact":"▼ Expand"}</button>
          </div>
          <GraphSlot id={99} history={history.current||[]} compact={true} tick={step}/>
        </div>
      </div>

      {/* DRIVER INPUTS — 3 stacked line charts */}
      <Box title="Driver Inputs">
        <DriverInputsPanel history={history.current||[]} tick={step}/>
      </Box>

      {/* Track Map */}
      <Box title="Track + Safety Tubes">
        <TubeTrackMap track={track} tubes={tubes} step={step%Math.max(1,track.length)} width={320} height={340}/>
      </Box>
    </Row>

    {/* ═══ ROW 2: THERMAL + ELLIPSES + WAVELET + EDRIVE ══════════════ */}
    <Row cols="1fr 1fr 1.4fr 0.5fr" minH={260}>
      <Box title="5-Node Thermal">
        <ThermalQuintet data={thermal5} step={step%Math.max(1,thermal5.length)} size={80}/>
      </Box>
      <Box title="4-Corner Friction Ellipses">
        <FourCornerEllipses f={f}/>
      </Box>
      <Box title="MPC Wavelet (Db4)">
        <WaveletHeatmap data={wavelets} width={380} height={200}/>
      </Box>
      <div style={{display:"flex",flexDirection:"column",gap:4}}>
        <Panel title="INV" color={C.cy}><MRow label="R" value={f.invTmpR||"—"} unit="°C" color={tc(+f.invTmpR,70,85)}/><MRow label="L" value={f.invTmpL||"—"} unit="°C" color={tc(+f.invTmpL,70,85)}/></Panel>
        <Panel title="MOT" color={C.am}><MRow label="R" value={f.motTmpR||"—"} unit="°C" color={tc(+f.motTmpR,80,100)}/><MRow label="L" value={f.motTmpL||"—"} unit="°C" color={tc(+f.motTmpL,80,100)}/></Panel>
        <Panel title="TV" color={C.pr}><MRow label="Tgt" value={f.tvTarget||"—"} unit="Nm"/><MRow label="Err" value={f.tvError||"—"} unit="Nm" color={Math.abs(+f.tvError)>10?C.am:C.gn}/></Panel>
      </div>
    </Row>

    {/* ═══ ROW 3: STABILITY + DAMPERS + AERO + RADAR ═════════════════ */}
    <Row cols="1fr 1.2fr 1fr 0.8fr" minH={240}>
      <Box title="β-ψ̇ Phase Plane"><StabilityPhasePlane f={f}/></Box>
      <Box title="Damper Saturation"><DamperBars f={f}/></Box>
      <Box title="Aero Platform"><AeroPlatform f={f}/></Box>
      <Box title="WMPC Cost Radar"><WMPCCostRadar f={f}/></Box>
    </Row>

    {/* ═══ ROW 4: AL CONSTRAINTS + ENERGY FLOW ═══════════════════════ */}
    <Row cols="1fr 1fr" minH={220}>
      <Sec title="AL Constraint Monitor" right={<div style={{display:"flex",gap:4}}><Pill active={alView==="slack"} label="Slack" onClick={()=>setAlView("slack")} color={C.cy}/><Pill active={alView==="lambda"} label="λ" onClick={()=>setAlView("lambda")} color={C.am}/></div>}>
        <GC><ResponsiveContainer width="100%" height={160}><AreaChart data={alW} margin={{top:8,right:12,bottom:8,left:8}}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="solve" {...AX}/><YAxis {...AX}/><Tooltip contentStyle={TT}/>
          {alView==="slack"?[["grip",C.red],["steer",C.am],["ax",C.cy],["track",C.gn],["vx",C.pr]].map(([k,c])=><Area key={k} type="monotone" dataKey={k} stackId="1" fill={c} fillOpacity={.25} stroke={c} strokeWidth={1.5} name={k}/>)
            :<Area type="monotone" dataKey="lambda_grip" stroke={C.am} fill={C.am} fillOpacity={.1} strokeWidth={2} name="λ_grip"/>}
          <Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/>
        </AreaChart></ResponsiveContainer>
        <div style={{display:"flex",gap:10,padding:"4px 8px",borderTop:`1px solid ${C.b1}`}}>
          <Lbl color={(af.grip||1)<.05?C.red:C.gn}>Binding: {["grip","steer","ax","track","vx"].filter(k=>(af[k]||1)<.05).length}/5</Lbl>
          <Lbl color={C.cy}>Iters: {af.iters||"—"}</Lbl><Lbl color={C.am}>ρ={f.alRho||"—"}</Lbl>
        </div></GC>
      </Sec>
      <Sec title="Energy Flow (H = T + V + H_res)"><GC>
        <ResponsiveContainer width="100%" height={160}><AreaChart data={enW} margin={{top:8,right:12,bottom:8,left:8}}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="t" {...AX}/><YAxis {...AX}/><Tooltip contentStyle={TT}/>
          <Area type="monotone" dataKey="ke" stackId="1" fill={C.e_ke} fillOpacity={.25} stroke={C.e_ke} strokeWidth={1.5} name="T_kin"/>
          <Area type="monotone" dataKey="pe_s" stackId="1" fill={C.e_spe} fillOpacity={.2} stroke={C.e_spe} strokeWidth={1} name="V_struct"/>
          <Area type="monotone" dataKey="h_net" stackId="1" fill={C.e_hnet} fillOpacity={.3} stroke={C.e_hnet} strokeWidth={2} name="H_res"/>
          <Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/>
        </AreaChart></ResponsiveContainer>
        <div style={{display:"flex",justifyContent:"space-between",padding:"4px 8px",borderTop:`1px solid ${C.b1}`}}>
          <div><Lbl>H=</Lbl><Val color={C.cy}>{ef.H||f.Htotal||"—"}</Val><Vu>J</Vu></div>
          <div><Lbl>dH/dt=</Lbl><Val color={+(ef.dH||f.dHdt)>0?C.red:C.gn}>{ef.dH||f.dHdt||"—"}</Val><Vu>W</Vu></div>
          <div><Lbl>R=</Lbl><Val color={C.e_diss}>{ef.r_diss||f.Rdiss||"—"}</Val><Vu>W</Vu></div>
          <div><Lbl>Q̇=</Lbl><Val color={C.e_therm}>{f.Qtherm||"—"}</Val><Vu>W</Vu></div>
        </div></GC>
      </Sec>
    </Row>

    {/* ═══ ROW 5: PACEJKA + TWIN HEALTH ══════════════════════════════ */}
    <Row cols="1.3fr 1fr" minH={240}>
      <Box title="Live Effective Pacejka — Thermal + GP ±2σ">
        <LivePacejkaCurve f={f}/>
      </Box>
      <div style={{display:"flex",flexDirection:"column",gap:6}}>
        <Sec title="Twin Health">
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:5}}>
            <Panel title="EKF a_x" color={C.cy}><div style={{display:"flex",gap:4,alignItems:"center"}}><StatusDot ok={Math.abs(+f.ekfAx)<0.03}/><Val color={C.cy}>{f.ekfAx||"—"}</Val></div></Panel>
            <Panel title="EKF ω_z" color={C.am}><div style={{display:"flex",gap:4,alignItems:"center"}}><StatusDot ok={Math.abs(+f.ekfWz)<0.03}/><Val color={C.am}>{f.ekfWz||"—"}</Val></div></Panel>
            <Panel title="Conf." color={C.gn} style={{textAlign:"center"}}><Val big color={+f.modConf>90?C.gn:C.am}>{f.modConf||"—"}</Val><Vu>%</Vu></Panel>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:5,marginTop:5}}>
            <Panel title="JAX fwd" color={C.gn} style={{textAlign:"center"}}><Val color={C.gn}>{f.jaxFwdMs||"—"}</Val><Vu>ms</Vu></Panel>
            <Panel title="JAX bwd" color={C.cy} style={{textAlign:"center"}}><Val color={C.cy}>{f.jaxBwdMs||"—"}</Val><Vu>ms</Vu></Panel>
            <Panel title="CONSTR." color={C.red}>
              <div style={{fontSize:8,fontFamily:C.dt}}>
                {+f.fricUtil>95?<div style={{color:C.red}}>μ binding</div>:null}
                {+f.passViol>1?<div style={{color:C.am}}>Passivity</div>:null}
                {!+f.fricUtil&&!+f.passViol&&<span style={{color:C.gn}}>✓ Clear</span>}
              </div>
            </Panel>
          </div>
        </Sec>
      </div>
    </Row>

    {/* ═══ ROW 6: CONFIGURABLE GRAPHS ════════════════════════════════ */}
    <Sec title="Configurable Telemetry Graphs">
      <LiveGraphPanel history={history} tick={step}/>
    </Sec>
  </div>);
}