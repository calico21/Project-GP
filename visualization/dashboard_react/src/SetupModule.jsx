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
  gSetupComparison,
} from "./data.js";

const Note=({children})=><div style={{marginTop:10,...GL,padding:"10px 14px",fontSize:10,color:C.md,fontFamily:C.dt,lineHeight:1.8}}>{children}</div>;
const Lbl=({children,color})=><span style={{fontSize:8,fontWeight:700,color:color||C.dm,fontFamily:C.dt,letterSpacing:1.5,textTransform:"uppercase"}}>{children}</span>;
const Val=({children,color,big})=><span style={{fontSize:big?20:13,fontWeight:700,color:color||C.w,fontFamily:C.dt}}>{children}</span>;
const Vu=({children})=><span style={{fontSize:7,color:C.dm,fontFamily:C.dt,marginLeft:2}}>{children}</span>;

// ── Parallel Coordinates ────────────────────────────────────────────
function ParCoords({pareto}){
  const W=820,H=340,pd={t:38,b:38,l:20,r:20};
  const axes=PN.slice(0,16),nA=axes.length,xS=(W-pd.l-pd.r)/(nA-1);
  const yT=pd.t,yB=H-pd.b,bG=Math.max(...pareto.map(p=>p.grip));
  return(
    <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:"100%"}}>
      <defs><linearGradient id="pcG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor={C.cy} stopOpacity=".5"/><stop offset="100%" stopColor={C.am} stopOpacity=".5"/></linearGradient><linearGradient id="pcB" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor={C.red}/><stop offset="100%" stopColor="#ff6040"/></linearGradient></defs>
      {axes.map((a,i)=>{const x=pd.l+i*xS;return<g key={a}><line x1={x} y1={yT} x2={x} y2={yB} stroke={C.b1}/><text x={x} y={yT-8} textAnchor="middle" fill={C.md} fontSize={7} fontFamily="Outfit" fontWeight={600} transform={`rotate(-40 ${x} ${yT-8})`}>{a}</text></g>;})}
      {pareto.map((s,si)=>{const best=s.grip>=bG-0.01;return<polyline key={si} points={axes.map((_,i)=>`${pd.l+i*xS},${yT+(1-(s.params[i]||0))*(yB-yT)}`).join(" ")} fill="none" stroke={best?"url(#pcB)":"url(#pcG)"} strokeWidth={best?2.5:1} strokeOpacity={best?1:0.14}/>;})}
    </svg>
  );
}

// ── Confidence Gauge — COMPLETELY REWRITTEN ─────────────────────────
function ConfGauge({score}){
  const ac=score>80?C.gn:score>60?C.am:C.red;
  // Simple horizontal bar gauge — no arc overlap issues
  return(
    <div style={{padding:"8px 0"}}>
      {/* Score */}
      <div style={{textAlign:"center",marginBottom:12}}>
        <span style={{fontSize:36,fontWeight:800,color:ac,fontFamily:"Outfit"}}>{score}</span>
        <span style={{fontSize:16,fontWeight:700,color:ac,fontFamily:"Outfit"}}>%</span>
      </div>
      {/* Bar track */}
      <div style={{position:"relative",height:14,background:C.b1,borderRadius:7,overflow:"hidden",margin:"0 4px"}}>
        {/* Gradient fill */}
        <div style={{position:"absolute",top:0,left:0,height:"100%",width:`${score}%`,borderRadius:7,background:`linear-gradient(90deg, ${C.red}, ${C.am} 50%, ${C.gn})`,transition:"width 0.5s ease",boxShadow:`0 0 12px ${ac}40`}}/>
        {/* Marker line */}
        <div style={{position:"absolute",top:-2,left:`${score}%`,width:2,height:18,background:C.w,borderRadius:1,transform:"translateX(-1px)",boxShadow:`0 0 6px ${ac}`}}/>
      </div>
      {/* Scale labels */}
      <div style={{display:"flex",justifyContent:"space-between",margin:"4px 4px 0",fontSize:8,fontFamily:C.dt,color:C.dm}}>
        <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
      </div>
      <div style={{textAlign:"center",marginTop:6,fontSize:8,fontFamily:C.dt,color:C.dm,letterSpacing:2}}>EPISTEMIC CONFIDENCE</div>
    </div>
  );
}

const GRIDS=[
  {key:"g1",label:"Optimizer Core",icon:"◆"},{key:"g2",label:"Kinematic & Aero",icon:"△"},
  {key:"g3",label:"Contact Patch",icon:"◉"},{key:"g4",label:"Energy Flow",icon:"⬡"},
  {key:"g5",label:"WMPC Horizon",icon:"◈"},{key:"g6",label:"Twin Fidelity",icon:"⬢"},
  {key:"g7",label:"Setup Delta",icon:"⟷"}, {key:"g8",label:"Param Sweep",icon:"〰"},
];
function SetupDeltaTab({ pareto }) {
  const [idxA, setIdxA] = React.useState(0);
  const [idxB, setIdxB] = React.useState(Math.min(pareto.length - 1, 5));
  const setupA = pareto[idxA];
  const setupB = pareto[idxB];
  if (!setupA || !setupB) return null;

  const selStyle = {
    background: C.bg, border: `1px solid ${C.b1}`,
    borderRadius: 4, padding: "4px 8px", fontSize: 10, fontFamily: C.dt, marginLeft: 8,
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 16, marginBottom: 14, alignItems: "center" }}>
        <div>
          <Lbl>SETUP A</Lbl>
          <select value={idxA} onChange={e => setIdxA(+e.target.value)} style={{ ...selStyle, color: C.cy }}>
            {pareto.map((p, i) => <option key={i} value={i}>#{i} (grip: {p.grip.toFixed(3)})</option>)}
          </select>
        </div>
        <span style={{ color: C.dm, fontSize: 16 }}>⟷</span>
        <div>
          <Lbl>SETUP B</Lbl>
          <select value={idxB} onChange={e => setIdxB(+e.target.value)} style={{ ...selStyle, color: C.am }}>
            {pareto.map((p, i) => <option key={i} value={i}>#{i} (grip: {p.grip.toFixed(3)})</option>)}
          </select>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Grip A" value={setupA.grip.toFixed(4)} sub={`gen ${setupA.gen}`} sentiment="neutral" delay={0} />
        <KPI label="Grip B" value={setupB.grip.toFixed(4)} sub={`gen ${setupB.gen}`} sentiment="neutral" delay={1} />
        <KPI label="ΔGrip" value={(setupB.grip - setupA.grip).toFixed(4)} sub={setupB.grip > setupA.grip ? "B is better" : "A is better"} sentiment={setupB.grip > setupA.grip ? "positive" : "negative"} delay={2} />
        <KPI label="ΔStability" value={(setupB.stability - setupA.stability).toFixed(2)} sub="rad/s" sentiment={Math.abs(setupB.stability - setupA.stability) < 0.5 ? "positive" : "amber"} delay={3} />
      </div>

      <GC style={{ padding: "12px 14px" }}>
        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.8, color: C.dm, fontFamily: C.dt, marginBottom: 10 }}>
          28-PARAMETER DELTA TABLE
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "140px 90px 90px 90px 70px", gap: 0, fontSize: 8, fontFamily: C.dt }}>
          {["Parameter", "Setup A", "Setup B", "Δ Absolute", "Δ %"].map(h => (
            <div key={h} style={{ color: C.dm, fontWeight: 700, letterSpacing: 1, padding: "6px 4px", borderBottom: `1px solid ${C.b1}` }}>{h}</div>
          ))}
          {PN.map((name, i) => {
            const vA = setupA.params?.[i] ?? 0;
            const vB = setupB.params?.[i] ?? 0;
            const delta = vB - vA;
            const pctDelta = vA !== 0 ? (delta / vA) * 100 : 0;
            const sig = Math.abs(pctDelta) > 10;
            return (
              <React.Fragment key={name}>
                <div style={{ color: sig ? C.cy : C.br, padding: "4px", borderBottom: `1px solid ${C.b1}08` }}>{name}</div>
                <div style={{ color: C.br, padding: "4px", borderBottom: `1px solid ${C.b1}08` }}>{vA.toFixed(4)}</div>
                <div style={{ color: C.br, padding: "4px", borderBottom: `1px solid ${C.b1}08` }}>{vB.toFixed(4)}</div>
                <div style={{ color: delta > 0 ? C.gn : delta < 0 ? C.red : C.dm, fontWeight: 600, padding: "4px", borderBottom: `1px solid ${C.b1}08` }}>
                  {delta > 0 ? "+" : ""}{delta.toFixed(4)}
                </div>
                <div style={{ color: sig ? C.am : C.dm, padding: "4px", borderBottom: `1px solid ${C.b1}08` }}>
                  {pctDelta.toFixed(1)}%
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </GC>
    </div>
  );
}
function ParamSweepTab({ pareto, sens }) {
  const ax = (props) => ({ tick: { fontSize: 9, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, ...props });
  const bestSetup = pareto.reduce((best, p) => p.grip > best.grip ? p : best, pareto[0]);
  const [sweepParam, setSweepParam] = React.useState(0);

  const sweepData = React.useMemo(() => {
    if (!bestSetup?.params) return [];
    const data = [];
    for (let i = 0; i < 50; i++) {
      const val = i / 49;
      const bestVal = bestSetup.params[sweepParam] || 0.5;
      const delta = val - bestVal;
      const sensVal = sens?.[sweepParam]?.dGrip || 0;
      const sensStab = sens?.[sweepParam]?.dStab || 0;
      const grip = bestSetup.grip + sensVal * delta * 8 - Math.abs(sensVal) * delta * delta * 40;
      const stability = (bestSetup.stability || 2.5) + sensStab * delta * 5 + 0.8 * delta * delta * 10;
      data.push({ val: +val.toFixed(3), grip: +grip.toFixed(4), stability: +stability.toFixed(3) });
    }
    return data;
  }, [bestSetup, sweepParam, sens]);

  const bestVal = bestSetup?.params?.[sweepParam] || 0.5;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
        <Lbl>SWEEP PARAMETER</Lbl>
        <select value={sweepParam} onChange={e => setSweepParam(+e.target.value)} style={{
          background: C.bg, color: C.cy, border: `1px solid ${C.b1}`,
          borderRadius: 4, padding: "6px 12px", fontSize: 11, fontFamily: C.dt,
        }}>
          {PN.map((name, i) => <option key={i} value={i}>{name} {PU[i] ? `[${PU[i]}]` : ""}</option>)}
        </select>
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: 9, color: C.dm, fontFamily: C.dt }}>
          Current best: <span style={{ color: C.gn, fontWeight: 700 }}>{bestVal.toFixed(4)}</span>
          {" · "}Sensitivity: <span style={{ color: (sens?.[sweepParam]?.dGrip || 0) > 0 ? C.gn : C.red, fontWeight: 700 }}>
            {(sens?.[sweepParam]?.dGrip || 0) > 0 ? "+" : ""}{(sens?.[sweepParam]?.dGrip || 0).toFixed(3)}
          </span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Sec title={`Grip Response — ${PN[sweepParam]}`}>
          <GC><ResponsiveContainer width="100%" height={280}>
            <AreaChart data={sweepData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="val" {...ax()} label={{ value: `${PN[sweepParam]} [normalized]`, position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...ax()} domain={["auto", "auto"]} label={{ value: "Grip [G]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine x={bestVal} stroke={C.gn} strokeDasharray="4 2" label={{ value: "OPTIMAL", fill: C.gn, fontSize: 7 }} />
              <Area dataKey="grip" stroke={C.cy} fill={`${C.cy}12`} strokeWidth={2} dot={false} name="Grip [G]" />
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>

        <Sec title={`Stability Response — ${PN[sweepParam]}`}>
          <GC><ResponsiveContainer width="100%" height={280}>
            <AreaChart data={sweepData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="val" {...ax()} label={{ value: `${PN[sweepParam]} [normalized]`, position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...ax()} domain={["auto", "auto"]} label={{ value: "Stability [rad/s]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine x={bestVal} stroke={C.gn} strokeDasharray="4 2" />
              <ReferenceLine y={5.0} stroke={C.red} strokeDasharray="4 2" label={{ value: "5.0 cap", fill: C.red, fontSize: 7 }} />
              <Area dataKey="stability" stroke={C.am} fill={`${C.am}12`} strokeWidth={2} dot={false} name="Stability [rad/s]" />
            </AreaChart>
          </ResponsiveContainer></GC>
        </Sec>
      </div>

      <Sec title="Grip vs Stability Trade-off (dual axis)">
        <GC><ResponsiveContainer width="100%" height={260}>
          <LineChart data={sweepData} margin={{ top: 8, right: 40, bottom: 24, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="val" {...ax()} label={{ value: `${PN[sweepParam]} [normalized]`, position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis yAxisId="g" {...ax()} label={{ value: "Grip [G]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
            <YAxis yAxisId="s" orientation="right" {...ax()} label={{ value: "Stab [rad/s]", angle: 90, position: "insideRight", fill: C.dm, fontSize: 9 }} />
            <Tooltip contentStyle={TT} />
            <ReferenceLine yAxisId="g" x={bestVal} stroke={C.gn} strokeDasharray="4 2" label={{ value: "BEST", fill: C.gn, fontSize: 8 }} />
            <Line yAxisId="g" dataKey="grip" stroke={C.cy} strokeWidth={2} dot={false} name="Grip" />
            <Line yAxisId="s" dataKey="stability" stroke={C.am} strokeWidth={2} dot={false} name="Stability" />
            <Legend wrapperStyle={{ fontSize: 9, fontFamily: C.hd }} />
          </LineChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Note>
        <strong style={{ color: C.cy }}>How to read:</strong> The sweep varies <strong style={{ color: C.cy }}>{PN[sweepParam]}</strong> from 0→1 (normalized bounds) while holding all other 27 parameters at their Pareto-optimal values. The green reference line marks the current best. Flat curves = insensitive parameter (safe to ignore). Steep curves with cliff edges = critical parameter (small change = large grip/stability shift). The quadratic response surface is a local approximation — actual landscape may have additional local optima.
      </Note>
    </div>
  );
}

export default function SetupModule({pareto,conv,sens,track}){
  const[grid,setGrid]=useState("g1");
  const trustRegion=useMemo(()=>gTrustRegion(),[]);
  const epistemic=useMemo(()=>gEpistemicConf(pareto),[pareto]);
  const freqResp=useMemo(()=>gFreqResponse(),[]);
  const yawLag=useMemo(()=>track?gYawPhaseLag(track):[],[track]);
  const rideHist=useMemo(()=>gRideHeightHist(),[]);
  const rcMig=useMemo(()=>gRollCenterMig(),[]);
  const lapDelta=useMemo(()=>track?gLapDelta(track):[],[track]);
  const loads=useMemo(()=>track?gLoadTransfer(track):[],[track]);
  const slipE=useMemo(()=>track?gSlipEnergy(track):[],[track]);
  const fricSat=useMemo(()=>track?gFrictionSatHist(track):[],[track]);
  const Ku=useMemo(()=>track?gUndersteerGrad(track):[],[track]);
  const hamiltonian=useMemo(()=>track?gHamiltonianEnergy(track):[],[track]);
  const dissipation=useMemo(()=>gDissipationBreakdown(),[]);
  const regen=useMemo(()=>gRegenEnvelope(),[]);
  const torqueVec=useMemo(()=>track?gTorqueVectoring(track):[],[track]);
  const horizon=useMemo(()=>gHorizonTraj(),[]);
  const wavelets=useMemo(()=>gWaveletCoeffs(),[]);
  const alSlack=useMemo(()=>track?gALSlack(track):[],[track]);
  const ctrlEffort=useMemo(()=>track?gControlEffort(track):[],[track]);
  const ekf=useMemo(()=>gEKFInnovation(),[]);
  const pacDrift=useMemo(()=>gPacejkaDrift(),[]);
  const psd=useMemo(()=>gPSDOverlay(),[]);
  const fidelity=useMemo(()=>gFidelitySpider(),[]);
  const damperHist=useMemo(()=>gDamperHist(),[]);
  const setupComp=useMemo(()=>gSetupComparison(),[]);

  const bG=Math.max(...pareto.map(p=>p.grip));
  const bS=Math.min(...pareto.filter(p=>p.grip>bG-0.05).map(p=>p.stability));
  const fidelityAvg=+(fidelity.reduce((a,f)=>a+f.score,0)/fidelity.length).toFixed(1);
  const ax=(props)=>({tick:{fontSize:9,fill:C.dm,fontFamily:C.dt},stroke:C.b1,...props});

  const prioColor=p=>p.priority==="HIGH"?C.red:p.priority==="MED"?C.am:C.dm;
  const deltaColor=p=>Math.abs(p.deltaPct)>15?C.red:Math.abs(p.deltaPct)>5?C.am:C.gn;

  return(<>
    <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:10,marginBottom:16}}>
      <KPI label="Peak Grip" value={`${bG.toFixed(3)}G`} sub="Pareto Max" sentiment="positive" delay={0}/>
      <KPI label="Stability" value={bS.toFixed(2)} sub="rad/s" sentiment="amber" delay={1}/>
      <KPI label="Pareto" value={pareto.length} sub="setups" delay={2}/>
      <KPI label="Confidence" value={`${epistemic.score}%`} sub="epistemic" sentiment={epistemic.score>80?"positive":"amber"} delay={3}/>
      <KPI label="Twin R²" value={`${fidelityAvg}%`} sub="fidelity" sentiment="positive" delay={4}/>
      <KPI label="Converged" value={conv[conv.length-1].bestGrip.toFixed(3)} sub={`${conv.length} iter`} sentiment="positive" delay={5}/>
    </div>

    <div style={{display:"flex",gap:5,marginBottom:16,flexWrap:"wrap"}}>
      {GRIDS.map(g=><Pill key={g.key} active={grid===g.key} label={`${g.icon} ${g.label}`} onClick={()=>setGrid(g.key)} color={C.red}/>)}
    </div>

    <FadeSlide keyVal={grid}>
      {/* ═══ GRID 1: OPTIMIZER CORE ═══════════════════════════════════ */}
      {grid==="g1"&&(<>
        <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:14}}>
          <Sec title="Pareto Front — Grip vs Stability"><GC><ResponsiveContainer width="100%" height={320}><ScatterChart margin={{top:10,right:20,bottom:26,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="grip" type="number" domain={["auto","auto"]} {...ax()} label={{value:"Lat Grip [G]",position:"bottom",offset:8,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}}/><YAxis dataKey="stability" type="number" {...ax()} label={{value:"Stab [rad/s]",angle:-90,position:"insideLeft",offset:10,style:{fontSize:9,fill:C.md,fontFamily:C.hd}}}/><Tooltip contentStyle={TT}/><ReferenceArea y1={0} y2={5} fill={C.gn} fillOpacity={.03}/><ReferenceLine y={5} stroke={C.red} strokeDasharray="6 3"/><Scatter data={pareto} fillOpacity={.85} r={4}>{pareto.map((e,i)=><Cell key={i} fill={e.grip>=bG-.01?C.gn:C.cy}/>)}</Scatter></ScatterChart></ResponsiveContainer></GC></Sec>
          <div>
            <Sec title="Epistemic Confidence">
              <GC style={{padding:"16px 14px"}}>
                <ConfGauge score={epistemic.score}/>
                <div style={{marginTop:10}}>
                  {epistemic.breakdown.map(b=>(
                    <div key={b.axis} style={{display:"flex",justifyContent:"space-between",alignItems:"center",fontSize:9,fontFamily:C.dt,marginBottom:4,padding:"0 4px"}}>
                      <span style={{color:C.md}}>{b.axis}</span>
                      <div style={{display:"flex",alignItems:"center",gap:6}}>
                        <div style={{width:50,height:4,background:C.b1,borderRadius:2,overflow:"hidden"}}><div style={{width:`${b.score}%`,height:"100%",background:b.score>80?C.gn:b.score>60?C.am:C.red,borderRadius:2}}/></div>
                        <span style={{color:b.score>80?C.gn:b.score>60?C.am:C.red,fontWeight:600,width:28,textAlign:"right"}}>{b.score}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </GC>
            </Sec>
          </div>
        </div>

        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Convergence History"><GC><ResponsiveContainer width="100%" height={220}><AreaChart data={conv} margin={{top:8,right:16,bottom:8,left:8}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="iter" {...ax()}/><YAxis {...ax()} domain={["auto","auto"]}/><Tooltip contentStyle={TT}/><Area type="monotone" dataKey="bestGrip" stroke={C.red} fill={`${C.red}0c`} strokeWidth={2} dot={false}/></AreaChart></ResponsiveContainer></GC></Sec>
          <Sec title="Trust Region — Fisher Damping"><GC><ResponsiveContainer width="100%" height={220}><ComposedChart data={trustRegion.filter((_,i)=>i%2===0)} margin={{top:8,right:16,bottom:8,left:8}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="iter" {...ax()}/><YAxis yAxisId="l" {...ax()}/><YAxis yAxisId="r" orientation="right" {...ax()}/><Tooltip contentStyle={TT}/><Line yAxisId="l" type="monotone" dataKey="fishNorm" stroke={C.am} strokeWidth={1.5} dot={false} name="‖F‖ Fisher"/><Area yAxisId="r" type="monotone" dataKey="dampFactor" stroke={C.pr} fill={`${C.pr}08`} strokeWidth={1} dot={false} name="Damp Factor"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></ComposedChart></ResponsiveContainer></GC></Sec>
        </div>

        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Sensitivity — ∂Grip / ∂Parameter"><GC><ResponsiveContainer width="100%" height={340}><BarChart data={sens} layout="vertical" margin={{top:8,right:16,bottom:8,left:55}}><CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false}/><XAxis type="number" {...ax()}/><YAxis dataKey="param" type="category" tick={{fontSize:9,fill:C.br,fontFamily:C.dt}} stroke={C.b1} width={50}/><Tooltip contentStyle={TT}/><ReferenceLine x={0} stroke={C.dm}/><Bar dataKey="dGrip" radius={[0,4,4,0]} barSize={11}>{sens.map((e,i)=><Cell key={i} fill={e.dGrip>0?C.gn:C.red} fillOpacity={.8}/>)}</Bar></BarChart></ResponsiveContainer></GC></Sec>
          <Sec title="Parallel Coordinates — 28D"><GC style={{padding:"12px"}}><div style={{width:"100%",height:340}}><ParCoords pareto={pareto}/></div></GC></Sec>
        </div>

        {/* ═══ MORL OPTIMIZATION PROCESS ═════════════════════════════ */}
        <Sec title="MORL-SB-TRPO Optimization Process">
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10,marginBottom:12}}>
            <GC><ResponsiveContainer width="100%" height={180}><LineChart data={conv} margin={{top:8,right:12,bottom:8,left:8}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="iter" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="kl" stroke={C.am} strokeWidth={1.5} dot={false} name="D_KL"/><ReferenceLine y={0.0094} stroke={C.red} strokeDasharray="4 2" label={{value:"δ=0.0094",fill:C.red,fontSize:7,fontFamily:C.dt}}/><Legend wrapperStyle={{fontSize:8,fontFamily:C.hd}}/></LineChart></ResponsiveContainer><div style={{fontSize:8,color:C.dm,fontFamily:C.dt,padding:"4px 8px"}}>KL divergence vs trust region bound δ</div></GC>
            <GC><ResponsiveContainer width="100%" height={180}><LineChart data={conv} margin={{top:8,right:12,bottom:8,left:8}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="iter" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="entropy" stroke={C.pr} strokeWidth={1.5} dot={false} name="Σlog(σ)"/><Legend wrapperStyle={{fontSize:8,fontFamily:C.hd}}/></LineChart></ResponsiveContainer><div style={{fontSize:8,color:C.dm,fontFamily:C.dt,padding:"4px 8px"}}>Policy entropy — collapse = premature convergence</div></GC>
            <GC style={{padding:"12px"}}>
              <Lbl color={C.cy}>CHEBYSHEV ENSEMBLE</Lbl>
              <div style={{marginTop:8}}>
                {Array.from({length:20},(_,i)=>{const w=0.5*(1-Math.cos(i*Math.PI/19));const isHigh=w>0.7;return(
                  <div key={i} style={{display:"flex",alignItems:"center",gap:6,marginBottom:2}}>
                    <span style={{fontSize:7,fontFamily:C.dt,color:C.dm,width:14}}>{i+1}</span>
                    <div style={{flex:1,height:5,background:C.b1,borderRadius:2,overflow:"hidden"}}>
                      <div style={{width:`${w*100}%`,height:"100%",background:isHigh?C.red:C.cy,borderRadius:2,opacity:0.6+w*0.4}}/>
                    </div>
                    <span style={{fontSize:7,fontFamily:C.dt,color:isHigh?C.red:C.dm,width:28}}>ω={w.toFixed(2)}</span>
                  </div>
                );})}
              </div>
              <div style={{fontSize:8,color:C.dm,fontFamily:C.dt,marginTop:6}}>Red = high-grip boundary (65% concentrated)</div>
            </GC>
          </div>
        </Sec>

        {/* ═══ 28-PARAMETER COMPARISON TABLE ════════════════════════ */}
        <Sec title="28-Parameter Setup — Current vs MORL Optimal">
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:14}}>
            <div style={{...GL,padding:"12px 14px",borderTop:`2px solid ${C.gn}`,textAlign:"center"}}><Lbl color={C.gn}>GRIP GAIN</Lbl><div style={{marginTop:6}}><Val big color={C.gn}>+{setupComp.totalGrip.toFixed(3)}</Val><Vu>G</Vu></div></div>
            <div style={{...GL,padding:"12px 14px",borderTop:`2px solid ${C.am}`,textAlign:"center"}}><Lbl color={C.am}>STAB SHIFT</Lbl><div style={{marginTop:6}}><Val big color={setupComp.totalStab>0?C.am:C.pr}>{setupComp.totalStab>0?"+":""}{setupComp.totalStab.toFixed(3)}</Val><Vu>rad/s</Vu></div></div>
            <div style={{...GL,padding:"12px 14px",borderTop:`2px solid ${C.red}`,textAlign:"center"}}><Lbl color={C.red}>HIGH PRIO</Lbl><div style={{marginTop:6}}><Val big color={C.red}>{setupComp.params.filter(p=>p.priority==="HIGH").length}</Val><Vu>changes</Vu></div></div>
            <div style={{...GL,padding:"12px 14px",borderTop:`2px solid ${C.cy}`,textAlign:"center"}}><Lbl color={C.cy}>TOTAL DIM</Lbl><div style={{marginTop:6}}><Val big color={C.cy}>28</Val><Vu>params</Vu></div></div>
          </div>

          {(()=>{const cats=[...new Set(setupComp.params.map(p=>p.category))];return cats.map(cat=>{const ps=setupComp.params.filter(p=>p.category===cat);const catGrip=ps.reduce((a,p)=>a+p.gripGain,0);return(
            <div key={cat} style={{...GL,marginBottom:8,overflow:"hidden"}}>
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"6px 12px",background:`${C.cy}05`,borderBottom:`1px solid ${C.b1}`}}>
                <div style={{display:"flex",alignItems:"center",gap:6}}><div style={{width:3,height:12,background:C.cy,borderRadius:2}}/><span style={{fontSize:10,fontWeight:700,color:C.w,fontFamily:C.hd,letterSpacing:1,textTransform:"uppercase"}}>{cat}</span><span style={{fontSize:7,color:C.dm,fontFamily:C.dt}}>({ps.length})</span></div>
                <span style={{fontSize:8,fontFamily:C.dt,color:C.gn,fontWeight:600}}>+{catGrip.toFixed(4)}G</span>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"110px 65px 50px 65px 55px 45px 1fr 60px 50px",gap:0,padding:"4px 12px 2px",borderBottom:`1px solid ${C.b1}15`}}>
                {["PARAM","CURRENT","UNIT","OPTIMAL","DELTA","Δ%","","GRIP+","PRIO"].map(h=><span key={h} style={{fontSize:6,fontWeight:700,color:C.dm,fontFamily:C.dt,letterSpacing:1}}>{h}</span>)}
              </div>
              {ps.map(p=>(
                <div key={p.name} style={{display:"grid",gridTemplateColumns:"110px 65px 50px 65px 55px 45px 1fr 60px 50px",gap:0,padding:"4px 12px",borderBottom:`1px solid ${C.b1}08`,alignItems:"center"}}>
                  <span style={{fontSize:9,fontFamily:C.dt,color:C.br,fontWeight:600}}>{p.name}</span>
                  <span style={{fontSize:9,fontFamily:C.dt,color:C.md}}>{p.current}</span>
                  <span style={{fontSize:7,fontFamily:C.dt,color:C.dm}}>{p.unit}</span>
                  <span style={{fontSize:9,fontFamily:C.dt,color:C.w,fontWeight:700}}>{p.optimal}</span>
                  <span style={{fontSize:9,fontFamily:C.dt,color:deltaColor(p),fontWeight:600}}>{p.delta>0?"+":""}{p.delta}</span>
                  <span style={{fontSize:8,fontFamily:C.dt,color:deltaColor(p)}}>{p.deltaPct>0?"+":""}{p.deltaPct}%</span>
                  <div style={{padding:"0 4px"}}><div style={{height:5,background:C.b1,borderRadius:2,overflow:"hidden",position:"relative"}}><div style={{position:"absolute",top:0,height:"100%",borderRadius:2,left:p.deltaPct>=0?"50%":`${50+p.deltaPct/2}%`,width:`${Math.min(50,Math.abs(p.deltaPct)/2)}%`,background:deltaColor(p)}}/><div style={{position:"absolute",left:"50%",top:0,width:1,height:"100%",background:C.dm}}/></div></div>
                  <span style={{fontSize:8,fontFamily:C.dt,color:C.gn,fontWeight:600}}>+{p.gripGain.toFixed(4)}</span>
                  <span style={{fontSize:7,fontFamily:C.dt,fontWeight:700,color:prioColor(p),background:`${prioColor(p)}10`,padding:"1px 5px",borderRadius:6,border:`1px solid ${prioColor(p)}20`,textAlign:"center"}}>{p.priority}</span>
                </div>
              ))}
            </div>
          );});})()}
          <Note><strong style={{color:C.cy}}>Reading:</strong> <strong style={{color:C.red}}>HIGH</strong> = &gt;15% delta, change first. <strong style={{color:C.am}}>MED</strong> = 5-15%, incremental. <strong style={{color:C.gn}}>LOW</strong> = &lt;5%, near-optimal. Change bar: center = zero, left = decrease, right = increase. Total grip gain assumes simultaneous changes — individual gains not perfectly additive due to 28D cross-coupling.</Note>
        </Sec>
      </>)}

      {/* ═══ GRID 2 ════════════════════════════════════════════════ */}
      {grid==="g2"&&(<>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Frequency Response (Bode)"><GC><ResponsiveContainer width="100%" height={280}><LineChart data={freqResp} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="freq" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3"/><ReferenceArea x1={1} x2={3} fill={C.gn} fillOpacity={.04}/><ReferenceArea x1={8} x2={15} fill={C.am} fillOpacity={.04}/><Line type="monotone" dataKey="front_dB" stroke={C.cy} strokeWidth={2} dot={false} name="Front"/><Line type="monotone" dataKey="rear_dB" stroke={C.am} strokeWidth={2} dot={false} name="Rear"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></LineChart></ResponsiveContainer></GC></Sec>
          <Sec title="Lap Delta"><GC><ResponsiveContainer width="100%" height={280}><ComposedChart data={lapDelta.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis yAxisId="l" {...ax()}/><YAxis yAxisId="r" orientation="right" {...ax()}/><Tooltip contentStyle={TT}/><ReferenceLine yAxisId="l" y={0} stroke={C.gn} strokeDasharray="3 3"/><Area yAxisId="l" type="monotone" dataKey="delta" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.8} dot={false} name="Δt"/><Line yAxisId="r" type="monotone" dataKey="vOptimal" stroke={C.gn} strokeWidth={1} dot={false} name="V_opt" strokeDasharray="4 2"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></ComposedChart></ResponsiveContainer></GC></Sec>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Yaw Phase Lag"><GC><ResponsiveContainer width="100%" height={250}><LineChart data={yawLag} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis {...ax()} domain={[0,50]}/><Tooltip contentStyle={TT}/><ReferenceArea y1={0} y2={25} fill={C.gn} fillOpacity={.04}/><Line type="monotone" dataKey="lagMs" stroke={C.pr} strokeWidth={1.5} dot={false}/></LineChart></ResponsiveContainer></GC></Sec>
          <Sec title="Ride Height Distribution"><GC><ResponsiveContainer width="100%" height={250}><BarChart data={rideHist.filter(d=>d.front>5||d.rear>5)} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="height" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Bar dataKey="front" fill={C.cy} fillOpacity={.5} name="Front" radius={[2,2,0,0]}/><Bar dataKey="rear" fill={C.am} fillOpacity={.5} name="Rear" radius={[2,2,0,0]}/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></BarChart></ResponsiveContainer></GC></Sec>
        </div>
        <Sec title="Roll Center Migration"><GC><ResponsiveContainer width="100%" height={240}><ScatterChart margin={{top:10,right:20,bottom:24,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="rcXf" type="number" {...ax()}/><YAxis dataKey="rcYf" type="number" {...ax()}/><Tooltip contentStyle={TT}/><Scatter data={rcMig} fill={C.cy} fillOpacity={.4} r={3} name="Front"/><Scatter data={rcMig.map(d=>({...d,rcXf:d.rcXr,rcYf:d.rcYr}))} fill={C.am} fillOpacity={.4} r={3} name="Rear"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></ScatterChart></ResponsiveContainer></GC></Sec>
      </>)}

      {/* ═══ GRID 3 ════════════════════════════════════════════════ */}
      {grid==="g3"&&(<>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Dynamic Fz — 4-Corner"><GC><ResponsiveContainer width="100%" height={280}><AreaChart data={loads.filter((_,i)=>i%3===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Area type="monotone" dataKey="Fz_fl" stroke={C.cy} fill={`${C.cy}06`} strokeWidth={1.2} dot={false} name="FL"/><Area type="monotone" dataKey="Fz_fr" stroke={C.gn} fill={`${C.gn}06`} strokeWidth={1.2} dot={false} name="FR"/><Area type="monotone" dataKey="Fz_rl" stroke={C.am} fill={`${C.am}06`} strokeWidth={1.2} dot={false} name="RL"/><Area type="monotone" dataKey="Fz_rr" stroke={C.red} fill={`${C.red}06`} strokeWidth={1.2} dot={false} name="RR"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></AreaChart></ResponsiveContainer></GC></Sec>
          <Sec title="Friction Saturation"><GC><ResponsiveContainer width="100%" height={280}><BarChart data={fricSat} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="bin" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Bar dataKey="count" radius={[2,2,0,0]} barSize={14}>{fricSat.map((e,i)=><Cell key={i} fill={e.bin>=.85?C.red:e.bin>=.6?C.am:C.cy} fillOpacity={.7}/>)}</Bar></BarChart></ResponsiveContainer></GC></Sec>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Slip Energy [J]"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={slipE.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Area type="monotone" dataKey="energy" stroke={C.red} fill={`${C.red}10`} strokeWidth={1.5} dot={false}/></AreaChart></ResponsiveContainer></GC></Sec>
          <Sec title="Understeer Gradient Ku"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={Ku} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis domain={[-5,5]} {...ax()}/><Tooltip contentStyle={TT}/><ReferenceArea y1={-1} y2={1} fill={C.gn} fillOpacity={.04}/><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3"/><Area type="monotone" dataKey="Ku" stroke={C.pr} fill={`${C.pr}08`} strokeWidth={1.5} dot={false}/></AreaChart></ResponsiveContainer></GC></Sec>
        </div>
      </>)}

      {/* ═══ GRID 4 ════════════════════════════════════════════════ */}
      {grid==="g4"&&(<>
        <Sec title="Hamiltonian H(q,p) [kJ]"><GC><ResponsiveContainer width="100%" height={300}><AreaChart data={hamiltonian} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Area type="monotone" dataKey="kinetic" stackId="1" stroke={C.cy} fill={`${C.cy}15`} strokeWidth={1.2} name="T (kinetic)"/><Area type="monotone" dataKey="potential" stackId="1" stroke={C.gn} fill={`${C.gn}15`} strokeWidth={1.2} name="V (potential)"/><Line type="monotone" dataKey="dissipated" stroke={C.red} strokeWidth={1.5} dot={false} name="Σ Dissipated" strokeDasharray="4 2"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></AreaChart></ResponsiveContainer></GC></Sec>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Dissipation Breakdown"><GC style={{padding:"16px"}}>{dissipation.map(d=>{const mx=Math.max(...dissipation.map(x=>x.energy));return<div key={d.component} style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}><div style={{width:100,fontSize:9,color:C.md,fontFamily:C.dt,textAlign:"right"}}>{d.component}</div><div style={{flex:1,height:14,background:C.b1,borderRadius:4,overflow:"hidden"}}><div style={{width:`${(d.energy/mx)*100}%`,height:"100%",background:d.color,borderRadius:4}}/></div><div style={{width:50,fontSize:10,color:C.br,fontFamily:C.dt,fontWeight:600}}>{d.energy} J</div></div>;})}</GC></Sec>
          <Sec title="Regen Envelope"><GC><ResponsiveContainer width="100%" height={260}><AreaChart data={regen} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="speed" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Area type="monotone" dataKey="motorMax" stroke={C.cy} fill={`${C.cy}06`} strokeWidth={1} dot={false} name="Motor" strokeDasharray="4 2"/><Area type="monotone" dataKey="batteryLimit" stroke={C.am} fill={`${C.am}06`} strokeWidth={1} dot={false} name="Battery" strokeDasharray="4 2"/><Line type="monotone" dataKey="actual" stroke={C.gn} strokeWidth={2} dot={false} name="Actual"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></AreaChart></ResponsiveContainer></GC></Sec>
        </div>
        <Sec title="Torque Vectoring"><GC><ResponsiveContainer width="100%" height={220}><BarChart data={torqueVec.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Bar dataKey="mechYaw" stackId="a" fill={C.cy} fillOpacity={.5} name="Mechanical"/><Bar dataKey="activeYaw" stackId="a" fill={C.gn} fillOpacity={.6} name="Active Diff"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></BarChart></ResponsiveContainer></GC></Sec>
      </>)}

      {/* ═══ GRID 5 ════════════════════════════════════════════════ */}
      {grid==="g5"&&(<>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Receding Horizon — 64-step"><GC><ResponsiveContainer width="100%" height={300}><ComposedChart data={horizon} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="xPred" type="number" {...ax()}/><YAxis dataKey="yPred" type="number" {...ax()}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="yBoundL" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3"/><Line type="monotone" dataKey="yBoundR" stroke={C.dm} strokeWidth={1} dot={false} strokeDasharray="3 3"/><Line type="monotone" dataKey="yPred" stroke={C.cy} strokeWidth={2} dot={false}/></ComposedChart></ResponsiveContainer></GC></Sec>
          <Sec title="Wavelet Sparsity"><GC style={{padding:"14px"}}>{["cA3","cD3","cD2","cD1"].map(lv=>{const co=wavelets.filter(w=>w.level===lv);return<div key={lv} style={{marginBottom:10}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}><span style={{fontSize:10,fontFamily:C.dt,color:lv==="cA3"?C.cy:C.am,fontWeight:700}}>{lv}</span><span style={{fontSize:9,fontFamily:C.dt,color:C.dm}}>{co.filter(w=>w.active).length}/{co.length}</span></div><div style={{display:"flex",gap:2}}>{co.map((w,j)=><div key={j} style={{flex:1,height:18,borderRadius:2,background:w.active?(lv==="cA3"?C.cy:C.am):C.b1,opacity:.3+w.mag*.7}}/>)}</div></div>;})}<Note>Sparsity: {((1-wavelets.filter(w=>w.active).length/wavelets.length)*100).toFixed(0)}%</Note></GC></Sec>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="AL Constraint Slack"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={alSlack} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><ReferenceArea y1={0} y2={.1} fill={C.red} fillOpacity={.05}/><Area type="monotone" dataKey="slackGrip" stroke={C.red} fill={`${C.red}08`} strokeWidth={1.5} dot={false} name="Grip"/><Area type="monotone" dataKey="slackSteer" stroke={C.am} fill={`${C.am}06`} strokeWidth={1.2} dot={false} name="Steer"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></AreaChart></ResponsiveContainer></GC></Sec>
          <Sec title="Control Effort"><GC><ResponsiveContainer width="100%" height={250}><AreaChart data={ctrlEffort} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="s" {...ax()}/><YAxis domain={[0,1]} {...ax()}/><Tooltip contentStyle={TT}/><ReferenceArea y1={.9} y2={1} fill={C.red} fillOpacity={.05}/><Line type="monotone" dataKey="steerUtil" stroke={C.pr} strokeWidth={1.5} dot={false} name="Steer"/><Line type="monotone" dataKey="brakeUtil" stroke={C.red} strokeWidth={1.5} dot={false} name="Brake"/><Line type="monotone" dataKey="throttleUtil" stroke={C.gn} strokeWidth={1.5} dot={false} name="Throttle"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></AreaChart></ResponsiveContainer></GC></Sec>
        </div>
      </>)}

      {/* ═══ GRID 6 ════════════════════════════════════════════════ */}
      {grid==="g6"&&(<>
        <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:14}}>
          <Sec title="EKF Innovation (ỹ = z − Hx)"><GC><ResponsiveContainer width="100%" height={280}><LineChart data={ekf.filter((_,i)=>i%2===0)} margin={{top:10,right:20,bottom:10,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="t" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><ReferenceLine y={0} stroke={C.gn} strokeDasharray="3 3"/><ReferenceArea y1={-.02} y2={.02} fill={C.gn} fillOpacity={.04}/><Line type="monotone" dataKey="innov_ax" stroke={C.cy} strokeWidth={1} dot={false} name="a_x"/><Line type="monotone" dataKey="innov_wz" stroke={C.am} strokeWidth={1} dot={false} name="ω_z"/><Line type="monotone" dataKey="innov_vy" stroke={C.pr} strokeWidth={1} dot={false} name="v_y"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></LineChart></ResponsiveContainer></GC></Sec>
          <Sec title="Fidelity Spider"><GC style={{padding:"14px"}}><ResponsiveContainer width="100%" height={240}><RadarChart data={fidelity} cx="50%" cy="50%" outerRadius="72%"><PolarGrid stroke={C.b1}/><PolarAngleAxis dataKey="axis" tick={{fontSize:8,fill:C.md,fontFamily:C.dt}}/><PolarRadiusAxis angle={90} domain={[70,100]} tick={{fontSize:8,fill:C.dm}}/><Radar dataKey="score" stroke={C.cy} fill={C.cy} fillOpacity={0.15} strokeWidth={2}/></RadarChart></ResponsiveContainer><div style={{textAlign:"center",marginTop:4}}><span style={{fontSize:22,fontWeight:800,color:C.gn,fontFamily:C.hd}}>{fidelityAvg}%</span><span style={{fontSize:10,color:C.dm,fontFamily:C.dt,marginLeft:8}}>aggregate</span></div></GC></Sec>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Sec title="Pacejka Drift"><GC><ResponsiveContainer width="100%" height={250}><ComposedChart data={pacDrift} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="lap" {...ax()}/><YAxis yAxisId="l" {...ax()} domain={["auto","auto"]}/><YAxis yAxisId="r" orientation="right" {...ax()}/><Tooltip contentStyle={TT}/><Line yAxisId="l" type="monotone" dataKey="muY_pct" stroke={C.am} strokeWidth={2} dot={{r:3,fill:C.am}} name="µ_y%"/><Bar yAxisId="r" dataKey="stiffness" fill={C.cy} fillOpacity={.25} name="C_α"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></ComposedChart></ResponsiveContainer></GC></Sec>
          <Sec title="Real vs Sim PSD"><GC><ResponsiveContainer width="100%" height={250}><LineChart data={psd} margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid strokeDasharray="3 3" stroke={GS}/><XAxis dataKey="freq" {...ax()}/><YAxis {...ax()}/><Tooltip contentStyle={TT}/><Line type="monotone" dataKey="real" stroke={C.am} strokeWidth={1.5} dot={false} name="Real"/><Line type="monotone" dataKey="sim" stroke={C.cy} strokeWidth={1.5} dot={false} name="Sim" strokeDasharray="4 2"/><Legend wrapperStyle={{fontSize:9,fontFamily:C.hd}}/></LineChart></ResponsiveContainer></GC></Sec>
        </div>
      </>)}
      {grid==="g7"&&(<SetupDeltaTab pareto={pareto}/>)}
      {grid==="g8"&&(<ParamSweepTab pareto={pareto} sens={sens}/>)}
    </FadeSlide>
  </>);
}