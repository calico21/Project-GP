// src/data.js — Project-GP Synthetic Data (all 6 grids)

const sd = (s) => { let h = 0; for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0; return () => { h ^= h << 13; h ^= h >> 17; h ^= h << 5; return (h >>> 0) / 4294967296; }; };
const RN = sd("projectgp2026ter26fsg");
const R2 = sd("grid2kinematic"); const R3 = sd("grid3contact"); const R4 = sd("grid4energy"); const R5 = sd("grid5wmpc"); const R6 = sd("grid6ekf");

export const PN = ["k_f","k_r","arb_f","arb_r","c_low_f","c_low_r","c_high_f","c_high_r","camber_f","camber_r","toe_f","toe_r","castor","anti_dive_f","anti_squat","diff_lock","heave_k_f","heave_k_r","bumpstop_f","bumpstop_r","h_rc_f","h_rc_r","h_cg","brake_bias","tire_p_f","tire_p_r","ackermann","steer_ratio"];
export const PU = ["N/m","N/m","N/m","N/m","Ns/m","Ns/m","Ns/m","Ns/m","deg","deg","deg","deg","deg","","","","N/m","N/m","mm","mm","mm","mm","mm","","kPa","kPa","",""];

// ── Grid 1: Optimizer Core ──────────────────────────────────────────
export function gP(n=52){const p=[];for(let i=0;i<n;i++){const w=.5*(1-Math.cos(i*Math.PI/(n-1)));p.push({grip:+(1.25+.3*w+.04*(RN()-.5)).toFixed(4),stability:+(.5+2.8*(1-w)+.15*(RN()-.5)).toFixed(2),gen:Math.floor(RN()*200),omega:+w.toFixed(3),params:PN.map(()=>RN())});}return p.sort((a,b)=>a.grip-b.grip);}
export function gCV(n=200){const d=[];let b=1.1;for(let i=0;i<n;i++){b=Math.max(b,1.1+.45*(1-Math.exp(-i/40))+.02*RN());d.push({iter:i,bestGrip:+b.toFixed(4),kl:+(.012*Math.exp(-i/80)+.001*RN()).toFixed(5),entropy:+(-1+.3*Math.exp(-i/60)+.05*RN()).toFixed(3)});}return d;}
export function gSN(){return PN.slice(0,14).map(p=>({param:p,dGrip:+((RN()-.3)*.6).toFixed(3),dStab:+((RN()-.4)*.4).toFixed(3)}));}

export function gTrustRegion(n=200){const d=[];for(let i=0;i<n;i++){const fishNorm=2.5*Math.exp(-i/60)+.3+.15*RN();const stepSize=Math.min(.02,0.005/fishNorm+.001*RN());const damp=Math.min(1,fishNorm/3);d.push({iter:i,fishNorm:+fishNorm.toFixed(3),stepSize:+(stepSize*1000).toFixed(2),dampFactor:+damp.toFixed(3)});}return d;}

export function gEpistemicConf(pareto){
  const vars=pareto.map(p=>p.params.reduce((a,v)=>a+Math.pow(v-.5,2),0)/28);
  const meanVar=vars.reduce((a,v)=>a+v,0)/vars.length;
  const conf=Math.max(0,Math.min(100,100*(1-meanVar*4)));
  const breakdown=[{axis:"Spring Rates",score:+(conf+3*RN()).toFixed(0)},{axis:"Damper Maps",score:+(conf-5+4*RN()).toFixed(0)},{axis:"ARB Stiffness",score:+(conf+2*RN()).toFixed(0)},{axis:"Geometry",score:+(conf-8+5*RN()).toFixed(0)},{axis:"Aero Platform",score:+(conf-3+3*RN()).toFixed(0)},{axis:"Thermal Model",score:+(conf-10+6*RN()).toFixed(0)}];
  return{score:+conf.toFixed(1),breakdown};
}

// ── Grid 2: Kinematic & Aero Platform ───────────────────────────────
export function gTK(n=360){const v=[];let cx=0,cy=0,psi=0;for(let i=0;i<n;i++){const si=i*.5,ki=.08*Math.sin(si/15)+.04*Math.sin(si/7)+.02*Math.sin(si/3.5);psi+=ki*.5;cx+=Math.cos(psi)*.5;cy+=Math.sin(psi)*.5;const vi=12+8/(1+3*Math.abs(ki))+1.5*RN();v.push({s:+si.toFixed(1),x:+cx.toFixed(2),y:+cy.toFixed(2),speed:+vi.toFixed(1),lat_g:+(vi*vi*ki/9.81).toFixed(3),lon_g:+(i>0?(vi-(v[i-1]?.speed||vi))/(.5/vi)/9.81:0).toFixed(3),curvature:+ki.toFixed(5)});}return v;}
export function gTT(n=200){const d=[];let a=25,b=25,c=25,e=25;for(let i=0;i<n;i++){const h=.4+.3*Math.sin(i/20)+.1*RN(),cl=.12;a=Math.min(128,Math.max(20,a+h*(.9+.2*RN())-cl*(a-25)/60));b=Math.min(128,Math.max(20,b+h*(1+.15*RN())-cl*(b-25)/60));c=Math.min(128,Math.max(20,c+h*(.75+.2*RN())-cl*(c-25)/60));e=Math.min(128,Math.max(20,e+h*(.8+.15*RN())-cl*(e-25)/60));d.push({s:i,tfl:+a.toFixed(1),tfr:+b.toFixed(1),trl:+c.toFixed(1),trr:+e.toFixed(1)});}return d;}
export function gSU(){const d=[];for(let i=0;i<150;i++){const t=i*.02;d.push({t:+t.toFixed(3),steer:+(12*Math.sin(t*2.5)).toFixed(2),roll:+(12*Math.sin(t*2.5)*.15+.3*RN()).toFixed(2),heave_f:+(-12.8+2.5*Math.sin(t*3)+.5*RN()).toFixed(2),heave_r:+(-14.2+2*Math.sin(t*3+.5)+.4*RN()).toFixed(2)});}return d;}

export function gFreqResponse(){const d=[];for(let f=.5;f<=25;f+=.5){const wnf=2*Math.PI*1.8,wnr=2*Math.PI*2.1,z=.35,w=2*Math.PI*f;d.push({freq:f,front_dB:+(20*Math.log10(1/Math.sqrt(Math.pow(1-Math.pow(w/wnf,2),2)+Math.pow(2*z*w/wnf,2)))).toFixed(1),rear_dB:+(20*Math.log10(1/Math.sqrt(Math.pow(1-Math.pow(w/wnr,2),2)+Math.pow(2*z*w/wnr,2)))).toFixed(1)});}return d;}

export function gYawPhaseLag(track){return track.filter((_,i)=>i%3===0).map((p,i)=>{const steerRate=Math.abs(12*2.5*Math.cos(i*.03*2.5));const yawRate=Math.abs(Number(p.lat_g)||0)*2.5;const lagMs=18+12*Math.exp(-steerRate/8)+5*R2();return{s:p.s,steerRate:+steerRate.toFixed(1),yawRate:+yawRate.toFixed(2),lagMs:+lagMs.toFixed(1)};});}

export function gRideHeightHist(){const d=[];for(let h=-25;h<=0;h+=.5){const hf=h+12.8,hr=h+14.2;const cf=Math.round(500*Math.exp(-hf*hf/8)*(1+.2*R2()));const cr=Math.round(400*Math.exp(-hr*hr/10)*(1+.2*R2()));d.push({height:+h.toFixed(1),front:Math.max(0,cf),rear:Math.max(0,cr)});}return d;}

export function gRollCenterMig(){const d=[];for(let i=0;i<80;i++){const lg=2*(i/80-.5)+.3*R2();const rcXf=+(40+lg*8+2*R2()).toFixed(1);const rcYf=+(lg*15+3*R2()).toFixed(1);const rcXr=+(60+lg*5+2*R2()).toFixed(1);const rcYr=+(lg*10+2*R2()).toFixed(1);d.push({lat_g:+lg.toFixed(2),rcXf,rcYf,rcXr,rcYr});}return d;}

export function gLapDelta(track){let cum=0;return track.map(p=>{const opt=12+10/(1+3*Math.abs(Number(p.curvature)||0));const act=Number(p.speed)||0;cum+=(0.5/Math.max(act,1)-0.5/Math.max(opt,1));return{s:p.s,delta:+cum.toFixed(4),vActual:act,vOptimal:+opt.toFixed(1)};});}

// ── Grid 3: Contact Patch & Energy ──────────────────────────────────
export function gLoadTransfer(track){return track.map(p=>{const m=300,hcg=.33,lf=.8525,lr=.6975,L=lf+lr,tw=1.2;const lg=Number(p.lat_g)||0,lon=Number(p.lon_g)||0;const dLat=m*9.81*Math.abs(lg)*hcg/tw,dLon=m*9.81*lon*hcg/L;const sf=m*9.81*lr/L/2,sr=m*9.81*lf/L/2;return{s:p.s,Fz_fl:+(sf-dLon/2+(lg>0?-dLat/2:dLat/2)).toFixed(0),Fz_fr:+(sf-dLon/2+(lg>0?dLat/2:-dLat/2)).toFixed(0),Fz_rl:+(sr+dLon/2+(lg>0?-dLat/2:dLat/2)).toFixed(0),Fz_rr:+(sr+dLon/2+(lg>0?dLat/2:-dLat/2)).toFixed(0)};});}

export function gSlipEnergy(track){return track.map(p=>{const lg=Math.abs(Number(p.lat_g)||0),lon=Math.abs(Number(p.lon_g)||0),v=Number(p.speed)||1;const slipE=1.35*300*9.81*Math.sqrt(lg*lg+lon*lon)*v*.001*(.08+.03*R3());return{s:p.s,x:p.x,y:p.y,energy:+slipE.toFixed(1),speed:p.speed};});}

export function gFrictionSatHist(track){const bins=[];for(let b=0;b<=1;b+=.05){bins.push({bin:+b.toFixed(2),count:0});}track.forEach(p=>{const lg=Math.abs(Number(p.lat_g)||0),lon=Math.abs(Number(p.lon_g)||0);const util=Math.min(1,Math.sqrt(lg*lg+lon*lon)/1.35);const idx=Math.min(bins.length-1,Math.floor(util*20));bins[idx].count++;});return bins;}

export function gUndersteerGrad(track){return track.filter((_,i)=>i%4===0).map(p=>{const lg=Number(p.lat_g)||0;const v=Number(p.speed)||1;const Ku=v>5?Math.abs(lg)>0.1?(Number(p.curvature)||0)*v*v/9.81/(lg+1e-6)-1:0:0;return{s:p.s,Ku:+Math.max(-5,Math.min(5,Ku*100)).toFixed(2),balance:lg>0?"oversteer":"understeer"};});}

// ── Grid 4: Port-Hamiltonian Energy ─────────────────────────────────
export function gHamiltonianEnergy(track){let cumDiss=0;return track.filter((_,i)=>i%2===0).map(p=>{const v=Number(p.speed)||0,m=300;const kinetic=.5*m*v*v;const lg=Math.abs(Number(p.lat_g)||0);const potential=m*9.81*.33+.5*35000*Math.pow(lg*.015,2);cumDiss+=v*.02*(.5+lg*.8)+2*R4();return{s:p.s,kinetic:+(kinetic/1000).toFixed(1),potential:+(potential/1000).toFixed(2),total:+((kinetic+potential)/1000).toFixed(1),dissipated:+(cumDiss/1000).toFixed(2)};});}

export function gDissipationBreakdown(){return[{component:"Tire Slip (lat)",energy:4850,color:"#e10600"},{component:"Tire Slip (lon)",energy:2340,color:"#ff6040"},{component:"Damper Fluid",energy:1890,color:"#ffab00"},{component:"Aero Drag",energy:3200,color:"#00d2ff"},{component:"Rolling Resistance",energy:680,color:"#b388ff"},{component:"Drivetrain Friction",energy:420,color:"#00e676"}];}

export function gRegenEnvelope(){const d=[];for(let v=0;v<=30;v+=1){const motorMax=Math.min(25,v*1.2);const batteryLimit=18+2*Math.sin(v/10);const actual=Math.min(motorMax,batteryLimit)*(0.85+0.1*R4());d.push({speed:v,motorMax:+motorMax.toFixed(1),batteryLimit:+batteryLimit.toFixed(1),actual:+actual.toFixed(1)});}return d;}

export function gTorqueVectoring(track){return track.filter((_,i)=>i%3===0).map(p=>{const lg=Number(p.lat_g)||0;const mechYaw=lg*450;const activeYaw=lg*120*(1+.2*R4());return{s:p.s,mechYaw:+mechYaw.toFixed(0),activeYaw:+activeYaw.toFixed(0),total:+(mechYaw+activeYaw).toFixed(0)};});}

// ── Grid 5: Diff-WMPC Horizon ───────────────────────────────────────
export function gHorizonTraj(){const d=[];let x=0,y=0,psi=0;for(let i=0;i<64;i++){const ki=.04*Math.sin(i/8)+.02*Math.sin(i/3);psi+=ki*.8;x+=Math.cos(psi)*.8;y+=Math.sin(psi)*.8;const wL=y+1.75+.3*R5(),wR=y-1.75-.3*R5();d.push({step:i,xPred:+x.toFixed(2),yPred:+y.toFixed(2),yBoundL:+wL.toFixed(2),yBoundR:+wR.toFixed(2),sigma:+(.1+.05*i/64+.03*R5()).toFixed(3)});}return d;}

export function gWaveletCoeffs(){const levels=["cA3","cD3","cD2","cD1"];const d=[];levels.forEach((lv,li)=>{const n=li===0?8:8*Math.pow(2,li);for(let i=0;i<Math.min(n,16);i++){const mag=li===0?.8+.4*R5():(1/(li+1))*R5();d.push({level:lv,idx:i,mag:+mag.toFixed(3),active:mag>.15});}});return d;}

export function gALSlack(track){return track.filter((_,i)=>i%4===0).map(p=>{const lg=Math.abs(Number(p.lat_g)||0);const slackGrip=Math.max(0,1.35-Math.sqrt(lg*lg+Math.pow(Number(p.lon_g)||0,2)));const slackSteer=Math.max(0,.45-Math.abs(Number(p.curvature)||0)*10);return{s:p.s,slackGrip:+slackGrip.toFixed(3),slackSteer:+slackSteer.toFixed(3),penalty:+(slackGrip<.1?500*Math.pow(0.1-slackGrip,2):0).toFixed(1)};});}

export function gControlEffort(track){return track.filter((_,i)=>i%3===0).map(p=>{const lg=Math.abs(Number(p.lat_g)||0);const lon=Math.abs(Number(p.lon_g)||0);return{s:p.s,steerUtil:+Math.min(1,lg/.8+.1*R5()).toFixed(3),brakeUtil:+Math.min(1,lon/.6+.08*R5()).toFixed(3),throttleUtil:+Math.min(1,(Number(p.speed)||0)/25+.05*R5()).toFixed(3)};});}

// ── Grid 6: Twin Fidelity & EKF ─────────────────────────────────────
export function gEKFInnovation(n=300){const d=[];for(let i=0;i<n;i++){const t=i*.01;d.push({t:+t.toFixed(2),innov_ax:+(.02*Math.sin(t*15)+.01*R6()-.005).toFixed(4),innov_wz:+(.015*Math.sin(t*12)+.008*R6()-.004).toFixed(4),innov_vy:+(.03*Math.sin(t*8)+.015*R6()-.007).toFixed(4)});}return d;}

export function gPacejkaDrift(nLaps=20){const d=[];let muY=1.35,stiff=45000;for(let i=1;i<=nLaps;i++){muY-=.003+.002*R6();stiff-=80+60*R6();d.push({lap:i,muY:+muY.toFixed(4),stiffness:+stiff.toFixed(0),muY_pct:+((muY/1.35)*100).toFixed(1)});}return d;}

export function gPSDOverlay(){const d=[];for(let f=.5;f<=30;f+=.5){const base=-20+15*Math.exp(-Math.pow(f-2,2)/3)+8*Math.exp(-Math.pow(f-12,2)/8);const real=base+2*R6()-1;const sim=base+1.5*R6()-.75;d.push({freq:f,real:+real.toFixed(1),sim:+sim.toFixed(1)});}return d;}

export function gFidelitySpider(){return[{axis:"Kinematic R²",score:94,fullMark:100},{axis:"Dynamic NRMSE",score:88,fullMark:100},{axis:"Thermal ΔT",score:91,fullMark:100},{axis:"Slip Angle",score:86,fullMark:100},{axis:"Load Transfer",score:93,fullMark:100},{axis:"Freq. Response",score:89,fullMark:100}];}

export function gDamperHist(){const bins=[];for(let v=-.3;v<=.3;v+=.025){const x=v/.12;bins.push({vel:+v.toFixed(3),front:Math.round(800*Math.exp(-x*x/2)*(1+.3*RN())),rear:Math.round(600*Math.exp(-x*x/1.8)*(1+.25*RN()))});}return bins;}

export function gGripUtil(track){return track.filter((_,i)=>i%4===0).map(p=>{const lg=Math.abs(Number(p.lat_g)||0),lon=Math.abs(Number(p.lon_g)||0);const mu=1.35;const combined=Math.sqrt(lg*lg+lon*lon);return{s:p.s,utilisation:+Math.min(combined/mu,1.0).toFixed(3),combined:+combined.toFixed(3)};});}

export function gDriverInputs(track){
  const R7=sd("driverinputs2026");
  return track.filter((_,i)=>i%2===0).map((p,i)=>{
    const k=Number(p.curvature)||0,v=Number(p.speed)||1,vOpt=12+10/(1+3*Math.abs(k));
    const optSteer=k*1.55*14*180/Math.PI;
    const actSteer=optSteer*(1+0.12*(R7()-.5))+1.5*R7();
    const optThr=v<vOpt?Math.min(1,(vOpt-v)/8+0.15):0;
    const actThr=Math.max(0,optThr*(0.85+0.3*R7())+0.05*R7());
    const optBrk=v>vOpt+1?Math.min(1,(v-vOpt)/6):0;
    const actBrk=Math.max(0,optBrk*(0.9+0.2*R7())+0.03*R7());
    return{
      s:p.s,
      steerAct:+actSteer.toFixed(1),steerOpt:+optSteer.toFixed(1),
      thrAct:+(actThr*100).toFixed(0),thrOpt:+(optThr*100).toFixed(0),
      brkAct:+(actBrk*100).toFixed(0),brkOpt:+(optBrk*100).toFixed(0),
    };
  });
}

export function gSetupComparison() {
  const R8 = sd("setupcomp2026");
  const cats = {
    "Springs": [0,1], "ARB": [2,3], "Damper Low-Spd": [4,5], "Damper High-Spd": [6,7],
    "Camber": [8,9], "Toe": [10,11], "Geometry": [12,13,14],
    "Differential": [15], "Heave Springs": [16,17], "Bumpstop": [18,19],
    "Roll Center": [20,21], "CG & Bias": [22,23], "Tire Pressure": [24,25],
    "Steering": [26,27]
  };
  const ranges = [
    [25000,55000],[25000,55000],[800,4000],[800,4000],[1200,4500],[1200,4500],[600,2200],[600,2200],
    [-3.5,-0.5],[-2.5,0],[-0.3,0.5],[-0.3,0.5],[3,8],[0.1,0.6],[0.1,0.6],
    [0.2,0.95],[15000,40000],[15000,40000],[2,12],[2,12],
    [20,80],[30,90],[280,350],[0.45,0.65],[75,110],[75,110],[0.4,1.0],[3,6]
  ];
  const current = ranges.map(([lo,hi]) => lo + (hi - lo) * (0.4 + 0.2 * R8()));
  const optimal = ranges.map(([lo,hi],i) => {
    const c = current[i];
    const shift = (hi - lo) * (R8() - 0.4) * 0.3;
    return Math.max(lo, Math.min(hi, c + shift));
  });

  const params = PN.map((name, i) => {
    const cur = current[i], opt = optimal[i], rng = ranges[i][1] - ranges[i][0];
    const delta = opt - cur;
    const deltaPct = (delta / rng) * 100;
    const gripGain = Math.abs(deltaPct) * 0.0008 * (1 + R8() * 0.5);
    const stabGain = Math.abs(deltaPct) * 0.0003 * (R8() > 0.5 ? 1 : -1);
    const cat = Object.entries(cats).find(([_, idxs]) => idxs.includes(i));
    return {
      name, unit: PU[i], category: cat ? cat[0] : "Other",
      current: +cur.toFixed(i >= 8 && i <= 14 ? 2 : 0),
      optimal: +opt.toFixed(i >= 8 && i <= 14 ? 2 : 0),
      delta: +delta.toFixed(i >= 8 && i <= 14 ? 3 : 1),
      deltaPct: +deltaPct.toFixed(1),
      gripGain: +gripGain.toFixed(4),
      stabGain: +stabGain.toFixed(4),
      priority: Math.abs(deltaPct) > 15 ? "HIGH" : Math.abs(deltaPct) > 5 ? "MED" : "LOW",
      min: ranges[i][0], max: ranges[i][1],
    };
  });
  const totalGrip = params.reduce((a, p) => a + p.gripGain, 0);
  const totalStab = params.reduce((a, p) => a + p.stabGain, 0);
  return { params, totalGrip: +totalGrip.toFixed(3), totalStab: +totalStab.toFixed(3) };
}

// ── Extended Grid 2: Compliance Steer + Ackermann Error ─────────────
export function gComplianceSteer(track){
  const R8=sd("compliance2026");
  return track.filter((_,i)=>i%3===0).map(p=>{
    const lg=Number(p.lat_g)||0;
    return{s:p.s,
      compSteer:+(lg*0.12+0.03*R8()).toFixed(3),
      ackErr:+(lg*0.8*(1-0.95)+0.2*R8()).toFixed(2),
      camberGain:+(-0.7*lg+0.1*R8()).toFixed(2),
    };
  });
}

// ── Extended Grid 3: Tire Wear Model ────────────────────────────────
export function gTireWear(nLaps=20){
  const R9=sd("tirewear2026");
  const d=[];let wFL=0,wFR=0,wRL=0,wRR=0;
  for(let i=1;i<=nLaps;i++){
    wFL+=0.08+0.03*R9();wFR+=0.09+0.03*R9();
    wRL+=0.06+0.02*R9();wRR+=0.065+0.02*R9();
    d.push({lap:i,wFL:+wFL.toFixed(2),wFR:+wFR.toFixed(2),wRL:+wRL.toFixed(2),wRR:+wRR.toFixed(2)});
  }
  return d;
}

// ── Extended Grid 4: Energy Budget ──────────────────────────────────
export function gEnergyBudget(track){
  const R10=sd("energy2026");
  let cumE=0;
  return track.filter((_,i)=>i%4===0).map(p=>{
    const v=Number(p.speed)||0,thr=Math.max(0,Number(p.lon_g)||0);
    const pDeploy=0.3*300*v*thr*(0.85+0.1*R10());
    const pRegen=Math.max(0,-Number(p.lon_g)||0)*0.3*300*v*0.35;
    cumE+=(pDeploy-pRegen)*0.02;
    return{s:p.s,deploy:+(pDeploy/1000).toFixed(2),regen:+(pRegen/1000).toFixed(2),cumKJ:+(cumE/1000).toFixed(1),socPct:+(100-cumE/3600/2.5*100).toFixed(1)};
  });
}

// ── Extended Grid 5: Solver Convergence per Step ────────────────────
export function gSolverIter(){
  const R11=sd("solver2026");
  const d=[];
  for(let i=0;i<80;i++){
    const nIter=Math.round(8+12*R11());
    const residual=0.1*Math.exp(-nIter/5)+0.005*R11();
    const solveMs=2+nIter*0.4+R11();
    d.push({step:i,nIter,residual:+residual.toFixed(4),solveMs:+solveMs.toFixed(1),feasible:residual<0.01?1:0});
  }
  return d;
}

// ── Extended Grid 6: Sensor Covariance ──────────────────────────────
export function gSensorCov(){
  return[
    {sensor:"IMU a_x",variance:0.015,unit:"m/s²",status:"GOOD"},
    {sensor:"IMU a_y",variance:0.012,unit:"m/s²",status:"GOOD"},
    {sensor:"Gyro ω_z",variance:0.008,unit:"rad/s",status:"GOOD"},
    {sensor:"WS FL",variance:0.25,unit:"rad/s",status:"GOOD"},
    {sensor:"WS FR",variance:0.28,unit:"rad/s",status:"GOOD"},
    {sensor:"WS RL",variance:0.22,unit:"rad/s",status:"GOOD"},
    {sensor:"WS RR",variance:0.24,unit:"rad/s",status:"GOOD"},
    {sensor:"Damper FL",variance:0.4,unit:"mm",status:"FAIR"},
    {sensor:"Damper FR",variance:0.45,unit:"mm",status:"FAIR"},
    {sensor:"Steer δ",variance:0.15,unit:"deg",status:"GOOD"},
    {sensor:"Brake P",variance:1.2,unit:"bar",status:"FAIR"},
    {sensor:"GPS pos",variance:12,unit:"mm",status:"WARN"},
  ];
}

// ═══════════════════════════════════════════════════════════════════════════════
// data.js — v4.0 PATCH (append to end of existing data.js)
// ═══════════════════════════════════════════════════════════════════════════════
// USAGE: git checkout src/data.js && cat src/data_v4_patch.js >> src/data.js
//
// NAMING: Two existing functions have v4 replacements with different signatures:
//   gEnergyBudget(track)  → kept as-is (used by SetupModule)
//   gEnergyBudgetPH()     → NEW standalone Port-Hamiltonian version
//   gWaveletCoeffs()      → kept as-is (used by SetupModule)
//   gWaveletCoeffsMPC()   → NEW version with ctl/level/idx/mag fields
// ═══════════════════════════════════════════════════════════════════════════════

// ── v4 RNG seeds ────────────────────────────────────────────────────────────
const R_TH = sd("thermal5node_gp4");
const R_EN = sd("energy_budget_gp4");
const R_AL = sd("alconstraint_gp4");
const R_WV = sd("wavelet_coeff_gp4");
const R_HN = sd("hnet_landscape_gp4");
const R_RM = sd("rmatrix_gp4");
const R_GP = sd("gpenvelope_gp4");
const R_HY = sd("hysteresis_gp4");
const R_FI = sd("fimeigen_gp4");
const R_TS = sd("truststeps_gp4");
const R_EP = sd("episodes_gp4");

// ── DOF labels for R_net dissipation matrix ─────────────────────────────────
export const R_DOF_LABELS = [
  "x","y","z","φ","θ","ψ",
  "z_fl","z_fr","z_rl","z_rr",
  "φ_wfl","φ_wfr","φ_wrl","φ_wrr",
];

// ── 5-Node Tire Thermal ─────────────────────────────────────────────────────
export function gThermal5(n=200){
  const d=[];
  let temps=[[90,82,70,55,40],[92,84,72,56,40],[85,78,66,52,38],[87,80,68,53,38]];
  for(let i=0;i<n;i++){
    const intensity=0.4+0.3*Math.sin(i/20)+0.1*R_TH();
    for(let c=0;c<4;c++){
      const heatIn=[intensity*3.5+R_TH()*2,intensity*2+R_TH(),intensity*1.2+R_TH()*0.5,intensity*0.6+R_TH()*0.3,intensity*0.15+R_TH()*0.1];
      const coolRate=[0.15,0.08,0.04,0.02,0.005];
      for(let j=0;j<5;j++) temps[c][j]=Math.min(165,Math.max(25,temps[c][j]+heatIn[j]-coolRate[j]*(temps[c][j]-30)));
    }
    d.push({s:i,fl:[...temps[0]],fr:[...temps[1]],rl:[...temps[2]],rr:[...temps[3]]});
  }
  return d;
}

// ── Port-Hamiltonian Energy Budget (standalone — NOT the old track-based one) ─
export function gEnergyBudgetPH(n=300){
  const d=[];let ke=4500,pe_s=120,pe_arb=35,h_net=15,diss_cum=0;
  for(let i=0;i<n;i++){
    const dt=0.02;
    ke+=(-50+100*R_EN())*dt; pe_s+=(-8+16*R_EN())*dt; pe_arb+=(-3+6*R_EN())*dt;
    const h_inj=Math.max(0,2*(R_EN()-0.6)); h_net+=h_inj*dt;
    const r_diss=8+12*R_EN(); diss_cum+=r_diss*dt;
    const H=ke+pe_s+pe_arb+h_net; const dH=(-50+100*R_EN())*dt+h_inj-r_diss;
    d.push({t:+(i*dt).toFixed(3),ke:+ke.toFixed(0),pe_s:+pe_s.toFixed(0),pe_arb:+pe_arb.toFixed(0),h_net:+h_net.toFixed(1),H:+H.toFixed(0),dH:+dH.toFixed(1),r_diss:+r_diss.toFixed(1),diss_cum:+diss_cum.toFixed(0)});
  }
  return d;
}

// ── AL Constraint Monitor ───────────────────────────────────────────────────
export function gALConstraints(n=200){
  const d=[];
  for(let i=0;i<n;i++){
    d.push({
      solve:i,
      grip:+(0.20*Math.exp(-i/80)+0.05*R_AL()).toFixed(3),
      steer:+(0.15+0.05*R_AL()).toFixed(3),
      ax:+(0.10+0.08*R_AL()).toFixed(3),
      track:+(0.30*Math.exp(-i/100)+0.03*R_AL()).toFixed(3),
      vx:+(0.20+0.06*R_AL()).toFixed(3),
      iters:Math.round(3+5*R_AL()),
      lambda_grip:+(1+4*(1-Math.exp(-i/60))+0.5*R_AL()).toFixed(2),
    });
  }
  return d;
}

// ── Wavelet Coefficients MPC (3 controls × 4 levels — NOT the old version) ──
export function gWaveletCoeffsMPC(){
  const d=[];const ctls=["steer","throttle","brake"];
  const levels=["cA3","cD3","cD2","cD1"];const sizes=[8,8,16,32];
  for(let c=0;c<3;c++){
    for(let l=0;l<4;l++){
      for(let k=0;k<sizes[l];k++){
        const baseMag=l===0?0.3:0.15/(l+1);
        d.push({ctl:ctls[c],level:levels[l],idx:k,mag:+(baseMag*(R_WV()-0.5)*2).toFixed(4)});
      }
    }
  }
  return d;
}

// ── Stochastic Tube Boundaries ──────────────────────────────────────────────
export function gTubePoints(track){
  const pts=[];
  for(let i=0;i<track.length;i++){
    const p=track[i];
    const halfWidth=1.8+1.2*Math.abs(p.curvature)*25+0.4*R_WV();
    let dx=0,dy=1;
    if(i<track.length-1){dx=+track[i+1].x-(+p.x);dy=+track[i+1].y-(+p.y);}
    else if(i>0){dx=+p.x-(+track[i-1].x);dy=+p.y-(+track[i-1].y);}
    const len=Math.sqrt(dx*dx+dy*dy)||1;
    const nx=-dy/len,ny=dx/len;
    pts.push({x:+p.x,y:+p.y,inX:+(+p.x+nx*halfWidth).toFixed(2),inY:+(+p.y+ny*halfWidth).toFixed(2),outX:+(+p.x-nx*halfWidth).toFixed(2),outY:+(+p.y-ny*halfWidth).toFixed(2),margin:+(4-halfWidth).toFixed(2)});
  }
  return pts;
}

// ── H_net Energy Landscape ──────────────────────────────────────────────────
export function gHnetLandscape(res=40){
  const d=[];
  for(let i=0;i<res;i++){for(let j=0;j<res;j++){
    const qf=(i/res-0.5)*40,qr=(j/res-0.5)*40;
    const r2=qf*qf+qr*qr;
    const H=50*Math.exp(-r2/400)+20*Math.exp(-((qf-10)**2)/100-((qr+5)**2)/100)+3*(R_HN()-0.5);
    d.push({i,j,qf:+qf.toFixed(1),qr:+qr.toFixed(1),H:+H.toFixed(2)});
  }}
  return d;
}

// ── R_net Dissipation Matrix (14×14 PSD) ────────────────────────────────────
export function gRMatrix(){
  const m=[];
  for(let i=0;i<14;i++){const row=[];for(let j=0;j<14;j++){
    if(i===j)row.push(+(5+15*R_RM()).toFixed(2));
    else if(j>i)row.push(+((R_RM()-0.5)*3).toFixed(2));
    else row.push(0);
  }m.push(row);}
  for(let i=0;i<14;i++)for(let j=0;j<i;j++)m[i][j]=m[j][i];
  return m;
}

// ── GP Uncertainty Envelope ─────────────────────────────────────────────────
export function gGPEnvelope(n=100){
  const d=[];
  for(let i=0;i<n;i++){
    const alpha=(i/n-0.5)*30;
    const mu=3000*Math.sin(1.4*Math.atan(12*(alpha*Math.PI/180)));
    const sigma=150+200*Math.abs(alpha)/15+100*R_GP();
    d.push({alpha:+alpha.toFixed(1),mu:+mu.toFixed(0),upper:+(mu+2*sigma).toFixed(0),lower:+(mu-2*sigma).toFixed(0),sigma:+sigma.toFixed(0)});
  }
  return d;
}

// ── Slip-Force Hysteresis ───────────────────────────────────────────────────
export function gHysteresis(n=200){
  const d=[];let Fy=0;
  for(let i=0;i<n;i++){
    const t=i*0.03,alpha=8*Math.sin(t*2);
    const Fy_ss=3000*Math.sin(1.4*Math.atan(12*(alpha*Math.PI/180)));
    Fy+=(Fy_ss-Fy)*Math.min(1,0.03/0.05);
    d.push({t:+t.toFixed(2),alpha:+alpha.toFixed(2),Fy:+Fy.toFixed(0),Fy_ss:+Fy_ss.toFixed(0)});
  }
  return d;
}

// ── Load Sensitivity ────────────────────────────────────────────────────────
export function gLoadSensitivity(n=50){
  const d=[];
  for(let i=0;i<n;i++){const Fz=i*60;d.push({Fz,mu:+(1.65-0.35*(Fz/3000)).toFixed(3),Ca:+(80+40*(1-Math.exp(-Fz/800))).toFixed(1)});}
  return d;
}

// ── Combined Slip Friction Surface ──────────────────────────────────────────
export function gFrictionSurface(res=30){
  const d=[];
  for(let ai=0;ai<res;ai++){for(let ki=0;ki<res;ki++){
    const alpha=(ai/res-0.5)*30,kappa=(ki/res-0.5)*0.6;
    const Fy=3000*Math.sin(1.4*Math.atan(12*(alpha*Math.PI/180)))*(1-0.4*Math.abs(kappa));
    const Fx=3200*Math.sin(1.6*Math.atan(10*kappa))*(1-0.5*Math.abs(alpha*Math.PI/180));
    const F=Math.sqrt(Fx*Fx+Fy*Fy);
    d.push({ai,ki,alpha:+alpha.toFixed(1),kappa:+kappa.toFixed(3),Fy:+Fy.toFixed(0),Fx:+Fx.toFixed(0),F:+F.toFixed(0)});
  }}
  return d;
}

// ── FIM Eigenspectrum ───────────────────────────────────────────────────────
export function gFIMEigen(){
  return Array.from({length:14},(_,i)=>({idx:i,eigenval:+(50*Math.exp(-i*0.5)+2*R_FI()).toFixed(2),label:PN[i]}));
}

// ── Trust Region Steps ──────────────────────────────────────────────────────
export function gTrustSteps(n=60){
  const d=[];let x=0,y=0;
  for(let i=0;i<n;i++){
    const dx=(R_TS()-0.5)*0.4,dy=(R_TS()-0.5)*0.4;
    const kl=0.005+0.01*R_TS();const accepted=kl<0.0094;
    if(accepted){x+=dx;y+=dy;}
    d.push({step:i,x:+x.toFixed(3),y:+y.toFixed(3),dx:+dx.toFixed(3),dy:+dy.toFixed(3),accepted,kl:+kl.toFixed(4)});
  }
  return d;
}

// ── MORL Episode Rewards ────────────────────────────────────────────────────
export function gEpisodes(n=150){
  const d=[];
  for(let i=0;i<n;i++){
    d.push({ep:i,grip:+(0.8+0.5*(1-Math.exp(-i/40))+0.08*(R_EP()-0.5)).toFixed(3),stab:+(2+1.5*(1-Math.exp(-i/50))+0.1*(R_EP()-0.5)).toFixed(3),combined:+(1.2+0.8*(1-Math.exp(-i/45))+0.06*(R_EP()-0.5)).toFixed(3)});
  }
  return d;
}

// ── Enhanced Pareto (adds id, hvContrib) ────────────────────────────────────
export function gP4(n=52){
  const RP=sd("pareto_gp4");const p=[];
  for(let i=0;i<n;i++){const w=0.5*(1-Math.cos(i*Math.PI/(n-1)));
    p.push({id:i,grip:+(1.25+0.3*w+0.04*(RP()-0.5)).toFixed(4),stability:+(0.5+2.8*(1-w)+0.15*(RP()-0.5)).toFixed(2),gen:Math.floor(RP()*200),omega:+w.toFixed(3),hvContrib:+(0.01+0.04*RP()).toFixed(4),params:PN.map(()=>RP())});}
  return p.sort((a,b)=>a.grip-b.grip);
}

// ── Enhanced Convergence (adds pgNorm, advMean, hvol) ───────────────────────
export function gCV4(n=200){
  const RCV=sd("convergence_gp4");const d=[];let b=1.1;
  for(let i=0;i<n;i++){b=Math.max(b,1.1+0.45*(1-Math.exp(-i/40))+0.02*RCV());
    d.push({iter:i,bestGrip:+b.toFixed(4),kl:+(0.012*Math.exp(-i/80)+0.001*RCV()).toFixed(5),entropy:+(-1+0.3*Math.exp(-i/60)+0.05*RCV()).toFixed(3),pgNorm:+(2*Math.exp(-i/50)+0.1*RCV()).toFixed(3),advMean:+(0.1*Math.exp(-i/30)+0.02*(RCV()-0.5)).toFixed(4),hvol:+(0.3+0.5*(1-Math.exp(-i/60))+0.02*RCV()).toFixed(4)});}
  return d;
}

// ── Enhanced Sensitivity (adds dTherm, uses all 28 params) ──────────────────
export function gSN4(){
  const RSN=sd("sensitivity_gp4");
  return PN.map((p,i)=>({param:p,unit:PU[i],dGrip:+((RSN()-0.3)*0.6).toFixed(3),dStab:+((RSN()-0.4)*0.4).toFixed(3),dTherm:+((RSN()-0.5)*0.2).toFixed(3)}));
}