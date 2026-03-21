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
  const vars=pareto.map(p=>p.params.reduce((a,v)=>a+(v-.5)**2,0)/28);
  const meanVar=vars.reduce((a,v)=>a+v,0)/vars.length;
  const conf=Math.max(0,Math.min(100,100*(1-meanVar*4)));
  const breakdown=[{axis:"Spring Rates",score:+(conf+3*RN()).toFixed(0)},{axis:"Damper Maps",score:+(conf-5+4*RN()).toFixed(0)},{axis:"ARB Stiffness",score:+(conf+2*RN()).toFixed(0)},{axis:"Geometry",score:+(conf-8+5*RN()).toFixed(0)},{axis:"Aero Platform",score:+(conf-3+3*RN()).toFixed(0)},{axis:"Thermal Model",score:+(conf-10+6*RN()).toFixed(0)}];
  return{score:+conf.toFixed(1),breakdown};
}

// ── Grid 2: Kinematic & Aero Platform ───────────────────────────────
export function gTK(n=360){const v=[];let cx=0,cy=0,psi=0;for(let i=0;i<n;i++){const si=i*.5,ki=.08*Math.sin(si/15)+.04*Math.sin(si/7)+.02*Math.sin(si/3.5);psi+=ki*.5;cx+=Math.cos(psi)*.5;cy+=Math.sin(psi)*.5;const vi=12+8/(1+3*Math.abs(ki))+1.5*RN();v.push({s:+si.toFixed(1),x:+cx.toFixed(2),y:+cy.toFixed(2),speed:+vi.toFixed(1),lat_g:+(vi*vi*ki/9.81).toFixed(3),lon_g:+(i>0?(vi-(v[i-1]?.speed||vi))/(.5/vi)/9.81:0).toFixed(3),curvature:+ki.toFixed(5)});}return v;}
export function gTT(n=200){const d=[];let a=25,b=25,c=25,e=25;for(let i=0;i<n;i++){const h=.4+.3*Math.sin(i/20)+.1*RN(),cl=.12;a=Math.min(128,Math.max(20,a+h*(.9+.2*RN())-cl*(a-25)/60));b=Math.min(128,Math.max(20,b+h*(1+.15*RN())-cl*(b-25)/60));c=Math.min(128,Math.max(20,c+h*(.75+.2*RN())-cl*(c-25)/60));e=Math.min(128,Math.max(20,e+h*(.8+.15*RN())-cl*(e-25)/60));d.push({s:i,tfl:+a.toFixed(1),tfr:+b.toFixed(1),trl:+c.toFixed(1),trr:+e.toFixed(1)});}return d;}
export function gSU(){const d=[];for(let i=0;i<150;i++){const t=i*.02;d.push({t:+t.toFixed(3),steer:+(12*Math.sin(t*2.5)).toFixed(2),roll:+(12*Math.sin(t*2.5)*.15+.3*RN()).toFixed(2),heave_f:+(-12.8+2.5*Math.sin(t*3)+.5*RN()).toFixed(2),heave_r:+(-14.2+2*Math.sin(t*3+.5)+.4*RN()).toFixed(2)});}return d;}

export function gFreqResponse(){const d=[];for(let f=.5;f<=25;f+=.5){const wnf=2*Math.PI*1.8,wnr=2*Math.PI*2.1,z=.35,w=2*Math.PI*f;d.push({freq:f,front_dB:+(20*Math.log10(1/Math.sqrt((1-(w/wnf)**2)**2+(2*z*w/wnf)**2))).toFixed(1),rear_dB:+(20*Math.log10(1/Math.sqrt((1-(w/wnr)**2)**2+(2*z*w/wnr)**2))).toFixed(1)});}return d;}

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

export function gWaveletCoeffs(){const levels=["cA3","cD3","cD2","cD1"];const d=[];levels.forEach((lv,li)=>{const n=li===0?8:8*(2**li);for(let i=0;i<Math.min(n,16);i++){const mag=li===0?.8+.4*R5():(1/(li+1))*R5();d.push({level:lv,idx:i,mag:+mag.toFixed(3),active:mag>.15});}});return d;}

export function gALSlack(track){return track.filter((_,i)=>i%4===0).map(p=>{const lg=Math.abs(Number(p.lat_g)||0);const slackGrip=Math.max(0,1.35-Math.sqrt(lg*lg+(Number(p.lon_g)||0)**2));const slackSteer=Math.max(0,.45-Math.abs(Number(p.curvature)||0)*10);return{s:p.s,slackGrip:+slackGrip.toFixed(3),slackSteer:+slackSteer.toFixed(3),penalty:+(slackGrip<.1?500*Math.pow(0.1-slackGrip,2):0).toFixed(1)};});}

export function gControlEffort(track){return track.filter((_,i)=>i%3===0).map(p=>{const lg=Math.abs(Number(p.lat_g)||0);const lon=Math.abs(Number(p.lon_g)||0);return{s:p.s,steerUtil:+Math.min(1,lg/.8+.1*R5()).toFixed(3),brakeUtil:+Math.min(1,lon/.6+.08*R5()).toFixed(3),throttleUtil:+Math.min(1,(Number(p.speed)||0)/25+.05*R5()).toFixed(3)};});}

// ── Grid 6: Twin Fidelity & EKF ─────────────────────────────────────
export function gEKFInnovation(n=300){const d=[];for(let i=0;i<n;i++){const t=i*.01;d.push({t:+t.toFixed(2),innov_ax:+(.02*Math.sin(t*15)+.01*R6()-.005).toFixed(4),innov_wz:+(.015*Math.sin(t*12)+.008*R6()-.004).toFixed(4),innov_vy:+(.03*Math.sin(t*8)+.015*R6()-.007).toFixed(4)});}return d;}

export function gPacejkaDrift(nLaps=20){const d=[];let muY=1.35,stiff=45000;for(let i=1;i<=nLaps;i++){muY-=.003+.002*R6();stiff-=80+60*R6();d.push({lap:i,muY:+muY.toFixed(4),stiffness:+stiff.toFixed(0),muY_pct:+((muY/1.35)*100).toFixed(1)});}return d;}

export function gPSDOverlay(){const d=[];for(let f=.5;f<=30;f+=.5){const base=-20+15*Math.exp(-Math.pow(f-2,2)/3)+8*Math.exp(-Math.pow(f-12,2)/8);const real=base+2*R6()-1;const sim=base+1.5*R6()-.75;d.push({freq:f,real:+real.toFixed(1),sim:+sim.toFixed(1)});}return d;}

export function gFidelitySpider(){return[{axis:"Kinematic R²",score:94,fullMark:100},{axis:"Dynamic NRMSE",score:88,fullMark:100},{axis:"Thermal ΔT",score:91,fullMark:100},{axis:"Slip Angle",score:86,fullMark:100},{axis:"Load Transfer",score:93,fullMark:100},{axis:"Freq. Response",score:89,fullMark:100}];}

export function gDamperHist(){const bins=[];for(let v=-.3;v<=.3;v+=.025){const x=v/.12;bins.push({vel:+v.toFixed(3),front:Math.round(800*Math.exp(-x*x/2)*(1+.3*RN())),rear:Math.round(600*Math.exp(-x*x/1.8)*(1+.25*RN()))});}return bins;}
