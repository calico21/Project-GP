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