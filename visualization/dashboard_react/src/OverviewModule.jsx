import React, { useState } from "react";
import { C, GL } from "./theme.js";
import { Eq, Pt, Hl, InfoBox } from "./components.jsx";

const ARCH = [
  { key: "dynamics", label: "Vehicle Dynamics", sub: "Neural Port-Hamiltonian · 46-DOF", color: C.cy, icon: "⬡",
    desc: "Symplectic Leapfrog · 5 substeps · Cholesky dissipation · FiLM energy",
    body: () => (
      <>
        <Pt>The chassis obeys the <Hl>Port-Hamiltonian</Hl> equation — structurally guaranteeing energy conservation and bounded dissipation. Neural components physically cannot violate thermodynamics.</Pt>
        <Eq>{"ẋ = (J − R)·∇H(x) + F_ext(x, u)"}</Eq>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, margin: "10px 0" }}>
          <InfoBox color={C.cy} title="J — INTERCONNECTION"><Pt>Skew-symmetric: <Hl>Jᵀ = −J</Hl> guarantees xᵀJx ≡ 0. Conservative energy exchange between chassis, suspension, and wheel subsystems.</Pt></InfoBox>
          <InfoBox color={C.red} title="R — CHOLESKY DISSIPATION"><Pt>R_net predicts lower-triangular L, then <Hl color={C.red}>R = LLᵀ</Hl>. Structurally PSD — can only remove energy, never inject.</Pt></InfoBox>
        </div>
        <Pt><Hl color={C.gn}>H_net</Hl> (128→64→1 FiLM MLP) learns residual energy, gated by susp_sq = Σ(q−z_eq)²+10⁻⁴, forcing dH/dq≡0 at equilibrium. <Hl color={C.am}>Störmer-Verlet</Hl> (5 substeps in jax.lax.scan) preserves symplectic 2-form — energy error oscillates O(h²) with zero secular drift.</Pt>
      </>
    ),
  },
  { key: "tire", label: "Tire Model", sub: "Pacejka MF6.2 + PINN + Sparse GP", color: C.gn, icon: "◉",
    desc: "Hoosier R20 TTC · 5-node Jaeger thermal · Matérn 5/2 uncertainty",
    body: () => (
      <>
        <Pt>Three-layer hybrid: analytical physics backbone + ML residual correction + calibrated uncertainty.</Pt>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, margin: "10px 0" }}>
          <InfoBox color={C.cy} title="PACEJKA MF6.2"><Pt>Full Magic Formula with load/camber sensitivity, combined-slip G_yk/G_xa, turn-slip φ_t = a/R_path.</Pt></InfoBox>
          <InfoBox color={C.am} title="5-NODE THERMAL"><Pt>Jaeger flash → 5-node ODE (Surface In/Mid/Out, Core, Gas). Gay-Lussac pressure → dynamic Fz.</Pt></InfoBox>
          <InfoBox color={C.gn} title="PINN + SPARSE GP"><Pt>Spectrally-normalized PINN drift on [sin(2α), κ³, T_norm]. 50 inducing-pt GP gives σ for LCB.</Pt></InfoBox>
        </div>
        <Eq>{"T_flash = q·a / (k · √(π · V_slide · a / α))"}</Eq>
        <Eq>{"k(d) = σ²·(1 + √5·d + 5d²/3)·exp(−√5·d)     [Matérn 5/2]"}</Eq>
      </>
    ),
  },
  { key: "wmpc", label: "Diff-WMPC", sub: "Wavelet MPC + Stochastic Tubes", color: C.am, icon: "◈",
    desc: "3-level Db4 DWT · 5-sigma UT · augmented Lagrangian",
    body: () => (
      <>
        <Pt>Control horizon in <Hl color={C.am}>frequency domain via 3-level Daubechies-4 DWT</Hl>. IDWT reconstruction inherently band-limits to smooth, drivable trajectories.</Pt>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, margin: "10px 0" }}>
          <InfoBox color={C.am} title="UNSCENTED TRANSFORM"><Pt>5 σ-points along grip/wind axes. All trajectories vmap-simulated in parallel → spatial μ_n and σ²_n → Stochastic Tube with log-barrier boundaries.</Pt></InfoBox>
          <InfoBox color={C.red} title="NaN GRADIENT RECOVERY"><Pt>Unstable inputs → NaN. Python interceptor returns ∇(10⁶+0.5‖c‖²)=c, a smooth bowl recovering to stability.</Pt></InfoBox>
        </div>
        <Eq>{"L_AL = f(x) + λᵀ·c(x) + (ρ/2)·‖c(x)‖²"}</Eq>
      </>
    ),
  },
  { key: "morl", label: "MORL-SB-TRPO", sub: "28-dim Setup Optimizer · Pareto", color: C.red, icon: "◆",
    desc: "ARD BO cold-start · Chebyshev ensemble · Adam + trust region",
    body: () => (
      <>
        <Pt>Maps the full <Hl color={C.red}>Pareto frontier</Hl> in 28D setup space. All 20 ensemble members get exact gradients through differentiable physics.</Pt>
        <Eq>{"max  Σ_k [ ω_k·Grip + (1−ω_k)·Stability + 0.005·Σlog(σ_k) ]"}</Eq>
        <Eq>{"s.t.  D_KL( π_k ‖ π_k_old ) ≤ 0.005·√(28/8) ≈ 0.0094"}</Eq>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, margin: "10px 0" }}>
          <InfoBox color={C.am} title="CHEBYSHEV"><Pt>ω_i = 0.5·(1−cos(iπ/(N−1))). ~65% of ensemble at the high-grip Pareto boundary.</Pt></InfoBox>
          <InfoBox color={C.pr} title="TRUST REGION"><Pt>KL cap + 10-iter lag. Entropy bonus 0.005·Σlog(σ) prevents collapse. Bottom 5 reinitialized every 200 iter.</Pt></InfoBox>
          <InfoBox color={C.gn} title="ARD BO COLD-START"><Pt>10+30 EI with per-dim lengthscales. Auto-prunes low-sensitivity dims → effective ~8-10D search.</Pt></InfoBox>
        </div>
      </>
    ),
  },
];

export default function OverviewModule() {
  const [expanded, setExpanded] = useState(null);
  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 14 }}>
        {ARCH.map((c) => {
          const isOpen = expanded === c.key;
          return (
            <div key={c.key} onClick={() => setExpanded(isOpen ? null : c.key)}
              style={{
                ...GL, borderLeft: `3px solid ${c.color}`, cursor: "pointer",
                padding: "18px 20px", gridColumn: isOpen ? "1 / -1" : undefined,
                transition: "all 0.35s cubic-bezier(0.25,0.1,0.25,1)",
              }}
              onMouseEnter={e => { e.currentTarget.style.background = C.hover; e.currentTarget.style.boxShadow = `0 8px 40px ${c.color}12, inset 0 1px 0 rgba(255,255,255,0.04)`; }}
              onMouseLeave={e => { e.currentTarget.style.background = C.card; e.currentTarget.style.boxShadow = GL.boxShadow; }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                <span style={{ fontSize: 22, color: c.color, filter: `drop-shadow(0 0 6px ${c.color}40)` }}>{c.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: c.color, letterSpacing: 1.5, fontFamily: C.hd, textTransform: "uppercase" }}>{c.label}</div>
                  <div style={{ fontSize: 11, fontWeight: 500, color: C.w, fontFamily: C.bd, marginTop: 1 }}>{c.sub}</div>
                </div>
                <div style={{ fontSize: 18, color: C.dm, transform: isOpen ? "rotate(90deg)" : "none", transition: "transform 0.3s", fontFamily: C.hd }}>›</div>
              </div>
              <div style={{ fontSize: 11, color: C.md, fontFamily: C.bd, lineHeight: 1.6 }}>{c.desc}</div>
              {isOpen && <div style={{ marginTop: 16, paddingTop: 16, borderTop: `1px solid ${c.color}15` }}>{c.body()}</div>}
            </div>
          );
        })}
      </div>
      {/* System status */}
      <div style={{ marginTop: 20, ...GL, display: "flex", alignItems: "center", gap: 20, padding: "14px 20px", flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div className="pulse-dot" style={{ width: 8, height: 8, borderRadius: "50%", background: C.gn, boxShadow: `0 0 10px ${C.gn}` }} />
          <span style={{ fontSize: 10, fontWeight: 700, color: C.gn, letterSpacing: 2, textTransform: "uppercase", fontFamily: C.dt }}>System Online</span>
        </div>
        <div style={{ width: 1, height: 20, background: C.b1 }} />
        {[["Engine","JAX NPH/LFNO"],["State","46-dim"],["Setup","28-dim"],["Stack","100% Differentiable"]].map(([k,v]) => (
          <div key={k} style={{ fontFamily: C.dt, fontSize: 10 }}>
            <span style={{ color: C.dm, letterSpacing: 1, textTransform: "uppercase" }}>{k} </span>
            <span style={{ color: C.br, fontWeight: 600 }}>{v}</span>
          </div>
        ))}
      </div>
    </>
  );
}
