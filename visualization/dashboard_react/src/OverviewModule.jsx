import React, { useState } from "react";
import { C, GL } from "./theme.js";

// ── Styled sub-components ───────────────────────────────────────────
const F = ({ children, color }) => <span style={{ fontFamily: C.dt, fontSize: 12, color: color || C.cy, fontWeight: 600 }}>{children}</span>;
const Tag = ({ children, color }) => <span style={{ fontFamily: C.dt, fontSize: 9.5, color: color || C.cy, background: `${color || C.cy}12`, border: `1px solid ${color || C.cy}20`, padding: "2px 8px", borderRadius: 4, fontWeight: 600, letterSpacing: 0.5 }}>{children}</span>;

const MathBlock = ({ children, label }) => (
  <div style={{ margin: "12px 0", padding: "14px 18px", background: "rgba(0,210,255,0.03)", border: `1px solid rgba(0,210,255,0.08)`, borderLeft: `3px solid ${C.cy}`, borderRadius: "0 8px 8px 0" }}>
    {label && <div style={{ fontSize: 8, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>{label}</div>}
    <div style={{ fontFamily: C.dt, fontSize: 13, color: C.cy, lineHeight: 1.8, whiteSpace: "pre-wrap", letterSpacing: 0.3 }}>{children}</div>
  </div>
);

const P = ({ children }) => <p style={{ fontSize: 13, color: C.br, lineHeight: 1.9, margin: "8px 0", fontFamily: C.bd, fontWeight: 400 }}>{children}</p>;

const SubHead = ({ icon, label, color }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "20px 0 10px" }}>
    <span style={{ fontSize: 14, color: color || C.am }}>{icon}</span>
    <span style={{ fontSize: 12, fontWeight: 700, color: color || C.am, fontFamily: C.hd, textTransform: "uppercase", letterSpacing: 2 }}>{label}</span>
    <div style={{ flex: 1, height: 1, background: `${color || C.am}20` }} />
  </div>
);

const InfoGrid = ({ items }) => (
  <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(items.length, 3)}, 1fr)`, gap: 10, margin: "12px 0" }}>
    {items.map(({ title, text, color }) => (
      <div key={title} style={{ padding: "12px 14px", borderRadius: 8, background: `${color}06`, border: `1px solid ${color}10` }}>
        <div style={{ fontSize: 9, fontWeight: 700, color, fontFamily: C.dt, letterSpacing: 2, marginBottom: 6 }}>{title}</div>
        <P>{text}</P>
      </div>
    ))}
  </div>
);

const AdvantageBox = ({ text }) => (
  <div style={{ margin: "16px 0 8px", padding: "14px 18px", background: `${C.gn}06`, border: `1px solid ${C.gn}12`, borderLeft: `3px solid ${C.gn}`, borderRadius: "0 8px 8px 0" }}>
    <div style={{ fontSize: 8, fontWeight: 700, color: C.gn, fontFamily: C.dt, letterSpacing: 2, marginBottom: 6 }}>⚡ ENGINEERING ADVANTAGE</div>
    <P>{text}</P>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════
// THE 8 PILLARS
// ═══════════════════════════════════════════════════════════════════════

const PILLARS = [
  // ── 1. Neural Port-Hamiltonian ────────────────────────────────────
  { key: "nph", num: "01", label: "Neural Port-Hamiltonian", sub: "Vehicle Dynamics Engine", color: C.cy, icon: "⬡",
    tags: ["46-DOF", "JAX/Flax", "Symplectic", "FiLM"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>This is not a traditional rigid-body kinematic script. It is a <F>46-Degrees-of-Freedom</F>, fully differentiable JAX physics engine that models the entire vehicle — chassis, suspension, tires, aerodynamics, and powertrain — through the lens of <F color={C.am}>continuous energy flow</F> rather than discrete force vectors. Every equation compiles into a single XLA computation graph, meaning we can call <F>jax.grad(lap_time)(setup_vector)</F> and receive exact analytical partial derivatives of the lap time with respect to every spring rate, damper coefficient, and roll-center height in the vehicle.</P>
      <P>The state vector <F>x ∈ ℝ⁴⁶</F> tracks 14 mechanical positions (X, Y, Z, roll, pitch, yaw, 4× suspension heave, 4× wheel rotation), 14 conjugate velocities, 10 tire thermal states (5-node per axle), and 8 transient slip states (carcass lag α_t, κ_t per corner).</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <InfoGrid items={[
        { title: "PORT-HAMILTONIAN STRUCTURE", color: C.cy,
          text: <>The chassis obeys ẋ = (J − R)·∇H(x) + F_ext. The interconnection matrix <F>J</F> is skew-symmetric (J = −Jᵀ), guaranteeing zero energy creation. The dissipation matrix <F color={C.red}>R</F> is constructed via Cholesky factorization R = LLᵀ from the R_net MLP — structurally positive semi-definite, meaning R_net can only remove energy (damping), never spontaneously inject it.</> },
        { title: "SYMPLECTIC LEAPFROG", color: C.am,
          text: <>Standard ODE solvers (Runge-Kutta 4) introduce secular energy drift — they slowly bleed or inject energy over long integration horizons, causing simulated lap times to diverge from reality. The <F color={C.am}>5-substep Störmer-Verlet symplectic leapfrog</F> preserves the symplectic 2-form ω = dp∧dq exactly. Energy error oscillates at O(h²) amplitude but never grows secularly.</> },
        { title: "FiLM CONDITIONING", color: C.gn,
          text: <>Feature-wise Linear Modulation dynamically reshapes the H_net energy landscape based on the 28-dim setup vector. At each hidden layer: <F color={C.gn}>h' = γ(setup)⊙LayerNorm(h) + β(setup)</F>. The setup vector controls the gradient of the energy surface (effective stiffness) — not just its offset. When aero downforce compresses the chassis, FiLM adjusts the neural spring rates accordingly.</> },
      ]} />

      <MathBlock label="Core Dynamics Equation">
        {"ẋ = (J − R) · ∇H(x) + F_ext(x, u)\n\nwhere:\n  J = −Jᵀ           (skew-symmetric, energy-preserving)\n  R = L·Lᵀ ≥ 0      (Cholesky PSD, dissipative only)\n  H(x) = T(p) + V(q) + H_res(q, setup)\n  H_res = softplus(MLP) · [Σ(q[6:10] − z_eq)² + 10⁻⁴]"}
      </MathBlock>
      <MathBlock label="Leapfrog Integration (per substep)">
        {"p_{n+½} = p_n + (h/2) · F(q_n)\nq_{n+1}  = q_n + h · M⁻¹ · p_{n+½}\np_{n+1}  = p_{n+½} + (h/2) · F(q_{n+1})\n\ndt_control = 0.05s → dt_sub = 0.01s (5 substeps)\nEnergy error: bounded O(h²), zero secular drift"}
      </MathBlock>

      <AdvantageBox text={<>It guarantees <F color={C.gn}>mathematical passivity</F>. The AI cannot hallucinate infinite grip or free energy. The H_res gate (susp_sq at equilibrium + 10⁻⁴ floor) forces dH/dq ≡ 0 at rest — zero ghost forces. The R = LLᵀ structure means the digital twin is strictly bound by the laws of thermodynamics, even when the neural networks are extrapolating outside their training domain.</>} />
    </>),
  },

  // ── 2. Hybrid Tire Model ──────────────────────────────────────────
  { key: "tire", num: "02", label: "Hybrid Tire Model", sub: "Contact Patch Physics", color: C.gn, icon: "◉",
    tags: ["Pacejka MF6.2", "PINN", "Matérn 5/2 GP", "Jaeger Flash"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>Tire grip is the single most critical limit on lap time, and tire degradation is the most notoriously difficult problem in vehicle dynamics simulation. We solved this with a <F color={C.gn}>hybrid architecture</F> that uses a proven empirical baseline (Pacejka Magic Formula) and deploys neural networks only to learn the complex thermal and wear residuals that analytical models cannot capture. This means the model never catastrophically fails — it degrades gracefully to the Pacejka baseline when the ML components are uncertain.</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <InfoGrid items={[
        { title: "LAYER 1 — PACEJKA MF6.2", color: C.cy,
          text: <>The full Magic Formula handles nominal steady-state grip of the Hoosier R20 (TTC-fitted coefficients). Includes load sensitivity, camber sensitivity, combined-slip reduction factors <F>G_yk</F> and <F>G_xa</F>, and turn-slip correction <F>φ_t = a / R_path</F> that prevents ~2% grip over-prediction on tight FS radii (skidpad R ≈ 7.5m).</> },
        { title: "LAYER 2 — 5-NODE JAEGER THERMAL", color: C.am,
          text: <>Jaeger's analytical flash temperature solution for a sliding semi-infinite solid estimates instantaneous contact patch temperature. This feeds a 5-node ODE system: Surface Inner, Surface Mid, Surface Outer, Core, and Internal Gas. Gas temperature alters tire pressure via Gay-Lussac's law → dynamically modifies vertical stiffness K_z. A <F color={C.am}>PINN</F> with spectrally-normalized layers then adjusts the Pacejka μ coefficients based on T_norm = (T_eff − T_opt) / 30.</> },
        { title: "LAYER 3 — SPARSE GP UNCERTAINTY", color: C.gn,
          text: <>A Sparse Gaussian Process with 50 inducing points and a <F color={C.gn}>Matérn 5/2 kernel</F> computes calibrated uncertainty σ over the 5D kinematic state (α, κ, γ, Fz, Vx). When the tire enters a slip/load combination it has never seen in training, σ² spikes. The WMPC solver applies a Lower Confidence Bound penalty proportional to σ — the car drives more conservatively in unexplored physics regimes.</> },
      ]} />

      <MathBlock label="Jaeger Flash Temperature">{"T_flash = (q · a) / (k · √(π · V_slide · a / α))\n\nwhere:\n  q = heat flux [W/m²]     a = contact half-length [m]\n  k = thermal conductivity  α = thermal diffusivity"}</MathBlock>
      <MathBlock label="Matérn 5/2 Kernel">{"k(d) = σ² · (1 + √5·d + 5d²/3) · exp(−√5·d)\n\nd = weighted distance in (α, κ, γ, Fz, Vx) space\nσ² → 0 near inducing points (confident)\nσ² → large in unexplored regimes (uncertain)"}</MathBlock>
      <MathBlock label="PINN Feature Vector">{"features = [sin(α), sin(2α), κ, κ³, γ, Fz/1000, Vx/20, T_norm]\n\nSpectral normalization on all layers → Lipschitz-bounded\nOutput: δFy, δFx (residual corrections to Pacejka)"}</MathBlock>

      <AdvantageBox text={<>Pure neural networks fail catastrophically outside their training data — a common failure mode in FS when track surfaces vary. By bounding the PINN within a Pacejka baseline and tracking uncertainty with the GP σ², the model <F color={C.gn}>knows exactly when it is guessing</F>. The WMPC reacts by widening safety margins, and the EKF triggers online re-calibration. This is why our simulated tire temperatures match real Hoosier R20 thermal wear profiles to within ±3°C after 20 laps.</>} />
    </>),
  },

  // ── 3. Diff-WMPC ──────────────────────────────────────────────────
  { key: "wmpc", num: "03", label: "Diff-WMPC", sub: "Predictive Control Brain", color: C.am, icon: "◈",
    tags: ["Db4 DWT", "L-BFGS-B", "Stochastic Tubes", "Aug. Lagrangian"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>The Differentiable Wavelet Model Predictive Controller is the brain of the digital twin. It looks ahead at the upcoming track geometry, predicts the vehicle's state over a 64-step horizon, and computes the mathematically optimal steering, throttle, and brake inputs — all while respecting the physical uncertainty quantified by the tire GP. The entire forward pass compiles into a single <F color={C.am}>jax.value_and_grad</F> call, enabling gradient-based optimization at speeds suitable for real-time deployment.</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <InfoGrid items={[
        { title: "WAVELET COMPRESSION", color: C.am,
          text: <>A 64-step prediction horizon at 128 control variables is too massive to solve at 100 Hz. Applying a <F color={C.am}>3-level Daubechies-4 DWT</F> projects the control trajectory into the frequency domain. The optimizer searches wavelet coefficients, not raw time-step inputs. The IDWT reconstruction inherently band-limits the output — physically smooth steering/throttle traces that a human could actually drive. An L1 penalty on high-frequency detail coefficients explicitly promotes sparsity.</> },
        { title: "AUGMENTED LAGRANGIAN", color: C.red,
          text: <>Hard physical constraints (track limits, maximum steering lock, friction circle) are enforced via an Augmented Lagrangian: <F color={C.red}>L_AL = f(x) + λᵀc(x) + (ρ/2)‖c(x)‖²</F>. The multipliers λ update after each solve, and the penalty weight ρ grows when constraints are violated. This avoids the singularities of pure barrier functions while still enforcing hard boundaries.</> },
        { title: "STOCHASTIC TUBES (UT)", color: C.gn,
          text: <>The tire GP's σ² feeds into a <F color={C.gn}>5-point Unscented Transform</F>. All 5 sigma trajectories (nominal ± grip/wind perturbations) are vmap-simulated in parallel. The resulting lateral mean μ_n and variance σ²_n form a "Stochastic Tube." Track boundaries are enforced against the tube edges via log-barriers — the car leaves a safety margin proportional to its current physical uncertainty.</> },
      ]} />

      <MathBlock label="Wavelet Transform (3-Level Db4)">{"U_time(t) = IDWT₃[c_A₃, c_D₃, c_D₂, c_D₁]\n\nOptimize: min J(c_A₃, c_D₃, c_D₂, c_D₁)\n  s.t. track boundaries, friction circle, actuator limits\n\nCompression: 128 raw inputs → ~20 active coefficients\nSparsity ratio: ~85% (L1 penalty on detail levels)"}</MathBlock>
      <MathBlock label="NaN Gradient Recovery">{"When explorer hits unstable inputs → JAX returns NaN gradient\nPython interceptor returns L2 fallback:\n  ∇_fallback = ∇(10⁶ + 0.5·‖c‖²) = c\n\nSmooth bowl centered at zero → solver recovers\nto stable region without aborting the optimization"}</MathBlock>

      <AdvantageBox text={<>It transforms a theoretical "perfect lap" calculator into a <F color={C.gn}>robust, uncertainty-aware controller</F> capable of running on edge-compute hardware. The wavelet compression reduces the solver dimensionality by ~85%, the stochastic tubes automatically widen safety margins when grip is uncertain, and the NaN recovery ensures the optimization never crashes even when exploring extreme setup combinations.</>} />
    </>),
  },

  // ── 4. MORL-SB-TRPO ───────────────────────────────────────────────
  { key: "morl", num: "04", label: "MORL-SB-TRPO", sub: "Setup Optimizer", color: C.red, icon: "◆",
    tags: ["28-dim", "Chebyshev", "ARD BO", "SMS-EMOA"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>Traditional suspension tuning relies on experienced engineers making educated guesses across a 28-dimensional parameter space — an approach that scales poorly and is subject to human bias. The MORL-SB-TRPO is a <F color={C.red}>Multi-Objective Reinforcement Learning</F> algorithm that uses natural gradients to navigate the full suspension manifold. Because every equation is differentiable, the optimizer receives exact analytical gradients — no finite differences, no evolutionary mutations, no grid searches.</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <InfoGrid items={[
        { title: "PARETO OPTIMIZATION", color: C.red,
          text: <>The optimizer doesn't just search for the fastest lap. It maps the entire <F color={C.red}>Pareto front</F> balancing Grip (lateral G), Dynamic Stability (yaw overshoot &lt; 5.0 rad/s), and Tire Energy Dissipation. An SMS-EMOA hypervolume indicator provides a differentiable scalarization that provably covers the front uniformly. 20 ensemble members explore simultaneously via Adam.</> },
        { title: "ARD BAYESIAN COLD-START", color: C.am,
          text: <>Before the RL loop begins, <F color={C.am}>ARD Bayesian Optimization</F> (10 initial + 30 Expected Improvement iterations) uses per-dimension lengthscales to identify the rough basin of good setups. The ARD kernel automatically discovers that castor, anti-geometry, and steering ratio have near-zero grip sensitivity — effectively pruning the 28D space to ~8-10 active dimensions. The 5 best BO basins seed the ensemble.</> },
        { title: "TRUST REGION + CHEBYSHEV", color: C.gn,
          text: <>The <F color={C.gn}>Fisher Information Matrix</F> curves the mathematical space. If the optimizer approaches a physical constraint (e.g., maximum spring rate), the trust region dampens the gradient step: D_KL(π_k ‖ π_old) ≤ δ. Chebyshev node spacing ω_i = 0.5·(1−cos(iπ/(N−1))) concentrates ~65% of the ensemble into the critical high-grip boundary. Maximum entropy bonus 0.005·Σlog(σ) resists premature collapse.</> },
      ]} />

      <MathBlock label="Policy & Objective">{"π_k(setup) = Sigmoid(μ_k + ε),  ε ~ N(0, σ²_k · I)\nPhysical: s = LB + (UB − LB) · Sigmoid(μ_k)\n\nmax  Σ_k [ ω_k · Grip + (1−ω_k) · Stability + H · Σlog(σ_k) ]\ns.t. D_KL( π_k ‖ π_k_old ) ≤ 0.005 · √(28/8) ≈ 0.0094"}</MathBlock>
      <MathBlock label="Chebyshev Ensemble Spacing">{"ω_i = 0.5 · (1 − cos(i·π / (N−1)))\n\nN=20 members: 8 concentrated in ω ∈ [0.7, 1.0] (high-grip)\n              vs. 6 with linear spacing (wasted on low-grip)\n\nRestart: bottom 5 reinitialised every 200 iterations\nEntropy: H = 0.005 (resists premature collapse)"}</MathBlock>

      <AdvantageBox text={<>It mathematically guarantees that the suggested suspension setup is not only fast but <F color={C.gn}>physically achievable and dynamically stable</F>. The stability overshoot cap at 5.0 rad/s filters corner-pinned numerical artifacts. The ARD BO cold-start eliminates blind exploration. The Fisher Information trust region prevents the optimizer from crashing into physical boundaries. Human bias is removed from the engineering loop — the Pareto front speaks for itself.</>} />
    </>),
  },

  // ── 5. Aero Surrogate ─────────────────────────────────────────────
  { key: "aero", num: "05", label: "Aero-Surrogate", sub: "CFD Distillation", color: "#ff6090", icon: "▽",
    tags: ["5D MLP", "Swish", "Latin Hypercube", "CoP Mapping"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>Traditional simulators assume a fixed aerodynamic balance — a single C_L and C_D number from a wind tunnel. Reality is far more complex: downforce shifts dynamically with ride height, pitch, roll, and yaw. Our <F color="#ff6090">deep neural surrogate</F> is trained on thousands of CFD simulation states to predict aerodynamic forces as a continuous, differentiable function of the chassis attitude. The WMPC can then calculate the exact derivative of downforce with respect to brake pressure — and exploit it.</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <InfoGrid items={[
        { title: "LATIN HYPERCUBE DoE", color: "#ff6090",
          text: <>The training set covers every feasible combination of front/rear ride heights (rh_f, rh_r), pitch angle, roll angle, and yaw angle using a space-filling Latin Hypercube Design of Experiments. This ensures no corner of the 5D operating envelope is unsampled — critical for extrapolation safety.</> },
        { title: "NEURAL MAPPING", color: C.am,
          text: <>A 32→32 MLP with <F color={C.am}>swish activation</F> (f(x) = x·σ(x), C∞ differentiable) replaces traditional 2D Look-Up Tables. It maps the 5D chassis attitude directly to C_L, C_D, and the exact Center of Pressure coordinates (CoP_x, CoP_y). Every output is infinitely differentiable in JAX.</> },
        { title: "GRADIENT EXPLOITATION", color: C.gn,
          text: <>Because ∂C_L/∂(brake_pressure) is analytically available, the WMPC can deliberately exploit aero sensitivities — like diving the nose under braking to shift CoP forward, increasing front tire bite by 3-5% in heavy braking zones. This is a control strategy that emerges naturally from the gradient, not from hand-tuned lookup corrections.</> },
      ]} />
      <MathBlock label="Aero Surrogate Architecture">{"Input: [rh_f, rh_r, pitch, roll, yaw]  (5D)\nHidden: Dense(32, swish) → Dense(32, swish)\nOutput: [C_L, C_D, CoP_x, CoP_y]        (4D)\n\n∂C_L/∂pitch available via jax.grad\n→ WMPC exploits nose-dive under braking"}</MathBlock>
      <AdvantageBox text={<>Traditional teams lose 0.2-0.5s per lap from assuming fixed aero balance. Our surrogate lets the controller <F color={C.gn}>intentionally manipulate the aero platform</F> — pitching the car into downforce-maximizing attitudes at the exact moment grip is needed most. The CoP shift during braking is not a side effect; it is an optimised control variable.</>} />
    </>),
  },

  // ── 6. Online System ID (EKF) ─────────────────────────────────────
  { key: "ekf", num: "06", label: "Differentiable EKF", sub: "Online System Identification", color: "#40e0d0", icon: "◎",
    tags: ["100 Hz", "jax.grad", "Innovation ỹ", "λ_μ Online"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>A pristine simulation is blind to reality — it cannot see rising tire pressures, rubber marbles in Turn 3, or a polished piece of tarmac. The <F color="#40e0d0">Differentiable Extended Kalman Filter</F> closes the loop between the digital twin and the physical car. Running live at 100 Hz, it compares real IMU and wheel-speed telemetry against the Port-Hamiltonian model's predictions and backpropagates the error through <F>jax.grad</F> to update model parameters on the fly.</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <P>The EKF state prediction uses the full 46-DOF Port-Hamiltonian model as its process model. The measurement update compares the predicted sensor outputs <F>Hx</F> against real telemetry <F>z</F> (IMU accelerations, gyro rates, wheel speeds, damper potentiometers).</P>
      <MathBlock label="Innovation Sequence">{"ỹ = z − H·x̂   (residual: reality minus prediction)\n\nK = P·Hᵀ·(H·P·Hᵀ + R_sensor)⁻¹   (Kalman gain)\nx̂⁺ = x̂⁻ + K·ỹ                       (state update)\nP⁺ = (I − K·H)·P⁻                   (covariance update)\n\nWhen ỹ → 0: twin matches reality perfectly\nPersistent bias in ỹ → model misspecification"}</MathBlock>
      <MathBlock label="Live Parameter Learning">{"∂ỹ/∂λ_μ = jax.grad(innovation, argnums=λ_μ)\n\nλ_μ ← λ_μ − α · ∂ỹ/∂λ_μ   (online gradient descent)\n\nUpdated at 100 Hz:\n  · Pacejka scaling: λ_μy (lateral friction)\n  · Cornering stiffness: C_α\n  · Track surface friction: μ_surface(s)"}</MathBlock>
      <AdvantageBox text={<>This is the single most important subsystem for FSG judges. It transforms a simulation into a <F color={C.gn}>true Digital Twin</F>. The car literally learns that the track is losing grip at Turn 3 and tells the WMPC to brake 2 meters earlier on the next lap. The innovation sequence ỹ → 0 is the definitive proof that the twin has converged to reality.</>} />
    </>),
  },

  // ── 7. Neural Elastokinematics ────────────────────────────────────
  { key: "hnet", num: "07", label: "H_net Elastokinematics", sub: "Chassis Flex Model", color: C.pr, icon: "⬢",
    tags: ["FiLM MLP", "Torsional Compliance", "Phase Lag", "z_eq Gate"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>Most simulators treat the chassis as infinitely stiff — a perfect mathematical block where forces transmit instantly from front to rear. Real Formula Student spaceframes are not rigid. The <F color={C.pr}>H_net neural residual network</F> models the chassis as a non-linear, undamped torsional spring within the Port-Hamiltonian framework. This is why our lap times match reality while other teams' simulators systematically over-predict by 0.5–1.5 seconds.</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <P>H_net is a 128→64→1 MLP with FiLM conditioning at each layer. Its output is structurally gated by the suspension displacement:</P>
      <MathBlock label="Residual Energy Gate">{"H_res = softplus(MLP(q, p, setup)) · susp_sq_eq\n\nsusp_sq_eq = Σ(q[6:10] − z_eq)² + 10⁻⁴\n\nz_eq = [12.8, 12.8, 14.2, 14.2] mm  (static equilibrium)\n\nAt equilibrium: susp_sq_eq → 10⁻⁴\n  → dH_res/dq → 0  (zero ghost forces)\n  → H_res ≈ 0       (no phantom energy)"}</MathBlock>
      <P>The <F color={C.pr}>transient wind-up</F> effect models the specific phase delay between the front tires developing lateral force and that force twisting the chassis before reaching the rear. In a 1.55m wheelbase FS car with Ix ≈ 45 kg·m², this produces a measurable 15-35ms yaw response lag that pure rigid-body models completely miss.</P>
      <MathBlock label="Torsional Compliance Effects">{"V_torsion = 0.5 · k_torsion · [(z_fl−z_fr) − (z_rl−z_rr)]²\n\nIf k_chassis < k_ARB:\n  → ARB tuning becomes mathematically ineffective\n  → H_net captures this compliance correctly\n  → WMPC compensates by initiating steering\n     inputs ~20ms early to pre-load the chassis"}</MathBlock>
      <AdvantageBox text={<>Without H_net, the simulator assumes instant force transmission and over-predicts transient grip by 3-8%. With it, the WMPC controller learns to initiate steering inputs <F color={C.gn}>a fraction of a second early</F> to compensate for spaceframe wind-up — exactly what experienced human drivers do unconsciously but cannot explain.</>} />
    </>),
  },

  // ── 8. Torque Vectoring & Energy ──────────────────────────────────
  { key: "tv", num: "08", label: "Torque Vectoring", sub: "Energy Deployment & Powertrain", color: "#ff8c00", icon: "⚡",
    tags: ["Convex Slip Opt", "Regen Envelope", "SoC Derating", "Yaw Moment"],
    content: () => (<>
      <SubHead icon="◎" label="What — The Concept" color={C.cy} />
      <P>Formula Student relies heavily on intelligent powertrain deployment, especially in EV categories where energy is finite and thermal limits are real. The <F color="#ff8c00">Predictive Torque Vectoring</F> layer distributes longitudinal torque across individual wheels to induce optimal yaw moments while strictly managing the battery's thermal budget and state of charge.</P>

      <SubHead icon="∑" label="How — The Mathematics" color={C.am} />
      <InfoGrid items={[
        { title: "CONVEX SLIP OPTIMIZATION", color: "#ff8c00",
          text: <>The WMPC's yaw rate target is decomposed into per-wheel torque requests that maximise the combined friction circle at each corner. The allocation is convex — guaranteed global optimum — and accounts for the instantaneous Fz distribution, tire temperature, and slip state.</> },
        { title: "THERMAL DERATING", color: C.red,
          text: <>Battery SoC, cell temperatures, and motor casing temperatures enter the Augmented Lagrangian as hard boundary penalties. If the inverter approaches 85°C, the solver automatically reduces peak torque while redistributing to cooler motors — maintaining yaw control authority even under thermal stress.</> },
        { title: "REGEN ENVELOPE", color: C.gn,
          text: <>The regen braking envelope is a dynamic 2D polygon: motor torque limit (speed-dependent back-EMF) intersected with battery charge current limit (SoC and temperature-dependent). The WMPC maximises energy recovery on every deceleration without locking the rear axle or exceeding cell C-rate limits.</> },
      ]} />
      <MathBlock label="Yaw Moment Allocation">{"M_yaw_target = WMPC yaw demand [Nm]\nM_mechanical  = Σ Fy_i × track_arm_i\nM_active      = (T_outer − T_inner) × r_wheel / diff_ratio\n\nConstraint: |T_i| ≤ T_max(ω_motor, T_inverter, SoC)\n            Σ|T_i| ≤ P_battery_max / ω_avg"}</MathBlock>
      <AdvantageBox text={<>If the front aero stalls mid-corner, the torque vectoring can instantaneously transfer power to the outside rear wheel, <F color={C.gn}>artificially inducing oversteer</F> to keep the car on the racing line. This active safety net means the WMPC can push the aero platform harder, knowing the powertrain can compensate for transient grip loss — extracting 100% of mechanical grip without risking thermal shutdown.</>} />
    </>),
  },
];

// ═══════════════════════════════════════════════════════════════════════
// MAIN COMPONENT — Split pane layout
// ═══════════════════════════════════════════════════════════════════════

export default function OverviewModule() {
  const [active, setActive] = useState("nph");
  const pillar = PILLARS.find(p => p.key === active);

  return (
    <div style={{ display: "flex", gap: 0, minHeight: "calc(100vh - 160px)" }}>
      {/* ── LEFT NAV ─────────────────────────────────────────── */}
      <div style={{ width: 260, flexShrink: 0, borderRight: `1px solid ${C.b1}`, paddingRight: 0, overflowY: "auto" }}>
        <div style={{ padding: "0 16px 12px", marginBottom: 8 }}>
          <div style={{ fontSize: 9, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 3, textTransform: "uppercase", marginBottom: 4 }}>Architecture Pillars</div>
          <div style={{ fontSize: 11, color: C.md, fontFamily: C.bd }}>8 subsystems powering the digital twin</div>
        </div>

        {PILLARS.map((p) => {
          const isA = active === p.key;
          return (
            <button key={p.key} onClick={() => setActive(p.key)} style={{
              width: "100%", display: "flex", alignItems: "flex-start", gap: 10,
              padding: "12px 16px", border: "none", borderRadius: 0,
              borderLeft: isA ? `3px solid ${p.color}` : "3px solid transparent",
              borderBottom: `1px solid ${C.b1}`,
              background: isA ? `${p.color}08` : "transparent",
              cursor: "pointer", textAlign: "left",
              transition: "all 0.2s ease",
            }}
              onMouseEnter={e => { if (!isA) e.currentTarget.style.background = C.hover; }}
              onMouseLeave={e => { if (!isA) e.currentTarget.style.background = "transparent"; }}
            >
              <span style={{ fontSize: 18, color: isA ? p.color : C.dm, lineHeight: 1, marginTop: 2, transition: "color 0.2s" }}>{p.icon}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, fontWeight: 600 }}>{p.num}</span>
                  <span style={{ fontSize: 11.5, fontWeight: isA ? 700 : 500, color: isA ? p.color : C.br, fontFamily: C.hd, transition: "color 0.2s" }}>{p.label}</span>
                </div>
                <div style={{ fontSize: 9.5, color: C.dm, fontFamily: C.dt, marginTop: 2, letterSpacing: 0.3 }}>{p.sub}</div>
              </div>
            </button>
          );
        })}

        {/* System status at bottom */}
        <div style={{ padding: "16px", borderTop: `1px solid ${C.b1}`, marginTop: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.gn, boxShadow: `0 0 8px ${C.gn}`, animation: "pulseGlow 2s infinite" }} />
            <span style={{ fontSize: 9, fontWeight: 700, color: C.gn, fontFamily: C.dt, letterSpacing: 1.5 }}>ALL SYSTEMS NOMINAL</span>
          </div>
          {[["State Vector", "46-dim"], ["Setup Space", "28-dim"], ["Differentiable", "100%"], ["Fidelity", "90.6%"]].map(([k, v]) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 9, fontFamily: C.dt, marginBottom: 2 }}>
              <span style={{ color: C.dm }}>{k}</span>
              <span style={{ color: C.br, fontWeight: 600 }}>{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── RIGHT CONTENT ────────────────────────────────────── */}
      <div style={{ flex: 1, overflowY: "auto", padding: "0 28px 28px" }}>
        {/* Header */}
        <div style={{ position: "sticky", top: 0, zIndex: 5, background: C.bg, paddingTop: 4, paddingBottom: 16, borderBottom: `1px solid ${pillar.color}15`, marginBottom: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <span style={{ fontSize: 32, color: pillar.color, filter: `drop-shadow(0 0 10px ${pillar.color}40)` }}>{pillar.icon}</span>
            <div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                <span style={{ fontSize: 10, color: C.dm, fontFamily: C.dt, fontWeight: 600 }}>{pillar.num}</span>
                <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: C.w, fontFamily: C.hd, letterSpacing: 1 }}>{pillar.label}</h2>
              </div>
              <div style={{ fontSize: 12, color: C.md, fontFamily: C.bd, marginTop: 2 }}>{pillar.sub}</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
            {pillar.tags.map(t => <Tag key={t} color={pillar.color}>{t}</Tag>)}
          </div>
        </div>

        {/* Body */}
        {pillar.content()}
      </div>
    </div>
  );
}