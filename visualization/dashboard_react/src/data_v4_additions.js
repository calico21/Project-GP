// ═══════════════════════════════════════════════════════════════════════════════
// data.js — v4.0 ADDITIONS
// ═══════════════════════════════════════════════════════════════════════════════
//
// INTEGRATION INSTRUCTIONS:
// ─────────────────────────────────────────────────────────────────────────────
// 1. Open your existing  visualization/dashboard_react/src/data.js
//
// 2. Add these NEW seeded RNG instances alongside your existing ones at the top:
//
//      const R_TH  = sd("thermal5node_gp4");
//      const R_EN  = sd("energy_budget_gp4");
//      const R_AL  = sd("alconstraint_gp4");
//      const R_WV  = sd("wavelet_coeff_gp4");
//      const R_HN  = sd("hnet_landscape_gp4");
//      const R_RM  = sd("rmatrix_gp4");
//      const R_GP  = sd("gpenvelope_gp4");
//      const R_HY  = sd("hysteresis_gp4");
//      const R_FI  = sd("fimeigen_gp4");
//      const R_TS  = sd("truststeps_gp4");
//      const R_EP  = sd("episodes_gp4");
//      const R_FS  = sd("friction_surf_gp4");
//      const R_LS  = sd("loadsens_gp4");
//
// 3. Paste ALL the functions below at the end of data.js, before the closing.
//
// 4. Add the new function names to your exports (they're all named exports).
//
// That's it. Nothing in the existing file changes — these are pure additions.
// ═══════════════════════════════════════════════════════════════════════════════


// ─────────────────────────────────────────────────────────────────────────────
// 5-NODE TIRE THERMAL MODEL
// ─────────────────────────────────────────────────────────────────────────────
// Generates per-corner [flash, surface, tread_bulk, carcass, gas] temperature
// arrays that evolve over distance. Models:
//   - Flash temp: spikes under high slip, decays fast (Jaeger transient)
//   - Surface:    tracks flash with ~0.5s thermal lag
//   - Bulk:       slow soak-through from surface
//   - Carcass:    structural heating from hysteresis
//   - Gas:        inflation gas slowly warming from carcass conduction
//
// Returns: Array<{ s: number, fl: number[5], fr: number[5], rl: number[5], rr: number[5] }>
//
export function gThermal5(n = 200) {
  const d = [];
  // Initial temps per corner × 5 nodes  [flash, surf, bulk, carcass, gas]
  let temps = [
    [90, 82, 70, 55, 40],   // FL — slightly loaded from static camber
    [92, 84, 72, 56, 40],   // FR
    [85, 78, 66, 52, 38],   // RL — less thermal from lighter rear
    [87, 80, 68, 53, 38],   // RR
  ];

  for (let i = 0; i < n; i++) {
    // Driving intensity envelope — simulates cornering/braking events
    const intensity = 0.4 + 0.3 * Math.sin(i / 20) + 0.1 * R_TH();

    for (let c = 0; c < 4; c++) {
      // Heat generation rate per node (flash gets the most, gas the least)
      const heatIn = [
        intensity * 3.5 + R_TH() * 2,     // flash — Jaeger surface heating
        intensity * 2.0 + R_TH(),          // surface — friction + flash soak
        intensity * 1.2 + R_TH() * 0.5,    // bulk — conduction from surface
        intensity * 0.6 + R_TH() * 0.3,    // carcass — hysteresis heating
        intensity * 0.15 + R_TH() * 0.1,   // gas — slow conduction from carcass
      ];
      // Cooling rate proportional to (T - T_ambient) with node-specific time constant
      const coolRate = [0.15, 0.08, 0.04, 0.02, 0.005];

      for (let j = 0; j < 5; j++) {
        temps[c][j] = Math.min(165, Math.max(25,
          temps[c][j] + heatIn[j] - coolRate[j] * (temps[c][j] - 30)
        ));
      }
    }

    d.push({
      s: i,
      fl: [...temps[0]],
      fr: [...temps[1]],
      rl: [...temps[2]],
      rr: [...temps[3]],
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// PORT-HAMILTONIAN ENERGY BUDGET
// ─────────────────────────────────────────────────────────────────────────────
// Tracks the Hamiltonian decomposition over time:
//   H(q,p) = T(p) + V_spring(q) + V_arb(q) + H_net_residual(q)
// Plus the dissipation rate from R_net and the time-derivative dH/dt.
//
// Returns: Array<{ t, ke, pe_s, pe_arb, h_net, H, dH, r_diss }>
//
export function gEnergyBudget(n = 300) {
  const d = [];
  let ke = 4500, pe_s = 120, pe_arb = 35, h_net = 15, diss_cum = 0;

  for (let i = 0; i < n; i++) {
    const dt = 0.02;
    // Stochastic perturbations simulating vehicle dynamics
    ke += (-50 + 100 * R_EN()) * dt;
    pe_s += (-8 + 16 * R_EN()) * dt;
    pe_arb += (-3 + 6 * R_EN()) * dt;

    // H_net injection — should be near-zero for a well-trained network.
    // Positive means phantom energy (passivity violation).
    const h_inj = Math.max(0, 2 * (R_EN() - 0.6));  // biased slightly positive to show the warning
    h_net += h_inj * dt;

    // Dissipation from R_net (always positive — guaranteed by Cholesky PSD structure)
    const r_diss = 8 + 12 * R_EN();
    diss_cum += r_diss * dt;

    const H = ke + pe_s + pe_arb + h_net;
    const dH = (-50 + 100 * R_EN()) * dt + h_inj - r_diss;

    d.push({
      t: +(i * dt).toFixed(3),
      ke: +ke.toFixed(0),
      pe_s: +pe_s.toFixed(0),
      pe_arb: +pe_arb.toFixed(0),
      h_net: +h_net.toFixed(1),
      H: +H.toFixed(0),
      dH: +dH.toFixed(1),
      r_diss: +r_diss.toFixed(1),
      diss_cum: +diss_cum.toFixed(0),
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// AUGMENTED LAGRANGIAN CONSTRAINT MONITOR
// ─────────────────────────────────────────────────────────────────────────────
// Per-solve snapshot of the 5 AL constraint slacks, multiplier λ, and
// iteration count. Simulates the solver converging over time as AL
// multipliers tighten the feasibility envelope.
//
// Returns: Array<{ solve, grip, steer, ax, track, vx, iters, lambda_grip }>
//
export function gALConstraints(n = 200) {
  const d = [];
  for (let i = 0; i < n; i++) {
    d.push({
      solve: i,
      // Constraint slacks — exponential decay toward binding
      grip:  +(0.20 * Math.exp(-i / 80) + 0.05 * R_AL()).toFixed(3),
      steer: +(0.15 + 0.05 * R_AL()).toFixed(3),
      ax:    +(0.10 + 0.08 * R_AL()).toFixed(3),
      track: +(0.30 * Math.exp(-i / 100) + 0.03 * R_AL()).toFixed(3),
      vx:    +(0.20 + 0.06 * R_AL()).toFixed(3),
      // Solver diagnostics
      iters: Math.round(3 + 5 * R_AL()),
      lambda_grip: +(1 + 4 * (1 - Math.exp(-i / 60)) + 0.5 * R_AL()).toFixed(2),
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// WAVELET COEFFICIENT DECOMPOSITION
// ─────────────────────────────────────────────────────────────────────────────
// Db4 3-level DWT of the 3 control channels (steer, throttle, brake).
// Produces coefficients at each level: cA3 (8 coeffs), cD3 (8), cD2 (16), cD1 (32).
// These map to the 64-step MPC horizon via inverse DWT.
//
// Returns: Array<{ ctl, level, idx, mag }>
//
export function gWaveletCoeffs() {
  const d = [];
  const ctls = ["steer", "throttle", "brake"];
  const levels = ["cA3", "cD3", "cD2", "cD1"];
  const sizes = [8, 8, 16, 32];

  for (let c = 0; c < 3; c++) {
    for (let l = 0; l < 4; l++) {
      for (let k = 0; k < sizes[l]; k++) {
        // Approximation (cA3) has larger magnitude; details decay with level
        const baseMag = l === 0 ? 0.3 : 0.15 / (l + 1);
        d.push({
          ctl: ctls[c],
          level: levels[l],
          idx: k,
          mag: +(baseMag * (R_WV() - 0.5) * 2).toFixed(4),
        });
      }
    }
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// STOCHASTIC TUBE BOUNDARIES
// ─────────────────────────────────────────────────────────────────────────────
// Computes UT 2σ safety tube around the predicted trajectory for the track map.
// Returns per-point inner/outer boundary coordinates offset along the track normal.
//
// Requires: track from gTK()
// Returns: Array<{ x, y, inX, inY, outX, outY, margin }>
//
export function gTubePoints(track) {
  const pts = [];
  for (let i = 0; i < track.length; i++) {
    const p = track[i];
    // Tube half-width: wider in corners (higher curvature → more uncertainty)
    const halfWidth = 1.8 + 1.2 * Math.abs(p.curvature) * 25 + 0.4 * R_WV();

    // Compute track normal direction
    let dx = 0, dy = 1;
    if (i < track.length - 1) {
      dx = +track[i + 1].x - +p.x;
      dy = +track[i + 1].y - +p.y;
    } else if (i > 0) {
      dx = +p.x - +track[i - 1].x;
      dy = +p.y - +track[i - 1].y;
    }
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    // Normal is perpendicular to tangent (rotated 90°)
    const nx = -dy / len;
    const ny = dx / len;

    pts.push({
      x: +p.x,
      y: +p.y,
      inX:  +(+p.x + nx * halfWidth).toFixed(2),
      inY:  +(+p.y + ny * halfWidth).toFixed(2),
      outX: +(+p.x - nx * halfWidth).toFixed(2),
      outY: +(+p.y - ny * halfWidth).toFixed(2),
      margin: +(4 - halfWidth).toFixed(2),   // positive = safe, negative = violation
    });
  }
  return pts;
}


// ─────────────────────────────────────────────────────────────────────────────
// H_NET ENERGY LANDSCAPE (2D grid for contour plot)
// ─────────────────────────────────────────────────────────────────────────────
// Evaluates H_net(q_front, q_rear) over a grid of suspension deflections
// with all other DOFs fixed at equilibrium. Produces a 2D scalar field.
//
// Returns: Array<{ i, j, qf, qr, H }>    (res×res entries)
//
export function gHnetLandscape(res = 40) {
  const d = [];
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      // Deflection range ±20mm around equilibrium (12.8mm front, 14.2mm rear)
      const qf = (i / res - 0.5) * 40;   // mm
      const qr = (j / res - 0.5) * 40;   // mm

      // Synthetic energy landscape: quadratic bowl + off-center bump + noise
      const r2 = qf * qf + qr * qr;
      const H = 50 * Math.exp(-r2 / 400)
              + 20 * Math.exp(-((qf - 10) ** 2) / 100 - ((qr + 5) ** 2) / 100)
              + 3 * (R_HN() - 0.5);

      d.push({ i, j, qf: +qf.toFixed(1), qr: +qr.toFixed(1), H: +H.toFixed(2) });
    }
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// R_NET DISSIPATION MATRIX (14×14 symmetric PSD)
// ─────────────────────────────────────────────────────────────────────────────
// The dissipation matrix from R_net's Cholesky factorization: R = LLᵀ + diag(softplus(d)).
// Diagonal entries are the direct damping per DOF. Off-diagonal entries are cross-coupling.
//
// Returns: number[][] (14×14)
//
export const R_DOF_LABELS = [
  "x", "y", "z", "φ", "θ", "ψ",
  "z_fl", "z_fr", "z_rl", "z_rr",
  "φ_wfl", "φ_wfr", "φ_wrl", "φ_wrr",
];

export function gRMatrix() {
  const m = [];
  for (let i = 0; i < 14; i++) {
    const row = [];
    for (let j = 0; j < 14; j++) {
      if (i === j) {
        // Diagonal: positive definite (damper-like)
        row.push(+(5 + 15 * R_RM()).toFixed(2));
      } else if (j > i) {
        // Upper triangle: cross-coupling (can be negative, but small)
        row.push(+((R_RM() - 0.5) * 3).toFixed(2));
      } else {
        row.push(0);  // placeholder — filled by symmetry below
      }
    }
    m.push(row);
  }
  // Enforce symmetry
  for (let i = 0; i < 14; i++) {
    for (let j = 0; j < i; j++) {
      m[i][j] = m[j][i];
    }
  }
  return m;
}


// ─────────────────────────────────────────────────────────────────────────────
// SPARSE GP UNCERTAINTY ENVELOPE
// ─────────────────────────────────────────────────────────────────────────────
// Matérn 5/2 posterior mean ± 2σ over the slip angle range.
// Narrow band where inducing points are dense; wide in extrapolation zones.
//
// Returns: Array<{ alpha, mu, upper, lower, sigma }>
//
export function gGPEnvelope(n = 100) {
  const d = [];
  for (let i = 0; i < n; i++) {
    const alpha = (i / n - 0.5) * 30;   // degrees
    const mu = 3000 * Math.sin(1.4 * Math.atan(12 * (alpha * Math.PI / 180)));
    // Uncertainty wider at extremes (fewer inducing points)
    const sigma = 150 + 200 * Math.abs(alpha) / 15 + 100 * R_GP();

    d.push({
      alpha: +alpha.toFixed(1),
      mu: +mu.toFixed(0),
      upper: +(mu + 2 * sigma).toFixed(0),
      lower: +(mu - 2 * sigma).toFixed(0),
      sigma: +sigma.toFixed(0),
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// SLIP-FORCE HYSTERESIS (Lissajous trace)
// ─────────────────────────────────────────────────────────────────────────────
// Sinusoidal slip angle input → transient Fy response with relaxation length.
// The loop width reveals the pneumatic trail and transient slip dynamics.
//
// Returns: Array<{ t, alpha, Fy, Fy_ss }>
//
export function gHysteresis(n = 200) {
  const d = [];
  let Fy = 0;

  for (let i = 0; i < n; i++) {
    const t = i * 0.03;
    const alpha = 8 * Math.sin(t * 2);  // sinusoidal sweep ±8°

    // Steady-state Pacejka response
    const Fy_ss = 3000 * Math.sin(1.4 * Math.atan(12 * (alpha * Math.PI / 180)));

    // First-order transient lag (relaxation length ≈ 0.15m at 15m/s → τ ≈ 0.01s)
    const tau = 0.05;
    const dt = 0.03;
    Fy += (Fy_ss - Fy) * Math.min(1, dt / tau);

    d.push({
      t: +t.toFixed(2),
      alpha: +alpha.toFixed(2),
      Fy: +Fy.toFixed(0),
      Fy_ss: +Fy_ss.toFixed(0),
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// LOAD SENSITIVITY CURVES
// ─────────────────────────────────────────────────────────────────────────────
// Peak friction μ and cornering stiffness Cα as functions of vertical load Fz.
// Classic Pacejka PDY1/PDY2 degressive characteristic.
//
// Returns: Array<{ Fz, mu, Ca }>
//
export function gLoadSensitivity(n = 50) {
  const d = [];
  for (let i = 0; i < n; i++) {
    const Fz = i * 60;   // 0..3000 N
    const mu = 1.65 - 0.35 * (Fz / 3000);                    // degressive
    const Ca = 80 + 40 * (1 - Math.exp(-Fz / 800));           // saturating
    d.push({ Fz, mu: +mu.toFixed(3), Ca: +Ca.toFixed(1) });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// COMBINED SLIP FRICTION SURFACE
// ─────────────────────────────────────────────────────────────────────────────
// Evaluates |F|(α, κ) over a 2D grid for the friction carpet heatmap.
// Uses simplified Pacejka combined slip: Fy(α)*(1-|κ|) and Fx(κ)*(1-|α|).
//
// Returns: Array<{ ai, ki, alpha, kappa, Fy, Fx, F }>
//
export function gFrictionSurface(res = 30) {
  const d = [];
  for (let ai = 0; ai < res; ai++) {
    for (let ki = 0; ki < res; ki++) {
      const alpha = (ai / res - 0.5) * 30;    // degrees
      const kappa = (ki / res - 0.5) * 0.6;   // ratio

      const Fy = 3000 * Math.sin(1.4 * Math.atan(12 * (alpha * Math.PI / 180)))
               * (1 - 0.4 * Math.abs(kappa));
      const Fx = 3200 * Math.sin(1.6 * Math.atan(10 * kappa))
               * (1 - 0.5 * Math.abs(alpha * Math.PI / 180));
      const F = Math.sqrt(Fx * Fx + Fy * Fy);

      d.push({
        ai, ki,
        alpha: +alpha.toFixed(1),
        kappa: +kappa.toFixed(3),
        Fy: +Fy.toFixed(0),
        Fx: +Fx.toFixed(0),
        F: +F.toFixed(0),
      });
    }
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// FIM EIGENSPECTRUM
// ─────────────────────────────────────────────────────────────────────────────
// Top eigenvalues of the Fisher Information Matrix from TRPO natural gradient.
// Steep eigenvalues = directions where policy is sensitive.
// Flat tail = directions the optimizer can move freely.
//
// Returns: Array<{ idx, eigenval, label }>
//
export function gFIMEigen() {
  return Array.from({ length: 14 }, (_, i) => ({
    idx: i,
    eigenval: +(50 * Math.exp(-i * 0.5) + 2 * R_FI()).toFixed(2),
    label: PN[i],
  }));
}


// ─────────────────────────────────────────────────────────────────────────────
// TRUST REGION STEP MAP
// ─────────────────────────────────────────────────────────────────────────────
// Projected policy updates onto the top-2 FIM eigenvectors.
// Each step is either accepted (KL ≤ δ) or rejected (KL > δ).
//
// Returns: Array<{ step, x, y, dx, dy, accepted, kl }>
//
export function gTrustSteps(n = 60) {
  const d = [];
  let x = 0, y = 0;

  for (let i = 0; i < n; i++) {
    const dx = (R_TS() - 0.5) * 0.4;
    const dy = (R_TS() - 0.5) * 0.4;
    const kl = 0.005 + 0.01 * R_TS();
    const accepted = kl < 0.0094;   // δ = 0.0094 is the trust region bound

    if (accepted) { x += dx; y += dy; }

    d.push({
      step: i,
      x: +x.toFixed(3), y: +y.toFixed(3),
      dx: +dx.toFixed(3), dy: +dy.toFixed(3),
      accepted,
      kl: +kl.toFixed(4),
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// MORL EPISODE REWARDS
// ─────────────────────────────────────────────────────────────────────────────
// Per-episode reward curves for the multi-objective RL optimizer.
// Three channels: grip reward, stability reward, combined scalarization.
//
// Returns: Array<{ ep, grip, stab, combined }>
//
export function gEpisodes(n = 150) {
  const d = [];
  for (let i = 0; i < n; i++) {
    d.push({
      ep: i,
      grip:     +(0.8 + 0.5 * (1 - Math.exp(-i / 40)) + 0.08 * (R_EP() - 0.5)).toFixed(3),
      stab:     +(2.0 + 1.5 * (1 - Math.exp(-i / 50)) + 0.10 * (R_EP() - 0.5)).toFixed(3),
      combined: +(1.2 + 0.8 * (1 - Math.exp(-i / 45)) + 0.06 * (R_EP() - 0.5)).toFixed(3),
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// EXTENDED CONVERGENCE DATA (adds pgNorm, advMean, hvol to existing gCV)
// ─────────────────────────────────────────────────────────────────────────────
// NOTE: Your existing gCV already produces iter, bestGrip, kl, entropy.
// This is a REPLACEMENT that adds three new fields per row.
// If you prefer not to replace gCV, use this as gCV4() instead.
//
export function gCV4(n = 200) {
  const RCV = sd("convergence_gp4");
  const d = [];
  let b = 1.1;
  for (let i = 0; i < n; i++) {
    b = Math.max(b, 1.1 + 0.45 * (1 - Math.exp(-i / 40)) + 0.02 * RCV());
    d.push({
      iter:     i,
      bestGrip: +b.toFixed(4),
      kl:       +(0.012 * Math.exp(-i / 80) + 0.001 * RCV()).toFixed(5),
      entropy:  +(-1 + 0.3 * Math.exp(-i / 60) + 0.05 * RCV()).toFixed(3),
      // v4.0 additions:
      pgNorm:   +(2 * Math.exp(-i / 50) + 0.1 * RCV()).toFixed(3),
      advMean:  +(0.1 * Math.exp(-i / 30) + 0.02 * (RCV() - 0.5)).toFixed(4),
      hvol:     +(0.3 + 0.5 * (1 - Math.exp(-i / 60)) + 0.02 * RCV()).toFixed(4),
    });
  }
  return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// EXTENDED SENSITIVITY (adds dTherm to existing gSN)
// ─────────────────────────────────────────────────────────────────────────────
// NOTE: Your existing gSN returns { param, dGrip, dStab }.
// This adds dTherm and uses all 28 params instead of just the first 14.
//
export function gSN4() {
  const RSN = sd("sensitivity_gp4");
  return PN.map((p, i) => ({
    param:  p,
    unit:   PU[i],
    dGrip:  +((RSN() - 0.3) * 0.6).toFixed(3),
    dStab:  +((RSN() - 0.4) * 0.4).toFixed(3),
    dTherm: +((RSN() - 0.5) * 0.2).toFixed(3),
  }));
}


// ─────────────────────────────────────────────────────────────────────────────
// EXTENDED PARETO (adds hvContrib to existing gP)
// ─────────────────────────────────────────────────────────────────────────────
// NOTE: Your existing gP returns { grip, stability, gen, omega, params }.
// This adds id and hvContrib for the interactive Pareto explorer.
//
export function gP4(n = 52) {
  const RP = sd("pareto_gp4");
  const p = [];
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos(i * Math.PI / (n - 1)));
    p.push({
      id: i,
      grip:     +(1.25 + 0.3 * w + 0.04 * (RP() - 0.5)).toFixed(4),
      stability:+(0.5 + 2.8 * (1 - w) + 0.15 * (RP() - 0.5)).toFixed(2),
      gen:      Math.floor(RP() * 200),
      omega:    +w.toFixed(3),
      hvContrib:+(0.01 + 0.04 * RP()).toFixed(4),
      params:   PN.map(() => RP()),
    });
  }
  return p.sort((a, b) => a.grip - b.grip);
}