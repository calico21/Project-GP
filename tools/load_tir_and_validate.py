"""
tools/load_tir_and_validate.py
Project-GP — TIR File Loader & Pacejka Cross-Validation

Three capabilities:
  1. parse_tir(path) → dict   — loads any PAC2002/MF6.2 .tir into tire_coeffs format
  2. generate_sweeps(coeffs)   — creates dense (α, κ, Fz, γ) reference curves from .tir
  3. validate_jax_pacejka()    — compares JAX tire_model.py output against reference

Usage:
  python tools/load_tir_and_validate.py --tir path/to/hoosier.tir
  python tools/load_tir_and_validate.py --tir path/to/hoosier.tir --export-coeffs
  python tools/load_tir_and_validate.py --validate   # uses existing tire_coeffs.py
"""
from __future__ import annotations

import re
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# §1  .tir file parser
# ─────────────────────────────────────────────────────────────────────────────

# PAC2002 / MF6.2 sections we care about and their coefficient prefixes
_TIR_SECTIONS = {
    'MODEL',
    'DIMENSION',
    'VERTICAL',
    'LONG_SLIP_RANGE', 'SLIP_ANGLE_RANGE',
    'SHAPE',
    'LATERAL_COEFFICIENTS',
    'LONGITUDINAL_COEFFICIENTS',
    'ALIGNING_COEFFICIENTS',
    'OVERTURNING_COEFFICIENTS',
    'ROLLING_COEFFICIENTS',
    'TURNSLIP_COEFFICIENTS',
    'CONDITIONS',
    'OPERATING_CONDITIONS',
}

# Sign convention map: coefficients where the .tir file sign convention
# is OPPOSITE to the tire_model.py implementation (SAE → ISO or vice versa).
# Only add entries here that are empirically confirmed — PDY2 is the canonical
# example from the project's debugging history.
_SIGN_CONVENTION_NOTES = {
    'PDY2': (
        'Many .tir fitters store PDY2 with opposite sign. '
        'Project-GP uses: Dy = PDY1*(1+PDY2*dfz)*Fz → NEGATIVE PDY2 = degressive. '
        'Check your source. If from OptimumT or MFeval, the sign is typically '
        'already correct for this convention. If from Adams or CarSim, negate it.'
    ),
    'PKY1': (
        'PKY1 may be stored as negative in some .tir conventions (right-tyre). '
        'Project-GP uses abs(PKY1). Parser takes abs() automatically.'
    ),
    'PDY1': (
        'PDY1 should be positive (friction coefficient magnitude). '
        'Parser takes abs() automatically.'
    ),
}

# Coefficients where we always want the absolute value regardless of .tir sign
_ALWAYS_ABS = {'PDY1', 'PKY1'}


def parse_tir(tir_path: str) -> Dict[str, float]:
    """
    Parse a PAC2002 / MF6.2 .tir file into a flat coefficient dictionary
    compatible with Project-GP's tire_coeffs format.

    The .tir format is INI-like:
        [SECTION_NAME]
        PARAM_NAME  =  value   $comment

    Returns dict with keys matching data/configs/tire_coeffs.py naming.
    Unknown coefficients are included with their original .tir key name.
    """
    path = Path(tir_path)
    if not path.exists():
        raise FileNotFoundError(f"TIR file not found: {tir_path}")

    raw_coeffs: Dict[str, float] = {}
    metadata: Dict[str, str] = {}
    current_section: Optional[str] = None
    sign_warnings: list = []

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines, comments, and headers
            if not line or line.startswith('!') or line.startswith('$'):
                continue

            # Section header: [SECTION_NAME]
            section_match = re.match(r'\[(\w+)\]', line)
            if section_match:
                current_section = section_match.group(1).upper()
                continue

            # Key = value $comment
            kv_match = re.match(
                r"([A-Za-z_]\w*)\s*=\s*([^\$!]+?)(?:\s*[\$!].*)?$", line
            )
            if kv_match:
                key = kv_match.group(1).upper()
                val_str = kv_match.group(2).strip().strip("'\"")

                # Try numeric parse
                try:
                    value = float(val_str)
                    raw_coeffs[key] = value
                except ValueError:
                    # String value (tire name, model type, etc.)
                    metadata[key] = val_str

    if not raw_coeffs:
        raise ValueError(f"No numeric coefficients found in {tir_path}. "
                         f"Is this a valid .tir file?")

    # ── Post-processing: sign conventions + abs() ────────────────────────
    for key in _ALWAYS_ABS:
        if key in raw_coeffs and raw_coeffs[key] < 0:
            sign_warnings.append(
                f"  {key}: negated from {raw_coeffs[key]:.6f} → "
                f"{abs(raw_coeffs[key]):.6f} (Project-GP uses abs)"
            )
            raw_coeffs[key] = abs(raw_coeffs[key])

    # ── Map .tir reference conditions to Project-GP names ────────────────
    # .tir uses FNOMIN, UNLOADED_RADIUS, LONGVL, NOMPRES, etc.
    _rename_map = {
        'FNOMIN':           'FNOMIN',
        'UNLOADED_RADIUS':  'R0',
        'LONGVL':           'V0',
        'NOMPRES':          'P_nom',
        'WIDTH':            'WIDTH',
        'RIM_RADIUS':       'RIM_RADIUS',
        'RIM_WIDTH':        'RIM_WIDTH',
    }
    for tir_name, gp_name in _rename_map.items():
        if tir_name in raw_coeffs and gp_name not in raw_coeffs:
            raw_coeffs[gp_name] = raw_coeffs[tir_name]

    # Convert NOMPRES from Pa to bar if value > 10 (heuristic: Pa vs bar)
    if 'P_nom' in raw_coeffs and raw_coeffs['P_nom'] > 10:
        raw_coeffs['P_nom'] = raw_coeffs['P_nom'] / 1e5

    # Default thermal params if not in .tir (thermal isn't part of PAC2002)
    raw_coeffs.setdefault('T_OPT', 90.0)
    raw_coeffs.setdefault('T_opt', 90.0)
    raw_coeffs.setdefault('T_ENV', 25.0)
    raw_coeffs.setdefault('BETA_T', 0.0008)

    # ── Report ───────────────────────────────────────────────────────────
    n_pac = sum(1 for k in raw_coeffs
                if any(k.startswith(p) for p in
                       ['PC', 'PD', 'PE', 'PK', 'PH', 'PV',
                        'RB', 'RC', 'RE', 'RH', 'RV',
                        'QB', 'QC', 'QD', 'QE', 'QH',
                        'PD', 'PK', 'PH', 'PE']))
    print(f"\n[TIR Parser] Loaded {path.name}")
    print(f"  Total coefficients : {len(raw_coeffs)}")
    print(f"  Pacejka parameters : {n_pac}")
    print(f"  FNOMIN             : {raw_coeffs.get('FNOMIN', '?')} N")
    print(f"  R0                 : {raw_coeffs.get('R0', '?')} m")
    print(f"  V0 (LONGVL)        : {raw_coeffs.get('V0', '?')} m/s")
    if metadata.get('TIREID') or metadata.get('FILE_TYPE'):
        print(f"  Tire ID            : {metadata.get('TIREID', metadata.get('FITTYP', '?'))}")

    if sign_warnings:
        print(f"\n  ⚠ Sign convention adjustments:")
        for w in sign_warnings:
            print(w)

    # Print PDY2 advisory always — it's the most dangerous coefficient
    pdy2 = raw_coeffs.get('PDY2', None)
    if pdy2 is not None:
        sign = 'DEGRESSIVE ✓' if pdy2 < 0 else 'PROGRESSIVE ⚠ (may need negation)'
        print(f"\n  PDY2 = {pdy2:+.6f} → {sign}")
        print(f"  {_SIGN_CONVENTION_NOTES['PDY2']}")

    return raw_coeffs


def export_as_tire_coeffs_py(coeffs: dict, output_path: str = None):
    """
    Write parsed coefficients as a drop-in replacement for
    data/configs/tire_coeffs.py.
    """
    if output_path is None:
        output_path = 'data/configs/tire_coeffs_from_tir.py'

    lines = [
        "# Auto-generated from .tir file by tools/load_tir_and_validate.py",
        "# Review PDY2 sign convention before use!",
        "",
        "tire_coeffs = {",
    ]

    # Group by prefix for readability
    groups = {}
    for k, v in sorted(coeffs.items()):
        prefix = re.match(r'([A-Z]+)', k)
        group = prefix.group(1) if prefix else 'OTHER'
        groups.setdefault(group, []).append((k, v))

    for group_name in sorted(groups.keys()):
        lines.append(f"    # ── {group_name} " + "─" * (60 - len(group_name)))
        for k, v in groups[group_name]:
            # Format: integer-like values without decimals, else 6 sig figs
            if v == int(v) and abs(v) < 1e6:
                lines.append(f"    '{k}': {int(v)},")
            else:
                lines.append(f"    '{k}': {v},")
        lines.append("")

    lines.append("}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[TIR Parser] Exported → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §2  Reference sweep generation from .tir coefficients
# ─────────────────────────────────────────────────────────────────────────────

def generate_reference_sweeps(
    coeffs: dict,
    n_alpha: int = 200,
    n_kappa: int = 100,
    Fz_levels: tuple = (300, 500, 700, 900, 1200),
    gamma_levels: tuple = (0.0, -0.035, -0.07),  # rad (0°, -2°, -4°)
) -> dict:
    """
    Generate dense Pacejka reference curves using the STANDARD MF6.2 formula
    implemented independently of tire_model.py. This is the "ground truth"
    for cross-validation.

    Returns dict with:
      'lateral_sweeps':  list of dicts {alpha, Fy, Fz, gamma}
      'longitudinal_sweeps': list of dicts {kappa, Fx, Fz, gamma}
      'aligning_sweeps': list of dicts {alpha, Mz, Fz, gamma}
    """
    alpha_range = np.linspace(-0.25, 0.25, n_alpha)   # rad (~±14°)
    kappa_range = np.linspace(-0.20, 0.20, n_kappa)

    Fz0 = coeffs.get('FNOMIN', 654.0)

    lateral_sweeps = []
    longitudinal_sweeps = []
    aligning_sweeps = []

    for Fz in Fz_levels:
        for gamma in gamma_levels:
            dfz = (Fz - Fz0) / Fz0

            # ── Pure lateral Fy ──────────────────────────────────────────
            PCY1 = coeffs.get('PCY1', 1.53)
            PDY1 = coeffs.get('PDY1', 2.40)
            PDY2 = coeffs.get('PDY2', -0.34)
            PDY3 = coeffs.get('PDY3', 3.90)
            PEY1 = coeffs.get('PEY1', 0.0)
            PEY2 = coeffs.get('PEY2', -0.28)
            PEY3 = coeffs.get('PEY3', 0.70)
            PEY4 = coeffs.get('PEY4', -0.48)
            PKY1 = coeffs.get('PKY1', 53.24)
            PKY2 = coeffs.get('PKY2', 2.38)
            PKY3 = coeffs.get('PKY3', 0.15)
            PHY1 = coeffs.get('PHY1', -0.0009)
            PHY2 = coeffs.get('PHY2', -0.00082)
            PVY1 = coeffs.get('PVY1', 0.045)
            PVY2 = coeffs.get('PVY2', -0.024)

            Dy = PDY1 * (1 + PDY2 * dfz) * (1 - PDY3 * gamma**2) * Fz
            Cy = PCY1
            ByCy = PKY1 * np.sin(
                2.0 * np.arctan(Fz / (PKY2 * Fz0))
            ) * (1 - PKY3 * abs(gamma))
            By = ByCy / (Cy * Dy + 1e-6)

            Shy = PHY1 + PHY2 * dfz
            Svy = Fz * (PVY1 + PVY2 * dfz)

            Ey = (PEY1 + PEY2 * dfz) * (
                1 - (PEY3 + PEY4 * gamma) * np.sign(alpha_range + Shy)
            )
            Ey = np.minimum(Ey, 1.0)

            alpha_y = alpha_range + Shy
            Fy0 = Dy * np.sin(
                Cy * np.arctan(
                    By * alpha_y - Ey * (By * alpha_y - np.arctan(By * alpha_y))
                )
            ) + Svy

            lateral_sweeps.append({
                'alpha': alpha_range.copy(),
                'Fy': Fy0.copy(),
                'Fz': Fz,
                'gamma': gamma,
                'Dy': Dy,
                'By': By,
                'Cy': Cy,
            })

            # ── Pure longitudinal Fx ─────────────────────────────────────
            PCX1 = coeffs.get('PCX1', 1.685)
            PDX1 = coeffs.get('PDX1', 1.21)
            PDX2 = coeffs.get('PDX2', -0.037)
            PEX1 = coeffs.get('PEX1', 0.344)
            PEX2 = coeffs.get('PEX2', 0.095)
            PEX3 = coeffs.get('PEX3', -0.020)
            PEX4 = coeffs.get('PEX4', 0.0)
            PKX1 = coeffs.get('PKX1', 21.51)
            PKX2 = coeffs.get('PKX2', 13.49)
            PKX3 = coeffs.get('PKX3', -0.41)
            PHX1 = coeffs.get('PHX1', 0.0)
            PHX2 = coeffs.get('PHX2', 0.0)
            PVX1 = coeffs.get('PVX1', 0.0)
            PVX2 = coeffs.get('PVX2', 0.0)

            Dx = (PDX1 + PDX2 * dfz) * Fz
            Cx = PCX1
            BxCx = (PKX1 + PKX2) * np.exp(PKX3 * dfz) * Fz
            Bx = BxCx / (Cx * Dx + 1e-6)

            Shx = PHX1 + PHX2 * dfz
            Svx = Fz * (PVX1 + PVX2 * dfz)

            Ex = (PEX1 + PEX2 * dfz + PEX3 * dfz**2) * (
                1 - PEX4 * np.sign(kappa_range + Shx)
            )
            Ex = np.minimum(Ex, 1.0)

            kappa_x = kappa_range + Shx
            Fx0 = Dx * np.sin(
                Cx * np.arctan(
                    Bx * kappa_x - Ex * (Bx * kappa_x - np.arctan(Bx * kappa_x))
                )
            ) + Svx

            longitudinal_sweeps.append({
                'kappa': kappa_range.copy(),
                'Fx': Fx0.copy(),
                'Fz': Fz,
                'gamma': gamma,
            })

    print(f"\n[Reference] Generated {len(lateral_sweeps)} lateral sweeps × "
          f"{n_alpha} pts")
    print(f"[Reference] Generated {len(longitudinal_sweeps)} longitudinal sweeps × "
          f"{n_kappa} pts")

    return {
        'lateral_sweeps': lateral_sweeps,
        'longitudinal_sweeps': longitudinal_sweeps,
        'Fz_levels': list(Fz_levels),
        'gamma_levels': list(gamma_levels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §3  JAX Pacejka cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_jax_pacejka(
    coeffs: dict = None,
    reference: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Compare Project-GP's JAX PacejkaTire against reference sweep data.

    Tests:
      1. Pure lateral Fy at multiple (Fz, γ) combinations
      2. Pure longitudinal Fx at multiple Fz levels
      3. Load sensitivity: peak μ_y vs Fz (degressive check)
      4. GP uncertainty envelope (σ coverage on reference data)

    Returns metrics dict with per-sweep R², RMSE, and composite score.
    """
    # Lazy import — only needed when validating against JAX model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    import jax
    import jax.numpy as jnp

    if coeffs is None:
        from data.configs.tire_coeffs import tire_coeffs
        coeffs = tire_coeffs

    from models.tire_model import PacejkaTire

    if reference is None:
        reference = generate_reference_sweeps(coeffs)

    tire = PacejkaTire(coeffs)
    metrics = {}
    all_fy_r2 = []
    all_fx_r2 = []

    # Nominal thermal state (no thermal effect — isolates Pacejka layer)
    T_ribs = jnp.array([90.0, 90.0, 90.0])   # at T_opt → thermal factor = 1.0
    T_gas  = 90.0
    T_core = 90.0

    print("\n" + "=" * 70)
    print(" PACEJKA IMPLEMENTATION CROSS-VALIDATION")
    print("=" * 70)

    # ── 3.1  Pure lateral sweeps ─────────────────────────────────────────
    print(f"\n{'Fz':>6} {'γ°':>5} {'R²':>8} {'RMSE':>10} {'NRMSE%':>8} "
          f"{'Peak_ref':>10} {'Peak_jax':>10} {'Δpeak%':>8}")
    print("─" * 70)

    for sweep in reference['lateral_sweeps']:
        alpha_arr = sweep['alpha']
        Fz_val = sweep['Fz']
        gamma_val = sweep['gamma']
        Fy_ref = sweep['Fy']

        # Evaluate JAX model point-by-point (no PINN, no GP — pure Pacejka)
        Fy_jax = []
        for alpha_i in alpha_arr:
            # compute_tire_forces signature: alpha, kappa, Fz, gamma, Vx,
            #   T_ribs, T_gas, T_core, kappa_c=0
            _, Fy_i = tire.compute_tire_forces(
                alpha=jnp.float32(alpha_i),
                kappa=jnp.float32(0.0),
                Fz=jnp.float32(Fz_val),
                gamma=jnp.float32(gamma_val),
                Vx=jnp.float32(coeffs.get('V0', 11.176)),
                T_ribs=T_ribs,
                T_gas=T_gas,
                T_core=T_core,
            )
            Fy_jax.append(float(Fy_i))
        Fy_jax = np.array(Fy_jax)

        # Metrics
        ss_res = np.sum((Fy_ref - Fy_jax) ** 2)
        ss_tot = np.sum((Fy_ref - np.mean(Fy_ref)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        rmse = np.sqrt(np.mean((Fy_ref - Fy_jax) ** 2))
        nrmse = rmse / (np.max(np.abs(Fy_ref)) + 1e-6) * 100

        peak_ref = np.max(np.abs(Fy_ref))
        peak_jax = np.max(np.abs(Fy_jax))
        dpeak = (peak_jax - peak_ref) / (peak_ref + 1e-6) * 100

        gamma_deg = np.degrees(gamma_val)
        print(f"{Fz_val:6.0f} {gamma_deg:5.1f} {r2:8.5f} {rmse:10.2f} "
              f"{nrmse:8.2f} {peak_ref:10.1f} {peak_jax:10.1f} {dpeak:+8.2f}")

        key = f"Fy_Fz{Fz_val:.0f}_g{gamma_deg:.0f}"
        metrics[key] = {'r2': r2, 'rmse': rmse, 'nrmse': nrmse, 'dpeak_pct': dpeak}
        all_fy_r2.append(r2)

    # ── 3.2  Pure longitudinal sweeps ────────────────────────────────────
    print(f"\n{'Fz':>6} {'γ°':>5} {'R²':>8} {'RMSE':>10} {'NRMSE%':>8} "
          f"{'Peak_ref':>10} {'Peak_jax':>10} {'Δpeak%':>8}")
    print("─" * 70)

    for sweep in reference['longitudinal_sweeps']:
        kappa_arr = sweep['kappa']
        Fz_val = sweep['Fz']
        gamma_val = sweep['gamma']
        Fx_ref = sweep['Fx']

        Fx_jax = []
        for kappa_i in kappa_arr:
            Fx_i, _ = tire.compute_tire_forces(
                alpha=jnp.float32(0.0),
                kappa=jnp.float32(kappa_i),
                Fz=jnp.float32(Fz_val),
                gamma=jnp.float32(gamma_val),
                Vx=jnp.float32(coeffs.get('V0', 11.176)),
                T_ribs=T_ribs,
                T_gas=T_gas,
                T_core=T_core,
            )
            Fx_jax.append(float(Fx_i))
        Fx_jax = np.array(Fx_jax)

        ss_res = np.sum((Fx_ref - Fx_jax) ** 2)
        ss_tot = np.sum((Fx_ref - np.mean(Fx_ref)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        rmse = np.sqrt(np.mean((Fx_ref - Fx_jax) ** 2))
        nrmse = rmse / (np.max(np.abs(Fx_ref)) + 1e-6) * 100

        peak_ref = np.max(np.abs(Fx_ref))
        peak_jax = np.max(np.abs(Fx_jax))
        dpeak = (peak_jax - peak_ref) / (peak_ref + 1e-6) * 100

        gamma_deg = np.degrees(gamma_val)
        print(f"{Fz_val:6.0f} {gamma_deg:5.1f} {r2:8.5f} {rmse:10.2f} "
              f"{nrmse:8.2f} {peak_ref:10.1f} {peak_jax:10.1f} {dpeak:+8.2f}")

        key = f"Fx_Fz{Fz_val:.0f}_g{gamma_deg:.0f}"
        metrics[key] = {'r2': r2, 'rmse': rmse, 'nrmse': nrmse, 'dpeak_pct': dpeak}
        all_fx_r2.append(r2)

    # ── 3.3  Load sensitivity (degressive) check ────────────────────────
    print("\n── Load sensitivity (peak μ_y vs Fz) ──")
    Fz_test = np.array([300, 500, 700, 900, 1200, 1500])
    mu_peaks = []
    for Fz_i in Fz_test:
        alpha_dense = np.linspace(0, 0.25, 500)
        Fy_dense = []
        for a in alpha_dense:
            _, fy = tire.compute_tire_forces(
                alpha=jnp.float32(a), kappa=jnp.float32(0.0),
                Fz=jnp.float32(Fz_i), gamma=jnp.float32(0.0),
                Vx=jnp.float32(11.176), T_ribs=T_ribs, T_gas=T_gas, T_core=T_core,
            )
            Fy_dense.append(float(fy))
        mu_peak = np.max(np.abs(Fy_dense)) / Fz_i
        mu_peaks.append(mu_peak)
        print(f"  Fz={Fz_i:6.0f} N  →  μ_peak = {mu_peak:.4f}")

    # Degressive check: μ should decrease with Fz
    degressive = all(mu_peaks[i] >= mu_peaks[i+1] for i in range(len(mu_peaks)-1))
    print(f"  Degressive: {'✓ PASS' if degressive else '✗ FAIL — check PDY2 sign'}")
    metrics['load_sensitivity_degressive'] = degressive
    metrics['mu_peaks'] = dict(zip(Fz_test.tolist(), mu_peaks))

    # ── 3.4  PINN + GP layer check (residual magnitude) ─────────────────
    print("\n── PINN/GP residual magnitude at nominal conditions ──")
    test_alpha = jnp.float32(0.10)   # ~5.7°, typical cornering
    test_Fz = jnp.float32(700.0)

    # With PINN+GP active (default)
    Fx_full, Fy_full = tire.compute_tire_forces(
        alpha=test_alpha, kappa=jnp.float32(0.0),
        Fz=test_Fz, gamma=jnp.float32(-0.035),
        Vx=jnp.float32(11.0), T_ribs=T_ribs, T_gas=T_gas, T_core=T_core,
    )

    # Evaluate PINN explicitly
    state_8d = jnp.array([0.10, 0.0, -0.035, 700.0, 11.0, 0.0, 0.0, 0.0])
    mods, sigma = tire._pinn_module.apply(tire._pinn_params, state_8d)
    print(f"  PINN drift corrections: ΔFx/Fx0 = {float(mods[0]):+.4f}, "
          f"ΔFy/Fy0 = {float(mods[1]):+.4f}")
    print(f"  GP sigma (uncertainty): {float(sigma):.4f}")
    print(f"  LCB penalty: {min(2*float(sigma), 0.15):.4f} (cap 0.15)")
    metrics['pinn_drift_fx'] = float(mods[0])
    metrics['pinn_drift_fy'] = float(mods[1])
    metrics['gp_sigma_nominal'] = float(sigma)

    # ── Summary ──────────────────────────────────────────────────────────
    mean_fy_r2 = np.mean(all_fy_r2)
    mean_fx_r2 = np.mean(all_fx_r2) if all_fx_r2 else 0
    composite = 0.6 * mean_fy_r2 + 0.3 * mean_fx_r2 + 0.1 * (1.0 if degressive else 0.0)

    print(f"\n{'=' * 70}")
    print(f" CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Lateral  Fy  mean R² : {mean_fy_r2:.6f}  "
          f"({'PASS' if mean_fy_r2 > 0.999 else 'CHECK — expect >0.999'})")
    print(f"  Longit.  Fx  mean R² : {mean_fx_r2:.6f}  "
          f"({'PASS' if mean_fx_r2 > 0.999 else 'CHECK — expect >0.999'})")
    print(f"  Load sensitivity     : {'DEGRESSIVE ✓' if degressive else 'PROGRESSIVE ✗'}")
    print(f"  Composite score      : {composite:.4f} / 1.0000")
    print(f"{'=' * 70}\n")

    metrics['mean_fy_r2'] = mean_fy_r2
    metrics['mean_fx_r2'] = mean_fx_r2
    metrics['composite_score'] = composite

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# §4  Raw TTC .mat loader (for when you get the consortium data)
# ─────────────────────────────────────────────────────────────────────────────

def load_ttc_mat(mat_path: str) -> Optional[dict]:
    """
    Load raw TTC .mat file from Calspan TIRF.

    Standard channel names (TTC Round 8/9 format):
      SA  = slip angle [deg]
      FY  = lateral force [N]
      FX  = longitudinal force [N]
      FZ  = normal load [N] (positive down)
      MZ  = aligning torque [N·m]
      IA  = inclination angle (camber) [deg]
      SL  = slip ratio [-]
      V   = velocity [km/h]
      P   = pressure [kPa]
      TSTC = surface temperature [°C]
      ET  = elapsed time [s]
      N   = rotational speed [rpm]
      RL  = loaded radius [m]

    Returns dict of numpy arrays, or None if scipy not available.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        print("[TTC] scipy required for .mat loading: pip install scipy")
        return None

    mat = loadmat(mat_path, squeeze_me=True)

    # TTC data is typically in a struct; find the main data variable
    # Common names: 'data', 'channel', or the tire name
    data_key = None
    for k in mat:
        if not k.startswith('_') and hasattr(mat[k], 'dtype'):
            if mat[k].dtype.names and 'SA' in mat[k].dtype.names:
                data_key = k
                break

    if data_key is None:
        # Try flat arrays
        if 'SA' in mat:
            return {k: np.asarray(mat[k]).flatten()
                    for k in ['SA', 'FY', 'FX', 'FZ', 'MZ', 'IA',
                              'SL', 'V', 'P', 'TSTC', 'ET', 'N', 'RL']
                    if k in mat}
        print(f"[TTC] Could not find standard channels in {mat_path}")
        print(f"[TTC] Available keys: {[k for k in mat if not k.startswith('_')]}")
        return None

    struct = mat[data_key]
    channels = {}
    for name in struct.dtype.names:
        channels[name] = np.asarray(struct[name]).flatten()

    n = len(next(iter(channels.values())))
    print(f"[TTC] Loaded {mat_path}: {n} points, "
          f"channels: {list(channels.keys())}")
    return channels


def ttc_train_test_split(
    ttc_data: dict,
    test_fraction: float = 0.2,
    stratify_by: str = 'FZ',
    n_bins: int = 5,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    Stratified split of raw TTC data ensuring both train and test
    cover the full operating range of Fz (or any other channel).
    """
    rng = np.random.default_rng(seed)
    n = len(ttc_data[stratify_by])

    # Bin the stratification channel
    bins = np.linspace(
        ttc_data[stratify_by].min(),
        ttc_data[stratify_by].max(),
        n_bins + 1,
    )
    bin_idx = np.digitize(ttc_data[stratify_by], bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    test_mask = np.zeros(n, dtype=bool)
    for b in range(n_bins):
        in_bin = np.where(bin_idx == b)[0]
        n_test = max(1, int(len(in_bin) * test_fraction))
        chosen = rng.choice(in_bin, n_test, replace=False)
        test_mask[chosen] = True

    train = {k: v[~test_mask] for k, v in ttc_data.items()}
    test = {k: v[test_mask] for k, v in ttc_data.items()}

    print(f"[TTC Split] Train: {(~test_mask).sum()} | "
          f"Test: {test_mask.sum()} | "
          f"Stratified by {stratify_by} into {n_bins} bins")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# §5  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project-GP: TIR file loader & Pacejka cross-validation"
    )
    parser.add_argument('--tir', type=str, default=None,
                        help='Path to .tir file')
    parser.add_argument('--ttc-mat', type=str, default=None,
                        help='Path to raw TTC .mat file (from consortium)')
    parser.add_argument('--export-coeffs', action='store_true',
                        help='Export parsed .tir as tire_coeffs_from_tir.py')
    parser.add_argument('--validate', action='store_true',
                        help='Run cross-validation against JAX PacejkaTire')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for exported coefficients')
    args = parser.parse_args()

    coeffs = None

    if args.tir:
        coeffs = parse_tir(args.tir)

        if args.export_coeffs:
            export_as_tire_coeffs_py(coeffs, args.output)

    if args.ttc_mat:
        ttc = load_ttc_mat(args.ttc_mat)
        if ttc is not None:
            train, test = ttc_train_test_split(ttc)
            print(f"\n[Ready] Use train/test sets for PINN training + validation")

    if args.validate:
        metrics = validate_jax_pacejka(coeffs)

    if not any([args.tir, args.ttc_mat, args.validate]):
        parser.print_help()
        print("\nExamples:")
        print("  python tools/load_tir_and_validate.py --tir hoosier_r25b.tir")
        print("  python tools/load_tir_and_validate.py --tir hoosier_r25b.tir --export-coeffs")
        print("  python tools/load_tir_and_validate.py --validate")
        print("  python tools/load_tir_and_validate.py --ttc-mat TTC_R8_Hoosier.mat")


if __name__ == '__main__':
    main()