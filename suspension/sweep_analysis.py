# suspension/sweep_analysis.py
# Project-GP — Full Kinematic Sweep Analysis (Optimum Kinematics equivalent)
# =============================================================================
#
# Computes every standard kinematic output that Optimum Kinematics reports,
# over the full suspension travel range, for a given (toe, camber) target.
#
# OUTPUTS (all as numpy arrays over the heave sweep):
#   Camber angle          [deg]
#   Toe angle             [deg]   (= bump steer curve)
#   Caster angle          [deg]   (variation with heave)
#   KPI                   [deg]   (variation with heave)
#   Motion ratio          [-]
#   Roll centre height    [mm]
#   Scrub radius          [mm]    (= mechanical trail projected at ground)
#   Track width change    [mm]    (wheel centre Y vs nominal)
#   Wheel centre height   [mm]
#   Wheel centre path     [(X,Y,Z) array in mm]
#
# DERIVATIVES (all analytically computed via IFD, not finite differences):
#   Camber gain           [deg/m]
#   Bump steer            [deg/m] = dToe/dz
#   dMR/dz                [1/m]
#   dRC/dz                [-]
#
# STEER AXIS GEOMETRY:
#   Kingpin inclination   [deg]   (static — from hardpoints only)
#   Caster angle          [deg]   (static)
#   Mechanical trail      [mm]
#   Scrub radius at nominal
#
# DESIGN RATIONALE:
#   Every curve is computed by jax.vmap(solve_at_heave)(z_array), making
#   the results differentiable w.r.t. (delta_L_tr, psi_shim).  This lets
#   the MORL optimizer compute dObjective/d(toe_target) and
#   dObjective/d(camber_target) exactly via the IFD chain, not by
#   finite-differencing the sweep.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from suspension.kinematics import SuspensionKinematics, KinematicOutputs


# ---------------------------------------------------------------------------
# §1  Sweep parameters
# ---------------------------------------------------------------------------

# Standard heave range for FS: -80 mm droop to +150 mm bump
Z_MIN_M =  -0.080
Z_MAX_M =   0.150
N_SWEEP =   500     # resolution (fine enough for smooth curves)

# Standard steer sweep: ±30 mm rack travel (≈ ±8° for typical FS rack)
RACK_MIN_MM = -30.0
RACK_MAX_MM =  30.0
N_STEER     =  100


# ---------------------------------------------------------------------------
# §2  SweepResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """
    All kinematic outputs for one axle over the heave sweep.
    All arrays are (N_SWEEP,) numpy, with z_mm as the independent axis.

    Angles in degrees, lengths in mm (for human-readable output).
    Gains are SI (deg/m or mm/m) for consistency with OKinematic exports.
    """
    axle:             str        # 'front' or 'rear'
    toe_target_deg:   float
    camber_target_deg: float

    # Independent variable
    z_mm:              np.ndarray   # heave [mm]

    # Primary kinematic outputs
    camber_deg:        np.ndarray   # [deg]
    toe_deg:           np.ndarray   # [deg] — bump steer curve
    caster_deg:        np.ndarray   # [deg] — caster variation with heave
    kpi_deg:           np.ndarray   # [deg] — KPI variation with heave
    motion_ratio:      np.ndarray   # [-]
    rc_height_mm:      np.ndarray   # [mm]
    scrub_radius_mm:   np.ndarray   # [mm]
    track_change_mm:   np.ndarray   # [mm] — wheel centre Y relative to nominal
    wheel_z_mm:        np.ndarray   # [mm] — wheel centre height

    # Derivatives at z=0
    camber_gain_deg_per_m:  float   # dCamber/dz [deg/m]
    bump_steer_deg_per_m:   float   # dToe/dz [deg/m]
    mr_at_zero:             float   # MR at z=0
    rc_at_zero_mm:          float   # roll centre height at z=0 [mm]
    drc_dz:                 float   # dRC/dz [-]

    # Static steer axis geometry (hardpoint-only, no heave dependency)
    kpi_static_deg:     float
    caster_static_deg:  float
    mech_trail_mm:      float

    # Steer kinematics (Ackermann correction, filled by steer_sweep)
    steer_lock_deg:     float = 0.0
    ackermann_pct:      float = 0.0   # positive = pro-Ackermann

    def print_summary(self):
        print(f"\n{'='*62}")
        print(f"  Kinematic Summary — {self.axle.upper()} axle")
        print(f"  Toe target: {self.toe_target_deg:+.3f}°   "
              f"Camber target: {self.camber_target_deg:+.2f}°")
        print(f"{'='*62}")
        print(f"  Camber gain        : {self.camber_gain_deg_per_m:+.2f} deg/m")
        print(f"  Bump steer         : {self.bump_steer_deg_per_m:+.4f} deg/m  "
              f"({'toe-in' if self.bump_steer_deg_per_m > 0 else 'toe-out'} in bump)")
        print(f"  Motion ratio (z=0) : {self.mr_at_zero:.4f}")
        print(f"  Roll centre (z=0)  : {self.rc_at_zero_mm:.1f} mm")
        print(f"  dRC/dz             : {self.drc_dz:+.4f} m/m")
        print(f"  KPI (static)       : {self.kpi_static_deg:.2f}°")
        print(f"  Caster (static)    : {self.caster_static_deg:.2f}°")
        print(f"  Mechanical trail   : {self.mech_trail_mm:.1f} mm")
        print(f"  Camber @ -80mm     : {self.camber_deg[0]:+.3f}°")
        print(f"  Camber @   0mm     : {self.camber_deg[self._z0_idx]:+.3f}°")
        print(f"  Camber @ +50mm     : {self.camber_deg[self._z50_idx]:+.3f}°")
        print(f"  Toe    @ -80mm     : {self.toe_deg[0]:+.4f}°")
        print(f"  Toe    @   0mm     : {self.toe_deg[self._z0_idx]:+.4f}°")
        print(f"  Toe    @ +50mm     : {self.toe_deg[self._z50_idx]:+.4f}°")
        print(f"  Track change range : [{self.track_change_mm.min():.1f}, "
              f"{self.track_change_mm.max():.1f}] mm")
        print(f"{'='*62}")

    @property
    def _z0_idx(self) -> int:
        return int(np.argmin(np.abs(self.z_mm)))

    @property
    def _z50_idx(self) -> int:
        return int(np.argmin(np.abs(self.z_mm - 50.0)))

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict for Excel export."""
        return {
            "z_mm":            self.z_mm,
            "camber_deg":      self.camber_deg,
            "toe_deg":         self.toe_deg,
            "caster_deg":      self.caster_deg,
            "kpi_deg":         self.kpi_deg,
            "motion_ratio":    self.motion_ratio,
            "rc_height_mm":    self.rc_height_mm,
            "scrub_radius_mm": self.scrub_radius_mm,
            "track_change_mm": self.track_change_mm,
            "wheel_z_mm":      self.wheel_z_mm,
        }


# ---------------------------------------------------------------------------
# §3  Caster and KPI variation with heave
# ---------------------------------------------------------------------------

def _caster_and_kpi_from_kp(e_kp: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    From the instantaneous kingpin unit vector e_kp, compute:
        Caster = arctan2(−e_kp[X], e_kp[Z])   [rad]
        KPI    = arctan2(  e_kp[Y], e_kp[Z])   [rad]

    Sign convention (ISO/SAE):
        Caster positive = top of kingpin leans rearward  (−X direction)
        KPI positive    = top of kingpin leans inboard   (+Y for left wheel)
    """
    caster = jnp.arctan2(-e_kp[0], e_kp[2])
    kpi    = jnp.arctan2( e_kp[1], e_kp[2])
    return caster, kpi


def _scrub_radius(
    e_kp: jax.Array,
    wheel_pos: jax.Array,
    R_wheel: float,
) -> jax.Array:
    """
    Scrub radius = lateral distance from the kingpin axis extension to the
    tyre contact patch centre in the ground plane (Z=0).

    The kingpin axis passes through the lower ball joint C and has direction e_kp.
    Its intersection with Z=0:
        P_ground = C − (C[Z] / e_kp[Z]) * e_kp   (if e_kp[Z] ≠ 0)

    Scrub radius = P_ground[Y] − wheel_pos[Y]
    (positive = kingpin meets ground outboard of contact patch)

    Note: for the left wheel, a positive scrub radius means the contact patch
    is inboard of the kingpin intercept — common for FS with narrow track.
    """
    C = wheel_pos   # lower ball joint passed as wheel_pos (caller's responsibility)
    # We use wheel_pos directly for simplicity here; the exact scrub
    # requires C (lower ball joint). The calling code passes C.
    t_ground = -C[2] / (e_kp[2] + 1e-12)
    P_ground = C + t_ground * e_kp
    return P_ground[1] - wheel_pos[1]   # Y-diff [m]


# ---------------------------------------------------------------------------
# §4  Main sweep function
# ---------------------------------------------------------------------------

def compute_sweep(
    kin:            SuspensionKinematics,
    toe_target_deg: float,
    camber_target_deg: float,
    axle:           str = "front",
    z_min:          float = Z_MIN_M,
    z_max:          float = Z_MAX_M,
    n_pts:          int   = N_SWEEP,
) -> SweepResult:
    """
    Compute the full kinematic sweep for the given toe and camber targets.

    Internally:
    1. Inverts the kinematic model to find (delta_L_tr, psi_shim) for the targets.
    2. Runs jax.vmap(solve_at_heave)(z_array) for all heave positions.
    3. Computes derivatives at z=0 via jax.grad (IFD-backed).
    4. Packages everything into a SweepResult.

    Args:
        kin              : SuspensionKinematics instance (left side)
        toe_target_deg   : target static toe [deg]; positive = toe-in
        camber_target_deg: target static camber [deg]; negative = top lean outboard
        axle             : 'front' or 'rear' (label only)
        z_min, z_max     : heave range [m]
        n_pts            : number of sweep points
    """
    toe_rad    = math.radians(toe_target_deg)
    camber_rad = math.radians(camber_target_deg)

    # ── Invert to get tie-rod and shim adjustments ────────────────────────────
    delta_L_tr = jnp.array(kin.delta_L_tr_from_toe(toe_rad),    dtype=jnp.float32)
    psi_shim   = jnp.array(kin.psi_shim_from_camber(camber_rad), dtype=jnp.float32)

    # ── Heave array ───────────────────────────────────────────────────────────
    z_array = jnp.linspace(z_min, z_max, n_pts, dtype=jnp.float32)

    # ── Vectorised solve ──────────────────────────────────────────────────────
    outputs: KinematicOutputs = jax.jit(
        lambda z: kin.sweep(z, delta_L_tr, psi_shim)
    )(z_array)

    # ── Caster and KPI variation ───────────────────────────────────────────────
    # e_kp is already in KinematicOutputs; vmap extracts it per-point
    caster_arr, kpi_arr = jax.vmap(_caster_and_kpi_from_kp)(outputs.e_kp)

    # ── Scrub radius ──────────────────────────────────────────────────────────
    # Use lower ball joint C as the point through which kingpin axis passes
    scrub_arr = jax.vmap(
        lambda C, e_kp, W: _scrub_radius(e_kp, C, kin.R_wheel)
    )(outputs.C, outputs.e_kp, outputs.wheel_pos)

    # ── Track width change ─────────────────────────────────────────────────────
    # Reference: wheel centre Y at z=0
    z0_idx        = int(jnp.argmin(jnp.abs(z_array)))
    wheel_y_nom   = float(outputs.wheel_pos[z0_idx, 1])
    track_change  = outputs.wheel_pos[:, 1] - wheel_y_nom

    # ── Derivatives at z=0 via IFD ────────────────────────────────────────────
    z0 = jnp.array(0.0, dtype=jnp.float32)

    camber_gain_rad_m = float(jax.grad(
        lambda z: kin.solve_at_heave(z, delta_L_tr, psi_shim).camber_rad
    )(z0))
    bump_steer_rad_m = float(jax.grad(
        lambda z: kin.solve_at_heave(z, delta_L_tr, psi_shim).toe_rad
    )(z0))
    drc_dz = float(jax.grad(
        lambda z: kin.solve_at_heave(z, delta_L_tr, psi_shim).roll_centre_z
    )(z0))

    out_z0 = kin.solve_at_heave(z0, delta_L_tr, psi_shim)

    # Convert to numpy
    def to_np(x): return np.array(x)

    return SweepResult(
        axle=axle,
        toe_target_deg=toe_target_deg,
        camber_target_deg=camber_target_deg,
        z_mm=to_np(z_array) * 1000.0,
        camber_deg=np.degrees(to_np(outputs.camber_rad)),
        toe_deg=np.degrees(to_np(outputs.toe_rad)),
        caster_deg=np.degrees(to_np(caster_arr)),
        kpi_deg=np.degrees(to_np(kpi_arr)),
        motion_ratio=to_np(outputs.motion_ratio),
        rc_height_mm=to_np(outputs.roll_centre_z) * 1000.0,
        scrub_radius_mm=to_np(scrub_arr) * 1000.0,
        track_change_mm=to_np(track_change) * 1000.0,
        wheel_z_mm=to_np(outputs.wheel_pos[:, 2]) * 1000.0,
        camber_gain_deg_per_m=math.degrees(camber_gain_rad_m),
        bump_steer_deg_per_m=math.degrees(bump_steer_rad_m),
        mr_at_zero=float(out_z0.motion_ratio),
        rc_at_zero_mm=float(out_z0.roll_centre_z) * 1000.0,
        drc_dz=drc_dz,
        kpi_static_deg=math.degrees(kin.kpi_rad),
        caster_static_deg=math.degrees(kin.caster_rad),
        mech_trail_mm=kin.mech_trail_m * 1000.0,
    )


# ---------------------------------------------------------------------------
# §5  Parallel steer sweep (Ackermann analysis)
# ---------------------------------------------------------------------------

@dataclass
class SteerSweepResult:
    """
    Ackermann geometry and toe change with steering input.
    Computes inner/outer wheel toe as a function of rack travel [mm].
    """
    axle:               str
    rack_travel_mm:     np.ndarray   # (N_STEER,) rack travel [mm]
    toe_outer_deg:      np.ndarray   # outer wheel toe [deg]
    toe_inner_deg:      np.ndarray   # inner wheel toe [deg]
    ackermann_pct:      np.ndarray   # Ackermann % at each rack position
    ideal_ackermann:    np.ndarray   # ideal Ackermann toe-in for outer wheel [deg]

    def print_summary(self):
        idx_max = np.argmax(np.abs(self.rack_travel_mm))
        print(f"\n  Steer Sweep ({self.axle}):")
        print(f"  Max steer: inner={self.toe_inner_deg[idx_max]:+.3f}°  "
              f"outer={self.toe_outer_deg[idx_max]:+.3f}°")
        print(f"  Ackermann at max steer: {self.ackermann_pct[idx_max]:+.1f}%  "
              f"(+100%=perfect, 0%=parallel, -100%=anti)")


def compute_steer_sweep(
    kin_left:      SuspensionKinematics,
    kin_right:     SuspensionKinematics,
    toe_target_deg: float,
    camber_target_deg: float,
    rack_min_mm:   float = RACK_MIN_MM,
    rack_max_mm:   float = RACK_MAX_MM,
    n_pts:         int   = N_STEER,
    wheelbase_m:   float = 1.550,
    axle:          str   = "front",
) -> SteerSweepResult:
    """
    Compute inner/outer toe angles as a function of rack travel.
    Ackermann % is defined as:
        Ack% = (toe_inner − toe_outer) / (ideal_ackermann) × 100
    where ideal_ackermann is the perfect Ackermann correction for the wheel geometry.

    The rack translates laterally; the left tie-rod gets longer by +rack/2
    and the right tie-rod gets shorter by −rack/2 (symmetric rack).

    Args:
        kin_left, kin_right : SuspensionKinematics for each side
        toe_target_deg      : static toe at zero steer [deg]
        rack_min_mm         : rack travel range (negative = right turn)
        n_pts               : sweep resolution
    """
    toe_rad    = math.radians(toe_target_deg)
    camber_rad = math.radians(camber_target_deg)

    dL_static_L = kin_left.delta_L_tr_from_toe(toe_rad)
    dL_static_R = kin_right.delta_L_tr_from_toe(toe_rad)
    ps_L = kin_left.psi_shim_from_camber(camber_rad)
    ps_R = kin_right.psi_shim_from_camber(camber_rad)

    rack_array  = np.linspace(rack_min_mm, rack_max_mm, n_pts)
    z0          = jnp.array(0.0, dtype=jnp.float32)
    ps_jL       = jnp.array(ps_L, dtype=jnp.float32)
    ps_jR       = jnp.array(ps_R, dtype=jnp.float32)

    toe_L_arr = np.zeros(n_pts)
    toe_R_arr = np.zeros(n_pts)

    for i, rack_mm in enumerate(rack_array):
        rack_m = rack_mm / 1000.0
        # Rack moves left → left tie-rod extends, right tie-rod shortens
        dL_L = jnp.array(dL_static_L + rack_m * 0.5, dtype=jnp.float32)
        dL_R = jnp.array(dL_static_R - rack_m * 0.5, dtype=jnp.float32)

        out_L = kin_left.solve_at_heave( z0, dL_L, ps_jL)
        out_R = kin_right.solve_at_heave(z0, dL_R, ps_jR)

        # Mirror right toe for consistent sign convention
        toe_L_arr[i] = math.degrees(float(out_L.toe_rad))
        # Right side: toe_rad sign is mirrored relative to left
        toe_R_arr[i] = -math.degrees(float(out_R.toe_rad))

    # Ideal Ackermann: for a turn radius R, the inner wheel must toe in more.
    # At rack travel x, the steering angle is approximately δ ≈ x / steering_ratio
    # where steering_ratio is in mm/rad.
    # Ackermann correction: δ_inner − δ_outer = track / wheelbase × δ_outer²
    # Simplified to first order: Δδ_ideal ≈ t_w² / (2 × L × R)
    # We approximate R from the outer wheel steering angle.
    t_w = 2.0 * kin_left.half_track
    outer_steer_rad = np.abs(np.radians(toe_L_arr))
    ideal_ackermann_deg = np.degrees(
        np.arctan(1.0 / (1.0 / np.tan(np.maximum(outer_steer_rad, 1e-4)) - t_w / wheelbase_m))
    ) - np.abs(toe_L_arr)
    ideal_ackermann_deg = np.where(outer_steer_rad < 0.001, 0.0, ideal_ackermann_deg)

    delta_toe = np.abs(toe_L_arr) - np.abs(toe_R_arr)   # inner − outer toe-in
    with np.errstate(divide="ignore", invalid="ignore"):
        ackermann_pct = np.where(
            np.abs(ideal_ackermann_deg) > 0.01,
            delta_toe / ideal_ackermann_deg * 100.0,
            0.0,
        )

    return SteerSweepResult(
        axle=axle,
        rack_travel_mm=rack_array,
        toe_outer_deg=toe_L_arr,
        toe_inner_deg=toe_R_arr,
        ackermann_pct=np.clip(ackermann_pct, -200.0, 200.0),
        ideal_ackermann=ideal_ackermann_deg,
    )


# ---------------------------------------------------------------------------
# §6  Roll analysis (symmetric heave vs roll input)
# ---------------------------------------------------------------------------

@dataclass
class RollAnalysisResult:
    """
    Camber and toe change during chassis roll (anti-symmetric heave).
    Roll angle φ [deg] is the independent variable.
    """
    roll_deg:         np.ndarray   # roll angle [deg] (positive = left side bump)
    camber_outer_deg: np.ndarray   # outer wheel camber [deg]
    camber_inner_deg: np.ndarray   # inner wheel camber [deg]
    camber_net_deg:   np.ndarray   # net camber change (outer − initial)
    toe_outer_deg:    np.ndarray
    toe_inner_deg:    np.ndarray
    rc_height_mm:     np.ndarray   # roll centre at each roll angle [mm]


def compute_roll_analysis(
    kin_left:          SuspensionKinematics,
    kin_right:         SuspensionKinematics,
    toe_target_deg:    float,
    camber_target_deg: float,
    roll_max_deg:      float = 3.0,   # max chassis roll [deg]
    n_pts:             int   = 100,
    track_m:           float = 1.230,
) -> RollAnalysisResult:
    """
    For a chassis roll angle φ, the left wheel heaves by +Δz = tan(φ)×t_w/2
    and the right wheel heaves by −Δz (anti-symmetric).

    This gives the camber-in-roll characteristic, which is the most important
    output for tyre slip angle sensitivity during cornering.
    """
    toe_rad    = math.radians(toe_target_deg)
    camber_rad = math.radians(camber_target_deg)

    dL_L = jnp.array(kin_left.delta_L_tr_from_toe(toe_rad),    dtype=jnp.float32)
    dL_R = jnp.array(kin_right.delta_L_tr_from_toe(toe_rad),   dtype=jnp.float32)
    ps_L = jnp.array(kin_left.psi_shim_from_camber(camber_rad), dtype=jnp.float32)
    ps_R = jnp.array(kin_right.psi_shim_from_camber(camber_rad),dtype=jnp.float32)

    roll_arr = np.linspace(-roll_max_deg, roll_max_deg, n_pts)
    half_track = track_m / 2.0

    cam_outer = np.zeros(n_pts)
    cam_inner = np.zeros(n_pts)
    toe_outer = np.zeros(n_pts)
    toe_inner = np.zeros(n_pts)
    rc_arr    = np.zeros(n_pts)

    # Nominal camber at φ=0
    out0 = kin_left.solve_at_heave(jnp.array(0.0), dL_L, ps_L)
    cam0 = float(out0.camber_rad)

    for i, phi_deg in enumerate(roll_arr):
        phi = math.radians(phi_deg)
        dz  = math.tan(phi) * half_track   # heave of outer (left) wheel

        # Left = outer wheel in left-hand turn (φ > 0 = left side bumps)
        z_L = jnp.array( dz, dtype=jnp.float32)
        z_R = jnp.array(-dz, dtype=jnp.float32)

        out_L = kin_left.solve_at_heave( z_L, dL_L, ps_L)
        out_R = kin_right.solve_at_heave(z_R, dL_R, ps_R)

        cam_outer[i] = math.degrees(float(out_L.camber_rad))
        cam_inner[i] = math.degrees(float(out_R.camber_rad))
        toe_outer[i] = math.degrees(float(out_L.toe_rad))
        toe_inner[i] = -math.degrees(float(out_R.toe_rad))
        rc_arr[i]    = float(out_L.roll_centre_z) * 1000.0

    return RollAnalysisResult(
        roll_deg=roll_arr,
        camber_outer_deg=cam_outer,
        camber_inner_deg=cam_inner,
        camber_net_deg=cam_outer - math.degrees(cam0),
        toe_outer_deg=toe_outer,
        toe_inner_deg=toe_inner,
        rc_height_mm=rc_arr,
    )


# ---------------------------------------------------------------------------
# §7  Full analysis runner
# ---------------------------------------------------------------------------

@dataclass
class FullKinematicReport:
    """
    Complete kinematic analysis for one axle pair.
    Contains sweep, steer, and roll results together with the derived VP entries.
    """
    axle:           str
    heave_sweep:    SweepResult
    steer_sweep:    Optional[SteerSweepResult]   # None for rear passive-steer axle
    roll_analysis:  RollAnalysisResult

    # VP-ready derived quantities
    vp_entries:     Dict[str, Any]

    def print_all(self):
        self.heave_sweep.print_summary()
        if self.steer_sweep is not None:
            self.steer_sweep.print_summary()


def run_full_kinematic_analysis(
    kin_left:           SuspensionKinematics,
    kin_right:          Optional[SuspensionKinematics],
    toe_target_deg:     float,
    camber_target_deg:  float,
    axle:               str,
    vp:                 Dict[str, Any],
) -> FullKinematicReport:
    """
    Run all three analyses and assemble the VP entry dict.

    For the rear axle (passive steer), kin_right can be None and the
    steer sweep is skipped.
    """
    print(f"\n[KinematicAnalysis/{axle}] Running heave sweep "
          f"(toe={toe_target_deg:+.3f}°, camber={camber_target_deg:+.2f}°)...")
    sweep = compute_sweep(kin_left, toe_target_deg, camber_target_deg, axle=axle)
    sweep.print_summary()

    steer = None
    if kin_right is not None and not vp.get("passive_rear_steer", False):
        print(f"[KinematicAnalysis/{axle}] Running steer sweep...")
        steer = compute_steer_sweep(
            kin_left, kin_right,
            toe_target_deg, camber_target_deg,
            axle=axle,
            wheelbase_m=vp.get("lf", 0.8525) + vp.get("lr", 0.6975),
        )
        steer.print_summary()

    print(f"[KinematicAnalysis/{axle}] Running roll analysis...")
    roll = compute_roll_analysis(
        kin_left,
        kin_right if kin_right is not None else kin_left,
        toe_target_deg, camber_target_deg,
        track_m=vp.get("track_front" if axle == "front" else "track_rear", 1.230),
    )

    # ── VP entries from this sweep ────────────────────────────────────────────
    axle_tag = "_f" if axle == "front" else "_r"
    vp_entries = {
        f"motion_ratio{axle_tag}_poly": list(
            float(x) for x in np.array(
                kin_left.kinematic_gains(
                    jnp.array(kin_left.delta_L_tr_from_toe(math.radians(toe_target_deg))),
                    jnp.array(kin_left.psi_shim_from_camber(math.radians(camber_target_deg))),
                ).mr_poly
            )
        ),
        f"bump_steer_quad{axle_tag}":     sweep.bump_steer_deg_per_m * (math.pi / 180.0) / 2.0,
        f"camber_per_m_travel{axle_tag}": sweep.camber_gain_deg_per_m,
        f"h_rc{axle_tag}":               sweep.rc_at_zero_mm / 1000.0,
        f"dh_rc_dz{axle_tag}":           sweep.drc_dz,
    }

    return FullKinematicReport(
        axle=axle,
        heave_sweep=sweep,
        steer_sweep=steer,
        roll_analysis=roll,
        vp_entries=vp_entries,
    )