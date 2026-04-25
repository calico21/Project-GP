# models/track_surface.py
# ═══════════════════════════════════════════════════════════════════════════════
# Project-GP — Dynamic Track Surface Model
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM (Flaw #6):
#   Road friction μ and track temperature are uniform, static boundary
#   conditions. The entire track has the same grip coefficient at every
#   point, at every time. This means:
#   1. The optimizer cannot learn that turn-in on a rubbered-in racing line
#      has 8-12% more grip than the dusty inside
#   2. The UKF state estimator never needs to adapt to grip transitions
#   3. Shadow zones (bridges, grandstands) that cool the track are invisible
#   4. The tire thermal model gets a constant T_track boundary condition
#
# SOLUTION:
#   Spatially-varying, time-evolving track surface model:
#
#   μ(s, n, t) = μ_base · Γ_rubber(s, n) · Γ_thermal(s, t) · Γ_moisture(s, t)
#
#   where:
#     s = distance along track centerline [m]
#     n = lateral offset from centerline [m] (negative = inside)
#     t = time [s]
#
#   Γ_rubber(s, n): racing-line rubber build-up — a Gaussian distribution
#     centered on the racing line that grows logarithmically with laps.
#     Inner regions that are never driven have Γ_rubber ≈ 0.92 (dusty).
#
#   Γ_thermal(s, t): track temperature field solved as 1D heat equation
#     along s, with solar heating, shadow zones, and wind cooling.
#     Track surface temperature drives the tire contact temperature.
#
#   Γ_moisture(s, t): moisture/oil contamination model — certain zones
#     (pit exit, curb regions) may have persistent low-grip patches.
#
# DISCRETIZATION:
#   Track is discretized into N_s × N_n cells (typical: 500 × 5).
#   Each cell stores: (μ_local, T_surface, rubber_level, moisture).
#   Interpolation between cells is bilinear (C⁰) — sufficient since
#   the vehicle samples at low spatial frequency relative to cell size.
#
# JAX CONTRACT: Pure JAX, JIT-safe, vmapped spatial queries.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class TrackSurfaceConfig(NamedTuple):
    """Track surface model parameters."""
    # Baseline friction
    mu_base: float = 1.0             # μ = 1.0 (then tire model supplies actual μ)
    mu_dusty: float = 0.88           # μ on unswept/dusty surface (relative)
    mu_rubbered: float = 1.08        # μ on well-rubbered surface (relative)
    mu_wet: float = 0.65             # μ on wet surface (relative)

    # Rubber build-up dynamics
    rubber_rate: float = 0.002       # rubber accumulation per tire pass [fraction]
    rubber_decay: float = 1e-5       # rubber degradation rate [1/s] (UV, rain)
    rubber_width: float = 1.5        # racing line width for rubber deposit [m]

    # Thermal
    T_surface_base: float = 30.0     # baseline track temperature [°C]
    T_air: float = 25.0              # ambient air temperature [°C]
    solar_flux: float = 800.0        # W/m² (mid-day summer)
    alpha_solar: float = 0.9         # asphalt solar absorptivity
    emissivity: float = 0.93         # asphalt thermal emissivity
    k_asphalt: float = 1.0           # W/m/K asphalt conductivity
    rho_c_asphalt: float = 2.0e6     # J/m³/K asphalt volumetric heat capacity
    h_wind: float = 15.0             # W/m²/K wind convection coefficient

    # Discretization
    N_s: int = 200                   # number of longitudinal cells
    N_n: int = 5                     # number of lateral cells
    track_length: float = 1000.0     # total track length [m]
    track_half_width: float = 5.0    # half track width [m]

    # Grip-temperature coupling
    T_grip_peak: float = 45.0        # track temp at peak asphalt grip [°C]
    T_grip_width: float = 20.0       # temperature window width [°C]


# ─────────────────────────────────────────────────────────────────────────────
# §2  Track Surface State
# ─────────────────────────────────────────────────────────────────────────────

class TrackSurfaceState(NamedTuple):
    """Full track surface state."""
    T_surface: jax.Array     # (N_s, N_n) surface temperature [°C]
    rubber_level: jax.Array  # (N_s, N_n) rubber accumulation ∈ [0, 1]
    moisture: jax.Array      # (N_s, N_n) moisture level ∈ [0, 1]
    shadow_mask: jax.Array   # (N_s,) shadow factor ∈ [0, 1] (0 = full shadow)

    @classmethod
    def default(cls, cfg: TrackSurfaceConfig = TrackSurfaceConfig()) -> 'TrackSurfaceState':
        """Initialize with uniform conditions."""
        return cls(
            T_surface=jnp.full((cfg.N_s, cfg.N_n), cfg.T_surface_base),
            rubber_level=jnp.zeros((cfg.N_s, cfg.N_n)),
            moisture=jnp.zeros((cfg.N_s, cfg.N_n)),
            shadow_mask=jnp.ones(cfg.N_s),  # no shadows by default
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3  Shadow Zone Configuration
# ─────────────────────────────────────────────────────────────────────────────

def create_shadow_mask(
    N_s: int,
    track_length: float,
    shadow_zones: list = None,
) -> jax.Array:
    """
    Create shadow mask from a list of (start_m, end_m, opacity) tuples.

    shadow_zones: list of (start [m], end [m], opacity ∈ [0,1])
      opacity = 0 → full shadow (no solar heating)
      opacity = 1 → full sun

    Default: one bridge shadow zone at 30% of track.
    """
    if shadow_zones is None:
        # Default: bridge at 30% of track, grandstand at 70%
        shadow_zones = [
            (0.28 * track_length, 0.32 * track_length, 0.2),   # bridge
            (0.68 * track_length, 0.73 * track_length, 0.5),   # grandstand
        ]

    s_grid = jnp.linspace(0, track_length, N_s)
    mask = jnp.ones(N_s)

    for start, end, opacity in shadow_zones:
        # Smooth transitions at shadow edges (sigmoid, 2m transition)
        enter = jax.nn.sigmoid(2.0 * (s_grid - start))
        exit_ = jax.nn.sigmoid(2.0 * (end - s_grid))
        in_shadow = enter * exit_
        mask = mask * (1.0 - in_shadow * (1.0 - opacity))

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# §4  Rubber Build-Up
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def update_rubber_level(
    rubber: jax.Array,        # (N_s, N_n) current rubber level
    car_s: jax.Array,         # car position along track [m]
    car_n: jax.Array,         # car lateral offset [m]
    cfg: TrackSurfaceConfig = TrackSurfaceConfig(),
    dt: float = 0.005,
) -> jax.Array:
    """
    Update rubber accumulation based on car position.

    Rubber deposits in a Gaussian pattern around the car's lateral position.
    The racing line gradually builds up μ over multiple laps.

    rubber(s, n) += rate · G(n - car_n; σ_w) · G(s - car_s; σ_s)

    Decay: rubber -= decay_rate · rubber · dt
    """
    ds = cfg.track_length / cfg.N_s
    dn = 2.0 * cfg.track_half_width / cfg.N_n

    s_grid = jnp.linspace(0, cfg.track_length, cfg.N_s)
    n_grid = jnp.linspace(-cfg.track_half_width, cfg.track_half_width, cfg.N_n)

    # Gaussian deposit centered on car position
    # Longitudinal: sharp deposit under contact patch (~0.15m)
    s_dist = jnp.abs(s_grid - car_s)
    # Handle wraparound for circuit
    s_dist = jnp.minimum(s_dist, cfg.track_length - s_dist)
    g_s = jnp.exp(-0.5 * (s_dist / 0.3) ** 2)  # σ_s = 0.3m

    # Lateral: wider deposit (full tire width ~0.2m, spread by traffic)
    g_n = jnp.exp(-0.5 * ((n_grid - car_n) / cfg.rubber_width) ** 2)

    # Outer product gives 2D deposit pattern
    deposit = jnp.outer(g_s, g_n) * cfg.rubber_rate * dt

    # Update with deposit and decay
    rubber_new = rubber + deposit - cfg.rubber_decay * rubber * dt
    return jnp.clip(rubber_new, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Track Thermal Evolution
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def update_track_temperature(
    T_surface: jax.Array,       # (N_s, N_n) current surface temperatures [°C]
    shadow_mask: jax.Array,     # (N_s,) solar occlusion factor
    cfg: TrackSurfaceConfig = TrackSurfaceConfig(),
    dt: float = 1.0,            # thermal update timestep (can be larger than sim dt)
) -> jax.Array:
    """
    Track surface temperature evolution.

    Energy balance per cell:
      C · dT/dt = Q_solar - Q_radiation - Q_convection - Q_conduction

    Q_solar = α · I_solar · shadow_factor
    Q_radiation = ε · σ_SB · T⁴ (linearized around T_base)
    Q_convection = h_wind · (T - T_air)
    Q_conduction = k/d · (T - T_bulk)  [conduction into asphalt bulk]

    Returns: updated T_surface (N_s, N_n)
    """
    sigma_SB = 5.67e-8  # Stefan-Boltzmann constant

    # Surface depth for heat capacity
    d_surface = 0.02  # m — top 2cm of asphalt exchanges heat rapidly
    C_cell = cfg.rho_c_asphalt * d_surface  # J/m²/K

    # Solar heating (per unit area, shadow-modulated)
    Q_solar = cfg.alpha_solar * cfg.solar_flux * shadow_mask[:, None]
    Q_solar = jnp.broadcast_to(Q_solar, T_surface.shape)

    # Radiative cooling (linearized: Q_rad ≈ 4·ε·σ·T₀³·(T-T₀))
    T_K = T_surface + 273.15
    T_ref_K = cfg.T_air + 273.15
    Q_rad = cfg.emissivity * sigma_SB * 4.0 * T_ref_K ** 3 * (T_K - T_ref_K)

    # Wind convection
    Q_conv = cfg.h_wind * (T_surface - cfg.T_air)

    # Conduction to bulk (slow — bulk at ~T_air + small offset)
    T_bulk = cfg.T_air + 2.0  # bulk temperature slightly above air
    Q_cond = cfg.k_asphalt / 0.1 * (T_surface - T_bulk)  # 0.1m depth to bulk

    # Energy balance
    dT_dt = (Q_solar - Q_rad - Q_conv - Q_cond) / C_cell
    T_new = T_surface + dT_dt * dt

    return jnp.clip(T_new, -10.0, 80.0)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Friction Query
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def query_track_friction(
    car_s: jax.Array,           # distance along track [m]
    car_n: jax.Array,           # lateral offset [m]
    state: TrackSurfaceState,
    cfg: TrackSurfaceConfig = TrackSurfaceConfig(),
) -> tuple:
    """
    Query the local friction coefficient and track temperature at (s, n).

    Bilinear interpolation from the grid.

    Returns:
        mu_local: local friction multiplier (to be applied to tire μ)
        T_track_local: local track surface temperature [°C]
    """
    # Map (s, n) to grid indices (fractional)
    s_frac = (car_s % cfg.track_length) / cfg.track_length * (cfg.N_s - 1)
    n_frac = ((car_n + cfg.track_half_width)
              / (2.0 * cfg.track_half_width) * (cfg.N_n - 1))

    # Clamp to grid bounds
    s_frac = jnp.clip(s_frac, 0.0, cfg.N_s - 1.001)
    n_frac = jnp.clip(n_frac, 0.0, cfg.N_n - 1.001)

    # Bilinear interpolation indices
    si = jnp.floor(s_frac).astype(jnp.int32)
    ni = jnp.floor(n_frac).astype(jnp.int32)
    sf = s_frac - si  # fractional part
    nf = n_frac - ni

    si1 = jnp.minimum(si + 1, cfg.N_s - 1)
    ni1 = jnp.minimum(ni + 1, cfg.N_n - 1)

    # ── Rubber level interpolation ────────────────────────────────────────
    r00 = state.rubber_level[si, ni]
    r10 = state.rubber_level[si1, ni]
    r01 = state.rubber_level[si, ni1]
    r11 = state.rubber_level[si1, ni1]
    rubber = (r00 * (1 - sf) * (1 - nf) + r10 * sf * (1 - nf)
              + r01 * (1 - sf) * nf + r11 * sf * nf)

    # ── Temperature interpolation ─────────────────────────────────────────
    t00 = state.T_surface[si, ni]
    t10 = state.T_surface[si1, ni]
    t01 = state.T_surface[si, ni1]
    t11 = state.T_surface[si1, ni1]
    T_local = (t00 * (1 - sf) * (1 - nf) + t10 * sf * (1 - nf)
               + t01 * (1 - sf) * nf + t11 * sf * nf)

    # ── Moisture interpolation ────────────────────────────────────────────
    m00 = state.moisture[si, ni]
    m10 = state.moisture[si1, ni]
    m01 = state.moisture[si, ni1]
    m11 = state.moisture[si1, ni1]
    moisture = (m00 * (1 - sf) * (1 - nf) + m10 * sf * (1 - nf)
                + m01 * (1 - sf) * nf + m11 * sf * nf)

    # ── Composite friction multiplier ─────────────────────────────────────
    # Rubber contribution: dusty → rubbered-in
    mu_rubber = cfg.mu_dusty + (cfg.mu_rubbered - cfg.mu_dusty) * rubber

    # Temperature contribution: Gaussian peak at T_grip_peak
    T_diff = T_local - cfg.T_grip_peak
    mu_thermal = 1.0 - 0.08 * (T_diff / cfg.T_grip_width) ** 2

    # Moisture contribution
    mu_moisture = 1.0 - moisture * (1.0 - cfg.mu_wet)

    # Combined multiplier (multiplicative, centered on 1.0)
    mu_local = cfg.mu_base * mu_rubber * mu_thermal * mu_moisture
    mu_local = jnp.clip(mu_local, 0.3, 1.5)

    return mu_local, T_local


# ─────────────────────────────────────────────────────────────────────────────
# §7  Per-Corner Friction Query
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def query_corner_friction(
    car_s: jax.Array,           # distance along track [m]
    car_n: jax.Array,           # lateral offset of CG [m]
    yaw: jax.Array,             # yaw angle [rad]
    state: TrackSurfaceState,
    cfg: TrackSurfaceConfig = TrackSurfaceConfig(),
    track_f: float = 1.20,      # front track width [m]
    track_r: float = 1.18,      # rear track width [m]
    lf: float = 0.8525,
    lr: float = 0.6975,
) -> tuple:
    """
    Query friction for each contact patch.

    Each tire contacts the track at a different (s, n) position.
    The friction can be DIFFERENT at each corner — asymmetric grip.

    Returns:
        mu_corners: (4,) local friction multipliers [FL, FR, RL, RR]
        T_corners: (4,) local track temperatures [°C]
    """
    cos_yaw = jnp.cos(yaw)
    sin_yaw = jnp.sin(yaw)

    # Contact patch positions in track coordinates
    # FL: forward-left, FR: forward-right, RL: rear-left, RR: rear-right
    offsets = jnp.array([
        [lf, track_f / 2],    # FL
        [lf, -track_f / 2],   # FR
        [-lr, track_r / 2],   # RL
        [-lr, -track_r / 2],  # RR
    ])

    # Rotate offsets by yaw angle
    s_offsets = offsets[:, 0] * cos_yaw - offsets[:, 1] * sin_yaw
    n_offsets = offsets[:, 0] * sin_yaw + offsets[:, 1] * cos_yaw

    s_corners = car_s + s_offsets
    n_corners = car_n + n_offsets

    # Query each corner
    mu_corners, T_corners = jax.vmap(
        lambda s, n: query_track_friction(s, n, state, cfg)
    )(s_corners, n_corners)

    return mu_corners, T_corners


# ─────────────────────────────────────────────────────────────────────────────
# §8  Full Track Update Step
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def track_surface_step(
    state: TrackSurfaceState,
    car_s: jax.Array,
    car_n: jax.Array,
    cfg: TrackSurfaceConfig = TrackSurfaceConfig(),
    dt_sim: float = 0.005,
    dt_thermal: float = 1.0,
    update_thermal: bool = True,
) -> TrackSurfaceState:
    """
    Full track surface update: rubber deposition + thermal evolution.

    Called every sim step for rubber update, every 1s for thermal update
    (thermal time constants are much longer than the sim dt).
    """
    # Rubber build-up
    rubber_new = update_rubber_level(state.rubber_level, car_s, car_n, cfg, dt_sim)

    # Thermal update (only when triggered — saves compute)
    T_new = jax.lax.cond(
        update_thermal,
        lambda t: update_track_temperature(t, state.shadow_mask, cfg, dt_thermal),
        lambda t: t,
        state.T_surface,
    )

    return TrackSurfaceState(
        T_surface=T_new,
        rubber_level=rubber_new,
        moisture=state.moisture,  # moisture evolves on longer timescales (rain model)
        shadow_mask=state.shadow_mask,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §9  Initialization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_track_surface(
    track_length: float = 1000.0,
    track_width: float = 10.0,
    n_laps_pre_rubbered: int = 0,
    shadow_zones: list = None,
    T_ambient: float = 25.0,
    solar_flux: float = 800.0,
) -> tuple:
    """
    Factory to create track surface state and config.

    Args:
        track_length: circuit length [m]
        track_width: total track width [m]
        n_laps_pre_rubbered: simulate this many laps of rubber build-up
        shadow_zones: list of (start_m, end_m, opacity) for shadow regions
        T_ambient: ambient temperature [°C]
        solar_flux: solar radiation [W/m²]

    Returns:
        state: TrackSurfaceState
        cfg: TrackSurfaceConfig
    """
    cfg = TrackSurfaceConfig(
        track_length=track_length,
        track_half_width=track_width / 2.0,
        T_air=T_ambient,
        T_surface_base=T_ambient + 10.0,  # track warmer than air
        solar_flux=solar_flux,
    )

    shadow = create_shadow_mask(cfg.N_s, track_length, shadow_zones)

    state = TrackSurfaceState(
        T_surface=jnp.full((cfg.N_s, cfg.N_n), cfg.T_surface_base),
        rubber_level=jnp.zeros((cfg.N_s, cfg.N_n)),
        moisture=jnp.zeros((cfg.N_s, cfg.N_n)),
        shadow_mask=shadow,
    )

    # Pre-rubber if requested (simulate ideal racing line deposition)
    if n_laps_pre_rubbered > 0:
        # Deposit rubber along the ideal racing line (n ≈ 0)
        n_grid = jnp.linspace(-cfg.track_half_width, cfg.track_half_width, cfg.N_n)
        racing_line_deposit = jnp.exp(-0.5 * (n_grid / 1.5) ** 2)
        per_lap = cfg.rubber_rate * track_length / (cfg.track_length / cfg.N_s)
        total_rubber = jnp.minimum(
            per_lap * n_laps_pre_rubbered * racing_line_deposit[None, :],
            1.0,
        )
        state = state._replace(
            rubber_level=jnp.broadcast_to(total_rubber, (cfg.N_s, cfg.N_n)))

    return state, cfg