# optimization/discrete_catalog_optimizer.py
# Project-GP — Gumbel-Softmax Catalog-Aware Setup Optimizer
# ═══════════════════════════════════════════════════════════════════════════
#
# Discrete catalog → Gumbel-Softmax → differentiable weighted sum → JAX grad
#
# Concrete(π, τ): y_k = exp((log π_k + g_k) / τ) / Z,  g_k ~ Gumbel(0,1)
# Selected value: v = Σ_k y_k · catalog[k]  (gradient flows through y_k)
# Straight-Through: forward=hard argmax, backward=soft Gumbel gradient (STE)
# Temperature schedule: τ(t) = τ_init · (τ_final/τ_init)^(t / n_anneal_steps)
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from models.vehicle_dynamics import SuspensionSetup, SETUP_LB, SETUP_UB, DEFAULT_SETUP


class PartCatalog(NamedTuple):
    """
    Available parts per garage inventory.
    Arrays are 1-D JAX constants — XLA constant-folds them.
    """
    spring_f:  jax.Array   # (K_sf,) N/m  e.g. [25k, 30k, 35k, 40k, 45k, 50k]
    spring_r:  jax.Array   # (K_sr,)
    arb_f:     jax.Array   # (K_af,) N·m/rad
    arb_r:     jax.Array   # (K_ar,)
    c_low_f:   jax.Array   # (K_cf,) N·s/m damper low-speed bump clicks → forces
    c_low_r:   jax.Array   # (K_cr,)

    @staticmethod
    def ter27_default() -> "PartCatalog":
        return PartCatalog(
            spring_f = jnp.array([25000., 30000., 35000., 40000., 45000., 50000., 55000.]),
            spring_r = jnp.array([28000., 33000., 38000., 43000., 48000., 53000.]),
            arb_f    = jnp.array([0., 300., 600., 1000., 1500., 2000., 3000.]),
            arb_r    = jnp.array([0., 300., 600., 1000., 1500., 2000.]),
            c_low_f  = jnp.array([1200., 1500., 1800., 2100., 2500., 3000., 3500.]),
            c_low_r  = jnp.array([1200., 1500., 1800., 2100., 2500., 3000.]),
        )


class DiscreteSetupLogits(NamedTuple):
    """
    Trainable parameters in logit space.
    Discrete params: logit vectors per catalog dimension.
    Continuous params: unconstrained scalars → physical via sigmoid rescaling.
    """
    logit_kf:   jax.Array    # (K_sf,)
    logit_kr:   jax.Array    # (K_sr,)
    logit_af:   jax.Array    # (K_af,)
    logit_ar:   jax.Array    # (K_ar,)
    logit_cf:   jax.Array    # (K_cf,)
    logit_cr:   jax.Array    # (K_cr,)
    # Remaining 22 continuous params in un-constrained logit space
    # Decoded via: p_i = LB_i + (UB_i - LB_i) * sigmoid(z_i)
    z_continuous: jax.Array  # (22,)

    @classmethod
    def init(cls, catalog: PartCatalog, key: jax.Array) -> "DiscreteSetupLogits":
        keys = jax.random.split(key, 7)
        return cls(
            logit_kf      = jnp.zeros(catalog.spring_f.shape[0]),
            logit_kr      = jnp.zeros(catalog.spring_r.shape[0]),
            logit_af      = jnp.zeros(catalog.arb_f.shape[0]),
            logit_ar      = jnp.zeros(catalog.arb_r.shape[0]),
            logit_cf      = jnp.zeros(catalog.c_low_f.shape[0]),
            logit_cr      = jnp.zeros(catalog.c_low_r.shape[0]),
            z_continuous  = jax.random.normal(keys[6], (22,)) * 0.1,
        )


def gumbel_softmax_select(
    logits:   jax.Array,        # (K,) categorical logits
    catalog:  jax.Array,        # (K,) physical values
    tau:      jax.Array,        # scalar temperature
    key:      jax.Array,
    hard:     bool = False,     # STE: hard forward, soft backward
) -> tuple[jax.Array, jax.Array]:
    """Returns (selected_value, soft_weights)."""
    # Gumbel noise: -log(-log(U)), U ~ Uniform(0,1)
    U       = jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0 - 1e-10)
    gumbel  = -jnp.log(-jnp.log(U))
    y_soft  = jax.nn.softmax((logits + gumbel) / tau)

    if hard:
        # STE: one_hot forward, soft gradient backward
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft), logits.shape[0])
        y      = y_hard + (y_soft - jax.lax.stop_gradient(y_soft))
    else:
        y = y_soft

    return jnp.dot(y, catalog), y


def decode_to_physical_setup(
    logits:  DiscreteSetupLogits,
    catalog: PartCatalog,
    tau:     jax.Array,
    key:     jax.Array,
    hard:    bool = False,
) -> jax.Array:
    """
    Decode mixed logits → 28-element physical SuspensionSetup vector.
    Discrete params selected via Gumbel-Softmax; continuous via sigmoid.
    """
    keys = jax.random.split(key, 6)

    k_f,  _ = gumbel_softmax_select(logits.logit_kf, catalog.spring_f, tau, keys[0], hard)
    k_r,  _ = gumbel_softmax_select(logits.logit_kr, catalog.spring_r, tau, keys[1], hard)
    arb_f,_ = gumbel_softmax_select(logits.logit_af, catalog.arb_f,   tau, keys[2], hard)
    arb_r,_ = gumbel_softmax_select(logits.logit_ar, catalog.arb_r,   tau, keys[3], hard)
    c_f,  _ = gumbel_softmax_select(logits.logit_cf, catalog.c_low_f, tau, keys[4], hard)
    c_r,  _ = gumbel_softmax_select(logits.logit_cr, catalog.c_low_r, tau, keys[5], hard)

    # Continuous params: indices 6–27 mapped via sigmoid into [LB, UB]
    lb_cont = jnp.array(SETUP_LB)[6:]    # (22,) physical lower bounds
    ub_cont = jnp.array(SETUP_UB)[6:]    # (22,) physical upper bounds
    cont    = lb_cont + (ub_cont - lb_cont) * jax.nn.sigmoid(logits.z_continuous)

    return jnp.concatenate([
        jnp.array([k_f, k_r, arb_f, arb_r, c_f, c_r]),
        cont,
    ])  # (28,)


def temperature_schedule(step: int, T_init: float = 1.0, T_final: float = 0.08,
                          n_anneal: int = 600) -> jax.Array:
    """Exponential annealing: τ(t) = τ_init · (τ_final/τ_init)^(t/T)."""
    ratio = jnp.log(T_final / T_init) / n_anneal
    return jnp.array(T_init) * jnp.exp(ratio * jnp.minimum(step, n_anneal))


class DiscreteSetupOptimizer:
    """
    Mixed discrete-continuous setup optimizer.
    Maintains Gumbel-Softmax logits; Adam operates entirely in logit space.
    Gradient descent finds optimal logit values; Gumbel-Softmax bridges
    to physical catalog selections with temperature-controlled sharpness.
    """

    def __init__(self, catalog: PartCatalog, lr: float = 5e-3, n_anneal: int = 600):
        self.catalog   = catalog
        self.n_anneal  = n_anneal
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(2.0),
            optax.adam(lr),
        )

    def init(self, key: jax.Array) -> tuple:
        logits    = DiscreteSetupLogits.init(self.catalog, key)
        opt_state = self._optimizer.init(logits)
        return logits, opt_state

    def step(
        self,
        logits:    DiscreteSetupLogits,
        opt_state,
        objective_fn,               # (setup_vec → scalar)
        step_idx:  int,
        key:       jax.Array,
    ) -> tuple:
        tau = temperature_schedule(step_idx, n_anneal=self.n_anneal)

        def loss(lgts):
            # Soft decode during training — gradients flow through y_soft
            setup = decode_to_physical_setup(lgts, self.catalog, tau, key, hard=False)
            return objective_fn(setup)

        val, grads  = jax.value_and_grad(loss)(logits)
        updates, new_opt_state = self._optimizer.update(grads, opt_state, logits)
        new_logits  = optax.apply_updates(logits, updates)
        return new_logits, new_opt_state, val, tau

    def get_final_setup(self, logits: DiscreteSetupLogits, key: jax.Array) -> jax.Array:
        """Decode with hard=True (STE): exact catalog selections for the race engineer."""
        tau = jnp.array(0.01)   # near-zero temperature for argmax behavior
        return decode_to_physical_setup(logits, self.catalog, tau, key, hard=True)

    def get_selected_parts(self, logits: DiscreteSetupLogits) -> dict:
        """Return human-readable selected part values (for garage checklist)."""
        c = self.catalog
        return {
            "k_f_Nm":  float(c.spring_f[jnp.argmax(logits.logit_kf)]),
            "k_r_Nm":  float(c.spring_r[jnp.argmax(logits.logit_kr)]),
            "arb_f":   float(c.arb_f[jnp.argmax(logits.logit_af)]),
            "arb_r":   float(c.arb_r[jnp.argmax(logits.logit_ar)]),
            "c_low_f": float(c.c_low_f[jnp.argmax(logits.logit_cf)]),
            "c_low_r": float(c.c_low_r[jnp.argmax(logits.logit_cr)]),
        }