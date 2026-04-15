"""
online_sysid.py — Online H_net parameter adaptation via jax.grad

Closes the reality gap between MirenaSim's Godot physics and Project-GP's
Port-Hamiltonian model by continuously minimising rollout residuals against
MirenaSim ground-truth state transitions.

Architecture:
  RingBuffer:     Thread-safe circular buffer of (s_t, u_t, s_{t+1}, w_t) tuples.
  loss_fn:        JIT-compiled, vmapped; gradient flows through Project-GP forward step.
  _adapt_loop():  Daemon thread; wakes on NEW_DATA event, runs one Adam step.
  get_params():   Atomic read of current best params (used by inference thread).

Loss:
  L = E_{(s,u,s',w) ~ D} [ Σ_i w_i * (f_GP(s,u)[obs_i] - s'[obs_i])² ]

  where w_i = precision from Car.msg covariance (1/σ²_i).
  Precision-weighting: high-confidence MirenaSim states pull harder.
  This prevents adapting toward states where Godot itself has high uncertainty.

Safety:
  - Learning rate is conservative (3e-5) to prevent catastrophic forgetting.
  - EMA of params provides a stability backstop.
  - If loss spikes (NaN or >10× baseline), adaptation pauses and logs.
  - Gradient clipping (global norm ≤ 1.0) prevents exploding updates.

Threading:
  Ring buffer writes from ROS2 thread. Adam step in daemon thread.
  Param read from inference thread. All via RLock + atomic swap.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

log = logging.getLogger(__name__)

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
RING_CAPACITY   = 512    # Trajectory buffer capacity (frames)
ADAPT_PERIOD    = 100    # Push N frames before triggering one adapt step
BATCH_SIZE      = 32     # Mini-batch size
LEARNING_RATE   = 3e-5   # Conservative Adam LR — prevents catastrophic forgetting
GRAD_CLIP_NORM  = 1.0    # Global gradient norm clipping
EMA_DECAY       = 0.995  # EMA decay for param stability backstop
LOSS_SPIKE_THR  = 10.0   # Pause adaptation if loss > SPIKE_THR × baseline
WARMUP_FRAMES   = 64     # Minimum frames in buffer before first adapt step

# Observable 6-DOF indices in the 46-DOF state vector (matches state_mapper.py)
_OBS_IDX = jnp.array([0, 1, 2, 3, 4, 5])


# ─── Ring Buffer ──────────────────────────────────────────────────────────────

class RingBuffer:
    """
    Lock-protected circular buffer for (s_t, u_t, s_{t+1}, precision_weights) tuples.

    Writes from ROS2 subscriber thread; reads from sysid adapt thread.
    Uses a single RLock (low contention at 50 Hz write rate).
    """

    def __init__(self, capacity: int, state_dim: int = 46, ctrl_dim: int = 2):
        self._cap = capacity
        self._s   = np.zeros((capacity, state_dim), dtype=np.float32)
        self._u   = np.zeros((capacity, ctrl_dim),  dtype=np.float32)
        self._s1  = np.zeros((capacity, state_dim), dtype=np.float32)
        self._w   = np.ones( (capacity, 6),         dtype=np.float32)  # precision weights
        self._head  = 0
        self._count = 0
        self._lock  = threading.RLock()

    def push(
        self,
        s_t:  np.ndarray,  # [46] or [6] state at t
        u_t:  np.ndarray,  # [2]  control at t
        s_t1: np.ndarray,  # [46] or [6] state at t+1 (MirenaSim ground truth)
        w_t:  np.ndarray,  # [6]  precision weights from Car.msg covariance
    ) -> None:
        """Thread-safe push. Overwrites oldest entry when full (circular)."""
        s_full  = _pad_to_46(s_t)
        s1_full = _pad_to_46(s_t1)

        with self._lock:
            i            = self._head % self._cap
            self._s[i]   = s_full
            self._u[i]   = u_t[:2]
            self._s1[i]  = s1_full
            self._w[i]   = w_t[:6]
            self._head  += 1
            self._count  = min(self._count + 1, self._cap)

    def sample(self, n: int) -> Optional[tuple]:
        """
        Returns a random mini-batch of n tuples, or None if buffer is too small.
        Copies data under lock, then releases — JAX arrays built outside lock.
        """
        with self._lock:
            if self._count < n:
                return None
            idx  = np.random.choice(self._count, size=n, replace=False)
            s_b  = self._s[:self._count][idx].copy()
            u_b  = self._u[:self._count][idx].copy()
            s1_b = self._s1[:self._count][idx].copy()
            w_b  = self._w[:self._count][idx].copy()

        return (
            jnp.array(s_b),
            jnp.array(u_b),
            jnp.array(s1_b),
            jnp.array(w_b),
        )

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


def _pad_to_46(s: np.ndarray) -> np.ndarray:
    """Zero-pad a 6-DOF observation to 46-DOF (unobserved dims = 0)."""
    if s.shape[0] == 46:
        return s.astype(np.float32)
    out      = np.zeros(46, dtype=np.float32)
    out[:len(s)] = s[:46]
    return out


# ─── JIT-compiled loss and gradient ──────────────────────────────────────────

def build_loss_and_grad(
    forward_fn: Callable,
) -> Callable:
    """
    Builds a JIT-compiled (loss, grad) function for one adaptation step.

    forward_fn signature:
        forward_fn(params: PyTree, state_46d: jnp.ndarray, control_2d: jnp.ndarray)
            → next_state_46d: jnp.ndarray  [46]

    The returned function can be called as:
        loss, grads = val_and_grad_fn(params, s_batch, u_batch, s1_batch, w_batch)

    All internal ops are XLA-compilable:
      - vmap over batch dimension (no Python loop)
      - jnp.sum / jnp.mean for reduction
      - No conditionals on dynamic values
    """
    @jax.jit
    def _loss(
        params:   dict,
        s_batch:  jnp.ndarray,   # [B, 46]
        u_batch:  jnp.ndarray,   # [B, 2]
        s1_batch: jnp.ndarray,   # [B, 46] ground truth
        w_batch:  jnp.ndarray,   # [B, 6]  precision weights
    ) -> jnp.ndarray:            # scalar
        # vmap the forward step over the batch — single XLA kernel
        pred_fn  = jax.vmap(lambda s, u: forward_fn(params, s, u))
        s1_pred  = pred_fn(s_batch, u_batch)                       # [B, 46]

        # Residual on observable DOFs only — MirenaSim only measures 6 states
        pred_obs = s1_pred[:, _OBS_IDX]                            # [B, 6]
        true_obs = s1_batch[:, _OBS_IDX]                           # [B, 6]
        res      = pred_obs - true_obs                             # [B, 6]

        # Precision-weighted MSE: down-weights noisy observations
        loss_per = jnp.sum(w_batch * res ** 2, axis=-1)            # [B]
        return jnp.mean(loss_per)                                  # scalar

    return jax.value_and_grad(_loss)


# ─── EMA parameter shadow ─────────────────────────────────────────────────────

def _ema_update(ema_params: dict, new_params: dict, decay: float) -> dict:
    """Exponential moving average: ema ← decay·ema + (1-decay)·new."""
    return jax.tree_util.tree_map(
        lambda e, n: decay * e + (1.0 - decay) * n,
        ema_params, new_params,
    )


# ─── OnlineSysID ─────────────────────────────────────────────────────────────

class OnlineSysID:
    """
    Background daemon that continuously adapts Project-GP's H_net parameters
    to minimise rollout residuals against MirenaSim ground-truth transitions.

    Usage:
        # Initialise with H_net params and the JIT-compiled forward step
        sysid = OnlineSysID(project_gp_forward_fn, initial_params)
        sysid.start()

        # In ROS2 /car/state callback, after computing s_{t+1}:
        sysid.push(s_t_np, u_t_np, s_t1_np, precision_weights_np)

        # In inference thread, before running powertrain_step:
        current_params = sysid.get_params()

        # On shutdown:
        sysid.stop()
    """

    def __init__(
        self,
        forward_fn:     Callable,         # (params, s[46], u[2]) → s_next[46]
        init_params:    dict,             # initial H_net params pytree
        ring_capacity:  int   = RING_CAPACITY,
        adapt_period:   int   = ADAPT_PERIOD,
        batch_size:     int   = BATCH_SIZE,
        learning_rate:  float = LEARNING_RATE,
    ):
        self._buf         = RingBuffer(ring_capacity)
        self._period      = adapt_period
        self._batch_sz    = batch_size
        self._push_count  = 0

        # Optax: Adam + global gradient norm clipping
        self._opt         = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP_NORM),
            optax.adam(learning_rate),
        )
        self._opt_state   = self._opt.init(init_params)

        # Params: live copy (inference thread reads this)
        self._params      = init_params
        self._ema_params  = init_params  # EMA backstop
        self._param_lock  = threading.RLock()

        # Loss baseline for spike detection
        self._loss_baseline: Optional[float] = None

        # JIT-compiled value_and_grad
        self._val_grad_fn = build_loss_and_grad(forward_fn)

        # Step counter for logging
        self._adapt_steps = 0

        # Threading primitives
        self._new_data = threading.Event()
        self._running  = threading.Event()
        self._thread   = threading.Thread(
            target=self._adapt_loop, daemon=True, name="sysid_adapt",
        )

    def start(self) -> None:
        self._running.set()
        self._thread.start()
        log.info("OnlineSysID: adaptation thread started.")

    def stop(self) -> None:
        self._running.clear()
        self._new_data.set()  # unblock waiting thread
        self._thread.join(timeout=2.0)
        log.info(f"OnlineSysID: stopped after {self._adapt_steps} adapt steps.")

    def push(
        self,
        s_t:  np.ndarray,  # [46] or [6] state at t
        u_t:  np.ndarray,  # [2]  control at t
        s_t1: np.ndarray,  # [46] or [6] state at t+1
        w_t:  np.ndarray,  # [6]  precision weights (1/σ²) from covariance
    ) -> None:
        """
        Push one (s_t, u_t, s_{t+1}, weights) tuple. Signals adapt thread
        every ADAPT_PERIOD pushes. Thread-safe; non-blocking.
        """
        self._buf.push(s_t, u_t, s_t1, w_t)
        self._push_count += 1
        if (self._push_count % self._period == 0
                and self._buf.count >= WARMUP_FRAMES):
            self._new_data.set()

    def get_params(self) -> dict:
        """
        Returns current adapted params. Uses EMA copy for stability
        (EMA lags by ~1/( 1-EMA_DECAY) = 200 steps but is smoother).
        Inference thread should call this before each powertrain_step().
        """
        with self._param_lock:
            # Shallow pytree copy — safe because JAX arrays are immutable
            return jax.tree_util.tree_map(lambda x: x, self._ema_params)

    def get_live_params(self) -> dict:
        """Returns the latest (non-EMA) Adam-updated params. Less stable."""
        with self._param_lock:
            return jax.tree_util.tree_map(lambda x: x, self._params)

    # ── Adapt loop ────────────────────────────────────────────────────────────

    def _adapt_loop(self) -> None:
        log.info("OnlineSysID: adapt loop running.")
        while self._running.is_set():
            triggered = self._new_data.wait(timeout=5.0)
            self._new_data.clear()

            if not self._running.is_set():
                break
            if not triggered:
                continue  # Timeout: no new data, keep waiting

            batch = self._buf.sample(self._batch_sz)
            if batch is None:
                continue

            s_b, u_b, s1_b, w_b = batch

            with self._param_lock:
                params     = self._params
                opt_state  = self._opt_state

            # ── Adaptation step ───────────────────────────────────────────────
            try:
                loss, grads = self._val_grad_fn(params, s_b, u_b, s1_b, w_b)

                # Spike detection: pause if loss blows up
                loss_val = float(loss)
                if not np.isfinite(loss_val):
                    log.warning(f"OnlineSysID: non-finite loss={loss_val}, skipping step.")
                    continue
                if self._loss_baseline is None:
                    self._loss_baseline = loss_val
                elif loss_val > LOSS_SPIKE_THR * self._loss_baseline:
                    log.warning(
                        f"OnlineSysID: loss spike {loss_val:.4f} > "
                        f"{LOSS_SPIKE_THR}× baseline {self._loss_baseline:.4f}, skipping."
                    )
                    continue
                else:
                    # EMA of loss baseline for slow drift tracking
                    self._loss_baseline = 0.99 * self._loss_baseline + 0.01 * loss_val

                updates, new_opt_state = self._opt.update(grads, opt_state, params)
                new_params             = optax.apply_updates(params, updates)
                new_ema                = _ema_update(self._ema_params, new_params, EMA_DECAY)

                with self._param_lock:
                    self._params      = new_params
                    self._ema_params  = new_ema
                    self._opt_state   = new_opt_state

                self._adapt_steps += 1
                if self._adapt_steps % 50 == 0:
                    log.info(
                        f"OnlineSysID: step {self._adapt_steps:5d} | "
                        f"loss={loss_val:.6f} | baseline={self._loss_baseline:.6f}"
                    )

            except Exception as exc:
                log.error(f"OnlineSysID: adapt step failed: {exc}", exc_info=True)
