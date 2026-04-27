# tests/conftest.py
# Project-GP — Shared test infrastructure
# =============================================================================
#
# Single source of truth for: project-root sys.path injection, JAX cache
# isolation, deterministic seeds, shared synthetic hardpoints, and a
# lightweight TestResult class compatible with both pytest and bare-Python
# execution. Imported transparently by pytest; for stand-alone scripts use:
#
#     from tests.conftest import (TestResult, suppress_jax_logs,
#                                 FRONT_HPTS, REAR_HPTS, get_vp, get_tc)
#
# Fixture scope is `module` — heavy JAX-traced objects (vehicle, kinematics,
# tire) are reused across all tests in a file to amortize JIT compile cost.
# =============================================================================

from __future__ import annotations

import os
import sys
import math
import time
import contextlib
import logging
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

# ── 1. Project root injection ────────────────────────────────────────────────
# Resolves regardless of where pytest is invoked from.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 2. JAX environment isolation ─────────────────────────────────────────────
# Stale .jax_cache from prior runs is the #1 cause of mysterious "old code is
# running" symptoms. Tests get a private cache directory that does not
# collide with the development cache.
os.environ.setdefault("JAX_PLATFORM_NAME", os.environ.get("JAX_PLATFORM_NAME", "cpu"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

try:
    import jax_config  # noqa: F401  — sets XLA cache, parallelism, etc.
except Exception:
    pass

import jax
import jax.numpy as jnp

# Suppress JAX/Flax info-level chatter inside the test suite.
for name in ("jax", "absl", "flax"):
    logging.getLogger(name).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# §3  TestResult — lightweight scoreboard usable from any context
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    """
    Aggregates [PASS]/[FAIL]/[WARN] outcomes with timing.
    Designed to coexist with pytest assertions: both can be used in the same
    file. Pytest sees `assert` failures; this class records them for reports.
    """

    __slots__ = ("name", "passed", "failed", "warned", "fails", "warns", "_t0")

    def __init__(self, name: str = "<suite>"):
        self.name   = name
        self.passed = 0
        self.failed = 0
        self.warned = 0
        self.fails: list[str] = []
        self.warns: list[str] = []
        self._t0    = time.perf_counter()

    # — Outcome verbs ————————————————————————————————————————————
    def ok(self, label: str) -> None:
        self.passed += 1
        print(f"  [PASS] {label}")

    def fail(self, label: str, reason: str = "") -> None:
        self.failed += 1
        msg = f"{label}: {reason}" if reason else label
        self.fails.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, label: str, reason: str = "") -> None:
        self.warned += 1
        msg = f"{label}: {reason}" if reason else label
        self.warns.append(msg)
        print(f"  [WARN] {msg}")

    # — Convenience predicates ——————————————————————————————————
    def check(self, cond: bool, label: str, fail_msg: str = "") -> bool:
        (self.ok if cond else self.fail)(label) if cond else self.fail(label, fail_msg)
        return cond

    def close(self, condition: bool, ok_label: str, fail_label: str,
              tol: float, value: float) -> bool:
        msg = f"|x|={abs(value):.3e} (tol={tol:.1e})"
        if condition:
            self.ok(f"{ok_label} — {msg}")
        else:
            self.fail(fail_label, msg)
        return condition

    def summary(self) -> bool:
        dt = time.perf_counter() - self._t0
        total = self.passed + self.failed
        print(f"\n{'─' * 62}")
        print(f"  {self.name}  —  {self.passed}/{total} PASS, "
              f"{self.warned} WARN, {self.failed} FAIL  ({dt:.1f}s)")
        if self.fails:
            print("  FAILURES:")
            for f in self.fails:
                print(f"    · {f}")
        print(f"{'─' * 62}")
        return self.failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# §4  Output suppression context — for tests that intentionally exercise
#      noisy code paths (training loops, MORL inits) and we only care about
#      the assertion outcome.
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_jax_logs() -> Iterator[None]:
    """Suppress stdout from the wrapped block (NOT stderr — tracebacks survive)."""
    import io
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved


# ─────────────────────────────────────────────────────────────────────────────
# §5  Synthetic hardpoints — match Velis geometry without requiring Excel
# ─────────────────────────────────────────────────────────────────────────────
# Re-exported from test_kinematics so all suspension tests share one source.

FRONT_HPTS = {
    "CHAS_LowFor": np.array([0.160,  0.160,  0.110]),
    "CHAS_LowAft": np.array([-0.160, 0.160,  0.130]),
    "CHAS_UppFor": np.array([0.120,  0.245,  0.267]),
    "CHAS_UppAft": np.array([-0.120, 0.245,  0.258]),
    "UPRI_LowPnt": np.array([0.00227, 0.583374, 0.12265]),
    "UPRI_UppPnt": np.array([-0.011496, 0.55563, 0.280]),
    "CHAS_TiePnt": np.array([0.050,  0.14478, 0.1445]),
    "UPRI_TiePnt": np.array([0.070,  0.571,   0.150]),
    "NSMA_PPAttPnt_L": np.array([-0.00351, 0.51471,  0.29418]),
    "CHAS_AttPnt_L":   np.array([-0.17933, 0.150,    0.61479]),
    "CHAS_RocAxi_L":   np.array([0.00067,  0.22753,  0.61211]),
    "CHAS_RocPiv_L":   np.array([0.00067,  0.19506,  0.57518]),
    "ROCK_RodPnt_L":   np.array([0.05998,  0.20185,  0.56921]),
    "ROCK_CoiPnt_L":   np.array([0.00067,  0.150,    0.61479]),
    "NSMA_UBarAttPnt_L": np.array([0.00067, 0.17253, 0.59498]),
    "UBAR_AttPnt_L":     np.array([-0.19933, 0.17253, 0.59499]),
    "CHAS_PivPnt_L":     np.array([-0.19933, 0.17253, 0.63499]),
    "Half Track_m": 0.615,
    "R_wheel":      0.2032,
    "spring_rate_N_per_m": 44000.0,
    "ubar_stiffness_Nm_per_rad": 286.5,
    "actuation_type": "pushrod",
}

REAR_HPTS = {
    "CHAS_LowFor": np.array([0.150,   0.240,   0.1262]),
    "CHAS_LowAft": np.array([-0.150,  0.240,   0.120]),
    "CHAS_UppFor": np.array([0.150,   0.240,   0.282]),
    "CHAS_UppAft": np.array([-0.150,  0.240,   0.250]),
    "UPRI_LowPnt": np.array([0.000,   0.57678, 0.11265]),
    "UPRI_UppPnt": np.array([0.000,   0.520001,0.280]),
    "CHAS_TiePnt": np.array([-0.095,  0.275,   0.125]),
    "UPRI_TiePnt": np.array([-0.080,  0.590,   0.1658]),
    "NSMA_PPAttPnt_L": np.array([0.00893, 0.49739, 0.29758]),
    "CHAS_AttPnt_L":   np.array([-0.030,  0.050,   0.430]),
    "CHAS_RocAxi_L":   np.array([0.07451,  0.11973, 0.58004]),
    "CHAS_RocPiv_L":   np.array([0.10743,  0.10826, 0.54713]),
    "ROCK_RodPnt_L":   np.array([0.14842,  0.20010, 0.57238]),
    "ROCK_CoiPnt_L":   np.array([0.09728,  0.050,   0.55728]),
    "NSMA_UBarAttPnt_L": np.array([0.09728, 0.080,  0.56328]),
    "UBAR_AttPnt_L":     np.array([0.000,   0.080,  0.450]),
    "CHAS_PivPnt_L":     np.array([0.020,   0.080,  0.436]),
    "Half Track_m": 0.615,
    "R_wheel":      0.2032,
    "spring_rate_N_per_m": 53000.0,
    "ubar_stiffness_Nm_per_rad": 286.5,
    "actuation_type": "pullrod",
    "passive_rear_steer": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# §6  Lazy loaders — heavy modules. Cached so first-call cost is paid once.
# ─────────────────────────────────────────────────────────────────────────────

_VP_CACHE = None
_TC_CACHE = None


def get_vp(car: str = "ter26") -> dict:
    """Returns the active vehicle_params dict, falling back across import paths."""
    global _VP_CACHE
    if _VP_CACHE is not None:
        return _VP_CACHE
    try:
        if car == "ter27":
            from config.vehicles.ter27 import vehicle_params_ter27 as VP
        else:
            from config.vehicles.ter26 import vehicle_params as VP
    except ImportError:
        # Legacy fall-through (older revisions used data.configs)
        from data.configs.vehicle_params import vehicle_params as VP  # type: ignore
    _VP_CACHE = VP
    return VP


def get_tc() -> dict:
    """Returns tire_coeffs dict, falling back across import paths."""
    global _TC_CACHE
    if _TC_CACHE is not None:
        return _TC_CACHE
    try:
        from config.tire_coeffs import tire_coeffs as TC
    except ImportError:
        from data.configs.tire_coeffs import tire_coeffs as TC  # type: ignore
    _TC_CACHE = TC
    return TC


# ─────────────────────────────────────────────────────────────────────────────
# §7  pytest fixtures (used only when invoked under pytest)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pytest

    @pytest.fixture(scope="module")
    def vp():
        return get_vp()

    @pytest.fixture(scope="module")
    def tc():
        return get_tc()

    @pytest.fixture(scope="module")
    def front_kin():
        from suspension.kinematics import SuspensionKinematics
        return SuspensionKinematics(FRONT_HPTS, side="left")

    @pytest.fixture(scope="module")
    def rear_kin():
        from suspension.kinematics import SuspensionKinematics
        return SuspensionKinematics(REAR_HPTS, side="left")

    @pytest.fixture(scope="module")
    def front_kin_right():
        from suspension.kinematics import SuspensionKinematics
        return SuspensionKinematics(FRONT_HPTS, side="right")

    @pytest.fixture(scope="module")
    def vehicle(vp, tc):
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
        return DifferentiableMultiBodyVehicle(vp, tc)

    @pytest.fixture(scope="module")
    def tire(tc):
        from models.tire_model import PacejkaTire
        return PacejkaTire(tc, rng_seed=0)

except ImportError:  # pytest not installed in this environment
    pytest = None    # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# §8  Numeric helpers shared across all test files
# ─────────────────────────────────────────────────────────────────────────────

def finite_grad(fn, *args, eps: float = 1e-4) -> jax.Array:
    """Central finite difference on the FIRST argument."""
    x = args[0]
    if jnp.ndim(x) == 0:
        return (fn(x + eps, *args[1:]) - fn(x - eps, *args[1:])) / (2 * eps)
    grads = []
    for i in range(x.shape[0]):
        e = jnp.zeros_like(x).at[i].set(eps)
        grads.append((fn(x + e, *args[1:]) - fn(x - e, *args[1:])) / (2 * eps))
    return jnp.array(grads)


def is_psd(M: jax.Array, tol: float = -1e-6) -> bool:
    """Returns True if M is symmetric and all eigenvalues ≥ tol."""
    M_np = np.array(M)
    if not np.allclose(M_np, M_np.T, atol=1e-5):
        return False
    eigs = np.linalg.eigvalsh(0.5 * (M_np + M_np.T))
    return bool(np.min(eigs) >= tol)


def all_finite(*arrs) -> bool:
    """True iff every element of every input array is finite."""
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(a)))) for a in arrs)