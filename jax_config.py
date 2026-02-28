"""
jax_config.py  —  Project-GP shared JAX startup configuration.

Import this module at the very top of every script that uses JAX,
BEFORE any other jax import:

    import jax_config          # must be first
    import jax
    import jax.numpy as jnp
    ...

Why order matters
-----------------
JAX reads XLA flags and cache settings at the moment the first JAX
operation is traced. Importing jax_config first guarantees the cache
directory and compilation flags are set before any tracing occurs.
Importing it after jax.numpy has been used is too late — the settings
are ignored silently.
"""

import os

# ── Locate project root (this file lives at the project root) ─────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR   = os.path.join(_HERE, '.jax_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)

# ── Memory allocation ─────────────────────────────────────────────────────────
# Disable the default "preallocate 90 % of GPU/device memory" behaviour.
# On CPU this has no effect; on GPU it prevents OOM when other processes
# share the device.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE',    'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION',   '0.80')

# ── Parallelism flags ─────────────────────────────────────────────────────────
# Allow XLA to use all available CPU threads for compilation.
# On an 8-core machine this cuts compile time by 3-5x.
cpu_count = os.cpu_count() or 4
os.environ.setdefault('XLA_FLAGS',
    f'--xla_cpu_multi_thread_eigen=true '
    f'--intra_op_parallelism_threads={cpu_count} '
    f'--inter_op_parallelism_threads={cpu_count}')

# ── Now import JAX and apply Python-side config ───────────────────────────────
import jax

# Persistent compilation cache — compiled XLA modules are stored on disk
# and reused across process restarts. First compile still takes ~3-5 min on CPU;
# subsequent runs (cache hit) take 5-30 seconds.
jax.config.update('jax_compilation_cache_dir', _CACHE_DIR)

# Only cache modules that took more than 5 seconds to compile.
# This avoids cluttering the cache with trivial small ops.
jax.config.update('jax_persistent_cache_min_compile_time_secs', 5.0)

# 64-bit floats off by default (JAX default). Keep 32-bit for physics:
# faster SIMD, smaller memory footprint, compatible with most ML libraries.
# Enable this only if you observe numerical precision issues in the ODE integrator.
# jax.config.update('jax_enable_x64', True)

print(f"[JAX Config] XLA cache : {_CACHE_DIR}")
print(f"[JAX Config] Device    : {jax.default_backend()}")
print(f"[JAX Config] Devices   : {jax.devices()}")