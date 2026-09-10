# benchmarks/metrics/gradients.py
# Project-GP — Gradient Accuracy Metrics
# ═══════════════════════════════════════════════════════════════════════════════
"""
Metrics for comparing gradient computation methods:
- Relative error vs finite-difference reference
- Cosine similarity
- Per-component analysis
"""

from __future__ import annotations

import jax.numpy as jnp


def gradient_relative_error(g_test: jnp.ndarray,
                            g_ref: jnp.ndarray) -> dict:
    """
    Compute relative gradient error: e = ‖g_test - g_ref‖ / ‖g_ref‖.

    Args:
        g_test: gradient vector to evaluate
        g_ref: reference gradient (typically finite-difference)

    Returns:
        dict with relative_error, cosine_similarity, max_component_error
    """
    diff = g_test - g_ref
    ref_norm = jnp.linalg.norm(g_ref)

    rel_error = float(jnp.linalg.norm(diff) / jnp.maximum(ref_norm, 1e-12))

    # Cosine similarity
    cos_sim = float(jnp.dot(g_test, g_ref) / (
        jnp.linalg.norm(g_test) * ref_norm + 1e-12))

    # Per-component relative error
    component_error = jnp.abs(diff) / (jnp.abs(g_ref) + 1e-12)

    return {
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
        "max_component_error": float(jnp.max(component_error)),
        "mean_component_error": float(jnp.mean(component_error)),
        "l2_error": float(jnp.linalg.norm(diff)),
        "ref_norm": float(ref_norm),
    }


def gradient_statistics(g: jnp.ndarray) -> dict:
    """
    Compute statistics of a gradient vector (detect explosion/vanishing).

    Args:
        g: gradient vector

    Returns:
        dict with norm, max, min, mean, std, n_zeros, has_nan, has_inf
    """
    return {
        "norm": float(jnp.linalg.norm(g)),
        "max": float(jnp.max(jnp.abs(g))),
        "min": float(jnp.min(jnp.abs(g))),
        "mean": float(jnp.mean(jnp.abs(g))),
        "std": float(jnp.std(g)),
        "n_zeros": int(jnp.sum(jnp.abs(g) < 1e-12)),
        "has_nan": bool(jnp.any(jnp.isnan(g))),
        "has_inf": bool(jnp.any(jnp.isinf(g))),
    }
