# benchmarks/baselines/__init__.py
# Project-GP — Baseline model registry
# ═══════════════════════════════════════════════════════════════════════════════

from benchmarks.baselines.neural_ode import NeuralODE
from benchmarks.baselines.pinn import PINN
from benchmarks.baselines.hnn import HNN
from benchmarks.baselines.phnn import PHNN
from benchmarks.baselines.passive_hnet import PassiveHNetBaseline

BASELINE_REGISTRY = {
    "NeuralODE": NeuralODE,
    "PINN": PINN,
    "HNN": HNN,
    "PHNN": PHNN,
    "PassiveHNet": PassiveHNetBaseline,
}

__all__ = ["BASELINE_REGISTRY", "NeuralODE", "PINN", "HNN", "PHNN", "PassiveHNetBaseline"]
