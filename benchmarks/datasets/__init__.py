# benchmarks/datasets/__init__.py
# Project-GP — Dataset generators for benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

from benchmarks.datasets.interpolation import generate_interpolation_dataset
from benchmarks.datasets.extrapolation import generate_extrapolation_splits
from benchmarks.datasets.noise import add_measurement_noise
from benchmarks.datasets.setup_generalization import generate_setup_generalization_splits

__all__ = [
    "generate_interpolation_dataset",
    "generate_extrapolation_splits",
    "add_measurement_noise",
    "generate_setup_generalization_splits",
]
