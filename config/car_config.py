# config/car_config.py
# Project-GP — Multi-Car Configuration Loader
# ═══════════════════════════════════════════════════════════════════════════════
# Single entry point for selecting the active car configuration.
#
# USAGE:
#   from config.car_config import get_car_config, get_design_bounds
#
#   cfg = get_car_config('ter27')
#   vp  = cfg['vehicle_params']
#   lb, ub = get_design_bounds('ter27')
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
from typing import Dict, Tuple, Any
import jax.numpy as jnp

_CAR_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _register_ter26():
    from config.vehicles.ter26 import vehicle_params
    _CAR_REGISTRY['ter26'] = {
        'vehicle_params': vehicle_params,
        'drivetrain':     'rwd',
        'setup_dim':      28,
        'season':         2026,
        'label':          'Ter26 RWD',
    }

def _register_ter27():
    from config.vehicles.ter27 import vehicle_params_ter27
    _CAR_REGISTRY['ter27'] = {
        'vehicle_params': vehicle_params_ter27,
        'drivetrain':     'awd',
        'setup_dim':      28,
        'season':         2027,
        'label':          'Ter27 4WD',
    }

def get_car_config(car_id: str = 'ter26') -> Dict[str, Any]:
    """Returns full configuration dict for the requested car. Lazy-loads on first access."""
    car_id = car_id.lower().strip()
    if car_id not in _CAR_REGISTRY:
        if car_id == 'ter26':
            _register_ter26()
        elif car_id == 'ter27':
            _register_ter27()
        else:
            raise ValueError(f"Unknown car_id '{car_id}'. Valid: 'ter26', 'ter27'")
    return _CAR_REGISTRY[car_id]

def get_design_bounds(car_id: str = 'ter26') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (lower_bounds, upper_bounds) arrays for the setup optimizer.
    Delegates to the per-car bounds defined in config/vehicles/.
    """
    cfg = get_car_config(car_id)
    if car_id == 'ter26':
        from config.vehicles.ter26 import get_design_bounds as _bounds
    else:
        from config.vehicles.ter27 import get_design_bounds as _bounds
    return _bounds()