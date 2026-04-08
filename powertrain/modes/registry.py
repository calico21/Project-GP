# powertrain/modes/registry.py
from __future__ import annotations
from enum import Enum
import jax
import jax.numpy as jnp
from functools import partial
from typing import Protocol, runtime_checkable

class ControlMode(str, Enum):
    SIMPLE   = "simple"
    ADVANCED = "advanced"

@runtime_checkable
class TorqueAllocator(Protocol):
    def allocate(self, state, params, ref) -> jnp.ndarray: ...

def build_powertrain(mode: ControlMode):
    """
    Returns JIT-compiled powertrain callables resolved at Python-level.
    mode is a static_argnums axis — zero runtime branching in XLA graph.
    """
    if mode is ControlMode.SIMPLE:
        from powertrain.modes.simple.torque_vectoring import allocate_simple
        from powertrain.modes.simple.traction_control import tc_simple
        _allocate = allocate_simple
        _tc       = tc_simple
    else:
        from powertrain.modes.advanced.torque_vectoring import allocate_socp
        from powertrain.modes.advanced.traction_control import tc_desc
        _allocate = allocate_socp
        _tc       = tc_desc

    # Both branches share identical call signatures — polymorph at import time,
    # not inside the traced function. XLA sees a single code path.
    allocate_jit = jax.jit(_allocate)
    tc_jit       = jax.jit(_tc)
    return allocate_jit, tc_jit