# powertrain/__init__.py
# Project-GP — Differentiable Powertrain Control Stack (Ter27 4WD)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Modules:
#   motor_model         PMSM electromechanical + thermal model
#   torque_vectoring    SOCP control allocator + CBF safety filter
#   traction_control    DESC extremum-seeking slip controller
#   launch_control      Neural predictive launch sequencer
#   virtual_impedance   PIO mitigation via virtual mechanical impedance
#   powertrain_manager  Unified coordinator binding all modules
#
# All modules are 100% native JAX. No numpy inside traced functions.
# Every path is differentiable — gradients flow from lap time through
# torque allocation to suspension setup parameters.
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = '0.1.0'