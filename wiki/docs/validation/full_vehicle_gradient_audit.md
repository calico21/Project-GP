# Full-vehicle gradient audit

`experiments/full_vehicle_gradient_audit.py` is the source-of-record diagnostic
for the unresolved one-step vehicle gradient audit. It is intentionally not a
production solver and does not alter `PassiveHNet` or the vehicle equations.

## Exact production trace

For the canonical audit case, `make_case()` constructs the following live
inputs: a 108-state x64 initial state (`make_initial_state(vx0=20)`), a
28-element x64 physical setup vector, six differentiated setup entries, fixed
controls, and `dt=0.005`.

```
theta -> setup.at[:n].set(theta)
      -> simulate_step(x0, controls, setup, dt, n_substeps=1)
      -> setup conversion (array path; x64 cast)
      -> lax.scan(checkpoint(_glrk4_step), length=1)
      -> _glrk4_step:
           dx0 = _compute_derivatives(x0, controls, setup)
           (k1, k2) = (dx0, dx0)
           lax.scan(Picard update, length=32)
             x_i = x0 + dt * A_i @ (k1, k2)
             k_i = clip(_compute_derivatives(x_i, controls, setup), -500, 500)
           x_next = x0 + dt/2 * (k1 + k2)
           x_next[28:108] = clip(x_next[28:108], -1000, 1000)
      -> benchmark objective sum((x_next[:28] - target[:28])**2)
```

The state vector is unpacked by `_compute_derivatives` as `q[0:14]`,
`p/M[14:28]`, thermal `[28:56]`, transient slip `[56:72]`, damper `[72:84]`,
and elastokinematic hysteresis `[84:108]`. All these blocks are live stage
variables. Setup affects the neural energy conditioning, springs, dampers,
bump stops, anti-roll bars, geometry, tire inputs, brake bias, and CG-height
load transfer. The external-force, thermal, slip, damper and hysteresis terms
are evaluated inside each stage.

`production AD` is `jax.grad` through this finite 32-iteration map.
`unrolled AD` in the audit is independently written but has the same initial
guess and iterations. `IFT` instead differentiates `F(z, theta)=0`, with a
complete `z in R^216`, at the terminal stages. It is a root-map reference only
when its reported residual is small. `FD` is a central difference of the
actual `simulate_step` objective, so it evaluates the finite production map.

## Gradient-sensitive operations found in the production path

The audit reports whether the stage-rate and output clips are active at every
FD perturbation. Additional hard or piecewise operations in the transitively
called production path include state, velocity, force, load, tire, thermal and
aero `clip`/`maximum`/`minimum` operations. The most material source locations
are `models/vehicle_dynamics.py:514,860-861,905-934,996-1032,1415,1423`,
`models/tire_model.py:606-733`, `models/tire_thermal_3d.py:179-330`,
`models/tire_transient.py:116-241`, and `models/aero_platform.py:206-381`.
`tire_model.py` deliberately uses `stop_gradient` for fixed GP factors; those
factors are not differentiated setup inputs. The only custom VJP found,
`_clip_adjoint`, is currently unused by this path. The Python setup-type and
shape branches are trace/static branches; the canonical audit takes the array
path. There is no `detach` in this path. The vehicle energy force intentionally
retains its setup dependency through the nested energy gradient.

## Running and interpreting

Run with x64 enabled:

```bash
JAX_PLATFORMS=cpu python experiments/full_vehicle_gradient_audit.py
```

The generated JSON has exact state-reconstruction errors by state block,
stage residual, singular values and condition number of `F_z`, the IFT linear
identity norm, objective hierarchy, FD sweep, vector-map FD checks for `z` and
`x_next`, and a discrepancy-sorted parameter table. Do not claim FD validates
the implicit root map unless the reported maps and residual justify it.
