# TTC tire-model benchmark protocol

## Decision and hypothesis

Evaluate only the repository's existing, fixed-coefficient Pacejka/MF6.2 tire
formulation on an independent, complete-run-held-out TTC data set. The initial
hypothesis is limited: its measured force error and load/slip-angle dependence
can be quantified on a new test run. It is not a claim of superiority over an
invented baseline or of full-vehicle validity.

## Data and partition

The available processed data are steady-state cornering samples from TTC Round
9 Hoosier runs 4, 5, and 6. The locked partition is train run **4** (35,300
samples), validation run **5** (31,131), and test run **6** (52,293). Complete
runs are exclusive to one partition. The archive's `is_test` flag is not used:
it assigns approximately 20% of rows within *each* run and would leak run
conditions across partitions.

## Preprocessing

Use only the stored SI channels: slip angle, longitudinal slip, Fx, Fy, Fz,
Mz, camber, pressure, belt speed, and three surface temperatures. All learnt
input/target statistics are fit on run 4 alone, serialized, and then applied
unchanged to runs 5 and 6. The pipeline must use `Vx_ms`, the actual archive
field; prior scripts request `Vx_mps` and silently substitute a constant.
Sign convention must be fixed from the training source/documentation before
test evaluation—never flipped after inspecting test correlation.

## Compared formulations

Phase 1 compares the existing analytical Pacejka/MF6.2 formulation with its
PINN/GP correction disabled. This is the only immediately qualified model.
Phase 2 may compare the existing `TireOperatorPINN` residual formulation only
after retraining it exclusively on run 4 with a corrected input adapter and
validation-only model selection. The persisted `pinn_params.bytes` has no
training split/provenance manifest, so it cannot be used as benchmark evidence.
No standard-GP uncertainty claim is permitted: the class name is historical;
the implementation uses a spectral-mixture kernel and a nonstandard variance
construction. No calibrated inducing-point artifact is currently present.

## Outcomes

Primary outcomes are Fy error (MAE, RMSE, signed bias in N) and lateral
friction-coefficient error `Fy/Fz`; report the same force metrics in fixed Fz
bins and Fy-versus-slip-angle curves at supportable load/camber conditions.
Report Fx and `Fx/Fz` as secondary near-zero-slip diagnostics only: all stored
`kappa` values are zero, so the data cannot validate longitudinal-slip or
combined-slip prediction. Report test coverage outside the training range for
slip angle, Fz, camber, pressure, belt speed, and temperatures.

Plots are: held-out Fy-versus-slip-angle curves, absolute Fy error versus Fz,
measured/predicted lateral friction versus Fz, and train-versus-test operating
range coverage. With only one independent test run, report point estimates and
condition-bin sample counts; do not present row-wise confidence intervals or
p-values as independent statistical evidence.

## Evidence and cost

A successful, reproducible result is conditional **Tier B independent tire
submodel evidence**. It does not validate the 108-state vehicle model, the GP
uncertainty interpretation, longitudinal-slip behavior, or generalization
beyond this tire/run family. The present pilot only builds and audits split and
scaler metadata; it performs no model execution or training and should take a
few seconds. Full training is explicitly deferred.
