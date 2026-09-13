# Data-source and comparability design study

## Decision

**No-go for a genuinely independent 108-state predictive benchmark with the
current repository and available data.** The missing ingredient is an
independent source of all 108 target states (or an independently validated
state-estimation pipeline that reconstructs them with quantified uncertainty).

Do not train or present a five-family 108-state predictive comparison until
that ingredient exists. The existing 28-state benchmark remains legacy Tier C
evidence and is not changed by this study.

## Candidate target sources

| Candidate | State / input coverage | Independent of PassiveHNet? | Splits and rollouts | Decision |
|---|---|---:|---|---|
| data/raw_can_logs/{1..5}.csv | Real CAN: steering, throttle/brake, IMU, GPS, vehicle speed, rear RPM/torque, limited temperatures; no 108-state reconstruction | Yes, subject to provenance confirmation | Separate sessions permit session-level train/validation/test and observed-channel rollouts | Best available independent source, but supports an **observed-channel telemetry benchmark**, not a 108-state target |
| data/telemetry/raw/{1..5}.csv, logs/usable_signals.csv | Alternate/raw decoded CAN representations; same limited observable class | Likely yes, subject to duplicate-session/provenance audit | Potential session splits; raw asynchronous signals require documented resampling | Supporting telemetry material, not a full-state source |
| data/ttc_round9/*.mat, processed.npz | Independent tyre-test channels: slip, load, force, temperature, pressure, speed | Yes | Run IDs provide hold-out runs; no vehicle trajectories | Valid **tyre submodel** benchmark only |
| trained/koopman_tv/telemetry_X.npy, telemetry_U.npy | 2-state / 2-input derived telemetry arrays | Uncertain; provenance is not persisted with arrays | Could be split after provenance audit; no 108 state | Insufficient for full vehicle |
| reports/calib_window0_debug.npz | Simulated/real comparison for three channels | Mixed/derived | Small diagnostic windows only | Not a target dataset |
| simulator/logs/*.csv, simulator/physics_server.py | 57–64 telemetry fields emitted by DifferentiableMultiBodyVehicle | No: server imports the production vehicle map | Rollouts possible, but circular | Production-model structural/self-consistency only |
| benchmarks/datasets/* and synthetic fallback | Current generator explicitly calls DifferentiableMultiBodyVehicle; existing implementation stores first 28 states | No | Arbitrary generated splits/rollouts, but circular | Synthetic self-consistency only |
| QP/optimization arrays and trained model files | Optimizer/surrogate artifacts, not vehicle state transitions | Not applicable | Not applicable | Not target sources |

The CPU environment can ingest and preprocess the stored CAN and tyre files.
A full 108-state learned rollout campaign through the production implicit map
should not be budgeted from the present evidence: prior derivative work already
exceeded ten CPU minutes. A measured compile/forward/backward timing pilot is
required before committing a training budget.

## Study types: keep them separate

### A. Independent predictive benchmark

Use real telemetry only for the channels genuinely observed and reliably
time-aligned. Split by complete sessions/laps, never random rows; fit scalers
on training sessions only. This can support observed-channel prediction claims,
not claims about unobserved 108-state auxiliary variables.

### B. Production-model structural and solver study

Use DifferentiableMultiBodyVehicle.simulate_step as the 108-state, 6-control,
28-setup production GLRK reference. This supports solver, differentiability,
energy-structure, and setup-sensitivity evidence. It is not a predictive
competition when it also creates the labels.

### C. Synthetic self-consistency study

Use generated outputs of the production map only to test implementation,
scaling, recovery, and failure modes. Label it synthetic self-consistency or
emulation; never predictive superiority.

## Minimum common interface for a future independent benchmark

x_t in R^108, u_t in R^6, s in R^28, fixed documented dt, and target
x_(t+1) in R^108. The state blocks are mechanical 0:28, thermal 28:56,
slip 56:72, and hysteresis/compliance 72:108.

Persist immutable train/validation/test_one_step/test_rollout tensors, IDs,
source-session IDs, SHA-256 hashes, source versions, and scalers. Fit
state/control/setup scalers on train only; retain centre, scale, constant-field
policy, units, and inverse transform. Evaluate normalized per-state error and
physical-unit group errors at 1/10/20/50/100/200 steps.

## Fair baseline definitions if an independent 108-state source becomes available

| Paper label | Exact definition |
|---|---|
| NeuralODE-108 | Controlled 108-state vector-field MLP using all 6 controls and 28 setup inputs |
| Mixed-state PINN-108 | 108-state controlled dynamics head; physics residual only on the documented 28-state mechanical sub-block |
| Controlled HNN + auxiliary head | Hamiltonian mechanical core with learned control-force map plus an explicit 80-state auxiliary dynamics head; never call this simply “HNN” |
| Controlled PHNN + auxiliary head | Port-Hamiltonian mechanical core with control port plus explicit learned 80-state auxiliary dynamics head; never call this simply “PHNN” |
| Production PassiveHNet-GLRK | Actual DifferentiableMultiBodyVehicle.simulate_step 108-state implicit GLRK map, with its complete 6-control/28-setup interface |

These are output- and input-matched, but they are not identically structured.
Report architecture, parameter count, optimizer, wall time, and control
dependence. Verify every wrapper has nonzero control sensitivity on a fixed
test case.

Production PassiveHNet cannot enter a predictive leaderboard against labels
generated by itself. It can be evaluated against an independent source, or be
reported separately as a physics reference in a structural study.

## Energy

No common learned-Hamiltonian comparison is valid: learned energy functions
have arbitrary offsets/scales and the generic baselines do not share the
production energy definition. If a standalone, unit-calibrated reference
mechanical-energy functional can be specified and validated, it may be
evaluated on every predicted state as a **common proxy**. Otherwise mark common
energy unavailable and retain energy/passivity claims for the structural study.

## Required artifacts, modules, and tests after the no-go is resolved

Required data artifact: independently sourced 108-state trajectories, or a
validated estimator with per-state uncertainty and control/setup provenance.

Then add isolated modules under experiments/paper_suite/benchmark_108/ for
data contracts, session splits, train-only scaling, model adapters, training,
evaluation, and reporting. Do not repurpose the legacy benchmark modules.

Lightweight gates:

1. State block, unit, and input/output shape contracts.
2. Session/lap-level split non-overlap and immutable hashes.
3. Train-only scaler leakage test and inverse-transform round trip.
4. Every model consumes controls and setup; output shape is exactly 108.
5. Production PassiveHNet adapter equals simulate_step at fixed inputs.
6. Checkpoint/optimizer/PRNG resume equivalence and manifest coverage.
7. Group-metric index mapping and common-energy availability contract.

## Go/no-go

**No-go.** Current data supports an independent observed-channel telemetry
benchmark and an independent tyre submodel benchmark, plus production-model
structural studies. It does not support a genuinely independent, scientifically
fair 108-state predictive benchmark.
