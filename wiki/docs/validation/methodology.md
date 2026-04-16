# Twin Fidelity Methodology

## The Central Question

A digital twin is not a simulation. A simulation predicts what *might* happen. A twin
tracks what *is* happening — and the gap between the two is quantified, bounded, and
continuously reduced.

The Twin Fidelity Score is a scalar $\mathcal{F} \in [0, 100]\%$ that answers: **how
well does the model reproduce measured vehicle behaviour when driven with the same
inputs?**

---

## Open-Loop Validation Protocol

The gold standard for model validation is **open-loop replay**:

1. Record real vehicle telemetry: steering angle $\delta(t)$, throttle $\tau(t)$,
   brake $\beta(t)$, and measured outputs $y_{\text{real}}(t)$
2. Drive the digital twin with the *same* inputs $\delta(t), \tau(t), \beta(t)$
3. Compare the twin's predicted outputs $y_{\text{sim}}(t)$ against $y_{\text{real}}(t)$

This is a strict test: the model must reproduce the correct state evolution from
the correct inputs, without feedback correction. Any unmodelled dynamics — tyre
degradation, aerodynamic hysteresis, thermal transients — will show up as divergence.

```
Real Car:    δ(t), τ(t), β(t)  ──→  Vehicle  ──→  y_real(t)
                                        ↕ compare
Digital Twin: δ(t), τ(t), β(t) ──→  46-DOF PH  ──→  y_sim(t)
```

---

## Validation Channels

Three primary channels are validated, chosen because they are (a) measurable with
standard FS instrumentation, (b) sensitive to the key modelling assumptions, and
(c) complementary in the frequency domain:

| Channel | Symbol | Sensitive To | Frequency Content |
|---|---|---|---|
| Velocity | $v_x(t)$ | Tyre longitudinal grip, powertrain model, aero drag | Low (0–2 Hz) |
| Yaw rate | $\dot{\psi}(t)$ | Tyre lateral grip, suspension compliance, CG height | Mid (0–5 Hz) |
| Lateral acceleration | $a_y(t)$ | Combined grip, load transfer, aero balance | Mid-High (0–10 Hz) |

---

## Metrics

### 1. Coefficient of Determination ($R^2$)

$$
R^2 = 1 - \frac{\sum_i (y_{\text{real},i} - y_{\text{sim},i})^2}{\sum_i (y_{\text{real},i} - \bar{y}_{\text{real}})^2}
$$

$R^2 = 1.0$: perfect prediction. $R^2 = 0.0$: model no better than the mean.
$R^2 < 0$: model worse than predicting the mean (catastrophic failure).

### 2. Cross-Correlation Peak and Lag

$$
\rho(\tau) = \frac{1}{N\sigma_s\sigma_r} \sum_i (y_{\text{sim},i} - \bar{y}_s)(y_{\text{real},i+\tau} - \bar{y}_r)
$$

The peak $\rho^* = \max_\tau |\rho(\tau)|$ measures shape similarity regardless of
phase. The lag $\tau^* = \arg\max |\rho(\tau)|$ detects systematic timing errors
(transport delay, actuator lag). A good twin has $\rho^* > 0.98$ and $|\tau^*| < 10$ ms.

### 3. PSD Residual (Frequency-Domain Fidelity)

$$
\mathcal{P} = \sqrt{\frac{1}{N_f} \sum_k \left(\log_{10} S_{\text{sim}}(f_k) - \log_{10} S_{\text{real}}(f_k)\right)^2}
$$

where $S(f)$ is the power spectral density estimated via Welch's method. This metric
catches frequency-band-specific errors that $R^2$ might average out: a model that
reproduces the low-frequency trend but misses the 3 Hz yaw oscillation will score
high $R^2$ but poor $\mathcal{P}$.

### 4. Composite Twin Fidelity Score

$$
\boxed{\mathcal{F} = 100 \times \left(
0.30 \cdot R^2_{v_x} + 0.25 \cdot R^2_{\dot{\psi}} + 0.20 \cdot R^2_{a_y}
+ 0.15 \cdot \bar{\rho}^* + 0.10 \cdot \max(0, 1 - \bar{\mathcal{P}})
\right)}
$$

The weights reflect engineering priority: velocity fidelity matters most (it
determines lap time directly), yaw rate fidelity matters for stability prediction,
and frequency-domain fidelity provides the "fine structure" assessment.

---

## Sensor Noise Model

Real sensors introduce noise that the model does not. To make the comparison fair,
the validation pipeline can inject sensor-realistic noise into the ground truth:

| Sensor | Noise Model | Magnitude |
|---|---|---|
| IMU accelerometer | White noise + bias drift | ±0.02 g RMS + 0.001 g/√Hz drift |
| IMU gyroscope | White noise + bias instability | ±0.01 rad/s RMS + 0.002 rad/s bias |
| Wheel speed (Hall) | Quantisation + 1-sample delay | $2\pi / 48$ rad resolution |
| GPS velocity | White noise | ±0.05 m/s RMS |
| Steering encoder | White noise | ±0.002 rad (14-bit) |

These are calibrated to the Bosch BMI088 IMU, 48-tooth Hall sensors, and u-blox M9N
GPS used on the Ter27 sensor stack.

---

## Validation Without the Physical Car

Before the car is built or available for testing, the methodology can still be
demonstrated using a **sim-to-sim-with-noise** protocol:

1. Run the physics server as the "ground truth" system
2. Inject sensor-realistic noise into the server's output (becoming the "measurement")
3. Run the digital twin with the same inputs (without noise)
4. Compute fidelity metrics

This demonstrates the complete validation pipeline end-to-end. When real car data
arrives, the only change is swapping the data source — the metrics, reports, and
dashboard visualisations are identical.

!!! warning "Credibility Note"
    Sim-to-sim validation will produce $\mathcal{F} > 95\%$ because the model is
    being compared against itself plus noise. This is expected and should be presented
    honestly: "The pipeline is validated end-to-end; the fidelity score reflects
    methodology readiness, not model calibration. Real-data fidelity will be lower
    and will drive the next round of model improvements."

---

## Running the Validation

```bash
# Offline demo (no server, no car data — demonstrates the pipeline)
python scripts/run_twin_fidelity_demo.py --duration 30

# With physics server
python scripts/run_twin_fidelity_demo.py --server --track fsg_autocross --duration 60

# With real car telemetry
python scripts/run_twin_fidelity_demo.py --real-telemetry path/to/motec.csv
```

Output: `reports/twin_fidelity/twin_fidelity_report.json` containing all metrics,
plus a human-readable `.txt` summary.
