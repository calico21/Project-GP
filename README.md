# Formula Student Driver & Setup Optimizer (Digital Twin)

**A Python-based engineering suite for vehicle dynamics simulation, driver analysis, and genetic setup optimization.**

![Status](https://img.shields.io/badge/Status-Competition%20Ready-brightgreen)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/Libraries-CasADi%20%7C%20NumPy%20%7C%20Pandas-orange)

## 🏎️ Project Overview

This software creates a **Digital Twin** of a Formula Student race car to solve two critical engineering problems:

1.  **Driver Analysis (The "Ghost Car"):**
    * Uses **Optimal Control (OCP)** to calculate the theoretical minimum lap time for a specific track.
    * Compares the "Perfect Lap" against real telemetry data to identify driver error versus vehicle limits.
    * Reverse-engineers track geometry directly from GPS logs.

2.  **Setup Optimization (The "Genetic Engineer"):**
    * Uses a **Genetic Algorithm (NSGA-II)** to find the optimal suspension stiffness and aero balance.
    * Visualizes the **Pareto Front** between **Ultimate Grip** (Qualifying) and **Drivability/Stability** (Endurance).

---

## 📂 Project Structure

```text
FS_Driver_Setup_Optimizer/
│
├── data/
│   └── logs/               # Place Vector .asc or SavvyCAN logs here
│
├── models/
│   ├── vehicle_dynamics.py # 7-DOF Bicycle Model equations (CasADi)
│   ├── tire_model.py       # Pacejka 5.2 Magic Formula implementation
│   └── track_model.py      # Curvature and path generation logic
│
├── optimization/
│   ├── ocp_solver.py       # The "Virtual Driver" (Time-Optimal Path Solver)
│   └── evolutionary.py     # Genetic Algorithm for finding spring/aero setups
│
├── telemetry/
│   ├── log_ingestion.py    # DBC decoder for raw CAN data
│   ├── track_generator.py  # Reverse-engineers track geometry from GPS logs
│   └── validation.py       # Statistical comparison (Sim vs. Real)
│
├── main.py                 # Primary executable for Lap Analysis
├── visualize_log.py        # Helper to debug CAN IDs
├── TER.dbc                 # CAN Database file
└── README.md               # Documentation