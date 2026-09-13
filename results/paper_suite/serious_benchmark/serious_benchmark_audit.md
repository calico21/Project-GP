# Serious-benchmark audit

## Decision

The 15 records are internally complete, but this benchmark is **not suitable as paper evidence of full-vehicle prediction, PassiveHNet superiority, energy behavior, or physical stability**. It is Tier C protocol/evaluation evidence for 28-state mechanical surrogate wrappers only.

## Fairness

All wrappers use the same 28-state target, per-seed deterministic split stream, 200-step rollout controls, and relative-RMSE divergence definition. No normalization was applied; this uniform absence is a scale defect. Raw arrays and target scales were not retained.

## Required next experiment

Build a matched 108-state benchmark with persisted split tensors/scalers and statewise normalization. Every model must receive equivalent forced-dynamics inputs, and any PassiveHNet claim must evaluate the production GLRK map. Record comparable energy quantities before model-comparison conclusions.
