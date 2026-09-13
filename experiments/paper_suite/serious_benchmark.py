"""Resumable protocol for the post-pilot learned-model benchmark.

This module deliberately does not run at import time or from the paper-suite
``run_all`` command.  The old baseline pilot remains an immutable historical
artifact; this is the protocol used for the next, independently testable run.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from benchmarks.baselines import BASELINE_REGISTRY
from benchmarks.metrics.stability import detect_numerical_failures, estimate_divergence_time
from benchmarks.metrics.trajectory import per_state_errors, rmse
from .common import SuiteConfig, jsonable, manifest, metadata

MODEL_NAMES = tuple(BASELINE_REGISTRY)
HORIZONS = (1, 10, 20, 50, 100, 200)


@dataclass(frozen=True)
class SeriousBenchmarkConfig:
    """Fixed protocol parameters.  ``paper`` is intentionally conservative."""
    seeds: tuple[int, ...] = (0, 1, 2)
    n_train: int = 1000
    n_validation: int = 200
    n_test_one_step: int = 200
    n_test_trajectories: int = 32
    horizon_steps: int = 200
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    dt: float = .005
    models: tuple[str, ...] = MODEL_NAMES

    def validate(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("The serious benchmark requires at least three distinct seeds.")
        if set(self.models) != set(MODEL_NAMES):
            raise ValueError(f"Models must be exactly {MODEL_NAMES} for a matched benchmark.")
        if self.horizon_steps < max(HORIZONS):
            raise ValueError("horizon_steps must be at least 200.")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(jsonable(payload), indent=2, sort_keys=True))


def config_fingerprint(config: SeriousBenchmarkConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def model_specifications(config: SeriousBenchmarkConfig) -> dict[str, dict[str, Any]]:
    """Persist constructor defaults, preventing architecture/config ambiguity."""
    output = {}
    for name in config.models:
        cls = BASELINE_REGISTRY[name]
        signature = inspect.signature(cls.__init__)
        defaults = {key: value.default for key, value in signature.parameters.items()
                    if key != "self" and value.default is not inspect.Parameter.empty}
        # dt is protocol-controlled, even if a class default differs later.
        defaults["dt"] = config.dt
        output[name] = {"class": f"{cls.__module__}.{cls.__qualname__}", "constructor": defaults}
    return output


def split_seed(seed: int, split: str) -> int:
    """Stable split derivation; split names never share a PRNG stream."""
    digest = hashlib.sha256(f"Project-GP/serious-benchmark/v1/{seed}/{split}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def split_metadata(config: SeriousBenchmarkConfig, seed: int) -> dict[str, dict[str, Any]]:
    counts = {"train": config.n_train, "validation": config.n_validation,
              "test_one_step": config.n_test_one_step, "test_rollout": config.n_test_trajectories}
    return {name: {"seed": split_seed(seed, name), "count": count,
                   "sample_ids": [f"{name}:{seed}:{i}" for i in range(count)]}
            for name, count in counts.items()}


def prepare_campaign(output: str | Path, config: SeriousBenchmarkConfig) -> Path:
    """Create/reopen a campaign ledger without generating data or training."""
    config.validate()
    root = Path(output) / "serious_benchmark"
    state_path = root / "campaign_state.json"
    fingerprint = config_fingerprint(config)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config_fingerprint"] != fingerprint:
            raise ValueError("Existing campaign has a different configuration; use a new output directory.")
        protocol_path = root / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        if "model_specifications" not in protocol:
            protocol["model_specifications"] = model_specifications(config)
            _json(protocol_path, protocol)
            manifest(SuiteConfig(output=str(root.parent)))
        return root
    units = {f"{model}/seed_{seed}": {"status": "pending"}
             for seed in config.seeds for model in config.models}
    _json(root / "protocol.json", {"protocol_version": 1, "config": asdict(config), "model_specifications": model_specifications(config),
                                    "config_fingerprint": fingerprint,
                                    "required_metrics": ["one_step_rmse", "rollout_rmse", "per_state_error",
                                                         "stability", "energy_diagnostics", "parameter_count",
                                                         "training_time_s", "inference_time_s"]})
    _json(state_path, {"config_fingerprint": fingerprint, "status": "prepared", "units": units,
                       "resume_rule": "completed units are never retrained; incomplete units resume from epoch checkpoints."})
    _json(root / "splits.json", {str(seed): split_metadata(config, seed) for seed in config.seeds})
    manifest(SuiteConfig(output=str(root.parent)))
    return root


def _tree_size(tree: Any) -> int:
    return sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree))


def save_checkpoint(path: Path, state: dict[str, Any], meta: dict[str, Any]) -> None:
    """Atomically persist parameter *and optimizer* state for exact resumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Flax list restoration requires a same-length template.  Histories have a
    # variable length, so keep them as explicit JSON provenance rather than in
    # the binary pytree; params/optimizer/RNG retain exact array fidelity.
    serializable_state = {key: value for key, value in state.items() if key != "history"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(serializable_state))
    os.replace(temporary, path)
    _json(path.with_suffix(".json"), {"metadata": meta, "history": state.get("history", [])})


def load_checkpoint(path: Path, template: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists() or not path.with_suffix(".json").exists():
        raise FileNotFoundError(f"Incomplete checkpoint: {path}")
    sidecar = json.loads(path.with_suffix(".json").read_text())
    fixed_template = {key: value for key, value in template.items() if key != "history"}
    state = serialization.from_bytes(fixed_template, path.read_bytes())
    state["history"] = sidecar.get("history", [])
    return state, sidecar.get("metadata", {})


def train_with_history(model, train: dict[str, jax.Array], validation: dict[str, jax.Array], *,
                       seed: int, epochs: int, learning_rate: float, batch_size: int,
                       resume: dict[str, Any] | None = None, on_epoch=None) -> dict[str, Any]:
    """Matched Adam training with epoch-level train/validation histories.

    The returned state contains the optimizer and PRNG state, so interruption
    within an individual model/seed unit is resumable rather than merely
    restartable.
    """
    optimizer = optax.adam(learning_rate)
    if resume is None:
        init_key, permutation_key = jax.random.split(jax.random.PRNGKey(seed))
        params = model.init_params(init_key)
        state = {"params": params, "opt_state": optimizer.init(params), "rng": permutation_key,
                 "next_epoch": 0, "history": []}
    else:
        state = resume
    n = int(train["x"].shape[0])
    if n == 0:
        raise ValueError("Training split is empty.")

    @jax.jit
    def step(params, opt_state, x, u, x_next, setup):
        loss, grads = jax.value_and_grad(model.loss_fn)(params, x, u, x_next, setup)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    for epoch in range(int(state["next_epoch"]), epochs):
        rng, key = jax.random.split(state["rng"])
        permutation = jax.random.permutation(key, n)
        params, opt_state = state["params"], state["opt_state"]
        batch_losses = []
        for start in range(0, n, batch_size):
            ids = permutation[start:min(start + batch_size, n)]
            params, opt_state, loss = step(params, opt_state, train["x"][ids], train["u"][ids], train["x_next"][ids], train["setup"][ids])
            batch_losses.append(float(loss))
        train_loss = float(model.loss_fn(params, train["x"], train["u"], train["x_next"], train["setup"]))
        validation_loss = float(model.loss_fn(params, validation["x"], validation["u"], validation["x_next"], validation["setup"]))
        state = {"params": params, "opt_state": opt_state, "rng": rng, "next_epoch": epoch + 1,
                 "history": [*state["history"], {"epoch": epoch + 1, "train_loss": train_loss,
                                                    "validation_loss": validation_loss,
                                                    "mean_batch_loss": float(np.mean(batch_losses))}]}
        if on_epoch:
            on_epoch(state)
    model.params = state["params"]
    return state


def _production_one_step(vehicle, n: int, seed: int, dt: float) -> dict[str, jax.Array]:
    """Independent labelled samples; each sample uses a fold-in-derived key."""
    from models.vehicle_dynamics import DEFAULT_SETUP
    from .baseline_real import _to_baseline
    rows = []
    base = jax.random.PRNGKey(seed)
    for i in range(n):
        k1, k2, k3 = jax.random.split(jax.random.fold_in(base, i), 3)
        x = vehicle.make_initial_state(vx0=float(jax.random.uniform(k1, (), minval=10., maxval=30.)))
        x = x.at[15].set(jax.random.uniform(k2, (), minval=-1., maxval=1.))
        u = jnp.array([jax.random.uniform(k3, (), minval=-.05, maxval=.05), 8., 8., 8., 8., 0.], dtype=x.dtype)
        x_next = vehicle.simulate_step(x, u, DEFAULT_SETUP, dt=dt, n_substeps=1)
        rows.append((_to_baseline(vehicle, x), u, _to_baseline(vehicle, x_next), DEFAULT_SETUP))
    return {key: jnp.stack([row[i] for row in rows]) for i, key in enumerate(("x", "u", "x_next", "setup"))}


def build_production_splits(config: SeriousBenchmarkConfig, seed: int) -> dict[str, dict[str, jax.Array]]:
    """Construct disjoint train/validation/test one-step data for one seed."""
    from config.tire_coeffs import tire_coeffs as TC
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    vehicle = DifferentiableMultiBodyVehicle(VP, TC)
    return {name: _production_one_step(vehicle, meta["count"], meta["seed"], config.dt)
            for name, meta in split_metadata(config, seed).items() if name != "test_rollout"}


def build_production_rollout_test(config: SeriousBenchmarkConfig, seed: int) -> dict[str, jax.Array]:
    """Independent randomized 200-step test trajectories, never reused for fitting."""
    from config.tire_coeffs import tire_coeffs as TC
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from models.vehicle_dynamics import DEFAULT_SETUP, DifferentiableMultiBodyVehicle
    from .baseline_real import _to_baseline
    vehicle = DifferentiableMultiBodyVehicle(VP, TC)
    base = jax.random.PRNGKey(split_seed(seed, "test_rollout"))
    initials, controls_all, targets = [], [], []
    for i in range(config.n_test_trajectories):
        ki, kc = jax.random.split(jax.random.fold_in(base, i))
        x = vehicle.make_initial_state(vx0=float(jax.random.uniform(ki, (), minval=10., maxval=30.)))
        x = x.at[15].set(jax.random.uniform(ki, (), minval=-1., maxval=1.))
        initial = x
        noise = jax.random.normal(kc, (config.horizon_steps, 2), dtype=x.dtype)
        controls = jnp.column_stack((jnp.clip(.04 * noise[:, 0], -.08, .08),
                                     jnp.clip(8. + 2. * noise[:, 1], 0., 15.),
                                     jnp.full((config.horizon_steps,), 8., dtype=x.dtype),
                                     jnp.full((config.horizon_steps,), 8., dtype=x.dtype),
                                     jnp.full((config.horizon_steps,), 8., dtype=x.dtype),
                                     jnp.zeros((config.horizon_steps,), dtype=x.dtype)))
        ref = []
        for control in controls:
            x = vehicle.simulate_step(x, control, DEFAULT_SETUP, dt=config.dt, n_substeps=1)
            ref.append(_to_baseline(vehicle, x))
        initials.append(_to_baseline(vehicle, initial))
        controls_all.append(controls); targets.append(jnp.stack(ref))
    return {"x0": jnp.stack(initials), "controls": jnp.stack(controls_all),
            "x_true": jnp.stack(targets), "setup": jnp.broadcast_to(DEFAULT_SETUP, (config.n_test_trajectories, DEFAULT_SETUP.size))}


def _energy_diagnostics(model_name: str) -> dict[str, str]:
    # These learned baselines do not share a calibrated physical-energy scale.
    # Recording an unavailable field is more honest than comparing arbitrary H.
    return {"status": "unavailable", "reason": f"{model_name} has no shared calibrated physical-energy observable in this benchmark protocol."}


def evaluate_model(model, params: dict[str, Any], one_step: dict[str, jax.Array], rollout: dict[str, jax.Array],
                   config: SeriousBenchmarkConfig) -> dict[str, Any]:
    start = time.perf_counter()
    predicted_one = jax.vmap(lambda x, u, s: model.predict_step(params, x, u, s))(one_step["x"], one_step["u"], one_step["setup"])
    predicted_one.block_until_ready()
    inference_time = time.perf_counter() - start
    predicted_rollout = jax.vmap(lambda x, controls, setup: model.predict_trajectory(params, x, controls, setup))(rollout["x0"], rollout["controls"], rollout["setup"])
    predicted_rollout.block_until_ready()
    true = rollout["x_true"]
    horizons = {str(h): float(rmse(true[:, :h], predicted_rollout[:, :h])) for h in HORIZONS}
    states = per_state_errors(true.reshape((-1, true.shape[-1])), predicted_rollout.reshape((-1, predicted_rollout.shape[-1])))
    stability = [estimate_divergence_time(predicted_rollout[i], true[i], config.dt) for i in range(true.shape[0])]
    failures = detect_numerical_failures(predicted_rollout.reshape((-1, predicted_rollout.shape[-1])))
    return {"one_step_rmse": float(rmse(one_step["x_next"], predicted_one)), "rollout_rmse": horizons,
            "per_state_error": states, "stability": stability, "numerical_failures": failures,
            "energy_diagnostics": _energy_diagnostics(model.name),
            "inference_time_s": inference_time, "inference_samples": int(one_step["x"].shape[0])}


def _update_unit(root: Path, unit: str, status: str) -> None:
    path = root / "campaign_state.json"; state = json.loads(path.read_text())
    state["units"][unit]["status"] = status
    statuses = [entry["status"] for entry in state["units"].values()]
    state["status"] = "completed" if all(item == "completed" for item in statuses) else "running"
    _json(path, state)


def run_campaign(output: str | Path, config: SeriousBenchmarkConfig) -> Path:
    """Explicitly execute/restart the benchmark.  Never called by preparation."""
    root = prepare_campaign(output, config)
    state = json.loads((root / "campaign_state.json").read_text())
    records_path = root / "results.json"
    records = json.loads(records_path.read_text())["records"] if records_path.exists() else []
    completed = {(r["model"], r["seed"]) for r in records}
    for seed in config.seeds:
        splits = build_production_splits(config, seed)
        rollout = build_production_rollout_test(config, seed)
        for model_name in config.models:
            if (model_name, seed) in completed:
                continue
            unit = f"{model_name}/seed_{seed}"; _update_unit(root, unit, "running")
            model = BASELINE_REGISTRY[model_name](dt=config.dt)
            checkpoint = root / "checkpoints" / model_name / f"seed_{seed}" / "training.msgpack"
            # Initialise only to construct an exact Flax restore template.
            initial = model.init_params(jax.random.PRNGKey(seed))
            optimizer = optax.adam(config.learning_rate)
            template = {"params": initial, "opt_state": optimizer.init(initial), "rng": jax.random.PRNGKey(seed), "next_epoch": 0}
            resume = load_checkpoint(checkpoint, template)[0] if checkpoint.exists() else None
            started = time.perf_counter()
            def persist(training_state):
                save_checkpoint(checkpoint, training_state, {"model": model_name, "seed": seed,
                                                              "config_fingerprint": config_fingerprint(config),
                                                              "kind": "epoch"})
            trained = train_with_history(model, splits["train"], splits["validation"], seed=seed,
                                         epochs=config.epochs, learning_rate=config.learning_rate,
                                         batch_size=config.batch_size, resume=resume, on_epoch=persist)
            training_time = time.perf_counter() - started
            final_path = checkpoint.with_name("final.msgpack")
            save_checkpoint(final_path, trained, {"model": model_name, "seed": seed,
                                                   "config_fingerprint": config_fingerprint(config), "kind": "final"})
            result = {"model": model_name, "seed": seed, "parameter_count": _tree_size(trained["params"]),
                      "training_time_s": training_time, "history": trained["history"],
                      **evaluate_model(model, trained["params"], splits["test_one_step"], rollout, config)}
            records.append(result); _json(records_path, {"records": records, "config": asdict(config)})
            _update_unit(root, unit, "completed")
    write_benchmark_artifacts(root, records, config)
    return root


def write_benchmark_artifacts(root: Path, records: list[dict[str, Any]], config: SeriousBenchmarkConfig) -> None:
    """Write JSON, flat CSV, LaTeX summary, figures, and refresh the manifest."""
    _json(root / "results.json", {"records": records, "config": asdict(config)})
    flat = [{k: v for k, v in record.items() if not isinstance(v, (dict, list))} for record in records]
    if flat:
        fields = sorted({key for row in flat for key in row})
        with (root / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(flat)
        lines = ["\\begin{tabular}{lrr}", "\\toprule", "Model & Seed & One-step RMSE \\\\ \\midrule"]
        lines += [f"{r['model']} & {r['seed']} & {r.get('one_step_rmse', float('nan')):.6g} \\\\" for r in records]
        lines += ["\\bottomrule", "\\end{tabular}"]
        (root / "table_serious_benchmark.tex").write_text("\n".join(lines))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(6, 3.5))
            axis.bar([f"{r['model']}\nseed {r['seed']}" for r in records], [r.get("one_step_rmse", np.nan) for r in records])
            axis.set(ylabel="independent-test one-step RMSE", title="Serious benchmark (per seed)")
            axis.tick_params(axis="x", rotation=30); figure.tight_layout()
            figure.savefig(root / "fig_serious_benchmark_one_step_rmse.png", dpi=300)
            figure.savefig(root / "fig_serious_benchmark_one_step_rmse.pdf", dpi=300)
            plt.close(figure)
        except ImportError:
            pass
    manifest(SuiteConfig(output=str(root.parent)))


def campaign_ready(output: str | Path, config: SeriousBenchmarkConfig) -> bool:
    root = Path(output) / "serious_benchmark"
    return (root / "protocol.json").exists() and (root / "campaign_state.json").exists() and (root / "splits.json").exists() and config.validate() is None
