import json
import jax
import jax.numpy as jnp
import numpy as np

from experiments.paper_suite.serious_benchmark import (
    SeriousBenchmarkConfig, campaign_ready, config_fingerprint, load_checkpoint,
    prepare_campaign, save_checkpoint, split_metadata, train_with_history,
    write_benchmark_artifacts, model_specifications,
)


class _TinyModel:
    """CPU-cheap common-interface model used only to test the protocol."""
    def init_params(self, key):
        return {"w": jax.random.normal(key, (1, 1)), "b": jnp.zeros((1,))}

    def loss_fn(self, params, x, u, x_next, setup):
        prediction = x @ params["w"] + params["b"]
        return jnp.mean((prediction - x_next) ** 2)


def _data():
    x = jnp.arange(8., dtype=jnp.float32).reshape(-1, 1)
    return {"x": x, "u": jnp.zeros((8, 1)), "x_next": 2 * x, "setup": jnp.zeros((8, 1))}


def test_protocol_requires_three_distinct_seeds_and_independent_splits(tmp_path):
    cfg = SeriousBenchmarkConfig(n_train=4, n_validation=2, n_test_one_step=2, n_test_trajectories=2, epochs=2, batch_size=2)
    root = prepare_campaign(tmp_path, cfg)
    assert campaign_ready(tmp_path, cfg)
    assert config_fingerprint(cfg) == json.loads((root / "protocol.json").read_text())["config_fingerprint"]
    assert set(model_specifications(cfg)) == {"NeuralODE", "PINN", "HNN", "PHNN", "PassiveHNet"}
    splits = split_metadata(cfg, 0)
    assert len({entry["seed"] for entry in splits.values()}) == 4
    ids = [sample for entry in splits.values() for sample in entry["sample_ids"]]
    assert len(ids) == len(set(ids))


def test_checkpoint_roundtrip_and_epoch_resume_are_reproducible(tmp_path):
    train = validation = _data()
    first = train_with_history(_TinyModel(), train, validation, seed=9, epochs=1, learning_rate=.05, batch_size=3)
    path = tmp_path / "checkpoint.msgpack"
    save_checkpoint(path, first, {"epoch": 1, "seed": 9})
    template = {"params": first["params"], "opt_state": first["opt_state"], "rng": first["rng"], "next_epoch": 0, "history": []}
    restored, meta = load_checkpoint(path, template)
    resumed = train_with_history(_TinyModel(), train, validation, seed=9, epochs=3, learning_rate=.05, batch_size=3, resume=restored)
    full = train_with_history(_TinyModel(), train, validation, seed=9, epochs=3, learning_rate=.05, batch_size=3)
    assert meta == {"epoch": 1, "seed": 9}
    assert resumed["history"] == full["history"]
    for a, b in zip(jax.tree_util.tree_leaves(resumed["params"]), jax.tree_util.tree_leaves(full["params"])):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_result_artifacts_are_written_in_machine_and_paper_formats(tmp_path):
    cfg = SeriousBenchmarkConfig(n_train=4, n_validation=2, n_test_one_step=2, n_test_trajectories=2, epochs=2, batch_size=2)
    root = prepare_campaign(tmp_path, cfg)
    records = [{"model": "NeuralODE", "seed": 0, "one_step_rmse": .25,
                "rollout_rmse": {"200": .5}, "per_state_error": {"x0": {"rmse": .1}}}]
    write_benchmark_artifacts(root, records, cfg)
    assert json.loads((root / "results.json").read_text())["records"] == records
    assert "one_step_rmse" in (root / "results.csv").read_text()
    assert "One-step RMSE" in (root / "table_serious_benchmark.tex").read_text()
