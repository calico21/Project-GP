import numpy as np

from experiments.paper_suite.ttc_residual_pinn import FEATURES, TTCAdditiveResidualPINN, _load_roles, coverage_partition, train_coverage_thresholds


def _archive(path):
    runs = np.array([4, 4, 5, 5, 6, 6], dtype=np.int8)
    fields = {name: np.arange(6, dtype=float) for name in ("alpha_rad", "kappa", "gamma_rad", "Fz_N", "Vx_ms", "P_kPa", "T_inner_C", "T_center_C", "T_outer_C", "Fy_N")}
    np.savez(path, **fields, run=runs, is_test=np.ones(6, dtype=np.int8))


def test_train_validation_loader_excludes_test_run_and_ignores_legacy_flag(tmp_path):
    archive = tmp_path / "processed.npz"; _archive(archive)
    roles, source = _load_roles(archive, ("train", "validation"))
    assert source["requested_runs"].tolist() == [4, 5]
    assert {role: len(values["Fy_N"]) for role, values in roles.items()} == {"train": 2, "validation": 2}
    assert all("is_test" not in values for values in roles.values())
    assert all(name in roles["train"] for name in FEATURES)


def test_residual_network_initialization_is_seed_deterministic():
    import jax
    import jax.numpy as jnp
    model = TTCAdditiveResidualPINN((16, 16))
    first = model.init(jax.random.PRNGKey(0), jnp.zeros(len(FEATURES)))
    second = model.init(jax.random.PRNGKey(0), jnp.zeros(len(FEATURES)))
    leaves_a = jax.tree_util.tree_leaves(first); leaves_b = jax.tree_util.tree_leaves(second)
    assert all(np.array_equal(a, b) for a, b in zip(leaves_a, leaves_b))


def test_coverage_thresholds_are_train_derived_and_freeze_speed_partition():
    train = {"alpha_rad": np.array([0., 1.]), "Fz_N": np.array([100., 200.]), "gamma_rad": np.array([0., .1]), "Vx_ms": np.array([10., 12.]), "P_kPa": np.array([70., 80.])}
    validation = {"alpha_rad": np.array([.5]), "Fz_N": np.array([150.]), "gamma_rad": np.array([.05]), "Vx_ms": np.array([14.]), "P_kPa": np.array([90.])}
    thresholds = train_coverage_thresholds(train)
    result = coverage_partition(validation, thresholds)
    assert thresholds["Vx_ms"] == {"min": 10.0, "max": 12.0}
    assert result["speed_interpolation_count"] == 0 and result["speed_extrapolation_count"] == 1
