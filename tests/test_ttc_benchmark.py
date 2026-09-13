import numpy as np
import pytest

from experiments.paper_suite.ttc_benchmark import CompleteRunSplit, TrainOnlyScaler, prepare_pilot


def _write_ttc(path):
    runs = np.array([4, 4, 5, 5, 6, 6], dtype=np.int8)
    values = {name: np.arange(6, dtype=float) for name in (
        "alpha_rad", "kappa", "gamma_rad", "Fz_N", "Vx_ms", "P_kPa", "T_inner_C", "T_center_C", "T_outer_C", "Fx_N", "Fy_N", "Mz_Nm",
    )}
    np.savez(path, **values, run=runs)


def test_complete_run_split_is_disjoint_and_deterministic():
    split = CompleteRunSplit()
    runs = np.array([4, 4, 5, 5, 6, 6])
    first, second = split.masks(runs), split.masks(runs)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert np.all(sum(mask.astype(int) for mask in first.values()) == 1)
    assert set(runs[first["train"]]) == {4}
    assert set(runs[first["validation"]]) == {5}
    assert set(runs[first["test"]]) == {6}


def test_split_rejects_run_leakage():
    with pytest.raises(ValueError, match="overlap"):
        CompleteRunSplit(train=(4,), validation=(4,), test=(6,)).validate({4, 5, 6})


def test_scaler_is_fit_from_train_rows_only():
    train = np.array([[0.0, 10.0], [2.0, 14.0]])
    held_out = np.array([[1000.0, -1000.0]])
    scaler = TrainOnlyScaler.fit(train)
    assert np.allclose(scaler.center, [1.0, 12.0])
    assert np.allclose(scaler.transform(train).mean(axis=0), [0.0, 0.0])
    assert np.max(np.abs(scaler.transform(held_out))) > 100


def test_pilot_ignores_legacy_row_test_flag_and_has_no_partition_leakage(tmp_path):
    archive = tmp_path / "processed.npz"; _write_ttc(archive)
    manifest = prepare_pilot(archive)
    assert manifest["partition_runs_observed"] == {"train": [4], "validation": [5], "test": [6]}
    assert manifest["stored_is_test_flag"].startswith("ignored")
