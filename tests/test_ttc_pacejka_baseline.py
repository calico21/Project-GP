import numpy as np

from experiments.paper_suite.ttc_benchmark import CompleteRunSplit, TrainOnlyScaler
from experiments.paper_suite.ttc_pacejka_baseline import choose_force_sign


def test_sign_mapping_is_deterministic_and_train_only():
    measured_train = np.array([1.0, 2.0, 3.0])
    raw_model_train = -measured_train
    first = choose_force_sign(measured_train, raw_model_train)
    second = choose_force_sign(measured_train, raw_model_train)
    assert first == second
    assert first["model_to_ttc_fy_multiplier"] == -1
    assert first["selection_partition"] == "train (Run 4) only"


def test_locked_test_run_cannot_participate_in_fit_mask():
    runs = np.array([4, 4, 5, 5, 6, 6])
    masks = CompleteRunSplit().masks(runs)
    fitted_rows = masks["train"]
    assert not np.any(fitted_rows & masks["validation"])
    assert not np.any(fitted_rows & masks["test"])
    assert set(runs[fitted_rows]) == {4}


def test_fitted_scaler_cannot_be_modified_by_evaluation_transform():
    scaler = TrainOnlyScaler.fit(np.array([[1.0], [3.0]]))
    before = (scaler.center.copy(), scaler.scale.copy())
    scaler.transform(np.array([[100.0], [200.0]]))
    assert np.array_equal(scaler.center, before[0])
    assert np.array_equal(scaler.scale, before[1])
