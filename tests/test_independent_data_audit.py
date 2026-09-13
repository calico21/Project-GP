import csv
import json

import numpy as np

from experiments.paper_suite.independent_data_audit import audit_can, audit_ttc, run_audit


def _can(path, rows):
    fields = ["timestamp", "ANGLE", "APPS_AV", "BPPS", "a_x", "a_y", "a_z", "v_x", "v_y", "v_z", "Yaw_Rate_z", "speed", "rlRPM", "rrRPM", "rlTRQ", "rrTRQ"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            writer.writerow({field: 0.005 * index for field in fields})


def test_independent_data_audit_requires_session_and_run_level_splits(tmp_path):
    can_dir = tmp_path / "can"; can_dir.mkdir()
    for name in ("1", "2", "3", "4", "5"):
        _can(can_dir / f"{name}.csv", 3)
    ttc = tmp_path / "processed.npz"
    values = {name: np.ones(6) for name in ("alpha_rad", "kappa", "Fy_N", "Fx_N", "Fz_N", "Mz_Nm", "gamma_rad", "P_kPa", "Vx_ms", "T_center_C", "T_inner_C", "T_outer_C")}
    np.savez(ttc, **values, run=np.array([4, 4, 4, 5, 5, 5]), is_test=np.array([0, 1, 0, 0, 1, 0]))
    assert audit_can(can_dir)["proposed_session_split"]["test"] == ["5"]
    assert audit_ttc(ttc)["existing_is_test_split"]["eligible_for_held_out_run_claim"] is False
    result = run_audit(tmp_path / "output", can_dir, ttc)
    payload = json.loads(result.read_text())
    assert "No independent 108-state benchmark" in payload["overall_decision"]
