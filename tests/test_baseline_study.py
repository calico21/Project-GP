from experiments.paper_suite.baseline_real import manoeuvres, OBS

def test_baseline_protocol_has_required_manoeuvres_and_observables():
    assert set(manoeuvres(3)) == {"straight", "steering_transient", "sustained_lateral", "brake_steer"}
    assert {"vx", "vy", "yaw_rate", "roll", "pitch", "front_heave", "rear_heave"} <= set(OBS)
