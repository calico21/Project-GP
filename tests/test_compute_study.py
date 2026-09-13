from experiments.paper_suite.common import SuiteConfig
def test_compute_config_has_positive_horizon():
    assert SuiteConfig(mode="smoke").horizon() > 0
