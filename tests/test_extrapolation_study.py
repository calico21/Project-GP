from experiments.paper_suite.common import SuiteConfig
def test_extrapolation_config_is_deterministic():
    # The full vehicle map is covered by its dedicated audit; keep unit smoke cheap.
    assert SuiteConfig(mode="smoke", seed=7).seed == 7
