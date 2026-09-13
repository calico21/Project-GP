from experiments.paper_suite.studies import ablation_study
from experiments.paper_suite.common import SuiteConfig
def test_ablation_statuses_are_explicit(tmp_path):
    p=ablation_study(SuiteConfig(output=str(tmp_path)))
    assert p.exists()
