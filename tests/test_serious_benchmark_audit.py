import json
from experiments.paper_suite.audit_serious_benchmark import audit_serious_benchmark

def test_audit_rejects_incomplete_campaign(tmp_path):
    root = tmp_path / "serious_benchmark"; root.mkdir()
    (root / "results.json").write_text(json.dumps({"records": []}))
    (root / "protocol.json").write_text(json.dumps({"config": {"models": ["A"], "seeds": [0]}}))
    (root / "campaign_state.json").write_text(json.dumps({"status": "prepared"}))
    try:
        audit_serious_benchmark(tmp_path)
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("incomplete campaign was accepted")
