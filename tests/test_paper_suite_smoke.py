import json
import jax
jax.config.update("jax_enable_x64", True)
from experiments.paper_suite.common import SuiteConfig
from experiments.paper_suite.solver import solve_stages_picard, solve_stages_newton
from experiments.full_vehicle_gradient_audit import make_case, AuditCase

def test_solver_dimensions_and_finite():
    v,x,u,s,_=make_case(AuditCase(n_setup_params=1)); p=solve_stages_picard(v,x,u,s,.005,iterations=1); n=solve_stages_newton(v,x,u,s,.005,iterations=1)
    assert p["stages"].shape == (216,) and n["stages"].shape == (216,)
    assert p["final_residual_inf"] >= 0 and n["final_residual_inf"] >= 0

def test_result_schema_writer(tmp_path):
    # Full vehicle studies are deliberately exercised from the CLI smoke run;
    # keeping this unit test short avoids recompiling the 216-state map twice.
    from experiments.paper_suite.common import write_result
    cfg=SuiteConfig(output=str(tmp_path)); write_result(cfg, "solver", {"stage_dimension":216}, rows=[{"finite":True}])
    report=json.loads((tmp_path/"solver"/"solver.json").read_text())
    assert report["metrics"]["stage_dimension"] == 216 and report["status"] == "completed"
