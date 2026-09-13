from __future__ import annotations
import argparse, time
from .common import SuiteConfig, manifest
from .studies import solver_study, gradient_study, baseline_study, energy_study, ablation_study, setup_optimization, extrapolation_study, compute_study
from .reporting import figures, tables
def main():
 p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("smoke","standard","paper"),default="smoke"); p.add_argument("--seed",type=int,default=0); p.add_argument("--output"); a=p.parse_args(); cfg=SuiteConfig(a.mode,a.seed,a.output or SuiteConfig().output)
 for name,fn in (("solver",solver_study),("baseline",baseline_study),("energy",energy_study),("gradients",gradient_study),("ablations",ablation_study),("setup optimization",setup_optimization),("extrapolation",extrapolation_study),("compute",compute_study),("figures",figures),("tables",tables)):
  print(f"[paper-suite] {name}",flush=True); t=time.perf_counter(); fn(cfg); print(f"[paper-suite] {name} completed in {time.perf_counter()-t:.2f}s",flush=True)
 manifest(cfg)
if __name__=="__main__": main()
