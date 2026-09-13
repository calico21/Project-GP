from __future__ import annotations
import argparse
from .common import SuiteConfig, manifest
from . import studies
from .reporting import figures, tables
def run(fn):
 p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("smoke","standard","paper"),default="smoke"); p.add_argument("--seed",type=int,default=0); p.add_argument("--output"); a=p.parse_args(); cfg=SuiteConfig(a.mode,a.seed,a.output or SuiteConfig().output); fn(cfg); manifest(cfg)
