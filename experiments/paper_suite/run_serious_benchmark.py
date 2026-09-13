"""Explicit launcher for the serious benchmark; preparation is the default."""
from __future__ import annotations
import argparse
from .common import SuiteConfig
from .serious_benchmark import SeriousBenchmarkConfig, prepare_campaign, run_campaign

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=SuiteConfig().output)
    parser.add_argument("--run", action="store_true", help="Execute the expensive training campaign (opt-in).")
    args = parser.parse_args()
    config = SeriousBenchmarkConfig()
    print(run_campaign(args.output, config) if args.run else prepare_campaign(args.output, config))
