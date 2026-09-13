from .setup_optimization_pilot import run_pilot
from .common import SuiteConfig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output", default=SuiteConfig().output)
    args = parser.parse_args(); run_pilot(SuiteConfig(seed=args.seed, output=args.output))
