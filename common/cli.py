import argparse

import yaml

from .config import DEFAULT_CONFIG
from fitting import FITTING_METHODS
from formulas import FORMULAS
from .runner import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit scaling laws and save JSON results.")
    parser.add_argument("--dataset", default="SFT-15", help="Dataset name in the configuration (default: SFT-15)")
    parser.add_argument("--formula", "--method", dest="formula", choices=tuple(FORMULAS),
                        default="ULD-IG", help="Scaling-law formula (default: ULD-IG)")
    parser.add_argument("--fit-method", "--fit-mode", dest="fit_method",
                        choices=sorted({name for methods in FITTING_METHODS.values() for name in methods}),
                        default="default", help="Formula-specific fitting method (default: unanchored)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a YAML configuration")
    parser.add_argument("--budget", type=int, nargs="+", dest="budgets", help="One or more configured budgets (default: all)")
    parser.add_argument("--anchor-weight", type=float, help="Use a fixed soft-anchor weight instead of the configured schedule")
    parser.add_argument("--output-dir", help="Results directory (default: results/ in the repository)")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true", help="Skip completed curves and retry failures")
    existing.add_argument("--overwrite", action="store_true", help="Refit and replace the selected result files")
    args = parser.parse_args(argv)
    try:
        run(dataset=args.dataset, formula=args.formula, fit_method=args.fit_method,
            config_path=args.config, budgets=args.budgets, anchor_weight=args.anchor_weight,
            output_dir=args.output_dir, resume=args.resume, overwrite=args.overwrite)
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f"Error: {exc}\n")
