from __future__ import annotations

import argparse
import logging

from .calibration.modeling import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate walk-forward forecast calibration models")
    parser.add_argument("--calibration-dir", default="data/calibration")
    parser.add_argument("--min-train-days", type=int, default=90)
    parser.add_argument("--shrinkage-k", type=float, default=30.0)
    parser.add_argument("--ml-refit-days", type=int, default=30)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    baseline, ml, rules = train_and_evaluate(
        calibration_dir=args.calibration_dir,
        min_train_days=args.min_train_days,
        shrinkage_k=args.shrinkage_k,
        ml_refit_days=args.ml_refit_days,
        run_ml=not args.skip_ml,
    )
    logging.info("Wrote baseline results: %s rows", len(baseline))
    logging.info("Wrote ML results: %s rows", len(ml))
    logging.info("Wrote recommended rules: %s rows", len(rules))


if __name__ == "__main__":
    main()
