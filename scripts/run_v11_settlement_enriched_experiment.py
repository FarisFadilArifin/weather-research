from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.station_stacking import (  # noqa: E402
    TARGET_SOURCE_SETTLEMENT_FIRST,
    StationStackingConfig,
    YearSplitFold,
)
from src.calibration.v11_settlement_enriched_experiment import (  # noqa: E402
    VARIANTS,
    frozen_validation_predictions,
    full_tuned_experiment_from_features,
)
from src.calibration.v11_settlement_enrichment import (  # noqa: E402
    FEATURE_VERSION,
    STATIONS,
    extended_prediction_metrics,
    paired_bootstrap_interval,
    promotion_decision,
)


BASELINE_DIR = REPO_ROOT / "data/calibration/station_stacking_v11_settlement_expanding_4fold"
OUTPUT_DIR = REPO_ROOT / "data/calibration/v11_settlement_enriched_v1"
FOLDS = (
    YearSplitFold("fold_2021_to_2022", 2021, 2021, 2022),
    YearSplitFold("fold_2021_2022_to_2023", 2021, 2022, 2023),
    YearSplitFold("fold_2021_2023_to_2024", 2021, 2023, 2024),
    YearSplitFold("fold_2021_2024_to_2025", 2021, 2024, 2025),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen and fully tune the isolated enriched V11 Settlement experiment")
    parser.add_argument("--stage", choices=["screen", "full", "all"], default="all")
    parser.add_argument("--stations", default=",".join(STATIONS))
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--prepared-dir", type=Path, default=OUTPUT_DIR / "prepared")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "experiment")
    parser.add_argument("--winner", choices=[value for value in VARIANTS if value != "cleaned_v11"])
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--stack-optuna-trials", type=int, default=30)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--retune-original", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = tuple(value.strip().upper() for value in args.stations.split(",") if value.strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, variants = _load_prepared_features(args.prepared_dir, stations)
    winner = args.winner
    if args.stage in {"screen", "all"}:
        winner = run_screen(args, stations, variants)
    if args.stage in {"full", "all"}:
        if not winner:
            winner_file = args.output_dir / "screen_winner.txt"
            if not winner_file.exists():
                raise ValueError("Full stage needs --winner or an existing screen_winner.txt")
            winner = winner_file.read_text(encoding="utf-8").strip()
        run_full(args, stations, variants, frames, winner)


def run_screen(args: argparse.Namespace, stations: tuple[str, ...], variants: dict[str, dict[str, pd.DataFrame]]) -> str:
    all_predictions: list[pd.DataFrame] = []
    for variant, station_frames in variants.items():
        for station in stations:
            selected = pd.read_csv(args.baseline_dir / f"{station}_year_split_selected_hyperparameters.csv")
            config = experiment_config(station, args.output_dir / "screen" / variant, args, trials=1)
            predictions = frozen_validation_predictions(station_frames[station], config, selected)
            predictions["variant"] = variant
            predictions["method"] = variant + "__" + predictions["method"].astype(str)
            all_predictions.append(predictions)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(args.output_dir / "screen_fold_predictions.csv", index=False)
    metrics = extended_prediction_metrics(predictions)
    metrics.to_csv(args.output_dir / "screen_metrics.csv", index=False)
    score = _screen_score(metrics)
    score.to_csv(args.output_dir / "screen_candidate_ranking.csv", index=False)
    winner = str(score.loc[score["variant"].ne("cleaned_v11")].iloc[0]["variant"])
    (args.output_dir / "screen_winner.txt").write_text(winner + "\n", encoding="utf-8")
    return winner


def _load_prepared_features(
    prepared_dir: Path,
    stations: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    manifest = prepared_dir / "prepared_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Prepared enrichment data is missing at {prepared_dir}. "
            "Run scripts/prepare_v11_settlement_enriched_features.py before training."
        )
    originals: dict[str, pd.DataFrame] = {}
    variants: dict[str, dict[str, pd.DataFrame]] = {variant: {} for variant in VARIANTS}
    for station in stations:
        originals[station] = pd.read_csv(prepared_dir / "original_v11" / f"{station}_features.csv")
        for variant in VARIANTS:
            variants[variant][station] = pd.read_csv(prepared_dir / variant / f"{station}_features.csv")
    return originals, variants


def run_full(
    args: argparse.Namespace,
    stations: tuple[str, ...],
    variants: dict[str, dict[str, pd.DataFrame]],
    raw_frames: dict[str, pd.DataFrame],
    winner: str,
) -> None:
    labels = [winner]
    if args.retune_original:
        labels.insert(0, "original_v11")
    pooled_test: list[pd.DataFrame] = []
    pooled_validation: list[pd.DataFrame] = []
    for label in labels:
        source_variant = "cleaned_v11" if label == "original_v11" else label
        for station in stations:
            output = args.output_dir / "full" / label
            config = experiment_config(station, output, args, trials=args.optuna_trials)
            feature_frame = raw_frames[station] if label == "original_v11" else variants[source_variant][station]
            artifacts = full_tuned_experiment_from_features(feature_frame, config, artifact_dir=output)
            validation = artifacts["validation_predictions"].copy()
            test = artifacts["test_predictions"].copy()
            validation["candidate"] = label
            test["candidate"] = label
            validation["method"] = label + "__" + validation["method"].astype(str)
            test["method"] = label + "__" + test["method"].astype(str)
            pooled_validation.append(validation)
            pooled_test.append(test)
    # If original was not retuned, import its untouched predictions only after selection.
    if not args.retune_original:
        for station in stations:
            for kind, target in (("validation", pooled_validation), ("test", pooled_test)):
                source = pd.read_csv(args.baseline_dir / f"{station}_year_split_{kind}_predictions.csv")
                source["station_id"] = station
                source["candidate"] = "original_v11"
                source["method"] = "original_v11__" + source["method"].astype(str)
                target.append(source)
    validation = pd.concat(pooled_validation, ignore_index=True)
    test = pd.concat(pooled_test, ignore_index=True)
    validation.to_csv(args.output_dir / "full_fold_predictions.csv", index=False)
    test.to_csv(args.output_dir / "final_2026_predictions.csv", index=False)
    validation_metrics = extended_prediction_metrics(validation)
    test_metrics = extended_prediction_metrics(test)
    validation_metrics.to_csv(args.output_dir / "full_validation_metrics.csv", index=False)
    test_metrics.to_csv(args.output_dir / "final_2026_metrics.csv", index=False)
    baseline_method = "original_v11__ridge_stack"
    candidate_method = f"{winner}__ridge_stack"
    _bootstrap_report(validation, baseline_method, candidate_method).to_csv(args.output_dir / "validation_paired_bootstrap.csv", index=False)
    _bootstrap_report(test, baseline_method, candidate_method).to_csv(args.output_dir / "final_2026_paired_bootstrap.csv", index=False)
    decision = promotion_decision(test_metrics, baseline=baseline_method, candidate=candidate_method)
    decision.to_csv(args.output_dir / "promotion_decision.csv", index=False)
    write_markdown_report(args.output_dir / "original_vs_enriched_report.md", winner, validation_metrics, test_metrics, decision)


def experiment_config(station: str, output_dir: Path, args: argparse.Namespace, *, trials: int) -> StationStackingConfig:
    return StationStackingConfig(
        station_id=station,
        project_root=REPO_ROOT,
        timing_mode="same_day_11am_live_safe",
        providers=("gfs", "hrrr", "nbm"),
        fast_mode=args.fast_mode,
        optuna_trials=trials,
        stack_optuna_trials=args.stack_optuna_trials,
        optuna_startup_trials=min(15, trials),
        stack_optuna_startup_trials=min(15, args.stack_optuna_trials),
        optuna_metric="mae_f",
        feature_version="v11",
        target_mode="remaining_warmup",
        target_source=TARGET_SOURCE_SETTLEMENT_FIRST,
        hyperparameter_space="wide",
        catboost_max_iterations=1200,
        catboost_max_depth=8,
        catboost_min_learning_rate=0.005,
        catboost_max_border_count=128,
        base_model_methods=("xgboost", "lightgbm", "catboost"),
        stack_enabled=True,
        year_split_folds=FOLDS,
        year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0},
        year_split_test_train_years=(2021, 2025),
        year_split_test_year=2026,
        output_dir=output_dir,
        climatology_normals_path=REPO_ROOT / "data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv",
    )


def _screen_score(metrics: pd.DataFrame) -> pd.DataFrame:
    pooled = metrics.loc[metrics["scope"].eq("pooled") & metrics["method"].str.contains("__(?:xgboost|lightgbm|catboost)$", regex=True)].copy()
    pooled["variant"] = pooled["method"].str.split("__").str[0]
    score = pooled.groupby("variant", as_index=False).agg(
        mean_mae_f=("mae_f", "mean"),
        mean_rmse_f=("rmse_f", "mean"),
        mean_bucket_hit_rate=("bucket_hit_rate", "mean"),
        worst_p95_abs_error_f=("p95_abs_error_f", "max"),
    )
    score["rank_score"] = score["mean_mae_f"].rank() + score["mean_rmse_f"].rank() + score["mean_bucket_hit_rate"].rank(ascending=False)
    return score.sort_values(["rank_score", "mean_mae_f", "mean_rmse_f"]).reset_index(drop=True)


def _bootstrap_report(predictions: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "ALL", predictions)]
    scopes.extend(("station", str(station), group) for station, group in predictions.groupby("station_id"))
    if "fold" in predictions:
        scopes.extend(("fold", str(fold), group) for fold, group in predictions.groupby("fold"))
    rows = []
    for scope, label, group in scopes:
        for metric in ("mae", "rmse", "bias", "bucket_hit_rate"):
            rows.append(
                {
                    "scope": scope,
                    "scope_value": label,
                    "metric": metric,
                    **paired_bootstrap_interval(group, baseline_method=baseline, candidate_method=candidate, metric=metric),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(path: Path, winner: str, validation: pd.DataFrame, test: pd.DataFrame, decision: pd.DataFrame) -> None:
    lines = [
        f"# {FEATURE_VERSION}: Original V11 vs {winner}",
        "",
        "The enriched candidate was selected from 2022–2025 expanding-fold validation only. The 2026 rows were opened once after selection.",
        "",
        "## Validation metrics",
        "",
        validation.to_markdown(index=False),
        "",
        "## Final 2026 metrics",
        "",
        test.to_markdown(index=False),
        "",
        "## Promotion checks",
        "",
        decision.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
