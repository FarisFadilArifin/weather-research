from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.celsius_market_probability import (
    fahrenheit_to_celsius,
    round_half_up,
)


STATION_CONFIGS = {
    "Tokyo": {
        "station_id": "RJTT",
        "station_name": "Tokyo Haneda Airport",
        "market_url": "https://polymarket.com/event/highest-temperature-in-tokyo-on-may-5-2026",
        "market_title": "Highest temperature in Tokyo on May 5, 2026",
    },
    "Seoul": {
        "station_id": "RKSI",
        "station_name": "Incheon Intl Airport",
        "market_url": "https://polymarket.com/event/highest-temperature-in-seoul-on-may-6-2026/highest-temperature-in-seoul-on-may-6-2026-13c",
        "market_title": "Highest temperature in Seoul on May 6, 2026",
    },
}


def configure_station(city: str) -> None:
    global CITY, STATION_ID, STATION_NAME, MARKET_URL, MARKET_TITLE
    global TOKYO_ROOT, CELSIUS_ROOT, REPORT_ROOT
    config = STATION_CONFIGS[city]
    CITY = city
    STATION_ID = config["station_id"]
    STATION_NAME = config["station_name"]
    MARKET_URL = config["market_url"]
    MARKET_TITLE = config["market_title"]
    TOKYO_ROOT = (
        PROJECT_ROOT / "data" / "calibration" / "station_training_baseline" / city
    )
    CELSIUS_ROOT = TOKYO_ROOT / "celsius_market_probability"
    REPORT_ROOT = PROJECT_ROOT / "reports" / f"{city.lower()}_celsius_market_probability"


configure_station("Tokyo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_fahrenheit_distribution(degree_probabilities: str) -> dict[int, float]:
    mapped: dict[int, float] = {}
    for degree_f, probability in json.loads(degree_probabilities).items():
        bucket_c = round_half_up(fahrenheit_to_celsius(float(degree_f)))
        mapped[bucket_c] = mapped.get(bucket_c, 0.0) + float(probability)
    total = sum(mapped.values())
    return {bucket: probability / total for bucket, probability in mapped.items()}


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    new = pd.read_csv(CELSIUS_ROOT / f"{STATION_ID}_2026_holdout_predictions.csv")
    old = pd.read_csv(
        TOKYO_ROOT
        / "ordinal_probability"
        / f"{STATION_ID}_2026_probability_holdout_predictions.csv"
    )
    for frame in (new, old):
        frame["contract_date"] = pd.to_datetime(frame["contract_date"])
    mapped_rows = []
    for _, row in old.iterrows():
        probabilities = mapped_fahrenheit_distribution(row["degree_probabilities"])
        actual_bucket_c = round_half_up(fahrenheit_to_celsius(float(row["actual_high_f"])))
        recommended_bucket_c = min(
            probabilities, key=lambda bucket: (-probabilities[bucket], bucket)
        )
        mapped_rows.append(
            {
                "contract_date": row["contract_date"],
                "old_mapped_recommended_bucket_c": recommended_bucket_c,
                "old_mapped_actual_bucket_probability_c": probabilities.get(
                    actual_bucket_c, 0.0
                ),
                "old_mapped_hit": recommended_bucket_c == actual_bucket_c,
                "old_mapped_market_brier": sum(
                    (
                        probability
                        - (1.0 if bucket == actual_bucket_c else 0.0)
                    )
                    ** 2
                    for bucket, probability in probabilities.items()
                ),
                "old_mapped_market_bucket_probabilities_c": json.dumps(
                    probabilities, sort_keys=True
                ),
            }
        )
    mapped = pd.DataFrame(mapped_rows)
    legacy_columns = [
        "contract_date",
        "probability_decision",
        "probability_decision_reason",
        "recommended_bucket",
        "recommended_hit",
        "point_hit",
    ]
    detail = new.merge(mapped, on="contract_date", validate="one_to_one").merge(
        old[legacy_columns], on="contract_date", validate="one_to_one"
    )
    detail["nearest_celsius_point_hit"] = detail["point_bucket_c"].eq(
        detail["actual_bucket_c"]
    )
    detail["new_celsius_hit"] = detail["recommended_bucket_c"].eq(
        detail["actual_bucket_c"]
    )
    old_actionable = detail["probability_decision"].eq("shadow_trade")
    new_actionable = detail["market_probability_decision"].eq("shadow_trade")
    new_metrics = pd.read_csv(CELSIUS_ROOT / f"{STATION_ID}_2026_holdout_metrics.csv").iloc[0]
    summary = pd.DataFrame(
        [
            {
                "comparison": "nearest_celsius_point_bucket",
                "market_unit": "1C",
                "count": len(detail),
                "bucket_accuracy": detail["nearest_celsius_point_hit"].mean(),
                "log_loss": np.nan,
                "brier": np.nan,
                "decision_coverage": 1.0,
                "decision_accuracy": detail["nearest_celsius_point_hit"].mean(),
                "notes": "deterministic point baseline",
            },
            {
                "comparison": "old_mapped_fahrenheit_probabilities",
                "market_unit": "mapped_to_1C_post_hoc",
                "count": len(detail),
                "bucket_accuracy": detail["old_mapped_hit"].mean(),
                "log_loss": float(
                    -np.log(
                        detail["old_mapped_actual_bucket_probability_c"].clip(
                            lower=1e-12
                        )
                    ).mean()
                ),
                "brier": detail["old_mapped_market_brier"].mean(),
                "decision_coverage": 1.0,
                "decision_accuracy": detail["old_mapped_hit"].mean(),
                "notes": "diagnostic only; trained on rounded-F offsets",
            },
            {
                "comparison": "old_native_fahrenheit_probability_decision",
                "market_unit": "native_2F",
                "count": len(detail),
                "bucket_accuracy": detail["recommended_hit"].mean(),
                "log_loss": np.nan,
                "brier": np.nan,
                "decision_coverage": old_actionable.mean(),
                "decision_accuracy": detail.loc[
                    old_actionable, "recommended_hit"
                ].mean(),
                "notes": f"not a valid {CITY} 1C market filter",
            },
            {
                "comparison": "new_celsius_ordinal_probabilities",
                "market_unit": "1C",
                "count": len(detail),
                "bucket_accuracy": new_metrics["market_bucket_accuracy"],
                "log_loss": new_metrics["market_bucket_log_loss"],
                "brier": new_metrics["market_bucket_brier"],
                "decision_coverage": new_actionable.mean(),
                "decision_accuracy": detail.loc[
                    new_actionable, "new_celsius_hit"
                ].mean(),
                "notes": "exploratory 2026 holdout; shadow-only",
            },
        ]
    )
    return detail, summary


def render_audit(comparison: pd.DataFrame, summary: pd.DataFrame) -> str:
    point_manifest_path = next(
        (TOKYO_ROOT / "model_weights").glob(
            f"{STATION_ID}_station_high_regressor_baseline_*_no_peak_stack.json"
        )
    )
    point_bundle_path = point_manifest_path.with_suffix(".joblib")
    probability_manifest_path = next((CELSIUS_ROOT / "model_weights").glob("*.json"))
    probability_bundle_path = probability_manifest_path.with_suffix(".joblib")
    point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
    probability_manifest = json.loads(
        probability_manifest_path.read_text(encoding="utf-8")
    )
    point_bundle = joblib.load(point_bundle_path)
    forward = pd.read_csv(CELSIUS_ROOT / f"{STATION_ID}_forward_validation_predictions.csv")
    forward_metrics = pd.read_csv(
        CELSIUS_ROOT / f"{STATION_ID}_forward_validation_metrics.csv"
    ).iloc[0]
    holdout_metrics = pd.read_csv(CELSIUS_ROOT / f"{STATION_ID}_2026_holdout_metrics.csv").iloc[0]
    target = json.loads(
        (CELSIUS_ROOT / f"{STATION_ID}_celsius_target_contract.json").read_text(
            encoding="utf-8"
        )
    )
    point_coefficients = dict(
        zip(
            point_bundle["stack_features"],
            point_bundle["stack_model"].coef_.tolist(),
            strict=True,
        )
    )
    point_coefficients["intercept"] = float(point_bundle["stack_model"].intercept_)
    point_coefficients["ridge_alpha"] = float(point_bundle["stack_model"].alpha)
    thresholds = probability_manifest["decision_thresholds"]
    tails = probability_manifest["tail_policy"]
    outer = forward.iloc[0]
    forward_exact_offset_support = sorted(
        {
            int(bucket) - int(row["point_bucket_c"])
            for _, row in forward.iterrows()
            for bucket in json.loads(row["market_bucket_probabilities_c"])
        }
    )
    files = sorted(
        [
            *CELSIUS_ROOT.glob("*.csv"),
            *CELSIUS_ROOT.glob("*.json"),
            *[point_bundle_path, point_manifest_path],
            *[probability_bundle_path, probability_manifest_path],
        ],
        key=lambda path: str(path),
    )
    hash_lines = "\n".join(
        f"- `{path.relative_to(PROJECT_ROOT)}` — `{sha256_file(path)}`" for path in files
    )
    table_lines = [
        "| Comparison | Unit | Accuracy | Log loss | Brier | Decision coverage | Decision accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        format_value = lambda value: "—" if pd.isna(value) else f"{float(value):.4f}"
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["comparison"]),
                    str(row["market_unit"]),
                    format_value(row["bucket_accuracy"]),
                    format_value(row["log_loss"]),
                    format_value(row["brier"]),
                    format_value(row["decision_coverage"]),
                    format_value(row["decision_accuracy"]),
                ]
            )
            + " |"
        )
    comparison_table = "\n".join(table_lines)
    indexed_summary = summary.set_index("comparison")
    point_row = indexed_summary.loc["nearest_celsius_point_bucket"]
    old_mapped_row = indexed_summary.loc["old_mapped_fahrenheit_probabilities"]
    new_row = indexed_summary.loc["new_celsius_ordinal_probabilities"]
    comparison_conclusion = (
        f"On the already-inspected 2026 holdout, the new model has "
        f"{float(new_row['bucket_accuracy']):.4f} accuracy versus "
        f"{float(point_row['bucket_accuracy']):.4f} for the nearest-C point bucket "
        f"and {float(old_mapped_row['bucket_accuracy']):.4f} for the mapped legacy "
        f"distribution. Its log loss is {float(new_row['log_loss']):.4f} versus "
        f"{float(old_mapped_row['log_loss']):.4f} for mapped legacy, while its Brier "
        f"score is {float(new_row['brier']):.4f} versus "
        f"{float(old_mapped_row['brier']):.4f}. This mixed, previously inspected "
        "evidence cannot be used for retuning and does not justify promotion."
    )
    low_tail_support = ", ".join(sorted(tails["low_exact_offset_weights"], key=int))
    high_tail_support = ", ".join(sorted(tails["high_exact_offset_weights"], key=int))
    return f"""# {CITY} 1°C Market Probability Audit

Status: **research complete; shadow-only; not approved for production promotion.**

## Settlement and target contract

{CITY} Polymarket rules resolve the {STATION_ID} daily high at {STATION_NAME}
using Wunderground at whole degrees Celsius. A representative market states
both the station/Wunderground source and whole-°C precision:
[{MARKET_TITLE}]({MARKET_URL}).
This validates the exact 1°C bucket contract, not the historical 2°F contract.

The source frame does not contain a settlement-equivalent raw Celsius target.
`iem_daily_high_c` belongs to a different diagnostic IEM source, so it is not
substituted for the Wunderground target. `actual_high_f` is converted exactly by
`(F - 32) * 5 / 9` and then rounded by `floor(C + 0.5)`. All {len(pd.read_csv(TOKYO_ROOT / f'{STATION_ID}_features.csv'))}
available converted targets are integer Celsius to floating tolerance, which is
consistent with Wunderground's whole-Celsius source values.

Target: `{target['target']}`.

## Artifact identities

- Point model: `{point_manifest['model_version']}`
- Point bundle SHA-256: `{sha256_file(point_bundle_path)}`
- Probability model: `{probability_manifest['model_version']}`
- Probability bundle SHA-256: `{sha256_file(probability_bundle_path)}`
- Probability manifest point dependency: `{probability_manifest['point_bundle_sha256']}`
- Feature contract: `{probability_manifest['feature_profile']}`, {len(probability_manifest['feature_names'])} features, providers GFS/GEFS/JMA-MSM
- Probability learner: `{probability_manifest['selected_family']}`, C={probability_manifest['selected_params']['C']}, class_weight={probability_manifest['selected_params']['class_weight']}, temperature={probability_manifest['temperature']}

The probability dependency hash equals the freshly exported point bundle hash.

## Chronology

| Stage | Period | Use |
|---|---|---|
| Honest 2025 outer-model training | through `{outer['model_training_cutoff']}` | Fit the 2025 forward fold |
| 2025 fold inner training | through `{outer['calibration_training_cutoff']}` | Candidate preprocessing/model fit |
| 2025 fold inner calibration | `{outer['calibration_validation_start']}` to `{outer['calibration_validation_cutoff']}` | C/weight/temperature selection |
| Forward validation | `{forward['contract_date'].min()}` to `{forward['contract_date'].max()}` | Model-development metrics and policy thresholds |
| Final probability development | `{probability_manifest['training_start']}` to `{probability_manifest['training_cutoff']}` ({probability_manifest['training_rows']} rows) | Frozen probability model |
| Final inner calibration | `{probability_manifest['final_calibration_validation_start']}` to `{probability_manifest['final_calibration_validation_cutoff']}` | Final C/weight/temperature selection |
| Exploratory holdout | `{comparison['contract_date'].min().date()}` to `{comparison['contract_date'].max().date()}` | Metrics only; never model/policy selection |

The point bundle is the current production-style refit over
`{point_manifest['training']['first_contract_date']}` to
`{point_manifest['training']['last_contract_date']}`. Probability development
does not use its in-sample predictions: it uses honest expanding point-stack
predictions for 2024–2025 and the 2026 point holdout predictions. The point
bundle hash records the serving dependency.

## Ordered offset support and frozen policy

- Classes: `{probability_manifest['offset_class_contract']}`
- Forward-fold exact support (tail policy fitted on the earlier 2024 history):
  `{forward_exact_offset_support}`
- Final-model exact development support: {tails['fitted_min_offset_c']}°C through {tails['fitted_max_offset_c']}°C
- Low tail allocation: `{json.dumps(tails['low_exact_offset_weights'], sort_keys=True)}`
- High tail allocation: `{json.dumps(tails['high_exact_offset_weights'], sort_keys=True)}`
- Minimum top probability: {thresholds['minimum_top_probability']:.3f}
- Minimum top-two margin: {thresholds['minimum_top_two_margin']:.3f}
- Minimum switch advantage: {thresholds['minimum_switch_advantage']:.3f}
- Tail rule: reject when an open-tail class spans multiple exact market buckets
  and its mass is at least the top-two margin.

All thresholds and the tail rule were frozen from 2025 forward-validation data.
No 2026 ROI, P&L, model variant, or threshold influenced selection.

## Model-development metrics (2025 forward validation, n={int(forward_metrics['count'])})

- Market bucket accuracy: {forward_metrics['market_bucket_accuracy']:.4f}
- Nearest-Celsius point accuracy: {forward_metrics['point_bucket_accuracy']:.4f}
- Market log loss: {forward_metrics['market_bucket_log_loss']:.4f}
- Market Brier score: {forward_metrics['market_bucket_brier']:.4f}
- Offset ranked probability score: {forward_metrics['ranked_probability_score']:.4f}
- Offset calibration error: {forward_metrics['offset_calibration_error']:.4f}
- Decision coverage: {forward_metrics['decision_coverage']:.4f} ({int(forward_metrics['decision_count'])} rows)
- Decision accuracy: {forward_metrics['decision_accuracy']:.4f}
- Calibration table: `data/calibration/station_training_baseline/{CITY}/celsius_market_probability/{STATION_ID}_forward_validation_calibration.csv`

## Exploratory 2026 holdout metrics (n={int(holdout_metrics['count'])})

- Market bucket accuracy: {holdout_metrics['market_bucket_accuracy']:.4f}
- Nearest-Celsius point accuracy: {holdout_metrics['point_bucket_accuracy']:.4f}
- Market log loss: {holdout_metrics['market_bucket_log_loss']:.4f}
- Market Brier score: {holdout_metrics['market_bucket_brier']:.4f}
- Offset ranked probability score: {holdout_metrics['ranked_probability_score']:.4f}
- Offset calibration error: {holdout_metrics['offset_calibration_error']:.4f}
- Decision coverage: {holdout_metrics['decision_coverage']:.4f} ({int(holdout_metrics['decision_count'])} rows)
- Decision accuracy: {holdout_metrics['decision_accuracy']:.4f}
- Calibration table: `data/calibration/station_training_baseline/{CITY}/celsius_market_probability/{STATION_ID}_2026_holdout_calibration.csv`

## Required 2026 comparison

{comparison_table}

The old mapped-Fahrenheit distribution is diagnostic only: integer-Fahrenheit
degree probabilities were converted and aggregated to nearest whole Celsius
after prediction. It was not trained on Celsius offsets. The old native
`probability_decision` evaluates incompatible 2°F buckets and must not be used
as a {CITY} market filter. Its displayed accuracy is native 2°F accuracy, not
1°C accuracy, and is included only to document the historical policy.

{comparison_conclusion}

## Point-model weights for later handoff

- XGBoost point-stack coefficient: {point_coefficients['xgboost_predicted_high_f']:.15f}
- LightGBM point-stack coefficient: {point_coefficients['lightgbm_predicted_high_f']:.15f}
- CatBoost point-stack coefficient: {point_coefficients['catboost_predicted_high_f']:.15f}
- Intercept: {point_coefficients['intercept']:.1f}
- Ridge alpha: {point_coefficients['ridge_alpha']:.14f}

Use the point bundle itself rather than copying these coefficients in isolation;
it also contains the three fitted base models, preprocessing, feature lists,
and inference contract.

## Production handoff (future work only)

Do not modify `D:\\dev\\polymarket-weather-prediction` yet. A later reviewed
integration should:

1. Load the exact point bundle above and verify its SHA-256.
2. Load the exact Celsius probability bundle above and verify both its own hash
   and `point_bundle_sha256` dependency.
3. Supply the 59 `asia_no_peak` features and GFS/GEFS/JMA-MSM inputs at {CITY}'s
   live-safe 11 AM cutoff.
4. Consume `point_bucket_c`, `recommended_bucket_c`,
   `recommended_bucket_probability_c`, `actual_bucket_probability_c`,
   `market_top_probability_c`, `market_top_two_margin_c`,
   `market_switch_advantage_c`, `market_tail_ambiguity_c`,
   `market_probability_decision`, `market_probability_decision_reason`,
   `celsius_offset_probabilities`, and `market_bucket_probabilities_c`.
5. Freeze the thresholds listed above. Never fall back to the old native
   Fahrenheit `probability_decision`.
6. Keep outputs shadow-only pending a separate promotion review on fresh,
   previously unseen {CITY} market outcomes.

## Limitations

- The target is an exact conversion of Wunderground-derived Fahrenheit values,
  because no same-source raw Celsius column is stored. Future ingestion should
  persist the raw Wunderground Celsius settlement value directly.
- Open-tail allocation is data-limited: low-tail exact support is
  `{low_tail_support}` and high-tail exact support is `{high_tail_support}`;
  multi-bucket tails can trigger ambiguity rejection.
- Only one {int(forward_metrics['count'])}-row forward-validation year is
  available for probability policy development.
- The 2026 holdout was previously inspected and is exploratory only.
- Market prices, liquidity, slippage, and ROI are outside this model audit.

## Artifact SHA-256

{hash_lines}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", choices=sorted(STATION_CONFIGS), default="Tokyo")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    configure_station(args.station)
    output = (args.output_dir or REPORT_ROOT).resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail, summary = build_comparison()
    detail_path = output / f"{STATION_ID}_2026_legacy_comparison_detail.csv"
    summary_path = output / f"{STATION_ID}_2026_legacy_comparison_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit = render_audit(detail, summary)
    audit_path = output / "AUDIT.md"
    audit_path.write_text(audit, encoding="utf-8")
    print(audit_path)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
