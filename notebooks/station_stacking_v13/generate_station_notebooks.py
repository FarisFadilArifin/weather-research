import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")

# Source-owned v13 contract markers:
# feature_version="v13"
# timing_mode=TIMING_MODE
# providers=PROVIDERS
# target_mode="remaining_warmup"
# base_model_methods=("xgboost", "lightgbm", "catboost")
# stack_enabled=True
# export_station_model_weights
# year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS
# hyperparameter_space="wide"
# MODEL_VERSION = "station_high_regressor_v13_weather_warmup_stack"
# OPTUNA_TRIALS = 30
# OPTUNA_STARTUP_TRIALS = 15
# STACK_OPTUNA_TRIALS = 30
# STACK_OPTUNA_STARTUP_TRIALS = 15


def _load_v11_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v11" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v11_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v11 notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rain_comparison_cells() -> list[dict]:
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Rain-Day V11 vs V13 MAE\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def _rain_day_flags(features: pd.DataFrame) -> pd.DataFrame:\n",
                "    work = features.copy()\n",
                "    work[\"contract_date\"] = pd.to_datetime(work[\"contract_date\"]).dt.strftime(\"%Y-%m-%d\")\n",
                "    rain_signal = pd.Series(False, index=work.index)\n",
                "    candidate_cols = [\n",
                "        \"observed_is_raining_at_as_of\",\n",
                "        \"v4_observed_precip_any\",\n",
                "        \"v4_any_forecast_precip\",\n",
                "        \"gfs_forecast_has_precip\",\n",
                "        \"hrrr_forecast_has_precip\",\n",
                "        \"nbm_forecast_has_precip\",\n",
                "    ]\n",
                "    for column in candidate_cols:\n",
                "        if column in work:\n",
                "            rain_signal |= pd.to_numeric(work[column], errors=\"coerce\").fillna(0).gt(0)\n",
                "    amount_cols = [\n",
                "        column\n",
                "        for column in work.columns\n",
                "        if column.endswith(\"forecast_precip_total_mm\")\n",
                "        or column.endswith(\"forecast_precip_max_1h_mm\")\n",
                "        or column in {\"observed_precip_recent_at_as_of\", \"precip_amount\"}\n",
                "    ]\n",
                "    for column in amount_cols:\n",
                "        rain_signal |= pd.to_numeric(work[column], errors=\"coerce\").fillna(0).gt(0.01)\n",
                "    return work.loc[rain_signal, [\"contract_date\"]].drop_duplicates()\n",
                "\n",
                "\n",
                "def _prediction_mae_by_method(predictions: pd.DataFrame, version: str, rainy_dates: pd.DataFrame) -> pd.DataFrame:\n",
                "    if predictions.empty:\n",
                "        return pd.DataFrame(columns=[\"version\", \"method\", \"rain_day_count\", \"mae_f\", \"rmse_f\"])\n",
                "    pred = predictions.copy()\n",
                "    pred[\"contract_date\"] = pd.to_datetime(pred[\"contract_date\"]).dt.strftime(\"%Y-%m-%d\")\n",
                "    pred = pred.merge(rainy_dates, on=\"contract_date\", how=\"inner\")\n",
                "    pred = pred.dropna(subset=[\"actual_high_f\", \"predicted_high_f\"])\n",
                "    if pred.empty:\n",
                "        return pd.DataFrame(columns=[\"version\", \"method\", \"rain_day_count\", \"mae_f\", \"rmse_f\"])\n",
                "    pred[\"error_f\"] = pred[\"actual_high_f\"] - pred[\"predicted_high_f\"]\n",
                "    return (\n",
                "        pred.groupby(\"method\", dropna=False)\n",
                "        .agg(\n",
                "            rain_day_count=(\"contract_date\", \"nunique\"),\n",
                "            prediction_count=(\"contract_date\", \"size\"),\n",
                "            mae_f=(\"error_f\", lambda s: float(np.mean(np.abs(s)))),\n",
                "            rmse_f=(\"error_f\", lambda s: float(np.sqrt(np.mean(np.square(s))))),\n",
                "        )\n",
                "        .reset_index()\n",
                "        .assign(version=version)\n",
                "        [[\"version\", \"method\", \"rain_day_count\", \"prediction_count\", \"mae_f\", \"rmse_f\"]]\n",
                "    )\n",
                "\n",
                "\n",
                "v11_pred_path = PROJECT_ROOT / \"data\" / \"calibration\" / \"station_stacking_v11\" / f\"{STATION_ID}_year_split_test_predictions.csv\"\n",
                "v11_feature_path = PROJECT_ROOT / \"data\" / \"calibration\" / \"station_stacking_v11\" / f\"{STATION_ID}_features.csv\"\n",
                "v11_predictions = pd.read_csv(v11_pred_path) if v11_pred_path.exists() else pd.DataFrame()\n",
                "rain_feature_frame = result.features if not result.features.empty else pd.read_csv(v11_feature_path)\n",
                "rainy_dates = _rain_day_flags(rain_feature_frame)\n",
                "\n",
                "rain_mae_comparison = pd.concat(\n",
                "    [\n",
                "        _prediction_mae_by_method(v11_predictions, \"v11\", rainy_dates),\n",
                "        _prediction_mae_by_method(result.test_predictions, \"v13\", rainy_dates),\n",
                "    ],\n",
                "    ignore_index=True,\n",
                ")\n",
                "rain_mae_comparison = rain_mae_comparison.sort_values([\"method\", \"version\"]).reset_index(drop=True)\n",
                "rain_mae_comparison\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rain_mae_wide = rain_mae_comparison.pivot_table(\n",
                "    index=\"method\",\n",
                "    columns=\"version\",\n",
                "    values=\"mae_f\",\n",
                "    aggfunc=\"first\",\n",
                ")\n",
                "if {\"v11\", \"v13\"}.issubset(rain_mae_wide.columns):\n",
                "    rain_mae_wide[\"v13_minus_v11_mae_f\"] = rain_mae_wide[\"v13\"] - rain_mae_wide[\"v11\"]\n",
                "rain_mae_wide.sort_values(\"v13_minus_v11_mae_f\" if \"v13_minus_v11_mae_f\" in rain_mae_wide else rain_mae_wide.columns[0])\n",
            ],
        },
    ]


def _notebook(station_id: str) -> dict:
    v11 = _load_v11_generator()
    notebook = v11._notebook(station_id)
    replacements = [
        ("Station Stacking v11", "Station Stacking v13"),
        ('feature_version="v11"', 'feature_version="v13"'),
        (
            'MODEL_VERSION = "station_high_regressor_v11_huber_ridge_stack"',
            'MODEL_VERSION = "station_high_regressor_v13_weather_warmup_stack"',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v13"',
        ),
        (
            '    ("v11", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"),',
            '    ("v11", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"),\n'
            '    ("v13", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v13"),',
        ),
        ("V11_DROPPED_FEATURE_COLUMNS", "V13_DROPPED_FEATURE_COLUMNS"),
        ("V11_FEATURE_COLUMNS", "V13_FEATURE_COLUMNS"),
        ("V11 Contract", "V13 Contract"),
        ("V11 Feature Coverage", "V13 Feature Coverage"),
        ("v11_feature_coverage", "v13_feature_coverage"),
        ("## V11", "## V13"),
        ("`feature_version=\\\"v11\\\"`", "`feature_version=\\\"v13\\\"`"),
        (
            "This version keeps the v9 remaining-warmup feature contract, trains XGBoost/LightGBM/CatBoost with Huber-style objectives, keeps ridge stacking enabled, and writes artifacts to `data/calibration/station_stacking_v11`.",
            "This version keeps the v11 remaining-warmup target and Huber/ridge stack, adds direct GFS/HRRR weather fields for rain/cloud/dewpoint/humidity, and writes artifacts to `data/calibration/station_stacking_v13`.",
        ),
        (
            '`feature_version="v11"` keeps the v9 feature contract and remaining-warmup target, but trains base learners with Huber-style objectives while retaining the ridge stack selected by validation MAE.',
            '`feature_version="v13"` keeps the v11 remaining-warmup target and stack semantics, then adds direct forecast-side weather features so rainy-day behavior can be evaluated against v11.',
        ),
        (
            'source_pipeline="notebooks/station_stacking_v11"',
            'source_pipeline="notebooks/station_stacking_v13"',
        ),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]
    insert_at = next(
        (
            index
            for index, cell in enumerate(notebook["cells"])
            if cell.get("cell_type") == "markdown" and "Feature Coverage" in "".join(cell.get("source", []))
        ),
        len(notebook["cells"]),
    )
    notebook["cells"][insert_at:insert_at] = _rain_comparison_cells()
    return notebook


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in TARGET_STATIONS:
        notebook = _notebook(station)
        path = out_dir / f"stacking_{station}_v13.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
