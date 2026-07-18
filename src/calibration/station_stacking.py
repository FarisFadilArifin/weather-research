from __future__ import annotations

import math
import os
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data_quality import (
    STRICT_QUALITY_ISSUES_COLUMN,
    STRICT_QUALITY_OK_COLUMN,
    add_strict_quality_flags,
    plausible_temperature_mask,
)


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")
TARGET_PROVIDERS = ("gfs", "hrrr")
OPTIONAL_PROVIDERS = ("nbm",)
TARGET = "actual_high_f"
TARGET_SOURCE_IEM_HOURLY = "iem_hourly"
TARGET_SOURCE_SETTLEMENT_FIRST = "settlement_first"
TARGET_SOURCE_WUNDERGROUND_ONLY = "wunderground_only"
OBSERVED_HIGH_SO_FAR_COLUMN = "observed_high_temp_through_as_of_f"
REMAINING_WARMUP_TARGET = "remaining_warmup_from_observed_high_so_far_f"
TARGET_MODE_DIRECT_HIGH = "actual_high"
TARGET_MODE_REMAINING_WARMUP = "remaining_warmup"
TRAINING_PROFILE_LEGACY = "legacy"
TRAINING_PROFILE_V20_ALIGNED = "v20_aligned"
TIMING_MODE_SAME_DAY_11AM = "same_day_11am"
TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE = "same_day_11am_live_safe"
TIMING_MODE_SAME_DAY_9AM_LIVE_SAFE = "same_day_9am_live_safe"
V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION = "v11_settlement_fix_temp"
V15_FEATURE_VERSIONS = ("v15_base", "v15_forecast_temp_at_as_of", "v15_precip_cloud")
V16_FEATURE_VERSIONS = ("v16_fused",)
V17_FEATURE_VERSIONS = ("v17_importance_015",)
V18_FEATURE_VERSIONS = ("v18",)
V18_1_FEATURE_VERSIONS = ("v18_1_nbm", "v18_1_rap")
V20_FEATURE_VERSION = "v20_peak_timing"
V20_KDAL_FIX_FEATURE_VERSION = "v20_kdal_nbm_physics"
V20_PEAK_TIMING_FEATURE_VERSIONS = (V20_FEATURE_VERSION, V20_KDAL_FIX_FEATURE_VERSION)
SUPPORTED_FEATURE_VERSIONS = (
    "base",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION,
    *V15_FEATURE_VERSIONS,
    *V16_FEATURE_VERSIONS,
    *V17_FEATURE_VERSIONS,
    *V18_FEATURE_VERSIONS,
    *V18_1_FEATURE_VERSIONS,
    V20_FEATURE_VERSION,
    V20_KDAL_FIX_FEATURE_VERSION,
)
CURRENT_OBS_TREND_FEATURE_VERSIONS = {"v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION, *V15_FEATURE_VERSIONS, *V16_FEATURE_VERSIONS, *V17_FEATURE_VERSIONS, *V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS, *V20_PEAK_TIMING_FEATURE_VERSIONS}
CLIMATOLOGY_FEATURE_VERSIONS = {"v9", "v10", "v11", "v12", "v13", "v14", V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION, *V15_FEATURE_VERSIONS, *V16_FEATURE_VERSIONS, *V17_FEATURE_VERSIONS, *V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS, *V20_PEAK_TIMING_FEATURE_VERSIONS}
HUBER_STACK_FEATURE_VERSIONS = {"v11", "v12", "v13", "v14", V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION, *V15_FEATURE_VERSIONS, *V16_FEATURE_VERSIONS, *V17_FEATURE_VERSIONS, *V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS, *V20_PEAK_TIMING_FEATURE_VERSIONS}
CATBOOST_HUBER_FEATURE_VERSIONS = {"v10", "v11", "v12", "v13", "v14", V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION, *V15_FEATURE_VERSIONS, *V16_FEATURE_VERSIONS, *V17_FEATURE_VERSIONS, *V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS, *V20_PEAK_TIMING_FEATURE_VERSIONS}
V14_ADDITIONAL_MIN_NON_NULL_FRACTION = 0.80
V15_ADDITIONAL_MIN_TRAIN_NON_NULL_FRACTION = 0.70
WEATHER_AGGREGATE_FEATURE_VERSIONS = {"v13", "v14", V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION, "v15_forecast_temp_at_as_of", "v15_precip_cloud", "v16_fused", "v17_importance_015", *V20_PEAK_TIMING_FEATURE_VERSIONS}
GUARDED_BLEND_CAPS_F = (1.0, 2.0, 3.0)
V18_NBM_RAP_SHARD_ROOT = Path("data/calibration/nbm_rap_features_shards_priority_20260702_full")
V20_PEAK_TIMING_SHARD_ROOTS = (
    Path("data/calibration/peak_timing_features_shards_full_history_2021_2023"),
    Path("data/calibration/peak_timing_features_shards_pilot_20260715"),
)
V18_BUCKET_LOG_LOSS_EPSILON = 1e-12

FORECAST_CACHE_PATTERNS = (
    ("sdk_nwp_0h_cache.csv", "sdk_11am_*/sdk_nwp_0h_cache.csv"),
    ("sdk_nwp_0h_cache.csv", "sdk_9am_*/sdk_nwp_0h_cache.csv"),
    ("direct_nbm_0h_cache.csv", "direct_nbm_*/direct_nbm_0h_cache.csv"),
)

CURRENT_OBSERVATION_CACHE_PATTERN = "sdk_current_obs_*/sdk_current_observations_*.csv"

V13_PROVIDER_NUMERIC_COLUMNS = [
    "forecast_temp_at_as_of_f",
    "dewpoint_at_as_of_f",
    "humidity_at_as_of",
    "wind_speed_at_as_of",
    "wind_direction_at_as_of",
    "cloud_cover_mean",
    "cloud_cover_max",
    "low_cloud_cover_mean",
    "low_cloud_cover_max",
    "pressure_mslp_mean",
    "pressure_surface_mean",
    "visibility_mean",
    "ceiling_min",
    "downward_shortwave_radiation_mean_w_m2",
    "shortwave_radiation_mean_w_m2",
]

FORECAST_COLUMNS = [
    "station_id",
    "station_name",
    "airport_name",
    "provider",
    "model",
    "source_label",
    "timing_mode",
    "cycle_selection_policy",
    "contract_date",
    "forecast_as_of",
    "issued_at",
    "forecast_window_start",
    "forecast_window_end",
    "horizon_hours",
    "raw_forecast_high_f",
    "forecast_hour_min",
    "forecast_hour_max",
    "grid_dist_km_mean",
    "precip_amount",
    "forecast_precip_total_mm",
    "forecast_precip_max_1h_mm",
    "forecast_precip_hours_count",
    "forecast_has_precip",
    "forecast_precip_intensity_code",
    "forecast_precip_intensity",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_direction_mean",
    "wind_gust_max",
    "dewpoint_mean_f",
    "humidity_mean",
    *V13_PROVIDER_NUMERIC_COLUMNS,
    "data_source",
    "source_file_or_url",
    "fetch_status",
    "unavailable_reason",
]

PROVIDER_NUMERIC_COLUMNS = [
    "horizon_hours",
    "raw_forecast_high_f",
    "forecast_hour_min",
    "forecast_hour_max",
    "grid_dist_km_mean",
    "precip_amount",
    "forecast_precip_total_mm",
    "forecast_precip_max_1h_mm",
    "forecast_precip_hours_count",
    "forecast_has_precip",
    "forecast_precip_intensity_code",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_direction_mean",
    "wind_gust_max",
    "dewpoint_mean_f",
    "humidity_mean",
]
PROVIDER_FORECAST_NUMERIC_COLUMNS = [*PROVIDER_NUMERIC_COLUMNS, *V13_PROVIDER_NUMERIC_COLUMNS]

PROVIDER_TEXT_COLUMNS = [
    "model",
    "source_label",
    "cycle_selection_policy",
    "forecast_as_of",
    "issued_at",
    "forecast_window_start",
    "forecast_window_end",
    "forecast_precip_intensity",
    "data_source",
    "source_file_or_url",
    "source_cache_dir",
]

OBSERVED_NUMERIC_COLUMNS = [
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_dewpoint_at_as_of_f",
    "observed_humidity_at_as_of",
    "observed_wind_speed_at_as_of",
    "observed_wind_direction_at_as_of",
    "observed_wind_gust_at_as_of",
    "observed_peak_wind_gust_at_as_of",
    "observed_peak_wind_direction_at_as_of",
    "observed_pressure_at_as_of",
    "observed_altimeter_inhg_at_as_of",
    "observed_sea_level_pressure_mb_at_as_of",
    "observed_visibility_at_as_of",
    "observed_ceiling_at_as_of",
    "observed_cloud_cover_at_as_of",
    "observed_precip_recent_at_as_of",
    "observed_snow_depth_at_as_of",
    "observed_as_of_age_minutes",
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
]

OBSERVED_TEXT_COLUMNS = [
    "observed_pressure_source",
    "observed_weather_code_at_as_of",
    "observed_as_of_time_local",
    "observed_as_of_time_utc",
    "observed_source",
    "observed_observation_type",
    "observed_qc_field",
    "observed_raw_metar",
    "observed_data_source",
    "observed_fetch_status",
    "observed_unavailable_reason",
]

OBSERVED_CATEGORICAL_FEATURES = [
    "observed_pressure_source",
    "observed_weather_code_at_as_of",
    "observed_precip_intensity",
    "observed_observation_type",
    "observed_fetch_status",
]

HIGH_COLUMNS = {provider: f"{provider}_high_f" for provider in TARGET_PROVIDERS}
HIGH_COLUMNS.update({provider: f"{provider}_high_f" for provider in OPTIONAL_PROVIDERS})
BASELINE_METHODS = [
    "gfs_raw",
    "hrrr_raw",
    "nbm_raw",
    "provider_mean",
    "provider_median",
    "best_raw_provider",
]
BASE_MODEL_METHODS = ["xgboost", "lightgbm", "catboost"]
DEFAULT_BASE_MODEL_METHODS = tuple(BASE_MODEL_METHODS)
STACK_METHOD = "ridge_stack"
V12_GUARDED_BLEND_METHODS = tuple(f"guarded_blend_cap_{cap:g}f" for cap in GUARDED_BLEND_CAPS_F)
YEAR_SPLIT_SCOREBOARD_METHODS = (
    *BASE_MODEL_METHODS,
    STACK_METHOD,
    *V12_GUARDED_BLEND_METHODS,
    "provider_mean",
    "provider_median",
    "nbm_raw",
    "hrrr_raw",
    "gfs_raw",
)
YEAR_SPLIT_VALIDATION_WEIGHTS = {2024: 0.35, 2025: 0.65}
STACK_FEATURE_SETS = {
    "models_only": tuple(BASE_MODEL_METHODS),
    "models_plus_raw": (*BASE_MODEL_METHODS, "hrrr_raw", "gfs_raw"),
}
REQUIRED_MODEL_PACKAGES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "optuna": "optuna",
}
V5_FEATURE_COLUMNS = [
    "v2_recent_heat_anomaly_f",
    "v2_recent_heat_momentum_f",
    "v2_morning_warmup_to_consensus_f",
    "v2_consensus_minus_7d_actual_f",
    "v2_spread_per_warmup_f",
    "v2_humidity_warmup_interaction",
    "v3_high_so_far_above_current_f",
    "v3_remaining_warmup_from_high_so_far_f",
    "v3_high_so_far_minus_lag_1d_f",
    "v3_high_so_far_minus_7d_actual_f",
    "v3_remaining_warmup_per_spread_f",
    "v3_humidity_remaining_warmup_interaction",
    "v4_forecast_precip_total_mean_mm",
    "v4_forecast_precip_total_max_mm",
    "v4_forecast_precip_total_spread_mm",
    "v4_forecast_precip_max_1h_mean_mm",
    "v4_forecast_precip_hours_mean",
    "v4_forecast_precip_intensity_mean",
    "v4_forecast_precip_intensity_max",
    "v4_any_forecast_precip",
    "v4_all_forecast_precip",
    "v4_observed_precip_any",
    "v4_observed_precip_recent_mm_est",
    "v4_forecast_total_minus_observed_recent_mm",
    "v4_forecast_observed_precip_match",
    "v4_forecast_wet_observed_dry",
    "v4_observed_wet_forecast_dry",
    "v4_precip_humidity_interaction",
    "v4_precip_remaining_warmup_interaction",
]
V6_FEATURE_COLUMNS = [
    *V5_FEATURE_COLUMNS,
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
]
V7_FEATURE_COLUMNS = V6_FEATURE_COLUMNS
V8_FEATURE_COLUMNS = [
    *V7_FEATURE_COLUMNS,
    "v8_provider_max_remaining_from_high_so_far_f",
    "v8_provider_min_remaining_from_high_so_far_f",
    "v8_provider_median_remaining_from_high_so_far_f",
    "v8_provider_spread_per_remaining_warmup_f",
    "v8_month_remaining_warmup_mean_f",
    "v8_month_remaining_warmup_count",
    "v8_recent_remaining_warmup_7d_mean_f",
    "v8_recent_remaining_warmup_30d_mean_f",
    "v8_provider_mean_remaining_vs_month_normal_f",
    "v8_cloud_cover_mean_remaining_warmup_interaction",
    "v8_cloud_cover_max_remaining_warmup_interaction",
    "v8_precip_total_remaining_warmup_interaction",
    "v8_precip_max_1h_remaining_warmup_interaction",
    "v8_wind_speed_mean_remaining_warmup_interaction",
    "v8_wind_gust_max_remaining_warmup_interaction",
    "v8_forecast_dewpoint_mean_f",
    "v8_forecast_dewpoint_depression_mean_f",
    "v8_dewpoint_mean_remaining_warmup_interaction",
    "v8_dewpoint_depression_remaining_warmup_interaction",
]
V9_CLIMATOLOGY_FEATURE_COLUMNS = [
    "climatology_high_10y_f",
    "climatology_high_10y_std_f",
    "climatology_high_10y_count",
    "provider_mean_minus_climatology_10y_f",
    "observed_temp_minus_climatology_10y_f",
    "observed_high_so_far_minus_climatology_10y_f",
]
V9_FEATURE_COLUMNS = [
    *V8_FEATURE_COLUMNS,
    *V9_CLIMATOLOGY_FEATURE_COLUMNS,
]
V10_FEATURE_COLUMNS = V9_FEATURE_COLUMNS
V11_FEATURE_COLUMNS = V9_FEATURE_COLUMNS
V12_FEATURE_COLUMNS = V9_FEATURE_COLUMNS
V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS = [
    "v11sf_forecast_temp_11am_mean_f",
    "v11sf_forecast_temp_11am_median_f",
    "v11sf_forecast_temp_11am_minus_observed_f",
    "v11sf_forecast_temp_11am_abs_error_f",
    "v11sf_forecast_temp_11am_warm_error_f",
    "v11sf_forecast_temp_11am_cool_error_f",
    "v11sf_forecast_temp_11am_spread_f",
    "v11sf_forecast_temp_11am_provider_count",
    "v11sf_forecast_temp_bias_remaining_warmup_interaction",
    "v11sf_observation_adjusted_provider_high_f",
    "v11sf_forecast_warmup_after_11am_f",
]
V13_ADDITIONAL_FEATURE_COLUMNS = [
    "v13_forecast_temp_at_as_of_mean_f",
    "v13_forecast_temp_at_as_of_minus_observed_mean_f",
    "v13_forecast_temp_at_as_of_spread_f",
    "v13_cloud_cover_mean_pct",
    "v13_cloud_cover_max_pct",
    "v13_low_cloud_cover_mean_pct",
    "v13_visibility_mean_m",
    "v13_ceiling_min_m",
    "v13_low_visibility_flag",
    "v13_low_ceiling_flag",
    "v13_pressure_mslp_mean_pa",
    "v13_shortwave_mean_w_m2",
    "v13_weather_available_provider_count",
    "v13_cloud_cover_remaining_warmup_interaction",
    "v13_low_cloud_remaining_warmup_interaction",
    "v13_precip_cloud_remaining_warmup_interaction",
    "v13_forecast_temp_bias_remaining_warmup_interaction",
]
V13_FEATURE_COLUMNS = [
    *V9_FEATURE_COLUMNS,
    *V13_ADDITIONAL_FEATURE_COLUMNS,
]
V14_ADDITIONAL_FEATURE_COLUMNS = [
    "v4_forecast_precip_total_mean_mm",
    "v4_forecast_precip_total_max_mm",
    "v4_forecast_precip_max_1h_mean_mm",
    "v4_forecast_precip_hours_mean",
    "v4_forecast_precip_intensity_mean",
    "v4_precip_remaining_warmup_interaction",
    "v13_cloud_cover_mean_pct",
    "v13_cloud_cover_max_pct",
    "v13_cloud_cover_remaining_warmup_interaction",
    "v13_precip_cloud_remaining_warmup_interaction",
    "v13_forecast_temp_at_as_of_mean_f",
    "v13_forecast_temp_at_as_of_minus_observed_mean_f",
    "v13_forecast_temp_at_as_of_spread_f",
    "v13_forecast_temp_bias_remaining_warmup_interaction",
    "v13_weather_available_provider_count",
]
V14_FEATURE_COLUMNS = [
    *V11_FEATURE_COLUMNS,
    *[column for column in V14_ADDITIONAL_FEATURE_COLUMNS if column not in V11_FEATURE_COLUMNS],
]
V15_BASE_FEATURE_COLUMNS = V11_FEATURE_COLUMNS
V15_FORECAST_TEMP_AT_AS_OF_FEATURE_COLUMNS = [
    "v13_forecast_temp_at_as_of_mean_f",
    "v13_forecast_temp_at_as_of_minus_observed_mean_f",
    "v13_forecast_temp_at_as_of_spread_f",
    "v13_forecast_temp_bias_remaining_warmup_interaction",
]
V15_PRECIP_CLOUD_FEATURE_COLUMNS = [
    "v13_cloud_cover_mean_pct",
    "v13_cloud_cover_max_pct",
    "v13_cloud_cover_remaining_warmup_interaction",
    "v13_precip_cloud_remaining_warmup_interaction",
]
V15_ADDITIONAL_FEATURE_COLUMNS = [
    *V15_FORECAST_TEMP_AT_AS_OF_FEATURE_COLUMNS,
    *V15_PRECIP_CLOUD_FEATURE_COLUMNS,
]
V15_FEATURE_COLUMNS = [
    *V15_BASE_FEATURE_COLUMNS,
    *[column for column in V15_ADDITIONAL_FEATURE_COLUMNS if column not in V15_BASE_FEATURE_COLUMNS],
]
V16_BLOCKED_BASE_FEATURE_COLUMNS = {
    "v8_cloud_cover_mean_remaining_warmup_interaction",
    "v8_cloud_cover_max_remaining_warmup_interaction",
}
V16_BASE_FEATURE_COLUMNS = [column for column in V11_FEATURE_COLUMNS if column not in V16_BLOCKED_BASE_FEATURE_COLUMNS]
V16_ADDITIONAL_FEATURE_COLUMNS = [
    *V15_FORECAST_TEMP_AT_AS_OF_FEATURE_COLUMNS,
    *V15_PRECIP_CLOUD_FEATURE_COLUMNS,
]
V16_FEATURE_COLUMNS = [
    *V16_BASE_FEATURE_COLUMNS,
    *[column for column in V16_ADDITIONAL_FEATURE_COLUMNS if column not in V16_BASE_FEATURE_COLUMNS],
]
V17_IMPORTANCE_015_FEATURE_COLUMNS = [
    "nbm_high_minus_observed_high_temp_f",
    "v8_provider_median_remaining_from_high_so_far_f",
    "observed_high_so_far_change_since_9am_f",
    "nbm_high_minus_observed_temp_f",
    "v3_remaining_warmup_per_spread_f",
    "v8_provider_max_remaining_from_high_so_far_f",
    "v8_provider_mean_remaining_vs_month_normal_f",
    "hrrr_high_minus_observed_high_temp_f",
    "observed_high_temp_minus_temp_at_as_of_f",
    "gfs_high_minus_observed_high_temp_f",
    "observed_cloud_cover_at_as_of",
    "v3_remaining_warmup_from_high_so_far_f",
    "v3_humidity_remaining_warmup_interaction",
    "v13_forecast_temp_bias_remaining_warmup_interaction",
    "gfs_high_minus_observed_temp_f",
    "hrrr_rolling_bias_30d_f",
    "observed_temp_change_last_3h_f",
]
V17_ADDITIONAL_FEATURE_COLUMNS = [
    "v13_forecast_temp_bias_remaining_warmup_interaction",
]
V17_FEATURE_COLUMNS = V17_IMPORTANCE_015_FEATURE_COLUMNS
V18_NBM_CURVE_FEATURE_COLUMNS = [
    "nbm_t11l_f",
    "nbm_t12l_f",
    "nbm_t13l_f",
    "nbm_t14l_f",
    "nbm_t15l_f",
    "nbm_t16l_f",
    "nbm_t17l_f",
    "nbm_t18l_f",
    "nbm_max_post11_f",
    "nbm_hour_of_max_local",
    "nbm_slope_11_14_f",
    "nbm_slope_14_17_f",
    "nbm_cooling_onset_hour_local",
]
V18_RAP_PHYSICS_FEATURE_COLUMNS = [
    "rap_t11l_f",
    "rap_t12l_f",
    "rap_t13l_f",
    "rap_t14l_f",
    "rap_t15l_f",
    "rap_t16l_f",
    "rap_t17l_f",
    "rap_t18l_f",
    "rap_dswrf_12_17_sum",
    "rap_lcdc_12_17_mean",
    "rap_mcdc_12_17_mean",
    "rap_hcdc_12_17_mean",
    "rap_boundary_layer_cloud_12_17_mean",
    "rap_hpbl_max_12_17",
    "rap_hpbl_growth_12_15",
    "rap_t925_15l_f",
    "rap_t850_18l_f",
    "rap_t925_minus_rap_t11_f",
    "rap_mixed_down_margin_f",
    "rap_pwat_12l",
    "rap_cape_15l",
    "rap_cin_15l",
    "rap_cin_abs_15l",
    "rap_wind_speed_11l_mph",
    "rap_wind_direction_11l_deg",
    "rap_wind_direction_12_17_mean_deg",
    "rap_deep_mixing_flag",
]
V18_STATION_SPECIFIC_FEATURE_COLUMNS = [
    "katl_ne_wind_component_11_15_mph",
    "katl_cad_like_flag",
    "kdal_dewpoint_gradient_west_east_f",
    "kdal_dryline_proximity_score",
    "kmia_rap_onshore_component_11l_mph",
    "kmia_rap_onshore_component_13_15_mph",
    "kmia_sea_breeze_index",
]
V18_ADDITIONAL_FEATURE_COLUMNS = [
    *V18_NBM_CURVE_FEATURE_COLUMNS,
    *V18_RAP_PHYSICS_FEATURE_COLUMNS,
    *V18_STATION_SPECIFIC_FEATURE_COLUMNS,
]
V18_FEATURE_COLUMNS = [
    *V11_FEATURE_COLUMNS,
    *[column for column in V18_ADDITIONAL_FEATURE_COLUMNS if column not in V11_FEATURE_COLUMNS],
]
V18_1_NBM_FEATURE_COLUMNS = [
    *V11_FEATURE_COLUMNS,
    *[column for column in V18_NBM_CURVE_FEATURE_COLUMNS if column not in V11_FEATURE_COLUMNS],
]
V18_1_RAP_FEATURE_COLUMNS = [
    *V11_FEATURE_COLUMNS,
    *[
        column
        for column in [*V18_RAP_PHYSICS_FEATURE_COLUMNS, *V18_STATION_SPECIFIC_FEATURE_COLUMNS]
        if column not in V11_FEATURE_COLUMNS
    ],
]
V18_AUDIT_COLUMNS = [
    "local_hours",
    "row_status",
    "nbm_core_fetch_status",
    "nbm_core_unavailable_reason",
    "nbm_issued_at",
    "nbm_forecast_as_of",
    "nbm_cycle_selection_policy",
    "nbm_hour_count_requested",
    "nbm_hour_count_returned",
    "rap_fetch_status",
    "rap_unavailable_reason",
    "rap_source_model",
    "physics_source_model",
    "physics_fetch_status",
    "physics_unavailable_reason",
    "rap_issued_at",
    "rap_forecast_as_of",
    "rap_cycle_selection_policy",
    "rap_hour_count_requested",
    "rap_hour_count_returned",
    "v18_shard_source_path",
    "v18_shard_duplicate_count",
]
V20_PEAK_TIMING_RAW_FEATURE_COLUMNS = [
    "nbm_t11l_f",
    "nbm_max_post11_f",
    "nbm_hour_of_max_local",
    "nbm_slope_11_14_f",
    "nbm_slope_14_17_f",
    "nbm_cooling_onset_hour_local",
    "hrrr_t11l_f",
    "hrrr_max_post11_f",
    "hrrr_hour_of_max_local",
    "hrrr_peak_at_window_end",
    "hrrr_slope_11_14_f",
    "hrrr_slope_14_to_peak_f",
    "hrrr_solar_energy_11_to_hrrr_peak_wh_m2",
    "hrrr_solar_energy_11_to_nbm_peak_wh_m2",
    "hrrr_precip_total_11_to_hrrr_peak_mm",
    "hrrr_precip_wet_hours_11_to_hrrr_peak",
    "hrrr_precip_total_11_to_nbm_peak_mm",
    "hrrr_precip_wet_hours_11_to_nbm_peak",
    "hrrr_no_precip_11_18",
    "hrrr_tcc_11_to_hrrr_peak_mean_pct",
    "hrrr_tcc_11_to_hrrr_peak_max_pct",
    "hrrr_tcc_11_to_nbm_peak_mean_pct",
    "hrrr_tcc_11_to_nbm_peak_max_pct",
    "hrrr_blcc_11_to_nbm_peak_mean_pct",
    "hrrr_lcc_11_to_nbm_peak_mean_pct",
]
V20_ENGINEERED_FEATURE_COLUMNS = [
    "v20_hrrr_t11_minus_observed_f",
    "v20_nbm_t11_minus_observed_f",
    "v20_hrrr_remaining_rise_f",
    "v20_nbm_remaining_rise_f",
    "v20_hrrr_observation_adjusted_high_f",
    "v20_nbm_observation_adjusted_high_f",
    "v20_adjusted_high_mean_f",
    "v20_adjusted_high_spread_f",
    "v20_model_high_difference_f",
    "v20_peak_hour_difference",
    "v20_solar_energy_11_14_wh_m2",
    "v20_solar_energy_15_18_wh_m2",
    "v20_tcc_change_11_to_hrrr_peak_pct",
    "v20_tcc_change_11_to_nbm_peak_pct",
    "v20_rain_before_hrrr_peak",
    "v20_rain_before_nbm_peak",
    "v20_rain_present_11_18",
    "v20_precip_onset_minus_hrrr_peak_hours_zero_filled",
    "v20_precip_onset_minus_nbm_peak_hours_zero_filled",
]
V20_FEATURE_COLUMNS = [
    *V11_FEATURE_COLUMNS,
    *V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS,
    *V20_PEAK_TIMING_RAW_FEATURE_COLUMNS,
    *V20_ENGINEERED_FEATURE_COLUMNS,
]
V20_KDAL_NBM_PHYSICS_RAW_FEATURE_COLUMNS = [
    "nbm_t11l_f",
    "nbm_max_post11_f",
    "nbm_hour_of_max_local",
    "nbm_slope_11_14_f",
    "nbm_slope_14_17_f",
    "nbm_cooling_onset_hour_local",
    "hrrr_solar_energy_11_to_hrrr_peak_wh_m2",
    "hrrr_solar_energy_11_to_nbm_peak_wh_m2",
    "hrrr_precip_total_11_to_hrrr_peak_mm",
    "hrrr_precip_wet_hours_11_to_hrrr_peak",
    "hrrr_precip_total_11_to_nbm_peak_mm",
    "hrrr_precip_wet_hours_11_to_nbm_peak",
    "hrrr_no_precip_11_18",
    "hrrr_tcc_11_to_hrrr_peak_mean_pct",
    "hrrr_tcc_11_to_hrrr_peak_max_pct",
    "hrrr_tcc_11_to_nbm_peak_mean_pct",
    "hrrr_tcc_11_to_nbm_peak_max_pct",
    "hrrr_blcc_11_to_nbm_peak_mean_pct",
    "hrrr_lcc_11_to_nbm_peak_mean_pct",
]
V20_KDAL_NBM_PHYSICS_ENGINEERED_FEATURE_COLUMNS = [
    "v20_nbm_t11_minus_observed_f",
    "v20_nbm_remaining_rise_f",
    "v20_nbm_observation_adjusted_high_f",
    "v20_solar_energy_11_14_wh_m2",
    "v20_solar_energy_15_18_wh_m2",
    "v20_tcc_change_11_to_nbm_peak_pct",
    "v20_rain_before_nbm_peak",
    "v20_rain_present_11_18",
    "v20_precip_onset_minus_nbm_peak_hours_zero_filled",
]
V20_KDAL_NBM_PHYSICS_FEATURE_COLUMNS = [
    *V11_FEATURE_COLUMNS,
    *V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS,
    *V20_KDAL_NBM_PHYSICS_RAW_FEATURE_COLUMNS,
    *V20_KDAL_NBM_PHYSICS_ENGINEERED_FEATURE_COLUMNS,
]
V20_AUDIT_COLUMNS = [
    "local_hours",
    "row_status",
    "schema_version",
    "feature_profile",
    "timing_mode_v20_peak",
    "nbm_core_fetch_status",
    "nbm_core_unavailable_reason",
    "hrrr_fetch_status",
    "hrrr_unavailable_reason",
    "hrrr_profile_complete",
    "v20_shard_source_path",
    "v20_shard_duplicate_count",
]
V8_DROPPED_FEATURE_COLUMNS = {
    "gfs_available",
    "gfs_missing",
    "hrrr_available",
    "hrrr_missing",
    "nbm_available",
    "nbm_missing",
    "provider_count_available",
    "gfs_as_of_hour_local",
    "hrrr_as_of_hour_local",
    "nbm_as_of_hour_local",
    "gfs_horizon_hours",
    "hrrr_horizon_hours",
    "nbm_horizon_hours",
    "gfs_forecast_window_hours",
    "hrrr_forecast_window_hours",
    "nbm_forecast_window_hours",
    "hrrr_forecast_hour_min",
    "hrrr_forecast_hour_max",
    "hrrr_forecast_lead_hours",
    "gfs_issue_hour_utc",
    "hrrr_issue_hour_local",
    "hrrr_issue_hour_utc",
    "nbm_issue_hour_local",
    "nbm_issue_hour_utc",
    "lat",
    "lon",
    "year",
    "is_active_polymarket_station",
    "gfs_grid_dist_km_mean",
    "hrrr_grid_dist_km_mean",
    "gfs_hrrr_grid_dist_km_mean_diff_f",
    "gfs_hrrr_grid_dist_km_mean_abs_diff_f",
    "observed_fetch_status",
    "observed_observation_type",
    "observed_pressure_source",
}
V9_DROPPED_FEATURE_COLUMNS = {
    *V8_DROPPED_FEATURE_COLUMNS,
    "climatology_source_start_year",
    "climatology_source_end_year",
    "actual_minus_climatology_10y_f_DIAGNOSTIC_ONLY",
}
V10_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V11_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V11_SETTLEMENT_FIX_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V12_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V13_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V14_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V15_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V16_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V17_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
V18_DROPPED_FEATURE_COLUMNS = {*V9_DROPPED_FEATURE_COLUMNS, *V18_AUDIT_COLUMNS}
V20_DROPPED_FEATURE_COLUMNS = {*V9_DROPPED_FEATURE_COLUMNS, *V20_AUDIT_COLUMNS}


@dataclass(frozen=True)
class StationStackingConfig:
    station_id: str
    project_root: str | Path = "."
    timing_mode: str = "same_day_11am"
    providers: tuple[str, ...] = TARGET_PROVIDERS
    min_train_rows: int = 180
    refit_days: int = 30
    min_meta_train_rows: int = 60
    random_state: int = 42
    fast_mode: bool = False
    fast_max_validation_blocks: int = 3
    optuna_trials: int | None = None
    stack_optuna_trials: int | None = None
    optuna_verbose: bool = False
    optuna_metric: str = "rmse_f"
    optuna_startup_trials: int = 30
    stack_optuna_startup_trials: int = 30
    optuna_storage_path: str | Path | None = None
    climatology_normals_path: str | Path | None = None
    target_mode: str = TARGET_MODE_DIRECT_HIGH
    target_source: str = TARGET_SOURCE_IEM_HOURLY
    feature_version: str = "base"
    training_profile: str = TRAINING_PROFILE_LEGACY
    hyperparameter_space: str = "default"
    catboost_max_iterations: int | None = None
    catboost_max_depth: int | None = None
    catboost_min_learning_rate: float | None = None
    catboost_max_border_count: int | None = None
    base_model_methods: tuple[str, ...] = DEFAULT_BASE_MODEL_METHODS
    stack_enabled: bool = True
    year_split_folds: tuple[Any, ...] | None = None
    year_split_validation_weights: dict[int, float] | None = None
    year_split_test_train_years: tuple[int, int] = (2021, 2025)
    year_split_test_year: int = 2026
    feature_importance_repeats: int | None = None
    max_feature_missing_fraction: float | None = None
    output_dir: str | Path | None = None

    def resolved_project_root(self) -> Path:
        return Path(self.project_root).resolve()

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir).resolve()
        return self.resolved_project_root() / "data" / "calibration" / "station_stacking"

    def resolved_optuna_storage_path(self) -> Path | None:
        if self.optuna_storage_path is not None:
            return Path(self.optuna_storage_path).resolve()
        if self.effective_feature_version in CURRENT_OBS_TREND_FEATURE_VERSIONS:
            return self.resolved_output_dir() / f"{self.station_id.upper()}_optuna.sqlite3"
        return None

    def resolved_optuna_storage_uri(self) -> str | None:
        path = self.resolved_optuna_storage_path()
        if path is None:
            return None
        return f"sqlite:///{path.as_posix()}"

    @property
    def effective_min_train_rows(self) -> int:
        return min(self.min_train_rows, 30) if self.fast_mode else self.min_train_rows

    @property
    def effective_min_meta_train_rows(self) -> int:
        return min(self.min_meta_train_rows, 15) if self.fast_mode else self.min_meta_train_rows

    @property
    def effective_refit_days(self) -> int:
        return min(self.refit_days, 14) if self.fast_mode else self.refit_days

    @property
    def effective_optuna_trials(self) -> int:
        if self.optuna_trials is not None:
            return max(1, int(self.optuna_trials))
        return 8 if self.fast_mode else 50

    @property
    def effective_stack_optuna_trials(self) -> int:
        if self.stack_optuna_trials is not None:
            return max(1, int(self.stack_optuna_trials))
        return 8 if self.fast_mode else min(self.effective_optuna_trials, 50)

    @property
    def effective_optuna_startup_trials(self) -> int:
        return max(0, int(self.optuna_startup_trials))

    @property
    def effective_stack_optuna_startup_trials(self) -> int:
        return max(0, int(self.stack_optuna_startup_trials))

    @property
    def effective_feature_version(self) -> str:
        version = str(self.feature_version or "base").strip().lower()
        if version in {"", "none"}:
            version = "base"
        if version not in SUPPORTED_FEATURE_VERSIONS:
            raise ValueError(
                "feature_version must be one of: "
                + ", ".join(f"'{item}'" for item in SUPPORTED_FEATURE_VERSIONS)
            )
        return version

    @property
    def effective_training_profile(self) -> str:
        value = str(self.training_profile or TRAINING_PROFILE_LEGACY).strip().lower().replace("-", "_")
        aliases = {
            "": TRAINING_PROFILE_LEGACY,
            "default": TRAINING_PROFILE_LEGACY,
            "legacy": TRAINING_PROFILE_LEGACY,
            "v20": TRAINING_PROFILE_V20_ALIGNED,
            "v20_aligned": TRAINING_PROFILE_V20_ALIGNED,
        }
        profile = aliases.get(value, value)
        if profile not in {TRAINING_PROFILE_LEGACY, TRAINING_PROFILE_V20_ALIGNED}:
            raise ValueError("training_profile must be one of: 'legacy' or 'v20_aligned'")
        return profile

    @property
    def effective_target_source(self) -> str:
        value = str(self.target_source or TARGET_SOURCE_IEM_HOURLY).strip().lower().replace("-", "_")
        aliases = {
            "iem": TARGET_SOURCE_IEM_HOURLY,
            "iem_hourly": TARGET_SOURCE_IEM_HOURLY,
            "hourly": TARGET_SOURCE_IEM_HOURLY,
            "actual_highs": TARGET_SOURCE_IEM_HOURLY,
            "settlement": TARGET_SOURCE_SETTLEMENT_FIRST,
            "settlement_first": TARGET_SOURCE_SETTLEMENT_FIRST,
            "polymarket": TARGET_SOURCE_SETTLEMENT_FIRST,
            "wunderground": TARGET_SOURCE_WUNDERGROUND_ONLY,
            "wunderground_only": TARGET_SOURCE_WUNDERGROUND_ONLY,
            "wu_only": TARGET_SOURCE_WUNDERGROUND_ONLY,
        }
        source = aliases.get(value, value)
        if source not in {TARGET_SOURCE_IEM_HOURLY, TARGET_SOURCE_SETTLEMENT_FIRST, TARGET_SOURCE_WUNDERGROUND_ONLY}:
            raise ValueError("target_source must be one of: 'iem_hourly', 'settlement_first', or 'wunderground_only'")
        return source

    @property
    def effective_target_mode(self) -> str:
        value = str(self.target_mode or TARGET_MODE_DIRECT_HIGH).strip().lower().replace("-", "_")
        aliases = {
            "direct": TARGET_MODE_DIRECT_HIGH,
            "direct_high": TARGET_MODE_DIRECT_HIGH,
            "high": TARGET_MODE_DIRECT_HIGH,
            "actual": TARGET_MODE_DIRECT_HIGH,
            "actual_high": TARGET_MODE_DIRECT_HIGH,
            "remaining": TARGET_MODE_REMAINING_WARMUP,
            "warmup": TARGET_MODE_REMAINING_WARMUP,
            "remaining_warmup": TARGET_MODE_REMAINING_WARMUP,
        }
        mode = aliases.get(value, value)
        if mode not in {TARGET_MODE_DIRECT_HIGH, TARGET_MODE_REMAINING_WARMUP}:
            raise ValueError("target_mode must be one of: 'actual_high' or 'remaining_warmup'")
        return mode

    @property
    def effective_hyperparameter_space(self) -> str:
        value = str(self.hyperparameter_space or "default").strip().lower().replace("-", "_")
        if value in {"", "normal", "standard"}:
            value = "default"
        if value in {"plus", "wideplus"}:
            value = "wide_plus"
        if value not in {"default", "wide", "wide_plus"}:
            raise ValueError("hyperparameter_space must be one of: 'default', 'wide', or 'wide_plus'")
        return value

    @property
    def effective_catboost_max_iterations(self) -> int | None:
        return None if self.catboost_max_iterations is None else max(50, int(self.catboost_max_iterations))

    @property
    def effective_catboost_max_depth(self) -> int | None:
        return None if self.catboost_max_depth is None else min(16, max(2, int(self.catboost_max_depth)))

    @property
    def effective_catboost_min_learning_rate(self) -> float | None:
        if self.catboost_min_learning_rate is None:
            return None
        return min(0.25, max(1e-5, float(self.catboost_min_learning_rate)))

    @property
    def effective_catboost_max_border_count(self) -> int | None:
        return None if self.catboost_max_border_count is None else min(255, max(16, int(self.catboost_max_border_count)))

    @property
    def effective_base_model_methods(self) -> tuple[str, ...]:
        methods: list[str] = []
        for raw_method in self.base_model_methods or ():
            method = str(raw_method).strip().lower()
            if not method:
                continue
            if method not in BASE_MODEL_METHODS:
                raise ValueError(f"base_model_methods must be drawn from: {', '.join(BASE_MODEL_METHODS)}")
            if method not in methods:
                methods.append(method)
        if not methods:
            raise ValueError("base_model_methods must include at least one supported model method")
        return tuple(methods)

    @property
    def effective_year_split_folds(self) -> tuple["YearSplitFold", ...]:
        if self.effective_training_profile == TRAINING_PROFILE_V20_ALIGNED:
            return V20_EXPANDING_FOLDS
        return tuple(self.year_split_folds) if self.year_split_folds is not None else YEAR_SPLIT_FOLDS

    @property
    def effective_year_split_validation_weights(self) -> dict[int, float] | None:
        if self.effective_training_profile == TRAINING_PROFILE_V20_ALIGNED:
            return {fold.validation_year: 1.0 for fold in V20_EXPANDING_FOLDS}
        return self.year_split_validation_weights

    @property
    def effective_year_split_test_train_years(self) -> tuple[int, int]:
        return (int(self.year_split_test_train_years[0]), int(self.year_split_test_train_years[1]))

    @property
    def effective_year_split_test_year(self) -> int:
        return int(self.year_split_test_year)

    @property
    def effective_optuna_metric(self) -> str:
        aliases = {
            "mae": "mae_f",
            "mean_absolute_error": "mae_f",
            "rmse": "rmse_f",
            "root_mean_squared_error": "rmse_f",
            "bucket": "bucket_log_loss",
            "bucket_logloss": "bucket_log_loss",
            "bucket_log_loss": "bucket_log_loss",
            "log_loss": "bucket_log_loss",
        }
        metric = aliases.get(str(self.optuna_metric).strip().lower(), str(self.optuna_metric).strip().lower())
        if metric not in {"mae_f", "rmse_f", "bucket_log_loss"}:
            raise ValueError("optuna_metric must be one of: 'mae_f', 'mae', 'rmse_f', 'rmse', or 'bucket_log_loss'")
        return metric

    @property
    def effective_feature_importance_repeats(self) -> int:
        if self.feature_importance_repeats is not None:
            return max(1, int(self.feature_importance_repeats))
        return 3 if self.fast_mode else 10

    @property
    def effective_max_feature_missing_fraction(self) -> float | None:
        if self.max_feature_missing_fraction is None:
            return None
        value = float(self.max_feature_missing_fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError("max_feature_missing_fraction must be between 0 and 1")
        return value


@dataclass
class StationStackingResult:
    station_id: str
    features: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_columns: pd.DataFrame
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class YearSplitFold:
    name: str
    train_start_year: int
    train_end_year: int
    validation_year: int


@dataclass
class YearSplitExperimentResult:
    station_id: str
    features: pd.DataFrame
    tuning_results: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    metrics: pd.DataFrame
    stack_tuning_results: pd.DataFrame
    scoreboard: pd.DataFrame
    bracket_predictions: pd.DataFrame
    bracket_metrics: pd.DataFrame
    feature_columns: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    feature_importance: pd.DataFrame
    output_paths: dict[str, Path]


YEAR_SPLIT_FOLDS = (
    YearSplitFold("fold_2021_2023_to_2024", 2021, 2023, 2024),
    YearSplitFold("fold_2022_2024_to_2025", 2022, 2024, 2025),
)
YEAR_SPLIT_EXPANDING_FOLDS = (
    YearSplitFold("fold_2021_2023_to_2024", 2021, 2023, 2024),
    YearSplitFold("fold_2021_2024_to_2025", 2021, 2024, 2025),
)
V20_EXPANDING_FOLDS = (
    YearSplitFold("fold_2021_to_2022", 2021, 2021, 2022),
    YearSplitFold("fold_2021_2022_to_2023", 2021, 2022, 2023),
    YearSplitFold("fold_2021_2023_to_2024", 2021, 2023, 2024),
    YearSplitFold("fold_2021_2024_to_2025", 2021, 2024, 2025),
)
V20_STACK_META_VALIDATION_YEARS = (2023, 2024, 2025)
YEAR_SPLIT_TEST_TRAIN_YEARS = (2021, 2025)
YEAR_SPLIT_TEST_YEAR = 2026


def missing_model_dependencies(methods: Iterable[str] | None = None) -> list[str]:
    if methods is None:
        packages = REQUIRED_MODEL_PACKAGES
    else:
        requested = {str(method).strip().lower() for method in methods}
        requested.add("optuna")
        packages = {
            package: module
            for package, module in REQUIRED_MODEL_PACKAGES.items()
            if package in requested
        }
    return sorted(package for package, module in packages.items() if find_spec(module) is None)


def require_model_dependencies(methods: Iterable[str] | None = None) -> None:
    missing = missing_model_dependencies(methods)
    if missing:
        missing_list = ", ".join(missing)
        raise ImportError(
            "Station stacking ML requires the configured gradient boosting packages. "
            f"Missing: {missing_list}. Install them with: python -m pip install -r requirements.txt"
        )


def missing_expected_model_methods(
    metrics: pd.DataFrame,
    base_model_methods: Iterable[str] | None = None,
    stack_enabled: bool = True,
) -> list[str]:
    expected = [*(base_model_methods or BASE_MODEL_METHODS)]
    if stack_enabled:
        expected.append(STACK_METHOD)
    if metrics.empty or "method" not in metrics:
        return expected
    methods = set(metrics["method"].dropna().astype(str))
    return [method for method in expected if method not in methods]


def provider_availability(
    project_root: str | Path = ".",
    timing_mode: str = "same_day_11am",
    providers: tuple[str, ...] = TARGET_PROVIDERS,
) -> pd.DataFrame:
    forecasts = load_same_day_provider_forecasts(project_root, timing_mode=timing_mode, providers=providers)
    if forecasts.empty:
        return pd.DataFrame(
            columns=["station_id", "provider", "row_count", "first_contract_date", "last_contract_date"]
        )
    grouped = forecasts.groupby(["station_id", "provider"], dropna=False)["contract_date"].agg(
        row_count="count",
        first_contract_date="min",
        last_contract_date="max",
    )
    return grouped.reset_index().sort_values(["station_id", "provider"]).reset_index(drop=True)


def load_current_observation_features(
    project_root: str | Path = ".",
    station_id: str | None = None,
    timing_mode: str = "same_day_11am",
) -> pd.DataFrame:
    root = Path(project_root)
    calibration_dir = root / "data" / "calibration"
    cache_paths = sorted(calibration_dir.glob(CURRENT_OBSERVATION_CACHE_PATTERN))
    frames: list[pd.DataFrame] = []
    required = ["station_id", "contract_date", "timing_mode", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS]
    for path in cache_paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        for column in required:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame[required].copy()
        frame["source_cache_dir"] = path.parent.name
        frame["source_cache_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["contract_date", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS])

    out = pd.concat(frames, ignore_index=True)
    out["station_id"] = out["station_id"].astype("string").str.upper()
    out["timing_mode"] = out["timing_mode"].astype("string")
    out["contract_date"] = out["contract_date"].astype("string").str[:10]
    out["observed_fetch_status"] = out["observed_fetch_status"].astype("string").str.lower()
    if station_id is not None:
        out = out.loc[out["station_id"].eq(station_id.upper())].copy()
    allowed_timing_modes = _current_observation_timing_modes(timing_mode)
    out = out.loc[out["timing_mode"].isin(allowed_timing_modes)].copy()
    if out.empty:
        return pd.DataFrame(columns=["contract_date", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS])
    for column in OBSERVED_NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["_timing_match_rank"] = out["timing_mode"].ne(timing_mode).astype(int)
    out["_observation_quality_rank"] = _current_observation_quality_rank(out, timing_mode)
    out = out.sort_values(
        [
            "station_id",
            "contract_date",
            "_timing_match_rank",
            "_observation_quality_rank",
            "source_cache_mtime",
            "source_cache_dir",
        ],
        ascending=[True, True, True, True, False, True],
    )
    out = out.drop_duplicates(["station_id", "contract_date"], keep="first")
    keep = ["contract_date", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS]
    return out[keep].sort_values("contract_date").reset_index(drop=True)


def _current_observation_timing_modes(timing_mode: str) -> tuple[str, ...]:
    mode = str(timing_mode)
    if mode == TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE:
        return (TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE, TIMING_MODE_SAME_DAY_11AM)
    return (mode,)


def _current_observation_quality_rank(frame: pd.DataFrame, timing_mode: str = TIMING_MODE_SAME_DAY_11AM) -> pd.Series:
    status = frame["observed_fetch_status"].astype("string").str.strip().str.lower()
    temp = pd.to_numeric(frame.get("observed_temp_at_as_of_f"), errors="coerce")
    high = pd.to_numeric(frame.get("observed_high_temp_through_as_of_f"), errors="coerce")
    age = pd.to_numeric(frame.get("observed_as_of_age_minutes"), errors="coerce")
    as_of_text = frame.get("observed_as_of_time_local", pd.Series(pd.NA, index=frame.index)).astype("string")
    clock = as_of_text.str.extract(r"T(?P<hour>\d{2}):(?P<minute>\d{2})")
    local_minutes = pd.to_numeric(clock["hour"], errors="coerce") * 60 + pd.to_numeric(clock["minute"], errors="coerce")
    window_hour = 9 if timing_mode == TIMING_MODE_SAME_DAY_9AM_LIVE_SAFE else 11
    in_window = local_minutes.between(window_hour * 60 - 10, window_hour * 60 + 10)

    return (
        status.ne("ok").fillna(True).astype(int) * 100
        + temp.isna().astype(int) * 20
        + high.isna().astype(int) * 10
        + age.gt(20).fillna(False).astype(int) * 5
        + (~in_window.fillna(False)).astype(int) * 5
    )


def load_same_day_provider_forecasts(
    project_root: str | Path = ".",
    timing_mode: str = "same_day_11am",
    providers: tuple[str, ...] = TARGET_PROVIDERS,
) -> pd.DataFrame:
    root = Path(project_root)
    frames: list[pd.DataFrame] = []
    calibration_dir = root / "data" / "calibration"
    cache_paths = sorted(
        {
            path
            for _, pattern in FORECAST_CACHE_PATTERNS
            for path in calibration_dir.glob(pattern)
            if _include_forecast_cache_path(path)
        }
    )
    for path in cache_paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        for column in FORECAST_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame[FORECAST_COLUMNS].copy()
        frame["source_cache_dir"] = path.parent.name
        frame["source_cache_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*FORECAST_COLUMNS, "source_cache_dir", "source_cache_mtime"])

    out = pd.concat(frames, ignore_index=True)
    out["station_id"] = out["station_id"].astype("string").str.upper()
    out["provider"] = out["provider"].astype("string").str.lower()
    out["timing_mode"] = out["timing_mode"].astype("string")
    out["contract_date"] = out["contract_date"].astype("string").str[:10]
    out["fetch_status"] = out["fetch_status"].astype("string").str.lower().fillna("ok")
    out = out.loc[
        out["provider"].isin(providers)
        & out["timing_mode"].eq(timing_mode)
        & out["fetch_status"].eq("ok")
    ].copy()
    if out.empty:
        return out
    for column in PROVIDER_FORECAST_NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["station_id", "provider", "contract_date", "raw_forecast_high_f"])
    out = out.loc[plausible_temperature_mask(out["raw_forecast_high_f"])].copy()
    if out.empty:
        return out
    out["_source_quality_rank"] = out["source_cache_dir"].map(_source_quality_rank)
    out = out.sort_values(
        [
            "station_id",
            "provider",
            "contract_date",
            "_source_quality_rank",
            "source_cache_mtime",
            "source_cache_dir",
        ],
        ascending=[True, True, True, True, False, True],
    )
    out = out.drop_duplicates(["station_id", "provider", "contract_date"], keep="first")
    out = out.drop(columns=["_source_quality_rank"]).reset_index(drop=True)
    return out


def build_station_wide_dataset(
    project_root: str | Path = ".",
    station_id: str = "KATL",
    timing_mode: str = "same_day_11am",
    providers: tuple[str, ...] = TARGET_PROVIDERS,
    feature_version: str = "base",
    target_source: str = TARGET_SOURCE_IEM_HOURLY,
    climatology_normals_path: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(project_root)
    station_id = station_id.upper()
    version = _normalize_feature_version(feature_version)
    if version == V20_KDAL_FIX_FEATURE_VERSION and station_id != "KDAL":
        raise ValueError(f"{V20_KDAL_FIX_FEATURE_VERSION} is limited to KDAL")
    actuals = _load_station_actuals(root, station_id, target_source=target_source)
    if version in V20_PEAK_TIMING_FEATURE_VERSIONS:
        actuals = _expand_v20_actual_date_spine(root, station_id, actuals, target_source=target_source)
    current_observations = load_current_observation_features(root, station_id, timing_mode=timing_mode)
    station_meta = _load_station_meta(root, station_id)
    forecasts = load_same_day_provider_forecasts(root, timing_mode=timing_mode, providers=providers)
    forecasts = forecasts.loc[forecasts["station_id"].eq(station_id)].copy()

    wide = actuals.copy()
    if not current_observations.empty:
        wide = wide.merge(current_observations, on="contract_date", how="left")
    provider_numeric_columns = _provider_numeric_columns_for_feature_version(version)
    for provider in providers:
        provider_frame = forecasts.loc[forecasts["provider"].eq(provider)].copy()
        provider_wide = _provider_wide(provider_frame, provider, numeric_columns=provider_numeric_columns)
        wide = wide.merge(provider_wide, on="contract_date", how="left")

    for key, value in station_meta.items():
        wide[key] = value

    if version in {*V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS}:
        wide = _merge_v18_nbm_rap_features(root, station_id, wide)
    if version in V20_PEAK_TIMING_FEATURE_VERSIONS:
        wide = _merge_v20_peak_timing_features(root, station_id, wide)

    wide = wide.sort_values("contract_date").reset_index(drop=True)
    wide = _add_calendar_features(wide)
    wide = _add_current_observation_derived_features(wide)
    wide = _add_provider_availability_features(wide, providers)
    wide = _add_provider_time_features(wide, providers, str(station_meta.get("timezone", "UTC")))
    wide = _add_ensemble_features(wide, providers)
    wide = _add_forecast_shape_features(wide, providers)
    wide = _add_provider_cross_model_features(wide, providers)
    wide = _add_lagged_actual_features(wide)
    wide = _add_lagged_provider_error_features(wide, providers)
    wide = _add_prior_month_provider_error_features(wide, providers)
    wide = _add_forecast_history_delta_features(wide, providers)
    wide = _add_observation_history_delta_features(wide)
    wide = _add_observation_forecast_delta_features(wide, providers)
    wide = add_versioned_feature_engineering(wide, feature_version=version, providers=providers)
    if version in CLIMATOLOGY_FEATURE_VERSIONS:
        wide = add_v9_climatology_features(
            wide,
            project_root=root,
            station_id=station_id,
            climatology_normals_path=climatology_normals_path,
        )
    wide = add_strict_quality_flags(wide, providers=providers)
    return wide


def raw_baseline_predictions(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    frame = _with_actual_quality_columns(frame, config)
    base = add_strict_quality_flags(frame, providers=config.providers)
    base = base.loc[base[STRICT_QUALITY_OK_COLUMN].fillna(False)].dropna(subset=[TARGET]).copy()
    for provider in config.providers:
        column = HIGH_COLUMNS[provider]
        if column not in base:
            continue
        pred = base.loc[base[column].notna(), ["contract_date", TARGET, column]].copy()
        if pred.empty:
            continue
        pred["method"] = f"{provider}_raw"
        pred["predicted_high_f"] = pred[column]
        pred["evaluation_scope"] = "provider_available_dates"
        rows.append(_prediction_columns(pred))

    for method, column in [("provider_mean", "provider_mean_high_f"), ("provider_median", "provider_median_high_f")]:
        if column not in base:
            continue
        pred = base.loc[base[column].notna(), ["contract_date", TARGET, column]].copy()
        if pred.empty:
            continue
        pred["method"] = method
        pred["predicted_high_f"] = pred[column]
        pred["evaluation_scope"] = "provider_available_dates"
        rows.append(_prediction_columns(pred))

    best = _walk_forward_best_raw_provider(base, config)
    if not best.empty:
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def walk_forward_base_model_predictions(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    modeling_frame, categorical, numeric = _modeling_frame(frame, config)
    if modeling_frame.empty or len(modeling_frame) <= config.effective_min_train_rows:
        return _empty_predictions()
    models = _build_base_model_pipelines(config, categorical, numeric)
    feature_cols = categorical + numeric
    rows: list[pd.DataFrame] = []
    dates = sorted(modeling_frame["contract_date"].unique())
    completed_blocks = 0
    for start_idx in range(0, len(dates), config.effective_refit_days):
        block_start = dates[start_idx]
        block_dates = dates[start_idx : start_idx + config.effective_refit_days]
        train = modeling_frame.loc[modeling_frame["contract_date"] < block_start].copy()
        valid = modeling_frame.loc[modeling_frame["contract_date"].isin(block_dates)].copy()
        if len(train) < config.effective_min_train_rows or valid.empty:
            continue
        for method, estimator in models.items():
            estimator.fit(train[feature_cols], _model_target_values(train, config))
            pred = valid[["contract_date", TARGET]].copy()
            pred["method"] = method
            pred["predicted_high_f"] = _prediction_output_to_high(estimator.predict(valid[feature_cols]), valid, config)
            pred["evaluation_scope"] = "walk_forward_model"
            rows.append(_prediction_columns(pred))
        completed_blocks += 1
        if config.fast_mode and completed_blocks >= config.fast_max_validation_blocks:
            break
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def walk_forward_stack_predictions(
    base_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    config: StationStackingConfig,
) -> pd.DataFrame:
    if not config.stack_enabled:
        return _empty_predictions()
    if base_predictions.empty or baseline_predictions.empty:
        return _empty_predictions()
    stack_source = _stack_source_frame(base_predictions, baseline_predictions, config.effective_base_model_methods)
    if stack_source.empty or len(stack_source) <= config.effective_min_meta_train_rows:
        return _empty_predictions()

    from sklearn.linear_model import RidgeCV

    baseline_methods = [f"{provider}_raw" for provider in config.providers]
    baseline_methods.extend(method for method in BASELINE_METHODS if not method.endswith("_raw"))
    stack_features = [f"{method}_predicted_high_f" for method in [*config.effective_base_model_methods, *baseline_methods]]
    if any(column not in stack_source for column in stack_features):
        return _empty_predictions()
    stack_source = stack_source.dropna(subset=stack_features + [TARGET]).sort_values("contract_date").reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    dates = sorted(stack_source["contract_date"].unique())
    completed_blocks = 0
    for start_idx in range(0, len(dates), config.effective_refit_days):
        block_start = dates[start_idx]
        block_dates = dates[start_idx : start_idx + config.effective_refit_days]
        train = stack_source.loc[stack_source["contract_date"] < block_start].copy()
        valid = stack_source.loc[stack_source["contract_date"].isin(block_dates)].copy()
        if len(train) < config.effective_min_meta_train_rows or valid.empty:
            continue
        model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))
        model.fit(train[stack_features], train[TARGET])
        pred = valid[["contract_date", TARGET]].copy()
        pred["method"] = STACK_METHOD
        pred["predicted_high_f"] = model.predict(valid[stack_features])
        pred["evaluation_scope"] = "walk_forward_stack"
        rows.append(_prediction_columns(pred))
        completed_blocks += 1
        if config.fast_mode and completed_blocks >= config.fast_max_validation_blocks:
            break
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "evaluation_scope",
                "method",
                "count",
                "mae_f",
                "rmse_f",
                "bias_f",
                "within_1f_pct",
                "within_2f_pct",
                "within_3f_pct",
                "first_contract_date",
                "last_contract_date",
            ]
        )
    metrics = (
        predictions.groupby(["evaluation_scope", "method"], dropna=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    common = _common_date_metrics(predictions)
    if not common.empty:
        metrics = pd.concat([metrics, common], ignore_index=True)
    return metrics.sort_values(["evaluation_scope", "mae_f", "method"]).reset_index(drop=True)


def run_station_stacking_experiment(config: StationStackingConfig) -> StationStackingResult:
    features = build_station_wide_dataset(
        config.resolved_project_root(),
        station_id=config.station_id,
        timing_mode=config.timing_mode,
        providers=config.providers,
        feature_version=config.effective_feature_version,
        target_source=config.effective_target_source,
        climatology_normals_path=config.climatology_normals_path,
    )
    baseline_predictions = raw_baseline_predictions(features, config)
    model_predictions = walk_forward_base_model_predictions(features, config)
    stack_predictions = walk_forward_stack_predictions(model_predictions, baseline_predictions, config)
    predictions = pd.concat(
        [frame for frame in [baseline_predictions, model_predictions, stack_predictions] if not frame.empty],
        ignore_index=True,
    ) if not baseline_predictions.empty or not model_predictions.empty or not stack_predictions.empty else _empty_predictions()
    metrics = summarize_predictions(predictions)
    categorical, numeric = feature_columns(features, config)
    feature_columns_frame = pd.DataFrame(
        [{"feature": feature, "kind": "categorical"} for feature in categorical]
        + [{"feature": feature, "kind": "numeric"} for feature in numeric]
    )

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    station = config.station_id.upper()
    paths = {
        "features": output_dir / f"{station}_features.csv",
        "predictions": output_dir / f"{station}_predictions.csv",
        "metrics": output_dir / f"{station}_metrics.csv",
        "feature_columns": output_dir / f"{station}_feature_columns.csv",
    }
    features.to_csv(paths["features"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    feature_columns_frame.to_csv(paths["feature_columns"], index=False)
    return StationStackingResult(
        station_id=station,
        features=features,
        predictions=predictions,
        metrics=metrics,
        feature_columns=feature_columns_frame,
        output_paths=paths,
    )


def run_station_year_split_experiment(config: StationStackingConfig) -> YearSplitExperimentResult:
    folds = config.effective_year_split_folds
    test_train_years = config.effective_year_split_test_train_years
    test_year = config.effective_year_split_test_year
    features = build_station_wide_dataset(
        config.resolved_project_root(),
        station_id=config.station_id,
        timing_mode=config.timing_mode,
        providers=config.providers,
        feature_version=config.effective_feature_version,
        target_source=config.effective_target_source,
        climatology_normals_path=config.climatology_normals_path,
    )
    modeling_frame, categorical, numeric = _modeling_frame(features, config)
    feature_columns_frame = pd.DataFrame(
        [{"feature": feature, "kind": "categorical"} for feature in categorical]
        + [{"feature": feature, "kind": "numeric"} for feature in numeric]
    )

    baseline_validation = year_split_baseline_predictions(modeling_frame, config, folds)
    tuning_results, validation_predictions, selected = tune_year_split_base_models(
        modeling_frame,
        config,
        categorical,
        numeric,
        folds,
    )
    test_predictions = year_split_test_predictions(
        modeling_frame,
        config,
        categorical,
        numeric,
        selected,
        train_years=test_train_years,
        test_year=test_year,
    )
    test_stack_predictions, stack_tuning_results = tune_year_split_stack_model(
        validation_predictions=pd.concat(
            [frame for frame in [baseline_validation, validation_predictions] if not frame.empty],
            ignore_index=True,
        )
        if not baseline_validation.empty or not validation_predictions.empty
        else _empty_year_split_predictions(),
        test_predictions=test_predictions,
        config=config,
        test_year=test_year,
    )
    if not test_stack_predictions.empty:
        test_predictions = pd.concat([test_predictions, test_stack_predictions], ignore_index=True)
    if config.effective_feature_version in {"v12", *V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS}:
        guarded_predictions = guarded_blend_predictions(test_predictions)
        if not guarded_predictions.empty:
            test_predictions = pd.concat([test_predictions, guarded_predictions], ignore_index=True)
    feature_importance = year_split_feature_importance(
        modeling_frame,
        config,
        categorical,
        numeric,
        selected,
        train_years=test_train_years,
        test_year=test_year,
    )
    validation_predictions = pd.concat(
        [frame for frame in [baseline_validation, validation_predictions] if not frame.empty],
        ignore_index=True,
    ) if not baseline_validation.empty or not validation_predictions.empty else _empty_year_split_predictions()
    metrics = summarize_year_split_predictions(validation_predictions, test_predictions)
    scoreboard = year_split_scoreboard(validation_predictions, test_predictions)
    bracket_predictions = year_split_bracket_predictions(test_predictions, test_year=test_year)
    bracket_metrics = year_split_bracket_metrics(bracket_predictions)

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    station = config.station_id.upper()
    paths = {
        "features": output_dir / f"{station}_features.csv",
        "year_split_tuning": output_dir / f"{station}_year_split_tuning.csv",
        "year_split_validation_predictions": output_dir / f"{station}_year_split_validation_predictions.csv",
        "year_split_test_predictions": output_dir / f"{station}_year_split_test_predictions.csv",
        "year_split_metrics": output_dir / f"{station}_year_split_metrics.csv",
        "year_split_selected_hyperparameters": output_dir / f"{station}_year_split_selected_hyperparameters.csv",
        "year_split_feature_importance": output_dir / f"{station}_year_split_feature_importance.csv",
        "year_split_stack_tuning": output_dir / f"{station}_year_split_stack_tuning.csv",
        "year_split_scoreboard": output_dir / f"{station}_year_split_scoreboard.csv",
        "year_split_bracket_predictions": output_dir / f"{station}_year_split_bracket_predictions.csv",
        "year_split_bracket_metrics": output_dir / f"{station}_year_split_bracket_metrics.csv",
        "feature_columns": output_dir / f"{station}_feature_columns.csv",
    }
    features.to_csv(paths["features"], index=False)
    tuning_results.to_csv(paths["year_split_tuning"], index=False)
    validation_predictions.to_csv(paths["year_split_validation_predictions"], index=False)
    test_predictions.to_csv(paths["year_split_test_predictions"], index=False)
    metrics.to_csv(paths["year_split_metrics"], index=False)
    selected.to_csv(paths["year_split_selected_hyperparameters"], index=False)
    feature_importance.to_csv(paths["year_split_feature_importance"], index=False)
    stack_tuning_results.to_csv(paths["year_split_stack_tuning"], index=False)
    scoreboard.to_csv(paths["year_split_scoreboard"], index=False)
    bracket_predictions.to_csv(paths["year_split_bracket_predictions"], index=False)
    bracket_metrics.to_csv(paths["year_split_bracket_metrics"], index=False)
    feature_columns_frame.to_csv(paths["feature_columns"], index=False)
    return YearSplitExperimentResult(
        station_id=station,
        features=features,
        tuning_results=tuning_results,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        metrics=metrics,
        stack_tuning_results=stack_tuning_results,
        scoreboard=scoreboard,
        bracket_predictions=bracket_predictions,
        bracket_metrics=bracket_metrics,
        feature_columns=feature_columns_frame,
        selected_hyperparameters=selected,
        feature_importance=feature_importance,
        output_paths=paths,
    )


def year_split_baseline_predictions(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    folds: tuple[YearSplitFold, ...] = YEAR_SPLIT_FOLDS,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if frame.empty:
        return _empty_year_split_predictions()
    frame = add_strict_quality_flags(_with_actual_quality_columns(frame, config), providers=config.providers)
    frame = frame.loc[frame[STRICT_QUALITY_OK_COLUMN].fillna(False)].copy()
    if frame.empty:
        return _empty_year_split_predictions()
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    for fold in folds:
        train = frame.loc[year.between(fold.train_start_year, fold.train_end_year)].copy()
        valid = frame.loc[year.eq(fold.validation_year)].copy()
        if train.empty or valid.empty:
            continue
        for provider in config.providers:
            column = HIGH_COLUMNS[provider]
            if column in valid:
                pred = valid.loc[valid[column].notna(), ["contract_date", TARGET, column]].copy()
                if pred.empty:
                    continue
                pred["method"] = f"{provider}_raw"
                pred["predicted_high_f"] = pred[column]
                pred["evaluation_scope"] = "year_split_validation"
                pred["fold"] = fold.name
                rows.append(_year_split_prediction_columns(pred))
        for method, column in [("provider_mean", "provider_mean_high_f"), ("provider_median", "provider_median_high_f")]:
            if column not in valid:
                continue
            pred = valid.loc[valid[column].notna(), ["contract_date", TARGET, column]].copy()
            if pred.empty:
                continue
            pred["method"] = method
            pred["predicted_high_f"] = pred[column]
            pred["evaluation_scope"] = "year_split_validation"
            pred["fold"] = fold.name
            rows.append(_year_split_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_year_split_predictions()


def tune_year_split_base_models(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    folds: tuple[YearSplitFold, ...] = YEAR_SPLIT_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), _empty_year_split_predictions(), pd.DataFrame()
    frame = _ensure_model_target_columns(
        add_strict_quality_flags(_with_actual_quality_columns(frame, config), providers=config.providers),
        config,
    )
    frame = frame.loc[frame[STRICT_QUALITY_OK_COLUMN].fillna(False)].copy()
    frame = _drop_missing_model_target(frame, config)
    if frame.empty:
        return pd.DataFrame(), _empty_year_split_predictions(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    metric_col = config.effective_optuna_metric
    for method in config.effective_base_model_methods:
        study = _create_optuna_study(config, method)
        method_rows: list[dict[str, Any]] = []

        def objective(trial) -> float:
            params = _suggest_hyperparameters(method, trial, config)
            param_key = f"trial_{trial.number}"
            fold_scores: list[tuple[YearSplitFold, float]] = []
            trial_rows: list[dict[str, Any]] = []
            fit_records: list[dict[str, Any]] = []
            _set_trial_checkpoint_attrs(
                trial,
                method=method,
                param_key=param_key,
                params=params,
                rows=trial_rows,
                fit_metadata=fit_records,
                status="running",
                error="",
            )
            for fold in folds:
                train = frame.loc[year.between(fold.train_start_year, fold.train_end_year)].copy()
                valid = frame.loc[year.eq(fold.validation_year)].copy()
                if train.empty or valid.empty:
                    continue
                try:
                    predicted, fit_metadata = _fit_predict_base_model(
                        config=config,
                        categorical=categorical,
                        numeric=numeric,
                        method=method,
                        params=params,
                        train=train,
                        valid=valid,
                        early_stopping=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "method": method,
                        "trial_number": trial.number,
                        "param_key": param_key,
                        "fold": fold.name,
                        "fold_weight": _year_split_fold_weight(fold, config),
                        "mae_f": pd.NA,
                        "rmse_f": pd.NA,
                        "count": 0,
                        "status": "failed",
                        "error": str(exc),
                        **{f"param_{key}": value for key, value in params.items()},
                    }
                    trial_rows.append(row)
                    method_rows.append(row)
                    _set_trial_checkpoint_attrs(
                        trial,
                        method=method,
                        param_key=param_key,
                        params=params,
                        rows=trial_rows,
                        fit_metadata=fit_records,
                        status="failed",
                        error=str(exc),
                    )
                    raise
                pred = valid[["contract_date", TARGET]].copy()
                pred["method"] = method
                pred["param_key"] = param_key
                pred["predicted_high_f"] = predicted
                pred["evaluation_scope"] = "year_split_validation"
                pred["fold"] = fold.name
                metrics = _metric_row(_prediction_columns(pred))
                mae = float(metrics["mae_f"])
                rmse = float(metrics["rmse_f"])
                bucket_log_loss = float(metrics["bucket_log_loss"])
                fold_scores.append((fold, float(metrics[metric_col])))
                row = {
                    "method": method,
                    "trial_number": trial.number,
                    "param_key": param_key,
                    "fold": fold.name,
                    "fold_weight": _year_split_fold_weight(fold, config),
                    "mae_f": mae,
                    "rmse_f": rmse,
                    "bucket_log_loss": bucket_log_loss,
                    "count": int(metrics["count"]),
                    "fit_numeric_features": fit_metadata["numeric_features"],
                    "fit_categorical_features": fit_metadata["categorical_features"],
                    "best_iteration": fit_metadata["best_iteration"],
                    "status": "ok",
                    "error": "",
                    **{f"param_{key}": value for key, value in params.items()},
                }
                trial_rows.append(row)
                method_rows.append(row)
                fit_records.append({"fold": fold.name, **fit_metadata})
                current_score = _weighted_fold_score(fold_scores, config)
                _set_trial_checkpoint_attrs(
                    trial,
                    method=method,
                    param_key=param_key,
                    params=params,
                    rows=trial_rows,
                    fit_metadata=fit_records,
                    status="running",
                    error="",
                    objective_value=current_score,
                )
                if hasattr(trial, "report"):
                    trial.report(current_score, step=len(fold_scores))
                if hasattr(trial, "should_prune") and trial.should_prune():
                    _set_trial_checkpoint_attrs(
                        trial,
                        method=method,
                        param_key=param_key,
                        params=params,
                        rows=trial_rows,
                        fit_metadata=fit_records,
                        status="pruned",
                        error="",
                        objective_value=current_score,
                    )
                    raise _trial_pruned_exception()
            if not fold_scores:
                _set_trial_checkpoint_attrs(
                    trial,
                    method=method,
                    param_key=param_key,
                    params=params,
                    rows=trial_rows,
                    fit_metadata=fit_records,
                    status="failed",
                    error="No usable validation folds.",
                )
                return float("inf")
            final_score = _weighted_fold_score(fold_scores, config)
            _set_trial_checkpoint_attrs(
                trial,
                method=method,
                param_key=param_key,
                params=params,
                rows=trial_rows,
                fit_metadata=fit_records,
                status="ok",
                error="",
                objective_value=final_score,
            )
            return final_score

        remaining_trials = _remaining_optuna_trials(study, config.effective_optuna_trials)
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False, catch=(Exception,))
        stored_rows = _study_tuning_rows(study)
        rows.extend(stored_rows if stored_rows else method_rows)
    tuning = pd.DataFrame(rows)
    selected = _selected_hyperparameters(tuning, metric_col=metric_col)
    validation_predictions = _validation_predictions_for_selected_params(frame, config, categorical, numeric, folds, selected)
    return tuning, validation_predictions, selected


def year_split_test_predictions(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    selected_hyperparameters: pd.DataFrame,
    train_years: tuple[int, int] = YEAR_SPLIT_TEST_TRAIN_YEARS,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_year_split_predictions()
    frame = _ensure_model_target_columns(
        add_strict_quality_flags(_with_actual_quality_columns(frame, config), providers=config.providers),
        config,
    )
    frame = frame.loc[frame[STRICT_QUALITY_OK_COLUMN].fillna(False)].copy()
    frame = _drop_missing_model_target(frame, config)
    if frame.empty:
        return _empty_year_split_predictions()
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    train = frame.loc[year.between(train_years[0], train_years[1])].copy()
    test = frame.loc[year.eq(test_year)].copy()
    rows: list[pd.DataFrame] = []
    if train.empty or test.empty:
        return _empty_year_split_predictions()
    for provider in config.providers:
        column = HIGH_COLUMNS[provider]
        if column in test:
            pred = test.loc[test[column].notna(), ["contract_date", TARGET, column]].copy()
            if not pred.empty:
                pred["method"] = f"{provider}_raw"
                pred["predicted_high_f"] = pred[column]
                pred["evaluation_scope"] = "year_split_test"
                pred["fold"] = f"train_{train_years[0]}_{train_years[1]}_test_{test_year}"
                rows.append(_year_split_prediction_columns(pred))
    for method, column in [("provider_mean", "provider_mean_high_f"), ("provider_median", "provider_median_high_f")]:
        if column not in test:
            continue
        pred = test.loc[test[column].notna(), ["contract_date", TARGET, column]].copy()
        if pred.empty:
            continue
        pred["method"] = method
        pred["predicted_high_f"] = pred[column]
        pred["evaluation_scope"] = "year_split_test"
        pred["fold"] = f"train_{train_years[0]}_{train_years[1]}_test_{test_year}"
        rows.append(_year_split_prediction_columns(pred))
    selected = selected_hyperparameters.copy()
    for _, row in selected.iterrows():
        method = str(row["method"])
        params = _params_from_selected_row(row)
        try:
            predicted, _ = _fit_predict_base_model(
                config=config,
                categorical=categorical,
                numeric=numeric,
                method=method,
                params=params,
                train=train,
                valid=test,
                early_stopping=False,
            )
        except Exception:
            continue
        pred = test[["contract_date", TARGET]].copy()
        pred["method"] = method
        pred["param_key"] = str(row["param_key"])
        pred["predicted_high_f"] = predicted
        pred["evaluation_scope"] = "year_split_test"
        pred["fold"] = f"train_{train_years[0]}_{train_years[1]}_test_{test_year}"
        rows.append(_year_split_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_year_split_predictions()


def tune_year_split_stack_model(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    config: StationStackingConfig,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuning_columns = [
        "method",
        "trial_number",
        "param_key",
        "feature_set",
        "alpha",
        "fit_intercept",
        "fold",
        "mae_f",
        "rmse_f",
        "bucket_log_loss",
        "count",
        "status",
        "error",
    ]
    if not config.stack_enabled:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)
    if validation_predictions.empty or test_predictions.empty:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)
    stack_methods = [*config.effective_base_model_methods, *(f"{provider}_raw" for provider in config.providers)]
    train_source = _year_split_stack_source_frame(validation_predictions, stack_methods)
    test_source = _year_split_stack_source_frame(test_predictions, stack_methods)
    if train_source.empty or test_source.empty:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)
    if len(train_source) < config.effective_min_meta_train_rows:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)

    from sklearn.linear_model import Ridge

    meta_train, meta_valid = _stack_meta_train_valid_split(train_source)
    if _uses_expanding_stack_validation(config):
        meta_splits = _v20_stack_meta_splits(train_source)
    else:
        meta_year = int(pd.to_datetime(meta_valid["contract_date"], errors="coerce").dt.year.max()) if not meta_valid.empty else -1
        meta_splits = [(meta_year, meta_train, meta_valid)]
    rows: list[dict[str, Any]] = []
    if not meta_splits:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)
    study = _create_stack_optuna_study(config)
    metric_col = config.effective_optuna_metric

    def objective(trial) -> float:
        params = _suggest_stack_hyperparameters(trial, config)
        feature_set = str(params["feature_set"])
        alpha = float(params["alpha"])
        fit_intercept = bool(params["fit_intercept"])
        stack_features = _stack_features_for_set(feature_set, config.effective_base_model_methods, config.providers)
        param_key = f"stack_trial_{trial.number}"
        fit_records: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        _set_trial_checkpoint_attrs(
            trial,
            method=STACK_METHOD,
            param_key=param_key,
            params=params,
            rows=trial_rows,
            fit_metadata=fit_records,
            status="running",
            error="",
        )
        objective_scores: list[float] = []
        for validation_year, raw_train, raw_valid in meta_splits:
            train = raw_train.dropna(subset=[*stack_features, TARGET]).copy()
            valid = raw_valid.dropna(subset=[*stack_features, TARGET]).copy()
            fit_records.append(
                {
                    "validation_year": validation_year,
                    "stack_features": stack_features,
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                }
            )
            if train.empty or valid.empty:
                return float("inf")
            model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
            model.fit(train[stack_features], train[TARGET])
            predicted = model.predict(valid[stack_features])
            pred = valid[["contract_date", TARGET]].copy()
            pred["method"] = STACK_METHOD
            pred["predicted_high_f"] = predicted
            pred["evaluation_scope"] = "year_split_stack_validation"
            metrics = _metric_row(_prediction_columns(pred))
            row = {
                "method": STACK_METHOD,
                "trial_number": trial.number,
                "param_key": param_key,
                "feature_set": feature_set,
                "alpha": alpha,
                "fit_intercept": fit_intercept,
                "fold": f"meta_to_{validation_year}",
                "mae_f": float(metrics["mae_f"]),
                "rmse_f": float(metrics["rmse_f"]),
                "bucket_log_loss": float(metrics["bucket_log_loss"]),
                "count": int(metrics["count"]),
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            trial_rows.append(row)
            objective_scores.append(float(metrics[metric_col]))
        objective_value = float(np.mean(objective_scores))
        _set_trial_checkpoint_attrs(
            trial,
            method=STACK_METHOD,
            param_key=param_key,
            params=params,
            rows=trial_rows,
            fit_metadata=fit_records,
            status="ok",
            error="",
            objective_value=objective_value,
        )
        return objective_value

    remaining_trials = _remaining_optuna_trials(study, config.effective_stack_optuna_trials)
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False, catch=(Exception,))
    stored_rows = _study_tuning_rows(study)
    if stored_rows:
        rows = stored_rows
    tuning = pd.DataFrame(rows, columns=tuning_columns)
    ok = tuning.loc[tuning["status"].eq("ok")].copy()
    if ok.empty:
        return _empty_year_split_predictions(), tuning
    selected, _, _ = _select_stack_tuning_candidate(
        tuning,
        metric_col,
        aggregate_folds=_uses_expanding_stack_validation(config),
    )
    stack_features = _stack_features_for_set(
        str(selected["feature_set"]), config.effective_base_model_methods, config.providers
    )
    train = train_source.dropna(subset=[*stack_features, TARGET]).copy()
    test = test_source.dropna(subset=[*stack_features, TARGET]).copy()
    if len(train) < config.effective_min_meta_train_rows or test.empty:
        return _empty_year_split_predictions(), tuning
    try:
        model = Ridge(alpha=float(selected["alpha"]), fit_intercept=bool(selected["fit_intercept"]))
        model.fit(train[stack_features], train[TARGET])
        predicted = model.predict(test[stack_features])
    except Exception:
        return _empty_year_split_predictions(), tuning
    pred = test[["contract_date", TARGET]].copy()
    pred["method"] = STACK_METHOD
    pred["param_key"] = str(selected["param_key"])
    pred["predicted_high_f"] = predicted
    pred["evaluation_scope"] = "year_split_test"
    meta_years = pd.to_datetime(train["contract_date"], errors="coerce").dt.year
    pred["fold"] = f"ridge_meta_train_{int(meta_years.min())}_{int(meta_years.max())}_test_{test_year}"
    return _year_split_prediction_columns(pred), tuning


def year_split_stack_test_predictions(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    config: StationStackingConfig,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    predictions, _ = tune_year_split_stack_model(validation_predictions, test_predictions, config, test_year)
    return predictions


def kdal_oof_residual_calibrated_stack_predictions(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    stack_tuning: pd.DataFrame,
    config: StationStackingConfig,
    *,
    min_month_rows: int = 60,
    shrinkage_rows: int = 60,
    correction_cap_f: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate KDAL stack predictions using only prior-year OOF stack residuals."""
    if config.station_id.upper() != "KDAL" or config.effective_feature_version != V20_KDAL_FIX_FEATURE_VERSION:
        raise ValueError("KDAL OOF calibration requires station KDAL and v20_kdal_nbm_physics")
    if validation_predictions.empty or test_predictions.empty or stack_tuning.empty:
        return _empty_year_split_predictions(), pd.DataFrame(), _empty_year_split_predictions()

    ok = stack_tuning.loc[stack_tuning["status"].eq("ok")].copy()
    if ok.empty:
        return _empty_year_split_predictions(), pd.DataFrame(), _empty_year_split_predictions()
    metric = config.effective_optuna_metric
    selected_scores = ok.groupby("param_key", as_index=False).agg(
        mean_metric=(metric, "mean"),
        worst_metric=(metric, "max"),
    )
    selected_key = selected_scores.sort_values(["mean_metric", "worst_metric", "param_key"]).iloc[0]["param_key"]
    selected = ok.loc[ok["param_key"].eq(selected_key)].iloc[0]
    stack_features = _stack_features_for_set(
        str(selected["feature_set"]), config.effective_base_model_methods, config.providers
    )

    selected_methods = [column.removesuffix("_predicted_high_f") for column in stack_features]
    source = _year_split_stack_source_frame(validation_predictions, selected_methods)
    if source.empty:
        return _empty_year_split_predictions(), pd.DataFrame(), _empty_year_split_predictions()
    from sklearn.linear_model import Ridge

    years = pd.to_datetime(source["contract_date"], errors="coerce").dt.year
    oof_rows: list[pd.DataFrame] = []
    for validation_year in sorted(int(year) for year in years.dropna().unique())[1:]:
        train = source.loc[years.lt(validation_year)].dropna(subset=[*stack_features, TARGET]).copy()
        valid = source.loc[years.eq(validation_year)].dropna(subset=[*stack_features, TARGET]).copy()
        if len(train) < config.effective_min_meta_train_rows or valid.empty:
            continue
        model = Ridge(alpha=float(selected["alpha"]), fit_intercept=bool(selected["fit_intercept"]))
        model.fit(train[stack_features], train[TARGET])
        pred = valid[["contract_date", TARGET]].copy()
        pred["method"] = "ridge_stack_oof_for_calibration"
        pred["param_key"] = str(selected_key)
        pred["predicted_high_f"] = model.predict(valid[stack_features])
        pred["evaluation_scope"] = "year_split_stack_oof_calibration"
        pred["fold"] = f"meta_train_before_{validation_year}_validate_{validation_year}"
        oof_rows.append(_year_split_prediction_columns(pred))
    if not oof_rows:
        return _empty_year_split_predictions(), pd.DataFrame(), _empty_year_split_predictions()
    oof = pd.concat(oof_rows, ignore_index=True)
    oof["month"] = pd.to_datetime(oof["contract_date"], errors="coerce").dt.month
    residual = pd.to_numeric(oof[TARGET], errors="coerce") - pd.to_numeric(oof["predicted_high_f"], errors="coerce")
    global_correction = float(residual.mean())
    calibration = (
        pd.DataFrame({"month": oof["month"], "residual_f": residual})
        .dropna()
        .groupby("month", as_index=False)
        .agg(month_rows=("residual_f", "size"), month_mean_residual_f=("residual_f", "mean"))
    )
    calibration["global_mean_residual_f"] = global_correction
    enough_rows = calibration["month_rows"].ge(int(min_month_rows))
    shrink_weight = calibration["month_rows"] / (calibration["month_rows"] + max(1, int(shrinkage_rows)))
    shrunk = shrink_weight * calibration["month_mean_residual_f"] + (1.0 - shrink_weight) * global_correction
    calibration["correction_f"] = shrunk.where(enough_rows, global_correction).clip(
        lower=-abs(float(correction_cap_f)), upper=abs(float(correction_cap_f))
    )
    calibration["selected_param_key"] = str(selected_key)
    calibration["correction_cap_f"] = abs(float(correction_cap_f))
    calibration["min_month_rows"] = int(min_month_rows)
    calibration["shrinkage_rows"] = int(shrinkage_rows)

    ridge_test = test_predictions.loc[test_predictions["method"].eq(STACK_METHOD)].copy()
    if ridge_test.empty:
        return _empty_year_split_predictions(), calibration, oof
    ridge_test["month"] = pd.to_datetime(ridge_test["contract_date"], errors="coerce").dt.month
    ridge_test = ridge_test.merge(calibration[["month", "correction_f"]], on="month", how="left")
    ridge_test["correction_f"] = ridge_test["correction_f"].fillna(global_correction).clip(
        lower=-abs(float(correction_cap_f)), upper=abs(float(correction_cap_f))
    )
    ridge_test["predicted_high_f"] = pd.to_numeric(ridge_test["predicted_high_f"], errors="coerce") + ridge_test[
        "correction_f"
    ]
    ridge_test["method"] = "ridge_stack_oof_calibrated"
    ridge_test["param_key"] = ridge_test["param_key"].astype("string") + "+oof_month_calibration"
    ridge_test["evaluation_scope"] = "year_split_test"
    calibrated = _year_split_prediction_columns(ridge_test)
    return calibrated, calibration, oof


def guarded_blend_predictions(
    test_predictions: pd.DataFrame,
    caps_f: tuple[float, ...] = GUARDED_BLEND_CAPS_F,
    *,
    stack_method: str = STACK_METHOD,
    provider_method: str = "provider_mean",
) -> pd.DataFrame:
    if test_predictions.empty:
        return _empty_year_split_predictions()
    base = test_predictions.loc[test_predictions["evaluation_scope"].eq("year_split_test")].copy()
    if base.empty:
        return _empty_year_split_predictions()
    index_columns = ["contract_date"]
    if "station_id" in base:
        index_columns.insert(0, "station_id")
    pivot = base.pivot_table(index=index_columns, columns="method", values="predicted_high_f", aggfunc="first")
    if stack_method not in pivot or provider_method not in pivot:
        return _empty_year_split_predictions()
    actuals = base.groupby(index_columns, dropna=False)[TARGET].first()
    folds = base.groupby(index_columns, dropna=False)["fold"].first() if "fold" in base else None
    rows: list[pd.DataFrame] = []
    for cap in caps_f:
        capped = pivot[provider_method] + (pivot[stack_method] - pivot[provider_method]).clip(
            lower=-float(cap),
            upper=float(cap),
        )
        pred = capped.rename("predicted_high_f").to_frame().join(actuals)
        pred = pred.reset_index()
        pred["method"] = f"guarded_blend_cap_{float(cap):g}f"
        pred["param_key"] = f"cap_{float(cap):g}f"
        pred["evaluation_scope"] = "year_split_test"
        if folds is not None:
            pred = pred.merge(folds.rename("fold").reset_index(), on=index_columns, how="left")
        else:
            pred["fold"] = pd.NA
        rows.append(_year_split_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_year_split_predictions()


def select_guarded_blend_cap(
    predictions: pd.DataFrame,
    caps_f: tuple[float, ...] = GUARDED_BLEND_CAPS_F,
    *,
    baseline_method: str = "provider_mean",
) -> pd.DataFrame:
    columns = [
        "method",
        "cap_f",
        "count",
        "mae_f",
        "rmse_f",
        "bucket_hit_pct",
        "beats_provider_mean_mae",
        "provider_mean_mae_f",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    candidate_methods = [f"guarded_blend_cap_{float(cap):g}f" for cap in caps_f]
    frame = predictions.loc[
        predictions["evaluation_scope"].eq("year_split_test")
        & predictions["method"].isin([baseline_method, *candidate_methods])
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    key_columns = ["contract_date"]
    if "station_id" in frame:
        key_columns.insert(0, "station_id")
    method_keys = frame.groupby("method", dropna=False).apply(
        lambda group: set(map(tuple, group[key_columns].astype(str).to_numpy())),
        include_groups=False,
    )
    required_methods = [method for method in [baseline_method, *candidate_methods] if method in method_keys.index]
    if baseline_method not in required_methods:
        return pd.DataFrame(columns=columns)
    common_keys = set.intersection(*(method_keys.loc[method] for method in required_methods))
    if not common_keys:
        return pd.DataFrame(columns=columns)
    key_frame = pd.DataFrame(list(common_keys), columns=key_columns)
    common = frame.merge(key_frame, on=key_columns, how="inner")
    baseline = common.loc[common["method"].eq(baseline_method)]
    provider_mean_mae = float(pd.to_numeric(baseline["absolute_error_f"], errors="coerce").mean())
    rows: list[dict[str, Any]] = []
    for cap in caps_f:
        method = f"guarded_blend_cap_{float(cap):g}f"
        group = common.loc[common["method"].eq(method)].copy()
        if group.empty:
            continue
        error = pd.to_numeric(group["error_f"], errors="coerce")
        abs_error = error.abs()
        actual_bracket = _round_half_up_series(group[TARGET]).map(_temperature_bracket_from_rounded)
        predicted_bracket = _round_half_up_series(group["predicted_high_f"]).map(_temperature_bracket_from_rounded)
        bracket_hit = actual_bracket.eq(predicted_bracket).where(actual_bracket.notna() & predicted_bracket.notna())
        mae = float(abs_error.mean())
        rows.append(
            {
                "method": method,
                "cap_f": float(cap),
                "count": int(abs_error.notna().sum()),
                "mae_f": mae,
                "rmse_f": float(np.sqrt((error**2).mean())),
                "bucket_hit_pct": float(bracket_hit.mean() * 100.0),
                "beats_provider_mean_mae": bool(mae < provider_mean_mae),
                "provider_mean_mae_f": provider_mean_mae,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["mae_f", "bucket_hit_pct", "cap_f"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def year_split_scoreboard(validation_predictions: pd.DataFrame, test_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["period", "method", "count", "mae_f", "rmse_f"]
    frames: list[pd.DataFrame] = []
    for period, predictions in [
        ("validation_2024_2025", validation_predictions),
        (f"test_{YEAR_SPLIT_TEST_YEAR}", test_predictions),
    ]:
        if predictions.empty:
            continue
        frame = predictions.loc[predictions["method"].isin(YEAR_SPLIT_SCOREBOARD_METHODS)].copy()
        if frame.empty:
            continue
        metrics = frame.groupby("method", dropna=False).apply(_metric_row, include_groups=False).reset_index()
        metrics["period"] = period
        frames.append(metrics[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sort_year_split_visible_methods(pd.concat(frames, ignore_index=True), include_period=True)[columns]


def year_split_bracket_predictions(
    test_predictions: pd.DataFrame,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    columns = [
        "contract_date",
        "method",
        "actual_high_f",
        "predicted_high_f",
        "error_f",
        "absolute_error_f",
        "actual_rounded_high_f",
        "predicted_rounded_high_f",
        "actual_bracket",
        "predicted_bracket",
        "bracket_hit",
        "bucket_log_loss",
    ]
    if test_predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = test_predictions.loc[
        test_predictions["evaluation_scope"].eq("year_split_test")
        & test_predictions["method"].isin(YEAR_SPLIT_SCOREBOARD_METHODS)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["actual_rounded_high_f"] = _round_half_up_series(frame[TARGET])
    frame["predicted_rounded_high_f"] = _round_half_up_series(frame["predicted_high_f"])
    frame["actual_bracket"] = frame["actual_rounded_high_f"].map(_temperature_bracket_from_rounded)
    frame["predicted_bracket"] = frame["predicted_rounded_high_f"].map(_temperature_bracket_from_rounded)
    missing_bracket = frame["actual_bracket"].isna() | frame["predicted_bracket"].isna()
    frame["bracket_hit"] = frame["actual_bracket"].eq(frame["predicted_bracket"]).mask(missing_bracket).astype("boolean")
    frame["bucket_log_loss"] = frame.groupby("method", group_keys=False).apply(_bucket_log_loss_series, include_groups=False)
    return _sort_year_split_visible_methods(frame[columns]).reset_index(drop=True)


def year_split_bracket_metrics(bracket_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "count",
        "mae_f",
        "rmse_f",
        "bucket_log_loss",
        "bracket_accuracy_pct",
        "p95_absolute_error_f",
        "large_miss_5f_pct",
    ]
    if bracket_predictions.empty:
        return pd.DataFrame(columns=columns)
    metrics = bracket_predictions.groupby("method", dropna=False).apply(_metric_row, include_groups=False).reset_index()
    bracket_accuracy = (
        bracket_predictions.groupby("method", dropna=False)["bracket_hit"]
        .mean()
        .mul(100)
        .rename("bracket_accuracy_pct")
        .reset_index()
    )
    metrics = metrics.merge(bracket_accuracy, on="method", how="left")
    return _sort_year_split_visible_methods(metrics[columns]).reset_index(drop=True)


def polymarket_temperature_bracket(value: Any) -> str | None:
    rounded = round_temperature_half_up(value)
    if rounded is None:
        return None
    lower = rounded if rounded % 2 == 0 else rounded - 1
    return f"{lower}-{lower + 1}"


def round_temperature_half_up(value: Any) -> int | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return int(math.floor(float(number) + 0.5))


def year_split_feature_importance(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    selected_hyperparameters: pd.DataFrame,
    train_years: tuple[int, int] = YEAR_SPLIT_TEST_TRAIN_YEARS,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    columns = [
        "method",
        "param_key",
        "feature",
        "importance_mean_mae_f",
        "importance_std_mae_f",
        "n_repeats",
        "train_start_year",
        "train_end_year",
        "test_year",
        "train_rows",
        "test_rows",
    ]
    if frame.empty or selected_hyperparameters.empty:
        return pd.DataFrame(columns=columns)
    frame = _ensure_model_target_columns(
        add_strict_quality_flags(_with_actual_quality_columns(frame, config), providers=config.providers),
        config,
    )
    frame = frame.loc[frame[STRICT_QUALITY_OK_COLUMN].fillna(False)].copy()
    frame = _drop_missing_model_target(frame, config)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    from sklearn.inspection import permutation_importance

    year = pd.to_numeric(frame.get("year"), errors="coerce")
    train = frame.loc[year.between(train_years[0], train_years[1])].copy()
    test = frame.loc[year.eq(test_year)].copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=columns)

    fit_categorical, fit_numeric = _fit_feature_columns(
        train,
        categorical,
        numeric,
        max_missing_fraction=config.effective_max_feature_missing_fraction,
    )
    feature_names = [*fit_categorical, *fit_numeric]
    if not feature_names:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, row in selected_hyperparameters.iterrows():
        method = str(row["method"])
        params = _params_from_selected_row(row)
        estimator = _build_base_model_pipeline(config, fit_categorical, fit_numeric, method, params)
        try:
            estimator.fit(train[feature_names], _model_target_values(train, config))
            importance = permutation_importance(
                estimator,
                test[feature_names],
                test[TARGET],
                scoring=_high_prediction_mae_scorer(config, test),
                n_repeats=config.effective_feature_importance_repeats,
                random_state=config.random_state,
                n_jobs=1,
            )
        except Exception:
            continue
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "method": method,
                    "param_key": str(row.get("param_key", "")),
                    "feature": feature,
                    "importance_mean_mae_f": float(importance.importances_mean[index]),
                    "importance_std_mae_f": float(importance.importances_std[index]),
                    "n_repeats": config.effective_feature_importance_repeats,
                    "train_start_year": train_years[0],
                    "train_end_year": train_years[1],
                    "test_year": test_year,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["importance_mean_mae_f", "importance_std_mae_f"],
        ascending=[False, False],
        ignore_index=True,
    )


def _high_prediction_mae_scorer(config: StationStackingConfig, frame: pd.DataFrame):
    def scorer(estimator: Any, x: Any, y_true: Any) -> float:
        if hasattr(x, "index"):
            rows = frame.loc[x.index]
            actual = pd.to_numeric(pd.Series(y_true, index=x.index), errors="coerce")
        else:
            rows = frame.iloc[: len(x)]
            actual = pd.to_numeric(pd.Series(y_true), errors="coerce")
        predicted_high = _prediction_output_to_high(estimator.predict(x), rows, config)
        mask = actual.notna().to_numpy() & np.isfinite(predicted_high)
        if not mask.any():
            return float("-inf")
        return -float(np.abs(actual.to_numpy(dtype=float)[mask] - predicted_high[mask]).mean())

    return scorer


def summarize_year_split_predictions(validation_predictions: pd.DataFrame, test_predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for predictions in [validation_predictions, test_predictions]:
        if predictions.empty:
            continue
        frame = predictions.groupby(["evaluation_scope", "method"], dropna=False).apply(_metric_row, include_groups=False).reset_index()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["evaluation_scope", "method", "count", "mae_f", "rmse_f", "bias_f", "within_1f_pct", "within_2f_pct", "within_3f_pct"]
        )
    return pd.concat(frames, ignore_index=True).sort_values(["evaluation_scope", "rmse_f", "method"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame, config: StationStackingConfig) -> tuple[list[str], list[str]]:
    categorical = [column for column in ["day_of_week", *OBSERVED_CATEGORICAL_FEATURES] if column in frame]
    excluded = {
        TARGET,
        REMAINING_WARMUP_TARGET,
        "contract_date",
        "station_id",
        "station_name",
        "airport_name",
        "city_label",
        "timezone",
        "country",
        "all_provider_highs_available",
        "observed_as_of_time_local",
        "observed_as_of_time_utc",
        "observed_source",
        "observed_qc_field",
        "observed_raw_metar",
        "observed_data_source",
        "observed_unavailable_reason",
        "iem_actual_high_f",
        "settlement_high_f",
        "target_source",
        "target_source_diff_f",
        "settlement_source",
        "settlement_quality_flag",
        "actual_source",
        "actual_data_quality_flag",
        "actual_raw_observation_count",
        STRICT_QUALITY_OK_COLUMN,
        STRICT_QUALITY_ISSUES_COLUMN,
    }
    excluded.update(column for column in frame.columns if column.endswith("_source_file_or_url"))
    excluded.update(column for column in frame.columns if column.endswith("_source_cache_dir"))
    excluded.update(column for column in frame.columns if column.endswith("_data_source"))
    excluded.update(column for column in frame.columns if column.endswith("_source_label"))
    excluded.update(column for column in frame.columns if column.endswith("_model"))
    excluded.update(column for column in frame.columns if column.endswith("_cycle_selection_policy"))
    excluded.update(column for column in frame.columns if column.endswith("_forecast_as_of"))
    excluded.update(column for column in frame.columns if column.endswith("_issued_at"))
    excluded.update(column for column in frame.columns if column.endswith("_forecast_window_start"))
    excluded.update(column for column in frame.columns if column.endswith("_forecast_window_end"))
    version = config.effective_feature_version
    if version not in CURRENT_OBS_TREND_FEATURE_VERSIONS:
        excluded.update(
            {
                "observed_temp_change_last_1h_f",
                "observed_temp_change_last_3h_f",
                "observed_morning_warmup_rate_f_per_hour",
                "observed_high_so_far_change_since_9am_f",
            }
        )
    if version == "v8":
        excluded.update(V8_DROPPED_FEATURE_COLUMNS)
    if version == "v9":
        excluded.update(V9_DROPPED_FEATURE_COLUMNS)
    if version == "v10":
        excluded.update(V10_DROPPED_FEATURE_COLUMNS)
    if version == "v11":
        excluded.update(V11_DROPPED_FEATURE_COLUMNS)
    if version == V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION:
        excluded.update(V11_SETTLEMENT_FIX_DROPPED_FEATURE_COLUMNS)
        excluded.update(V20_PEAK_TIMING_RAW_FEATURE_COLUMNS)
        excluded.update(V20_ENGINEERED_FEATURE_COLUMNS)
    if version == "v12":
        excluded.update(V12_DROPPED_FEATURE_COLUMNS)
    if version == "v13":
        excluded.update(V13_DROPPED_FEATURE_COLUMNS)
    if version == "v14":
        excluded.update(V14_DROPPED_FEATURE_COLUMNS)
    if version in V15_FEATURE_VERSIONS:
        excluded.update(V15_DROPPED_FEATURE_COLUMNS)
    if version in V16_FEATURE_VERSIONS:
        excluded.update(V16_DROPPED_FEATURE_COLUMNS)
    if version in V17_FEATURE_VERSIONS:
        excluded.update(V17_DROPPED_FEATURE_COLUMNS)
    if version in {*V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS}:
        excluded.update(V18_DROPPED_FEATURE_COLUMNS)
    if version in V20_PEAK_TIMING_FEATURE_VERSIONS:
        excluded.update(V20_DROPPED_FEATURE_COLUMNS)
    categorical = [column for column in categorical if column not in excluded]

    numeric: list[str] = []
    for column in frame.columns:
        if column in excluded or column in categorical:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            if pd.to_numeric(series, errors="coerce").notna().any():
                numeric.append(column)
    if version == "v14":
        additional = set(V14_ADDITIONAL_FEATURE_COLUMNS)
        base = set(V11_FEATURE_COLUMNS)
        gated_numeric: list[str] = []
        for column in numeric:
            if column in base:
                gated_numeric.append(column)
                continue
            if _is_v14_blocked_weather_feature(column):
                continue
            if column in additional and not _passes_v14_added_feature_coverage(frame[column]):
                continue
            gated_numeric.append(column)
        numeric = gated_numeric
    if version in {"v15_forecast_temp_at_as_of", "v15_precip_cloud"}:
        additional = set(_v15_additional_feature_columns_for_version(version))
        base = set(V15_BASE_FEATURE_COLUMNS)
        gated_numeric = []
        for column in numeric:
            if column in base:
                gated_numeric.append(column)
                continue
            if _is_v15_blocked_weather_feature(column, additional):
                continue
            if column in additional and not _passes_v15_added_feature_train_coverage(frame, column, config):
                continue
            gated_numeric.append(column)
        numeric = gated_numeric
    if version == V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION:
        additional = set(V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS)
        base = set(V11_FEATURE_COLUMNS)
        gated_numeric = []
        for column in numeric:
            if column in base:
                gated_numeric.append(column)
                continue
            if _is_blocked_weather_feature(column, additional):
                continue
            gated_numeric.append(column)
        numeric = gated_numeric
    if version == "v16_fused":
        additional = set(V16_ADDITIONAL_FEATURE_COLUMNS)
        base = set(V16_BASE_FEATURE_COLUMNS)
        gated_numeric = []
        for column in numeric:
            if column in V16_BLOCKED_BASE_FEATURE_COLUMNS:
                continue
            if column in base:
                gated_numeric.append(column)
                continue
            if _is_v16_blocked_weather_feature(column, additional):
                continue
            if column in additional and not _passes_v15_added_feature_train_coverage(frame, column, config):
                continue
            gated_numeric.append(column)
        numeric = gated_numeric
    if version == "v17_importance_015":
        allowed = set(V17_IMPORTANCE_015_FEATURE_COLUMNS)
        additional = set(V17_ADDITIONAL_FEATURE_COLUMNS)
        categorical = []
        numeric = [
            column
            for column in numeric
            if column in allowed
            and (column not in additional or _passes_v15_added_feature_train_coverage(frame, column, config))
        ]
    if version == "v18":
        additional = set(V18_ADDITIONAL_FEATURE_COLUMNS)
        allowed = set(V18_FEATURE_COLUMNS)
        numeric = [
            column
            for column in numeric
            if column in allowed
            and (column not in additional or _passes_v15_added_feature_train_coverage(frame, column, config))
        ]
    if version == "v18_1_nbm":
        additional = set(V18_NBM_CURVE_FEATURE_COLUMNS)
        allowed = set(V18_1_NBM_FEATURE_COLUMNS)
        numeric = [
            column
            for column in numeric
            if column in allowed
            and (column not in additional or _passes_v15_added_feature_train_coverage(frame, column, config))
        ]
    if version == "v18_1_rap":
        additional = set([*V18_RAP_PHYSICS_FEATURE_COLUMNS, *V18_STATION_SPECIFIC_FEATURE_COLUMNS])
        allowed = set(V18_1_RAP_FEATURE_COLUMNS)
        numeric = [
            column
            for column in numeric
            if column in allowed
            and (column not in additional or _passes_v15_added_feature_train_coverage(frame, column, config))
        ]
    if version in V20_PEAK_TIMING_FEATURE_VERSIONS:
        allowed = set(
            V20_KDAL_NBM_PHYSICS_FEATURE_COLUMNS
            if version == V20_KDAL_FIX_FEATURE_VERSION
            else V20_FEATURE_COLUMNS
        )
        numeric = [column for column in numeric if column in allowed]
    return categorical, numeric


def _passes_v14_added_feature_coverage(series: pd.Series) -> bool:
    non_null_fraction = pd.to_numeric(series, errors="coerce").notna().mean()
    return bool(non_null_fraction >= V14_ADDITIONAL_MIN_NON_NULL_FRACTION)


def _v15_additional_feature_columns_for_version(version: str) -> tuple[str, ...]:
    if version == "v15_forecast_temp_at_as_of":
        return tuple(V15_FORECAST_TEMP_AT_AS_OF_FEATURE_COLUMNS)
    if version == "v15_precip_cloud":
        return tuple(V15_PRECIP_CLOUD_FEATURE_COLUMNS)
    return ()


def _passes_v15_added_feature_train_coverage(
    frame: pd.DataFrame,
    column: str,
    config: StationStackingConfig,
) -> bool:
    coverage_frame = _v15_train_coverage_frame(frame, config)
    non_null_fraction = pd.to_numeric(coverage_frame[column], errors="coerce").notna().mean()
    return bool(non_null_fraction >= V15_ADDITIONAL_MIN_TRAIN_NON_NULL_FRACTION)


def _v15_train_coverage_frame(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    if "year" not in frame:
        return frame
    train_start_year, train_end_year = config.year_split_test_train_years
    years = pd.to_numeric(frame["year"], errors="coerce")
    train_mask = years.between(train_start_year, train_end_year, inclusive="both")
    if not bool(train_mask.any()):
        return frame
    return frame.loc[train_mask]


def _is_v14_blocked_weather_feature(column: str) -> bool:
    return _is_blocked_weather_feature(column, set(V14_ADDITIONAL_FEATURE_COLUMNS))


def _is_v15_blocked_weather_feature(column: str, additional: set[str]) -> bool:
    return _is_blocked_weather_feature(column, additional)


def _is_v16_blocked_weather_feature(column: str, additional: set[str]) -> bool:
    return _is_blocked_weather_feature(column, additional)


def _is_blocked_weather_feature(column: str, allowed_additional: set[str]) -> bool:
    if column in allowed_additional:
        return False
    lowered = str(column).lower()
    weather_tokens = (
        "cloud",
        "ceiling",
        "dewpoint",
        "forecast_precip",
        "forecast_has_precip",
        "forecast_temp_at_as_of",
        "humidity",
        "precip",
        "precip_amount",
        "pressure",
        "shortwave",
        "visibility",
        "wind_",
    )
    if lowered.startswith("v13_"):
        return True
    if lowered.startswith(("v4_", "v8_")) and any(token in lowered for token in weather_tokens):
        return True
    if lowered.startswith(("gfs_", "hrrr_", "nbm_")) and any(token in lowered for token in weather_tokens):
        return True
    return False


def _source_quality_rank(source_cache_dir: str) -> int:
    name = str(source_cache_dir).lower()
    if "smoke" in name:
        return 5
    if "retry" in name or "gated" in name:
        return 4
    if "fixed" in name:
        return 4
    if "direct_gfs" in name or "direct_hrrr" in name:
        return 0
    if "direct_nbm" in name:
        return 2
    if "sdk_11am_nbm" in name:
        return 0
    return 1


def _include_forecast_cache_path(path: Path) -> bool:
    name = path.parent.name.lower()
    if "direct_nbm" in name:
        value = os.getenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "")
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return True


def _load_station_actuals(
    root: Path,
    station_id: str,
    target_source: str = TARGET_SOURCE_IEM_HOURLY,
) -> pd.DataFrame:
    path = root / "data" / "processed" / "actual_highs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing actual highs file: {path}")
    actuals = pd.read_csv(path)
    required = {"station_code", "date_local", "actual_high_f"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = actuals.loc[actuals["station_code"].astype(str).str.upper().eq(station_id)].copy()
    out = out.rename(
        columns={
            "date_local": "contract_date",
            "actual_high_f": "iem_actual_high_f",
            "source": "actual_source",
            "data_quality_flag": "actual_data_quality_flag",
            "raw_observation_count": "actual_raw_observation_count",
        }
    )
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["iem_actual_high_f"] = pd.to_numeric(out["iem_actual_high_f"], errors="coerce")
    out[TARGET] = out["iem_actual_high_f"]
    out["target_source"] = TARGET_SOURCE_IEM_HOURLY
    out["settlement_high_f"] = pd.NA
    out["settlement_source"] = pd.NA
    out["settlement_quality_flag"] = pd.NA
    optional = ["actual_source", "actual_data_quality_flag", "actual_raw_observation_count"]
    for column in optional:
        if column not in out:
            out[column] = pd.NA
    out["actual_raw_observation_count"] = pd.to_numeric(out["actual_raw_observation_count"], errors="coerce")
    out = _apply_settlement_first_actuals(root, station_id, out, target_source=target_source)
    out["target_source_diff_f"] = pd.to_numeric(out["settlement_high_f"], errors="coerce") - pd.to_numeric(
        out["iem_actual_high_f"],
        errors="coerce",
    )
    keep = [
        "contract_date",
        TARGET,
        "iem_actual_high_f",
        "settlement_high_f",
        "target_source",
        "target_source_diff_f",
        "settlement_source",
        "settlement_quality_flag",
        *optional,
    ]
    return out[keep].dropna(subset=["contract_date"]).sort_values("contract_date").reset_index(drop=True)


def _apply_settlement_first_actuals(
    root: Path,
    station_id: str,
    actuals: pd.DataFrame,
    target_source: str,
) -> pd.DataFrame:
    source = str(target_source or TARGET_SOURCE_IEM_HOURLY).strip().lower().replace("-", "_")
    if source not in {TARGET_SOURCE_SETTLEMENT_FIRST, TARGET_SOURCE_WUNDERGROUND_ONLY}:
        return actuals
    settlement_path = root / "data" / "processed" / "settlement_actual_highs.csv"
    if not settlement_path.exists():
        if source == TARGET_SOURCE_WUNDERGROUND_ONLY:
            out = actuals.copy()
            out[TARGET] = pd.NA
            out["target_source"] = TARGET_SOURCE_WUNDERGROUND_ONLY
            out["actual_source"] = "missing_wunderground"
            out["actual_data_quality_flag"] = "missing_wunderground"
            out["actual_raw_observation_count"] = pd.NA
            return out
        return actuals
    settlements = pd.read_csv(settlement_path)
    required = {"station_id", "contract_date", "settlement_high_f"}
    missing = required - set(settlements.columns)
    if missing:
        raise ValueError(f"{settlement_path} missing required columns: {sorted(missing)}")
    settlements = settlements.loc[settlements["station_id"].astype(str).str.upper().eq(station_id)].copy()
    if settlements.empty:
        if source == TARGET_SOURCE_WUNDERGROUND_ONLY:
            out = actuals.copy()
            out[TARGET] = pd.NA
            out["target_source"] = TARGET_SOURCE_WUNDERGROUND_ONLY
            out["actual_source"] = "missing_wunderground"
            out["actual_data_quality_flag"] = "missing_wunderground"
            out["actual_raw_observation_count"] = pd.NA
            return out
        return actuals
    settlements["contract_date"] = settlements["contract_date"].astype(str).str[:10]
    settlements["settlement_high_f"] = pd.to_numeric(settlements["settlement_high_f"], errors="coerce")
    rename = {
        "quality_flag": "settlement_quality_flag",
    }
    settlements = settlements.rename(columns=rename)
    for column in ["settlement_source", "settlement_quality_flag"]:
        if column not in settlements:
            settlements[column] = pd.NA
    if source == TARGET_SOURCE_WUNDERGROUND_ONLY:
        settlements = settlements.loc[
            settlements["settlement_source"].astype("string").str.strip().str.lower().eq("wunderground_station_history")
            & settlements["settlement_quality_flag"].astype("string").str.strip().str.lower().eq("ok")
            & settlements["settlement_high_f"].notna()
        ].copy()
    settlements = settlements.dropna(subset=["contract_date"]).sort_values("contract_date")
    settlements = settlements.drop_duplicates("contract_date", keep="last")

    out = actuals.merge(
        settlements[["contract_date", "settlement_high_f", "settlement_source", "settlement_quality_flag"]],
        on="contract_date",
        how="left",
        suffixes=("", "_settlement"),
    )
    for column in ["settlement_high_f", "settlement_source", "settlement_quality_flag"]:
        settlement_column = f"{column}_settlement"
        if settlement_column in out:
            out[column] = out[settlement_column].combine_first(out[column])
            out = out.drop(columns=[settlement_column])
    has_settlement = pd.to_numeric(out["settlement_high_f"], errors="coerce").notna()
    out.loc[has_settlement, TARGET] = pd.to_numeric(out.loc[has_settlement, "settlement_high_f"], errors="coerce")
    out.loc[has_settlement, "target_source"] = source
    out.loc[has_settlement, "actual_source"] = out.loc[has_settlement, "settlement_source"].fillna("settlement")
    out.loc[has_settlement, "actual_data_quality_flag"] = out.loc[has_settlement, "settlement_quality_flag"].fillna("ok")
    if source == TARGET_SOURCE_WUNDERGROUND_ONLY:
        missing_wunderground = ~has_settlement
        out.loc[missing_wunderground, TARGET] = pd.NA
        out.loc[missing_wunderground, "target_source"] = TARGET_SOURCE_WUNDERGROUND_ONLY
        out.loc[missing_wunderground, "actual_source"] = "missing_wunderground"
        out.loc[missing_wunderground, "actual_data_quality_flag"] = "missing_wunderground"
        out["actual_raw_observation_count"] = pd.NA
    return out


def _load_station_meta(root: Path, station_id: str) -> dict[str, Any]:
    path = root / "data" / "processed" / "station_registry.csv"
    if not path.exists():
        return {"station_id": station_id, "timezone": "UTC"}
    frame = pd.read_csv(path)
    code_col = "station_code" if "station_code" in frame.columns else "station_id"
    row = frame.loc[frame[code_col].astype(str).str.upper().eq(station_id)].head(1)
    if row.empty:
        return {"station_id": station_id, "timezone": "UTC"}
    values = row.iloc[0].to_dict()
    values["station_id"] = station_id
    for column in ["lat", "lon"]:
        if column in values:
            values[column] = pd.to_numeric(values[column], errors="coerce")
    return values


def _with_actual_quality_columns(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    quality_columns = ["actual_source", "actual_data_quality_flag", "actual_raw_observation_count"]
    if frame.empty or "contract_date" not in frame or "station_id" not in frame:
        return frame
    if all(column in frame for column in quality_columns):
        return frame
    station = config.station_id.upper()
    station_values = frame["station_id"].astype("string").str.upper().dropna().unique()
    if len(station_values) and any(value != station for value in station_values):
        return frame
    try:
        actuals = _load_station_actuals(
            config.resolved_project_root(),
            station,
            target_source=config.effective_target_source,
        )
    except (FileNotFoundError, ValueError):
        return frame
    if actuals.empty:
        return frame
    quality = actuals[["contract_date", *quality_columns]].drop_duplicates("contract_date")
    out = frame.copy()
    merge_columns = ["contract_date", *[column for column in quality_columns if column not in out]]
    if len(merge_columns) == 1:
        return out
    return out.merge(quality[merge_columns], on="contract_date", how="left")


def _provider_numeric_columns_for_feature_version(feature_version: str) -> list[str]:
    version = _normalize_feature_version(feature_version)
    columns = [*PROVIDER_NUMERIC_COLUMNS]
    if version in WEATHER_AGGREGATE_FEATURE_VERSIONS:
        columns.extend(V13_PROVIDER_NUMERIC_COLUMNS)
    return list(dict.fromkeys(columns))


def load_v18_nbm_rap_features(
    project_root: str | Path = ".",
    *,
    shard_root: str | Path | None = None,
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
) -> pd.DataFrame:
    root = Path(project_root)
    source_root = Path(shard_root) if shard_root is not None else root / V18_NBM_RAP_SHARD_ROOT
    if not source_root.is_absolute():
        source_root = root / source_root
    files = sorted(source_root.glob("**/nbm_rap_features.csv"))
    if not files:
        return _empty_v18_nbm_rap_features()
    frames: list[pd.DataFrame] = []
    for file_index, path in enumerate(files):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame = frame.copy()
        frame["v18_shard_source_path"] = str(path)
        frame["_v18_file_index"] = file_index
        frames.append(frame)
    if not frames:
        return _empty_v18_nbm_rap_features()
    combined = pd.concat(frames, ignore_index=True)
    required = {"station_id", "contract_date"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"V18 NBM/RAP shard data missing required columns: {sorted(missing)}")
    combined["station_id"] = combined["station_id"].astype("string").str.upper().str.strip()
    combined["contract_date"] = combined["contract_date"].astype("string").str[:10]
    if "timing_mode" in combined:
        combined = combined.loc[combined["timing_mode"].astype("string").str.strip().eq(timing_mode)].copy()
    if combined.empty:
        return _empty_v18_nbm_rap_features()
    for column in V18_ADDITIONAL_FEATURE_COLUMNS:
        if column in combined:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined["_v18_both_core_ok"] = (
        _status_ok(combined, "nbm_core_fetch_status") & _status_ok(combined, "rap_fetch_status")
    ).astype(int)
    combined["_v18_physics_ok"] = _status_ok(combined, "physics_fetch_status").astype(int)
    combined["_v18_row_ok"] = _status_ok(combined, "row_status").astype(int)
    present_feature_columns = [column for column in V18_ADDITIONAL_FEATURE_COLUMNS if column in combined]
    if present_feature_columns:
        combined["_v18_feature_non_null_count"] = combined[present_feature_columns].notna().sum(axis=1)
    else:
        combined["_v18_feature_non_null_count"] = 0
    combined["v18_shard_duplicate_count"] = combined.groupby(["station_id", "contract_date"])["contract_date"].transform("size")
    combined = combined.sort_values(
        [
            "station_id",
            "contract_date",
            "_v18_both_core_ok",
            "_v18_physics_ok",
            "_v18_row_ok",
            "_v18_feature_non_null_count",
            "_v18_file_index",
        ],
        ascending=[True, True, False, False, False, False, True],
    )
    combined = combined.drop_duplicates(["station_id", "contract_date"], keep="first")
    combined = combined.drop(
        columns=[
            "_v18_both_core_ok",
            "_v18_physics_ok",
            "_v18_row_ok",
            "_v18_feature_non_null_count",
            "_v18_file_index",
        ],
        errors="ignore",
    )
    keep = ["station_id", "contract_date", *V18_AUDIT_COLUMNS, *V18_ADDITIONAL_FEATURE_COLUMNS]
    for column in keep:
        if column not in combined:
            combined[column] = pd.NA
    return combined[keep].sort_values(["station_id", "contract_date"]).reset_index(drop=True)


def summarize_v18_nbm_rap_readiness(
    project_root: str | Path = ".",
    *,
    stations: Iterable[str] = TARGET_STATIONS,
    shard_root: str | Path | None = None,
) -> pd.DataFrame:
    features = load_v18_nbm_rap_features(project_root, shard_root=shard_root)
    columns = [
        "station_id",
        "first_contract_date",
        "last_contract_date",
        "rows",
        "nbm_ok_rows",
        "rap_ok_rows",
        "both_ok_rows",
        "duplicate_key_rows",
    ]
    rows: list[dict[str, Any]] = []
    for station in sorted({str(item).upper() for item in stations}):
        group = features.loc[features["station_id"].astype("string").str.upper().eq(station)].copy()
        if group.empty:
            rows.append(
                {
                    "station_id": station,
                    "first_contract_date": pd.NA,
                    "last_contract_date": pd.NA,
                    "rows": 0,
                    "nbm_ok_rows": 0,
                    "rap_ok_rows": 0,
                    "both_ok_rows": 0,
                    "duplicate_key_rows": 0,
                }
            )
            continue
        nbm_ok = _status_ok(group, "nbm_core_fetch_status")
        rap_ok = _status_ok(group, "rap_fetch_status")
        duplicates = pd.to_numeric(group.get("v18_shard_duplicate_count"), errors="coerce").fillna(1).gt(1)
        rows.append(
            {
                "station_id": station,
                "first_contract_date": str(group["contract_date"].min()),
                "last_contract_date": str(group["contract_date"].max()),
                "rows": int(len(group)),
                "nbm_ok_rows": int(nbm_ok.sum()),
                "rap_ok_rows": int(rap_ok.sum()),
                "both_ok_rows": int((nbm_ok & rap_ok).sum()),
                "duplicate_key_rows": int(duplicates.sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def load_v20_peak_timing_features(
    project_root: str | Path = ".",
    *,
    shard_roots: Iterable[str | Path] | None = None,
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
) -> pd.DataFrame:
    """Load completed V20 shards and keep the best row per station/date."""
    root = Path(project_root)
    roots = tuple(shard_roots) if shard_roots is not None else V20_PEAK_TIMING_SHARD_ROOTS
    files: list[Path] = []
    for raw_source_root in roots:
        source_root = Path(raw_source_root)
        if not source_root.is_absolute():
            source_root = root / source_root
        files.extend(source_root.glob("**/peak_timing_features.csv"))
    files = sorted(set(files))
    if not files:
        return pd.DataFrame(columns=["station_id", "contract_date", *V20_PEAK_TIMING_RAW_FEATURE_COLUMNS])

    frames: list[pd.DataFrame] = []
    for file_index, path in enumerate(files):
        try:
            shard = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if shard.empty:
            continue
        shard = shard.copy()
        shard["v20_shard_source_path"] = str(path)
        shard["_v20_file_index"] = file_index
        frames.append(shard)
    if not frames:
        return pd.DataFrame(columns=["station_id", "contract_date", *V20_PEAK_TIMING_RAW_FEATURE_COLUMNS])

    combined = pd.concat(frames, ignore_index=True)
    required = {"station_id", "contract_date", "timing_mode"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"V20 peak-timing shard data missing required columns: {sorted(missing)}")
    combined["station_id"] = combined["station_id"].astype("string").str.upper().str.strip()
    combined["contract_date"] = combined["contract_date"].astype("string").str[:10]
    combined = combined.loc[combined["timing_mode"].astype("string").str.strip().eq(timing_mode)].copy()
    if combined.empty:
        return pd.DataFrame(columns=["station_id", "contract_date", *V20_PEAK_TIMING_RAW_FEATURE_COLUMNS])

    numeric_candidates = set(V20_PEAK_TIMING_RAW_FEATURE_COLUMNS)
    numeric_candidates.update(f"hrrr_dswrf_{hour}l_w_m2" for hour in range(11, 19))
    numeric_candidates.update(f"hrrr_tcc_{hour}l_pct" for hour in range(11, 19))
    numeric_candidates.update(
        {
            "hrrr_precip_onset_hour_local",
            "hrrr_precip_onset_minus_hrrr_peak_hours",
            "hrrr_precip_onset_minus_nbm_peak_hours",
        }
    )
    for column in numeric_candidates:
        if column in combined:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")

    combined["_v20_both_core_ok"] = (
        _status_ok(combined, "nbm_core_fetch_status") & _status_ok(combined, "hrrr_fetch_status")
    ).astype(int)
    combined["_v20_profile_complete"] = pd.to_numeric(
        combined.get("hrrr_profile_complete", pd.Series(0, index=combined.index)), errors="coerce"
    ).fillna(0).gt(0).astype(int)
    combined["_v20_row_ok"] = _status_ok(combined, "row_status").astype(int)
    present = [column for column in numeric_candidates if column in combined]
    combined["_v20_feature_non_null_count"] = combined[present].notna().sum(axis=1) if present else 0
    combined["v20_shard_duplicate_count"] = combined.groupby(["station_id", "contract_date"])[
        "contract_date"
    ].transform("size")
    combined = combined.sort_values(
        [
            "station_id",
            "contract_date",
            "_v20_both_core_ok",
            "_v20_profile_complete",
            "_v20_row_ok",
            "_v20_feature_non_null_count",
            "_v20_file_index",
        ],
        ascending=[True, True, False, False, False, False, False],
    ).drop_duplicates(["station_id", "contract_date"], keep="first")
    combined = combined.rename(columns={"timing_mode": "timing_mode_v20_peak"})
    return combined.drop(
        columns=[
            "_v20_both_core_ok",
            "_v20_profile_complete",
            "_v20_row_ok",
            "_v20_feature_non_null_count",
            "_v20_file_index",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def _merge_v20_peak_timing_features(root: Path, station_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    features = load_v20_peak_timing_features(root)
    station_features = features.loc[features["station_id"].eq(station_id.upper())].copy()
    if station_features.empty:
        return frame
    out = frame.copy()
    out["contract_date"] = out["contract_date"].astype("string").str[:10]
    merge_columns = [column for column in station_features.columns if column != "station_id"]
    return out.merge(station_features[merge_columns], on="contract_date", how="left", suffixes=("", "_v20_peak"))


def _expand_v20_actual_date_spine(
    root: Path,
    station_id: str,
    actuals: pd.DataFrame,
    *,
    target_source: str,
) -> pd.DataFrame:
    peak = load_v20_peak_timing_features(root)
    peak_dates = peak.loc[peak["station_id"].eq(station_id.upper()), ["contract_date"]].drop_duplicates()
    if peak_dates.empty:
        return actuals
    out = peak_dates.merge(actuals, on="contract_date", how="outer")
    out = _apply_settlement_first_actuals(root, station_id, out, target_source=target_source)
    out["target_source_diff_f"] = pd.to_numeric(out.get("settlement_high_f"), errors="coerce") - pd.to_numeric(
        out.get("iem_actual_high_f"), errors="coerce"
    )
    return out.sort_values("contract_date").reset_index(drop=True)


def v20_peak_timing_readiness(
    features: pd.DataFrame,
    *,
    station_id: str,
    folds: tuple[YearSplitFold, ...] = V20_EXPANDING_FOLDS,
    max_missing_fraction: float = 0.03,
    start_date: str = "2021-01-01",
    end_date: str = "2026-07-14",
) -> tuple[bool, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit partial pulls without allowing incomplete data to reach tuning."""
    expected_dates = pd.date_range(start_date, end_date, freq="D")
    indexed = features.copy()
    indexed["contract_date"] = pd.to_datetime(indexed.get("contract_date"), errors="coerce")
    indexed = indexed.dropna(subset=["contract_date"]).drop_duplicates("contract_date").set_index("contract_date")
    indexed = indexed.reindex(expected_dates)
    peak_ready = _numeric_series(indexed, "hrrr_t11l_f").notna() & _numeric_series(indexed, "nbm_t11l_f").notna()
    target_ready = _numeric_series(indexed, TARGET).notna()

    summary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    years = pd.Series(indexed.index.year, index=indexed.index)
    for year in range(2021, 2027):
        year_mask = years.eq(year)
        expected = int(year_mask.sum())
        peak_count = int(peak_ready.loc[year_mask].sum())
        target_count = int(target_ready.loc[year_mask].sum())
        peak_missing_fraction = 1.0 - peak_count / expected if expected else 1.0
        target_missing_fraction = 1.0 - target_count / expected if expected else 1.0
        summary_rows.append(
            {
                "station_id": station_id.upper(),
                "year": year,
                "expected_days": expected,
                "peak_ready_days": peak_count,
                "wunderground_target_days": target_count,
                "peak_missing_fraction": peak_missing_fraction,
                "target_missing_fraction": target_missing_fraction,
                "peak_ready": peak_missing_fraction <= max_missing_fraction,
                "target_ready": target_missing_fraction <= max_missing_fraction,
            }
        )
        for contract_date in indexed.index[year_mask & ~peak_ready]:
            missing_rows.append({"station_id": station_id.upper(), "contract_date": contract_date.date().isoformat(), "missing": "peak_timing"})
        for contract_date in indexed.index[year_mask & ~target_ready]:
            missing_rows.append({"station_id": station_id.upper(), "contract_date": contract_date.date().isoformat(), "missing": "wunderground_target"})

    fold_rows: list[dict[str, Any]] = []
    feature_dates = pd.to_datetime(features.get("contract_date"), errors="coerce")
    feature_years = feature_dates.dt.year
    for fold in (*folds, YearSplitFold("test_refit_2021_2025", 2021, 2025, 2026)):
        train = features.loc[
            feature_years.between(fold.train_start_year, fold.train_end_year)
            & pd.to_numeric(features.get(TARGET), errors="coerce").notna()
        ].copy()
        for feature in [*V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS, *V20_PEAK_TIMING_RAW_FEATURE_COLUMNS, *V20_ENGINEERED_FEATURE_COLUMNS]:
            values = pd.to_numeric(train.get(feature, pd.Series(np.nan, index=train.index)), errors="coerce")
            missing_fraction = float(values.isna().mean()) if len(values) else 1.0
            fold_rows.append(
                {
                    "station_id": station_id.upper(),
                    "fold": fold.name,
                    "train_start_year": fold.train_start_year,
                    "train_end_year": fold.train_end_year,
                    "train_rows": len(train),
                    "feature": feature,
                    "missing_fraction": missing_fraction,
                    "retained": bool(values.notna().any() and missing_fraction <= max_missing_fraction),
                }
            )

    summary = pd.DataFrame(summary_rows)
    missing_dates = pd.DataFrame(missing_rows, columns=["station_id", "contract_date", "missing"])
    fold_missingness = pd.DataFrame(fold_rows)
    ready = bool(summary["peak_ready"].all() and summary["target_ready"].all())
    return ready, summary, missing_dates, fold_missingness


def _merge_v18_nbm_rap_features(root: Path, station_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    features = load_v18_nbm_rap_features(root)
    if features.empty:
        return frame
    station_features = features.loc[features["station_id"].astype("string").str.upper().eq(station_id)].copy()
    if station_features.empty:
        return frame
    out = frame.copy()
    out["station_id"] = out["station_id"].astype("string").str.upper()
    station_features["contract_date"] = station_features["contract_date"].astype("string").str[:10]
    return out.merge(station_features, on=["station_id", "contract_date"], how="left", suffixes=("", "_v18_shard"))


def _status_ok(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype("string").str.strip().str.lower().eq("ok")


def _empty_v18_nbm_rap_features() -> pd.DataFrame:
    return pd.DataFrame(columns=["station_id", "contract_date", *V18_AUDIT_COLUMNS, *V18_ADDITIONAL_FEATURE_COLUMNS])


def _provider_wide(
    frame: pd.DataFrame,
    provider: str,
    numeric_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    numeric_columns = tuple(numeric_columns or PROVIDER_NUMERIC_COLUMNS)
    output_columns = _provider_wide_columns(provider, numeric_columns=numeric_columns)
    if frame.empty:
        return pd.DataFrame(columns=["contract_date", *output_columns])
    keep = ["contract_date", *numeric_columns, *PROVIDER_TEXT_COLUMNS]
    for column in keep:
        if column not in frame:
            frame[column] = pd.NA
    out = frame[keep].drop_duplicates("contract_date", keep="first").copy()
    rename: dict[str, str] = {
        "raw_forecast_high_f": HIGH_COLUMNS[provider],
    }
    for column in numeric_columns:
        if column != "raw_forecast_high_f":
            rename[column] = f"{provider}_{column}"
    for column in PROVIDER_TEXT_COLUMNS:
        rename[column] = f"{provider}_{column}"
    out = out.rename(columns=rename)
    for column in output_columns:
        if column not in out:
            out[column] = pd.NA
    return out[["contract_date", *output_columns]]


def _provider_wide_columns(provider: str, numeric_columns: Iterable[str] | None = None) -> list[str]:
    numeric_columns = tuple(numeric_columns or PROVIDER_NUMERIC_COLUMNS)
    columns = [HIGH_COLUMNS[provider]]
    columns.extend(f"{provider}_{column}" for column in numeric_columns if column != "raw_forecast_high_f")
    columns.extend(f"{provider}_{column}" for column in PROVIDER_TEXT_COLUMNS)
    return columns


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["contract_date"], errors="coerce")
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["day_of_year"] = dates.dt.dayofyear
    out["day_of_year_sin"] = np.sin(2 * math.pi * out["day_of_year"] / 366)
    out["day_of_year_cos"] = np.cos(2 * math.pi * out["day_of_year"] / 366)
    out["day_of_week"] = dates.dt.day_name()
    out["is_weekend"] = dates.dt.dayofweek >= 5
    return out


def _add_current_observation_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "observed_temp_at_as_of_f" not in out:
        return out
    temp = pd.to_numeric(out.get("observed_temp_at_as_of_f"), errors="coerce")
    high_temp = pd.to_numeric(out.get("observed_high_temp_through_as_of_f"), errors="coerce")
    dewpoint = pd.to_numeric(out.get("observed_dewpoint_at_as_of_f"), errors="coerce")
    humidity = pd.to_numeric(out.get("observed_humidity_at_as_of"), errors="coerce")
    wind_speed = pd.to_numeric(out.get("observed_wind_speed_at_as_of"), errors="coerce")
    wind_direction = pd.to_numeric(out.get("observed_wind_direction_at_as_of"), errors="coerce")
    visibility = pd.to_numeric(out.get("observed_visibility_at_as_of"), errors="coerce")
    precip_recent = pd.to_numeric(out.get("observed_precip_recent_at_as_of"), errors="coerce")
    weather_code = out.get("observed_weather_code_at_as_of")
    weather_text = weather_code.astype("string").str.upper() if weather_code is not None else pd.Series(pd.NA, index=out.index)
    rain_code = weather_text.str.contains(r"(?:^|\s)[-+]?(?:RA|SHRA|TSRA|FZRA)\b", regex=True, na=False)
    drizzle_code = weather_text.str.contains(r"(?:^|\s)[-+]?(?:DZ|FZDZ)\b", regex=True, na=False)
    snow_code = weather_text.str.contains(r"(?:^|\s)[-+]?(?:SN|SHSN|BLSN)\b", regex=True, na=False)
    precip_code = rain_code | drizzle_code | snow_code
    light_precip_code = weather_text.str.contains(r"(?:^|\s)-(?:RA|SHRA|TSRA|FZRA|DZ|FZDZ|SN|SHSN)\b", regex=True, na=False)
    heavy_precip_code = weather_text.str.contains(r"(?:^|\s)\+(?:RA|SHRA|TSRA|FZRA|DZ|FZDZ|SN|SHSN)\b", regex=True, na=False)
    moderate_precip_code = precip_code & ~light_precip_code & ~heavy_precip_code

    out["observed_dewpoint_depression_f"] = temp - dewpoint
    out["observed_high_temp_minus_temp_at_as_of_f"] = high_temp - temp
    out["observed_heat_index_at_as_of_f"] = _heat_index_f(temp, humidity)
    out["observed_wind_chill_at_as_of_f"] = _wind_chill_f(temp, wind_speed)
    radians = 2 * math.pi * wind_direction / 360
    out["observed_wind_dir_sin"] = np.sin(radians)
    out["observed_wind_dir_cos"] = np.cos(radians)
    out["observed_is_raining_at_as_of"] = rain_code | drizzle_code | precip_recent.fillna(0).gt(0)
    out["observed_is_drizzle_at_as_of"] = drizzle_code
    out["observed_is_snowing_at_as_of"] = snow_code
    out["observed_is_fog_or_mist_at_as_of"] = (
        weather_text.str.contains(r"\b(?:FG|BR|HZ)\b", regex=True, na=False)
        | visibility.le(3)
    )
    out["observed_is_thunder_at_as_of"] = weather_text.str.contains(r"\bTS\b|TSRA|VCTS", regex=True, na=False)
    recent_precip = precip_recent.fillna(0)
    intensity_code = np.select(
        [
            heavy_precip_code | recent_precip.ge(0.10),
            moderate_precip_code | recent_precip.ge(0.03),
            light_precip_code | precip_code | recent_precip.gt(0),
        ],
        [3, 2, 1],
        default=0,
    )
    out["observed_precip_intensity_code"] = intensity_code
    out["observed_precip_intensity"] = pd.Series(intensity_code, index=out.index).map(
        {0: "none", 1: "light", 2: "moderate", 3: "heavy"}
    )
    return out


def _heat_index_f(temp_f: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_f, errors="coerce")
    humidity = pd.to_numeric(humidity_pct, errors="coerce")
    heat_index = (
        -42.379
        + 2.04901523 * temp
        + 10.14333127 * humidity
        - 0.22475541 * temp * humidity
        - 0.00683783 * temp**2
        - 0.05481717 * humidity**2
        + 0.00122874 * temp**2 * humidity
        + 0.00085282 * temp * humidity**2
        - 0.00000199 * temp**2 * humidity**2
    )
    return heat_index.where(temp.ge(80) & humidity.ge(40))


def _wind_chill_f(temp_f: pd.Series, wind_speed_mph: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_f, errors="coerce")
    wind_speed = pd.to_numeric(wind_speed_mph, errors="coerce")
    wind_chill = 35.74 + 0.6215 * temp - 35.75 * wind_speed**0.16 + 0.4275 * temp * wind_speed**0.16
    return wind_chill.where(temp.le(50) & wind_speed.gt(3))


def _add_provider_availability_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    high_cols = [HIGH_COLUMNS[provider] for provider in providers]
    for provider, column in HIGH_COLUMNS.items():
        if provider not in providers:
            continue
        if column not in out:
            out[column] = np.nan
        out[f"{provider}_available"] = out[column].notna()
        out[f"{provider}_missing"] = out[column].isna()
    out["provider_count_available"] = out[high_cols].notna().sum(axis=1)
    out["all_provider_highs_available"] = out[high_cols].notna().all(axis=1)
    return out


def _add_provider_time_features(frame: pd.DataFrame, providers: tuple[str, ...], timezone: str) -> pd.DataFrame:
    out = frame.copy()
    tz = ZoneInfo(timezone) if timezone else ZoneInfo("UTC")
    for provider in providers:
        issued_col = f"{provider}_issued_at"
        as_of_col = f"{provider}_forecast_as_of"
        if issued_col not in out:
            continue
        issued = pd.to_datetime(out[issued_col], errors="coerce", utc=True)
        as_of = pd.to_datetime(out.get(as_of_col), errors="coerce", utc=True)
        out[f"{provider}_issue_hour_utc"] = issued.dt.hour
        out[f"{provider}_issue_hour_local"] = issued.dt.tz_convert(tz).dt.hour
        out[f"{provider}_as_of_hour_local"] = as_of.dt.tz_convert(tz).dt.hour
        out[f"{provider}_forecast_lead_hours"] = (as_of - issued).dt.total_seconds() / 3600
    return out


def _add_ensemble_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    high_cols = [HIGH_COLUMNS[provider] for provider in providers]
    highs = out[high_cols]
    out["provider_mean_high_f"] = highs.mean(axis=1)
    out["provider_median_high_f"] = highs.median(axis=1)
    out["provider_min_high_f"] = highs.min(axis=1)
    out["provider_max_high_f"] = highs.max(axis=1)
    out["provider_spread_high_f"] = out["provider_max_high_f"] - out["provider_min_high_f"]
    out["provider_std_high_f"] = highs.std(axis=1)
    ranks = highs.rank(axis=1, method="average", ascending=True)
    for provider, column in HIGH_COLUMNS.items():
        if provider not in providers:
            continue
        out[f"{provider}_minus_provider_mean_high_f"] = out[column] - out["provider_mean_high_f"]
        out[f"{provider}_rank_high"] = ranks[column]
        out[f"{provider}_is_warmest"] = out[column].eq(out["provider_max_high_f"])
        out[f"{provider}_is_coldest"] = out[column].eq(out["provider_min_high_f"])
    return out


def _add_forecast_shape_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for provider in providers:
        hour_min = out.get(f"{provider}_forecast_hour_min")
        hour_max = out.get(f"{provider}_forecast_hour_max")
        if hour_min is not None and hour_max is not None:
            out[f"{provider}_forecast_window_hours"] = hour_max - hour_min + 1
        if f"{provider}_cloud_cover_max" in out and f"{provider}_cloud_cover_mean" in out:
            out[f"{provider}_cloud_variability"] = out[f"{provider}_cloud_cover_max"] - out[f"{provider}_cloud_cover_mean"]
        if f"{provider}_wind_speed_max" in out and f"{provider}_wind_speed_mean" in out:
            out[f"{provider}_wind_gustiness"] = out[f"{provider}_wind_speed_max"] - out[f"{provider}_wind_speed_mean"]
        if HIGH_COLUMNS[provider] in out and f"{provider}_dewpoint_mean_f" in out:
            out[f"{provider}_dewpoint_depression_f"] = out[HIGH_COLUMNS[provider]] - out[f"{provider}_dewpoint_mean_f"]
        if f"{provider}_wind_direction_mean" in out:
            radians = 2 * math.pi * pd.to_numeric(out[f"{provider}_wind_direction_mean"], errors="coerce") / 360
            out[f"{provider}_wind_dir_sin"] = np.sin(radians)
            out[f"{provider}_wind_dir_cos"] = np.cos(radians)
    return out


def _add_provider_cross_model_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    if len(providers) < 2:
        return out
    provider_pairs = [(left, right) for index, left in enumerate(providers) for right in providers[index + 1 :]]
    feature_map = {
        "high_f": lambda provider: HIGH_COLUMNS[provider],
        "dewpoint_mean_f": lambda provider: f"{provider}_dewpoint_mean_f",
        "humidity_mean": lambda provider: f"{provider}_humidity_mean",
        "wind_speed_mean": lambda provider: f"{provider}_wind_speed_mean",
        "wind_speed_max": lambda provider: f"{provider}_wind_speed_max",
        "wind_gust_max": lambda provider: f"{provider}_wind_gust_max",
        "precip_amount": lambda provider: f"{provider}_precip_amount",
        "forecast_precip_total_mm": lambda provider: f"{provider}_forecast_precip_total_mm",
        "forecast_precip_max_1h_mm": lambda provider: f"{provider}_forecast_precip_max_1h_mm",
        "forecast_precip_hours_count": lambda provider: f"{provider}_forecast_precip_hours_count",
        "forecast_precip_intensity_code": lambda provider: f"{provider}_forecast_precip_intensity_code",
        "grid_dist_km_mean": lambda provider: f"{provider}_grid_dist_km_mean",
        "forecast_temp_at_as_of_f": lambda provider: f"{provider}_forecast_temp_at_as_of_f",
        "cloud_cover_mean": lambda provider: f"{provider}_cloud_cover_mean",
        "cloud_cover_max": lambda provider: f"{provider}_cloud_cover_max",
        "low_cloud_cover_mean": lambda provider: f"{provider}_low_cloud_cover_mean",
        "visibility_mean": lambda provider: f"{provider}_visibility_mean",
        "ceiling_min": lambda provider: f"{provider}_ceiling_min",
        "pressure_mslp_mean": lambda provider: f"{provider}_pressure_mslp_mean",
        "shortwave_radiation_mean_w_m2": lambda provider: f"{provider}_shortwave_radiation_mean_w_m2",
    }
    for left, right in provider_pairs:
        prefix = f"{left}_{right}"
        for feature_name, column_for in feature_map.items():
            left_col = column_for(left)
            right_col = column_for(right)
            if left_col not in out or right_col not in out:
                continue
            left_values = pd.to_numeric(out[left_col], errors="coerce")
            right_values = pd.to_numeric(out[right_col], errors="coerce")
            out[f"{prefix}_{feature_name}_diff_f"] = left_values - right_values
            out[f"{prefix}_{feature_name}_abs_diff_f"] = (left_values - right_values).abs()
    return out


def _add_lagged_actual_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    actual = pd.to_numeric(out[TARGET], errors="coerce")
    shifted = actual.shift(1)
    out["actual_high_lag_1d"] = shifted
    out["actual_high_lag_2d"] = actual.shift(2)
    out["actual_high_lag_3d"] = actual.shift(3)
    out["actual_high_trend_1d"] = out["actual_high_lag_1d"] - out["actual_high_lag_2d"]
    out["actual_high_trend_3d"] = out["actual_high_lag_1d"] - actual.shift(4)
    for window in (3, 7, 14, 30):
        out[f"actual_high_roll_{window}d_mean"] = shifted.rolling(window, min_periods=1).mean()
        out[f"actual_high_roll_{window}d_std"] = shifted.rolling(window, min_periods=2).std()
        out[f"actual_high_roll_{window}d_min"] = shifted.rolling(window, min_periods=1).min()
        out[f"actual_high_roll_{window}d_max"] = shifted.rolling(window, min_periods=1).max()
    return out


def _add_lagged_provider_error_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    actual = pd.to_numeric(out[TARGET], errors="coerce")
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col not in out:
            continue
        error = actual - pd.to_numeric(out[high_col], errors="coerce")
        shifted_error = error.shift(1)
        shifted_abs_error = error.abs().shift(1)
        out[f"{provider}_error_lag_1d_f"] = shifted_error
        out[f"{provider}_abs_error_lag_1d_f"] = shifted_abs_error
        out[f"{provider}_rolling_bias_7d_f"] = shifted_error.rolling(7, min_periods=2).mean()
        out[f"{provider}_rolling_bias_14d_f"] = shifted_error.rolling(14, min_periods=3).mean()
        out[f"{provider}_rolling_bias_30d_f"] = shifted_error.rolling(30, min_periods=5).mean()
        out[f"{provider}_rolling_mae_7d_f"] = shifted_abs_error.rolling(7, min_periods=2).mean()
        out[f"{provider}_rolling_mae_14d_f"] = shifted_abs_error.rolling(14, min_periods=3).mean()
        out[f"{provider}_rolling_mae_30d_f"] = shifted_abs_error.rolling(30, min_periods=5).mean()
        out[f"{provider}_high_plus_rolling_bias_7d_f"] = pd.to_numeric(out[high_col], errors="coerce") + out[f"{provider}_rolling_bias_7d_f"]
        out[f"{provider}_high_plus_rolling_bias_14d_f"] = pd.to_numeric(out[high_col], errors="coerce") + out[f"{provider}_rolling_bias_14d_f"]
        out[f"{provider}_high_plus_rolling_bias_30d_f"] = pd.to_numeric(out[high_col], errors="coerce") + out[f"{provider}_rolling_bias_30d_f"]
    return out


def _add_prior_month_provider_error_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    actual = pd.to_numeric(out[TARGET], errors="coerce")
    month = pd.to_numeric(out.get("month"), errors="coerce")
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col not in out:
            continue
        error = actual - pd.to_numeric(out[high_col], errors="coerce")
        abs_error = error.abs()
        prior_month_bias = error.groupby(month, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=2).mean()
        )
        prior_month_mae = abs_error.groupby(month, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=2).mean()
        )
        prior_month_count = error.groupby(month, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=1).count()
        )
        out[f"{provider}_prior_month_bias_f"] = prior_month_bias
        out[f"{provider}_prior_month_mae_f"] = prior_month_mae
        out[f"{provider}_prior_month_error_count"] = prior_month_count
        out[f"{provider}_high_plus_prior_month_bias_f"] = pd.to_numeric(out[high_col], errors="coerce") + prior_month_bias
    return out


def _add_forecast_history_delta_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col not in out:
            continue
        out[f"{provider}_minus_actual_high_lag_1d_f"] = out[high_col] - out["actual_high_lag_1d"]
        out[f"{provider}_minus_actual_high_roll_7d_mean_f"] = out[high_col] - out["actual_high_roll_7d_mean"]
        out[f"{provider}_minus_actual_high_roll_30d_mean_f"] = out[high_col] - out["actual_high_roll_30d_mean"]
    return out


def _add_observation_history_delta_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "observed_temp_at_as_of_f" not in out:
        return out
    observed_temp = pd.to_numeric(out["observed_temp_at_as_of_f"], errors="coerce")
    observed_high_temp = pd.to_numeric(out.get("observed_high_temp_through_as_of_f"), errors="coerce")
    out["observed_temp_minus_actual_high_lag_1d_f"] = observed_temp - out.get("actual_high_lag_1d")
    out["observed_temp_minus_actual_high_roll_7d_mean_f"] = observed_temp - out.get("actual_high_roll_7d_mean")
    out["observed_temp_minus_actual_high_roll_30d_mean_f"] = observed_temp - out.get("actual_high_roll_30d_mean")
    out["observed_high_temp_minus_actual_high_lag_1d_f"] = observed_high_temp - out.get("actual_high_lag_1d")
    out["observed_high_temp_minus_actual_high_roll_7d_mean_f"] = observed_high_temp - out.get("actual_high_roll_7d_mean")
    out["observed_high_temp_minus_actual_high_roll_30d_mean_f"] = observed_high_temp - out.get("actual_high_roll_30d_mean")
    if "observed_dewpoint_at_as_of_f" in out:
        dewpoint = pd.to_numeric(out["observed_dewpoint_at_as_of_f"], errors="coerce")
        out["observed_dewpoint_minus_actual_high_roll_7d_mean_f"] = dewpoint - out.get("actual_high_roll_7d_mean")
    return out


def _add_observation_forecast_delta_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    observed_temp = pd.to_numeric(out.get("observed_temp_at_as_of_f"), errors="coerce")
    observed_high_temp = pd.to_numeric(out.get("observed_high_temp_through_as_of_f"), errors="coerce")
    observed_dewpoint = pd.to_numeric(out.get("observed_dewpoint_at_as_of_f"), errors="coerce")
    observed_humidity = pd.to_numeric(out.get("observed_humidity_at_as_of"), errors="coerce")
    observed_wind = pd.to_numeric(out.get("observed_wind_speed_at_as_of"), errors="coerce")
    observed_pressure = pd.to_numeric(out.get("observed_pressure_at_as_of"), errors="coerce")
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col in out:
            out[f"{provider}_high_minus_observed_temp_f"] = pd.to_numeric(out[high_col], errors="coerce") - observed_temp
            out[f"{provider}_high_minus_observed_high_temp_f"] = (
                pd.to_numeric(out[high_col], errors="coerce") - observed_high_temp
            )
        if f"{provider}_forecast_temp_at_as_of_f" in out:
            out[f"{provider}_forecast_temp_at_as_of_minus_observed_temp_f"] = (
                pd.to_numeric(out[f"{provider}_forecast_temp_at_as_of_f"], errors="coerce") - observed_temp
            )
        if f"{provider}_dewpoint_mean_f" in out:
            out[f"{provider}_dewpoint_minus_observed_dewpoint_f"] = (
                pd.to_numeric(out[f"{provider}_dewpoint_mean_f"], errors="coerce") - observed_dewpoint
            )
        if f"{provider}_humidity_mean" in out:
            out[f"{provider}_humidity_minus_observed_humidity"] = (
                pd.to_numeric(out[f"{provider}_humidity_mean"], errors="coerce") - observed_humidity
            )
        if f"{provider}_wind_speed_mean" in out:
            out[f"{provider}_wind_speed_minus_observed_wind_speed"] = (
                pd.to_numeric(out[f"{provider}_wind_speed_mean"], errors="coerce") - observed_wind
            )
        if f"{provider}_pressure_mslp_mean" in out:
            out[f"{provider}_pressure_minus_observed_pressure"] = (
                pd.to_numeric(out[f"{provider}_pressure_mslp_mean"], errors="coerce") - observed_pressure
            )
    return out


def add_versioned_feature_engineering(
    frame: pd.DataFrame,
    *,
    feature_version: str = "base",
    providers: tuple[str, ...] = TARGET_PROVIDERS,
) -> pd.DataFrame:
    version = _normalize_feature_version(feature_version)
    if version == "base":
        return frame
    out = add_v5_feature_engineering(frame, providers=providers)
    if version in {"v8", "v9", "v10", "v11", "v12", "v13", "v14", V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION, *V15_FEATURE_VERSIONS, *V16_FEATURE_VERSIONS, *V17_FEATURE_VERSIONS, *V18_FEATURE_VERSIONS, *V18_1_FEATURE_VERSIONS, *V20_PEAK_TIMING_FEATURE_VERSIONS}:
        out = add_v8_feature_engineering(out, providers=providers)
        if version == V11_SETTLEMENT_FIX_TEMP_FEATURE_VERSION:
            return add_v11_settlement_fix_temp_feature_engineering(out, providers=providers)
        if version in V20_PEAK_TIMING_FEATURE_VERSIONS:
            out = add_v11_settlement_fix_temp_feature_engineering(out, providers=providers)
            return add_v20_peak_timing_feature_engineering(out)
        if version in WEATHER_AGGREGATE_FEATURE_VERSIONS:
            return add_v13_feature_engineering(out, providers=providers)
        return out
    if version in {"v6", "v7"}:
        return out
    return out


def add_v5_feature_engineering(frame: pd.DataFrame, providers: tuple[str, ...] = TARGET_PROVIDERS) -> pd.DataFrame:
    out = frame.copy()
    observed_temp = _numeric_series(out, "observed_temp_at_as_of_f")
    high_so_far = _numeric_series(out, "observed_high_temp_through_as_of_f")
    observed_humidity = _numeric_series(out, "observed_humidity_at_as_of")
    provider_mean = _numeric_series(out, "provider_mean_high_f")
    provider_spread = _numeric_series(out, "provider_spread_high_f")
    lag_1d = _numeric_series(out, "actual_high_lag_1d")
    roll_7d = _numeric_series(out, "actual_high_roll_7d_mean")
    roll_30d = _numeric_series(out, "actual_high_roll_30d_mean")

    warmup_to_consensus = provider_mean - observed_temp
    remaining_warmup = provider_mean - high_so_far
    out["v2_recent_heat_anomaly_f"] = lag_1d - roll_30d
    out["v2_recent_heat_momentum_f"] = roll_7d - roll_30d
    out["v2_morning_warmup_to_consensus_f"] = warmup_to_consensus
    out["v2_consensus_minus_7d_actual_f"] = provider_mean - roll_7d
    out["v2_spread_per_warmup_f"] = provider_spread / warmup_to_consensus.abs().clip(lower=1.0)
    out["v2_humidity_warmup_interaction"] = (observed_humidity / 100.0) * warmup_to_consensus
    out["v3_high_so_far_above_current_f"] = high_so_far - observed_temp
    out["v3_remaining_warmup_from_high_so_far_f"] = remaining_warmup
    out["v3_high_so_far_minus_lag_1d_f"] = high_so_far - lag_1d
    out["v3_high_so_far_minus_7d_actual_f"] = high_so_far - roll_7d
    out["v3_remaining_warmup_per_spread_f"] = remaining_warmup / provider_spread.abs().clip(lower=1.0)
    out["v3_humidity_remaining_warmup_interaction"] = (observed_humidity / 100.0) * remaining_warmup

    precip_total = _provider_matrix(out, providers, "forecast_precip_total_mm", fallback="precip_amount")
    precip_max_1h = _provider_matrix(out, providers, "forecast_precip_max_1h_mm")
    precip_hours = _provider_matrix(out, providers, "forecast_precip_hours_count")
    precip_intensity = _provider_matrix(out, providers, "forecast_precip_intensity_code")
    has_precip = _provider_matrix(out, providers, "forecast_has_precip").fillna(0).clip(lower=0, upper=1)

    out["v4_forecast_precip_total_mean_mm"] = precip_total.mean(axis=1)
    out["v4_forecast_precip_total_max_mm"] = precip_total.max(axis=1)
    out["v4_forecast_precip_total_spread_mm"] = precip_total.max(axis=1) - precip_total.min(axis=1)
    out["v4_forecast_precip_max_1h_mean_mm"] = precip_max_1h.mean(axis=1)
    out["v4_forecast_precip_hours_mean"] = precip_hours.mean(axis=1)
    out["v4_forecast_precip_intensity_mean"] = precip_intensity.mean(axis=1)
    out["v4_forecast_precip_intensity_max"] = precip_intensity.max(axis=1)
    out["v4_any_forecast_precip"] = has_precip.max(axis=1).fillna(0).astype(int)
    out["v4_all_forecast_precip"] = has_precip.min(axis=1).fillna(0).astype(int)

    observed_any = (
        _boolean_numeric_series(out, "observed_is_raining_at_as_of")
        | _boolean_numeric_series(out, "observed_is_drizzle_at_as_of")
        | _boolean_numeric_series(out, "observed_is_snowing_at_as_of")
    ).astype(int)
    observed_recent_mm = _numeric_series(out, "observed_precip_recent_at_as_of") * 25.4
    out["v4_observed_precip_any"] = observed_any
    out["v4_observed_precip_recent_mm_est"] = observed_recent_mm
    out["v4_forecast_total_minus_observed_recent_mm"] = out["v4_forecast_precip_total_mean_mm"] - observed_recent_mm
    out["v4_forecast_observed_precip_match"] = out["v4_any_forecast_precip"].eq(observed_any).astype(int)
    out["v4_forecast_wet_observed_dry"] = (out["v4_any_forecast_precip"].eq(1) & observed_any.eq(0)).astype(int)
    out["v4_observed_wet_forecast_dry"] = (observed_any.eq(1) & out["v4_any_forecast_precip"].eq(0)).astype(int)
    out["v4_precip_humidity_interaction"] = out["v4_forecast_precip_total_mean_mm"] * (observed_humidity / 100.0)
    out["v4_precip_remaining_warmup_interaction"] = out["v4_forecast_precip_total_mean_mm"] * remaining_warmup
    return out


def add_v8_feature_engineering(frame: pd.DataFrame, providers: tuple[str, ...] = TARGET_PROVIDERS) -> pd.DataFrame:
    out = frame.copy()
    high_so_far = _numeric_series(out, "observed_high_temp_through_as_of_f")
    provider_mean = _numeric_series(out, "provider_mean_high_f")
    provider_max = _numeric_series(out, "provider_max_high_f")
    provider_min = _numeric_series(out, "provider_min_high_f")
    provider_median = _numeric_series(out, "provider_median_high_f")
    provider_spread = _numeric_series(out, "provider_spread_high_f")
    provider_mean_remaining = provider_mean - high_so_far

    out["v8_provider_max_remaining_from_high_so_far_f"] = provider_max - high_so_far
    out["v8_provider_min_remaining_from_high_so_far_f"] = provider_min - high_so_far
    out["v8_provider_median_remaining_from_high_so_far_f"] = provider_median - high_so_far
    out["v8_provider_spread_per_remaining_warmup_f"] = provider_spread / provider_mean_remaining.abs().clip(lower=1.0)

    actual_remaining = _numeric_series(out, TARGET) - high_so_far
    shifted_remaining = actual_remaining.shift(1)
    month = pd.to_numeric(out.get("month"), errors="coerce")
    out["v8_month_remaining_warmup_mean_f"] = actual_remaining.groupby(month, dropna=False).transform(
        lambda series: series.shift(1).expanding(min_periods=2).mean()
    )
    out["v8_month_remaining_warmup_count"] = actual_remaining.groupby(month, dropna=False).transform(
        lambda series: series.shift(1).expanding(min_periods=1).count()
    )
    out["v8_recent_remaining_warmup_7d_mean_f"] = shifted_remaining.rolling(7, min_periods=2).mean()
    out["v8_recent_remaining_warmup_30d_mean_f"] = shifted_remaining.rolling(30, min_periods=5).mean()
    out["v8_provider_mean_remaining_vs_month_normal_f"] = (
        provider_mean_remaining - out["v8_month_remaining_warmup_mean_f"]
    )

    cloud_mean = _provider_matrix(out, providers, "cloud_cover_mean")
    cloud_max = _provider_matrix(out, providers, "cloud_cover_max")
    precip_total = _provider_matrix(out, providers, "forecast_precip_total_mm", fallback="precip_amount")
    precip_max_1h = _provider_matrix(out, providers, "forecast_precip_max_1h_mm")
    wind_speed = _provider_matrix(out, providers, "wind_speed_mean")
    wind_gust = _provider_matrix(out, providers, "wind_gust_max")
    dewpoint = _provider_matrix(out, providers, "dewpoint_mean_f")

    cloud_mean_avg = cloud_mean.mean(axis=1)
    cloud_max_avg = cloud_max.mean(axis=1)
    precip_total_avg = precip_total.mean(axis=1)
    precip_max_1h_avg = precip_max_1h.mean(axis=1)
    wind_speed_avg = wind_speed.mean(axis=1)
    wind_gust_avg = wind_gust.mean(axis=1)
    dewpoint_avg = dewpoint.mean(axis=1)
    dewpoint_depression = provider_mean - dewpoint_avg

    out["v8_cloud_cover_mean_remaining_warmup_interaction"] = (cloud_mean_avg / 100.0) * provider_mean_remaining
    out["v8_cloud_cover_max_remaining_warmup_interaction"] = (cloud_max_avg / 100.0) * provider_mean_remaining
    out["v8_precip_total_remaining_warmup_interaction"] = precip_total_avg * provider_mean_remaining
    out["v8_precip_max_1h_remaining_warmup_interaction"] = precip_max_1h_avg * provider_mean_remaining
    out["v8_wind_speed_mean_remaining_warmup_interaction"] = wind_speed_avg * provider_mean_remaining
    out["v8_wind_gust_max_remaining_warmup_interaction"] = wind_gust_avg * provider_mean_remaining
    out["v8_forecast_dewpoint_mean_f"] = dewpoint_avg
    out["v8_forecast_dewpoint_depression_mean_f"] = dewpoint_depression
    out["v8_dewpoint_mean_remaining_warmup_interaction"] = dewpoint_avg * provider_mean_remaining
    out["v8_dewpoint_depression_remaining_warmup_interaction"] = dewpoint_depression * provider_mean_remaining
    return out


def add_v13_feature_engineering(frame: pd.DataFrame, providers: tuple[str, ...] = TARGET_PROVIDERS) -> pd.DataFrame:
    out = frame.copy()
    observed_temp = _numeric_series(out, "observed_temp_at_as_of_f")
    high_so_far = _numeric_series(out, "observed_high_temp_through_as_of_f")
    provider_mean = _numeric_series(out, "provider_mean_high_f")
    provider_mean_remaining = provider_mean - high_so_far

    forecast_temp_at_as_of = _provider_matrix(out, providers, "forecast_temp_at_as_of_f")
    cloud_mean = _provider_matrix(out, providers, "cloud_cover_mean")
    cloud_max = _provider_matrix(out, providers, "cloud_cover_max")
    low_cloud_mean = _provider_matrix(out, providers, "low_cloud_cover_mean")
    precip_total = _provider_matrix(out, providers, "forecast_precip_total_mm", fallback="precip_amount")
    visibility = _provider_matrix(out, providers, "visibility_mean")
    ceiling = _provider_matrix(out, providers, "ceiling_min")
    pressure = _provider_matrix(out, providers, "pressure_mslp_mean")
    shortwave = _provider_matrix(
        out,
        providers,
        "downward_shortwave_radiation_mean_w_m2",
        fallback="shortwave_radiation_mean_w_m2",
    )

    forecast_temp_mean = forecast_temp_at_as_of.mean(axis=1)
    forecast_temp_spread = forecast_temp_at_as_of.max(axis=1) - forecast_temp_at_as_of.min(axis=1)
    cloud_mean_avg = cloud_mean.mean(axis=1)
    cloud_max_avg = cloud_max.mean(axis=1)
    low_cloud_avg = low_cloud_mean.mean(axis=1)
    precip_total_avg = precip_total.mean(axis=1)
    visibility_min = visibility.min(axis=1)
    ceiling_min = ceiling.min(axis=1)

    out["v13_forecast_temp_at_as_of_mean_f"] = forecast_temp_mean
    out["v13_forecast_temp_at_as_of_minus_observed_mean_f"] = forecast_temp_mean - observed_temp
    out["v13_forecast_temp_at_as_of_spread_f"] = forecast_temp_spread
    out["v13_cloud_cover_mean_pct"] = cloud_mean_avg
    out["v13_cloud_cover_max_pct"] = cloud_max_avg
    out["v13_low_cloud_cover_mean_pct"] = low_cloud_avg
    out["v13_visibility_mean_m"] = visibility.mean(axis=1)
    out["v13_ceiling_min_m"] = ceiling_min
    out["v13_low_visibility_flag"] = visibility_min.lt(5000).where(visibility_min.notna()).astype("Int64")
    out["v13_low_ceiling_flag"] = ceiling_min.lt(1500).where(ceiling_min.notna()).astype("Int64")
    out["v13_pressure_mslp_mean_pa"] = pressure.mean(axis=1)
    out["v13_shortwave_mean_w_m2"] = shortwave.mean(axis=1)
    out["v13_weather_available_provider_count"] = pd.concat(
        [
            forecast_temp_at_as_of.notna().sum(axis=1),
            cloud_mean.notna().sum(axis=1),
            precip_total.notna().sum(axis=1),
            visibility.notna().sum(axis=1),
            ceiling.notna().sum(axis=1),
        ],
        axis=1,
    ).max(axis=1)
    out["v13_cloud_cover_remaining_warmup_interaction"] = (cloud_mean_avg / 100.0) * provider_mean_remaining
    out["v13_low_cloud_remaining_warmup_interaction"] = (low_cloud_avg / 100.0) * provider_mean_remaining
    out["v13_precip_cloud_remaining_warmup_interaction"] = (
        precip_total_avg * (cloud_mean_avg / 100.0) * provider_mean_remaining
    )
    out["v13_forecast_temp_bias_remaining_warmup_interaction"] = (
        out["v13_forecast_temp_at_as_of_minus_observed_mean_f"] * provider_mean_remaining
    )
    return out


def add_v11_settlement_fix_temp_feature_engineering(
    frame: pd.DataFrame,
    providers: tuple[str, ...] = TARGET_PROVIDERS,
) -> pd.DataFrame:
    """Add live-safe 11 AM forecast/observation temperature alignment features."""
    out = frame.copy()
    observed_temp = _numeric_series(out, "observed_temp_at_as_of_f")
    high_so_far = _numeric_series(out, "observed_high_temp_through_as_of_f")
    provider_mean_high = _numeric_series(out, "provider_mean_high_f")
    forecast_temps = _provider_matrix(out, providers, "forecast_temp_at_as_of_f")

    provider_count = forecast_temps.notna().sum(axis=1).astype("int64")
    forecast_mean = forecast_temps.mean(axis=1, skipna=True).where(provider_count.gt(0))
    forecast_median = forecast_temps.median(axis=1, skipna=True).where(provider_count.gt(0))
    forecast_spread = (forecast_temps.max(axis=1, skipna=True) - forecast_temps.min(axis=1, skipna=True)).where(
        provider_count.gt(0)
    )
    signed_delta = forecast_mean - observed_temp
    remaining_warmup = provider_mean_high - high_so_far

    out["v11sf_forecast_temp_11am_mean_f"] = forecast_mean
    out["v11sf_forecast_temp_11am_median_f"] = forecast_median
    out["v11sf_forecast_temp_11am_minus_observed_f"] = signed_delta
    out["v11sf_forecast_temp_11am_abs_error_f"] = signed_delta.abs()
    out["v11sf_forecast_temp_11am_warm_error_f"] = signed_delta.clip(lower=0)
    out["v11sf_forecast_temp_11am_cool_error_f"] = (-signed_delta).clip(lower=0)
    out["v11sf_forecast_temp_11am_spread_f"] = forecast_spread
    out["v11sf_forecast_temp_11am_provider_count"] = provider_count
    out["v11sf_forecast_temp_bias_remaining_warmup_interaction"] = signed_delta * remaining_warmup
    out["v11sf_observation_adjusted_provider_high_f"] = provider_mean_high - signed_delta
    out["v11sf_forecast_warmup_after_11am_f"] = provider_mean_high - forecast_mean
    return out


def add_v20_peak_timing_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Add curated, live-safe HRRR/NBM afternoon-peak features."""
    out = frame.copy()
    observed_temp = _numeric_series(out, "observed_temp_at_as_of_f")
    hrrr_t11 = _numeric_series(out, "hrrr_t11l_f")
    nbm_t11 = _numeric_series(out, "nbm_t11l_f")
    hrrr_max = _numeric_series(out, "hrrr_max_post11_f")
    nbm_max = _numeric_series(out, "nbm_max_post11_f")
    hrrr_peak_hour = _numeric_series(out, "hrrr_hour_of_max_local")
    nbm_peak_hour = _numeric_series(out, "nbm_hour_of_max_local")

    out["v20_hrrr_t11_minus_observed_f"] = hrrr_t11 - observed_temp
    out["v20_nbm_t11_minus_observed_f"] = nbm_t11 - observed_temp
    out["v20_hrrr_remaining_rise_f"] = hrrr_max - hrrr_t11
    out["v20_nbm_remaining_rise_f"] = nbm_max - nbm_t11
    out["v20_hrrr_observation_adjusted_high_f"] = observed_temp + out["v20_hrrr_remaining_rise_f"]
    out["v20_nbm_observation_adjusted_high_f"] = observed_temp + out["v20_nbm_remaining_rise_f"]
    adjusted = out[["v20_hrrr_observation_adjusted_high_f", "v20_nbm_observation_adjusted_high_f"]]
    adjusted_count = adjusted.notna().sum(axis=1)
    out["v20_adjusted_high_mean_f"] = adjusted.mean(axis=1, skipna=True).where(adjusted_count.gt(0))
    out["v20_adjusted_high_spread_f"] = (
        adjusted.max(axis=1, skipna=True) - adjusted.min(axis=1, skipna=True)
    ).where(adjusted_count.gt(0))
    out["v20_model_high_difference_f"] = hrrr_max - nbm_max
    out["v20_peak_hour_difference"] = hrrr_peak_hour - nbm_peak_hour

    early_solar = [_numeric_series(out, f"hrrr_dswrf_{hour}l_w_m2") for hour in range(11, 15)]
    late_solar = [_numeric_series(out, f"hrrr_dswrf_{hour}l_w_m2") for hour in range(15, 19)]
    out["v20_solar_energy_11_14_wh_m2"] = pd.concat(early_solar, axis=1).sum(axis=1, min_count=4)
    out["v20_solar_energy_15_18_wh_m2"] = pd.concat(late_solar, axis=1).sum(axis=1, min_count=4)

    tcc_11 = _numeric_series(out, "hrrr_tcc_11l_pct")
    out["v20_tcc_change_11_to_hrrr_peak_pct"] = _v20_hourly_value(
        out, "hrrr_tcc_{hour}l_pct", hrrr_peak_hour
    ) - tcc_11
    out["v20_tcc_change_11_to_nbm_peak_pct"] = _v20_hourly_value(
        out, "hrrr_tcc_{hour}l_pct", nbm_peak_hour
    ) - tcc_11

    wet_hrrr = _numeric_series(out, "hrrr_precip_wet_hours_11_to_hrrr_peak")
    wet_nbm = _numeric_series(out, "hrrr_precip_wet_hours_11_to_nbm_peak")
    no_precip = _numeric_series(out, "hrrr_no_precip_11_18")
    rain_present = no_precip.eq(0).where(no_precip.notna())
    out["v20_rain_before_hrrr_peak"] = wet_hrrr.gt(0).where(wet_hrrr.notna()).astype("Int64")
    out["v20_rain_before_nbm_peak"] = wet_nbm.gt(0).where(wet_nbm.notna()).astype("Int64")
    out["v20_rain_present_11_18"] = rain_present.astype("Int64")
    out["v20_precip_onset_minus_hrrr_peak_hours_zero_filled"] = _numeric_series(
        out, "hrrr_precip_onset_minus_hrrr_peak_hours"
    ).fillna(0.0)
    out["v20_precip_onset_minus_nbm_peak_hours_zero_filled"] = _numeric_series(
        out, "hrrr_precip_onset_minus_nbm_peak_hours"
    ).fillna(0.0)
    return out


def _v20_hourly_value(frame: pd.DataFrame, column_template: str, hours: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    rounded_hours = pd.to_numeric(hours, errors="coerce").round().astype("Int64")
    for hour in range(11, 19):
        column = column_template.format(hour=hour)
        if column in frame:
            result = result.where(~rounded_hours.eq(hour), pd.to_numeric(frame[column], errors="coerce"))
    return result


def add_v9_climatology_features(
    frame: pd.DataFrame,
    *,
    project_root: str | Path = ".",
    station_id: str | None = None,
    climatology_normals_path: str | Path | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    normals_path = _resolve_v9_climatology_normals_path(project_root, climatology_normals_path)
    normals = _load_v9_climatology_normals(normals_path)
    station = str(station_id or "").upper()
    if not station and "station_code" in out and out["station_code"].notna().any():
        station = str(out["station_code"].dropna().astype("string").iloc[0]).upper()

    dates = pd.to_datetime(out["contract_date"], errors="coerce")
    station_codes = (
        out["station_code"].astype("string").str.upper()
        if "station_code" in out
        else pd.Series(station, index=out.index, dtype="string")
    )
    join_keys = pd.DataFrame(
        {
            "_v9_row_id": out.index,
            "station_code": station_codes.fillna(station),
            "target_year": dates.dt.year.astype("Int64"),
            "month_day": dates.dt.strftime("%m-%d"),
        },
        index=out.index,
    )
    normal_columns = [
        "station_code",
        "target_year",
        "month_day",
        "climatology_high_10y_f",
        "climatology_high_10y_std_f",
        "climatology_high_10y_count",
        "climatology_source_start_year",
        "climatology_source_end_year",
    ]
    joined = join_keys.merge(normals[normal_columns], on=["station_code", "target_year", "month_day"], how="left")
    joined = joined.set_index("_v9_row_id").reindex(out.index)

    for column in normal_columns[3:]:
        out[column] = joined[column]

    climatology = pd.to_numeric(out["climatology_high_10y_f"], errors="coerce")
    out["provider_mean_minus_climatology_10y_f"] = _numeric_series(out, "provider_mean_high_f") - climatology
    out["observed_temp_minus_climatology_10y_f"] = _numeric_series(out, "observed_temp_at_as_of_f") - climatology
    out["observed_high_so_far_minus_climatology_10y_f"] = (
        _numeric_series(out, "observed_high_temp_through_as_of_f") - climatology
    )
    out["actual_minus_climatology_10y_f_DIAGNOSTIC_ONLY"] = _numeric_series(out, TARGET) - climatology
    return out


def _resolve_v9_climatology_normals_path(
    project_root: str | Path,
    climatology_normals_path: str | Path | None,
) -> Path:
    if climatology_normals_path is not None:
        path = Path(climatology_normals_path)
        return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()
    root = Path(project_root)
    candidates = [
        root / "data" / "calibration" / "station_stacking_v9" / "station_rolling_10y_daily_high_normals.csv",
        root / "outputs" / "climatology_all_stations" / "station_rolling_10y_daily_high_normals.csv",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "V9 climatology normals not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_v9_climatology_normals(path: Path) -> pd.DataFrame:
    normals = pd.read_csv(path)
    required = {
        "station_code",
        "target_year",
        "month_day",
        "climatology_high_10y_f",
        "climatology_high_10y_std_f",
        "climatology_high_10y_count",
        "climatology_source_start_year",
        "climatology_source_end_year",
    }
    missing = sorted(required - set(normals.columns))
    if missing:
        raise ValueError(f"V9 climatology normals missing required columns: {', '.join(missing)}")
    normals = normals.copy()
    normals["station_code"] = normals["station_code"].astype("string").str.upper()
    normals["target_year"] = pd.to_numeric(normals["target_year"], errors="coerce").astype("Int64")
    normals["month_day"] = normals["month_day"].astype("string")
    for column in required - {"station_code", "target_year", "month_day"}:
        normals[column] = pd.to_numeric(normals[column], errors="coerce")
    return normals


def _normalize_feature_version(feature_version: str) -> str:
    version = str(feature_version or "base").strip().lower()
    if version in {"", "none"}:
        version = "base"
    if version not in SUPPORTED_FEATURE_VERSIONS:
        raise ValueError(
            "feature_version must be one of: "
            + ", ".join(f"'{item}'" for item in SUPPORTED_FEATURE_VERSIONS)
        )
    return version


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _boolean_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0, index=frame.index, dtype="int64")
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).gt(0).astype(int)
    return series.astype("string").str.lower().isin({"1", "true", "yes", "y"}).astype(int)


def _provider_matrix(
    frame: pd.DataFrame,
    providers: tuple[str, ...],
    primary: str,
    fallback: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {provider: _provider_feature_series(frame, provider, primary, fallback) for provider in providers},
        index=frame.index,
    )


def _provider_feature_series(
    frame: pd.DataFrame,
    provider: str,
    primary: str,
    fallback: str | None,
) -> pd.Series:
    primary_column = f"{provider}_{primary}"
    if primary_column in frame:
        return _numeric_series(frame, primary_column)
    if fallback is not None:
        return _numeric_series(frame, f"{provider}_{fallback}")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["contract_date", "method", "evaluation_scope", TARGET, "predicted_high_f"]].copy()
    out["error_f"] = out[TARGET] - out["predicted_high_f"]
    out["absolute_error_f"] = out["error_f"].abs()
    return out


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["contract_date", "method", "evaluation_scope", TARGET, "predicted_high_f", "error_f", "absolute_error_f"]
    )


def _walk_forward_best_raw_provider(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    high_cols = [HIGH_COLUMNS[provider] for provider in config.providers]
    required = [TARGET, *high_cols]
    complete = frame.dropna(subset=required).sort_values("contract_date").reset_index(drop=True)
    if len(complete) <= config.effective_min_train_rows:
        return _empty_predictions()
    rows: list[pd.DataFrame] = []
    dates = sorted(complete["contract_date"].unique())
    completed_blocks = 0
    for start_idx in range(0, len(dates), config.effective_refit_days):
        block_start = dates[start_idx]
        block_dates = dates[start_idx : start_idx + config.effective_refit_days]
        train = complete.loc[complete["contract_date"] < block_start]
        valid = complete.loc[complete["contract_date"].isin(block_dates)].copy()
        if len(train) < config.effective_min_train_rows or valid.empty:
            continue
        mae_by_provider = {
            provider: float((train[TARGET] - train[HIGH_COLUMNS[provider]]).abs().mean())
            for provider in config.providers
        }
        best_provider = min(mae_by_provider, key=mae_by_provider.get)
        pred = valid[["contract_date", TARGET]].copy()
        pred["method"] = "best_raw_provider"
        pred["predicted_high_f"] = valid[HIGH_COLUMNS[best_provider]]
        pred["evaluation_scope"] = "complete_provider_walk_forward"
        rows.append(_prediction_columns(pred))
        completed_blocks += 1
        if config.fast_mode and completed_blocks >= config.fast_max_validation_blocks:
            break
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def _modeling_frame(frame: pd.DataFrame, config: StationStackingConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = _ensure_model_target_columns(_with_actual_quality_columns(frame, config), config)
    frame = add_strict_quality_flags(frame, providers=config.providers)
    categorical, numeric = feature_columns(frame, config)
    frame = _drop_missing_model_target(frame, config)
    if frame.empty:
        return frame, categorical, numeric
    required = [HIGH_COLUMNS[provider] for provider in config.providers]
    clean = frame.dropna(subset=required).loc[frame["all_provider_highs_available"].fillna(False)].copy()
    clean = clean.loc[clean[STRICT_QUALITY_OK_COLUMN].fillna(False)].copy()
    clean = clean.sort_values("contract_date").reset_index(drop=True)
    numeric = [column for column in numeric if column in clean and clean[column].notna().any()]
    categorical = [column for column in categorical if column in clean]
    return clean, categorical, numeric


def _ensure_model_target_columns(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    out = frame.copy()
    if config.effective_target_mode != TARGET_MODE_REMAINING_WARMUP:
        return out
    actual = pd.to_numeric(out.get(TARGET), errors="coerce")
    high_so_far = pd.to_numeric(out.get(OBSERVED_HIGH_SO_FAR_COLUMN), errors="coerce")
    out[REMAINING_WARMUP_TARGET] = actual - high_so_far
    return out


def _drop_missing_model_target(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    required = _unique_columns([TARGET, *_model_target_required_columns(config)])
    if any(column not in frame for column in required):
        return frame.iloc[0:0].copy()
    return frame.dropna(subset=required).copy()


def _model_target_column(config: StationStackingConfig) -> str:
    if config.effective_target_mode == TARGET_MODE_REMAINING_WARMUP:
        return REMAINING_WARMUP_TARGET
    return TARGET


def _model_target_required_columns(config: StationStackingConfig) -> list[str]:
    if config.effective_target_mode == TARGET_MODE_REMAINING_WARMUP:
        return [OBSERVED_HIGH_SO_FAR_COLUMN, REMAINING_WARMUP_TARGET]
    return [TARGET]


def _model_target_values(frame: pd.DataFrame, config: StationStackingConfig) -> pd.Series:
    frame = _ensure_model_target_columns(frame, config)
    return pd.to_numeric(frame[_model_target_column(config)], errors="coerce")


def _prediction_output_to_high(predicted: Any, frame: pd.DataFrame, config: StationStackingConfig) -> np.ndarray:
    predicted_array = np.asarray(predicted, dtype=float)
    if config.effective_target_mode != TARGET_MODE_REMAINING_WARMUP:
        return predicted_array
    high_so_far = pd.to_numeric(frame[OBSERVED_HIGH_SO_FAR_COLUMN], errors="coerce").to_numpy(dtype=float)
    predicted_high = high_so_far + predicted_array
    return np.maximum(predicted_high, high_so_far)


def _unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _build_base_model_pipelines(
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
) -> dict[str, Any]:
    return {
        method: _build_base_model_pipeline(config, categorical, numeric, method, params={})
        for method in config.effective_base_model_methods
    }


def _build_base_model_pipeline(
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    method: str,
    params: dict[str, Any],
):
    try:
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError("Station stacking notebooks need scikit-learn.") from exc

    return Pipeline(
        [
            ("prep", _build_preprocessor(categorical, numeric)),
            ("model", _build_base_model_estimator(config, method, params)),
        ]
    )


def _build_preprocessor(categorical: list[str], numeric: list[str]):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise ImportError("Station stacking notebooks need scikit-learn.") from exc

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )


def _build_base_model_estimator(
    config: StationStackingConfig,
    method: str,
    params: dict[str, Any],
    early_stopping_rounds: int | None = None,
):
    n_estimators = int(params.get("n_estimators", 120 if config.fast_mode else 900))
    cat_iterations = int(params.get("iterations", 120 if config.fast_mode else 900))
    if method == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Station stacking xgboost models need xgboost.") from exc
        robust_objective = config.effective_feature_version in HUBER_STACK_FEATURE_VERSIONS
        estimator_params = {
            "objective": "reg:pseudohubererror" if robust_objective else "reg:squarederror",
            "n_estimators": n_estimators,
            "learning_rate": float(params.get("learning_rate", 0.035)),
            "max_depth": int(params.get("max_depth", 3)),
            "min_child_weight": float(params.get("min_child_weight", 1.0)),
            "gamma": float(params.get("gamma", 0.0)),
            "subsample": float(params.get("subsample", 0.9)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.9)),
            "reg_alpha": float(params.get("reg_alpha", 0.0)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "random_state": config.random_state,
            "n_jobs": -1,
            "eval_metric": "mae" if robust_objective else "rmse",
        }
        if early_stopping_rounds is not None:
            estimator_params["early_stopping_rounds"] = early_stopping_rounds
        return XGBRegressor(**estimator_params)
    if method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Station stacking lightgbm models need lightgbm.") from exc
        estimator_params = {
            "n_estimators": n_estimators,
            "learning_rate": float(params.get("learning_rate", 0.035)),
            "num_leaves": int(params.get("num_leaves", 31)),
            "max_depth": int(params.get("max_depth", -1)),
            "min_child_samples": int(params.get("min_child_samples", 20)),
            "min_split_gain": float(params.get("min_split_gain", 0.0)),
            "bagging_fraction": float(params.get("bagging_fraction", params.get("subsample", 0.9))),
            "bagging_freq": int(params.get("bagging_freq", 1)),
            "feature_fraction": float(params.get("feature_fraction", params.get("colsample_bytree", 0.9))),
            "lambda_l1": float(params.get("lambda_l1", params.get("reg_alpha", 0.0))),
            "lambda_l2": float(params.get("lambda_l2", params.get("reg_lambda", 0.0))),
            "random_state": config.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        if config.effective_feature_version in HUBER_STACK_FEATURE_VERSIONS:
            estimator_params["objective"] = "huber"
            estimator_params["metric"] = "mae"
            estimator_params["alpha"] = float(params.get("huber_alpha", 0.9))
        return LGBMRegressor(**estimator_params)
    if method == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("Station stacking catboost models need catboost.") from exc
        cat_params = {
            "iterations": cat_iterations,
            "learning_rate": float(params.get("learning_rate", 0.035)),
            "depth": int(params.get("depth", 6)),
            "l2_leaf_reg": float(params.get("l2_leaf_reg", 3.0)),
            "random_strength": float(params.get("random_strength", 1.0)),
            "bagging_temperature": float(params.get("bagging_temperature", 1.0)),
            "border_count": int(params.get("border_count", 128)),
            "rsm": float(params.get("rsm", 1.0)),
            "bootstrap_type": "Bayesian",
            "loss_function": str(params.get("loss_function", "RMSE")),
            "random_seed": config.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        if config.effective_feature_version in CATBOOST_HUBER_FEATURE_VERSIONS:
            huber_delta = float(params.get("huber_delta", 1.0))
            cat_params["loss_function"] = f"Huber:delta={huber_delta:g}"
            cat_params["eval_metric"] = "MAE"
        elif "eval_metric" in params:
            cat_params["eval_metric"] = str(params["eval_metric"])
        return CatBoostRegressor(**cat_params)
    raise ValueError(f"Unknown base model method: {method}")


def _fit_predict_base_model(
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    method: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    early_stopping: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = _ensure_model_target_columns(train, config)
    valid = _ensure_model_target_columns(valid, config)
    fit_categorical, fit_numeric = _fit_feature_columns(
        train,
        categorical,
        numeric,
        max_missing_fraction=config.effective_max_feature_missing_fraction,
    )
    feature_names = [*fit_categorical, *fit_numeric]
    if not feature_names:
        raise ValueError("No non-empty calibration features are available for this fit.")

    preprocessor = _build_preprocessor(fit_categorical, fit_numeric)
    x_train = preprocessor.fit_transform(train[feature_names])
    x_valid = preprocessor.transform(valid[feature_names])
    y_train = _model_target_values(train, config)
    y_valid = _model_target_values(valid, config)
    early_stopping_rounds = _early_stopping_rounds(config) if early_stopping else None
    estimator = _build_base_model_estimator(config, method, params, early_stopping_rounds=early_stopping_rounds)
    _fit_base_estimator(
        estimator=estimator,
        method=method,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        early_stopping_rounds=early_stopping_rounds,
    )
    metadata = {
        "numeric_features": ",".join(fit_numeric),
        "categorical_features": ",".join(fit_categorical),
        "best_iteration": _best_iteration(estimator),
        "target_mode": config.effective_target_mode,
        "model_target": _model_target_column(config),
    }
    return _prediction_output_to_high(estimator.predict(x_valid), valid, config), metadata


def _fit_feature_columns(
    train: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    max_missing_fraction: float | None = None,
) -> tuple[list[str], list[str]]:
    def passes_missingness(column: str, *, numeric_column: bool) -> bool:
        if column not in train:
            return False
        values = pd.to_numeric(train[column], errors="coerce") if numeric_column else train[column]
        if max_missing_fraction is None:
            return bool(values.notna().any()) if numeric_column else True
        if not values.notna().any():
            return False
        return bool(values.isna().mean() <= float(max_missing_fraction))

    fit_categorical = [column for column in categorical if passes_missingness(column, numeric_column=False)]
    fit_numeric = [column for column in numeric if passes_missingness(column, numeric_column=True)]
    return fit_categorical, fit_numeric


def _fit_base_estimator(
    estimator: Any,
    method: str,
    x_train: Any,
    y_train: pd.Series,
    x_valid: Any,
    y_valid: pd.Series,
    early_stopping_rounds: int | None,
) -> None:
    if early_stopping_rounds is None:
        estimator.fit(x_train, y_train)
        return
    try:
        if method == "xgboost":
            estimator.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
            return
        if method == "lightgbm":
            import lightgbm as lgb
            eval_metric = "rmse"
            get_params = getattr(estimator, "get_params", None)
            if callable(get_params):
                metric = get_params().get("metric")
                if isinstance(metric, str) and metric:
                    eval_metric = metric

            estimator.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric=eval_metric,
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            return
        if method == "catboost":
            estimator.fit(
                x_train,
                y_train,
                eval_set=(x_valid, y_valid),
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=True,
                verbose=False,
            )
            return
    except TypeError:
        pass
    estimator.fit(x_train, y_train)


def _early_stopping_rounds(config: StationStackingConfig) -> int:
    return 20 if config.fast_mode else 50


def _best_iteration(estimator: Any) -> int | None:
    for attr in ("best_iteration", "best_iteration_"):
        value = getattr(estimator, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    if hasattr(estimator, "get_best_iteration"):
        try:
            value = estimator.get_best_iteration()
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None
    return None


def _year_split_fold_weight(fold: YearSplitFold, config: StationStackingConfig | None = None) -> float:
    if config is not None and config.effective_year_split_validation_weights is not None:
        return float(config.effective_year_split_validation_weights.get(fold.validation_year, 1.0))
    return float(YEAR_SPLIT_VALIDATION_WEIGHTS.get(fold.validation_year, 1.0))


def _weighted_fold_score(
    fold_scores: list[tuple[YearSplitFold, float]], config: StationStackingConfig | None = None
) -> float:
    if not fold_scores:
        return float("inf")
    weights = np.asarray([_year_split_fold_weight(fold, config) for fold, _ in fold_scores], dtype=float)
    scores = np.asarray([score for _, score in fold_scores], dtype=float)
    if float(weights.sum()) <= 0:
        return float(np.mean(scores))
    return float(np.average(scores, weights=weights))


def _trial_pruned_exception() -> Exception:
    import optuna

    return optuna.TrialPruned()


def _create_stack_optuna_study(config: StationStackingConfig):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO if config.optuna_verbose else optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=config.random_state + 1000,
        n_startup_trials=config.effective_stack_optuna_startup_trials,
    )
    return optuna.create_study(
        direction="minimize",
        sampler=sampler,
        **_optuna_study_storage_kwargs(config, stage="stack", method=STACK_METHOD),
    )


def _stack_meta_train_valid_split(stack_source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stack_source.empty:
        return pd.DataFrame(), pd.DataFrame()
    years = pd.to_datetime(stack_source["contract_date"], errors="coerce").dt.year
    available_years = sorted(int(year) for year in years.dropna().unique())
    validation_year = available_years[-1] if available_years else None
    train = stack_source.loc[years.lt(validation_year)].copy() if validation_year is not None else pd.DataFrame()
    valid = stack_source.loc[years.eq(validation_year)].copy() if validation_year is not None else pd.DataFrame()
    if not train.empty and not valid.empty:
        return train, valid
    ordered = stack_source.sort_values("contract_date").reset_index(drop=True)
    split_at = max(1, int(len(ordered) * 0.5))
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def _uses_expanding_stack_validation(config: StationStackingConfig) -> bool:
    """Select expanding stack validation from the explicit training profile."""
    return config.effective_training_profile == TRAINING_PROFILE_V20_ALIGNED


def _select_stack_tuning_candidate(
    tuning: pd.DataFrame,
    metric_col: str,
    *,
    aggregate_folds: bool,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Select one Ridge trial identically for evaluation and artifact export."""
    if metric_col not in {"mae_f", "rmse_f", "bucket_log_loss"}:
        raise ValueError("metric_col must be 'mae_f', 'rmse_f', or 'bucket_log_loss'")
    ok = tuning.loc[tuning["status"].astype(str).str.lower().eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful ridge stack tuning rows are available.")
    if aggregate_folds:
        aggregate = ok.groupby("param_key", as_index=False).agg(
            mean_metric=(metric_col, "mean"),
            worst_metric=(metric_col, "max"),
        )
        selected_key = aggregate.sort_values(
            ["mean_metric", "worst_metric", "param_key"]
        ).iloc[0]["param_key"]
    else:
        selected_key = ok.sort_values([metric_col, "param_key"]).iloc[0]["param_key"]
    selected_rows = ok.loc[ok["param_key"].eq(selected_key)].copy()
    selected = selected_rows.iloc[0]
    summary = {
        "mean_metric": float(pd.to_numeric(selected_rows[metric_col], errors="coerce").mean()),
        "worst_metric": float(pd.to_numeric(selected_rows[metric_col], errors="coerce").max()),
        "fold_count": int(selected_rows["fold"].nunique()) if "fold" in selected_rows else int(len(selected_rows)),
        "selection_rule": (
            "mean_metric_then_worst_metric_then_param_key"
            if aggregate_folds
            else "best_metric_then_param_key"
        ),
    }
    return selected, selected_rows, summary


def _v20_stack_meta_splits(stack_source: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    years = pd.to_datetime(stack_source["contract_date"], errors="coerce").dt.year
    eligible = years.between(
        min(V20_EXPANDING_FOLDS, key=lambda fold: fold.validation_year).validation_year,
        max(V20_EXPANDING_FOLDS, key=lambda fold: fold.validation_year).validation_year,
    )
    splits: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for validation_year in V20_STACK_META_VALIDATION_YEARS:
        train = stack_source.loc[eligible & years.lt(validation_year)].copy()
        valid = stack_source.loc[eligible & years.eq(validation_year)].copy()
        if not train.empty and not valid.empty:
            splits.append((validation_year, train, valid))
    return splits


def _stack_features_for_set(
    feature_set: str,
    base_model_methods: Iterable[str] | None = None,
    providers: Iterable[str] | None = None,
) -> list[str]:
    if feature_set not in STACK_FEATURE_SETS:
        raise ValueError(f"Unknown stack feature set: {feature_set}")
    base_methods = tuple(base_model_methods or BASE_MODEL_METHODS)
    if feature_set == "models_only":
        methods = base_methods
    else:
        raw_methods = tuple(f"{provider}_raw" for provider in (providers or ("hrrr", "gfs")))
        methods = (*base_methods, *raw_methods)
    return [f"{method}_predicted_high_f" for method in methods]


def _suggest_stack_hyperparameters(trial, config: StationStackingConfig) -> dict[str, Any]:
    space = config.effective_hyperparameter_space
    if space == "wide_plus":
        alpha_low, alpha_high = 1e-6, 1e6
    elif space == "wide":
        alpha_low, alpha_high = 1e-6, 1e5
    else:
        alpha_low, alpha_high = 1e-4, 1e3
    return {
        "feature_set": trial.suggest_categorical("feature_set", tuple(STACK_FEATURE_SETS)),
        "alpha": trial.suggest_float("alpha", alpha_low, alpha_high, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }


def _create_optuna_study(config: StationStackingConfig, method: str):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO if config.optuna_verbose else optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=config.random_state + BASE_MODEL_METHODS.index(method),
        n_startup_trials=config.effective_optuna_startup_trials,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    return optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        **_optuna_study_storage_kwargs(config, stage="base", method=method),
    )


def _optuna_study_storage_kwargs(config: StationStackingConfig, *, stage: str, method: str) -> dict[str, Any]:
    storage_uri = config.resolved_optuna_storage_uri()
    if storage_uri is None:
        return {}
    storage_path = config.resolved_optuna_storage_path()
    if storage_path is not None:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "storage": storage_uri,
        "study_name": _optuna_study_name(config, stage=stage, method=method),
        "load_if_exists": True,
    }


def _optuna_study_name(config: StationStackingConfig, *, stage: str, method: str) -> str:
    station = config.station_id.upper()
    version = config.effective_feature_version
    metric = config.effective_optuna_metric
    space = config.effective_hyperparameter_space
    target = config.effective_target_mode
    target_part = "" if target == TARGET_MODE_DIRECT_HIGH else f"_{target}"
    space_part = "" if space == "default" else f"_{space}"
    profile = config.effective_training_profile
    profile_part = "" if profile == TRAINING_PROFILE_LEGACY else f"_{profile}"
    return f"{station}_{version}{target_part}{profile_part}_{stage}_{method}_{metric}{space_part}"


def _remaining_optuna_trials(study: Any, target_trials: int) -> int:
    finished_trials = 0
    for trial in _study_trials(study):
        state = getattr(trial, "state", None)
        is_finished = getattr(state, "is_finished", None)
        if state is None or not callable(is_finished) or bool(is_finished()):
            finished_trials += 1
    return max(0, int(target_trials) - finished_trials)


def _study_trials(study: Any) -> list[Any]:
    if hasattr(study, "get_trials"):
        return list(study.get_trials(deepcopy=False))
    return list(getattr(study, "trials", []))


def _study_tuning_rows(study: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in _study_trials(study):
        attrs = getattr(trial, "user_attrs", {}) or {}
        trial_rows = attrs.get("tuning_rows")
        if not isinstance(trial_rows, list):
            continue
        rows.extend(dict(row) for row in trial_rows if isinstance(row, dict))
    return rows


def _set_trial_checkpoint_attrs(
    trial: Any,
    *,
    method: str,
    param_key: str,
    params: dict[str, Any],
    rows: list[dict[str, Any]],
    fit_metadata: list[dict[str, Any]],
    status: str,
    error: str,
    objective_value: float | None = None,
) -> None:
    if not hasattr(trial, "set_user_attr"):
        return
    attrs: dict[str, Any] = {
        "method": method,
        "param_key": param_key,
        "params": params,
        "tuning_rows": rows,
        "fold_metrics": _fold_metric_attrs(rows),
        "fit_metadata": fit_metadata,
        "status": status,
        "error": error,
    }
    if objective_value is not None:
        attrs["objective_value"] = objective_value
    for key, value in attrs.items():
        trial.set_user_attr(key, _optuna_jsonable(value))


def _fold_metric_attrs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("fold", "fold_weight", "mae_f", "rmse_f", "bucket_log_loss", "count", "status", "error")
    return [{key: row.get(key) for key in keys if key in row} for row in rows]


def _optuna_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _optuna_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_optuna_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _optuna_jsonable(value.item())
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _suggest_hyperparameters(method: str, trial, config: StationStackingConfig) -> dict[str, Any]:
    space = config.effective_hyperparameter_space
    wide = space in {"wide", "wide_plus"}
    wide_plus = space == "wide_plus"
    if method == "xgboost":
        max_estimators = 400 if config.fast_mode else (6000 if wide_plus else 3500 if wide else 2000)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50 if wide else 80, max_estimators),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.0005 if wide_plus else 0.001 if wide else 0.003,
                0.25 if wide else 0.15,
                log=True,
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 12 if wide else 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.01 if wide else 0.1, 100.0 if wide else 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 15.0 if wide else 5.0),
            "subsample": trial.suggest_float("subsample", 0.35 if wide else 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35 if wide else 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-10 if wide else 1e-8, 100.0 if wide else 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4 if wide else 1e-3, 200.0 if wide else 50.0, log=True),
        }
        if config.effective_feature_version in HUBER_STACK_FEATURE_VERSIONS:
            params["objective"] = "reg:pseudohubererror"
        return params
    if method == "lightgbm":
        max_estimators = 400 if config.fast_mode else (3500 if wide else 2000)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50 if wide else 80, max_estimators),
            "learning_rate": trial.suggest_float("learning_rate", 0.001 if wide else 0.003, 0.25 if wide else 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 512 if wide else 256),
            "max_depth": trial.suggest_int("max_depth", 2, 14 if wide else 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2 if wide else 5, 250 if wide else 150),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0 if wide else 2.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.35 if wide else 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.35 if wide else 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-10 if wide else 1e-8, 100.0 if wide else 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4 if wide else 1e-3, 200.0 if wide else 50.0, log=True),
        }
        if config.effective_feature_version in HUBER_STACK_FEATURE_VERSIONS:
            params["objective"] = "huber"
            params["huber_alpha"] = trial.suggest_categorical("huber_alpha", (0.75, 0.85, 0.9, 0.95))
        return params
    if method == "catboost":
        max_iterations = 400 if config.fast_mode else (6000 if wide_plus else 3500 if wide else 2000)
        if config.effective_catboost_max_iterations is not None:
            max_iterations = min(max_iterations, config.effective_catboost_max_iterations)
        max_depth = 12 if wide else 10
        if config.effective_catboost_max_depth is not None:
            max_depth = min(max_depth, config.effective_catboost_max_depth)
        min_learning_rate = 0.001 if wide else 0.003
        if config.effective_catboost_min_learning_rate is not None:
            min_learning_rate = max(min_learning_rate, config.effective_catboost_min_learning_rate)
        max_border_count = config.effective_catboost_max_border_count or 255
        params = {
            "iterations": trial.suggest_int("iterations", 50 if wide else 80, max_iterations),
            "learning_rate": trial.suggest_float("learning_rate", min_learning_rate, 0.25 if wide else 0.15, log=True),
            "depth": trial.suggest_int("depth", 2, max_depth),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.05 if wide else 0.5, 200.0 if wide else 50.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 20.0 if wide else 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 20.0 if wide else 10.0),
            "border_count": trial.suggest_int("border_count", 16 if wide else 32, max_border_count),
            "rsm": trial.suggest_float("rsm", 0.35 if wide else 0.5, 1.0),
        }
        if config.effective_feature_version in CATBOOST_HUBER_FEATURE_VERSIONS:
            huber_delta_choices = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0) if wide_plus else (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
            params["huber_delta"] = trial.suggest_categorical("huber_delta", huber_delta_choices)
        return params
    raise ValueError(f"Unknown base model method: {method}")


def _selected_hyperparameters(tuning: pd.DataFrame, metric_col: str = "rmse_f") -> pd.DataFrame:
    if metric_col not in {"mae_f", "rmse_f", "bucket_log_loss"}:
        raise ValueError("metric_col must be 'mae_f', 'rmse_f', or 'bucket_log_loss'")
    mean_metric_col = f"mean_validation_{metric_col}"
    selected_columns = [
        "method",
        "param_key",
        "mean_validation_rmse_f",
        "mean_validation_mae_f",
        "mean_validation_bucket_log_loss",
        "worst_validation_rmse_f",
        "worst_validation_mae_f",
        "worst_validation_bucket_log_loss",
    ]
    if tuning.empty:
        return pd.DataFrame(columns=selected_columns)
    ok = tuning.loc[tuning["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(columns=selected_columns)
    if "fold" in ok:
        fold_counts = ok.groupby(["method", "param_key"], dropna=False)["fold"].nunique()
        complete_fold_count = int(fold_counts.max()) if not fold_counts.empty else 0
        if complete_fold_count > 1:
            complete_keys = fold_counts.loc[fold_counts.eq(complete_fold_count)].index
            ok = ok.set_index(["method", "param_key"]).loc[complete_keys].reset_index()
    if "fold_weight" not in ok:
        ok["fold_weight"] = 1.0

    def weighted_metrics(group: pd.DataFrame) -> pd.Series:
        weights = pd.to_numeric(group["fold_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(group), dtype=float)
        bucket_loss = (
            pd.to_numeric(group["bucket_log_loss"], errors="coerce")
            if "bucket_log_loss" in group
            else pd.Series(np.nan, index=group.index)
        )
        return pd.Series(
            {
                "mean_validation_rmse_f": float(np.average(pd.to_numeric(group["rmse_f"], errors="coerce"), weights=weights)),
                "mean_validation_mae_f": float(np.average(pd.to_numeric(group["mae_f"], errors="coerce"), weights=weights)),
                "mean_validation_bucket_log_loss": float(np.average(bucket_loss.fillna(float("inf")), weights=weights)),
                "worst_validation_rmse_f": float(pd.to_numeric(group["rmse_f"], errors="coerce").max()),
                "worst_validation_mae_f": float(pd.to_numeric(group["mae_f"], errors="coerce").max()),
                "worst_validation_bucket_log_loss": float(bucket_loss.fillna(float("inf")).max()),
            }
        )

    grouped = ok.groupby(["method", "param_key"], dropna=False).apply(weighted_metrics, include_groups=False).reset_index()
    grouped = grouped.sort_values(["method", mean_metric_col, f"worst_validation_{metric_col}", "param_key"])
    selected = grouped.groupby("method", dropna=False).head(1).reset_index(drop=True)
    param_columns = [column for column in tuning.columns if column.startswith("param_") and column != "param_key"]
    params = ok[["method", "param_key", *param_columns]].drop_duplicates(["method", "param_key"])
    return selected.merge(params, on=["method", "param_key"], how="left")


def _params_from_selected_row(row: pd.Series) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in row.items():
        if not str(key).startswith("param_") or pd.isna(value):
            continue
        params[str(key).removeprefix("param_")] = value
    return params


def _filter_predictions_to_selected_params(predictions: list[pd.DataFrame], selected: pd.DataFrame) -> pd.DataFrame:
    if not predictions or selected.empty:
        return _empty_year_split_predictions()
    selected_keys = set(zip(selected["method"].astype(str), selected["param_key"].astype(str), strict=False))
    frames = []
    for frame in predictions:
        if frame.empty or "param_key" not in frame:
            continue
        keys = list(zip(frame["method"].astype(str), frame["param_key"].astype(str), strict=False))
        mask = [key in selected_keys for key in keys]
        if any(mask):
            frames.append(frame.loc[mask].copy())
    return pd.concat(frames, ignore_index=True) if frames else _empty_year_split_predictions()


def _validation_predictions_for_selected_params(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    folds: tuple[YearSplitFold, ...],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or selected.empty:
        return _empty_year_split_predictions()
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    rows: list[pd.DataFrame] = []
    for _, selected_row in selected.iterrows():
        method = str(selected_row["method"])
        params = _params_from_selected_row(selected_row)
        for fold in folds:
            train = frame.loc[year.between(fold.train_start_year, fold.train_end_year)].copy()
            valid = frame.loc[year.eq(fold.validation_year)].copy()
            if train.empty or valid.empty:
                continue
            try:
                predicted, _ = _fit_predict_base_model(
                    config=config,
                    categorical=categorical,
                    numeric=numeric,
                    method=method,
                    params=params,
                    train=train,
                    valid=valid,
                    early_stopping=False,
                )
            except Exception:
                continue
            pred = valid[["contract_date", TARGET]].copy()
            pred["method"] = method
            pred["param_key"] = str(selected_row["param_key"])
            pred["predicted_high_f"] = predicted
            pred["evaluation_scope"] = "year_split_validation"
            pred["fold"] = fold.name
            rows.append(_year_split_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_year_split_predictions()


def _year_split_best_raw_provider(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: StationStackingConfig,
    fold: str,
    evaluation_scope: str,
) -> pd.DataFrame:
    if train.empty or valid.empty:
        return _empty_year_split_predictions()
    mae_by_provider = {
        provider: float((train[TARGET] - pd.to_numeric(train[HIGH_COLUMNS[provider]], errors="coerce")).abs().mean())
        for provider in config.providers
        if HIGH_COLUMNS[provider] in train
    }
    if not mae_by_provider:
        return _empty_year_split_predictions()
    best_provider = min(mae_by_provider, key=mae_by_provider.get)
    column = HIGH_COLUMNS[best_provider]
    pred = valid.loc[valid[column].notna(), ["contract_date", TARGET, column]].copy()
    if pred.empty:
        return _empty_year_split_predictions()
    pred["method"] = "best_raw_provider"
    pred["predicted_high_f"] = pred[column]
    pred["evaluation_scope"] = evaluation_scope
    pred["fold"] = fold
    return _year_split_prediction_columns(pred)


def _year_split_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = _prediction_columns(frame)
    out["fold"] = frame.get("fold", pd.Series(pd.NA, index=frame.index)).to_numpy()
    out["param_key"] = frame.get("param_key", pd.Series(pd.NA, index=frame.index)).to_numpy()
    return out[
        [
            "contract_date",
            "fold",
            "method",
            "param_key",
            "evaluation_scope",
            TARGET,
            "predicted_high_f",
            "error_f",
            "absolute_error_f",
        ]
    ]


def _empty_year_split_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "contract_date",
            "fold",
            "method",
            "param_key",
            "evaluation_scope",
            TARGET,
            "predicted_high_f",
            "error_f",
            "absolute_error_f",
        ]
    )


def _year_split_stack_source_frame(predictions: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    combined = predictions.loc[predictions["method"].isin(methods)].copy()
    if combined.empty:
        return pd.DataFrame()
    pivot = combined.pivot_table(
        index="contract_date",
        columns="method",
        values="predicted_high_f",
        aggfunc="first",
    )
    pivot.columns = [f"{column}_predicted_high_f" for column in pivot.columns]
    required = [f"{method}_predicted_high_f" for method in methods]
    if any(column not in pivot for column in required):
        return pd.DataFrame()
    actuals = combined.groupby("contract_date", dropna=False)[TARGET].first()
    return pivot.join(actuals).reset_index()


def _round_half_up_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.floor(values + 0.5).astype("Int64")


def _temperature_bracket_from_rounded(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    rounded = int(value)
    lower = rounded if rounded % 2 == 0 else rounded - 1
    return f"{lower}-{lower + 1}"


def _sort_year_split_visible_methods(frame: pd.DataFrame, include_period: bool = False) -> pd.DataFrame:
    out = frame.copy()
    method_order = {method: index for index, method in enumerate(YEAR_SPLIT_SCOREBOARD_METHODS)}
    sort_columns = []
    if include_period and "period" in out:
        period_order = {"validation_2024_2025": 0, f"test_{YEAR_SPLIT_TEST_YEAR}": 1}
        out["_period_order"] = out["period"].map(period_order).fillna(len(period_order))
        sort_columns.append("_period_order")
    out["_method_order"] = out["method"].map(method_order).fillna(len(method_order))
    sort_columns.append("_method_order")
    if "contract_date" in out:
        sort_columns.append("contract_date")
    out = out.sort_values(sort_columns).drop(columns=[column for column in ["_period_order", "_method_order"] if column in out])
    return out.reset_index(drop=True)


def _stack_source_frame(
    base_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    base_model_methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    combined = pd.concat([base_predictions, baseline_predictions], ignore_index=True)
    combined = combined.loc[combined["method"].isin([*(base_model_methods or BASE_MODEL_METHODS), *BASELINE_METHODS])].copy()
    if combined.empty:
        return pd.DataFrame()
    pivot = combined.pivot_table(
        index="contract_date",
        columns="method",
        values="predicted_high_f",
        aggfunc="first",
    )
    pivot.columns = [f"{column}_predicted_high_f" for column in pivot.columns]
    actuals = combined.groupby("contract_date", dropna=False)[TARGET].first()
    out = pivot.join(actuals).reset_index()
    return out


def _metric_row(group: pd.DataFrame) -> pd.Series:
    error = pd.to_numeric(group["error_f"], errors="coerce")
    abs_error = error.abs()
    bucket_hit = _bucket_hit_series(group)
    bucket_log_loss = _bucket_log_loss(group)
    return pd.Series(
        {
            "count": int(error.notna().sum()),
            "mae_f": float(abs_error.mean()),
            "rmse_f": float(np.sqrt((error**2).mean())),
            "bias_f": float(error.mean()),
            "bucket_log_loss": bucket_log_loss,
            "bucket_accuracy_pct": float(bucket_hit.mean() * 100.0),
            "p95_absolute_error_f": float(abs_error.quantile(0.95)) if abs_error.notna().any() else float("nan"),
            "large_miss_5f_pct": float((abs_error >= 5.0).mean() * 100),
            "within_1f_pct": float((abs_error <= 1).mean() * 100),
            "within_2f_pct": float((abs_error <= 2).mean() * 100),
            "within_3f_pct": float((abs_error <= 3).mean() * 100),
            "first_contract_date": group["contract_date"].min(),
            "last_contract_date": group["contract_date"].max(),
        }
    )


def _bucket_hit_series(frame: pd.DataFrame) -> pd.Series:
    if "bracket_hit" in frame:
        return frame["bracket_hit"].astype("boolean")
    actual = _round_half_up_series(frame[TARGET]).map(_temperature_bracket_from_rounded)
    predicted = _round_half_up_series(frame["predicted_high_f"]).map(_temperature_bracket_from_rounded)
    return actual.eq(predicted).where(actual.notna() & predicted.notna()).astype("boolean")


def _bucket_log_loss(frame: pd.DataFrame) -> float:
    losses = _bucket_log_loss_series(frame)
    return float(losses.mean()) if losses.notna().any() else float("inf")


def _bucket_log_loss_series(frame: pd.DataFrame) -> pd.Series:
    actual = pd.to_numeric(frame[TARGET], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_high_f"], errors="coerce")
    valid = actual.notna() & predicted.notna()
    losses = pd.Series(np.nan, index=frame.index, dtype="float64")
    if not bool(valid.any()):
        return losses
    residual = actual.loc[valid] - predicted.loc[valid]
    error_mean = float(residual.mean())
    error_std = _residual_std_f(residual)
    for index, actual_value in actual.loc[valid].items():
        bounds = _polymarket_bucket_bounds(actual_value)
        if bounds is None:
            continue
        lower, upper = bounds
        mean_actual = float(predicted.loc[index]) + error_mean
        prob = _normal_interval_probability(mean_actual, error_std, lower, upper)
        losses.loc[index] = -math.log(max(V18_BUCKET_LOG_LOSS_EPSILON, prob))
    return losses


def _polymarket_bucket_bounds(value: Any) -> tuple[float, float] | None:
    rounded = round_temperature_half_up(value)
    if rounded is None:
        return None
    lower_int = rounded if rounded % 2 == 0 else rounded - 1
    return float(lower_int) - 0.5, float(lower_int + 1) + 0.5


def _normal_interval_probability(mean: float, std: float, lower: float, upper: float) -> float:
    return max(0.0, min(1.0, _normal_cdf(upper, mean, std) - _normal_cdf(lower, mean, std)))


def _normal_cdf(value: float, mean: float, std: float) -> float:
    return 0.5 * (1.0 + math.erf((float(value) - float(mean)) / (float(std) * math.sqrt(2.0))))


def _residual_std_f(residual: pd.Series) -> float:
    values = pd.to_numeric(residual, errors="coerce").dropna()
    if len(values) >= 2:
        std = float(values.std(ddof=0))
        if math.isfinite(std) and std > 0:
            return max(0.25, std)
    mae = float(values.abs().mean()) if len(values) else 1.0
    return max(0.75, mae if math.isfinite(mae) and mae > 0 else 1.0)


def _common_date_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    method_dates = predictions.groupby("method")["contract_date"].apply(set)
    if method_dates.empty:
        return pd.DataFrame()
    common_dates = set.intersection(*method_dates.tolist())
    if not common_dates:
        return pd.DataFrame()
    common = predictions.loc[predictions["contract_date"].isin(common_dates)].copy()
    common["evaluation_scope"] = "common_dates_all_methods"
    return (
        common.groupby(["evaluation_scope", "method"], dropna=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
