from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..direct_nwp_fetch import GFS_V16_LAYOUT_START_UTC


FEATURE_VERSION = "v11_settlement_enriched_v1"
PROVIDERS = ("gfs", "hrrr", "nbm")
STATIONS = ("KATL", "KDAL")
MIN_COVERAGE = 0.90
MISSING_INDICATOR_THRESHOLD = 0.01

FORECAST_RAW_FIELDS = (
    "temp_k_2m",
    "dewpoint_k_2m",
    "relative_humidity_pct_2m",
    "precip_mm_1h",
    "cloud_cover_pct",
    "wind_u_ms_10m",
    "wind_v_ms_10m",
    "wind_speed_ms_10m",
    "wind_direction_deg_10m",
)

HARD_REQUIRED = (
    "gfs_high_f",
    "hrrr_high_f",
    "nbm_high_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
)

# These are forbidden even when an upstream cache happens to contain them.
FORBIDDEN_PATTERNS = (
    "gust",
    "peak_wind",
    "weather_code",
    "heat_index",
    "wind_chill",
    "observed_precip",
    "observed_is_raining",
    "observed_is_drizzle",
    "observed_is_snowing",
    "observed_precip_intensity",
    "rain_forecast_match",
    "ceiling_min",
)
FORBIDDEN_EXACT = {"observed_ceiling_at_as_of"}

OBSERVED_BASE_FIELDS = (
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_dewpoint_at_as_of_f",
    "observed_humidity_at_as_of",
    "observed_pressure_at_as_of",
    "observed_visibility_at_as_of",
    "observed_cloud_cover_at_as_of",
    "observed_wind_speed_at_as_of",
    "observed_wind_u_at_as_of_mph",
    "observed_wind_v_at_as_of_mph",
)

OBSERVED_ENRICHED_FIELDS = (
    "observed_dewpoint_f_change_1h",
    "observed_dewpoint_f_change_3h",
    "observed_humidity_pct_change_1h",
    "observed_humidity_pct_change_3h",
    "observed_pressure_hpa_change_1h",
    "observed_pressure_hpa_change_3h",
    "observed_visibility_miles_change_1h",
    "observed_visibility_miles_change_3h",
    "observed_cloud_pct_change_1h",
    "observed_cloud_pct_change_3h",
    "observed_wind_speed_mph_change_1h",
    "observed_wind_speed_mph_change_3h",
    "observed_wind_u_mph_change_1h",
    "observed_wind_u_mph_change_3h",
    "observed_wind_v_mph_change_1h",
    "observed_wind_v_mph_change_3h",
    "observed_temperature_acceleration_f_per_h2",
    "observed_morning_temperature_range_f",
    "observed_minutes_since_high_so_far_strict_increase",
    "observed_calm_wind",
    "observed_variable_wind",
    "observed_cloud_category",
    "observed_ceiling_present",
)


@dataclass(frozen=True)
class CoverageGate:
    feature: str
    admitted: bool
    minimum_station_year_coverage: float
    maximum_missingness: float
    reason: str
    parent_features: tuple[str, ...] = ()


def enrichment_cache_root(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "calibration" / FEATURE_VERSION


def hourly_partition_path(
    root: str | Path,
    provider: str,
    station_id: str,
    contract_date: str,
) -> Path:
    day = str(contract_date)[:10]
    return Path(root) / "raw_forecast" / provider.lower() / station_id.upper() / day[:4] / f"{day}.csv"


def observation_partition_path(root: str | Path, source: str, station_id: str, year: int) -> Path:
    return Path(root) / "raw_observations" / source.lower() / station_id.upper() / f"{int(year)}.csv"


def normalize_hourly_forecast(
    values_by_fxx: Mapping[int, Mapping[str, Any]],
    *,
    provider: str,
    station_id: str,
    contract_date: str,
    issue_utc: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    issue = pd.to_datetime(issue_utc, utc=True, errors="coerce")
    legacy_gfs = provider.lower() == "gfs" and pd.notna(issue) and issue.to_pydatetime() < GFS_V16_LAYOUT_START_UTC
    for fxx, values in sorted(values_by_fxx.items()):
        row: dict[str, Any] = {
            "provider": provider.lower(),
            "station_id": station_id.upper(),
            "contract_date": str(contract_date)[:10],
            "issue_utc": issue.isoformat() if pd.notna(issue) else pd.NA,
            "forecast_hour": int(fxx),
            "valid_utc": (issue + pd.Timedelta(hours=int(fxx))).isoformat() if pd.notna(issue) else pd.NA,
            "source_temporal_resolution_hours": 3 if legacy_gfs else 1,
            "legacy_temporal_interpolation": bool(legacy_gfs),
            "precip_is_incremental": bool(values.get("_precip_is_incremental", False)),
        }
        for field in FORECAST_RAW_FIELDS:
            row[field] = _number(values.get(field))
        if row["wind_speed_ms_10m"] is None and row["wind_u_ms_10m"] is not None and row["wind_v_ms_10m"] is not None:
            row["wind_speed_ms_10m"] = math.hypot(row["wind_u_ms_10m"], row["wind_v_ms_10m"])
        if row["wind_direction_deg_10m"] is None and row["wind_u_ms_10m"] is not None and row["wind_v_ms_10m"] is not None:
            row["wind_direction_deg_10m"] = _uv_to_direction(row["wind_u_ms_10m"], row["wind_v_ms_10m"])
        if row["wind_u_ms_10m"] is None and row["wind_v_ms_10m"] is None and row["wind_speed_ms_10m"] is not None and row["wind_direction_deg_10m"] is not None:
            radians = math.radians(row["wind_direction_deg_10m"])
            row["wind_u_ms_10m"] = -row["wind_speed_ms_10m"] * math.sin(radians)
            row["wind_v_ms_10m"] = -row["wind_speed_ms_10m"] * math.cos(radians)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and not frame["precip_is_incremental"].fillna(False).all():
        frame["precip_mm_1h"] = _hourly_precip_increments(frame["precip_mm_1h"])
    return frame


def summarize_hourly_forecast(hourly: pd.DataFrame) -> dict[str, Any]:
    if hourly.empty:
        return {}
    ordered = hourly.sort_values("forecast_hour").copy()
    first = ordered.iloc[0]
    temp_f = _kelvin_to_f(ordered.get("temp_k_2m"))
    dew_f = _kelvin_to_f(ordered.get("dewpoint_k_2m"))
    precip = _numeric(ordered.get("precip_mm_1h")).clip(lower=0)
    cloud = _numeric(ordered.get("cloud_cover_pct"))
    u_mph = _numeric(ordered.get("wind_u_ms_10m")) * 2.2369362921
    v_mph = _numeric(ordered.get("wind_v_ms_10m")) * 2.2369362921
    speed_mph = _numeric(ordered.get("wind_speed_ms_10m")) * 2.2369362921
    if speed_mph.isna().all() and not u_mph.isna().all() and not v_mph.isna().all():
        speed_mph = np.hypot(u_mph, v_mph)
    mean_u = u_mph.mean()
    mean_v = v_mph.mean()
    direction = _uv_to_direction(mean_u, mean_v) if pd.notna(mean_u) and pd.notna(mean_v) else np.nan
    wet = precip.gt(0.01)
    source_resolution = _numeric(ordered.get("source_temporal_resolution_hours")).max()
    hourly_precip_recoverable = pd.isna(source_resolution) or source_resolution <= 1
    return {
        "high_f": temp_f.max(),
        "temp_at_11am_f": temp_f.iloc[0] if len(temp_f) else np.nan,
        "dewpoint_at_11am_f": dew_f.iloc[0] if len(dew_f) else np.nan,
        "dewpoint_remaining_mean_f": dew_f.mean(),
        "humidity_at_11am_pct": _at_first(ordered, "relative_humidity_pct_2m"),
        "humidity_remaining_mean_pct": _numeric(ordered.get("relative_humidity_pct_2m")).mean(),
        "precip_total_mm": precip.sum(min_count=1),
        "precip_max_hourly_mm": precip.max() if hourly_precip_recoverable else np.nan,
        "precip_wet_hour_count": int(wet.sum()) if precip.notna().any() and hourly_precip_recoverable else np.nan,
        "precip_any": float(wet.any()) if precip.notna().any() else np.nan,
        "cloud_at_11am_pct": _at_first(ordered, "cloud_cover_pct"),
        "cloud_remaining_mean_pct": cloud.mean(),
        "cloud_remaining_max_pct": cloud.max(),
        "wind_u_at_11am_mph": u_mph.iloc[0] if len(u_mph) else np.nan,
        "wind_v_at_11am_mph": v_mph.iloc[0] if len(v_mph) else np.nan,
        "wind_speed_at_11am_mph": speed_mph.iloc[0] if len(speed_mph) else np.nan,
        "wind_speed_remaining_mean_mph": speed_mph.mean(),
        "wind_speed_remaining_max_mph": speed_mph.max(),
        "wind_vector_mean_direction_sin": math.sin(math.radians(direction)) if pd.notna(direction) else np.nan,
        "wind_vector_mean_direction_cos": math.cos(math.radians(direction)) if pd.notna(direction) else np.nan,
        "forecast_hour_count": int(len(ordered)),
        "source_temporal_resolution_hours": source_resolution,
        "legacy_temporal_interpolation": bool(_numeric(ordered.get("legacy_temporal_interpolation")).fillna(0).max()),
    }


def prefix_provider_summary(summary: Mapping[str, Any], provider: str) -> dict[str, Any]:
    return {f"{provider.lower()}_{key}": value for key, value in summary.items()}


def add_cross_provider_features(frame: pd.DataFrame, admitted: Iterable[str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    allowed = set(admitted) if admitted is not None else set(out.columns)
    suffixes = sorted(
        {
            column.split("_", 1)[1]
            for column in out.columns
            if any(column.startswith(f"{provider}_") for provider in PROVIDERS)
            and column.split("_", 1)[1] not in {"high_f", "forecast_hour_count"}
        }
    )
    for suffix in suffixes:
        parents = [f"{provider}_{suffix}" for provider in PROVIDERS]
        if not all(parent in out and parent in allowed for parent in parents):
            continue
        values = out[parents].apply(pd.to_numeric, errors="coerce")
        out[f"provider_mean_{suffix}"] = values.mean(axis=1)
        out[f"provider_spread_{suffix}"] = values.max(axis=1) - values.min(axis=1)
        out[f"provider_count_{suffix}"] = values.notna().sum(axis=1)
    return out


def add_forecast_observation_deltas(frame: pd.DataFrame, admitted: Iterable[str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    allowed = set(admitted) if admitted is not None else set(out.columns)
    mappings = {
        "dewpoint_at_11am_f": "observed_dewpoint_at_as_of_f",
        "humidity_at_11am_pct": "observed_humidity_at_as_of",
        "cloud_at_11am_pct": "observed_cloud_cover_at_as_of",
        "wind_speed_at_11am_mph": "observed_wind_speed_at_as_of",
        "wind_u_at_11am_mph": "observed_wind_u_at_as_of_mph",
        "wind_v_at_11am_mph": "observed_wind_v_at_as_of_mph",
    }
    for provider in PROVIDERS:
        for suffix, observed in mappings.items():
            forecast = f"{provider}_{suffix}"
            if forecast in out and observed in out and forecast in allowed and observed in allowed:
                out[f"{forecast}_minus_{observed}"] = _numeric(out[forecast]) - _numeric(out[observed])
    return out


def summarize_observation_day(
    observations: pd.DataFrame,
    *,
    contract_date: str,
    timezone: str,
    window_start: str = "10:40",
    window_end: str = "11:00",
) -> dict[str, Any]:
    """Build live-safe observation features from normalized IEM/AWC-like rows.

    Only the last report in 10:40--11:00 local is used as the current report.
    Histories are restricted to timestamps at or before it.
    """
    if observations.empty:
        return {}
    frame = observations.copy()
    time_col = "observed_at_utc" if "observed_at_utc" in frame else "observed_at"
    frame["_utc"] = pd.to_datetime(frame.get(time_col), errors="coerce", utc=True)
    frame = frame.dropna(subset=["_utc"]).sort_values("_utc")
    local = frame["_utc"].dt.tz_convert(timezone)
    day = pd.Timestamp(contract_date).date()
    frame = frame.loc[local.dt.date.eq(day)].copy()
    if frame.empty:
        return {}
    local = frame["_utc"].dt.tz_convert(timezone)
    start_h, start_m = map(int, window_start.split(":"))
    end_h, end_m = map(int, window_end.split(":"))
    mins = local.dt.hour * 60 + local.dt.minute
    candidates = frame.loc[mins.between(start_h * 60 + start_m, end_h * 60 + end_m)].copy()
    if candidates.empty:
        return {}
    current = candidates.iloc[-1]
    history = frame.loc[frame["_utc"].le(current["_utc"])].copy()
    temp = _row_number(current, "temp_f")
    dew = _row_number(current, "dewpoint_f")
    humidity = _row_number(current, "relative_humidity_pct")
    if humidity is None:
        humidity = _relative_humidity(temp, dew)
    wind_speed = _knots_to_mph(_row_number(current, "wind_speed_kt"))
    wind_dir = _row_number(current, "wind_dir_degrees")
    wind_u, wind_v = _wind_components(wind_speed, wind_dir)
    pressure = _row_number(current, "sea_level_pressure_mb") or _inhg_to_hpa(_row_number(current, "altimeter_inhg"))
    cloud_pct = _cloud_cover_pct(current)
    ceiling_present = _ceiling_present(current)
    changes: dict[str, Any] = {}
    current_values = {
        "dewpoint_f": dew,
        "humidity_pct": humidity,
        "pressure_hpa": pressure,
        "visibility_miles": _row_number(current, "visibility_miles"),
        "cloud_pct": cloud_pct,
        "wind_speed_mph": wind_speed,
        "wind_u_mph": wind_u,
        "wind_v_mph": wind_v,
    }
    for hours in (1, 3):
        prior = _nearest_prior(history, current["_utc"] - pd.Timedelta(hours=hours))
        prior_values = _observation_values(prior) if prior is not None else {}
        for name, value in current_values.items():
            old = prior_values.get(name)
            changes[f"observed_{name}_change_{hours}h"] = value - old if value is not None and old is not None else np.nan
    temp_1h = _nearest_prior_value(history, current["_utc"] - pd.Timedelta(hours=1), "temp_f")
    temp_2h = _nearest_prior_value(history, current["_utc"] - pd.Timedelta(hours=2), "temp_f")
    acceleration = temp - 2 * temp_1h + temp_2h if None not in (temp, temp_1h, temp_2h) else np.nan
    history_temp = _numeric(history.get("temp_f"))
    last_increase_minutes = _minutes_since_last_high_increase(history, current["_utc"])
    variable_wind = wind_dir is None or _is_variable_wind(current)
    return {
        "observed_temp_at_as_of_f": temp,
        "observed_high_temp_through_as_of_f": history_temp.max(),
        "observed_dewpoint_at_as_of_f": dew,
        "observed_humidity_at_as_of": humidity,
        "observed_pressure_at_as_of": pressure,
        "observed_visibility_at_as_of": current_values["visibility_miles"],
        "observed_cloud_cover_at_as_of": cloud_pct,
        "observed_wind_speed_at_as_of": wind_speed,
        "observed_wind_u_at_as_of_mph": wind_u,
        "observed_wind_v_at_as_of_mph": wind_v,
        "observed_temperature_acceleration_f_per_h2": acceleration,
        "observed_morning_temperature_range_f": history_temp.max() - history_temp.min(),
        "observed_minutes_since_high_so_far_strict_increase": last_increase_minutes,
        "observed_calm_wind": bool(wind_speed is not None and wind_speed < 1.0),
        "observed_variable_wind": bool(variable_wind),
        "observed_cloud_category": _cloud_category(current),
        "observed_ceiling_present": ceiling_present,
        "observed_as_of_time_utc": current["_utc"].isoformat(),
        **changes,
    }


def coverage_inventory(
    frame: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    years: Iterable[int],
    stations: Iterable[str] = STATIONS,
    threshold: float = MIN_COVERAGE,
    reproducible_features: Iterable[str] | None = None,
    parent_map: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    data["year"] = pd.to_numeric(data.get("year", pd.to_datetime(data["contract_date"]).dt.year), errors="coerce")
    year_set = {int(year) for year in years}
    station_set = {station.upper() for station in stations}
    data = data.loc[data["year"].isin(year_set) & data["station_id"].astype(str).str.upper().isin(station_set)]
    reproducible = set(reproducible_features or candidate_features)
    parents = parent_map or {}
    rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        group_coverages: list[float] = []
        for station in sorted(station_set):
            for year in sorted(year_set):
                group = data.loc[data["station_id"].astype(str).str.upper().eq(station) & data["year"].eq(year)]
                group_coverages.append(float(group[feature].notna().mean()) if feature in group and len(group) else 0.0)
        min_coverage = min(group_coverages, default=0.0)
        parent_features = tuple(parents.get(feature, ()))
        parents_ok = all(parent in reproducible for parent in parent_features)
        is_forbidden = feature in FORBIDDEN_EXACT or any(pattern in feature for pattern in FORBIDDEN_PATTERNS)
        admitted = min_coverage >= threshold and feature in reproducible and parents_ok and not is_forbidden
        reason = "admitted"
        if is_forbidden:
            reason = "forbidden_by_contract"
        elif feature not in reproducible:
            reason = "not_live_reproducible"
        elif not parents_ok:
            reason = "parent_gate_failed"
        elif min_coverage < threshold:
            reason = "coverage_below_threshold"
        rows.append(
            {
                "feature": feature,
                "admitted": admitted,
                "minimum_station_year_coverage": min_coverage,
                "maximum_missingness": 1.0 - min_coverage,
                "reason": reason,
                "parent_features": "|".join(parent_features),
            }
        )
    return pd.DataFrame(rows)


def expanding_fold_coverage_inventory(
    frame: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    folds: Sequence[Any],
    stations: Iterable[str] = STATIONS,
    threshold: float = MIN_COVERAGE,
    reproducible_features: Iterable[str] | None = None,
    parent_map: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Evaluate each gate on that fold's training years, then keep only the intersection.

    This makes the shared model schema safe for the earliest fold; no validation or
    2026 row can influence admission.
    """
    inventories: list[pd.DataFrame] = []
    for fold in folds:
        years = range(int(fold.train_start_year), int(fold.train_end_year) + 1)
        inventory = coverage_inventory(
            frame,
            candidate_features,
            years=years,
            stations=stations,
            threshold=threshold,
            reproducible_features=reproducible_features,
            parent_map=parent_map,
        )
        inventory["fold"] = str(fold.name)
        inventories.append(inventory)
    if not inventories:
        return pd.DataFrame()
    detail = pd.concat(inventories, ignore_index=True)
    admitted_all = detail.groupby("feature")["admitted"].transform("all")
    detail["admitted_all_folds"] = admitted_all
    return detail


def apply_feature_contract(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    keep_non_candidates: Iterable[str] = (),
    add_missing_indicators: bool = True,
) -> pd.DataFrame:
    admitted = set(inventory.loc[inventory["admitted"].fillna(False), "feature"].astype(str))
    keep = set(keep_non_candidates) | admitted
    keep -= FORBIDDEN_EXACT
    keep = {column for column in keep if not any(pattern in column for pattern in FORBIDDEN_PATTERNS)}
    out = frame[[column for column in frame.columns if column in keep]].copy()
    if add_missing_indicators:
        missingness = out.apply(lambda values: pd.to_numeric(values, errors="coerce").isna().mean())
        for column in sorted(admitted):
            if column in out and pd.api.types.is_numeric_dtype(out[column]) and missingness.get(column, 0.0) > MISSING_INDICATOR_THRESHOLD:
                out[f"{column}__missing"] = out[column].isna().astype("int8")
    return out


def validate_hard_requirements(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in HARD_REQUIRED if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing hard-required columns: {missing_columns}")
    missing_rows = frame[list(HARD_REQUIRED)].isna().any(axis=1)
    if missing_rows.any():
        raise ValueError(f"{int(missing_rows.sum())} rows have missing hard-required forecast/observation values; they may not be imputed")


def parity_report(iem: pd.DataFrame, awc: pd.DataFrame, fields: Sequence[str], *, tolerance: Mapping[str, float] | None = None) -> pd.DataFrame:
    tolerances = dict(tolerance or {})
    keys = [key for key in ("station_id", "contract_date") if key in iem and key in awc]
    if not keys:
        raise ValueError("Parity inputs need station_id and contract_date keys")
    merged = iem.merge(awc, on=keys, how="outer", suffixes=("_iem", "_awc"), indicator=True)
    rows: list[dict[str, Any]] = []
    for field in fields:
        raw_left = merged.get(f"{field}_iem", pd.Series(pd.NA, index=merged.index))
        raw_right = merged.get(f"{field}_awc", pd.Series(pd.NA, index=merged.index))
        left = _numeric(raw_left)
        right = _numeric(raw_right)
        paired = left.notna() & right.notna()
        categorical = not paired.any() and (raw_left.notna() & raw_right.notna()).any()
        if categorical:
            paired = raw_left.notna() & raw_right.notna()
            abs_diff = raw_left.astype("string").str.lower().ne(raw_right.astype("string").str.lower()).astype(float)
            left_missingness = float(raw_left.isna().mean())
            right_missingness = float(raw_right.isna().mean())
        else:
            abs_diff = (left - right).abs()
            left_missingness = float(left.isna().mean())
            right_missingness = float(right.isna().mean())
        allowed = float(tolerances.get(field, 0.1))
        rows.append(
            {
                "feature": field,
                "row_match_rate": float(merged["_merge"].eq("both").mean()),
                "iem_missingness": left_missingness,
                "awc_missingness": right_missingness,
                "paired_count": int(paired.sum()),
                "median_abs_difference": float(abs_diff[paired].median()) if paired.any() else np.nan,
                "p95_abs_difference": float(abs_diff[paired].quantile(0.95)) if paired.any() else np.nan,
                "tolerance": allowed,
                "parity_pass": bool(
                    merged["_merge"].eq("both").mean() >= 0.90
                    and abs(left_missingness - right_missingness) <= 0.05
                    and paired.any()
                    and float(abs_diff[paired].quantile(0.95)) <= allowed
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_interval(
    predictions: pd.DataFrame,
    *,
    baseline_method: str,
    candidate_method: str,
    metric: str = "mae",
    iterations: int = 5000,
    seed: int = 20260713,
) -> dict[str, float]:
    base = predictions.loc[predictions["method"].eq(baseline_method), ["station_id", "contract_date", "actual_high_f", "predicted_high_f"]]
    cand = predictions.loc[predictions["method"].eq(candidate_method), ["station_id", "contract_date", "predicted_high_f"]]
    paired = base.merge(cand, on=["station_id", "contract_date"], suffixes=("_baseline", "_candidate"))
    if paired.empty:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "count": 0}
    actual = paired["actual_high_f"].to_numpy(float)
    bpred = paired["predicted_high_f_baseline"].to_numpy(float)
    cpred = paired["predicted_high_f_candidate"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, len(paired), len(paired))
        values[index] = _metric(actual[sample], cpred[sample], metric) - _metric(actual[sample], bpred[sample], metric)
    estimate = _metric(actual, cpred, metric) - _metric(actual, bpred, metric)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "count": int(len(paired)),
    }


def extended_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "actual_high_f", "predicted_high_f", "method"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing metric columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    scopes = [("pooled", "ALL", predictions)]
    scopes.extend(("station", station, group) for station, group in predictions.groupby("station_id"))
    if "fold" in predictions:
        scopes.extend(("fold", str(fold), group) for fold, group in predictions.groupby("fold"))
    for scope, station, scoped in scopes:
        for method, group in scoped.groupby("method"):
            actual = pd.to_numeric(group["actual_high_f"], errors="coerce")
            predicted = pd.to_numeric(group["predicted_high_f"], errors="coerce")
            valid = actual.notna() & predicted.notna()
            error = predicted[valid] - actual[valid]
            absolute = error.abs()
            rows.append(
                {
                    "scope": scope,
                    "station_id": station,
                    "method": method,
                    "count": int(valid.sum()),
                    "mae_f": float(absolute.mean()) if valid.any() else np.nan,
                    "rmse_f": float(np.sqrt(np.mean(error**2))) if valid.any() else np.nan,
                    "bias_f": float(error.mean()) if valid.any() else np.nan,
                    "p90_abs_error_f": float(absolute.quantile(0.90)) if valid.any() else np.nan,
                    "p95_abs_error_f": float(absolute.quantile(0.95)) if valid.any() else np.nan,
                    "bucket_hit_rate": float((np.rint(predicted[valid]) == np.rint(actual[valid])).mean()) if valid.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def promotion_decision(metrics: pd.DataFrame, *, baseline: str, candidate: str) -> pd.DataFrame:
    required = {"scope", "station_id", "method", "bucket_hit_rate", "mae_f", "rmse_f", "p95_abs_error_f", "bias_f"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics missing promotion columns: {sorted(missing)}")
    base = metrics.loc[metrics["method"].eq(baseline)].set_index(["scope", "station_id"])
    cand = metrics.loc[metrics["method"].eq(candidate)].set_index(["scope", "station_id"])
    joined = base.join(cand, lsuffix="_baseline", rsuffix="_candidate", how="inner").reset_index()
    pooled = joined.loc[joined["scope"].eq("pooled")]
    station = joined.loc[joined["scope"].eq("station")]
    checks = {
        "pooled_bucket_gain_at_least_1pp": bool((pooled["bucket_hit_rate_candidate"] - pooled["bucket_hit_rate_baseline"] >= 0.01).all() and not pooled.empty),
        "no_station_bucket_loss_over_1pp": bool((station["bucket_hit_rate_candidate"] - station["bucket_hit_rate_baseline"] >= -0.01).all() and not station.empty),
        "mae_worsening_at_most_005f": bool((pooled["mae_f_candidate"] - pooled["mae_f_baseline"] <= 0.05).all() and not pooled.empty),
        "rmse_worsening_at_most_005f": bool((pooled["rmse_f_candidate"] - pooled["rmse_f_baseline"] <= 0.05).all() and not pooled.empty),
        "p95_worsening_at_most_010f": bool((pooled["p95_abs_error_f_candidate"] - pooled["p95_abs_error_f_baseline"] <= 0.10).all() and not pooled.empty),
        "absolute_bias_at_most_025f": bool(pooled["bias_f_candidate"].abs().le(0.25).all() and not pooled.empty),
    }
    return pd.DataFrame([{"check": name, "passed": passed} for name, passed in checks.items()] + [{"check": "overall", "passed": all(checks.values())}])


def write_contract_manifest(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_version": FEATURE_VERSION,
        "stations": list(STATIONS),
        "providers": list(PROVIDERS),
        "forecast_raw_fields": list(FORECAST_RAW_FIELDS),
        "hard_required": list(HARD_REQUIRED),
        "forbidden_patterns": list(FORBIDDEN_PATTERNS),
        "minimum_station_year_coverage": MIN_COVERAGE,
        "missing_indicator_threshold": MISSING_INDICATOR_THRESHOLD,
        "observation_window_local": ["10:40", "11:00"],
        "imputation": "training-fold median numeric; explicit missing category categorical; never zero",
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _numeric(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    return pd.to_numeric(values, errors="coerce")


def _kelvin_to_f(values: Any) -> pd.Series:
    return (_numeric(values) - 273.15) * 9 / 5 + 32


def _at_first(frame: pd.DataFrame, column: str) -> Any:
    return _number(frame.iloc[0].get(column)) if len(frame) else np.nan


def _uv_to_direction(u: float, v: float) -> float:
    return float((math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0)


def _row_number(row: pd.Series, column: str) -> float | None:
    return _number(row.get(column))


def _relative_humidity(temp_f: float | None, dew_f: float | None) -> float | None:
    if temp_f is None or dew_f is None:
        return None
    temp_c = (temp_f - 32) * 5 / 9
    dew_c = (dew_f - 32) * 5 / 9
    return max(0.0, min(100.0, 100 * math.exp((17.625 * dew_c) / (243.04 + dew_c) - (17.625 * temp_c) / (243.04 + temp_c))))


def _knots_to_mph(value: float | None) -> float | None:
    return value * 1.150779448 if value is not None else None


def _inhg_to_hpa(value: float | None) -> float | None:
    return value * 33.8638866667 if value is not None else None


def _wind_components(speed: float | None, direction: float | None) -> tuple[float | None, float | None]:
    if speed is None or direction is None:
        return None, None
    radians = math.radians(direction)
    return -speed * math.sin(radians), -speed * math.cos(radians)


def _nearest_prior(frame: pd.DataFrame, timestamp: pd.Timestamp, tolerance: str = "45min") -> pd.Series | None:
    distance = (frame["_utc"] - timestamp).abs()
    if distance.empty or distance.min() > pd.Timedelta(tolerance):
        return None
    return frame.loc[distance.idxmin()]


def _nearest_prior_value(frame: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float | None:
    row = _nearest_prior(frame, timestamp)
    return _row_number(row, column) if row is not None else None


def _observation_values(row: pd.Series | None) -> dict[str, float | None]:
    if row is None:
        return {}
    temp = _row_number(row, "temp_f")
    dew = _row_number(row, "dewpoint_f")
    humidity = _row_number(row, "relative_humidity_pct") or _relative_humidity(temp, dew)
    speed = _knots_to_mph(_row_number(row, "wind_speed_kt"))
    direction = _row_number(row, "wind_dir_degrees")
    u, v = _wind_components(speed, direction)
    return {
        "dewpoint_f": dew,
        "humidity_pct": humidity,
        "pressure_hpa": _row_number(row, "sea_level_pressure_mb") or _inhg_to_hpa(_row_number(row, "altimeter_inhg")),
        "visibility_miles": _row_number(row, "visibility_miles"),
        "cloud_pct": _cloud_cover_pct(row),
        "wind_speed_mph": speed,
        "wind_u_mph": u,
        "wind_v_mph": v,
    }


def _cloud_category(row: pd.Series) -> str:
    text = " ".join(str(row.get(column, "")) for column in ("sky_cover", "cloud_cover", "raw_metar")).upper()
    for code, category in (("OVC", "overcast"), ("BKN", "broken"), ("SCT", "scattered"), ("FEW", "few"), ("CLR", "clear"), ("SKC", "clear")):
        if code in text:
            return category
    return "missing"


def _cloud_cover_pct(row: pd.Series) -> float | None:
    direct = _row_number(row, "cloud_cover_pct")
    if direct is not None:
        return direct
    return {"clear": 0.0, "few": 20.0, "scattered": 45.0, "broken": 75.0, "overcast": 100.0}.get(_cloud_category(row))


def _ceiling_present(row: pd.Series) -> bool:
    text = " ".join(str(row.get(column, "")) for column in ("sky_cover", "cloud_cover", "raw_metar")).upper()
    return "BKN" in text or "OVC" in text or _row_number(row, "ceiling_ft") is not None


def _is_variable_wind(row: pd.Series) -> bool:
    text = str(row.get("raw_metar", "")).upper()
    return " VRB" in f" {text}" or "VRB" in str(row.get("wind_dir", "")).upper()


def _minutes_since_last_high_increase(history: pd.DataFrame, current_time: pd.Timestamp) -> float:
    temp = _numeric(history.get("temp_f"))
    running = temp.cummax()
    strict = running.gt(running.shift(1)) & temp.notna()
    if not strict.any():
        return np.nan
    last_time = history.loc[strict, "_utc"].iloc[-1]
    return float((current_time - last_time).total_seconds() / 60)


def _metric(actual: np.ndarray, predicted: np.ndarray, metric: str) -> float:
    error = predicted - actual
    if metric == "mae":
        return float(np.mean(np.abs(error)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if metric == "bias":
        return float(np.mean(error))
    if metric == "bucket_hit_rate":
        return float(np.mean(np.rint(predicted) == np.rint(actual)))
    raise ValueError(f"Unsupported metric: {metric}")


def _hourly_precip_increments(values: pd.Series) -> pd.Series:
    precipitation = pd.to_numeric(values, errors="coerce").clip(lower=0)
    clean = precipitation.dropna()
    if len(clean) < 2:
        return precipitation
    decreases = int(clean.diff().dropna().lt(-0.01).sum())
    # GFS/NBM accumulations are monotone; HRRR hourly fields are not. Convert
    # only when the sequence clearly behaves like an accumulation.
    if decreases > max(1, len(clean) // 4):
        return precipitation
    increments = precipitation.diff()
    if len(increments):
        increments.iloc[0] = precipitation.iloc[0]
    return increments.clip(lower=0)
