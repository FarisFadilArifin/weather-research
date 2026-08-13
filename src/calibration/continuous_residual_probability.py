from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bucket_probability import (
    MANDATORY_SOURCE_FEATURES,
    canonical_two_degree_bucket,
    degree_to_bucket_probabilities,
    round_half_up,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "station_continuous_residual_probability_research"
LOG_EPSILON = 1e-12
TAIL_TOLERANCE = 1e-12
MODEL_FEATURES = (
    "point_prediction_f",
    "provider_spread_high_f",
    "base_prediction_spread_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "point_rounding_remainder_f",
    "point_distance_to_round_boundary_f",
    "day_of_year_sin",
    "day_of_year_cos",
)
PEAK_FEATURES = (
    "v20_adjusted_high_mean_f",
    "v20_adjusted_high_spread_f",
    "v20_peak_hour_difference",
    "v20_solar_energy_11_14_wh_m2",
    "v20_solar_energy_15_18_wh_m2",
)
BOUNDARY_BINS = (-1e-12, 0.10, 0.25, 0.500000000001)
BOUNDARY_LABELS = ("[0.00,0.10)", "[0.10,0.25)", "[0.25,0.50]")


@dataclass(frozen=True)
class DistributionPrediction:
    family: str
    location: float
    scale: float
    state: Mapping[str, Any]


def continuous_residual(actual_high_f: float, point_prediction_f: float) -> float:
    """The target is deliberately unrounded."""
    return float(actual_high_f) - float(point_prediction_f)


def settlement_interval(degree: int) -> tuple[float, float]:
    """Latent-temperature interval reported as degree under ROUND_HALF_UP.

    Temperatures are positive in this application. At positive half-degree ties,
    ROUND_HALF_UP selects the upper integer, so [d-.5, d+.5) is appropriate;
    exact endpoints have zero mass for all implemented continuous distributions.
    """
    return float(degree) - 0.5, float(degree) + 0.5


def prepare_probability_frame(
    features: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    *,
    station_id: str,
    include_peak_features: bool,
) -> pd.DataFrame:
    station = station_id.upper()
    if station == "KDAL" and include_peak_features:
        raise ValueError("KDAL cannot enable peak features")
    required_point = {"contract_date", "actual_high_f", "predicted_high_f"}
    required_base = {"contract_date", "method", "predicted_high_f"}
    if missing := sorted(required_point - set(point_predictions)):
        raise ValueError("point predictions missing: " + ",".join(missing))
    if missing := sorted(required_base - set(base_predictions)):
        raise ValueError("base predictions missing: " + ",".join(missing))
    point = point_predictions[list(required_point)].copy()
    point["contract_date"] = pd.to_datetime(point["contract_date"], errors="coerce")
    point = point.rename(columns={"predicted_high_f": "point_prediction_f"})
    if point["contract_date"].duplicated().any():
        conflicts = point.groupby("contract_date")[["actual_high_f", "point_prediction_f"]].nunique(dropna=False)
        if conflicts.gt(1).any(axis=None):
            raise ValueError("conflicting station/date point rows")
        point = point.drop_duplicates("contract_date")
    base = base_predictions.loc[
        base_predictions["method"].isin(("xgboost", "lightgbm", "catboost")),
        ["contract_date", "method", "predicted_high_f"],
    ].copy()
    base["contract_date"] = pd.to_datetime(base["contract_date"], errors="coerce")
    if base.duplicated(["contract_date", "method"]).any():
        conflicts = base.groupby(["contract_date", "method"])["predicted_high_f"].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError("conflicting station/date base rows")
        base = base.drop_duplicates(["contract_date", "method"])
    base = base.pivot(index="contract_date", columns="method", values="predicted_high_f")
    base = base.rename(columns=lambda value: f"{value}_predicted_high_f").reset_index()
    source = features.copy()
    source["contract_date"] = pd.to_datetime(source["contract_date"], errors="coerce")
    if source["contract_date"].duplicated().any():
        raise ValueError("feature frame has duplicate contract dates")
    if "station_id" in source and source["station_id"].dropna().astype(str).str.upper().ne(station).any():
        raise ValueError("feature source station identity mismatch")
    out = point.merge(base, on="contract_date", how="inner", validate="one_to_one")
    out = out.merge(source, on="contract_date", how="left", validate="one_to_one", suffixes=("", "_source"))
    mandatory = [
        *MANDATORY_SOURCE_FEATURES,
        "point_prediction_f",
        "xgboost_predicted_high_f",
        "lightgbm_predicted_high_f",
        "catboost_predicted_high_f",
    ]
    if missing := sorted(set(mandatory) - set(out)):
        raise ValueError("missing mandatory live-safe features: " + ",".join(missing))
    for name in mandatory:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=mandatory + ["actual_high_f", "contract_date"]).copy()
    out["station_id"] = station
    out["point_degree_f"] = out["point_prediction_f"].map(round_half_up)
    out["actual_degree_f"] = pd.to_numeric(out["actual_high_f"], errors="coerce").map(round_half_up)
    out["continuous_residual_f"] = out["actual_high_f"] - out["point_prediction_f"]
    out["point_rounding_remainder_f"] = out["point_prediction_f"] - out["point_degree_f"]
    out["point_distance_to_round_boundary_f"] = 0.5 - out["point_rounding_remainder_f"].abs()
    out["point_signed_distance_to_round_boundary_f"] = np.where(
        out["point_rounding_remainder_f"].ge(0),
        out["point_rounding_remainder_f"] - 0.5,
        out["point_rounding_remainder_f"] + 0.5,
    )
    base_cols = [f"{name}_predicted_high_f" for name in ("xgboost", "lightgbm", "catboost")]
    base_values = out[base_cols].apply(pd.to_numeric, errors="coerce")
    out["base_prediction_spread_f"] = base_values.max(axis=1) - base_values.min(axis=1)
    if "provider_spread_high_f" not in out:
        provider = out[["gfs_high_f", "hrrr_high_f", "nbm_high_f"]]
        out["provider_spread_high_f"] = provider.max(axis=1) - provider.min(axis=1)
    dates = out["contract_date"]
    out["month"] = dates.dt.month
    out["year"] = dates.dt.year
    if "day_of_year_sin" not in out:
        out["day_of_year_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    if "day_of_year_cos" not in out:
        out["day_of_year_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
    feature_names = [*MODEL_FEATURES, *(PEAK_FEATURES if include_peak_features else ())]
    for name in feature_names:
        if name not in out:
            out[name] = np.nan
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out["feature_profile"] = "peak_augmented" if include_peak_features else "common_no_peak"
    return out.sort_values("contract_date").reset_index(drop=True)


def _design(frame: pd.DataFrame, feature_names: Sequence[str], state: Mapping[str, Any] | None = None):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    x = frame.reindex(columns=feature_names)
    if state is None:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = StandardScaler()
        values = scaler.fit_transform(imputer.fit_transform(x))
        return values, {"imputer": imputer, "scaler": scaler}
    return state["scaler"].transform(state["imputer"].transform(x)), state


def fit_distribution(
    family: str,
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str] = MODEL_FEATURES,
    bandwidth: float = 1.0,
    prior_strength: float = 30.0,
    alpha: float = 3.0,
) -> dict[str, Any]:
    residuals = frame["continuous_residual_f"].to_numpy(float)
    if len(residuals) < 20:
        raise ValueError("at least 20 residuals are required")
    if family in {"seasonal_empirical", "conditional_empirical"}:
        return {
            "family": family,
            "residuals": residuals,
            "months": frame["month"].to_numpy(int),
            "boundary_band": boundary_band(frame["point_distance_to_round_boundary_f"]).astype(str).to_numpy(),
            "spread": frame["provider_spread_high_f"].to_numpy(float),
            "bandwidth": float(bandwidth),
            "prior_strength": float(prior_strength),
        }
    if family not in {"gaussian", "student_t"}:
        raise ValueError(f"unsupported continuous family: {family}")
    from sklearn.linear_model import Ridge

    x, transform = _design(frame, feature_names)
    mean_model = Ridge(alpha=alpha).fit(x, residuals)
    fitted_mean = mean_model.predict(x)
    log_scale_target = np.log(np.maximum(np.abs(residuals - fitted_mean), 0.15))
    scale_model = Ridge(alpha=alpha).fit(x, log_scale_target)
    raw_scale = np.exp(scale_model.predict(x))
    # E|N(0,s)| = s*sqrt(2/pi); convert absolute residual regression to sigma.
    scale_factor = math.sqrt(math.pi / 2.0)
    return {
        "family": family,
        "feature_names": list(feature_names),
        "transform": transform,
        "mean_model": mean_model,
        "scale_model": scale_model,
        "scale_factor": scale_factor,
        "minimum_scale": 0.20,
        "df": 5.0,
    }


def predict_distributions(state: Mapping[str, Any], rows: pd.DataFrame) -> list[DistributionPrediction]:
    family = str(state["family"])
    if family in {"gaussian", "student_t"}:
        x, _ = _design(rows, state["feature_names"], state["transform"])
        locations = state["mean_model"].predict(x)
        scales = np.maximum(
            np.exp(state["scale_model"].predict(x)) * float(state["scale_factor"]),
            float(state["minimum_scale"]),
        )
        return [
            DistributionPrediction(family, float(mu), float(scale), {"df": state.get("df", 5.0)})
            for mu, scale in zip(locations, scales, strict=True)
        ]
    residuals = np.asarray(state["residuals"], dtype=float)
    months = np.asarray(state["months"], dtype=int)
    bands = np.asarray(state["boundary_band"], dtype=str)
    spreads = np.asarray(state["spread"], dtype=float)
    outputs = []
    row_bands = boundary_band(rows["point_distance_to_round_boundary_f"]).astype(str)
    for (_, row), band in zip(rows.iterrows(), row_bands, strict=True):
        weights = np.ones(len(residuals), dtype=float)
        same_month = months == int(row["month"])
        weights *= np.where(same_month, 2.0, 1.0)
        if family == "conditional_empirical":
            weights *= np.where(bands == band, 2.0, 1.0)
            spread_scale = max(float(np.nanstd(spreads)), 0.25)
            weights *= np.exp(-0.5 * ((spreads - float(row["provider_spread_high_f"])) / spread_scale) ** 2)
        prior = float(state["prior_strength"])
        weights = weights + prior / max(len(weights), 1)
        weights /= weights.sum()
        mean = float(np.sum(weights * residuals))
        scale = float(max(np.sqrt(np.sum(weights * (residuals - mean) ** 2)), 0.20))
        outputs.append(
            DistributionPrediction(
                family,
                mean,
                scale,
                {"samples": residuals, "weights": weights, "bandwidth": float(state["bandwidth"])},
            )
        )
    return outputs


def predict_continuous_bundle(
    bundle: Mapping[str, Any], feature_values: Mapping[str, Any]
) -> dict[str, Any]:
    """Predict with the exact calibrated state represented by an exported bundle."""
    point = _finite_float(feature_values.get("point_prediction_f"), "point_prediction_f")
    observed_high = _finite_float(
        feature_values.get("observed_high_temp_through_as_of_f"),
        "observed_high_temp_through_as_of_f",
    )
    raw = predict_distributions(bundle["model_state"], pd.DataFrame([dict(feature_values)]))[0]
    multiplier = _finite_float(
        bundle.get("calibration_scale_multiplier", 1.0),
        "calibration_scale_multiplier",
    )
    calibrated = calibrated_prediction(raw, multiplier)
    degrees = integrate_settlement_degrees(
        point, calibrated, observed_high_f=observed_high
    )
    buckets = degree_to_bucket_probabilities(degrees)
    return {
        "model_family": calibrated.family,
        "calibration_scale_multiplier": multiplier,
        "predicted_residual_mean_f": calibrated.location,
        "predicted_residual_scale_f": calibrated.scale,
        "predicted_final_mean_f": point + calibrated.location,
        "degree_probabilities": {str(key): value for key, value in degrees.items()},
        "bucket_probabilities": buckets,
    }


def strict_json_data(value: Any) -> Any:
    """Return a JSON-standard representation, mapping non-finite numbers to null."""
    if isinstance(value, Mapping):
        return {str(key): strict_json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [strict_json_data(item) for item in value]
    if isinstance(value, np.generic):
        return strict_json_data(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normal_cdf(value: np.ndarray | float) -> np.ndarray | float:
    from scipy.special import ndtr
    return ndtr(value)


def _normal_ppf(value: np.ndarray | float) -> np.ndarray | float:
    from scipy.special import ndtri
    return ndtri(value)


def distribution_cdf(prediction: DistributionPrediction, value: float) -> float:
    z = (float(value) - prediction.location) / prediction.scale
    if prediction.family == "gaussian":
        return float(_normal_cdf(z))
    if prediction.family == "student_t":
        from scipy.stats import t
        return float(t.cdf(z, df=float(prediction.state.get("df", 5.0))))
    samples = np.asarray(prediction.state["samples"], dtype=float)
    weights = np.asarray(prediction.state["weights"], dtype=float)
    samples = samples - float(np.sum(weights * samples)) + prediction.location
    bandwidth = float(prediction.state["bandwidth"])
    return float(np.sum(weights * _normal_cdf((float(value) - samples) / bandwidth)))


def distribution_pdf(prediction: DistributionPrediction, value: float) -> float | None:
    z = (float(value) - prediction.location) / prediction.scale
    if prediction.family == "gaussian":
        return math.exp(-0.5 * z * z) / (prediction.scale * math.sqrt(2.0 * math.pi))
    if prediction.family == "student_t":
        from scipy.stats import t
        return float(t.pdf(z, df=float(prediction.state.get("df", 5.0))) / prediction.scale)
    return None


def distribution_ppf(prediction: DistributionPrediction, probability: float) -> float:
    probability = float(np.clip(probability, 1e-14, 1 - 1e-14))
    if prediction.family == "gaussian":
        return prediction.location + prediction.scale * float(_normal_ppf(probability))
    if prediction.family == "student_t":
        from scipy.stats import t
        return prediction.location + prediction.scale * float(t.ppf(probability, df=float(prediction.state.get("df", 5.0))))
    samples = np.asarray(prediction.state["samples"], dtype=float)
    weights = np.asarray(prediction.state["weights"], dtype=float)
    samples = samples - float(np.sum(weights * samples)) + prediction.location
    bandwidth = float(prediction.state["bandwidth"])
    # Deterministic stratified mixture sample. This is used only for quantile
    # diagnostics/support selection; settlement mass itself uses the exact CDF.
    count = 511
    grid = (np.arange(count, dtype=float) + 0.5) / count
    order = np.argsort(samples)
    ordered_samples = samples[order]
    ordered_weights = weights[order] / weights.sum()
    component = np.searchsorted(np.cumsum(ordered_weights), grid, side="left")
    noise_grid = np.mod(grid * 0.6180339887498949 + 0.1732050807568877, 1.0)
    noise_grid = np.clip(noise_grid, 1e-9, 1 - 1e-9)
    mixture_sample = np.sort(ordered_samples[np.minimum(component, len(samples) - 1)] + bandwidth * _normal_ppf(noise_grid))
    return float(np.quantile(mixture_sample, probability, method="linear"))


def calibrated_prediction(prediction: DistributionPrediction, multiplier: float) -> DistributionPrediction:
    if prediction.family in {"gaussian", "student_t"}:
        return DistributionPrediction(prediction.family, prediction.location, prediction.scale * multiplier, prediction.state)
    state = dict(prediction.state)
    state["bandwidth"] = float(state["bandwidth"]) * multiplier
    return DistributionPrediction(prediction.family, prediction.location, prediction.scale * multiplier, state)


def truncated_cdf(prediction: DistributionPrediction, value: float, floor: float | None) -> float:
    if floor is None:
        return distribution_cdf(prediction, value)
    minimum_degree = round_half_up(float(floor))
    lower = minimum_degree - 0.5
    if value <= lower:
        return 0.0
    base = distribution_cdf(prediction, lower)
    remaining = 1.0 - base
    if remaining <= 1e-15:
        return 1.0
    return float(np.clip((distribution_cdf(prediction, value) - base) / remaining, 0.0, 1.0))


def truncated_ppf(prediction: DistributionPrediction, probability: float, floor: float | None) -> float:
    if floor is None:
        return distribution_ppf(prediction, probability)
    lower = round_half_up(float(floor)) - 0.5
    base = distribution_cdf(prediction, lower)
    return distribution_ppf(prediction, base + float(probability) * (1.0 - base))


def integrate_settlement_degrees(
    point_prediction_f: float,
    prediction: DistributionPrediction,
    *,
    observed_high_f: float | None = None,
    tail_tolerance: float = TAIL_TOLERANCE,
) -> dict[int, float]:
    """Integrate exact half-degree cells; preserve numerical tail mass at edge cells."""
    final_prediction = DistributionPrediction(
        prediction.family,
        float(point_prediction_f) + prediction.location,
        prediction.scale,
        prediction.state,
    )
    minimum_degree = round_half_up(observed_high_f) if observed_high_f is not None else round_half_up(
        truncated_ppf(final_prediction, tail_tolerance / 2, None)
    )
    maximum_degree = round_half_up(truncated_ppf(final_prediction, 1 - tail_tolerance / 2, observed_high_f))
    maximum_degree = max(maximum_degree, minimum_degree)
    degrees = np.arange(minimum_degree, maximum_degree + 1, dtype=int)
    boundaries = np.arange(minimum_degree - 0.5, maximum_degree + 1.5, 1.0)
    cdf_values = _distribution_cdf_array(final_prediction, boundaries)
    if observed_high_f is not None:
        floor_edge = round_half_up(float(observed_high_f)) - 0.5
        base = float(_distribution_cdf_array(final_prediction, np.asarray([floor_edge]))[0])
        cdf_values = np.clip((cdf_values - base) / max(1.0 - base, 1e-15), 0.0, 1.0)
    probabilities = {
        int(degree): float(max(value, 0.0))
        for degree, value in zip(degrees, np.diff(cdf_values), strict=True)
    }
    # The computed support is quantile-driven, not arbitrary. Attach any remaining
    # floating-point/tail mass to edge settlement cells so no probability vanishes.
    low_edge = minimum_degree - 0.5
    high_edge = maximum_degree + 0.5
    probabilities[minimum_degree] += truncated_cdf(final_prediction, low_edge, observed_high_f)
    probabilities[maximum_degree] += 1.0 - truncated_cdf(final_prediction, high_edge, observed_high_f)
    total = float(sum(probabilities.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("continuous degree probabilities have invalid mass")
    output = {degree: float(max(value, 0.0) / total) for degree, value in probabilities.items()}
    if not math.isclose(sum(output.values()), 1.0, abs_tol=1e-10):
        raise AssertionError("continuous degree probabilities do not sum to one")
    return output


def _distribution_cdf_array(prediction: DistributionPrediction, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    z = (values - prediction.location) / prediction.scale
    if prediction.family == "gaussian":
        return np.asarray(_normal_cdf(z), dtype=float)
    if prediction.family == "student_t":
        from scipy.stats import t
        return np.asarray(t.cdf(z, df=float(prediction.state.get("df", 5.0))), dtype=float)
    samples = np.asarray(prediction.state["samples"], dtype=float)
    weights = np.asarray(prediction.state["weights"], dtype=float)
    samples = samples - float(np.sum(weights * samples)) + prediction.location
    bandwidth = float(prediction.state["bandwidth"])
    return np.asarray(_normal_cdf((values[:, None] - samples[None, :]) / bandwidth) @ weights, dtype=float)


def quantile_crps(prediction: DistributionPrediction, actual: float, *, floor: float | None = None, points: int = 199) -> float:
    probabilities = (np.arange(points, dtype=float) + 0.5) / points
    if prediction.family in {"seasonal_empirical", "conditional_empirical"}:
        raw = np.asarray(prediction.state["samples"], dtype=float)
        weights = np.asarray(prediction.state["weights"], dtype=float)
        ordered = np.argsort(raw)
        raw = raw[ordered]
        weights = weights[ordered] / weights.sum()
        indices = np.searchsorted(np.cumsum(weights), probabilities, side="left")
        centered = raw[np.minimum(indices, len(raw) - 1)] - float(np.sum(weights * raw))
        noise = float(prediction.state["bandwidth"]) * np.asarray(_normal_ppf(probabilities))
        # A fixed coprime rotation avoids spuriously pairing residual and kernel quantiles.
        samples = prediction.location + centered + np.roll(noise, points // 3)
        if floor is not None:
            lower = round_half_up(float(floor)) - 0.5
            retained = np.sort(samples[samples >= lower])
            if len(retained) >= 3:
                pick = np.minimum((probabilities * len(retained)).astype(int), len(retained) - 1)
                samples = retained[pick]
            else:
                samples = np.full(points, lower)
    else:
        samples = np.asarray([truncated_ppf(prediction, p, floor) for p in probabilities])
    first = float(np.mean(np.abs(samples - float(actual))))
    ordered = np.sort(samples)
    weights = 2 * np.arange(1, points + 1) - points - 1
    pair_half = float(np.sum(weights * ordered) / (points * points))
    return first - pair_half


def bucket_log_loss(actual_bucket: str, probabilities: Mapping[str, float], epsilon: float = LOG_EPSILON) -> float:
    return -math.log(max(float(probabilities.get(actual_bucket, 0.0)), epsilon))


def multiclass_brier(actual_bucket: str, probabilities: Mapping[str, float], labels: Sequence[str] | None = None) -> float:
    labels = list(labels or probabilities.keys())
    return float(sum((float(probabilities.get(label, 0.0)) - (label == actual_bucket)) ** 2 for label in labels))


def ranked_probability_score(actual_bucket: str, probabilities: Mapping[str, float], labels: Sequence[str] | None = None) -> float:
    labels = list(labels or sorted(probabilities, key=lambda label: int(label.split("-", 1)[0])))
    actual_index = labels.index(actual_bucket)
    cumulative = 0.0
    score = 0.0
    for index, label in enumerate(labels[:-1]):
        cumulative += float(probabilities.get(label, 0.0))
        score += (cumulative - (1.0 if index >= actual_index else 0.0)) ** 2
    return score / max(len(labels) - 1, 1)


def boundary_band(distance: pd.Series | Sequence[float]) -> pd.Categorical:
    return pd.cut(distance, bins=BOUNDARY_BINS, labels=BOUNDARY_LABELS, right=False, include_lowest=True)


def reliability_rows(predictions: pd.DataFrame, *, bins: int = 10, low_count_threshold: int = 20) -> pd.DataFrame:
    rows = []
    for _, prediction in predictions.iterrows():
        probs = _json_map(prediction["bucket_probabilities_json"])
        for bucket, probability in probs.items():
            rows.append({
                "station_id": prediction["station_id"], "year": prediction["validation_year"],
                "model_family": prediction["model_family"], "bucket": bucket,
                "probability": float(probability), "occurred": bucket == prediction["actual_bucket"],
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    long["probability_bin"] = pd.cut(long["probability"], np.linspace(0, 1, bins + 1), include_lowest=True, right=True)
    output = []
    for keys, group in long.groupby(["station_id", "year", "model_family", "bucket", "probability_bin"], observed=True):
        interval = keys[-1]
        output.append({
            "station_id": keys[0], "year": keys[1], "model_family": keys[2], "bucket": keys[3],
            "probability_bin_lower": float(interval.left), "probability_bin_upper": float(interval.right),
            "mean_predicted_probability": float(group["probability"].mean()),
            "empirical_frequency": float(group["occurred"].mean()), "sample_count": int(len(group)),
            "absolute_calibration_gap": abs(float(group["probability"].mean()) - float(group["occurred"].mean())),
            "low_count": bool(len(group) < low_count_threshold),
        })
    return pd.DataFrame(output)


def common_date_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    models = sorted(predictions["model_family"].unique())
    counts = predictions.groupby("contract_date")["model_family"].nunique()
    dates = counts.index[counts.eq(len(models))]
    common = predictions.loc[predictions["contract_date"].isin(dates)].copy()
    return summarize_metrics(common, coverage="common_dates")


def summarize_metrics(predictions: pd.DataFrame, *, coverage: str = "all_serveable") -> pd.DataFrame:
    rows = []
    for (family, period), group in _period_groups(predictions):
        rows.append({
            "model_family": family, "period": period, "coverage": coverage, "count": int(len(group)),
            "continuous_crps": _nanmean(group.get("continuous_crps")),
            "continuous_nll": _nanmean(group.get("continuous_nll")),
            "pit_mean": _nanmean(group.get("pit")), "pit_variance": _nanvar(group.get("pit")),
            "predictive_mean_mae_f": _nanmean(group.get("predictive_mean_absolute_error_f")),
            "bucket_log_loss": _nanmean(group.get("bucket_log_loss")),
            "bucket_brier": _nanmean(group.get("bucket_brier")), "ranked_probability_score": _nanmean(group.get("ranked_probability_score")),
            "top_bucket_accuracy": _nanmean(group.get("top_bucket_hit")), "top_two_bucket_accuracy": _nanmean(group.get("top_two_bucket_hit")),
            "point_bucket_accuracy": _nanmean(group.get("point_bucket_hit")),
            "mean_actual_bucket_probability": _nanmean(group.get("actual_bucket_probability")),
            "calibration_error": calibration_error(group),
        })
    return pd.DataFrame(rows)


def interval_coverage_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, period), group in _period_groups(predictions):
        for level in (50, 80, 90, 95):
            rows.append({"model_family": family, "period": period, "nominal_coverage": level / 100,
                         "empirical_coverage": _nanmean(group.get(f"interval_{level}_covered")),
                         "average_interval_width_f": _nanmean(group.get(f"interval_{level}_width_f")), "count": int(len(group))})
    return pd.DataFrame(rows)


def pit_histogram_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    continuous = predictions.loc[pd.to_numeric(predictions.get("pit"), errors="coerce").notna()].copy()
    if continuous.empty:
        return pd.DataFrame()
    continuous["pit_bin"] = pd.cut(continuous["pit"], np.linspace(0, 1, 11), include_lowest=True)
    rows = []
    for (family, year, interval), group in continuous.groupby(["model_family", "validation_year", "pit_bin"], observed=True):
        rows.append({"model_family": family, "year": year, "pit_bin_lower": float(interval.left), "pit_bin_upper": float(interval.right), "sample_count": int(len(group)), "fraction": float(len(group) / len(continuous.loc[(continuous.model_family == family) & (continuous.validation_year == year)]))})
    return pd.DataFrame(rows)


def boundary_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["boundary_band"] = boundary_band(frame["point_distance_to_round_boundary_f"])
    rows = []
    for (family, year, band), group in frame.groupby(["model_family", "validation_year", "boundary_band"], observed=True):
        rows.append({"model_family": family, "year": year, "boundary_band": str(band), "sample_count": int(len(group)),
                     "point_bucket_accuracy": _nanmean(group["point_bucket_hit"]), "top_probability_bucket_accuracy": _nanmean(group["top_bucket_hit"]),
                     "bucket_log_loss": _nanmean(group["bucket_log_loss"]), "bucket_brier": _nanmean(group["bucket_brier"]),
                     "continuous_crps": _nanmean(group.get("continuous_crps")), "top_two_accuracy": _nanmean(group["top_two_bucket_hit"]),
                     "mean_actual_bucket_probability": _nanmean(group["actual_bucket_probability"])})
    return pd.DataFrame(rows)


def calibration_error(group: pd.DataFrame) -> float:
    if group.empty:
        return float("nan")
    confidence = pd.to_numeric(group["top_bucket_probability"], errors="coerce")
    correct = pd.to_numeric(group["top_bucket_hit"], errors="coerce")
    total = 0.0
    for low in np.linspace(0, 0.9, 10):
        selected = confidence.ge(low) & confidence.lt(low + 0.1)
        if selected.any():
            total += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return total


def assert_cutoffs(predictions: pd.DataFrame) -> None:
    dates = pd.to_datetime(predictions["contract_date"])
    model = pd.to_datetime(predictions["model_training_cutoff"])
    calibration = pd.to_datetime(predictions["calibration_training_cutoff"])
    start = pd.to_datetime(predictions["calibration_training_start"])
    if not (model < dates).all() or not (calibration < dates).all() or not (start <= calibration).all():
        raise AssertionError("model/calibration cutoffs must precede every prediction date")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _period_groups(predictions: pd.DataFrame):
    for family, family_group in predictions.groupby("model_family"):
        for year in sorted(family_group["validation_year"].unique()):
            yield (family, str(year)), family_group.loc[family_group["validation_year"].eq(year)]
        forward = family_group.loc[family_group["validation_year"].isin((2024, 2025))]
        if not forward.empty:
            yield (family, "2024-2025"), forward


def _nanmean(values: Any) -> float:
    if values is None:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _nanvar(values: Any) -> float:
    if values is None:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.var(ddof=0)) if numeric.notna().any() else float("nan")


def _json_map(value: Any) -> dict[str, float]:
    raw = json.loads(value) if isinstance(value, str) else value
    return {str(key): float(probability) for key, probability in raw.items()}
