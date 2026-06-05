from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    label: str
    lower: float | None
    upper: float | None


def normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0 or not math.isfinite(std):
        raise ValueError("error_std_f must be positive and finite")
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))


def parse_bucket(label: str, continuity: bool = True) -> Bucket:
    text = label.strip().replace("°F", "").replace("°C", "")
    lower_text = text.lower()
    pad = 0.5 if continuity else 0.0

    range_match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
    if range_match and not any(word in lower_text for word in ("below", "lower", "above", "higher")):
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        lo, hi = min(lo, hi), max(lo, hi)
        return Bucket(label, lo - pad, hi + pad)

    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if ("or below" in lower_text or "or lower" in lower_text or "below" in lower_text) and nums:
        return Bucket(label, None, nums[0] + pad)
    if ("or above" in lower_text or "or higher" in lower_text or "above" in lower_text) and nums:
        return Bucket(label, nums[0] - pad, None)
    if len(nums) == 1:
        return Bucket(label, nums[0] - pad, nums[0] + pad)
    raise ValueError(f"Could not parse bucket label: {label!r}")


def bucket_probabilities(
    forecast_high_f: float,
    error_mean_f: float,
    error_std_f: float,
    buckets: list[str],
    continuity: bool = True,
) -> dict[str, float]:
    if not buckets:
        return {}
    mean_actual = forecast_high_f + error_mean_f
    parsed = [parse_bucket(bucket, continuity=continuity) for bucket in buckets]
    probs: dict[str, float] = {}
    for bucket in parsed:
        lo_prob = 0.0 if bucket.lower is None else normal_cdf(bucket.lower, mean_actual, error_std_f)
        hi_prob = 1.0 if bucket.upper is None else normal_cdf(bucket.upper, mean_actual, error_std_f)
        probs[bucket.label] = max(0.0, hi_prob - lo_prob)

    total = sum(probs.values())
    if total > 0:
        probs = {label: value / total for label, value in probs.items()}
    return probs
