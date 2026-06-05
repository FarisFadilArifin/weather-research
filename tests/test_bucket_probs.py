import math

import pytest

from src.bucket_probs import bucket_probabilities, parse_bucket


def test_parse_below_range_and_above_buckets():
    assert parse_bucket("89 or below").upper == 89.5
    middle = parse_bucket("90-91")
    assert middle.lower == 89.5
    assert middle.upper == 91.5
    assert parse_bucket("96 or above").lower == 95.5


def test_bucket_probabilities_sum_to_one():
    probs = bucket_probabilities(
        forecast_high_f=92,
        error_mean_f=0,
        error_std_f=2,
        buckets=["89 or below", "90-91", "92-93", "94-95", "96 or above"],
    )
    assert set(probs) == {"89 or below", "90-91", "92-93", "94-95", "96 or above"}
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    assert probs["92-93"] > probs["96 or above"]


def test_bucket_probabilities_use_error_mean_as_actual_shift():
    cool = bucket_probabilities(90, -2, 1, ["88 or below", "89-91", "92 or above"])
    warm = bucket_probabilities(90, 2, 1, ["88 or below", "89-91", "92 or above"])
    assert cool["88 or below"] > warm["88 or below"]
    assert warm["92 or above"] > cool["92 or above"]


def test_invalid_std_raises():
    with pytest.raises(ValueError):
        bucket_probabilities(90, 0, 0, ["89 or below", "90 or above"])
