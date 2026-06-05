from datetime import date

from src.actuals_fetch import local_day_window


def test_local_day_window_regular_day():
    start_local, end_local, start_utc, next_utc = local_day_window("2026-05-17", "America/Chicago")
    assert start_local.isoformat().startswith("2026-05-17T00:00:00")
    assert end_local.isoformat().startswith("2026-05-17T23:59:59")
    assert (next_utc - start_utc).total_seconds() == 24 * 3600


def test_local_day_window_spring_dst_has_23_utc_hours():
    _, _, start_utc, next_utc = local_day_window(date(2026, 3, 8), "America/New_York")
    assert (next_utc - start_utc).total_seconds() == 23 * 3600


def test_local_day_window_fall_dst_has_25_utc_hours():
    _, _, start_utc, next_utc = local_day_window("2026-11-01", "America/New_York")
    assert (next_utc - start_utc).total_seconds() == 25 * 3600
