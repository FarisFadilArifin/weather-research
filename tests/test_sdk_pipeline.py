from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from src.calibration import sdk_pipeline
from src.calibration.dataset import build_calibration_samples
from src.merge_sdk_nwp_shards import merge_sdk_nwp_shards


def test_sdk_inspection_writes_station_registry_and_archive_limits(tmp_path) -> None:
    registry, availability = sdk_pipeline.inspect_sdk(
        sdk_cache_dir=tmp_path,
        start_date="2016-01-01",
        end_date="2026-05-27",
    )
    assert set(registry["station_id"]) == set(sdk_pipeline.TARGET_STATIONS)
    starts = dict(zip(availability["provider"], availability["archive_start"], strict=False))
    assert starts["hrrr"] == "2014-07-30"
    assert starts["gfs"] == "2021-01-01"
    assert starts["nbm"] == "2020-01-01"
    assert (tmp_path / sdk_pipeline.SDK_STATION_REGISTRY_FILE).exists()
    assert (tmp_path / sdk_pipeline.SDK_AVAILABILITY_FILE).exists()


def test_sdk_actual_backfill_uses_obs_and_resumes(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_obs(station, start, end, **kwargs):
        calls.append((station, start, end))
        return pd.DataFrame(
            {
                "date": pd.date_range(start, end, freq="D").date.astype(str),
                "station": [station[1:]] * len(pd.date_range(start, end, freq="D")),
                "obs_high_f": [80.0] * len(pd.date_range(start, end, freq="D")),
                "source": ["iem"] * len(pd.date_range(start, end, freq="D")),
                "obs_count": [24] * len(pd.date_range(start, end, freq="D")),
            }
        )

    monkeypatch.setattr(sdk_pipeline, "_load_obs", lambda: fake_obs)
    first = sdk_pipeline.backfill_sdk_actuals(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        chunk_days=2,
    )
    second = sdk_pipeline.backfill_sdk_actuals(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        chunk_days=2,
    )
    assert len(calls) == 1
    assert len(first) == 2
    assert len(second) == 2
    assert set(second["fetch_status"]) == {"ok"}


def test_sdk_nwp_backfill_uses_forecast_nwp_and_resumes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_NWP_SUBPROCESS", "0")
    calls: list[tuple[object, str, int]] = []

    def fake_forecast_nwp(station, model, *, cycle, fxx, **kwargs):
        calls.append((station, model, fxx))
        stations = station if isinstance(station, list) else [station]
        return pd.DataFrame(
            {
                "station": stations,
                "valid_at": [(cycle + timedelta(hours=fxx)).isoformat()] * len(stations),
                "temp_k_2m": [300.0 + fxx / 100] * len(stations),
                "dewpoint_k_2m": [290.0] * len(stations),
                "wind_u_ms_10m": [3.0] * len(stations),
                "wind_v_ms_10m": [4.0] * len(stations),
                "relative_humidity_pct_2m": [50.0] * len(stations),
                "grid_dist_km": [1.0] * len(stations),
            }
        )

    monkeypatch.setattr(sdk_pipeline, "_load_forecast_nwp", lambda station_meta: fake_forecast_nwp)
    first = sdk_pipeline.backfill_sdk_nwp(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        models=["hrrr"],
        start_date="2026-01-01",
        end_date="2026-01-01",
    )
    second = sdk_pipeline.backfill_sdk_nwp(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        models=["hrrr"],
        start_date="2026-01-01",
        end_date="2026-01-01",
    )
    assert calls
    assert isinstance(calls[0][0], list)
    assert len(first) == 1
    assert len(second) == 1
    assert second.iloc[0]["provider"] == "hrrr"
    assert second.iloc[0]["raw_forecast_high_f"] > 70
    assert second.iloc[0]["timing_mode"] == "strict_6am"


def test_fresh_after_6am_cycle_selection_uses_first_local_6am_cycle() -> None:
    katl_as_of = sdk_pipeline.forecast_as_of_for_timing("2026-01-15", "America/New_York", "fresh_after_6am")
    katl_cycle, katl_fxx = sdk_pipeline.choose_cycle(
        "hrrr",
        "2026-01-15",
        "America/New_York",
        katl_as_of,
        timing_mode="fresh_after_6am",
        forecast_window_start=katl_as_of,
        forecast_window_end=sdk_pipeline.local_datetime_utc("2026-01-16", "America/New_York", 0),
    )
    assert katl_as_of == datetime(2026, 1, 15, 12, tzinfo=UTC)
    assert katl_cycle == datetime(2026, 1, 15, 11, tzinfo=UTC)
    assert min(katl_fxx) == 1

    dst_as_of = sdk_pipeline.forecast_as_of_for_timing("2026-07-15", "America/New_York", "fresh_after_6am")
    dst_cycle, _ = sdk_pipeline.choose_cycle(
        "hrrr",
        "2026-07-15",
        "America/New_York",
        dst_as_of,
        timing_mode="fresh_after_6am",
        forecast_window_start=dst_as_of,
        forecast_window_end=sdk_pipeline.local_datetime_utc("2026-07-16", "America/New_York", 0),
    )
    assert dst_as_of == datetime(2026, 7, 15, 11, tzinfo=UTC)
    assert dst_cycle == datetime(2026, 7, 15, 10, tzinfo=UTC)

    nbm_cycle, nbm_fxx, nbm_as_of, _, _ = sdk_pipeline.choose_fresh_after_6am_cycle(
        "nbm",
        "2026-01-15",
        "America/New_York",
    )
    assert nbm_as_of == datetime(2026, 1, 15, 12, tzinfo=UTC)
    assert nbm_cycle == datetime(2026, 1, 15, 11, tzinfo=UTC)
    assert min(nbm_fxx) == 1

    gfs_cycle, gfs_fxx, gfs_as_of, _, _ = sdk_pipeline.choose_fresh_after_6am_cycle(
        "gfs",
        "2026-01-15",
        "America/New_York",
    )
    assert gfs_as_of == datetime(2026, 1, 15, 12, tzinfo=UTC)
    assert gfs_cycle == datetime(2026, 1, 15, 12, tzinfo=UTC)
    assert min(gfs_fxx) == 0


def test_fresh_after_6am_gfs_can_shift_as_of_later_when_cycle_schedule_requires_it() -> None:
    cycle, fxx, as_of, _, _ = sdk_pipeline.choose_fresh_after_6am_cycle(
        "gfs",
        "2026-01-15",
        "America/Los_Angeles",
    )
    assert cycle == datetime(2026, 1, 15, 18, tzinfo=UTC)
    assert as_of == datetime(2026, 1, 15, 18, tzinfo=UTC)
    assert min(fxx) == 0


def test_fresh_after_6am_remaining_day_window_excludes_pre_7am_hours() -> None:
    as_of = sdk_pipeline.forecast_as_of_for_timing("2026-01-15", "America/Chicago", "fresh_after_6am")
    start, end = sdk_pipeline.forecast_window_for_timing("2026-01-15", "America/Chicago", "fresh_after_6am", as_of)
    cycle, fxx = sdk_pipeline.choose_cycle(
        "hrrr",
        "2026-01-15",
        "America/Chicago",
        as_of,
        timing_mode="fresh_after_6am",
        forecast_window_start=start,
        forecast_window_end=end,
    )
    assert as_of == datetime(2026, 1, 15, 13, tzinfo=UTC)
    assert cycle == datetime(2026, 1, 15, 12, tzinfo=UTC)
    assert min(fxx) == 1
    assert cycle + timedelta(hours=min(fxx)) == as_of
    assert all(cycle + timedelta(hours=hour) >= as_of for hour in fxx)


def test_same_day_11am_cycle_selection_uses_11am_local_snapshot(monkeypatch) -> None:
    def fake_cycle_hours(model: str) -> tuple[int, ...]:
        return (0, 6, 12, 18) if model == "gfs" else tuple(range(24))

    monkeypatch.setattr(sdk_pipeline, "model_cycle_hours", fake_cycle_hours)
    katl_as_of = sdk_pipeline.forecast_as_of_for_timing("2026-01-15", "America/New_York", "same_day_11am")
    katl_start, katl_end = sdk_pipeline.forecast_window_for_timing(
        "2026-01-15",
        "America/New_York",
        "same_day_11am",
        katl_as_of,
    )
    katl_cycle, katl_fxx = sdk_pipeline.choose_cycle(
        "hrrr",
        "2026-01-15",
        "America/New_York",
        katl_as_of,
        timing_mode="same_day_11am",
        forecast_window_start=katl_start,
        forecast_window_end=katl_end,
    )
    assert katl_as_of == datetime(2026, 1, 15, 16, tzinfo=UTC)
    assert katl_cycle == datetime(2026, 1, 15, 16, tzinfo=UTC)
    assert min(katl_fxx) == 0
    assert max(katl_fxx) == 12

    kdal_as_of = sdk_pipeline.forecast_as_of_for_timing("2026-01-15", "America/Chicago", "same_day_11am")
    kdal_start, kdal_end = sdk_pipeline.forecast_window_for_timing(
        "2026-01-15",
        "America/Chicago",
        "same_day_11am",
        kdal_as_of,
    )
    kdal_cycle, _ = sdk_pipeline.choose_cycle(
        "hrrr",
        "2026-01-15",
        "America/Chicago",
        kdal_as_of,
        timing_mode="same_day_11am",
        forecast_window_start=kdal_start,
        forecast_window_end=kdal_end,
    )
    assert kdal_as_of == datetime(2026, 1, 15, 17, tzinfo=UTC)
    assert kdal_cycle == datetime(2026, 1, 15, 17, tzinfo=UTC)

    klax_as_of = sdk_pipeline.forecast_as_of_for_timing("2026-01-15", "America/Los_Angeles", "same_day_11am")
    klax_start, klax_end = sdk_pipeline.forecast_window_for_timing(
        "2026-01-15",
        "America/Los_Angeles",
        "same_day_11am",
        klax_as_of,
    )
    gfs_cycle, gfs_fxx = sdk_pipeline.choose_cycle(
        "gfs",
        "2026-01-15",
        "America/Los_Angeles",
        klax_as_of,
        timing_mode="same_day_11am",
        forecast_window_start=klax_start,
        forecast_window_end=klax_end,
    )
    assert klax_as_of == datetime(2026, 1, 15, 19, tzinfo=UTC)
    assert gfs_cycle == datetime(2026, 1, 15, 18, tzinfo=UTC)
    assert min(gfs_fxx) == 1
    assert max(gfs_fxx) == 13

    nbm_cycle, nbm_fxx = sdk_pipeline.choose_cycle(
        "nbm",
        "2026-01-15",
        "America/New_York",
        katl_as_of,
        timing_mode="same_day_11am",
        forecast_window_start=katl_start,
        forecast_window_end=katl_end,
    )
    assert nbm_cycle == datetime(2026, 1, 15, 15, tzinfo=UTC)
    assert min(nbm_fxx) == 1
    assert max(nbm_fxx) == 13


def test_same_day_11am_window_excludes_pre_11am_hours(monkeypatch) -> None:
    monkeypatch.setattr(sdk_pipeline, "model_cycle_hours", lambda model: tuple(range(24)))
    as_of = sdk_pipeline.forecast_as_of_for_timing("2026-07-15", "America/New_York", "same_day_11am")
    start, end = sdk_pipeline.forecast_window_for_timing("2026-07-15", "America/New_York", "same_day_11am", as_of)
    cycle, fxx = sdk_pipeline.choose_cycle(
        "nbm",
        "2026-07-15",
        "America/New_York",
        as_of,
        timing_mode="same_day_11am",
        forecast_window_start=start,
        forecast_window_end=end,
    )
    assert as_of == datetime(2026, 7, 15, 15, tzinfo=UTC)
    assert start == as_of
    assert end == datetime(2026, 7, 16, 4, tzinfo=UTC)
    assert cycle == datetime(2026, 7, 15, 14, tzinfo=UTC)
    assert min(fxx) == 1
    assert max(fxx) == 13
    assert all(cycle + timedelta(hours=hour) >= as_of for hour in fxx)
    assert all(cycle + timedelta(hours=hour) < end for hour in fxx)


def test_nbm_record_filter_drops_ensemble_std_dev_duplicates() -> None:
    records = [
        SimpleNamespace(record_no=134, variable="TMP", level="2 m above ground", forecast_period="1 hour fcst"),
        SimpleNamespace(
            record_no=135,
            variable="TMP",
            level="2 m above ground",
            forecast_period="1 hour fcst:ens std dev",
        ),
        SimpleNamespace(record_no=35, variable="DPT", level="2 m above ground", forecast_period="1 hour fcst"),
        SimpleNamespace(
            record_no=36,
            variable="DPT",
            level="2 m above ground",
            forecast_period="1 hour fcst:ens std dev",
        ),
    ]
    clean = sdk_pipeline._drop_ensemble_std_dev_records(records)
    assert [record.record_no for record in clean] == [35, 134]

    parsed_like_sdk = [
        SimpleNamespace(record_no=134, variable="TMP", level="2 m above ground", forecast_period="1 hour fcst"),
        SimpleNamespace(record_no=135, variable="TMP", level="2 m above ground", forecast_period="1 hour fcst"),
    ]
    assert [record.record_no for record in sdk_pipeline._drop_ensemble_std_dev_records(parsed_like_sdk)] == [134]


def test_nbm_same_day_11am_recent_date_gate() -> None:
    today = date(2026, 5, 29)
    assert sdk_pipeline.nbm_same_day_11am_supported_date(date(2026, 5, 28), today)
    assert sdk_pipeline.nbm_same_day_11am_supported_date(date(2026, 5, 27), today)
    assert not sdk_pipeline.nbm_same_day_11am_supported_date(date(2026, 5, 26), today)
    assert not sdk_pipeline.nbm_same_day_11am_supported_date(date(2026, 5, 29), today)


def test_batched_nwp_backfill_writes_multiple_station_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_NWP_SUBPROCESS", "0")
    calls: list[tuple[list[str], int]] = []

    def fake_forecast_nwp(station, model, *, cycle, fxx, **kwargs):
        stations = station if isinstance(station, list) else [station]
        calls.append((stations, fxx))
        return pd.DataFrame(
            {
                "station": stations,
                "valid_at": [(cycle + timedelta(hours=fxx)).isoformat()] * len(stations),
                "temp_k_2m": [295.0 + index + fxx / 100 for index, _ in enumerate(stations)],
                "dewpoint_k_2m": [290.0] * len(stations),
                "wind_u_ms_10m": [3.0] * len(stations),
                "wind_v_ms_10m": [4.0] * len(stations),
                "relative_humidity_pct_2m": [50.0] * len(stations),
                "grid_dist_km": [1.0] * len(stations),
            }
        )

    monkeypatch.setattr(sdk_pipeline, "_load_forecast_nwp", lambda station_meta: fake_forecast_nwp)
    frame = sdk_pipeline.backfill_sdk_nwp(
        sdk_cache_dir=tmp_path,
        stations=["KATL", "KLGA"],
        models=["hrrr"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        timing_mode="fresh_after_6am",
        max_batches=1,
        fxx_workers=3,
    )
    assert len(frame) == 2
    assert len(calls) > 1
    assert all(isinstance(stations, list) and len(stations) == 2 for stations, _ in calls)
    assert set(frame["station_id"]) == {"KATL", "KLGA"}
    assert set(frame["timing_mode"]) == {"fresh_after_6am"}


def test_nwp_summary_populates_maximal_common_features() -> None:
    request = sdk_pipeline.NwpRequest(
        station_id="KATL",
        station_name="Atlanta",
        airport_name="Atlanta",
        timezone="America/New_York",
        contract_date="2026-01-01",
        model="gfs",
        forecast_as_of=datetime(2026, 1, 1, 16, tzinfo=UTC),
        cycle=datetime(2026, 1, 1, 15, tzinfo=UTC),
        fxx_hours=(1, 2),
        timing_mode="same_day_11am",
        cycle_selection_policy="test",
        forecast_window_start=datetime(2026, 1, 1, 16, tzinfo=UTC),
        forecast_window_end=datetime(2026, 1, 2, 5, tzinfo=UTC),
    )
    hourly = pd.DataFrame(
        {
            "station": ["KATL", "KATL"],
            "valid_at": ["2026-01-01T16:00:00+00:00", "2026-01-01T17:00:00+00:00"],
            "temp_k_2m": [300.0, 302.0],
            "dewpoint_k_2m": [290.0, 291.0],
            "wind_u_ms_10m": [3.0, 4.0],
            "wind_v_ms_10m": [4.0, 3.0],
            "wind_gust_ms": [8.0, 9.0],
            "relative_humidity_pct_2m": [50.0, 60.0],
            "precip_mm_1h": [1.0, 2.0],
            "pressure_pa_mslp": [101000.0, 101100.0],
            "pressure_pa_surface": [99000.0, 99100.0],
            "grid_dist_km": [1.0, 1.2],
        }
    )
    row = sdk_pipeline._summarize_nwp_request(request, hourly)
    assert round(row["forecast_temp_at_as_of_f"], 2) == 80.33
    assert row["precip_amount"] == 3.0
    assert round(row["wind_speed_at_as_of"], 2) == 11.18
    assert round(row["wind_gust_max"], 2) == 20.13
    assert row["humidity_at_as_of"] == 50.0
    assert row["pressure_mslp_mean"] == 101050.0


def test_batched_nwp_fetch_keeps_partial_hours_when_one_fxx_fails(monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_NWP_SUBPROCESS", "0")
    monkeypatch.setattr(sdk_pipeline, "_nwp_fxx_fetch_retries", lambda: 0)
    request = sdk_pipeline.NwpRequest(
        station_id="KATL",
        station_name="Atlanta",
        airport_name="Atlanta",
        timezone="America/New_York",
        contract_date="2026-01-01",
        model="nbm",
        forecast_as_of=datetime(2026, 1, 1, 16, tzinfo=UTC),
        cycle=datetime(2026, 1, 1, 15, tzinfo=UTC),
        fxx_hours=(1, 2, 3),
        timing_mode="same_day_11am",
        cycle_selection_policy="test",
        forecast_window_start=datetime(2026, 1, 1, 16, tzinfo=UTC),
        forecast_window_end=datetime(2026, 1, 2, 5, tzinfo=UTC),
    )

    def fake_forecast_nwp(station, model, *, cycle, fxx, **kwargs):
        if fxx == 2:
            raise RuntimeError("Server disconnected without sending a response.")
        stations = station if isinstance(station, list) else [station]
        return pd.DataFrame(
            {
                "station": stations,
                "valid_at": [(cycle + timedelta(hours=fxx)).isoformat()] * len(stations),
                "temp_k_2m": [300.0 + fxx] * len(stations),
                "dewpoint_k_2m": [290.0] * len(stations),
                "wind_u_ms_10m": [3.0] * len(stations),
                "wind_v_ms_10m": [4.0] * len(stations),
                "relative_humidity_pct_2m": [50.0] * len(stations),
            }
        )

    rows = sdk_pipeline._fetch_and_summarize_nwp_batch(fake_forecast_nwp, [request])
    assert len(rows) == 1
    assert rows[0]["fetch_status"] == "ok"
    assert rows[0]["forecast_hour_fetch_status"] == "partial"
    assert rows[0]["forecast_hour_count_requested"] == 3
    assert rows[0]["forecast_hour_count_returned"] == 2
    assert rows[0]["forecast_hour_missing"] == "2"


def test_nbm_uses_subprocess_fetch_by_default() -> None:
    request = sdk_pipeline.NwpRequest(
        station_id="KATL",
        station_name="Atlanta",
        airport_name="Atlanta",
        timezone="America/New_York",
        contract_date="2025-01-01",
        model="nbm",
        forecast_as_of=datetime(2025, 1, 1, 16, tzinfo=UTC),
        cycle=datetime(2025, 1, 1, 15, tzinfo=UTC),
        fxx_hours=(1,),
        timing_mode="same_day_11am",
        cycle_selection_policy="test",
        forecast_window_start=datetime(2025, 1, 1, 16, tzinfo=UTC),
        forecast_window_end=datetime(2025, 1, 2, 5, tzinfo=UTC),
    )

    assert sdk_pipeline._use_nbm_subprocess_fetch(request)


def test_hrrr_uses_subprocess_fetch_by_default() -> None:
    request = sdk_pipeline.NwpRequest(
        station_id="KATL",
        station_name="Atlanta",
        airport_name="Atlanta",
        timezone="America/New_York",
        contract_date="2025-01-01",
        model="hrrr",
        forecast_as_of=datetime(2025, 1, 1, 16, tzinfo=UTC),
        cycle=datetime(2025, 1, 1, 15, tzinfo=UTC),
        fxx_hours=(1,),
        timing_mode="same_day_11am",
        cycle_selection_policy="test",
        forecast_window_start=datetime(2025, 1, 1, 16, tzinfo=UTC),
        forecast_window_end=datetime(2025, 1, 2, 5, tzinfo=UTC),
    )

    assert sdk_pipeline._use_nwp_subprocess_fetch(request)


def test_nbm_variable_patch_uses_lean_stable_feature_set_by_default(monkeypatch) -> None:
    monkeypatch.delenv("WEATHER_RESEARCH_NBM_ENABLE_WIND", raising=False)
    variable_map = {
        "temp_k_2m": ("TMP", "2 m above ground"),
        "dewpoint_k_2m": ("DPT", "2 m above ground"),
        "relative_humidity_pct_2m": ("RH", "2 m above ground"),
        "wind_u_ms_10m": ("UGRD", "10 m above ground"),
        "wind_v_ms_10m": ("VGRD", "10 m above ground"),
        "wind_gust_ms": ("GUST", "10 m above ground"),
        "precip_mm_1h": ("APCP", "surface"),
        "pressure_pa_mslp": ("MSLMA", "mean sea level"),
    }

    sdk_pipeline._patch_nbm_variable_map(variable_map)

    assert "wind_u_ms_10m" not in variable_map
    assert "wind_v_ms_10m" not in variable_map
    assert "pressure_pa_mslp" not in variable_map
    assert {"temp_k_2m", "dewpoint_k_2m", "relative_humidity_pct_2m", "wind_gust_ms", "precip_mm_1h"}.issubset(variable_map)


def test_nbm_subprocess_code_falls_back_to_station_fetches() -> None:
    assert "except Exception:" in sdk_pipeline._NBM_SUBPROCESS_CODE
    assert "for station in stations:" in sdk_pipeline._NBM_SUBPROCESS_CODE
    assert "forecast_nwp([station]" in sdk_pipeline._NBM_SUBPROCESS_CODE


def test_fresh_after_6am_backfill_supports_gfs_and_nbm(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_NWP_SUBPROCESS", "0")
    calls: list[tuple[str, list[str], int]] = []

    def fake_forecast_nwp(station, model, *, cycle, fxx, **kwargs):
        stations = station if isinstance(station, list) else [station]
        calls.append((model, stations, fxx))
        return pd.DataFrame(
            {
                "station": stations,
                "valid_at": [(cycle + timedelta(hours=fxx)).isoformat()] * len(stations),
                "temp_k_2m": [296.0 + fxx / 100] * len(stations),
                "dewpoint_k_2m": [290.0] * len(stations),
                "wind_u_ms_10m": [3.0] * len(stations),
                "wind_v_ms_10m": [4.0] * len(stations),
                "relative_humidity_pct_2m": [50.0] * len(stations),
                "grid_dist_km": [1.0] * len(stations),
            }
        )

    monkeypatch.setattr(sdk_pipeline, "_load_forecast_nwp", lambda station_meta: fake_forecast_nwp)
    frame = sdk_pipeline.backfill_sdk_nwp(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        models=["gfs", "nbm"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        timing_mode="fresh_after_6am",
        max_batches=2,
    )
    assert set(frame["provider"]) == {"gfs", "nbm"}
    assert set(frame["timing_mode"]) == {"fresh_after_6am"}
    assert {model for model, _, _ in calls} == {"gfs", "nbm"}


def test_same_day_11am_backfill_supports_all_nwp_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_NWP_SUBPROCESS", "0")
    calls: list[tuple[str, list[str], int]] = []

    def fake_forecast_nwp(station, model, *, cycle, fxx, **kwargs):
        stations = station if isinstance(station, list) else [station]
        calls.append((model, stations, fxx))
        return pd.DataFrame(
            {
                "station": stations,
                "valid_at": [(cycle + timedelta(hours=fxx)).isoformat()] * len(stations),
                "temp_k_2m": [294.0 + fxx / 100] * len(stations),
                "dewpoint_k_2m": [288.0] * len(stations),
                "wind_u_ms_10m": [2.0] * len(stations),
                "wind_v_ms_10m": [3.0] * len(stations),
                "relative_humidity_pct_2m": [55.0] * len(stations),
                "grid_dist_km": [1.0] * len(stations),
            }
        )

    monkeypatch.setattr(sdk_pipeline, "_load_forecast_nwp", lambda station_meta: fake_forecast_nwp)
    monkeypatch.setattr(
        sdk_pipeline,
        "nwp_archive_starts",
        lambda: {"hrrr": datetime(2014, 7, 30).date(), "gfs": datetime(2021, 1, 1).date(), "nbm": datetime(2020, 1, 1).date()},
    )
    monkeypatch.setattr(sdk_pipeline, "nbm_same_day_11am_supported_date", lambda contract_day: True)
    frame = sdk_pipeline.backfill_sdk_nwp(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        models=["hrrr", "gfs", "nbm"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        timing_mode="same_day_11am",
        max_batches=3,
    )
    assert set(frame["provider"]) == {"hrrr", "gfs", "nbm"}
    assert set(frame["timing_mode"]) == {"same_day_11am"}
    assert {model for model, _, _ in calls} == {"hrrr", "gfs", "nbm"}


def test_sdk_nwp_weather_feature_enrichment_revisits_old_ok_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_NWP_SUBPROCESS", "0")
    calls: list[int] = []
    cache = tmp_path / sdk_pipeline.SDK_NWP_FILE
    pd.DataFrame(
        [
            {
                "station_id": "KATL",
                "provider": "gfs",
                "model": "gfs",
                "timing_mode": "same_day_11am",
                "contract_date": "2026-01-01",
                "raw_forecast_high_f": 70.0,
                "fetch_status": "ok",
            }
        ]
    ).to_csv(cache, index=False)

    def fake_forecast_nwp(station, model, *, cycle, fxx, **kwargs):
        calls.append(fxx)
        return pd.DataFrame(
            {
                "station": ["KATL"],
                "valid_at": [(cycle + timedelta(hours=fxx)).isoformat()],
                "temp_k_2m": [294.0 + fxx / 100],
                "dewpoint_k_2m": [288.0],
                "wind_u_ms_10m": [2.0],
                "wind_v_ms_10m": [3.0],
                "relative_humidity_pct_2m": [55.0],
                "grid_dist_km": [1.0],
            }
        )

    monkeypatch.setattr(sdk_pipeline, "_load_forecast_nwp", lambda station_meta: fake_forecast_nwp)
    frame = sdk_pipeline.backfill_sdk_nwp(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        models=["gfs"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        timing_mode="same_day_11am",
        include_weather_features=True,
    )

    assert calls
    row = frame.iloc[0]
    assert row[sdk_pipeline.WEATHER_FEATURE_FLAG] is True or str(row[sdk_pipeline.WEATHER_FEATURE_FLAG]).lower() == "true"
    assert round(row["dewpoint_mean_f"], 2) == 58.73

    sdk_pipeline.backfill_sdk_nwp(
        sdk_cache_dir=tmp_path,
        stations=["KATL"],
        models=["gfs"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        timing_mode="same_day_11am",
        include_weather_features=True,
    )
    assert len(calls) == len(set(calls))


def test_direct_nbm_backfill_batches_stations_and_labels_source(tmp_path, monkeypatch) -> None:
    calls: list[tuple[list[str], tuple[int, ...]]] = []

    def fake_extract(stations, settings, raw_dir, issue_utc, fxx_hours, force_refresh, variable):
        calls.append((sorted(stations), tuple(fxx_hours)))
        return {
            station: {fxx: 70.0 + index + fxx / 100 for fxx in fxx_hours}
            for index, station in enumerate(sorted(stations))
        }

    monkeypatch.setattr("src.nws_fetch._extract_nbm_run_points", fake_extract)
    frame = sdk_pipeline.backfill_direct_nbm(
        cache_dir=tmp_path,
        stations=["KATL", "KLGA"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        max_batches=1,
    )
    assert len(frame) == 2
    assert calls
    assert calls[0][0] == ["KATL", "KLGA"]
    assert set(frame["provider"]) == {"nbm"}
    assert set(frame["model"]) == {"nbm"}
    assert set(frame["timing_mode"]) == {"same_day_11am"}
    assert set(frame["data_source"]) == {"direct_noaa_nbm_archive_grib2"}
    assert str(frame.iloc[0]["source_label"]).startswith("noaa_nbm_archive_tmp_")


def test_direct_nbm_backfill_can_include_weather_features(tmp_path, monkeypatch) -> None:
    def fake_extract(stations, settings, raw_dir, issue_utc, fxx_hours, force_refresh, feature_fields=None):
        return {
            station: {
                fxx: {
                    "temp_k_2m": 300.0 + fxx / 100,
                    "dewpoint_k_2m": 290.0,
                    "relative_humidity_pct_2m": 55.0,
                    "precip_mm_1h": 0.5,
                    "cloud_cover_pct": 70.0,
                    "wind_speed_ms_10m": 5.0,
                    "wind_direction_deg_10m": 180.0,
                    "wind_gust_ms": 8.0,
                    "ceiling_m": 1000.0,
                    "visibility_m": 10000.0,
                }
                for fxx in fxx_hours
            }
            for station in stations
        }

    monkeypatch.setattr("src.nws_fetch._extract_nbm_run_feature_points", fake_extract)
    frame = sdk_pipeline.backfill_direct_nbm(
        cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        include_weather_features=True,
    )
    row = frame.iloc[0]
    assert row["fetch_status"] == "ok"
    assert row["cloud_cover_mean"] == 70.0
    assert row["precip_amount"] > 0
    assert row["humidity_at_as_of"] == 55.0
    assert round(row["wind_speed_at_as_of"], 2) == 11.18
    assert row["ceiling_min"] == 1000.0


def test_direct_nbm_weather_feature_enrichment_revisits_core_rows_once(tmp_path, monkeypatch) -> None:
    point_calls = 0
    feature_calls = 0

    def fake_extract_points(stations, settings, raw_dir, issue_utc, fxx_hours, force_refresh, variable):
        nonlocal point_calls
        point_calls += 1
        return {station: {fxx: 72.0 + fxx / 100 for fxx in fxx_hours} for station in stations}

    def fake_extract_features(stations, settings, raw_dir, issue_utc, fxx_hours, force_refresh, feature_fields=None):
        nonlocal feature_calls
        feature_calls += 1
        return {
            station: {
                fxx: {
                    "temp_k_2m": 295.0 + fxx / 100,
                    "dewpoint_k_2m": 285.0,
                    "relative_humidity_pct_2m": 60.0,
                    "precip_mm_1h": 0.25,
                    "cloud_cover_pct": 80.0,
                    "wind_speed_ms_10m": 4.0,
                    "wind_direction_deg_10m": 190.0,
                    "wind_gust_ms": 7.0,
                    "ceiling_m": 900.0,
                    "visibility_m": 12000.0,
                }
                for fxx in fxx_hours
            }
            for station in stations
        }

    monkeypatch.setattr("src.nws_fetch._extract_nbm_run_points", fake_extract_points)
    monkeypatch.setattr("src.nws_fetch._extract_nbm_run_feature_points", fake_extract_features)

    core = sdk_pipeline.backfill_direct_nbm(
        cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-01",
    )
    assert len(core) == 1
    assert point_calls == 1
    assert not bool(core.iloc[0][sdk_pipeline.DIRECT_NBM_WEATHER_FEATURE_FLAG])

    enriched = sdk_pipeline.backfill_direct_nbm(
        cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        include_weather_features=True,
    )
    row = enriched.iloc[0]
    assert len(enriched) == 1
    assert feature_calls == 1
    assert bool(row[sdk_pipeline.DIRECT_NBM_WEATHER_FEATURE_FLAG])
    assert row["cloud_cover_mean"] == 80.0
    assert row["humidity_at_as_of"] == 60.0

    sdk_pipeline.backfill_direct_nbm(
        cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-01",
        include_weather_features=True,
    )
    assert feature_calls == 1


def test_direct_nbm_resume_skips_completed_rows(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_extract(stations, settings, raw_dir, issue_utc, fxx_hours, force_refresh, variable):
        nonlocal calls
        calls += 1
        return {station: {fxx: 72.0 for fxx in fxx_hours} for station in stations}

    monkeypatch.setattr("src.nws_fetch._extract_nbm_run_points", fake_extract)
    sdk_pipeline.backfill_direct_nbm(
        cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-01",
    )
    sdk_pipeline.backfill_direct_nbm(
        cache_dir=tmp_path,
        stations=["KATL"],
        start_date="2026-01-01",
        end_date="2026-01-01",
    )
    assert calls == 1


def test_sdk_dataset_builder_uses_only_sdk_caches(tmp_path) -> None:
    sdk_dir = tmp_path / "data" / "calibration" / "sdk"
    sdk_dir.mkdir(parents=True)
    sdk_pipeline.write_station_registry(sdk_dir, ["KATL"])
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-01-01"],
            "actual_high_f": [75.0],
            "fetch_status": ["ok"],
        }
    ).to_csv(sdk_dir / sdk_pipeline.SDK_ACTUALS_FILE, index=False)
    pd.DataFrame(
        {
            "station_id": ["KATL", "KATL", "KATL"],
            "provider": ["hrrr", "gfs", "nbm"],
            "model": ["hrrr", "gfs", "nbm"],
            "timing_mode": ["same_day_11am", "same_day_11am", "same_day_11am"],
            "contract_date": ["2026-01-01", "2026-01-01", "2026-01-01"],
            "forecast_as_of": ["2026-01-01T16:00:00+00:00"] * 3,
            "issued_at": ["2026-01-01T15:00:00+00:00"] * 3,
            "raw_forecast_high_f": [73.0, 74.0, 72.0],
            "dewpoint_mean_f": [60.0, 61.0, 62.0],
            "humidity_mean": [50.0, 51.0, 52.0],
            "wind_speed_mean": [5.0, 6.0, pd.NA],
            "wind_speed_max": [7.0, 8.0, pd.NA],
            "fetch_status": ["ok", "ok", "ok"],
        }
    ).to_csv(sdk_dir / sdk_pipeline.SDK_NWP_FILE, index=False)
    samples = build_calibration_samples(
        project_root=tmp_path,
        calibration_dir=tmp_path / "data" / "calibration",
        source_mode="sdk",
        sdk_cache_dir=sdk_dir,
        include_timing_modes=["same_day_11am"],
    )
    assert len(samples) == 2
    assert set(samples["provider"]) == {"gfs", "hrrr"}
    assert "nbm" not in set(samples["provider"])
    assert set(samples["timing_mode"]) == {"same_day_11am"}
    assert set(samples["data_source"]) == {"mostlyright.weather.forecast_nwp"}
    assert samples.loc[samples["provider"].eq("hrrr"), "calibration_bias_f"].iloc[0] == 2.0


def test_sdk_dataset_builder_includes_direct_nbm_cache_with_lineage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    sdk_dir = tmp_path / "data" / "calibration" / "sdk"
    sdk_dir.mkdir(parents=True)
    sdk_pipeline.write_station_registry(sdk_dir, ["KATL"])
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-01-01"],
            "actual_high_f": [75.0],
            "fetch_status": ["ok"],
        }
    ).to_csv(sdk_dir / sdk_pipeline.SDK_ACTUALS_FILE, index=False)
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "provider": ["nbm"],
            "model": ["nbm"],
            "timing_mode": ["same_day_11am"],
            "contract_date": ["2026-01-01"],
            "forecast_as_of": ["2026-01-01T16:00:00+00:00"],
            "issued_at": ["2026-01-01T15:00:00+00:00"],
            "raw_forecast_high_f": [73.0],
            "data_source": ["direct_noaa_nbm_archive_grib2"],
            "fetch_status": ["ok"],
        }
    ).to_csv(sdk_dir / sdk_pipeline.DIRECT_NBM_FILE, index=False)
    samples = build_calibration_samples(
        project_root=tmp_path,
        calibration_dir=tmp_path / "data" / "calibration",
        source_mode="sdk",
        sdk_cache_dir=sdk_dir,
        include_providers=["nbm"],
        include_timing_modes=["same_day_11am"],
    )
    assert len(samples) == 1
    assert samples.iloc[0]["provider"] == "nbm"
    assert samples.iloc[0]["model"] == "nbm"
    assert samples.iloc[0]["data_source"] == "direct_noaa_nbm_archive_grib2"
    assert samples.iloc[0]["calibration_bias_f"] == 2.0


def test_sdk_dataset_builder_joins_safe_11am_observed_features(tmp_path) -> None:
    sdk_dir = tmp_path / "data" / "calibration" / "sdk"
    obs_dir = sdk_dir / "sdk_current_obs_2021_2026"
    obs_dir.mkdir(parents=True)
    sdk_pipeline.write_station_registry(sdk_dir, ["KATL"])
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-01-01"],
            "actual_high_f": [80.0],
            "fetch_status": ["ok"],
        }
    ).to_csv(sdk_dir / sdk_pipeline.SDK_ACTUALS_FILE, index=False)
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "provider": ["hrrr"],
            "model": ["hrrr"],
            "timing_mode": ["same_day_11am"],
            "contract_date": ["2026-01-01"],
            "forecast_as_of": ["2026-01-01T16:00:00+00:00"],
            "issued_at": ["2026-01-01T15:00:00+00:00"],
            "raw_forecast_high_f": [78.0],
            "dewpoint_mean_f": [60.0],
            "humidity_mean": [55.0],
            "wind_speed_mean": [4.0],
            "wind_speed_max": [8.0],
            "fetch_status": ["ok"],
        }
    ).to_csv(sdk_dir / sdk_pipeline.SDK_NWP_FILE, index=False)
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-01-01"],
            "timing_mode": ["same_day_11am"],
            "observed_fetch_status": ["ok"],
            "observed_temp_at_as_of_f": [70.0],
            "observed_dewpoint_at_as_of_f": [59.0],
            "observed_humidity_at_as_of": [68.0],
            "observed_wind_speed_at_as_of": [6.0],
            "observed_pressure_at_as_of": [1012.0],
            "observed_visibility_at_as_of": [10.0],
            "observed_as_of_age_minutes": [5],
        }
    ).to_csv(obs_dir / "sdk_current_observations_11am.csv", index=False)

    samples = build_calibration_samples(
        project_root=tmp_path,
        calibration_dir=tmp_path / "data" / "calibration",
        source_mode="sdk",
        sdk_cache_dir=sdk_dir,
        include_timing_modes=["same_day_11am"],
    )

    assert len(samples) == 1
    row = samples.iloc[0]
    assert row["actual_high_f"] == 80.0
    assert row["calibration_bias_f"] == 2.0
    assert row["observed_temp_at_as_of_f"] == 70.0
    assert row["observed_as_of_age_minutes"] == 5


def test_merge_sdk_nwp_shards_keeps_latest_by_timing_mode(tmp_path) -> None:
    shard_one = tmp_path / "shard_one"
    shard_two = tmp_path / "shard_two"
    shard_one.mkdir()
    shard_two.mkdir()
    pd.DataFrame(
        {
            "station_id": ["KATL", "KATL"],
            "contract_date": ["2024-01-01", "2024-01-01"],
            "provider": ["hrrr", "hrrr"],
            "model": ["hrrr", "hrrr"],
            "timing_mode": ["fresh_after_6am", "strict_6am"],
            "raw_forecast_high_f": [50.0, 49.0],
            "fetch_status": ["ok", "ok"],
        }
    ).to_csv(shard_one / sdk_pipeline.SDK_NWP_FILE, index=False)
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2024-01-01"],
            "provider": ["hrrr"],
            "model": ["hrrr"],
            "timing_mode": ["fresh_after_6am"],
            "raw_forecast_high_f": [51.0],
            "fetch_status": ["ok"],
        }
    ).to_csv(shard_two / sdk_pipeline.SDK_NWP_FILE, index=False)
    merged = merge_sdk_nwp_shards(
        shard_dirs=[shard_one, shard_two],
        output_sdk_cache_dir=tmp_path / "main",
    )
    assert len(merged) == 2
    fresh = merged.loc[merged["timing_mode"] == "fresh_after_6am"].iloc[0]
    assert fresh["raw_forecast_high_f"] == 51.0
