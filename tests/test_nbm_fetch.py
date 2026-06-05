from datetime import UTC, datetime

import pytest
import requests

from src.nws_fetch import (
    NBM_TMAX,
    TransientNbmDownloadError,
    _choose_nbm_tmax_issue_time,
    _nbm_byte_ranges,
    _nbm_get_with_retries,
    nbm_file_url,
)


def test_nbm_file_url_uses_archive_pattern():
    url = nbm_file_url(
        "https://noaa-nbm-grib2-pds.s3.amazonaws.com",
        datetime(2026, 5, 18, 0, tzinfo=UTC),
        24,
    )
    assert url == (
        "https://noaa-nbm-grib2-pds.s3.amazonaws.com/"
        "blend.20260518/00/core/blend.t00z.core.f024.co.grib2"
    )


def test_nbm_byte_ranges_select_deterministic_temperature_only():
    idx = "\n".join(
        [
            "1:0:d=2026051800:APTMP:2 m above ground:1 hour fcst:",
            "2:100:d=2026051800:TCDC:entire atmosphere:1 hour fcst:",
            "3:220:d=2026051800:TMP:2 m above ground:1 hour fcst:",
            "4:400:d=2026051800:TMP:2 m above ground:1 hour fcst:ens std dev",
            "5:500:d=2026051800:DPT:2 m above ground:1 hour fcst:",
        ]
    )
    assert _nbm_byte_ranges(idx) == [(220, 399)]


def test_nbm_byte_ranges_select_tmax_when_requested():
    idx = "\n".join(
        [
            "69:100:d=2026051800:MAXREF:1000 m above ground:23-24 hour max fcst:",
            "70:220:d=2026051800:TMAX:2 m above ground:12-24 hour max fcst:",
            "71:400:d=2026051800:TMAX:2 m above ground:12-24 hour max fcst:ens std dev",
            "72:500:d=2026051800:TMP:2 m above ground:24 hour fcst:",
        ]
    )
    assert _nbm_byte_ranges(idx, variable=NBM_TMAX) == [(220, 399)]


def test_nbm_byte_ranges_select_non_probability_feature_field():
    idx = "\n".join(
        [
            "1:100:d=2026051800:APCP:surface:0-1 hour acc fcst:prob >0.254:prob fcst 255/255",
            "2:220:d=2026051800:APCP:surface:0-1 hour acc fcst:",
            "3:400:d=2026051800:TCDC:surface:1 hour fcst:",
            "4:600:d=2026051800:TCDC:surface:1 hour fcst:ens std dev",
            "5:800:d=2026051800:VIS:surface:1 hour fcst:prob <1609.34:probability forecast",
            "6:1000:d=2026051800:VIS:surface:1 hour fcst:",
            "7:1200:d=2026051800:WIND:10 m above ground:1 hour fcst:",
            "8:1400:d=2026051800:WIND:10 m above ground:1 hour fcst:ens std dev",
            "9:1600:d=2026051800:WDIR:10 m above ground:1 hour fcst:",
            "10:1800:d=2026051800:CEIL:cloud ceiling:1 hour fcst:",
            "11:2000:d=2026051800:CEIL:cloud ceiling:1 hour fcst:prob <304.8:probability forecast",
        ]
    )
    assert _nbm_byte_ranges(idx, variable="APCP", level="surface") == [(220, 399)]
    assert _nbm_byte_ranges(idx, variable="TCDC", level="surface") == [(400, 599)]
    assert _nbm_byte_ranges(idx, variable="VIS", level="surface") == [(1000, 1199)]
    assert _nbm_byte_ranges(idx, variable="WIND", level="10 m above ground") == [(1200, 1399)]
    assert _nbm_byte_ranges(idx, variable="WDIR", level="10 m above ground") == [(1600, 1799)]
    assert _nbm_byte_ranges(idx, variable="CEIL", level="cloud ceiling") == [(1800, 1999)]


def test_choose_nbm_tmax_uses_single_field_ending_at_00z():
    issue, fxx_hours = _choose_nbm_tmax_issue_time(
        datetime(2026, 5, 17, 4, tzinfo=UTC),
        "2026-05-18",
        {"nws": {}},
        72,
        12,
    )
    assert issue == datetime(2026, 5, 17, 4, tzinfo=UTC)
    assert fxx_hours == [44]


def test_choose_nbm_tmax_respects_allowed_cycles():
    issue, fxx_hours = _choose_nbm_tmax_issue_time(
        datetime(2026, 5, 17, 4, tzinfo=UTC),
        "2026-05-18",
        {"nws": {"nbm_allowed_cycle_hours": [0, 6, 12, 18]}},
        72,
        12,
    )
    assert issue == datetime(2026, 5, 17, 0, tzinfo=UTC)
    assert fxx_hours == [48]


def test_nbm_get_with_retries_aborts_on_dns_failure(monkeypatch):
    calls = 0

    def fail_dns(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise requests.exceptions.ConnectionError("DNS failed")

    monkeypatch.setattr(requests, "get", fail_dns)

    with pytest.raises(TransientNbmDownloadError, match="rerun later"):
        _nbm_get_with_retries(
            "https://noaa-nbm-grib2-pds.s3.amazonaws.com/test.idx",
            {"nws": {"nbm_download_retries": 2, "nbm_retry_backoff_seconds": 0}},
            timeout=1,
            headers={"User-Agent": "test"},
        )

    assert calls == 2
