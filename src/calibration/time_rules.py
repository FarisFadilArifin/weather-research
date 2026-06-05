from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo


ZERO_HOUR_LOCAL_HOUR = 6


def parse_contract_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def forecast_as_of_local(contract_date: str | date | datetime, timezone: str) -> datetime:
    day = parse_contract_date(contract_date)
    return datetime.combine(day, datetime.min.time(), tzinfo=ZoneInfo(timezone)).replace(
        hour=ZERO_HOUR_LOCAL_HOUR
    )


def forecast_as_of_utc(contract_date: str | date | datetime, timezone: str) -> datetime:
    return forecast_as_of_local(contract_date, timezone).astimezone(UTC)


def local_day_utc_bounds(contract_date: str | date | datetime, timezone: str) -> tuple[datetime, datetime]:
    day = parse_contract_date(contract_date)
    start_local = datetime.combine(day, datetime.min.time(), tzinfo=ZoneInfo(timezone))
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def forecast_hours_for_local_day(cycle_utc: datetime, contract_date: str | date | datetime, timezone: str) -> list[int]:
    if cycle_utc.tzinfo is None or cycle_utc.tzinfo.utcoffset(cycle_utc) is None:
        raise ValueError("cycle_utc must be timezone-aware")
    cycle_utc = cycle_utc.astimezone(UTC)
    start_utc, end_utc = local_day_utc_bounds(contract_date, timezone)
    hours: list[int] = []
    valid = start_utc
    while valid < end_utc:
        fxx = int((valid - cycle_utc).total_seconds() // 3600)
        if fxx >= 0:
            hours.append(fxx)
        valid += timedelta(hours=1)
    return hours
