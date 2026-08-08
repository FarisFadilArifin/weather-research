from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, time
import json
from pathlib import Path
import re
import sys
import time as time_module
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.weather_no_walkforward import (
    YES_PROBABILITY_MAXIMUMS,
    economic_metrics,
    filter_candidates,
    frozen_median_parameters,
    monthly_walk_forward,
    parameter_grid,
)


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
ENTRY_TIME = time(11, 15)
BASE_RISK_USDC = 4.0
HIGH_RISK_USDC = 6.0
HIGH_EDGE_THRESHOLD = 0.15
EXECUTION_PENALTY = 0.01
REPORT_ROOT = PROJECT_ROOT / "reports" / "dallas_tokyo_no_walkforward"
CACHE_ROOT = PROJECT_ROOT / "data" / "polymarket" / "dallas_tokyo_no_walkforward"
USER_AGENT = "weather-research/0.1 dallas-tokyo-no-backtest"

STATIONS = {
    "Dallas": {
        "station_id": "KDAL",
        "tag_id": 100916,
        "slug_prefix": "highest-temperature-in-dallas-",
        "timezone": "America/Chicago",
        "unit": "F",
    },
    "Tokyo": {
        "station_id": "RJTT",
        "tag_id": 104122,
        "slug_prefix": "highest-temperature-in-tokyo-",
        "timezone": "Asia/Tokyo",
        "unit": "C",
    },
}


def request_json(url: str, *, params: Mapping[str, Any], retries: int = 6) -> Any:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=dict(params),
                timeout=60,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - bounded public API retry.
            error = exc
            if attempt + 1 < retries:
                time_module.sleep(min(12.0, 0.75 * (2**attempt)))
    raise RuntimeError(f"request failed for {url}: {error}") from error


def _jsonish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return []


def parse_bucket(label: str, unit: str) -> tuple[int | None, int | None]:
    normalized = label.replace("º", "°").strip()
    values = [int(value) for value in re.findall(r"(?<!\d)-?\d+", normalized)]
    if not values or unit not in normalized.upper():
        raise ValueError(f"unsupported {unit} bucket label: {label}")
    lower_text = normalized.lower()
    if "or below" in lower_text:
        return None, values[0]
    if "or higher" in lower_text:
        return values[0], None
    if len(values) == 2:
        return values[0], values[1]
    return values[0], values[0]


def event_markets(event: Mapping[str, Any], unit: str) -> list[dict[str, Any]]:
    rows = []
    for market in event.get("markets", []):
        label = str(market.get("groupItemTitle") or "").strip()
        lower, upper = parse_bucket(label, unit)
        outcomes = [str(value).lower() for value in _jsonish(market.get("outcomes"))]
        prices = [float(value) for value in _jsonish(market.get("outcomePrices"))]
        tokens = [str(value) for value in _jsonish(market.get("clobTokenIds"))]
        yes_index = outcomes.index("yes")
        no_index = outcomes.index("no")
        schedule = market.get("feeSchedule") or {}
        rows.append(
            {
                "market_id": str(market.get("id")),
                "market_slug": str(market.get("slug")),
                "bucket_label": label,
                "lower": lower,
                "upper": upper,
                "is_open_tail": lower is None or upper is None,
                "no_token_id": tokens[no_index],
                "settled_yes": prices[yes_index] >= 0.99,
                "fees_enabled": bool(market.get("feesEnabled")),
                "fee_rate": float(schedule.get("rate", 0.0)),
                "fee_exponent": float(schedule.get("exponent", 1.0)),
            }
        )
    return rows


def market_for_degree(markets: list[dict[str, Any]], degree: int) -> dict[str, Any]:
    matches = [
        market
        for market in markets
        if (market["lower"] is None or degree >= market["lower"])
        and (market["upper"] is None or degree <= market["upper"])
    ]
    if len(matches) != 1:
        raise ValueError(f"degree {degree} maps to {len(matches)} markets")
    return matches[0]


def map_degree_probabilities(
    probabilities: Mapping[str, float], markets: list[dict[str, Any]]
) -> dict[str, float]:
    mapped = {market["market_id"]: 0.0 for market in markets}
    for degree, probability in probabilities.items():
        market = market_for_degree(markets, int(degree))
        mapped[market["market_id"]] += float(probability)
    return mapped


def fetch_events(city: str, start: pd.Timestamp, end: pd.Timestamp, refresh: bool) -> list[dict[str, Any]]:
    config = STATIONS[city]
    cache = CACHE_ROOT / city.lower() / "events.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = request_json(
            GAMMA_EVENTS_URL,
            params={
                "tag_id": config["tag_id"],
                "closed": "true",
                "limit": 100,
                "offset": offset,
                "end_date_min": f"{start.date()}T00:00:00Z",
                "end_date_max": f"{(end + pd.Timedelta(days=1)).date()}T23:59:59Z",
            },
        )
        if not page:
            break
        events.extend(page)
        offset += len(page)
        if len(page) < 100:
            break
    selected = {
        str(event["id"]): event
        for event in events
        if str(event.get("slug", "")).startswith(str(config["slug_prefix"]))
    }
    result = sorted(selected.values(), key=lambda event: str(event.get("endDate")))
    cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def load_dallas_predictions() -> pd.DataFrame:
    root = PROJECT_ROOT / "data" / "calibration" / "station_training_baseline" / "KDAL"
    paths = sorted((root / "ordinal_challenger_v1").glob("*_2026_exploratory_predictions.csv"))
    if len(paths) != 3:
        raise ValueError(f"expected three KDAL ordinal members, found {len(paths)}")
    members = []
    for index, path in enumerate(paths):
        frame = pd.read_csv(path, parse_dates=["contract_date"])
        frame = frame[
            ["contract_date", "point_degree_f", "recommended_bucket", "shadow_trade", "degree_probabilities"]
        ].rename(
            columns={
                "point_degree_f": f"point_degree_{index}",
                "recommended_bucket": f"recommended_bucket_{index}",
                "shadow_trade": f"confidence_vote_{index}",
                "degree_probabilities": f"degree_probabilities_{index}",
            }
        )
        members.append(frame)
    result = members[0]
    for member in members[1:]:
        result = result.merge(member, on="contract_date", validate="one_to_one")
    features = pd.read_csv(
        root / "KDAL_features.csv",
        usecols=["contract_date", "provider_spread_high_f"],
        parse_dates=["contract_date"],
    )
    result = result.merge(features, on="contract_date", validate="one_to_one")
    result["point_degree"] = result["point_degree_0"].astype(int)
    result["confidence_gate_passed"] = (
        result[[f"confidence_vote_{index}" for index in range(3)]].astype(bool).sum(axis=1) >= 2
    )
    result["probability_maps"] = result.apply(
        lambda row: [json.loads(row[f"degree_probabilities_{index}"]) for index in range(3)],
        axis=1,
    )
    return result[
        [
            "contract_date", "point_degree", "confidence_gate_passed",
            "provider_spread_high_f", "probability_maps",
        ]
    ]


def load_tokyo_predictions() -> pd.DataFrame:
    root = PROJECT_ROOT / "data" / "calibration" / "station_training_baseline" / "Tokyo"
    frame = pd.read_csv(
        root / "celsius_market_probability" / "RJTT_2026_holdout_predictions.csv",
        parse_dates=["contract_date"],
    )
    features = pd.read_csv(
        root / "RJTT_features.csv",
        usecols=["contract_date", "provider_spread_high_f"],
        parse_dates=["contract_date"],
    )
    frame = frame.merge(features, on="contract_date", validate="one_to_one")
    frame["point_degree"] = frame["point_bucket_c"].astype(int)
    frame["recommended_degree"] = frame["recommended_bucket_c"].astype(int)
    frame["confidence_gate_passed"] = frame["market_probability_decision"].eq("shadow_trade")
    frame["probability_maps"] = frame["market_bucket_probabilities_c"].map(
        lambda value: [json.loads(value)]
    )
    return frame[
        [
            "contract_date", "point_degree", "recommended_degree", "confidence_gate_passed",
            "provider_spread_high_f", "probability_maps",
        ]
    ]


def entry_timestamp(date: pd.Timestamp, timezone: str) -> int:
    local = datetime.combine(date.date(), ENTRY_TIME, tzinfo=ZoneInfo(timezone))
    return int(local.astimezone(UTC).timestamp())


def build_candidates(city: str, predictions: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    config = STATIONS[city]
    events_by_date = {
        pd.to_datetime(event["endDate"], utc=True).tz_convert(None).normalize(): event
        for event in events
    }
    rows = []
    for prediction in predictions.itertuples(index=False):
        date = pd.Timestamp(prediction.contract_date).normalize()
        event = events_by_date.get(date)
        if event is None:
            continue
        markets = event_markets(event, str(config["unit"]))
        member_maps = [map_degree_probabilities(value, markets) for value in prediction.probability_maps]
        median_probabilities = {
            market["market_id"]: float(
                np.median([member[market["market_id"]] for member in member_maps])
            )
            for market in markets
        }
        recommended_market_id = max(median_probabilities, key=median_probabilities.get)
        point_market_id = market_for_degree(markets, int(prediction.point_degree))["market_id"]
        timestamp = entry_timestamp(date, str(config["timezone"]))
        for market in markets:
            probability = median_probabilities[market["market_id"]]
            if probability > 0.40:
                continue
            rows.append(
                {
                    "city": city,
                    "station_id": config["station_id"],
                    "contract_date": date,
                    "event_slug": event["slug"],
                    "market_slug": market["market_slug"],
                    "bucket_label": market["bucket_label"],
                    "model_yes_probability": probability,
                    "model_no_probability": 1.0 - probability,
                    "provider_spread_high_f": float(prediction.provider_spread_high_f),
                    "confidence_gate_passed": bool(prediction.confidence_gate_passed),
                    "is_open_tail": market["is_open_tail"],
                    "is_point_bucket": market["market_id"] == point_market_id,
                    "is_recommended_bucket": market["market_id"] == recommended_market_id,
                    "no_token_id": market["no_token_id"],
                    "no_win": not market["settled_yes"],
                    "fees_enabled": market["fees_enabled"],
                    "fee_rate": market["fee_rate"],
                    "fee_exponent": market["fee_exponent"],
                    "entry_timestamp": timestamp,
                }
            )
    return pd.DataFrame(rows)


def price_cache_path(city: str, token: str, timestamp: int) -> Path:
    return CACHE_ROOT / city.lower() / "no_price_history" / f"{token}_{timestamp}.json"


def fetch_price(city: str, token: str, timestamp: int, refresh: bool) -> dict[str, Any]:
    path = price_cache_path(city, token, timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = request_json(
            CLOB_HISTORY_URL,
            params={
                "market": token,
                "startTs": timestamp - 1800,
                "endTs": timestamp + 60,
                "fidelity": 1,
            },
        )
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    history = [
        (int(item["t"]), float(item["p"]))
        for item in payload.get("history", [])
        if int(item["t"]) <= timestamp
    ]
    if not history:
        return {"reference_price": np.nan, "price_age_seconds": np.nan}
    selected_time, selected_price = max(history)
    age = timestamp - selected_time
    if age > 300 or not 0.0 < selected_price < 1.0:
        return {"reference_price": np.nan, "price_age_seconds": age}
    return {"reference_price": selected_price, "price_age_seconds": age}


def add_prices(city: str, candidates: pd.DataFrame, workers: int, refresh: bool) -> pd.DataFrame:
    requests_to_make = sorted(
        {(str(row.no_token_id), int(row.entry_timestamp)) for row in candidates.itertuples(index=False)}
    )
    prices: dict[tuple[str, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_price, city, token, timestamp, refresh): (token, timestamp)
            for token, timestamp in requests_to_make
        }
        for future in as_completed(futures):
            prices[futures[future]] = future.result()
    frame = candidates.copy()
    selected = [
        prices[(str(token), int(timestamp))]
        for token, timestamp in zip(frame["no_token_id"], frame["entry_timestamp"], strict=True)
    ]
    frame["no_reference_price"] = [value["reference_price"] for value in selected]
    frame["price_age_seconds"] = [value["price_age_seconds"] for value in selected]
    frame["no_fill_price"] = (frame["no_reference_price"] + EXECUTION_PENALTY).clip(upper=0.99)
    fee_base = frame["no_fill_price"] * (1.0 - frame["no_fill_price"])
    frame["fee_per_share"] = np.where(
        frame["fees_enabled"],
        frame["fee_rate"] * np.power(fee_base, frame["fee_exponent"]),
        0.0,
    )
    frame["no_entry_cost"] = frame["no_fill_price"] + frame["fee_per_share"]
    frame["no_edge"] = frame["model_no_probability"] - frame["no_entry_cost"]
    frame["no_available"] = frame["no_entry_cost"].between(0.01, 0.999)
    frame["no_net_return"] = np.where(
        frame["no_available"],
        np.where(frame["no_win"], 1.0 / frame["no_entry_cost"] - 1.0, -1.0),
        np.nan,
    )
    return frame


def scale_trade_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["risk_usdc"] = np.where(
        trades["no_edge"].ge(HIGH_EDGE_THRESHOLD), HIGH_RISK_USDC, BASE_RISK_USDC
    )
    trades["fixed_4_pnl_usdc"] = trades["no_net_return"] * BASE_RISK_USDC
    trades["sized_pnl_usdc"] = trades["no_net_return"] * trades["risk_usdc"]
    return trades


def pnl_drawdown(values: pd.Series) -> float:
    cumulative = values.cumsum()
    return float((cumulative.cummax() - cumulative).max()) if len(values) else float("nan")


def render_report(results: Mapping[str, Mapping[str, Any]]) -> str:
    sections = []
    for city, result in results.items():
        folds = result["folds"]
        frozen_monthly = result["frozen_monthly"]
        clean = result["clean_metrics"]
        frozen = result["frozen_metrics"]
        sensitivity = result["threshold_sensitivity"]
        fold_lines = [
            "| Test month | Selected parameters from prior month | Trades | Win rate | Fixed $4 P&L |",
            "|---|---|---:|---:|---:|",
        ]
        for row in folds.itertuples(index=False):
            fold_lines.append(
                f"| {row.test_month} | `{row.selected_parameters}` | {row.trade_count} "
                f"| {row.win_rate:.1%} | {row.total_pnl_usdc:+.2f} |"
            )
        month_lines = [
            "| Month | Trades | Win rate | Fixed $4 P&L | $4/$6 sized P&L |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in frozen_monthly.itertuples(index=False):
            month_lines.append(
                f"| {row.month} | {row.trade_count} | {row.win_rate:.1%} "
                f"| {row.fixed_pnl_usdc:+.2f} | {row.sized_pnl_usdc:+.2f} |"
            )
        sensitivity_lines = [
            "| Maximum model YES probability | Trades | Win rate | Fixed $4 P&L | 95% LCB |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in sensitivity.itertuples(index=False):
            sensitivity_lines.append(
                f"| {row.max_yes_probability:.0%} | {row.trade_count} | {row.win_rate:.1%} "
                f"| {row.total_pnl_usdc:+.2f} | {row.mean_return_lcb_95:.1%} |"
            )
        sections.append(
            f"""## {city}

### Clean monthly walk-forward

{chr(10).join(fold_lines)}

Combined: {clean['trade_count']} trades, {clean['win_rate']:.1%} win rate,
{clean['total_pnl_usdc']:+.2f} USDC fixed-$4 P&L, {clean['mean_return_lcb_95']:.1%} one-sided
95% mean-return lower bound, and {clean['max_drawdown_usdc']:.2f} USDC maximum drawdown.

### Frozen median rule

`{json.dumps(result['frozen_parameters'], sort_keys=True, separators=(',', ':'))}`

{chr(10).join(month_lines)}

Combined: {frozen['trade_count']} trades, {frozen['win_rate']:.1%} win rate,
{frozen['fixed_pnl_usdc']:+.2f} USDC fixed-$4 P&L and {frozen['sized_pnl_usdc']:+.2f} USDC
under $4/$6 edge sizing. This frozen-rule table is a hindsight diagnostic; the clean
walk-forward table above is the robustness estimate.

Holding the other frozen parameters fixed, the low-probability threshold sensitivity is:

{chr(10).join(sensitivity_lines)}
"""
        )
    return f"""# Dallas and Tokyo 2026 Polymarket Buy-NO Backtest

Status: **historical economic research using public Polymarket NO-token price history; not a live strategy.**

The sweep tests {len(parameter_grid()):,} combinations per station: maximum model YES probability
(5%-40%), NO price caps (55%-95%), minimum model NO edge (0%-20%), provider-spread caps,
confidence-gate on/off, open-tail exclusion on/off, and bucket scope. At most one NO trade is chosen
per event. Entry is 11:15 local using the last public CLOB price at or before entry plus 1¢,
market-specific fees, and settlement hold.

{chr(10).join(sections)}

## Limitations

- Public price history is not historical executable ask depth; the 1¢ penalty is only a proxy.
- January has no earlier 2026 month and therefore appears only in the hindsight frozen-rule table.
- Results cover only dates for which the current research prediction artifacts and resolved markets overlap.
- Thousands of parameter combinations create multiple-testing risk. Do not promote without fresh shadow evidence.
"""


def run_station(city: str, workers: int, refresh_events: bool, refresh_prices: bool) -> dict[str, Any]:
    predictions = load_dallas_predictions() if city == "Dallas" else load_tokyo_predictions()
    start = predictions["contract_date"].min()
    end = predictions["contract_date"].max()
    events = fetch_events(city, start, end, refresh_events)
    candidates = build_candidates(city, predictions, events)
    priced = add_prices(city, candidates, workers, refresh_prices)
    folds, clean_trades = monthly_walk_forward(priced)
    clean_trades = scale_trade_pnl(clean_trades)
    clean_metrics = economic_metrics(clean_trades)
    clean_metrics["total_pnl_usdc"] = float(clean_trades["fixed_4_pnl_usdc"].sum())
    clean_metrics["max_drawdown_usdc"] = pnl_drawdown(clean_trades["fixed_4_pnl_usdc"])
    folds["total_pnl_usdc"] = folds["total_pnl_per_1usd"] * BASE_RISK_USDC
    frozen_parameters = frozen_median_parameters(folds)
    frozen_trades = scale_trade_pnl(filter_candidates(priced, frozen_parameters))
    frozen_metrics_raw = economic_metrics(frozen_trades)
    frozen_metrics = {
        **frozen_metrics_raw,
        "fixed_pnl_usdc": float(frozen_trades["fixed_4_pnl_usdc"].sum()),
        "sized_pnl_usdc": float(frozen_trades["sized_pnl_usdc"].sum()),
        "max_drawdown_usdc": pnl_drawdown(frozen_trades["fixed_4_pnl_usdc"]),
    }
    threshold_rows = []
    for threshold in YES_PROBABILITY_MAXIMUMS:
        parameters = {**frozen_parameters, "max_yes_probability": threshold}
        threshold_trades = scale_trade_pnl(filter_candidates(priced, parameters))
        metrics = economic_metrics(threshold_trades)
        threshold_rows.append(
            {
                "max_yes_probability": threshold,
                **metrics,
                "total_pnl_usdc": float(threshold_trades["fixed_4_pnl_usdc"].sum()),
                "max_drawdown_usdc": pnl_drawdown(threshold_trades["fixed_4_pnl_usdc"]),
            }
        )
    monthly_rows = []
    for month, group in frozen_trades.groupby(frozen_trades["contract_date"].dt.to_period("M")):
        monthly_rows.append(
            {
                "month": str(month),
                "trade_count": len(group),
                "win_rate": float(group["no_win"].mean()),
                "fixed_pnl_usdc": float(group["fixed_4_pnl_usdc"].sum()),
                "sized_pnl_usdc": float(group["sized_pnl_usdc"].sum()),
            }
        )
    return {
        "events": events,
        "priced": priced,
        "folds": folds,
        "clean_trades": clean_trades,
        "clean_metrics": clean_metrics,
        "frozen_parameters": frozen_parameters,
        "frozen_trades": frozen_trades,
        "frozen_metrics": frozen_metrics,
        "frozen_monthly": pd.DataFrame(monthly_rows),
        "threshold_sensitivity": pd.DataFrame(threshold_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--refresh-events", action="store_true")
    parser.add_argument("--refresh-prices", action="store_true")
    args = parser.parse_args()
    results = {
        city: run_station(city, max(1, args.workers), args.refresh_events, args.refresh_prices)
        for city in STATIONS
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for city, result in results.items():
        prefix = STATIONS[city]["station_id"]
        result["priced"].drop(columns=["no_token_id"]).to_csv(
            REPORT_ROOT / f"{prefix}_2026_no_candidate_frame.csv", index=False
        )
        result["folds"].to_csv(REPORT_ROOT / f"{prefix}_2026_no_walkforward_folds.csv", index=False)
        result["clean_trades"].drop(columns=["no_token_id"]).to_csv(
            REPORT_ROOT / f"{prefix}_2026_no_walkforward_trades.csv", index=False
        )
        result["frozen_trades"].drop(columns=["no_token_id"]).to_csv(
            REPORT_ROOT / f"{prefix}_2026_no_frozen_trades.csv", index=False
        )
        result["frozen_monthly"].to_csv(
            REPORT_ROOT / f"{prefix}_2026_no_frozen_monthly.csv", index=False
        )
        result["threshold_sensitivity"].to_csv(
            REPORT_ROOT / f"{prefix}_2026_no_probability_threshold_sensitivity.csv", index=False
        )
    report_path = REPORT_ROOT / "REPORT.md"
    report_path.write_text(render_report(results), encoding="utf-8")
    metadata = {
        city: {
            "matched_events": len(result["events"]),
            "priced_candidates": int(result["priced"]["no_available"].sum()),
            "clean_metrics": result["clean_metrics"],
            "frozen_parameters": result["frozen_parameters"],
            "frozen_metrics": result["frozen_metrics"],
            "probability_threshold_sensitivity": result["threshold_sensitivity"].to_dict("records"),
        }
        for city, result in results.items()
    }
    (REPORT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(report_path)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
