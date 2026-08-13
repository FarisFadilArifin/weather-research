from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, time
import hashlib
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

from src.calibration.seoul_polymarket_walkforward import (
    DEFAULT_FOLDS,
    DEFAULT_FAMILIES,
    PRICE_CAPS,
    economic_metrics,
    expand_families_with_price_caps,
    family_by_name,
    filter_mask,
    select_winner,
    walk_forward_backtest,
)


STATION_ROOT = PROJECT_ROOT / "data" / "calibration" / "station_training_baseline" / "Seoul"
PREDICTION_PATH = STATION_ROOT / "celsius_market_probability" / "RKSI_2026_holdout_predictions.csv"
FEATURE_PATH = STATION_ROOT / "RKSI_features.csv"
CACHE_ROOT = PROJECT_ROOT / "data" / "polymarket" / "seoul_2026_walkforward"
EVENT_CACHE = CACHE_ROOT / "gamma_highest_temperature_events.json"
PRICE_CACHE = CACHE_ROOT / "price_history"
REPORT_ROOT = PROJECT_ROOT / "reports" / "seoul_polymarket_2026_walkforward"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
SEOUL_TAG_ID = 102936
USER_AGENT = "weather-research/0.1 seoul-polymarket-backtest"
LOCAL_TZ = ZoneInfo("Asia/Seoul")
ENTRY_LOCAL_TIME = time(11, 15)
START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-07-25")
FULL_GRID_FAMILIES = expand_families_with_price_caps(DEFAULT_FAMILIES)


def request_json(
    url: str,
    *,
    params: Mapping[str, Any],
    retries: int = 5,
    timeout: int = 45,
) -> Any:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=dict(params),
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - network retry preserves final context.
            error = exc
            if attempt + 1 < retries:
                time_module.sleep(0.75 * (2**attempt))
    raise RuntimeError(f"request failed for {url} params={params}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_events(*, refresh: bool) -> list[dict[str, Any]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if EVENT_CACHE.exists() and not refresh:
        return json.loads(EVENT_CACHE.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = request_json(
            GAMMA_EVENTS_URL,
            params={
                "tag_id": SEOUL_TAG_ID,
                "closed": "true",
                "limit": 100,
                "offset": offset,
                "end_date_min": "2026-01-01T00:00:00Z",
                "end_date_max": "2026-07-26T23:59:59Z",
            },
            timeout=90,
        )
        if not page:
            break
        events.extend(page)
        offset += len(page)
        if len(page) < 100:
            break
    highest = [
        event
        for event in events
        if str(event.get("slug", "")).startswith("highest-temperature-in-seoul-")
    ]
    by_id = {str(event["id"]): event for event in highest}
    highest = sorted(by_id.values(), key=lambda event: str(event.get("endDate")))
    EVENT_CACHE.write_text(json.dumps(highest, indent=2, ensure_ascii=False), encoding="utf-8")
    return highest


def _jsonish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return []


def parse_bucket_label(label: str) -> tuple[int | None, int | None]:
    normalized = label.replace("º", "°").strip()
    match = re.search(r"(-?\d+)\s*°?C", normalized, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported Seoul bucket label: {label}")
    value = int(match.group(1))
    lower = normalized.lower()
    if "or below" in lower:
        return None, value
    if "or higher" in lower:
        return value, None
    return value, value


def market_rows(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in event.get("markets", []):
        label = str(market.get("groupItemTitle") or "").strip()
        lower, upper = parse_bucket_label(label)
        outcomes = [str(value).lower() for value in _jsonish(market.get("outcomes"))]
        prices = [float(value) for value in _jsonish(market.get("outcomePrices"))]
        tokens = [str(value) for value in _jsonish(market.get("clobTokenIds"))]
        yes_index = outcomes.index("yes")
        schedule = market.get("feeSchedule") or {}
        rows.append(
            {
                "market_id": str(market.get("id")),
                "market_slug": str(market.get("slug")),
                "condition_id": str(market.get("conditionId")),
                "bucket_label": label,
                "lower_c": lower,
                "upper_c": upper,
                "yes_token_id": tokens[yes_index],
                "settled_yes": prices[yes_index] >= 0.99,
                "fees_enabled": bool(market.get("feesEnabled")),
                "fee_rate": float(schedule.get("rate", 0.0)),
                "fee_exponent": float(schedule.get("exponent", 1.0)),
            }
        )
    return rows


def market_for_degree(markets: list[dict[str, Any]], degree_c: int) -> dict[str, Any]:
    matches = [
        market
        for market in markets
        if (market["lower_c"] is None or degree_c >= market["lower_c"])
        and (market["upper_c"] is None or degree_c <= market["upper_c"])
    ]
    if len(matches) != 1:
        raise ValueError(f"degree {degree_c} maps to {len(matches)} markets")
    return matches[0]


def entry_timestamp(contract_date: pd.Timestamp) -> int:
    local = datetime.combine(contract_date.date(), ENTRY_LOCAL_TIME, tzinfo=LOCAL_TZ)
    return int(local.astimezone(UTC).timestamp())


def _price_cache_path(token_id: str, timestamp: int) -> Path:
    return PRICE_CACHE / f"{token_id}_{timestamp}.json"


def fetch_price_history(
    token_id: str,
    timestamp: int,
    *,
    refresh: bool,
) -> dict[str, Any]:
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    path = _price_cache_path(token_id, timestamp)
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = request_json(
            CLOB_HISTORY_URL,
            params={
                "market": token_id,
                "startTs": timestamp - 1800,
                "endTs": timestamp + 60,
                "fidelity": 1,
            },
        )
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    history = [
        {"t": int(item["t"]), "p": float(item["p"])}
        for item in payload.get("history", [])
        if int(item["t"]) <= timestamp
    ]
    if not history:
        return {"reference_price": np.nan, "price_timestamp": pd.NaT, "price_age_seconds": np.nan}
    selected = max(history, key=lambda item: item["t"])
    age = timestamp - selected["t"]
    if age > 300 or not 0.0 < selected["p"] < 1.0:
        return {"reference_price": np.nan, "price_timestamp": pd.NaT, "price_age_seconds": age}
    return {
        "reference_price": selected["p"],
        "price_timestamp": pd.to_datetime(selected["t"], unit="s", utc=True),
        "price_age_seconds": age,
    }


def _market_probability(
    exact_probabilities: Mapping[str, float],
    markets: list[dict[str, Any]],
    selected_market_id: str,
) -> float:
    total = 0.0
    for degree, probability in exact_probabilities.items():
        try:
            mapped = market_for_degree(markets, int(degree))
        except ValueError:
            continue
        if mapped["market_id"] == selected_market_id:
            total += float(probability)
    return total


def build_base_rows(
    predictions: pd.DataFrame,
    events: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    events_by_date = {
        pd.to_datetime(event["endDate"], utc=True).tz_convert(None).normalize(): event
        for event in events
    }
    rows: list[dict[str, Any]] = []
    price_requests: set[tuple[str, int]] = set()
    for prediction in predictions.itertuples(index=False):
        contract_date = pd.Timestamp(prediction.contract_date).normalize()
        event = events_by_date.get(contract_date)
        if event is None:
            continue
        markets = market_rows(event)
        winners = [market for market in markets if market["settled_yes"]]
        if len(winners) != 1:
            raise ValueError(f"event {event['slug']} has {len(winners)} winning buckets")
        actual_market = market_for_degree(markets, int(prediction.actual_bucket_c))
        settlement_matches = actual_market["market_id"] == winners[0]["market_id"]
        probabilities = json.loads(prediction.market_bucket_probabilities_c)
        timestamp = entry_timestamp(contract_date)
        row: dict[str, Any] = {
            "contract_date": contract_date,
            "event_slug": event["slug"],
            "event_id": str(event["id"]),
            "actual_bucket_c": int(prediction.actual_bucket_c),
            "settled_bucket_label": winners[0]["bucket_label"],
            "settlement_matches_notebook": settlement_matches,
            "market_top_probability_c": float(prediction.market_top_probability_c),
            "market_top_two_margin_c": float(prediction.market_top_two_margin_c),
            "provider_spread_high_f": float(prediction.provider_spread_high_f),
            "market_probability_decision": str(prediction.market_probability_decision),
            "entry_timestamp": timestamp,
        }
        for action, degree in (
            ("point", int(prediction.point_bucket_c)),
            ("probability", int(prediction.recommended_bucket_c)),
        ):
            try:
                market = market_for_degree(markets, degree)
            except ValueError:
                row.update(
                    {
                        f"{action}_degree_c": degree,
                        f"{action}_market_id": None,
                        f"{action}_market_slug": None,
                        f"{action}_bucket_label": None,
                        f"{action}_token_id": None,
                        f"{action}_win": False,
                        f"{action}_fees_enabled": False,
                        f"{action}_fee_rate": 0.0,
                        f"{action}_fee_exponent": 1.0,
                        f"{action}_model_probability": np.nan,
                    }
                )
                continue
            price_requests.add((market["yes_token_id"], timestamp))
            row.update(
                {
                    f"{action}_degree_c": degree,
                    f"{action}_market_id": market["market_id"],
                    f"{action}_market_slug": market["market_slug"],
                    f"{action}_bucket_label": market["bucket_label"],
                    f"{action}_token_id": market["yes_token_id"],
                    f"{action}_win": bool(market["settled_yes"]),
                    f"{action}_fees_enabled": market["fees_enabled"],
                    f"{action}_fee_rate": market["fee_rate"],
                    f"{action}_fee_exponent": market["fee_exponent"],
                    f"{action}_model_probability": _market_probability(
                        probabilities, markets, market["market_id"]
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), sorted(price_requests)


def download_prices(
    requests_to_make: list[tuple[str, int]],
    *,
    refresh: bool,
    workers: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_price_history, token, timestamp, refresh=refresh): (token, timestamp)
            for token, timestamp in requests_to_make
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
    return results


def apply_execution_assumptions(
    base: pd.DataFrame,
    price_data: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    slippage_cents: int,
) -> pd.DataFrame:
    frame = base.copy()
    slippage = slippage_cents / 100.0
    for action in ("point", "probability"):
        prices = []
        for token, timestamp in zip(
            frame[f"{action}_token_id"], frame["entry_timestamp"], strict=True
        ):
            if token is None or pd.isna(token):
                prices.append(
                    {
                        "reference_price": np.nan,
                        "price_timestamp": pd.NaT,
                        "price_age_seconds": np.nan,
                    }
                )
            else:
                prices.append(price_data[(str(token), int(timestamp))])
        frame[f"{action}_reference_price"] = [item["reference_price"] for item in prices]
        frame[f"{action}_price_timestamp"] = [item["price_timestamp"] for item in prices]
        frame[f"{action}_price_age_seconds"] = [item["price_age_seconds"] for item in prices]
        frame[f"{action}_fill_price"] = (frame[f"{action}_reference_price"] + slippage).clip(upper=0.99)
        fee_base = frame[f"{action}_fill_price"] * (1.0 - frame[f"{action}_fill_price"])
        frame[f"{action}_fee_per_share"] = np.where(
            frame[f"{action}_fees_enabled"],
            frame[f"{action}_fee_rate"] * np.power(fee_base, frame[f"{action}_fee_exponent"]),
            0.0,
        )
        frame[f"{action}_entry_cost"] = frame[f"{action}_fill_price"] + frame[f"{action}_fee_per_share"]
        frame[f"{action}_edge"] = frame[f"{action}_model_probability"] - frame[f"{action}_entry_cost"]
        frame[f"{action}_available"] = (
            frame[f"{action}_entry_cost"].notna()
            & frame[f"{action}_entry_cost"].between(0.01, 0.999)
        )
        frame[f"{action}_net_return"] = np.where(
            frame[f"{action}_available"],
            np.where(frame[f"{action}_win"], 1.0 / frame[f"{action}_entry_cost"] - 1.0, -1.0),
            np.nan,
        )
    return frame


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTION_PATH, parse_dates=["contract_date"])
    features = pd.read_csv(FEATURE_PATH, usecols=["contract_date", "provider_spread_high_f"])
    features["contract_date"] = pd.to_datetime(features["contract_date"], errors="raise")
    frame = predictions.merge(features, on="contract_date", how="left", validate="one_to_one")
    frame = frame.loc[frame["contract_date"].between(START_DATE, END_DATE)].copy()
    if frame["provider_spread_high_f"].isna().any():
        raise ValueError("provider spread is missing from Seoul prediction rows")
    return frame


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


RISK_PER_TRADE_USDC = 4.0
HIGH_CONFIDENCE_RISK_USDC = 6.0
HIGH_CONFIDENCE_EDGE_THRESHOLD = 0.15


def scale_economic_outputs(
    folds: pd.DataFrame,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = folds.copy()
    trades = trades.copy()
    summary = summary.copy()
    for table in (folds, summary):
        table["total_pnl_usdc"] = table["total_pnl_per_1usd"] * RISK_PER_TRADE_USDC
        table["max_drawdown_usdc"] = table["max_drawdown"] * RISK_PER_TRADE_USDC
    trades["pnl_usdc"] = trades["net_return"] * RISK_PER_TRADE_USDC
    return folds, trades, summary


def median_selected_filter_backtest(
    frame: pd.DataFrame,
    fold_results: pd.DataFrame,
    winner: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    family = family_by_name(winner, FULL_GRID_FAMILIES)
    selections = [
        json.loads(value)
        for value in fold_results.loc[
            fold_results["family"].eq(winner), "selected_parameters"
        ]
    ]
    parameter_names = sorted({key for selection in selections for key in selection})
    median_parameters = {
        key: float(np.median([selection[key] for selection in selections]))
        for key in parameter_names
    }
    monthly_rows: list[dict[str, Any]] = []
    trade_rows: list[pd.DataFrame] = []
    evaluation_windows = [("january", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31"))]
    evaluation_windows.extend(
        (fold.name, pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end))
        for fold in DEFAULT_FOLDS
    )
    for month, validation_start, validation_end in evaluation_windows:
        validation = frame[
            frame["contract_date"].between(validation_start, validation_end)
        ]
        mask = filter_mask(validation, family, median_parameters)
        selected = validation.loc[mask]
        metrics = economic_metrics(
            selected[f"{family.action}_net_return"],
            selected[f"{family.action}_win"],
        )
        monthly_rows.append(
            {
                "month": month,
                "validation_start": validation["contract_date"].min().date().isoformat(),
                "validation_end": validation["contract_date"].max().date().isoformat(),
                "median_parameters": json.dumps(
                    median_parameters, sort_keys=True, separators=(",", ":")
                ),
                **metrics,
                "total_pnl_usdc": metrics["total_pnl_per_1usd"] * RISK_PER_TRADE_USDC,
                "max_drawdown_usdc": metrics["max_drawdown"] * RISK_PER_TRADE_USDC,
            }
        )
        trades = selected[
            [
                "contract_date",
                f"{family.action}_market_slug",
                f"{family.action}_bucket_label",
                f"{family.action}_entry_cost",
                f"{family.action}_model_probability",
                f"{family.action}_edge",
                f"{family.action}_win",
                f"{family.action}_net_return",
            ]
        ].copy()
        trades.columns = [
            "contract_date", "market_slug", "bucket_label", "entry_cost",
            "model_probability", "edge", "win", "net_return",
        ]
        trades["pnl_usdc"] = trades["net_return"] * RISK_PER_TRADE_USDC
        trades["risk_usdc"] = np.where(
            trades["edge"] >= HIGH_CONFIDENCE_EDGE_THRESHOLD,
            HIGH_CONFIDENCE_RISK_USDC,
            RISK_PER_TRADE_USDC,
        )
        trades["sized_pnl_usdc"] = trades["net_return"] * trades["risk_usdc"]
        trades["month"] = month
        month_trades = trades
        cumulative = month_trades["sized_pnl_usdc"].cumsum()
        drawdown = (cumulative.cummax() - cumulative).max()
        monthly_rows[-1].update(
            {
                "sizing_pnl_usdc": float(month_trades["sized_pnl_usdc"].sum()),
                "sizing_max_drawdown_usdc": float(drawdown),
                "high_confidence_trades": int(
                    month_trades["risk_usdc"].eq(HIGH_CONFIDENCE_RISK_USDC).sum()
                ),
            }
        )
        trade_rows.append(trades)
    median_trades = pd.concat(trade_rows, ignore_index=True)
    overall = economic_metrics(median_trades["net_return"], median_trades["win"])
    overall["total_pnl_usdc"] = overall["total_pnl_per_1usd"] * RISK_PER_TRADE_USDC
    overall["max_drawdown_usdc"] = overall["max_drawdown"] * RISK_PER_TRADE_USDC
    cumulative = median_trades["sized_pnl_usdc"].cumsum()
    overall["sizing_pnl_usdc"] = float(median_trades["sized_pnl_usdc"].sum())
    overall["sizing_max_drawdown_usdc"] = float((cumulative.cummax() - cumulative).max())
    overall["high_confidence_trades"] = int(
        median_trades["risk_usdc"].eq(HIGH_CONFIDENCE_RISK_USDC).sum()
    )
    return median_parameters, pd.DataFrame(monthly_rows), median_trades, overall


def median_price_cap_sensitivity(
    frame: pd.DataFrame,
    median_parameters: Mapping[str, float],
    winner: str,
) -> pd.DataFrame:
    family = family_by_name(winner, FULL_GRID_FAMILIES)
    windows = [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31"))]
    windows.extend(
        (pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end))
        for fold in DEFAULT_FOLDS
    )
    rows: list[dict[str, Any]] = []
    for cap in PRICE_CAPS:
        selected_parts = []
        parameters = {**median_parameters, "max_price": float(cap)}
        for start, end in windows:
            validation = frame[frame["contract_date"].between(start, end)]
            selected_parts.append(validation.loc[filter_mask(validation, family, parameters)])
        selected = pd.concat(selected_parts, ignore_index=True)
        metrics = economic_metrics(
            selected[f"{family.action}_net_return"],
            selected[f"{family.action}_win"],
        )
        rows.append(
            {
                "max_price": cap,
                "trade_count": metrics["trade_count"],
                "coverage": metrics["trade_count"] / int(frame[f"{family.action}_available"].sum()),
                "win_rate": metrics["win_rate"],
                "total_pnl_usdc": metrics["total_pnl_per_1usd"] * RISK_PER_TRADE_USDC,
                "mean_return": metrics["mean_return"],
                "mean_return_lcb_95": metrics["mean_return_lcb_95"],
                "max_drawdown_usdc": metrics["max_drawdown"] * RISK_PER_TRADE_USDC,
            }
        )
    return pd.DataFrame(rows).sort_values("max_price", ignore_index=True)


def render_report(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    winner: str,
    median_parameters: Mapping[str, float],
    median_monthly: pd.DataFrame,
    median_overall: Mapping[str, Any],
    price_cap_sensitivity: pd.DataFrame,
) -> str:
    winner_row = summary.loc[summary["family"].eq(winner)].iloc[0]
    winner_folds = folds.loc[folds["family"].eq(winner)]
    point = summary.loc[summary["family"].eq("point_all")].iloc[0]
    probability = summary.loc[summary["family"].eq("probability_all")].iloc[0]
    notebook = summary.loc[summary["family"].eq("notebook_policy_reference")].iloc[0]
    july = winner_folds.loc[winner_folds["fold"].eq("july")].iloc[0]
    winner_trades = trades.loc[trades["family"].eq(winner)]
    largest_winners = winner_trades.nlargest(2, "pnl_usdc")
    largest_winner_pnl = float(largest_winners["pnl_usdc"].sum())
    largest_winner_share = largest_winner_pnl / float(winner_row["total_pnl_usdc"])
    largest_winner_labels = ", ".join(
        f"{pd.Timestamp(row.contract_date).date()} ({row.pnl_usdc:+.2f} USDC)"
        for row in largest_winners.itertuples(index=False)
    )
    realized_best = summary.sort_values(
        ["total_pnl_usdc", "trade_count"], ascending=[False, False]
    ).iloc[0]
    fold_lines = [
        "| Test month | Training cutoff | Frozen parameters | Trades | Win rate | Net P&L | Mean return |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in winner_folds.itertuples(index=False):
        fold_lines.append(
            f"| {row.fold} | {row.train_end} | `{row.selected_parameters}` | {row.trade_count} "
            f"| {_format_pct(row.win_rate)} | {row.total_pnl_usdc:+.2f} | {_format_pct(row.mean_return)} |"
        )
    median_lines = [
        "| Month | Trades | Win rate | Fixed $4 P&L | KDAL-style sized P&L | High-risk trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in median_monthly.itertuples(index=False):
        median_lines.append(
            f"| {row.month} | {row.trade_count} | {_format_pct(row.win_rate)} "
            f"| {row.total_pnl_usdc:+.2f} | {row.sizing_pnl_usdc:+.2f} "
            f"| {row.high_confidence_trades} |"
        )
    price_cap_lines = [
        "| Max entry price | Trades | Win rate | Net P&L | 95% LCB | Max DD |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in price_cap_sensitivity.itertuples(index=False):
        price_cap_lines.append(
            f"| {row.max_price:.0%} | {row.trade_count} | {_format_pct(row.win_rate)} "
            f"| {row.total_pnl_usdc:+.2f} | {_format_pct(row.mean_return_lcb_95)} "
            f"| {row.max_drawdown_usdc:.2f} |"
        )
    ranking_lines = [
        "| Rank | Family | Trades | Coverage | Win rate | Net P&L | Mean return | 95% LCB | Max DD |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary.head(10).itertuples(index=False), start=1):
        ranking_lines.append(
            f"| {rank} | `{row.family}` | {row.trade_count} | {_format_pct(row.coverage)} "
            f"| {_format_pct(row.win_rate)} | {row.total_pnl_usdc:+.2f} "
            f"| {_format_pct(row.mean_return)} | {_format_pct(row.mean_return_lcb_95)} "
            f"| {row.max_drawdown_usdc:.2f} |"
        )
    scenario_lines = [
        "| Execution penalty | Winner family | Trades | Net P&L | Mean return | 95% LCB |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in scenario_summary.itertuples(index=False):
        scenario_lines.append(
            f"| {row.slippage_cents}¢ | `{row.winner}` | {row.trade_count} "
            f"| {row.total_pnl_usdc:+.2f} | {_format_pct(row.mean_return)} "
            f"| {_format_pct(row.mean_return_lcb_95)} |"
        )
    median_age = pd.concat(
        [frame["point_price_age_seconds"], frame["probability_price_age_seconds"]]
    ).dropna().median()
    mismatch_count = int((~frame["settlement_matches_notebook"]).sum())
    fee_rows = int(frame["point_fees_enabled"].sum())
    return f"""# Seoul 2026 Polymarket Walk-Forward Backtest

Status: **historical economic backtest using public Polymarket data; research only.**

The joint grid contains {len(FULL_GRID_FAMILIES)} filter families and
{sum(len(family.parameter_grid) for family in FULL_GRID_FAMILIES):,} parameter combinations. Every
eligible non-price filter is crossed with entry-price caps from 20¢ through 80¢.

## Best filter

The least-bad eligible walk-forward family at the base 1¢ execution penalty was `{winner}`.
Across the six out-of-sample months it made {int(winner_row['trade_count'])} trades at
{_format_pct(winner_row['coverage'])} coverage, won {_format_pct(winner_row['win_rate'])}, and
earned {winner_row['total_pnl_usdc']:+.2f} USDC using a fixed {RISK_PER_TRADE_USDC:.0f} USDC risk per trade. Mean
net return was {_format_pct(winner_row['mean_return'])}; its one-sided 95% lower confidence bound
was {_format_pct(winner_row['mean_return_lcb_95'])}, profit factor was
{winner_row['profit_factor']:.2f}, and maximum drawdown was {winner_row['max_drawdown_usdc']:.2f} USDC.

**No candidate had a non-negative 95% lower confidence bound.** Therefore this 2026 sample does
not establish a robustly profitable filter. The highest realized P&L belonged to
`{realized_best['family']}` at {realized_best['total_pnl_usdc']:+.2f} USDC, but its lower bound was
{_format_pct(realized_best['mean_return_lcb_95'])}; it is not a validated promotion candidate.

The result is also concentrated: the two largest winners, {largest_winner_labels}, contributed
{largest_winner_pnl:+.2f} USDC, or {_format_pct(largest_winner_share)} of the family's total P&L.
This tail dependence is another reason not to treat the realized headline return as robust.

The final July rule was selected using June 1-30 only:
`{july['selected_parameters']}`. In July 1-25 it made {int(july['trade_count'])} trades,
earned {july['total_pnl_usdc']:+.2f} USDC, and returned {_format_pct(july['mean_return'])} per
trade.

## Walk-forward results

{chr(10).join(fold_lines)}

Each filter's thresholds were selected on the immediately preceding calendar month only. January
selected February, February selected March, and so on through June selecting July. Candidate
selection maximized the one-sided 95% lower confidence bound of net return with minimum trade-count
and coverage constraints. February through July were disjoint forward test folds.

## Median frozen filter

Taking the median of the six monthly parameter selections gives
`{json.dumps(dict(median_parameters), sort_keys=True, separators=(',', ':'))}`. Freezing that one
filter and applying it uniformly to every January-July market produced
{int(median_overall['trade_count'])} trades, {_format_pct(median_overall['win_rate'])} wins,
{median_overall['total_pnl_usdc']:+.2f} USDC P&L, {_format_pct(median_overall['mean_return'])}
mean return, and {median_overall['max_drawdown_usdc']:.2f} USDC maximum drawdown.

{chr(10).join(median_lines)}

Only January 1 through July 25 can be reported. August-December 2026 markets were not available as
resolved historical outcomes at the backtest cutoff and are intentionally not fabricated.

This is a hindsight diagnostic, not a clean walk-forward estimate: later monthly selections help
define the median filter that is then applied to earlier months.

### Price-cap sensitivity

Holding the median provider-spread and edge thresholds fixed, the following table varies the
entry-price cap from 20¢ through 80¢:

{chr(10).join(price_cap_lines)}

The same trades were also evaluated with the existing KDAL-style probability sizing rule: $4 base
risk, increasing to $6 when model edge is at least 0.15. This produced
{median_overall['sizing_pnl_usdc']:+.2f} USDC with {median_overall['sizing_max_drawdown_usdc']:.2f} USDC
maximum drawdown across {median_overall['high_confidence_trades']} high-risk trades.

## Baselines

- Buy every point-model bucket: {int(point['trade_count'])} trades,
  {point['total_pnl_usdc']:+.2f} USDC P&L, {_format_pct(point['mean_return'])} mean return.
- Buy every ordinal recommended bucket: {int(probability['trade_count'])} trades,
  {probability['total_pnl_usdc']:+.2f} USDC P&L, {_format_pct(probability['mean_return'])} mean return.
- Existing notebook confidence policy: {int(notebook['trade_count'])} trades,
  {notebook['total_pnl_usdc']:+.2f} USDC P&L, {_format_pct(notebook['mean_return'])} mean return.

{chr(10).join(ranking_lines)}

## Polymarket pricing and settlement contract

- Markets: {len(frame)} Seoul daily-high events matched to RKSI model predictions from
  {frame['contract_date'].min().date()} through {frame['contract_date'].max().date()}.
- Entry time: 11:15 Asia/Seoul. The reference is the last public CLOB price-history point at or
  before entry, with median age {median_age:.0f} seconds.
- Execution: reference price plus 1¢, held to binary settlement, fixed {RISK_PER_TRADE_USDC:.0f} USDC risk per trade.
- Fees: each Gamma market's own `feesEnabled` and `feeSchedule` fields; {fee_rows} matched dates
  had fees enabled for the point action. Fee per share is
  `rate * (p * (1-p)) ** exponent`.
- Settlement: Gamma's resolved YES outcome. Notebook Wunderground settlement mapping mismatched
  {mismatch_count} events; those rows remain in the economic backtest because Gamma is the
  authoritative Polymarket settlement source.
- Historical CLOB price history is not a historical ask book. The added execution penalty is a
  conservative proxy, and the sensitivity table shows how the result moves at wider penalties.

## Execution sensitivity

{chr(10).join(scenario_lines)}

## Limitations

- Public price history does not reconstruct historical depth, executable ask size, queue position,
  partial fills, or rejected FOK orders.
- Results recycle a fixed {RISK_PER_TRADE_USDC:.0f} USDC risk and do not model overlapping capital or wallet limits.
- Multiple filter families were compared. Walk-forward testing reduces leakage but does not remove
  multiple-testing risk from a single January-July market regime.
- Treat the July fold as the cleanest evidence and keep any rule shadow-only until it survives new
  Seoul markets with captured bid/ask depth.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Seoul 2026 model trades on Polymarket")
    parser.add_argument("--refresh-events", action="store_true")
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    predictions = load_predictions()
    events = fetch_events(refresh=args.refresh_events)
    base, price_requests = build_base_rows(predictions, events)
    if base.empty:
        raise ValueError("no Seoul model dates matched Polymarket events")
    price_data = download_prices(
        price_requests,
        refresh=args.refresh_prices,
        workers=max(1, args.workers),
    )

    scenario_rows: list[dict[str, Any]] = []
    scenario_results: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for slippage_cents in (0, 1, 2, 3):
        frame = apply_execution_assumptions(
            base,
            price_data,
            slippage_cents=slippage_cents,
        )
        folds, trades, summary = walk_forward_backtest(frame, families=FULL_GRID_FAMILIES)
        folds, trades, summary = scale_economic_outputs(folds, trades, summary)
        winner = select_winner(summary)
        winner_row = summary.loc[summary["family"].eq(winner)].iloc[0]
        scenario_rows.append(
            {
                "slippage_cents": slippage_cents,
                "winner": winner,
                "trade_count": int(winner_row["trade_count"]),
                "coverage": float(winner_row["coverage"]),
                "win_rate": float(winner_row["win_rate"]),
                "total_pnl_per_1usd": float(winner_row["total_pnl_per_1usd"]),
                "total_pnl_usdc": float(winner_row["total_pnl_usdc"]),
                "mean_return": float(winner_row["mean_return"]),
                "mean_return_lcb_95": float(winner_row["mean_return_lcb_95"]),
                "max_drawdown": float(winner_row["max_drawdown"]),
                "max_drawdown_usdc": float(winner_row["max_drawdown_usdc"]),
            }
        )
        scenario_results[slippage_cents] = (frame, folds, trades, summary)

    frame, folds, trades, summary = scenario_results[1]
    winner = select_winner(summary)
    median_parameters, median_monthly, median_trades, median_overall = (
        median_selected_filter_backtest(frame, folds, winner)
    )
    price_cap_sensitivity = median_price_cap_sensitivity(frame, median_parameters, winner)
    scenario_summary = pd.DataFrame(scenario_rows)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    frame_path = REPORT_ROOT / "RKSI_2026_polymarket_daily_frame.csv"
    fold_path = REPORT_ROOT / "RKSI_2026_walkforward_fold_results.csv"
    trade_path = REPORT_ROOT / "RKSI_2026_walkforward_trades.csv"
    summary_path = REPORT_ROOT / "RKSI_2026_walkforward_family_summary.csv"
    scenario_path = REPORT_ROOT / "RKSI_2026_slippage_sensitivity.csv"
    median_monthly_path = REPORT_ROOT / "RKSI_2026_median_filter_monthly_results.csv"
    median_trade_path = REPORT_ROOT / "RKSI_2026_median_filter_trades.csv"
    price_cap_path = REPORT_ROOT / "RKSI_2026_median_filter_price_cap_sensitivity.csv"
    report_path = REPORT_ROOT / "REPORT.md"
    metadata_path = REPORT_ROOT / "run_metadata.json"
    public_frame = frame.drop(columns=["point_token_id", "probability_token_id"])
    public_frame.to_csv(frame_path, index=False)
    folds.to_csv(fold_path, index=False)
    trades.to_csv(trade_path, index=False)
    summary.to_csv(summary_path, index=False)
    scenario_summary.to_csv(scenario_path, index=False)
    median_monthly.to_csv(median_monthly_path, index=False)
    median_trades.to_csv(median_trade_path, index=False)
    price_cap_sensitivity.to_csv(price_cap_path, index=False)
    report_path.write_text(
        render_report(
            frame,
            folds,
            summary,
            trades,
            scenario_summary,
            winner,
            median_parameters,
            median_monthly,
            median_overall,
            price_cap_sensitivity,
        ),
        encoding="utf-8",
    )
    metadata = {
        "station_id": "RKSI",
        "source_notebook": "notebooks/station_training_baseline/stations/Seoul/train_Seoul.ipynb",
        "decision_time_local": "11:15 Asia/Seoul",
        "base_execution_penalty_cents": 1,
        "walk_forward_contract": "previous_calendar_month_selects_next_calendar_month",
        "risk_per_trade_usdc": RISK_PER_TRADE_USDC,
        "median_filter_parameters": median_parameters,
        "median_filter_is_hindsight_diagnostic": True,
        "sizing_policy": {
            "base_loss_usdc": RISK_PER_TRADE_USDC,
            "high_loss_usdc": HIGH_CONFIDENCE_RISK_USDC,
            "high_edge_threshold": HIGH_CONFIDENCE_EDGE_THRESHOLD,
        },
        "gamma_api": GAMMA_EVENTS_URL,
        "clob_price_history_api": CLOB_HISTORY_URL,
        "event_cache_sha256": sha256_file(EVENT_CACHE),
        "prediction_sha256": sha256_file(PREDICTION_PATH),
        "feature_sha256": sha256_file(FEATURE_PATH),
        "winner": winner,
        "robust_positive_filter_exists": bool(summary["robust_positive"].any()),
        "selection_excludes_current_fold": True,
        "outputs_sha256": {
            path.name: sha256_file(path)
            for path in (
                frame_path,
                fold_path,
                trade_path,
                summary_path,
                scenario_path,
                median_monthly_path,
                median_trade_path,
                price_cap_path,
                report_path,
            )
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    winner_row = summary.loc[summary["family"].eq(winner)].iloc[0]
    print(report_path)
    print(
        json.dumps(
            {
                "matched_dates": len(frame),
                "point_price_available": int(frame["point_available"].sum()),
                "probability_price_available": int(frame["probability_available"].sum()),
                "settlement_mismatches": int((~frame["settlement_matches_notebook"]).sum()),
                "winner": winner,
                "winner_metrics": winner_row.to_dict(),
                "median_filter_parameters": median_parameters,
                "median_filter_metrics": median_overall,
                "price_cap_sensitivity": price_cap_sensitivity.to_dict("records"),
                "slippage_sensitivity": scenario_rows,
            },
            default=str,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
