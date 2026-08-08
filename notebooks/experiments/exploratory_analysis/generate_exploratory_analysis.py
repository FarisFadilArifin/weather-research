"""Generate the repository-wide exploratory analysis notebook.

The generated notebook is intentionally read-only: it inventories every local
artifact under data/ and samples large shard collections instead of loading all
of them at once.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "exploratory_analysis.ipynb"


def markdown(source: str) -> dict:
    normalized = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(f"markdown:{normalized}".encode()).hexdigest()[:8],
        "metadata": {},
        "source": normalized.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    normalized = dedent(source).strip()
    return {
        "cell_type": "code",
        "id": hashlib.sha1(f"code:{normalized}".encode()).hexdigest()[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # Weather research: all-data exploratory analysis

        **Status:** current, general-purpose data audit notebook.

        This notebook explores everything currently pulled into `data/`: raw API
        responses and shards, processed research tables, calibration datasets,
        exports, model artifacts, and temporary pull outputs. It is read-only by
        default and is designed to remain usable as the local data lake grows.

        The workflow has two levels:

        1. inventory **every file** and summarize coverage, size, freshness, and
           possible duplicates;
        2. load the important consolidated tables and sample one representative
           file from each raw/calibration family for schema inspection.

        Run from top to bottom. Change the configuration in the next cell when a
        deeper or faster pass is needed.
        """
    ),
    code(
        """
        from __future__ import annotations

        import hashlib
        import json
        import os
        import re
        import shutil
        import sqlite3
        import subprocess
        import warnings
        from collections import Counter
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display

        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_colwidth", 120)
        plt.style.use("seaborn-v0_8-whitegrid")

        def find_repo_root(start: Path | None = None) -> Path:
            current = (start or Path.cwd()).resolve()
            for candidate in (current, *current.parents):
                if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
                    return candidate
            raise FileNotFoundError("Could not locate repo root containing pyproject.toml and data/")

        ROOT = find_repo_root()
        DATA = ROOT / "data"

        # Safe defaults for interactive use. Inventory still covers every file.
        MAX_ROWS_PER_TABLE = 250_000
        SCHEMA_SAMPLE_ROWS = 500
        MAX_SCHEMA_FAMILIES = None  # set an integer for an even faster schema pass
        DEEP_FILE_METADATA = False  # True adds exact size/freshness but is slow during active backfills
        HASH_DUPLICATE_CANDIDATES = False  # full hashes can be expensive on very large files
        ACTIVE_TABLE = "processed/model_errors.csv"
        RANDOM_SEED = 42

        print(f"Repository: {ROOT}")
        print(f"Data root:  {DATA}")
        """
    ),
    markdown(
        """
        ## 1. Complete artifact inventory

        This scan reads filesystem metadata only, so all files are represented
        without loading their contents. `logical_family` collapses daily shards
        into source-level groups while retaining separate calibration experiments.
        """
    ),
    code(
        """
        def logical_family(relative_path: str) -> str:
            parts = Path(relative_path).parts
            if not parts:
                return "unknown"
            if parts[0] in {"raw", "calibration"} and len(parts) >= 2:
                return "/".join(parts[:2])
            if parts[0] == "tmp" and len(parts) >= 2:
                return "/".join(parts[:2])
            return parts[0]

        def scan_paths(folder: Path) -> list[Path]:
            # ripgrep's directory walker is much faster on this shard-heavy tree.
            if shutil.which("rg"):
                result = subprocess.run(["rg", "--files", str(folder)], check=True,
                                        capture_output=True, text=True)
                return [Path(line) for line in result.stdout.splitlines() if line]
            return [Path(dp) / name for dp, _, names in os.walk(folder) for name in names]

        records = []
        for path in scan_paths(DATA):
            stat = path.stat() if DEEP_FILE_METADATA else None
            rel = path.relative_to(DATA).as_posix()
            records.append({
                "relative_path": rel,
                "logical_family": logical_family(rel),
                "top_level": Path(rel).parts[0],
                "extension": path.suffix.lower() or "[none]",
                "size_bytes": stat.st_size if stat else np.nan,
                "size_mb": stat.st_size / 2**20 if stat else np.nan,
                "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC") if stat else pd.NaT,
                "name": path.name,
                "path": path,
            })

        files = pd.DataFrame(records).sort_values("relative_path").reset_index(drop=True)
        assert not files.empty, f"No files found under {DATA}"

        overview = pd.DataFrame({
            "metric": ["files", "directories represented", "total size (GB)", "newest modification (UTC)"],
            "value": [len(files), files["path"].map(lambda p: p.parent).nunique(),
                      round(files["size_bytes"].sum(min_count=1) / 2**30, 3), files["modified_utc"].max()],
        })
        display(overview)
        display(files.drop(columns="path").sort_values("size_bytes", ascending=False).head(25))
        """
    ),
    code(
        """
        by_area = (files.groupby("top_level", as_index=False)
                   .agg(files=("relative_path", "size"), size_gb=("size_bytes", lambda s: s.sum(min_count=1) / 2**30),
                        newest_utc=("modified_utc", "max"))
                   .sort_values("size_gb", ascending=False))
        by_format = (files.groupby("extension", as_index=False)
                     .agg(files=("relative_path", "size"), size_gb=("size_bytes", lambda s: s.sum(min_count=1) / 2**30))
                     .sort_values("files", ascending=False))
        display(by_area)
        display(by_format)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        by_area.set_index("top_level")["files"].sort_values().plot.barh(ax=axes[0], title="File count by data area")
        if by_area["size_gb"].notna().any():
            by_area.set_index("top_level")["size_gb"].sort_values().plot.barh(ax=axes[1], title="Disk usage by data area (GB)")
        else:
            axes[1].text(.5, .5, "Set DEEP_FILE_METADATA=True for disk usage", ha="center", va="center")
            axes[1].set_title("Disk usage by data area (GB)")
        axes[0].set_xlabel("files")
        axes[1].set_xlabel("GB")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 2. Coverage, freshness, and duplicate candidates

        The filename parser surfaces station codes and ISO dates wherever they
        occur, which makes shard coverage visible even before opening the files.
        Duplicate candidates are conservative: same basename and size. Optional
        full SHA-256 verification is available for candidate groups.
        """
    ),
    code(
        """
        station_pattern = re.compile(r"(?<![A-Z0-9])([KC][A-Z]{3})(?![A-Z0-9])", re.I)
        date_pattern = re.compile(r"(20\\d{2}-\\d{2}-\\d{2})")

        files["station_hint"] = files["relative_path"].str.extract(station_pattern, expand=False).str.upper()
        files["date_hint"] = pd.to_datetime(files["relative_path"].str.extract(date_pattern, expand=False), errors="coerce")
        pull_coverage = (files.dropna(subset=["date_hint"])
                         .groupby(["logical_family", "station_hint"], dropna=False)
                         .agg(files=("relative_path", "size"), first_date=("date_hint", "min"),
                              last_date=("date_hint", "max"), size_mb=("size_mb", "sum"))
                         .reset_index().sort_values(["logical_family", "station_hint"], na_position="last"))
        display(pull_coverage.head(100))

        monthly_pulls = (files.dropna(subset=["date_hint"])
                         .assign(month=lambda x: x["date_hint"].dt.to_period("M").dt.to_timestamp())
                         .groupby(["month", "logical_family"]).size().unstack(fill_value=0))
        if not monthly_pulls.empty:
            monthly_pulls.plot(figsize=(14, 5), title="Date-stamped artifact coverage by logical family")
            plt.ylabel("files")
            plt.tight_layout()
            plt.show()
        """
    ),
    code(
        """
        backup_mask = files["relative_path"].str.contains(
            r"before_|backup|copy|smoke|probe|tmp|\\.bak", case=False, regex=True
        )
        backup_summary = (files.loc[backup_mask]
                          .groupby("logical_family", as_index=False)
                          .agg(candidate_files=("relative_path", "size"), size_mb=("size_mb", "sum"))
                          .sort_values("size_mb", ascending=False))
        duplicate_candidates = (files.dropna(subset=["size_bytes"]).groupby(["name", "size_bytes"], as_index=False)
                                .agg(copies=("relative_path", "size"), paths=("relative_path", list))
                                .query("copies > 1").sort_values(["copies", "size_bytes"], ascending=False))
        display(backup_summary.head(30))
        display(duplicate_candidates.head(30))

        if HASH_DUPLICATE_CANDIDATES and not duplicate_candidates.empty:
            candidate_paths = {p for paths in duplicate_candidates["paths"] for p in paths}
            def sha256(rel: str) -> str:
                digest = hashlib.sha256()
                with (DATA / rel).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(2**20), b""):
                        digest.update(chunk)
                return digest.hexdigest()
            verified_hashes = pd.DataFrame({"relative_path": sorted(candidate_paths)})
            verified_hashes["sha256"] = verified_hashes["relative_path"].map(sha256)
            display(verified_hashes.groupby("sha256").filter(lambda g: len(g) > 1).sort_values("sha256"))
        """
    ),
    markdown(
        """
        ## 3. Representative schema pass across all data families

        One representative CSV/JSON/JSONL/SQLite file is inspected from every
        logical family. This finds broken or surprising schemas without opening
        thousands of daily shards. The complete file inventory above remains the
        source of truth for file-level coverage.
        """
    ),
    code(
        """
        def read_sample(path: Path, nrows: int = SCHEMA_SAMPLE_ROWS) -> pd.DataFrame:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path, nrows=nrows, low_memory=False)
            if suffix == ".jsonl":
                return pd.read_json(path, lines=True, nrows=nrows)
            if suffix == ".json":
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    return pd.json_normalize(payload[:nrows])
                if isinstance(payload, dict):
                    for value in payload.values():
                        if isinstance(value, list):
                            return pd.json_normalize(value[:nrows])
                    return pd.json_normalize(payload)
            if suffix == ".sqlite3":
                with sqlite3.connect(path) as conn:
                    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
                    if tables:
                        safe_name = tables[0].replace('"', '""')
                        return pd.read_sql(f'SELECT * FROM "{safe_name}" LIMIT {int(nrows)}', conn)
            raise ValueError(f"Unsupported tabular format: {path.suffix}")

        schema_files = files[files["extension"].isin([".csv", ".json", ".jsonl", ".sqlite3"])].copy()
        # Prefer the smallest nonempty representative. In particular, JSON row
        # limits cannot be applied until after decoding the document.
        representatives = (schema_files
                           .sort_values("size_bytes", ascending=True, na_position="last")
                           .drop_duplicates("logical_family"))
        if MAX_SCHEMA_FAMILIES is not None:
            representatives = representatives.head(MAX_SCHEMA_FAMILIES)

        schema_records = []
        schema_samples = {}
        for row in representatives.itertuples(index=False):
            try:
                sample = read_sample(row.path)
                schema_samples[row.logical_family] = sample
                schema_records.append({
                    "logical_family": row.logical_family, "representative": row.relative_path,
                    "sample_rows": len(sample), "columns": sample.shape[1],
                    "numeric_columns": sample.select_dtypes(include=np.number).shape[1],
                    "sample_missing_pct": round(sample.isna().mean().mean() * 100, 2), "status": "ok",
                })
            except Exception as exc:
                schema_records.append({
                    "logical_family": row.logical_family, "representative": row.relative_path,
                    "sample_rows": np.nan, "columns": np.nan, "numeric_columns": np.nan,
                    "sample_missing_pct": np.nan, "status": f"{type(exc).__name__}: {exc}",
                })
        schema_catalog = pd.DataFrame(schema_records).sort_values("logical_family")
        display(schema_catalog)
        """
    ),
    markdown(
        """
        ## 4. Consolidated analysis-table catalog

        These are the compact, analysis-ready tables: all top-level CSVs from
        `processed`, `outputs`, `exports`, `calibration_strict`, plus the root of
        `calibration`. Large tables are capped by `MAX_ROWS_PER_TABLE`; the
        catalog clearly marks capped loads.
        """
    ),
    code(
        """
        def discover_analysis_tables() -> list[Path]:
            paths = []
            for area in ["processed", "outputs", "exports", "calibration_strict"]:
                folder = DATA / area
                if folder.exists():
                    paths.extend(folder.glob("*.csv"))
            calibration = DATA / "calibration"
            if calibration.exists():
                paths.extend(calibration.glob("*.csv"))
            return sorted(set(paths))

        def count_csv_rows(path: Path) -> int:
            with path.open("rb") as handle:
                return max(sum(chunk.count(b"\\n") for chunk in iter(lambda: handle.read(2**20), b"")) - 1, 0)

        tables = {}
        table_records = []
        for path in discover_analysis_tables():
            key = path.relative_to(DATA).as_posix()
            try:
                total_rows = count_csv_rows(path)
                frame = pd.read_csv(path, nrows=MAX_ROWS_PER_TABLE, low_memory=False)
                tables[key] = frame
                table_records.append({
                    "table": key, "rows_on_disk": total_rows, "rows_loaded": len(frame),
                    "load_capped": total_rows > len(frame), "columns": frame.shape[1],
                    "memory_mb": frame.memory_usage(deep=True).sum() / 2**20,
                    "missing_pct": frame.isna().mean().mean() * 100 if frame.shape[1] else 0,
                    "duplicate_rows_loaded": int(frame.duplicated().sum()), "status": "ok",
                })
            except Exception as exc:
                table_records.append({"table": key, "status": f"{type(exc).__name__}: {exc}"})

        table_catalog = pd.DataFrame(table_records).sort_values("table")
        display(table_catalog)
        print(f"Loaded {len(tables):,} consolidated tables ({sum(len(x) for x in tables.values()):,} rows in memory).")
        """
    ),
    markdown(
        """
        ## 5. Generic deep dive

        Set `ACTIVE_TABLE` in the configuration cell to any key shown in the
        catalog. The profile adapts to numeric, categorical, date, station, model,
        and provider columns.
        """
    ),
    code(
        """
        if ACTIVE_TABLE not in tables:
            ACTIVE_TABLE = next(iter(tables))
            print(f"Configured table was unavailable; using {ACTIVE_TABLE}")
        active = tables[ACTIVE_TABLE].copy()
        print(f"{ACTIVE_TABLE}: {active.shape[0]:,} loaded rows × {active.shape[1]:,} columns")
        display(active.head())

        column_profile = pd.DataFrame({
            "dtype": active.dtypes.astype(str),
            "non_null": active.notna().sum(),
            "missing_pct": active.isna().mean().mul(100).round(2),
            "unique": active.nunique(dropna=True),
        }).sort_values(["missing_pct", "unique"], ascending=[False, False])
        display(column_profile)
        """
    ),
    code(
        """
        missing = active.isna().mean().mul(100).sort_values(ascending=False)
        missing = missing[missing > 0].head(40)
        if not missing.empty:
            ax = missing.sort_values().plot.barh(figsize=(10, max(4, len(missing) * 0.22)),
                                                title=f"Missingness — {ACTIVE_TABLE}")
            ax.set_xlabel("missing values (%)")
            plt.tight_layout()
            plt.show()
        else:
            print("No missing values in loaded rows.")

        numeric = active.select_dtypes(include=np.number)
        if not numeric.empty:
            display(numeric.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)

        categorical = active.select_dtypes(include=["object", "string", "category", "bool"])
        low_cardinality = [c for c in categorical if active[c].nunique(dropna=True) <= 30]
        for column in low_cardinality[:12]:
            print(f"\\n{column}")
            display(active[column].value_counts(dropna=False).head(20).to_frame("rows"))
        """
    ),
    code(
        """
        date_columns = [c for c in active if any(token in c.lower() for token in ("date", "time", "issued", "as_of", "fetched"))]
        date_coverage = []
        for column in date_columns:
            parsed = pd.to_datetime(active[column], errors="coerce", utc=True)
            if parsed.notna().any():
                date_coverage.append({"column": column, "parsed_rows": parsed.notna().sum(),
                                      "first": parsed.min(), "last": parsed.max()})
        display(pd.DataFrame(date_coverage))

        dimension_columns = [c for c in ["station_code", "station_id", "provider", "model", "source",
                                          "settlement_source", "data_source", "quality_flag"] if c in active]
        for column in dimension_columns:
            display(active[column].value_counts(dropna=False).head(30).rename("rows").to_frame())
        """
    ),
    markdown(
        """
        ## 6. Domain-specific weather and forecast checks

        These cells use the consolidated model-error, actual-high, settlement,
        and calibration tables when available. They skip cleanly on a partial pull.
        """
    ),
    code(
        """
        errors = tables.get("processed/model_errors.csv")
        if errors is not None and {"abs_error_f", "error_f"}.issubset(errors):
            dims = [c for c in ["station_code", "provider", "model"] if c in errors]
            error_summary = (errors.groupby(dims, dropna=False)
                             .agg(rows=("abs_error_f", "size"), mae_f=("abs_error_f", "mean"),
                                  median_ae_f=("abs_error_f", "median"), bias_f=("error_f", "mean"),
                                  p95_ae_f=("abs_error_f", lambda s: s.quantile(.95)))
                             .reset_index().sort_values("mae_f"))
            display(error_summary)

            if "station_code" in errors and "provider" in errors:
                pivot = errors.pivot_table(index="station_code", columns="provider", values="abs_error_f", aggfunc="mean")
                pivot.plot.bar(figsize=(13, 5), title="Mean absolute forecast error by station and provider")
                plt.ylabel("MAE (°F)")
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.show()

            if "forecast_horizon_hours" in errors:
                horizon = (errors.assign(horizon_bin=pd.cut(errors["forecast_horizon_hours"],
                                                            [-np.inf, 6, 12, 24, 48, 72, np.inf]))
                           .groupby(["horizon_bin", "provider"], observed=True)["abs_error_f"].mean().unstack())
                display(horizon)
                horizon.plot(marker="o", figsize=(11, 4), title="MAE by forecast horizon")
                plt.ylabel("MAE (°F)")
                plt.tight_layout()
                plt.show()
        else:
            print("processed/model_errors.csv is unavailable or lacks error columns.")
        """
    ),
    code(
        """
        actuals = tables.get("processed/actual_highs.csv")
        settlements = tables.get("processed/settlement_actual_highs.csv")
        if actuals is not None and settlements is not None:
            left = actuals.rename(columns={"station_code": "station", "date_local": "date"}).copy()
            right = settlements.rename(columns={"station_id": "station", "contract_date": "date"}).copy()
            left["date"] = pd.to_datetime(left["date"], errors="coerce")
            right["date"] = pd.to_datetime(right["date"], errors="coerce")
            comparison = left.merge(right, on=["station", "date"], how="inner", suffixes=("_actual", "_settlement"))
            if not comparison.empty and {"actual_high_f", "settlement_high_f"}.issubset(comparison):
                comparison["settlement_minus_actual_f"] = comparison["settlement_high_f"] - comparison["actual_high_f"]
                settlement_summary = (comparison.groupby("station")
                                      .agg(overlap_days=("date", "size"), mean_difference_f=("settlement_minus_actual_f", "mean"),
                                           mae_difference_f=("settlement_minus_actual_f", lambda s: s.abs().mean()),
                                           max_abs_difference_f=("settlement_minus_actual_f", lambda s: s.abs().max()))
                                      .sort_values("mae_difference_f", ascending=False))
                display(settlement_summary)
                display(comparison.loc[comparison["settlement_minus_actual_f"].abs() >= 2]
                        .sort_values("settlement_minus_actual_f", key=lambda s: s.abs(), ascending=False).head(50))
        """
    ),
    code(
        """
        samples = tables.get("calibration/calibration_samples.csv")
        if samples is not None:
            coverage_dims = [c for c in ["station_id", "provider", "timing_mode", "data_source"] if c in samples]
            calibration_coverage = (samples.groupby(coverage_dims, dropna=False)
                                    .agg(rows=(coverage_dims[0], "size"),
                                         first_date=("contract_date", "min"), last_date=("contract_date", "max"),
                                         target_missing_pct=("actual_high_f", lambda s: s.isna().mean() * 100))
                                    .reset_index())
            display(calibration_coverage)

            weather_features = [c for c in samples if any(token in c.lower() for token in
                                ["dewpoint", "humidity", "wind", "observed_", "disagreement", "cloud", "rain"])]
            if weather_features:
                feature_coverage = samples[weather_features].notna().mean().mul(100).sort_values()
                feature_coverage.plot.barh(figsize=(10, max(4, len(feature_coverage) * .23)),
                                           title="Calibration weather-feature coverage")
                plt.xlabel("non-missing (%)")
                plt.tight_layout()
                plt.show()
        """
    ),
    markdown(
        """
        ## 7. Model and database artifact catalog

        Binary model files are intentionally not deserialized during EDA. SQLite
        files are inspected only for table names and sizes, avoiding mutation of
        active Optuna studies.
        """
    ),
    code(
        """
        binary_artifacts = files[files["extension"].isin([".joblib", ".pkl", ".pickle", ".bin", ".model"])]
        display(binary_artifacts.drop(columns="path").sort_values("size_bytes", ascending=False))

        sqlite_rows = []
        for row in files.loc[files["extension"] == ".sqlite3"].itertuples(index=False):
            try:
                uri = f"file:{row.path.as_posix()}?mode=ro"
                with sqlite3.connect(uri, uri=True) as conn:
                    names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)["name"].tolist()
                sqlite_rows.append({"relative_path": row.relative_path, "size_mb": row.size_mb,
                                    "tables": ", ".join(names), "status": "ok"})
            except Exception as exc:
                sqlite_rows.append({"relative_path": row.relative_path, "size_mb": row.size_mb,
                                    "tables": "", "status": f"{type(exc).__name__}: {exc}"})
        sqlite_catalog = pd.DataFrame(sqlite_rows)
        display(sqlite_catalog)
        """
    ),
    markdown(
        """
        ## 8. Automated audit findings

        This final table turns the exploration into a short review queue. It is
        advisory: backups and experimental artifacts may be intentional.
        """
    ),
    code(
        """
        findings = []
        def add_finding(severity: str, area: str, finding: str, detail) -> None:
            findings.append({"severity": severity, "area": area, "finding": finding, "detail": detail})

        if backup_mask.any():
            add_finding("info", "storage", "Backup/test-like artifacts present",
                        f"{backup_mask.sum():,} files, {files.loc[backup_mask, 'size_mb'].sum():,.1f} MB")
        if not duplicate_candidates.empty:
            add_finding("info", "storage", "Same-name/same-size duplicate candidates",
                        f"{len(duplicate_candidates):,} candidate groups; hash to confirm")
        failed_schemas = schema_catalog[schema_catalog["status"] != "ok"]
        if not failed_schemas.empty:
            add_finding("warning", "schemas", "Representative files could not be read",
                        f"{len(failed_schemas):,} logical families")
        if "status" in table_catalog:
            failed_tables = table_catalog[table_catalog["status"] != "ok"]
            if not failed_tables.empty:
                add_finding("warning", "tables", "Consolidated tables could not be read", len(failed_tables))
        high_missing = table_catalog.loc[pd.to_numeric(table_catalog.get("missing_pct"), errors="coerce") >= 50, "table"].tolist()
        if high_missing:
            add_finding("review", "completeness", "Tables are at least 50% missing overall", ", ".join(high_missing))
        stale_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
        newest_by_area = pd.to_datetime(by_area["newest_utc"], errors="coerce", utc=True)
        stale_areas = by_area.loc[newest_by_area.notna() & (newest_by_area < stale_cutoff), "top_level"].tolist()
        if stale_areas:
            add_finding("review", "freshness", "No artifact updated in the last 30 days", ", ".join(stale_areas))
        if not findings:
            add_finding("ok", "overall", "No automatic warnings triggered", "Review domain plots for research conclusions")

        audit_findings = pd.DataFrame(findings)
        display(audit_findings)

        # Optional exports (uncomment to persist compact audit outputs):
        # out = ROOT / "outputs" / "eda"
        # out.mkdir(parents=True, exist_ok=True)
        # files.drop(columns="path").to_csv(out / "file_inventory.csv", index=False)
        # table_catalog.to_csv(out / "table_catalog.csv", index=False)
        # audit_findings.to_csv(out / "audit_findings.csv", index=False)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} ({len(cells)} cells)")
