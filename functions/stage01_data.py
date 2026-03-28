from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .common import PipelineConfig
from .state import PipelineState


AQI_BREAKPOINTS = {
    "PM2.5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "PM10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "O3_8h_ppm": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "CO_8h": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
}


def _normalize_24h_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_dt = pd.to_datetime(date_series, errors="coerce", dayfirst=True)
    time_str = time_series.astype(str).str.strip()
    is_24 = time_str.eq("24:00")
    time_str = time_str.where(~is_24, "00:00")

    ts = pd.to_datetime(
        date_dt.dt.strftime("%Y-%m-%d") + " " + time_str,
        errors="coerce",
        format="%Y-%m-%d %H:%M",
    )
    ts = ts + pd.to_timedelta(is_24.astype(int), unit="D")
    return ts


def load_air_quality_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if {"Date", "Time"}.issubset(df.columns):
        df["datetime"] = _normalize_24h_time(df["Date"], df["Time"])
        df = df.drop(columns=["Date", "Time"])
    else:
        raise ValueError("Expected Date and Time columns not found in CSV.")

    for col in df.columns:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    return df


def describe_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    ).sort_values("missing_pct", ascending=False)


def drop_sparse_initial_period(df: pd.DataFrame, required_cols: List[str], missing_threshold: float) -> pd.DataFrame:
    temp = df.copy().dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    row_missing = temp[required_cols].isna().mean(axis=1)
    stable_idx = row_missing.le(missing_threshold)
    if stable_idx.any():
        temp = temp.loc[stable_idx.idxmax() :].reset_index(drop=True)
    return temp


def impute_time_series(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    temp = df.copy().dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    temp[cols] = temp[cols].interpolate(method="time", limit_direction="both")
    temp[cols] = temp[cols].ffill().bfill()
    return temp.reset_index()


def winsorize_iqr(df: pd.DataFrame, cols: List[str], k: float) -> pd.DataFrame:
    temp = df.copy()
    for col in cols:
        q1 = temp[col].quantile(0.25)
        q3 = temp[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        temp[col] = temp[col].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)
    return temp


def sub_index(value: float, breakpoints: List[Tuple[float, float, int, int]]) -> float:
    if pd.isna(value):
        return np.nan
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= value <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (value - bp_lo) + i_lo
    return np.nan


def derive_aqi(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    if "O3 8hr" in temp.columns:
        temp["O3_8h_ppm"] = np.where(temp["O3 8hr"] > 1.0, temp["O3 8hr"] / 1000.0, temp["O3 8hr"])

    pollutant_to_col = {
        "PM2.5": "PM2.5",
        "PM10": "PM10",
        "O3_8h_ppm": "O3_8h_ppm",
        "CO_8h": "CO 8hr" if "CO 8hr" in temp.columns else "CO",
    }

    subindex_cols = []
    for pollutant, col in pollutant_to_col.items():
        if col in temp.columns and pollutant in AQI_BREAKPOINTS:
            out_col = f"AQI_{pollutant}"
            temp[out_col] = temp[col].apply(lambda v: sub_index(v, AQI_BREAKPOINTS[pollutant]))
            subindex_cols.append(out_col)

    if not subindex_cols:
        raise ValueError("No valid pollutant columns found for AQI derivation.")

    temp["AQI_value"] = temp[subindex_cols].max(axis=1)
    bins = [-np.inf, 50, 100, 150, 200, 300, np.inf]
    labels = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
    temp["AQI_class"] = pd.cut(temp["AQI_value"], bins=bins, labels=labels)
    return temp


def run_stage01(config: PipelineConfig) -> PipelineState:
    state = PipelineState()
    df_raw = load_air_quality_data(config.csv_path)
    quality_raw = describe_data_quality(df_raw)

    required_for_keep = [c for c in ["PM2.5", "PM10", "NO2", "O3", "CO"] if c in df_raw.columns]
    num_cols = [c for c in df_raw.columns if c != "datetime"]

    df1 = drop_sparse_initial_period(df_raw, required_for_keep, config.missing_threshold)
    df2 = impute_time_series(df1, cols=num_cols)
    df_clean = winsorize_iqr(df2, cols=num_cols, k=config.winsor_k)

    df_clean["hour"] = df_clean["datetime"].dt.hour
    df_clean["dayofweek"] = df_clean["datetime"].dt.dayofweek
    df_clean["month"] = df_clean["datetime"].dt.month

    df_aqi = derive_aqi(df_clean)
    quality_clean = describe_data_quality(df_clean)

    overview = {
        "raw_shape": df_raw.shape,
        "clean_shape": df_clean.shape,
        "date_start": df_clean["datetime"].min(),
        "date_end": df_clean["datetime"].max(),
        "aqi_class_distribution": df_aqi["AQI_class"].value_counts(dropna=False),
    }

    state.put_data(df_raw=df_raw, df_clean=df_clean, df_aqi=df_aqi)
    state.put_metrics(quality_raw=quality_raw, quality_clean=quality_clean, stage01_overview=overview)

    if config.save_artifacts:
        out = config.artifact_dir / "stage01"
        out.mkdir(parents=True, exist_ok=True)
        quality_raw.to_csv(out / "quality_raw.csv", index=True)
        quality_clean.to_csv(out / "quality_clean.csv", index=True)
        df_clean.to_csv(out / "clean_dataset.csv", index=False)
        state.put_artifacts(stage01_dir=out)

    return state

