"""Feature engineering: cleaning, calendar, rolling, lagged and anomaly signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_EXCLUDE = {
    "target",
    "target_name",
    "anomaly",
    "anomaly_flag",
    "season",
    "date",
    "TMAX",
    "TMIN",
    "PRCP",
    "SNOW",
    "SNWD",
}

TEMP_ELEMENTS = ("TMAX", "TMIN")
SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
SEASON_CODE = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    index = df.index
    df["year"] = index.year
    df["month"] = index.month
    df["dayofyear"] = index.dayofyear
    season = index.month.map(SEASON_BY_MONTH)
    df["season"] = season
    df["season_code"] = season.map(SEASON_CODE).astype("float64")
    angle = 2.0 * np.pi * df["dayofyear"].to_numpy(dtype="float64") / 365.25
    df["doy_sin"] = np.sin(angle)
    df["doy_cos"] = np.cos(angle)
    return df


def compute_climatology(df: pd.DataFrame, elements: list[str], window: int = 15, min_years: int = 5) -> pd.DataFrame:
    work = pd.DataFrame({"doy": df.index.dayofyear})
    for element in elements:
        if element in df.columns:
            work[element] = df[element].to_numpy(dtype="float64")
    climatology = pd.DataFrame(index=pd.RangeIndex(1, 367, name="doy"))
    for element in elements:
        if element not in work.columns:
            continue
        grp = work.groupby("doy")[element]
        mean = grp.mean()
        count = grp.count()
        mean = mean.where(count >= min_years)
        values = mean.reindex(range(1, 367)).to_numpy(dtype="float64")
        size = values.shape[0]
        smoothed = np.full(size, np.nan)
        for i in range(size):
            neighbours = [values[(i + offset) % size] for offset in range(-window, window + 1)]
            neighbours = [value for value in neighbours if not np.isnan(value)]
            if neighbours:
                smoothed[i] = float(np.mean(neighbours))
        climatology[f"{element}_clim"] = smoothed
    return climatology


def attach_climatology(df: pd.DataFrame, climatology: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    doy = df.index.dayofyear
    for element in elements:
        column = f"{element}_clim"
        if column in climatology.columns:
            df[column] = doy.map(climatology[column]).to_numpy(dtype="float64")
    return df


def add_anomaly_columns(df: pd.DataFrame, elements: list[str], target_element: str = "TMAX") -> pd.DataFrame:
    for element in elements:
        column = f"{element}_clim"
        if column in df.columns:
            df[f"{element}_anomaly"] = df[element] - df[column]
    if f"{target_element}_anomaly" in df.columns:
        df["anomaly"] = df[f"{target_element}_anomaly"]
    return df


def add_lagged_features(df: pd.DataFrame, lags: list[int], elements: list[str]) -> pd.DataFrame:
    for element in elements:
        for lag in lags:
            df[f"{element}_lag{lag}"] = df[element].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int], elements: list[str], precip_context: bool = True) -> pd.DataFrame:
    for element in elements:
        shifted = df[element].shift(1)
        for window in windows:
            df[f"{element}_roll{window}_mean"] = shifted.rolling(window).mean()
            df[f"{element}_roll{window}_std"] = shifted.rolling(window).std()
    if precip_context and "PRCP" in df.columns:
        precip = df["PRCP"].shift(1)
        df["PRCP_lag1"] = df["PRCP"].shift(1)
        for window in windows:
            df[f"PRCP_roll{window}_sum"] = precip.rolling(window).sum()
    if "SNOW" in df.columns:
        df["SNOW_lag1"] = df["SNOW"].shift(1)
    if "SNWD" in df.columns:
        df["SNWD_lag1"] = df["SNWD"].shift(1)
    return df


def add_anomaly_lag_features(df: pd.DataFrame, lags: list[int], elements: list[str]) -> pd.DataFrame:
    for element in elements:
        if f"{element}_anomaly" not in df.columns:
            continue
        for lag in lags:
            df[f"{element}_anomaly_lag{lag}"] = df[f"{element}_anomaly"].shift(lag)
    return df


def feature_columns(df: pd.DataFrame, include_current: bool = False) -> list[str]:
    columns = []
    for column in df.columns:
        if column in FEATURE_EXCLUDE:
            continue
        if column.endswith("_clim") or column.endswith("_anomaly"):
            if include_current and column.endswith("_anomaly") and "_lag" not in column:
                columns.append(column)
            continue
        if df[column].dtype == object:
            continue
        columns.append(column)
    return columns


def build_feature_frame(df: pd.DataFrame, config) -> tuple[pd.DataFrame, list[str]]:
    features_cfg = config["features"]
    elements = list(config["data"]["elements"])
    lags = list(features_cfg["lag_days"])
    windows = list(features_cfg["rolling_windows"])
    include_current = bool(features_cfg.get("include_current_observation", False))

    frame = df.copy()
    frame = add_calendar_features(frame)
    climatology = compute_climatology(
        frame,
        [element for element in elements if element in frame.columns],
        window=int(features_cfg["climatology_window"]),
        min_years=int(features_cfg["min_years_for_baseline"]),
    )
    present_elements = [element for element in elements if element in frame.columns]
    frame = attach_climatology(frame, climatology, present_elements)
    frame = add_anomaly_columns(frame, present_elements, target_element="TMAX")

    lag_elements = [element for element in TEMP_ELEMENTS if element in frame.columns]
    frame = add_lagged_features(frame, lags, lag_elements)
    frame = add_rolling_features(
        frame,
        windows,
        lag_elements,
        precip_context=bool(features_cfg.get("include_precip_context", True)),
    )
    frame = add_anomaly_lag_features(frame, lags, lag_elements)

    columns = feature_columns(frame, include_current=include_current)
    return frame, columns
