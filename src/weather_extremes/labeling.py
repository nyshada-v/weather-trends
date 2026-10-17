"""Extreme-event labeling from seasonal-baseline anomalies."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "normal", 1: "extreme_hot", 2: "extreme_cold"}
CLASS_ORDER = ["normal", "extreme_hot", "extreme_cold"]
LABEL_TO_CODE = {name: code for code, name in CLASS_NAMES.items()}


def compute_thresholds(anomaly: pd.Series, hot_quantile: float, cold_quantile: float) -> dict:
    clean = anomaly.dropna()
    if clean.empty:
        raise ValueError("cannot compute thresholds from an empty anomaly series")
    return {
        "hot_threshold": float(clean.quantile(hot_quantile)),
        "cold_threshold": float(clean.quantile(cold_quantile)),
        "hot_quantile": float(hot_quantile),
        "cold_quantile": float(cold_quantile),
        "n_observations": int(clean.shape[0]),
    }


def label_extremes(
    df: pd.DataFrame,
    hot_quantile: float = 0.95,
    cold_quantile: float = 0.05,
    anomaly_col: str = "anomaly",
) -> tuple[pd.DataFrame, dict]:
    if anomaly_col not in df.columns:
        raise KeyError(f"missing anomaly column {anomaly_col!r}; run feature engineering first")
    thresholds = compute_thresholds(df[anomaly_col], hot_quantile, cold_quantile)
    anomaly = df[anomaly_col]
    labels = np.zeros(len(df), dtype="float64")
    labels[anomaly.to_numpy() > thresholds["hot_threshold"]] = LABEL_TO_CODE["extreme_hot"]
    labels[anomaly.to_numpy() < thresholds["cold_threshold"]] = LABEL_TO_CODE["extreme_cold"]
    labels[anomaly.isna().to_numpy()] = np.nan
    df["target"] = labels
    df["target_name"] = pd.Series(labels, index=df.index).map(CLASS_NAMES)
    df["anomaly_flag"] = df[anomaly_col].abs()
    thresholds["class_counts"] = (
        df["target_name"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict()
    )
    return df, thresholds
