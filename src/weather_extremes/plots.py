"""Figure generation for trends, anomalies, model performance and diagnostics."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve

from .labeling import CLASS_NAMES, CLASS_ORDER

sns.set_theme(style="whitegrid")
PALETTE = {"normal": "#4C72B0", "extreme_hot": "#C44E52", "extreme_cold": "#55A868"}


def _ensure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_temperature_trend(frame: pd.DataFrame, station_name: str, path: str) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame.index, frame["TMAX"], color="#B0B0B0", linewidth=0.5, label="Daily TMAX")
    rolling = frame["TMAX"].rolling(365, min_periods=90).mean()
    ax.plot(frame.index, rolling, color="#C44E52", linewidth=2.0, label="365-day mean TMAX")
    if "target_name" in frame.columns:
        hot = frame[frame["target_name"] == "extreme_hot"]
        cold = frame[frame["target_name"] == "extreme_cold"]
        ax.scatter(hot.index, hot["TMAX"], s=6, color=PALETTE["extreme_hot"], label="extreme hot", alpha=0.6)
        ax.scatter(cold.index, cold["TMAX"], s=6, color=PALETTE["extreme_cold"], label="extreme cold", alpha=0.6)
    ax.set_title(f"Temperature trend and extreme flags - {station_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("TMAX (deg C)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_climatology(frame: pd.DataFrame, path: str) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(12, 5))
    doy = frame.index.dayofyear
    grouped = frame.groupby(doy)["TMAX"]
    mean = grouped.mean()
    low = grouped.quantile(0.10)
    high = grouped.quantile(0.90)
    ax.fill_between(mean.index, low, high, color="#4C72B0", alpha=0.2, label="10th-90th percentile")
    ax.plot(mean.index, mean, color="#4C72B0", label="Seasonal baseline (TMAX)")
    ax2 = ax.twinx()
    tmin = frame.groupby(doy)["TMIN"].mean()
    ax2.plot(tmin.index, tmin, color="#55A868", linewidth=1.2, label="Seasonal baseline (TMIN)")
    ax2.set_ylabel("TMIN (deg C)", color="#55A868")
    ax2.tick_params(axis="y", colors="#55A868")
    ax.set_title("Seasonal baseline (day-of-year climatology)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("TMAX (deg C)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_anomaly_distribution(frame: pd.DataFrame, thresholds: dict, path: str) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    anomaly = frame["anomaly"].dropna()
    ax.hist(anomaly, bins=80, color="#4C72B0", alpha=0.7)
    ax.axvline(thresholds["hot_threshold"], color=PALETTE["extreme_hot"], linestyle="--", label="hot threshold")
    ax.axvline(thresholds["cold_threshold"], color=PALETTE["extreme_cold"], linestyle="--", label="cold threshold")
    ax.set_title("Deseasonalised temperature anomaly distribution")
    ax.set_xlabel("TMAX anomaly (deg C)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annual_trend(frame: pd.DataFrame, path: str) -> dict:
    _ensure(path)
    annual = frame.groupby(frame["year"])[["TMAX", "TMIN"]].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    summary = {}
    for column, color in (("TMAX", "#C44E52"), ("TMIN", "#55A868")):
        years = annual.index.to_numpy(dtype="float64")
        values = annual[column].to_numpy(dtype="float64")
        mask = ~np.isnan(values)
        ax.plot(years[mask], values[mask], marker="o", markersize=3, color=color, label=f"{column} annual mean")
        if mask.sum() > 2:
            result = stats.linregress(years[mask], values[mask])
            ax.plot(years[mask], result.intercept + result.slope * years[mask], color=color, linestyle="--", linewidth=1.5)
            summary[f"{column}_slope_per_decade"] = float(result.slope * 10.0)
            summary[f"{column}_p_value"] = float(result.pvalue)
    ax.set_title("Annual mean temperature trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (deg C)")
    ax.legend()
    figs = summary
    text_lines = [f"{key.split('_slope_per_decade')[0]}: {value:+.3f} deg C/decade" for key, value in figs.items() if key.endswith("slope_per_decade")]
    if text_lines:
        ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return figs


def plot_event_frequency(frame: pd.DataFrame, path: str) -> None:
    _ensure(path)
    counts = (
        frame.groupby(["year", "target_name"]).size().unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(counts))
    for name in CLASS_ORDER:
        values = counts[name].to_numpy(dtype="float64")
        ax.bar(counts.index, values, bottom=bottom, label=name, color=PALETTE.get(name, None))
        bottom += values
    ax.set_title("Extreme-event frequency per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of days")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, path: str) -> None:
    _ensure(path)
    matrix = np.array(metrics["confusion_matrix"], dtype="float64")
    labels = metrics["confusion_labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true: pd.Series, predictions: pd.DataFrame, path: str) -> None:
    _ensure(path)
    y_true = pd.Series(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    for code, name in CLASS_NAMES.items():
        column = f"p_{name}"
        if column not in predictions.columns:
            continue
        binary = (y_true == code).astype(int)
        if binary.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(binary, predictions[column])
        ax.plot(fpr, tpr, label=f"{name}")
    binary_extreme = (y_true != 0).astype(int)
    if "p_extreme" in predictions.columns and binary_extreme.nunique() > 1:
        fpr, tpr, _ = roc_curve(binary_extreme, predictions["p_extreme"])
        ax.plot(fpr, tpr, linestyle="--", color="black", label="extreme vs normal")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_title("ROC curves (one-vs-rest)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, model_name: str, path: str, top_n: int = 20) -> None:
    _ensure(path)
    if importance.empty:
        return
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#4C72B0")
    ax.set_title(f"Feature importance - {model_name} (top {len(top)})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_season_error(breakdown: pd.DataFrame, path: str) -> None:
    _ensure(path)
    if breakdown.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    positions = np.arange(len(breakdown))
    ax.bar(positions - width / 2, breakdown["false_positives"], width, label="false positives", color="#C44E52")
    ax.bar(positions + width / 2, breakdown["false_negatives"], width, label="false negatives", color="#4C72B0")
    ax.set_xticks(positions)
    ax.set_xticklabels(breakdown["season"])
    ax.set_title("Seasonal error breakdown")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_probability_timeline(
    test_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    alert_threshold: float,
    path: str,
) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(predictions.index, predictions["p_extreme"], color="#4C72B0", linewidth=0.8, label="P(extreme)")
    ax.axhline(alert_threshold, color="#C44E52", linestyle="--", linewidth=1.2, label=f"alert threshold = {alert_threshold:g}")
    actual_extreme = predictions.index[(test_frame["target"] != 0).to_numpy()]
    ax.scatter(actual_extreme, np.full(len(actual_extreme), ax.get_ylim()[1] * 0.0), color="black", marker="|", s=40, label="actual extreme")
    ax.set_title("Predicted extreme probability over the test window")
    ax.set_xlabel("Date")
    ax.set_ylabel("P(extreme)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
