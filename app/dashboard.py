"""Streamlit dashboard for weather trend analysis and extreme detection."""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from weather_extremes.config import load_config
from weather_extremes.labeling import CLASS_ORDER
from weather_extremes.pipeline import run_pipeline

st.set_page_config(page_title="Weather Extremes", page_icon="🌡️", layout="wide")


def load_config_safe():
    try:
        return load_config()
    except Exception as exc:
        st.error(f"Could not load config.yaml: {exc}")
        st.stop()


def read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_frame(path_base: str) -> pd.DataFrame | None:
    for extension, reader in (("parquet", pd.read_parquet), ("csv", lambda p: pd.read_csv(p, index_col=0, parse_dates=True))):
        path = f"{path_base}.{extension}"
        if os.path.exists(path):
            try:
                return reader(path)
            except Exception:
                continue
    return None


def available_models(metrics_dir: str) -> list[str]:
    models = []
    for path in glob.glob(os.path.join(metrics_dir, "metrics_*_test.json")):
        name = os.path.basename(path)[len("metrics_") : -len("_test.json")]
        if name != "selected":
            models.append(name)
    return sorted(models)


def figure_path(figures_dir: str, name: str) -> str | None:
    path = os.path.join(figures_dir, name)
    return path if os.path.exists(path) else None


def metric_card_row(metrics: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.4f}")
    columns[1].metric("Precision (macro)", f"{metrics.get('precision_macro', float('nan')):.4f}")
    columns[2].metric("Recall (macro)", f"{metrics.get('recall_macro', float('nan')):.4f}")
    columns[3].metric("F1 (macro)", f"{metrics.get('f1_macro', float('nan')):.4f}")
    columns = st.columns(4)
    columns[0].metric("F1 (extreme, macro)", f"{metrics.get('f1_extreme_macro', float('nan')):.4f}")
    auc = metrics.get("roc_auc_extreme_vs_normal")
    columns[1].metric("ROC-AUC (extreme vs normal)", f"{auc:.4f}" if auc is not None else "n/a")
    ovr = metrics.get("roc_auc_ovr_macro")
    columns[2].metric("ROC-AUC (OVR macro)", f"{ovr:.4f}" if ovr is not None else "n/a")
    columns[3].metric("Observations (test)", f"{metrics.get('n_observations', 0):,}")


def render_sidebar(config) -> str:
    with st.sidebar:
        st.header("Controls")
        metrics_dir = config["output"]["metrics_dir"]
        models = available_models(metrics_dir)
        if not models:
            st.warning("No metrics found. Run the pipeline first.")
            st.stop()
        configured = config["model"].get("primary")
        default_index = models.index(configured) if configured in models else 0
        selected = st.selectbox("Model", models, index=default_index)
        st.divider()
        st.subheader("Run pipeline")
        st.caption("Re-download/process data and retrain the models.")
        if st.button("Run / refresh pipeline", use_container_width=True):
            with st.spinner("Running pipeline..."):
                try:
                    run_pipeline(config)
                    st.success("Pipeline finished.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
        st.divider()
        st.caption(f"Project root: {PROJECT_ROOT}")
    return selected


def overview_tab(config, summary: dict, metrics: dict) -> None:
    station = summary.get("station", {})
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Station")
        st.write(
            {
                "station_id": station.get("station_id"),
                "name": station.get("name"),
                "state": station.get("state"),
                "latitude": station.get("latitude"),
                "longitude": station.get("longitude"),
                "elevation_m": station.get("elevation"),
            }
        )
    with right:
        st.subheader("Scope")
        st.write(
            {
                "window": f"{config['data']['start_date']} to {config['data']['end_date']}",
                "train": summary.get("split", {}).get("train"),
                "validation": summary.get("split", {}).get("val"),
                "test": summary.get("split", {}).get("test"),
                "selected_model": summary.get("selected_model"),
            }
        )

    st.subheader("Accuracy and performance metrics")
    metric_card_row(metrics)

    thresholds = summary.get("thresholds", {})
    if thresholds:
        st.subheader("Extreme-event labeling")
        cols = st.columns(3)
        cols[0].metric("Hot threshold (deg C anomaly)", f"{thresholds.get('hot_threshold', float('nan')):.2f}")
        cols[1].metric("Cold threshold (deg C anomaly)", f"{thresholds.get('cold_threshold', float('nan')):.2f}")
        counts = thresholds.get("class_counts", {})
        cols[2].metric("Total labeled days", f"{thresholds.get('n_observations', 0):,}")
        st.bar_chart(pd.Series(counts, name="days"))

    trend = summary.get("trend", {})
    if trend:
        st.subheader("Long-term trend")
        tcols = st.columns(2)
        if "TMAX_slope_per_decade" in trend:
            tcols[0].metric("TMAX trend (deg C / decade)", f"{trend['TMAX_slope_per_decade']:+.3f}")
        if "TMIN_slope_per_decade" in trend:
            tcols[1].metric("TMIN trend (deg C / decade)", f"{trend['TMIN_slope_per_decade']:+.3f}")


def trends_tab(config) -> None:
    frame = read_frame(os.path.join(config["data"]["processed_dir"], "features_labeled"))
    if frame is None:
        st.info("Processed features not found. Run the pipeline.")
        return
    station_name = config["data"].get("station_name", config["data"]["station_id"])
    st.subheader(f"Daily temperature - {station_name}")
    window = st.select_slider("Smoothing window (days)", options=[1, 7, 30, 90, 365], value=30)
    series = pd.DataFrame({"TMAX": frame["TMAX"], "TMIN": frame["TMIN"]})
    if window > 1:
        series = series.rolling(window, min_periods=1).mean()
    st.line_chart(series)

    st.subheader("Annual mean temperature")
    annual = frame.groupby("year")[["TMAX", "TMIN"]].mean()
    st.line_chart(annual)

    st.subheader("Extreme-event frequency per year")
    counts = (
        frame.groupby(["year", "target_name"]).size().unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0)
    )
    st.bar_chart(counts)

    image = figure_path(config["output"]["figures_dir"], "annual_trend.png")
    if image:
        st.image(image, caption="Annual trend with fitted slopes")


def performance_tab(config, model_name: str, metrics: dict) -> None:
    figures_dir = config["output"]["figures_dir"]
    st.subheader("Confusion matrix")
    matrix = metrics.get("confusion_matrix")
    labels = metrics.get("confusion_labels", CLASS_ORDER)
    if matrix:
        st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), use_container_width=True)
    image = figure_path(figures_dir, "confusion_matrix.png")
    if image:
        st.image(image, caption="Confusion matrix")

    st.subheader("Per-class metrics")
    per_class = metrics.get("per_class", {})
    if per_class:
        st.dataframe(pd.DataFrame(per_class).T, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curves")
        image = figure_path(figures_dir, "roc_curves.png")
        if image:
            st.image(image)
    with right:
        st.subheader("Seasonal error breakdown")
        image = figure_path(figures_dir, "season_error_breakdown.png")
        if image:
            st.image(image)

    st.subheader("Operational alert view")
    st.write(
        {
            "alert_threshold": metrics.get("alert_threshold"),
            "alert_precision": metrics.get("alert_precision"),
            "alert_recall": metrics.get("alert_recall"),
            "alert_f1": metrics.get("alert_f1"),
        }
    )

    st.subheader(f"Feature importance - {model_name}")
    importance_path = os.path.join(config["output"]["metrics_dir"], f"feature_importance_{model_name}.csv")
    if os.path.exists(importance_path):
        importance = pd.read_csv(importance_path).head(20).set_index("feature")
        st.bar_chart(importance)
        st.dataframe(importance.reset_index(), use_container_width=True)


def predictions_tab(config, model_name: str) -> None:
    metrics_dir = config["output"]["metrics_dir"]
    path = os.path.join(metrics_dir, f"predictions_test_{model_name}.csv")
    if not os.path.exists(path):
        st.info("Predictions not found. Run the pipeline.")
        return
    predictions = pd.read_csv(path, index_col=0, parse_dates=True)
    st.subheader("Predicted extreme probability over the test window")
    st.line_chart(predictions[["p_extreme"]])

    threshold = st.slider("Alert threshold", 0.0, 1.0, 0.5, 0.01)
    actual_extreme = (predictions["target"] != 0).astype(int)
    alerts = (predictions["p_extreme"] >= threshold).astype(int)
    tp = int(((actual_extreme == 1) & (alerts == 1)).sum())
    fp = int(((actual_extreme == 0) & (alerts == 1)).sum())
    fn = int(((actual_extreme == 1) & (alerts == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    cols = st.columns(4)
    cols[0].metric("Precision @ threshold", f"{precision:.3f}")
    cols[1].metric("Recall @ threshold", f"{recall:.3f}")
    cols[2].metric("True positives", f"{tp:,}")
    cols[3].metric("False positives", f"{fp:,}")

    st.subheader("Highest-probability alerts")
    top = predictions.sort_values("p_extreme", ascending=False).head(25)
    top = top.assign(actual=top["target"].map({0: "normal", 1: "extreme_hot", 2: "extreme_cold"}))
    st.dataframe(top[["pred_name", "p_extreme", "actual"]], use_container_width=True)

    st.subheader("False-positive review (flagged but actually normal)")
    false_positives = predictions[(predictions["p_extreme"] >= threshold) & (predictions["target"] == 0)]
    st.dataframe(false_positives[["pred_name", "p_extreme"]].sort_values("p_extreme", ascending=False).head(25), use_container_width=True)


def about_tab(config) -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        1. **Data foundation** - NOAA GHCN-Daily daily summaries, quality-flagged values removed, aligned to a daily index.
        2. **Feature engineering** - cleaning, calendar/cyclical signals, causal rolling statistics, lagged temperatures, seasonal-baseline anomalies.
        3. **Labeling** - day-of-year climatology anomaly thresholded into `normal` / `extreme_hot` / `extreme_cold`.
        4. **Models** - logistic regression and random forest, trained on a chronological train/validation/test split.
        5. **Evaluation** - accuracy, precision, recall, F1, ROC-AUC, confusion matrix, alert performance, seasonal error breakdown.
        """
    )
    st.subheader("Configuration")
    st.json(
        {
            "station_id": config["data"]["station_id"],
            "window": [config["data"]["start_date"], config["data"]["end_date"]],
            "elements": config["data"]["elements"],
            "labeling": dict(config["labeling"]),
            "split": dict(config["split"]),
            "models": config["model"]["models"],
            "primary": config["model"].get("primary"),
        }
    )
    figure = figure_path(config["output"]["figures_dir"], "temperature_trend.png")
    if figure:
        st.image(figure, caption="Temperature trend and extreme flags")


def main() -> None:
    config = load_config_safe()
    selected = render_sidebar(config)
    summary = read_json(os.path.join(config["output"]["metrics_dir"], "run_summary.json"))
    metrics = read_json(os.path.join(config["output"]["metrics_dir"], f"metrics_{selected}_test.json"))

    st.title("🌡️ Weather Trend Analysis and Extreme Temperature Pattern Detection")
    st.caption(f"Station {config['data']['station_id']} | model: {selected}")

    tabs = st.tabs(["Overview", "Trends", "Model performance", "Predictions", "About"])
    with tabs[0]:
        overview_tab(config, summary, metrics)
    with tabs[1]:
        trends_tab(config)
    with tabs[2]:
        performance_tab(config, selected, metrics)
    with tabs[3]:
        predictions_tab(config, selected)
    with tabs[4]:
        about_tab(config)


if __name__ == "__main__":
    main()
