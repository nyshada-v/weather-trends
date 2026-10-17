"""End-to-end orchestration of the extreme-temperature detection workflow."""

from __future__ import annotations

import json
import logging
import os

import joblib
import pandas as pd

from . import evaluate as evaluate_module
from . import plots
from .config import Config, _merge, load_config, resolve_paths
from .data_loader import (
    download_station_dly,
    download_station_metadata,
    get_station_info,
    parse_dly,
    parse_station_metadata,
    values_to_wide,
)
from .features import build_feature_frame
from .labeling import CLASS_ORDER, label_extremes
from .model import build_models, coefficient_table, feature_importance, fit_model, predict_model, time_split

logger = logging.getLogger("weather_extremes")


def _ensure_dirs(config: Config) -> None:
    dirs = [
        config["data"]["raw_dir"],
        config["data"]["interim_dir"],
        config["data"]["processed_dir"],
        config["output"]["reports_dir"],
        config["output"]["figures_dir"],
        config["output"]["metrics_dir"],
        config["output"]["models_dir"],
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def load_dataset(config: Config) -> tuple[pd.DataFrame, dict]:
    data_cfg = config["data"]
    station_id = data_cfg["station_id"]
    raw_dir = data_cfg["raw_dir"]
    elements = list(data_cfg["elements"])
    base_url = data_cfg["base_url"]

    if data_cfg.get("download", True):
        download_station_dly(station_id, raw_dir, base_url=base_url, force=bool(data_cfg.get("force_download", False)))
        download_station_metadata(raw_dir, base_url=base_url, force=bool(data_cfg.get("force_download", False)))

    dly_path = os.path.join(raw_dir, f"{station_id}.dly")
    dly = parse_dly(dly_path, station_id=station_id)
    wide = values_to_wide(dly, elements, units=data_cfg.get("units", "metric"))
    full_index = pd.date_range(data_cfg["start_date"], data_cfg["end_date"], freq="D")
    wide = wide.reindex(full_index)
    wide.index.name = "date"

    metadata_path = os.path.join(raw_dir, "ghcnd-stations.txt")
    station = None
    if os.path.exists(metadata_path):
        metadata = parse_station_metadata(metadata_path)
        try:
            station = get_station_info(station_id, metadata)
        except KeyError:
            station = None
    station_dict = (
        {
            "station_id": station.station_id,
            "name": station.name,
            "state": station.state,
            "latitude": station.latitude,
            "longitude": station.longitude,
            "elevation": station.elevation,
        }
        if station
        else {"station_id": station_id}
    )
    return wide, station_dict


def prepare_features(config: Config, wide: pd.DataFrame):
    frame, feature_cols = build_feature_frame(wide, config)
    labeling_cfg = config["labeling"]
    frame, thresholds = label_extremes(
        frame,
        hot_quantile=float(labeling_cfg["hot_quantile"]),
        cold_quantile=float(labeling_cfg["cold_quantile"]),
    )
    required = [column for column in feature_cols if column in frame.columns] + ["target"]
    frame = frame.dropna(subset=required)
    frame = frame.sort_index()
    return frame, feature_cols, thresholds


def run_pipeline(config: Config | None = None, overrides: dict | None = None) -> dict:
    if config is None:
        config = load_config(overrides=overrides)
    elif overrides:
        config = Config(_merge(dict(config), overrides))
        resolve_paths(config)
    _ensure_dirs(config)

    seed = int(config["project"]["random_seed"])
    logger.info("Loading station %s", config["data"]["station_id"])
    wide, station = load_dataset(config)
    interim_path = os.path.join(config["data"]["interim_dir"], "station_daily.parquet")
    try:
        wide.to_parquet(interim_path)
    except Exception:
        wide.to_csv(os.path.join(config["data"]["interim_dir"], "station_daily.csv"))

    frame, feature_cols, thresholds = prepare_features(config, wide)
    processed_path = os.path.join(config["data"]["processed_dir"], "features_labeled.parquet")
    try:
        frame.to_parquet(processed_path)
    except Exception:
        frame.to_csv(os.path.join(config["data"]["processed_dir"], "features_labeled.csv"))

    split_cfg = config["split"]
    train, val, test = time_split(frame, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    logger.info("Split sizes -> train=%d val=%d test=%d", len(train), len(val), len(test))

    models = build_models(config)
    results = {
        "station": station,
        "thresholds": thresholds,
        "feature_columns": feature_cols,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "models": {},
        "trend": {},
    }

    alert_threshold = float(config["evaluation"]["alert_threshold"])
    roc_mode = str(config["evaluation"].get("roc_multi_class", "ovr"))
    selected_name = config["model"].get("primary")
    selection_scores = {}

    fig_dir = config["output"]["figures_dir"]
    metrics_dir = config["output"]["metrics_dir"]
    models_dir = config["output"]["models_dir"]

    for name, model in models.items():
        fit_model(model, train, feature_cols)
        val_pred = predict_model(model, val, feature_cols)
        test_pred = predict_model(model, test, feature_cols)
        val_metrics = evaluate_module.evaluate_model(val["target"], val_pred, alert_threshold, roc_mode)
        test_metrics = evaluate_module.evaluate_model(test["target"], test_pred, alert_threshold, roc_mode)
        importance = feature_importance(model, [c for c in feature_cols if c in train.columns])
        results["models"][name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "feature_importance": importance.to_dict(orient="records"),
        }
        selection_scores[name] = val_metrics.get("f1_extreme_macro") or 0.0
        importance.to_csv(os.path.join(metrics_dir, f"feature_importance_{name}.csv"), index=False)
        if name == "logistic_regression":
            coefficient_table(model, [c for c in feature_cols if c in train.columns]).to_csv(
                os.path.join(metrics_dir, "coefficients_logistic_regression.csv"), index=False
            )
        evaluate_module.write_metrics(test_metrics, os.path.join(metrics_dir, f"metrics_{name}_test.json"))
        joblib.dump(model, os.path.join(models_dir, f"{name}.joblib"))
        plots.plot_feature_importance(importance, name, os.path.join(fig_dir, f"feature_importance_{name}.png"))

    if selected_name is None or selected_name == "auto":
        selected_name = max(selection_scores, key=selection_scores.get)
    results["selected_model"] = selected_name
    results["selection_scores"] = selection_scores

    selected_model = models[selected_name]
    test_pred = predict_model(selected_model, test, feature_cols)
    val_pred = predict_model(selected_model, val, feature_cols)
    results["test_metrics"] = results["models"][selected_name]["test"]

    predictions_path = os.path.join(metrics_dir, f"predictions_test_{selected_name}.csv")
    test_pred.assign(target=test["target"], target_name=test["target_name"], season=test["season"]).to_csv(predictions_path)

    breakdown = evaluate_module.season_error_breakdown(
        test["season"], test["target"], test_pred["pred"], test_pred["p_extreme"]
    )
    breakdown.to_csv(os.path.join(metrics_dir, f"season_error_breakdown_{selected_name}.csv"), index=False)
    results["season_error_breakdown"] = breakdown.to_dict(orient="records")

    plots.plot_temperature_trend(frame, station.get("name", station["station_id"]), os.path.join(fig_dir, "temperature_trend.png"))
    plots.plot_climatology(frame, os.path.join(fig_dir, "seasonal_climatology.png"))
    plots.plot_anomaly_distribution(frame, thresholds, os.path.join(fig_dir, "anomaly_distribution.png"))
    results["trend"] = plots.plot_annual_trend(frame, os.path.join(fig_dir, "annual_trend.png"))
    plots.plot_event_frequency(frame, os.path.join(fig_dir, "event_frequency.png"))
    plots.plot_confusion_matrix(results["test_metrics"], os.path.join(fig_dir, "confusion_matrix.png"))
    plots.plot_roc_curves(test["target"], test_pred, os.path.join(fig_dir, "roc_curves.png"))
    plots.plot_season_error(breakdown, os.path.join(fig_dir, "season_error_breakdown.png"))
    plots.plot_probability_timeline(test, test_pred, alert_threshold, os.path.join(fig_dir, "probability_timeline.png"))

    evaluate_module.write_metrics(results["test_metrics"], os.path.join(metrics_dir, "metrics_selected_test.json"))
    with open(os.path.join(metrics_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "station": station,
                "thresholds": thresholds,
                "split": results["split"],
                "selected_model": selected_name,
                "selection_scores": selection_scores,
                "test_metrics": {key: value for key, value in results["test_metrics"].items() if key != "classification_report"},
                "trend": results["trend"],
            },
            handle,
            indent=2,
        )
    logger.info("Selected model: %s", selected_name)
    logger.info("Test accuracy: %.4f | macro-F1: %.4f", results["test_metrics"]["accuracy"], results["test_metrics"]["f1_macro"])
    return results
