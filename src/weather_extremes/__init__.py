"""Weather Trend Analysis and Extreme Temperature Pattern Detection."""

from .config import load_config, Config
from .data_loader import load_station_data, get_station_info, parse_dly
from .features import build_feature_frame, FEATURE_EXCLUDE
from .labeling import label_extremes, CLASS_NAMES, CLASS_ORDER
from .model import build_models, time_split, fit_model, predict_model
from .evaluate import evaluate_model, season_error_breakdown
from .pipeline import run_pipeline

__all__ = [
    "load_config",
    "Config",
    "load_station_data",
    "get_station_info",
    "parse_dly",
    "build_feature_frame",
    "FEATURE_EXCLUDE",
    "label_extremes",
    "CLASS_NAMES",
    "CLASS_ORDER",
    "build_models",
    "time_split",
    "fit_model",
    "predict_model",
    "evaluate_model",
    "season_error_breakdown",
    "run_pipeline",
]

__version__ = "0.1.0"
