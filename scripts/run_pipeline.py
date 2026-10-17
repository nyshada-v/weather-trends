"""Command-line entry point for the weather extremes pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from weather_extremes.config import load_config
from weather_extremes.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather trend analysis and extreme temperature detection")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--station", default=None, help="GHCN station id, e.g. USW00094728")
    parser.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    parser.add_argument("--models", nargs="+", default=None, help="models to train")
    parser.add_argument("--primary", default=None, help="primary model name or 'auto'")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--hot-quantile", type=float, default=None)
    parser.add_argument("--cold-quantile", type=float, default=None)
    parser.add_argument("--no-download", action="store_true", help="use cached raw files only")
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {"data": {}, "model": {}, "labeling": {}, "project": {}}
    if args.station:
        overrides["data"]["station_id"] = args.station
    if args.start:
        overrides["data"]["start_date"] = args.start
    if args.end:
        overrides["data"]["end_date"] = args.end
    if args.no_download:
        overrides["data"]["download"] = False
    if args.models:
        overrides["model"]["models"] = args.models
    if args.primary:
        overrides["model"]["primary"] = args.primary
    if args.seed is not None:
        overrides["project"]["random_seed"] = args.seed
    if args.hot_quantile is not None:
        overrides["labeling"]["hot_quantile"] = args.hot_quantile
    if args.cold_quantile is not None:
        overrides["labeling"]["cold_quantile"] = args.cold_quantile
    return {key: value for key, value in overrides.items() if value}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    config = load_config(args.config, overrides=build_overrides(args))
    results = run_pipeline(config)
    metrics = results["test_metrics"]
    print("\n===== RESULTS =====")
    print(f"Station:        {results['station'].get('name', results['station']['station_id'])}")
    print(f"Selected model: {results['selected_model']}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision(macro): {metrics['precision_macro']:.4f}")
    print(f"Recall(macro):    {metrics['recall_macro']:.4f}")
    print(f"F1(macro):        {metrics['f1_macro']:.4f}")
    print(f"F1(extreme):      {metrics['f1_extreme_macro']:.4f}")
    auc_extreme = metrics.get("roc_auc_extreme_vs_normal")
    print(f"ROC-AUC(extreme vs normal): {auc_extreme:.4f}" if auc_extreme is not None else "ROC-AUC(extreme vs normal): n/a")
    print(f"ROC-AUC(OVR macro):         {metrics['roc_auc_ovr_macro']:.4f}" if metrics.get("roc_auc_ovr_macro") else "ROC-AUC(OVR macro): n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
