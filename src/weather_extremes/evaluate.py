"""Evaluation metrics, confusion matrices and error breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .labeling import CLASS_NAMES, CLASS_ORDER


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(y_true).shape[0] < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def evaluate_model(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    alert_threshold: float = 0.5,
    roc_multi_class: str = "ovr",
) -> dict:
    y_true = pd.Series(y_true).astype(int)
    y_pred = predictions["pred"].astype(int)
    labels = [code for code in sorted(CLASS_NAMES) if code in set(y_true.unique()) | set(y_pred.unique())]
    target_names = [CLASS_NAMES[code] for code in labels]

    metrics = {
        "n_observations": int(y_true.shape[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)),
    }

    per_class = {}
    for code, name in zip(labels, target_names):
        per_class[name] = {
            "precision": float(precision_score((y_true == code).astype(int), (y_pred == code).astype(int), zero_division=0)),
            "recall": float(recall_score((y_true == code).astype(int), (y_pred == code).astype(int), zero_division=0)),
            "f1": float(f1_score((y_true == code).astype(int), (y_pred == code).astype(int), zero_division=0)),
            "support": int((y_true == code).sum()),
        }
    metrics["per_class"] = per_class
    extreme_f1 = [per_class[name]["f1"] for name in ("extreme_hot", "extreme_cold") if name in per_class]
    metrics["f1_extreme_macro"] = float(np.mean(extreme_f1)) if extreme_f1 else None

    metrics["roc_auc_ovr_macro"] = None
    if len(labels) > 1:
        try:
            auc_scores = predictions[[f"p_{CLASS_NAMES[code]}" for code in labels]].to_numpy(dtype="float64")
            y_for_auc = np.array([labels.index(value) for value in y_true])
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_for_auc,
                    auc_scores,
                    multi_class=roc_multi_class,
                    average="macro",
                    labels=range(len(labels)),
                )
            )
        except Exception:
            metrics["roc_auc_ovr_macro"] = None

    y_extreme = (y_true != 0).astype(int)
    metrics["roc_auc_extreme_vs_normal"] = _safe_roc_auc(y_extreme.to_numpy(), predictions["p_extreme"].to_numpy(dtype="float64"))

    alert_pred = (predictions["p_extreme"] >= alert_threshold).astype(int)
    metrics["alert_threshold"] = float(alert_threshold)
    metrics["alert_precision"] = float(precision_score(y_extreme, alert_pred, zero_division=0))
    metrics["alert_recall"] = float(recall_score(y_extreme, alert_pred, zero_division=0))
    metrics["alert_f1"] = float(f1_score(y_extreme, alert_pred, zero_division=0))

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics["confusion_labels"] = target_names
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0, digits=4
    )
    return metrics


def season_error_breakdown(
    seasons: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    p_extreme: pd.Series | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "season": pd.Series(seasons).to_numpy(),
            "y_true": pd.Series(y_true).astype(int).to_numpy(),
            "y_pred": pd.Series(y_pred).astype(int).to_numpy(),
        }
    )
    if p_extreme is not None:
        frame["p_extreme"] = pd.Series(p_extreme).to_numpy(dtype="float64")
    rows = []
    for season, group in frame.groupby("season"):
        true_extreme = group["y_true"] != 0
        pred_extreme = group["y_pred"] != 0
        true_hot = group["y_true"] == 1
        true_cold = group["y_true"] == 2
        row = {
            "season": season,
            "n": int(group.shape[0]),
            "accuracy": float((group["y_true"] == group["y_pred"]).mean()),
            "extreme_rate_true": float(true_extreme.mean()),
            "false_positives": int((~true_extreme & pred_extreme).sum()),
            "false_negatives": int((true_extreme & ~pred_extreme).sum()),
            "missed_hot": int((true_hot & (group["y_pred"] != 1)).sum()),
            "missed_cold": int((true_cold & (group["y_pred"] != 2)).sum()),
            "hit_hot": int((true_hot & (group["y_pred"] == 1)).sum()),
            "hit_cold": int((true_cold & (group["y_pred"] == 2)).sum()),
        }
        if "p_extreme" in group.columns:
            row["mean_alert_probability"] = float(group["p_extreme"].mean())
        rows.append(row)
    order = ["winter", "spring", "summer", "autumn"]
    breakdown = pd.DataFrame(rows)
    if not breakdown.empty:
        breakdown["season"] = pd.Categorical(breakdown["season"], categories=order, ordered=True)
        breakdown = breakdown.sort_values("season").reset_index(drop=True)
    return breakdown


def write_metrics(metrics: dict, path: str) -> None:
    import json

    serialisable = {key: value for key, value in metrics.items() if key != "classification_report"}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2)
