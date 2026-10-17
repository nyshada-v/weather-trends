"""Baseline model construction, time-aware splitting and prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .labeling import CLASS_NAMES, CLASS_ORDER

MODEL_FACTORIES = {
    "logistic_regression": lambda params, seed: Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=seed, **params))]
    ),
    "random_forest": lambda params, seed: Pipeline(
        [("clf", RandomForestClassifier(random_state=seed, **params))]
    ),
    "gradient_boosting": lambda params, seed: Pipeline(
        [("clf", GradientBoostingClassifier(random_state=seed, **params))]
    ),
}


def build_models(config) -> dict:
    seed = int(config["project"]["random_seed"])
    models = {}
    for name in config["model"]["models"]:
        if name not in MODEL_FACTORIES:
            raise KeyError(f"unknown model {name!r}")
        params = dict(config["model"].get(name, {}) or {})
        models[name] = MODEL_FACTORIES[name](params, seed)
    return models


def final_estimator(model):
    return model.named_steps["clf"] if hasattr(model, "named_steps") else model


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float):
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    ordered = df.sort_index()
    n = ordered.shape[0]
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train : n_train + n_val]
    test = ordered.iloc[n_train + n_val :]
    return train, val, test


def fit_model(model, train: pd.DataFrame, feature_columns: list[str], target_col: str = "target"):
    est = final_estimator(model)
    available = [column for column in feature_columns if column in train.columns]
    X = train[available].astype("float64")
    y = train[target_col].astype(int)
    model.fit(X, y)
    return model


def predict_model(model, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    est = final_estimator(model)
    available = [column for column in feature_columns if column in frame.columns]
    X = frame[available].astype("float64")
    proba = model.predict_proba(X)
    classes = np.asarray(est.classes_)
    result = pd.DataFrame(index=frame.index)
    for position, code in enumerate(classes):
        result[f"p_{CLASS_NAMES[int(code)]}"] = proba[:, position]
    for name in CLASS_ORDER:
        column = f"p_{name}"
        if column not in result.columns:
            result[column] = 0.0
    result = result[[f"p_{name}" for name in CLASS_ORDER]]
    argmax = np.argmax(proba, axis=1)
    result["pred"] = classes[argmax]
    result["pred_name"] = result["pred"].map(CLASS_NAMES)
    result["p_extreme"] = result["p_extreme_hot"] + result["p_extreme_cold"]
    return result


def feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    est = final_estimator(model)
    available = list(feature_columns)
    if hasattr(est, "feature_importances_"):
        return (
            pd.DataFrame({"feature": available, "importance": est.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    if hasattr(est, "coef_"):
        coef = np.atleast_2d(est.coef_)
        return (
            pd.DataFrame({"feature": available, "importance": np.abs(coef).mean(axis=0)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame({"feature": available, "importance": np.nan})


def coefficient_table(model, feature_columns: list[str]) -> pd.DataFrame:
    est = final_estimator(model)
    if not hasattr(est, "coef_"):
        return pd.DataFrame()
    coef = np.atleast_2d(est.coef_)
    rows = []
    for position, code in enumerate(np.asarray(est.classes_)):
        for column_index, feature in enumerate(feature_columns):
            rows.append(
                {
                    "class": CLASS_NAMES[int(code)],
                    "feature": feature,
                    "coefficient": float(coef[position, column_index]),
                }
            )
    return pd.DataFrame(rows)
