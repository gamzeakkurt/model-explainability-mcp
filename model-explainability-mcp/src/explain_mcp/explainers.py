"""SHAP computations. All functions return plain JSON-friendly dicts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from .registry import MAX_BACKGROUND_ROWS, ModelEntry, RegistryError

CAVEAT = (
    "SHAP values describe how the model uses its inputs, not real-world causation. "
    "Correlated features can share credit."
)


def _positive_class_values(values: np.ndarray, expected: Any) -> tuple[np.ndarray, float]:
    """Normalise SHAP output to (n_samples, n_features) for the positive class."""
    values = np.asarray(values)
    exp = np.asarray(expected).ravel()
    if values.ndim == 3:  # (n, features, classes) in newer shap
        return values[:, :, -1], float(exp[-1])
    if isinstance(expected, (list, np.ndarray)) and len(exp) > 1:  # list-of-arrays style
        return values, float(exp[-1])
    return values, float(exp[0])


def get_explainer(entry: ModelEntry) -> Any:
    if entry.explainer is None:
        try:
            entry.explainer = shap.TreeExplainer(entry.model)
        except Exception:
            background = shap.sample(entry.data, min(MAX_BACKGROUND_ROWS, len(entry.data)))
            fn = entry.model.predict_proba if entry.task == "classification" else entry.model.predict
            entry.explainer = shap.KernelExplainer(fn, background)
    return entry.explainer


def shap_for(entry: ModelEntry, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = get_explainer(entry)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = np.stack(values, axis=-1)
    return _positive_class_values(values, explainer.expected_value)


def predict_row(entry: ModelEntry, X: pd.DataFrame) -> float:
    if entry.task == "classification":
        return float(entry.model.predict_proba(X)[0, -1])
    return float(entry.model.predict(X)[0])


def global_importance(entry: ModelEntry, top_k: int = 10, max_rows: int = 500) -> dict:
    X = entry.data.head(max_rows)
    values, _ = shap_for(entry, X)
    entry.shap_cache = (X, values)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    total = mean_abs.sum() or 1.0
    ranked = [
        {
            "feature": X.columns[i],
            "mean_abs_shap": round(float(mean_abs[i]), 5),
            "share_of_total": round(float(mean_abs[i] / total), 4),
        }
        for i in order[:top_k]
    ]
    return {"rows_used": len(X), "top_features": ranked, "caveat": CAVEAT}


def explain_row(entry: ModelEntry, row_id: int, top_k: int = 8) -> dict:
    if not 0 <= row_id < len(entry.data):
        raise RegistryError(f"row_id must be between 0 and {len(entry.data) - 1}")
    X = entry.data.iloc[[row_id]]
    return explain_frame(entry, X, top_k, actual=None if entry.target is None else entry.target.iloc[row_id])


def explain_frame(entry: ModelEntry, X: pd.DataFrame, top_k: int = 8, actual: Any = None) -> dict:
    values, base = shap_for(entry, X)
    row_values = values[0]
    order = np.argsort(np.abs(row_values))[::-1]
    contribs = [
        {
            "feature": X.columns[i],
            "value": _py(X.iloc[0, i]),
            "shap": round(float(row_values[i]), 5),
            "direction": "raises" if row_values[i] > 0 else "lowers",
        }
        for i in order[:top_k]
    ]
    rest = float(row_values[order[top_k:]].sum()) if len(order) > top_k else 0.0
    out = {
        "prediction": round(predict_row(entry, X), 5),
        "base_value": round(base, 5),
        "contributions": contribs,
        "other_features_net_shap": round(rest, 5),
        "output_scale_note": (
            "SHAP values are in the model's raw output space (log-odds for many "
            "classifiers); 'prediction' is the final probability/value."
        ),
        "caveat": CAVEAT,
    }
    if actual is not None:
        out["actual"] = _py(actual)
    return out


def what_if(entry: ModelEntry, row_id: int, changes: dict[str, Any]) -> dict:
    if not 0 <= row_id < len(entry.data):
        raise RegistryError(f"row_id must be between 0 and {len(entry.data) - 1}")
    unknown = [c for c in changes if c not in entry.data.columns]
    if unknown:
        raise RegistryError(f"Unknown features: {unknown}. Available: {list(entry.data.columns)}")
    before = entry.data.iloc[[row_id]].copy()
    after = before.copy()
    for col, val in changes.items():
        after[col] = val
    p_before, p_after = predict_row(entry, before), predict_row(entry, after)
    return {
        "changes": {c: {"from": _py(before[c].iloc[0]), "to": _py(after[c].iloc[0])} for c in changes},
        "prediction_before": round(p_before, 5),
        "prediction_after": round(p_after, 5),
        "delta": round(p_after - p_before, 5),
    }


def compare_rows(entry: ModelEntry, row_a: int, row_b: int, top_k: int = 6) -> dict:
    for r in (row_a, row_b):
        if not 0 <= r < len(entry.data):
            raise RegistryError(f"row ids must be between 0 and {len(entry.data) - 1}")
    X = entry.data.iloc[[row_a, row_b]]
    values, _ = shap_for(entry, X)
    diff = values[0] - values[1]
    order = np.argsort(np.abs(diff))[::-1][:top_k]
    return {
        "prediction_a": round(predict_row(entry, X.iloc[[0]]), 5),
        "prediction_b": round(predict_row(entry, X.iloc[[1]]), 5),
        "biggest_differences": [
            {
                "feature": X.columns[i],
                "value_a": _py(X.iloc[0, i]),
                "value_b": _py(X.iloc[1, i]),
                "shap_a_minus_b": round(float(diff[i]), 5),
            }
            for i in order
        ],
        "caveat": CAVEAT,
    }


def find_misclassified(entry: ModelEntry, n: int = 5, max_rows: int = 2000) -> dict:
    if entry.target is None:
        raise RegistryError("This model was loaded without a target column; cannot find errors.")
    X = entry.data.head(max_rows)
    y = entry.target.head(max_rows)
    if entry.task == "classification":
        proba = entry.model.predict_proba(X)[:, -1]
        pred = entry.model.predict(X)
        wrong = np.where(pred != y.to_numpy())[0]
        conf = np.abs(proba[wrong] - 0.5)  # most confident wrong answers first
        picked = wrong[np.argsort(conf)[::-1][:n]]
        rows = [
            {"row_id": int(i), "predicted": _py(pred[i]), "actual": _py(y.iloc[i]), "probability": round(float(proba[i]), 4)}
            for i in picked
        ]
        return {"errors_found": int(len(wrong)), "rows_checked": len(X), "worst": rows}
    err = np.abs(entry.model.predict(X) - y.to_numpy())
    picked = np.argsort(err)[::-1][:n]
    rows = [
        {"row_id": int(i), "predicted": round(float(entry.model.predict(X.iloc[[i]])[0]), 4), "actual": _py(y.iloc[i]), "abs_error": round(float(err[i]), 4)}
        for i in picked
    ]
    return {"rows_checked": len(X), "worst": rows}


def partial_dependence(entry: ModelEntry, feature: str, points: int = 10) -> dict:
    if feature not in entry.data.columns:
        raise RegistryError(f"Unknown feature '{feature}'. Available: {list(entry.data.columns)}")
    X = entry.data.head(500).copy()
    grid = np.unique(np.quantile(X[feature], np.linspace(0.05, 0.95, points)))
    curve = []
    for g in grid:
        Xg = X.copy()
        Xg[feature] = g
        pred = entry.model.predict_proba(Xg)[:, -1] if entry.task == "classification" else entry.model.predict(Xg)
        curve.append({"value": _py(g), "avg_prediction": round(float(np.mean(pred)), 5)})
    trend = curve[-1]["avg_prediction"] - curve[0]["avg_prediction"]
    return {"feature": feature, "curve": curve, "overall_change": round(trend, 5), "caveat": CAVEAT}


def _py(v: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON."""
    if isinstance(v, np.generic):
        v = v.item()
    return round(v, 5) if isinstance(v, float) else v
