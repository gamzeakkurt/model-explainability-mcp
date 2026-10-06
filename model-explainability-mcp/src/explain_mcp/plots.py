"""Plot rendering. Returns PNG bytes so the server can send them as MCP images."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

from .explainers import global_importance, shap_for  # noqa: E402
from .registry import ModelEntry, RegistryError  # noqa: E402


def _png() -> bytes:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close("all")
    return buf.getvalue()


def importance_bar(entry: ModelEntry, top_k: int = 10) -> bytes:
    data = global_importance(entry, top_k)["top_features"][::-1]
    plt.figure(figsize=(7, 0.45 * len(data) + 1))
    plt.barh([d["feature"] for d in data], [d["mean_abs_shap"] for d in data], color="#2a6fdb")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global feature importance")
    return _png()


def waterfall(entry: ModelEntry, row_id: int, top_k: int = 10) -> bytes:
    if not 0 <= row_id < len(entry.data):
        raise RegistryError(f"row_id must be between 0 and {len(entry.data) - 1}")
    X = entry.data.iloc[[row_id]]
    values, base = shap_for(entry, X)
    exp = shap.Explanation(values=values[0], base_values=base, data=X.iloc[0].to_numpy(), feature_names=list(X.columns))
    shap.plots.waterfall(exp, max_display=top_k, show=False)
    return _png()


def beeswarm(entry: ModelEntry, top_k: int = 10, max_rows: int = 300) -> bytes:
    X = entry.data.head(max_rows)
    values, base = shap_for(entry, X)
    exp = shap.Explanation(values=values, base_values=np.full(len(X), base), data=X.to_numpy(), feature_names=list(X.columns))
    shap.plots.beeswarm(exp, max_display=top_k, show=False)
    return _png()
