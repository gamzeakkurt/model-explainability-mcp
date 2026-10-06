"""MCP server exposing model explainability tools to Claude."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from . import explainers, plots, registry

mcp = FastMCP(
    "model-explainability",
    instructions=(
        "Explain machine learning model predictions. Call load_model first, then use the "
        "explain_* / global_* tools. Always mention the caveat that SHAP is not causal."
    ),
)


@mcp.tool()
def load_model(model_id: str, model_path: str, dataset_path: str, target: str | None = None) -> dict:
    """Register a trained model (joblib/pickle) and its dataset (CSV) under an id.

    Paths must be inside the allowed directory (env EXPLAIN_MCP_ALLOWED_DIR, default: cwd).
    `target` is the label column in the CSV; it is removed from the features.
    """
    e = registry.register(model_id, model_path, dataset_path, target)
    return {"model_id": e.model_id, "task": e.task, "model_type": type(e.model).__name__, "rows": len(e.data), "features": list(e.data.columns)}


@mcp.tool()
def list_models() -> list[dict]:
    """List models currently loaded."""
    return [{"model_id": e.model_id, "task": e.task, "model_type": type(e.model).__name__, "rows": len(e.data)} for e in registry.all_entries()]


@mcp.tool()
def model_summary(model_id: str) -> dict:
    """Features, task type, and basic performance of a loaded model."""
    e = registry.get(model_id)
    out: dict[str, Any] = {"model_type": type(e.model).__name__, "task": e.task, "rows": len(e.data), "features": list(e.data.columns)}
    if e.target is not None:
        pred = e.model.predict(e.data)
        if e.task == "classification":
            out["accuracy"] = round(float((pred == e.target.to_numpy()).mean()), 4)
            out["class_balance"] = {str(k): round(float(v), 4) for k, v in e.target.value_counts(normalize=True).items()}
        else:
            from sklearn.metrics import r2_score

            out["r2"] = round(float(r2_score(e.target, pred)), 4)
        out["note"] = "Metrics are computed on the loaded dataset, which may include training data."
    return out


@mcp.tool()
def global_feature_importance(model_id: str, top_k: int = 10) -> dict:
    """Rank features by mean absolute SHAP value across the dataset."""
    return explainers.global_importance(registry.get(model_id), top_k)


@mcp.tool()
def explain_prediction(model_id: str, row_id: int, top_k: int = 8) -> dict:
    """Explain one row's prediction: base value, per-feature SHAP contributions, final prediction."""
    return explainers.explain_row(registry.get(model_id), row_id, top_k)


@mcp.tool()
def what_if(model_id: str, row_id: int, changes: dict[str, Any]) -> dict:
    """Re-predict a row after editing feature values, e.g. changes={"income": 60000}."""
    return explainers.what_if(registry.get(model_id), row_id, changes)


@mcp.tool()
def compare_predictions(model_id: str, row_a: int, row_b: int) -> dict:
    """Show which features explain the difference between two rows' predictions."""
    return explainers.compare_rows(registry.get(model_id), row_a, row_b)


@mcp.tool()
def partial_dependence(model_id: str, feature: str, points: int = 10) -> dict:
    """Average prediction as one feature sweeps from low to high values."""
    return explainers.partial_dependence(registry.get(model_id), feature, points)


@mcp.tool()
def find_misclassified(model_id: str, n: int = 5) -> dict:
    """Return the worst prediction errors (requires a target column)."""
    return explainers.find_misclassified(registry.get(model_id), n)


@mcp.tool()
def plot_feature_importance(model_id: str, top_k: int = 10) -> Image:
    """Bar chart (PNG) of global feature importance."""
    return Image(data=plots.importance_bar(registry.get(model_id), top_k), format="png")


@mcp.tool()
def plot_explanation(model_id: str, row_id: int, top_k: int = 10) -> Image:
    """SHAP waterfall plot (PNG) for one row."""
    return Image(data=plots.waterfall(registry.get(model_id), row_id, top_k), format="png")


@mcp.tool()
def plot_beeswarm(model_id: str, top_k: int = 10) -> Image:
    """SHAP beeswarm plot (PNG) showing feature effects across many rows."""
    return Image(data=plots.beeswarm(registry.get(model_id), top_k), format="png")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
