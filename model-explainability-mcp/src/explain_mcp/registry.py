"""In-memory registry of loaded models and their datasets.

MCP tool calls are independent, so loaded models are kept here under an id.
Loading is restricted to an allowed directory because unpickling a model file
can execute arbitrary code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ALLOWED_DIR_ENV = "EXPLAIN_MCP_ALLOWED_DIR"
MAX_BACKGROUND_ROWS = 200


class RegistryError(ValueError):
    """Raised for user-facing problems (bad path, unknown model id, ...)."""


@dataclass
class ModelEntry:
    model_id: str
    model: Any
    data: pd.DataFrame  # feature columns only
    target: pd.Series | None
    task: str  # "classification" or "regression"
    explainer: Any = field(default=None, repr=False)
    shap_cache: Any = field(default=None, repr=False)


_models: dict[str, ModelEntry] = {}


def allowed_dir() -> Path:
    return Path(os.environ.get(ALLOWED_DIR_ENV, Path.cwd())).resolve()


def resolve_path(path: str) -> Path:
    """Resolve a user-supplied path and make sure it is inside the allowed dir."""
    base = allowed_dir()
    p = Path(path).expanduser()
    p = (base / p).resolve() if not p.is_absolute() else p.resolve()
    if base != p and base not in p.parents:
        raise RegistryError(
            f"Path {p} is outside the allowed directory {base}. "
            f"Set {ALLOWED_DIR_ENV} to change it."
        )
    if not p.is_file():
        raise RegistryError(f"File not found: {p}")
    return p


def register(model_id: str, model_path: str, dataset_path: str, target: str | None) -> ModelEntry:
    model = joblib.load(resolve_path(model_path))
    df = pd.read_csv(resolve_path(dataset_path))

    y = None
    if target:
        if target not in df.columns:
            raise RegistryError(f"Target column '{target}' not in dataset columns: {list(df.columns)}")
        y = df[target]
        df = df.drop(columns=[target])

    task = "classification" if hasattr(model, "predict_proba") else "regression"
    entry = ModelEntry(model_id=model_id, model=model, data=df, target=y, task=task)
    _models[model_id] = entry
    return entry


def get(model_id: str) -> ModelEntry:
    try:
        return _models[model_id]
    except KeyError:
        known = ", ".join(_models) or "none loaded"
        raise RegistryError(f"Unknown model_id '{model_id}'. Loaded models: {known}") from None


def all_entries() -> list[ModelEntry]:
    return list(_models.values())


def clear() -> None:
    _models.clear()
