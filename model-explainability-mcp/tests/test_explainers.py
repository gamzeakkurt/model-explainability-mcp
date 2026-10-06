import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from explain_mcp import explainers, registry
from explain_mcp.registry import ModelEntry, RegistryError


@pytest.fixture(scope="module")
def entry():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400), "noise": rng.normal(size=400)})
    y = ((2 * X.a - X.b) > 0).astype(int)
    model = LGBMClassifier(n_estimators=50, random_state=0, verbose=-1).fit(X, y)
    return ModelEntry("t", model, X, y, "classification")


def test_global_importance_ranks_signal_over_noise(entry):
    top = explainers.global_importance(entry, top_k=3)["top_features"]
    assert top[0]["feature"] == "a"
    assert top[-1]["feature"] == "noise"


def test_explain_row_is_consistent(entry):
    out = explainers.explain_row(entry, 0)
    assert 0 <= out["prediction"] <= 1
    assert {c["feature"] for c in out["contributions"]} == {"a", "b", "noise"}
    assert "caveat" in out


def test_what_if_moves_prediction(entry):
    row = int(entry.data.index[entry.data.a < -1][0])
    out = explainers.what_if(entry, row, {"a": 3.0})
    assert out["delta"] > 0


def test_what_if_rejects_unknown_feature(entry):
    with pytest.raises(RegistryError):
        explainers.what_if(entry, 0, {"nope": 1})


def test_row_bounds(entry):
    with pytest.raises(RegistryError):
        explainers.explain_row(entry, 10_000)


def test_partial_dependence_increases_with_a(entry):
    assert explainers.partial_dependence(entry, "a")["overall_change"] > 0


def test_path_outside_allowed_dir_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(registry.ALLOWED_DIR_ENV, str(tmp_path))
    with pytest.raises(RegistryError, match="outside the allowed"):
        registry.resolve_path("/etc/passwd")
