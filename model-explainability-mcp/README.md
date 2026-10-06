# Model Explainability MCP

An [MCP](https://modelcontextprotocol.io) server that lets Claude explain machine learning predictions using [SHAP](https://shap.readthedocs.io). The server does the math; Claude explains the results in plain language.

> "Why was application #812 denied?"
> "Which features matter most overall?"
> "What happens if the debt-to-income ratio drops to 0.3?"

## Tools

| Tool | What it does |
|---|---|
| `load_model` | Register a joblib model + CSV dataset under an id |
| `list_models`, `model_summary` | Inspect loaded models and basic metrics |
| `global_feature_importance` | Rank features by mean absolute SHAP |
| `explain_prediction` | Per-feature SHAP contributions for one row |
| `what_if` | Re-predict a row after changing feature values |
| `compare_predictions` | Why two rows got different outputs |
| `partial_dependence` | How the average prediction moves as one feature changes |
| `find_misclassified` | Worst errors (needs a target column) |
| `plot_feature_importance`, `plot_explanation`, `plot_beeswarm` | PNG plots Claude can display |

Tree models (LightGBM, XGBoost, sklearn forests) use the fast `TreeExplainer`; other sklearn-style models fall back to `KernelExplainer`.

## Quick start

```bash
git clone <your-repo-url> && cd model-explainability-mcp
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/train_demo_model.py   # creates demo/model.joblib + demo/credit.csv
```

### Claude Desktop

Add to `claude_desktop_config.json` (use absolute paths):

```json
{
  "mcpServers": {
    "model-explainability": {
      "command": "/ABSOLUTE/PATH/model-explainability-mcp/.venv/bin/explain-mcp",
      "env": { "EXPLAIN_MCP_ALLOWED_DIR": "/ABSOLUTE/PATH/model-explainability-mcp" }
    }
  }
}
```

### Claude Code

```bash
claude mcp add model-explainability \
  -e EXPLAIN_MCP_ALLOWED_DIR=$(pwd) -- $(pwd)/.venv/bin/explain-mcp
```

## Try it

1. "Load `demo/model.joblib` with dataset `demo/credit.csv`, target `approved`, as `credit`."
2. "Which features matter most for the credit model?"
3. "Explain the prediction for row 12 and show the waterfall plot."
4. "What if row 12 had a debt_to_income of 0.2?"
5. "Show me the most confident mistakes the model makes."

## Safety

- **Loading is restricted** to `EXPLAIN_MCP_ALLOWED_DIR` (default: the working directory).
- **Pickle/joblib files can execute code when loaded.** Only load models you trust.
- Tools are read-only: nothing is written to disk besides in-memory state.
- SHAP explains how the *model* behaves, not real-world causes. Every explanation includes a caveat field.
- SHAP values for classifiers are in raw (log-odds) space; the final `prediction` is a probability.

## Development

```bash
pytest
```

## Roadmap

- [ ] Counterfactual search ("smallest change that flips the decision")
- [ ] LIME as a second opinion
- [ ] Fairness checks by sensitive group
- [ ] Drift detection on new data
- [ ] Keras/PyTorch support

## License

MIT
