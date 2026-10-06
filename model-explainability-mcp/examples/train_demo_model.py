"""Train a demo credit-approval model and save it with its dataset.

Run from the project root:  python examples/train_demo_model.py
Creates demo/model.joblib and demo/credit.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

rng = np.random.default_rng(42)
n = 2000

df = pd.DataFrame(
    {
        "income": rng.normal(55000, 18000, n).clip(15000),
        "debt_to_income": rng.beta(2, 5, n).round(3),
        "credit_history_years": rng.integers(0, 30, n),
        "num_late_payments": rng.poisson(1.0, n),
        "loan_amount": rng.normal(15000, 6000, n).clip(1000),
    }
)

score = (
    (df.income - 55000) / 18000 * 0.8
    - df.debt_to_income * 6
    + df.credit_history_years * 0.08
    - df.num_late_payments * 0.5
    - (df.loan_amount - 15000) / 6000 * 0.3
    + 1.2
    + rng.normal(0, 0.5, n)
)
df["approved"] = (score > 0).astype(int)

out = Path(__file__).resolve().parent.parent / "demo"
out.mkdir(exist_ok=True)

X, y = df.drop(columns="approved"), df["approved"]
model = LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=0, verbose=-1).fit(X, y)

joblib.dump(model, out / "model.joblib")
df.round(3).to_csv(out / "credit.csv", index=False)
print(f"Saved demo/model.joblib and demo/credit.csv  (train accuracy {model.score(X, y):.3f})")
