"""
Model pipeline (my part): train and evaluate models on Jothi's preprocessed data.

Expects Jothi's preprocessing step to save:
    data/processed/train.csv
    data/processed/test.csv
with columns: Year, Country, Spending_USD, Life_Expectancy
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

TARGET = "Life_Expectancy"
NUM_COLS = ["Year", "Spending_USD"]
CAT_COLS = ["Country"]

# 1. Load data -------------------------------------------------------------
if os.path.exists("data/processed/train.csv"):
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    print("Loaded Jothi's processed train/test files")
else:
    # Fallback so the script still runs before preprocessing is pushed
    df = pd.read_csv("data/clean/healthexp.csv")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    print("Processed files not found, split data/clean/healthexp.csv 80/20")

X_train, y_train = train[NUM_COLS + CAT_COLS], train[TARGET]
X_test, y_test = test[NUM_COLS + CAT_COLS], test[TARGET]
print(f"Train rows: {len(train)}, Test rows: {len(test)}")

# 2. Build pipeline: encode features -> model ------------------------------
def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("prep", prep), ("model", model)])

# 3. Train + evaluate Linear Regression on the test set -------------------
pipe = make_pipeline(LinearRegression())
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(f"\nLinear Regression  test R2 = {r2_score(y_test, pred):.3f}   "
      f"test MAE = {mean_absolute_error(y_test, pred):.2f} years")
errors = np.abs(pred - y_test)
print(f"Test accuracy (within 1 year):   {(errors <= 1).mean():.1%}")
print(f"Test accuracy (within 0.5 year): {(errors <= 0.5).mean():.1%}")

# 4. Predict 2020 and compare with the real 2020 values -------------------
# Train on 1970-2019 only, so the model never sees 2020.
full = pd.concat([train, test])
past = full[full.Year <= 2019]
y2020 = full[full.Year == 2020].copy()

pipe = make_pipeline(LinearRegression())
pipe.fit(past[NUM_COLS + CAT_COLS], past[TARGET])
y2020["Predicted_Life_Expectancy"] = pipe.predict(y2020[NUM_COLS + CAT_COLS])
y2020["Error"] = y2020["Predicted_Life_Expectancy"] - y2020[TARGET]

print("\n2020 predictions vs. actual (model trained on 1970-2019):")
print(y2020[["Country", TARGET, "Predicted_Life_Expectancy", "Error"]]
      .round(2).sort_values("Country").to_string(index=False))
err2020 = y2020["Error"].abs()
print(f"\n2020 MAE = {err2020.mean():.2f} years   "
      f"2020 R2 = {r2_score(y2020[TARGET], y2020['Predicted_Life_Expectancy']):.3f}")
print(f"2020 accuracy (within 1 year): {(err2020 <= 1).mean():.1%}")

y2020.sort_values("Country").to_csv("predictions_2020.csv", index=False)
print("\nSaved predictions_2020.csv")
