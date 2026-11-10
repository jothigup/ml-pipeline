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

# 4. Predict 2021 -----------------------------------------------------------
# Retrain on ALL data (train + test) so the model uses every year up to 2020.
full = pd.concat([train, test])
pipe = make_pipeline(LinearRegression())
pipe.fit(full[NUM_COLS + CAT_COLS], full[TARGET])

# 2021 spending isn't in the data, so estimate it per country by
# fitting a straight-line trend to that country's 2015-2020 spending.
rows = []
for country, g in full[full.Year >= 2015].groupby("Country"):
    slope, intercept = np.polyfit(g.Year, g.Spending_USD, 1)
    rows.append({"Year": 2021, "Country": country,
                 "Spending_USD": slope * 2021 + intercept})
future = pd.DataFrame(rows)
future["Predicted_Life_Expectancy_2021"] = pipe.predict(future[NUM_COLS + CAT_COLS])

print("\nPredicted life expectancy for 2021:")
print(future.round({"Spending_USD": 0, "Predicted_Life_Expectancy_2021": 1})
      .to_string(index=False))
future.to_csv("predictions_2021.csv", index=False)
print("\nSaved predictions_2021.csv")
