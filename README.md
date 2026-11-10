# ml-pipeline

A simple end-to-end ML pipeline for AIPI 510: predict a country's **life expectancy** from the year, the country and its health spending per person.

## Data
`data/clean/healthexp.csv`: 274 rows, 6 countries (Canada, France, Germany, Great Britain, Japan, USA), 1970–2020.

| Column | Use |
|---|---|
| Year | feature (numeric) |
| Country | feature (one-hot encoded) |
| Spending_USD | feature (numeric) |
| Life_Expectancy | **target** |

## Pipeline
1. **Load** `data/clean/healthexp.csv` (cleaned by Jothi) and split it 80% train / 20% test (`random_state=42`).
2. **Preprocess**: StandardScaler on Year and Spending_USD, OneHotEncoder on Country.
3. **Model**: Linear Regression.
4. **Evaluate** on the held-out test set (R² and Mean Absolute Error).
5. **Predict 2020**: retrain on 1970–2019 only, predict each country's 2020 life expectancy, and compare with the real 2020 values (we don't have real 2021 data to check against).

## Results (test set)
**Test accuracy = 100% within 1 year** (83.6% within 0.5 year) · **Test R² = 0.989** · **Test MAE = 0.29 years**

| Metric (55 test rows) | Value |
|---|---|
| Test accuracy: predictions within 1 year of true value | **100%** |
| Test accuracy: predictions within 0.5 year of true value | 83.6% |
| Test R² (variance explained) | 0.989 |
| Test MAE (average error) | 0.29 years |

Since this is a regression problem (predicting a number), "accuracy" is the % of test predictions within 1 year of the true life expectancy.

## 2020 predictions vs. actual
Model trained on 1970–2019 only, then asked to predict 2020.

| Country | Actual 2020 | Predicted 2020 | Error |
|---|---|---|---|
| Canada | 81.7 | 83.1 | +1.4 |
| France | 82.3 | 83.4 | +1.1 |
| Germany | 81.1 | 81.4 | +0.3 |
| Great Britain | 80.4 | 82.0 | +1.6 |
| Japan | 84.7 | 85.1 | +0.4 |
| USA | 77.0 | 78.7 | +1.7 |

**2020 MAE = 1.08 years** · **2020 accuracy (within 1 year) = 33.3%** (2 of 6 countries) · 2020 R² = 0.72

**Why 2020 is worse than the test set:** the model **over-predicted every country**, because COVID-19 lowered life expectancy in 2020 and the model only knows the long-term upward trend. The two countries it got closest (Japan, Germany) had smaller COVID drops that year, and Japan's life expectancy actually rose. This shows the model works well under normal conditions but can't foresee sudden shocks.

## How to run
```bash
pip install -r requirements.txt
python train.py
```
