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
5. **Predict 2021**: retrain on all data (1970–2020), estimate each country's 2021 spending from its 2015–2020 trend, and predict life expectancy.

## Results (test set)
**Test accuracy = 100% within 1 year** (83.6% within 0.5 year) · **Test R² = 0.989** · **Test MAE = 0.29 years**

| Metric (55 test rows) | Value |
|---|---|
| Test accuracy: predictions within 1 year of true value | **100%** |
| Test accuracy: predictions within 0.5 year of true value | 83.6% |
| Test R² (variance explained) | 0.989 |
| Test MAE (average error) | 0.29 years |

Since this is a regression problem (predicting a number), "accuracy" is the % of test predictions within 1 year of the true life expectancy.

## 2021 predictions
| Country | Est. 2021 spending (USD) | Predicted life expectancy |
|---|---|---|
| Canada | 5,849 | 83.3 |
| France | 5,538 | 83.5 |
| Germany | 7,168 | 81.4 |
| Great Britain | 4,984 | 82.2 |
| Japan | 4,693 | 85.3 |
| USA | 12,015 | 78.6 |

**Limitation:** the model follows long-term trends. It can't see shocks like COVID-19, which lowered life expectancy in 2020–2021, so these 2021 predictions are probably too high for some countries (especially the USA).

## How to run
```bash
pip install -r requirements.txt
python train.py
```
