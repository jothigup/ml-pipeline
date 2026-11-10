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
1. **Load** Jothi's processed train/test data (`data/processed/`).
2. **Preprocess**: StandardScaler on Year and Spending_USD, OneHotEncoder on Country.
3. **Model**: Linear Regression.
4. **Evaluate** on the held-out test set (R² and Mean Absolute Error).
5. **Predict 2021**: retrain on all data (1970–2020), estimate each country's 2021 spending from its 2015–2020 trend, and predict life expectancy.

## Results (test set)
| Model | Test R² | Test MAE |
|---|---|---|
| Linear Regression | 0.989 | 0.29 years |

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
