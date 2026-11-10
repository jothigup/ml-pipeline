import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_split_data(filepath: str, target_col: str, test_year: int = 2020):
    """Train on years before test_year and use test_year as the test set."""
    df = pd.read_csv(filepath)
    train_df = df[df["Year"] < test_year]
    test_df = df[df["Year"] == test_year]
    if train_df.empty or test_df.empty:
        raise ValueError(f"Dataset must contain years before and equal to {test_year}.")

    feature_columns = [column for column in df.columns if column != target_col]
    return (
        train_df[feature_columns],
        test_df[feature_columns],
        train_df[target_col],
        test_df[target_col],
    )

def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Constructs the preprocessing steps and pairs them with a regressor."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    return pipeline

def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fits the pipeline on the training dataset."""
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tolerance_years: float = 1.0,
) -> tuple[float, float, float]:
    """Prints regression metrics, including percent of predictions within a year tolerance."""
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy_within_tolerance = (
        abs(y_test.to_numpy() - y_pred) <= tolerance_years
    ).mean() * 100
    print(f"Mean Absolute Error: {mae:.4f}")
    r2 = r2_score(y_test, y_pred)
    print(f"R² score: {r2:.4f}")
    print(
        f"Within {tolerance_years:.1f} year(s) of actual life expectancy: "
        f"{accuracy_within_tolerance:.1f}%"
    )
    return mae, r2, accuracy_within_tolerance

if __name__ == "__main__":
    # 1. Configuration & Setup
    DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "clean" / "healthexp.csv"
    TARGET_COLUMN = "Life_Expectancy"
    NUM_FEATURES = ["Year", "Spending_USD"]
    CAT_FEATURES = ["Country"]

    # 2. Execution Flow
    X_train, X_test, y_train, y_test = load_and_split_data(
        DATA_PATH, TARGET_COLUMN, test_year=2020
    )
    
    pipeline = build_pipeline(NUM_FEATURES, CAT_FEATURES)
    
    trained_pipeline = train_pipeline(pipeline, X_train, y_train)
    
    evaluate_pipeline(trained_pipeline, X_test, y_test)