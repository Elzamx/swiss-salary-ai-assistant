from __future__ import annotations
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["years_experience"]
CATEGORICAL = ["job_title", "education", "location"]
TEXT = "skills"
TARGET = "salary_chf"


def split_semicolon_skills(s):
    return str(s).split(";")


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
            ("skills", TfidfVectorizer(tokenizer=split_semicolon_skills, token_pattern=None, lowercase=True, min_df=2), TEXT),
        ]
    )


def candidate_models(random_state=42):
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(n_estimators=180, max_depth=16, min_samples_leaf=4, random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=180, max_depth=18, min_samples_leaf=3, random_state=random_state, n_jobs=-1),
    }


def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_select(df: pd.DataFrame, model_dir: str = "models", random_state: int = 42):
    X = df[NUMERIC + CATEGORICAL + [TEXT]]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    results = {}
    pipelines = {}
    for name, model in candidate_models(random_state).items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = evaluate(y_test, pred)
        pipelines[name] = pipe
    best_name = min(results, key=lambda n: results[n]["MAE"])
    import os, json
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipelines[best_name], f"{model_dir}/salary_model.joblib")
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    return best_name, results
