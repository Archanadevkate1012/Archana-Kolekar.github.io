"""Phase 3 training pipeline with outlier handling + ensemble refinement."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from training.ensemble import build_ensemble
from training.evaluate import evaluate_model
from training.outlier_handling import remove_outliers_iqr


DATA_PATH = Path("data/creditcard.csv")
ARTIFACT_DIR = Path("artifacts")
TARGET_COLUMN = "Class"


def train_refined_model() -> None:
    """Train, evaluate, and persist the refined fraud detection model."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected training dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    df = remove_outliers_iqr(df, feature_columns)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_ensemble(random_state=42)
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, output_dir=str(ARTIFACT_DIR))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "refined_model.pkl")
    joblib.dump(feature_columns, ARTIFACT_DIR / "feature_names.pkl")


if __name__ == "__main__":
    train_refined_model()
