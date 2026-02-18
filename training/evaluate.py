"""Model evaluation helpers including misclassification analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "artifacts") -> pd.DataFrame:
    """Evaluate model and export confusion matrix and false-negative examples."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()

    false_negatives = X_test[(y_test == 1) & (preds == 0)]
    print("False Negatives Count:", len(false_negatives))
    false_negatives.to_csv(output_path / "false_negatives.csv", index=False)

    return false_negatives
