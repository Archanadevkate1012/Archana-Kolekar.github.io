"""Flask deployment alternative for single-record inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "refined_model.pkl"
FEATURE_PATH = ARTIFACT_DIR / "feature_names.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({"error": f"Missing required features: {missing}"}), 400

    df = pd.DataFrame([data], columns=features)
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return jsonify({
        "fraud_prediction": prediction,
        "fraud_probability": probability,
    })


if __name__ == "__main__":
    app.run(debug=True)
