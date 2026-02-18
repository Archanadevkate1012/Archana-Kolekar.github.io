# Fraud Detection – Phase 3

This repo now contains a clean, modular Phase 3 pipeline for:

- Misclassification analysis
- Outlier handling
- Ensemble refinement
- Streamlit dashboard deployment
- Flask API alternative

## Project structure

- `training/outlier_handling.py` – IQR-based outlier filtering
- `training/ensemble.py` – soft-voting ensemble builder
- `training/evaluate.py` – ROC-AUC, report, confusion matrix, false-negative export
- `training/refine_train.py` – end-to-end refined training pipeline
- `streamlit_app.py` – interactive fraud dashboard
- `deployment/flask_app.py` – REST inference endpoint

## Run training

```bash
python -m training.refine_train
```

Expected input dataset:
- `data/creditcard.csv` with target column `Class`

Generated artifacts:
- `artifacts/refined_model.pkl`
- `artifacts/feature_names.pkl`
- `artifacts/confusion_matrix.png`
- `artifacts/false_negatives.csv`

## Run Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

## Run Flask API

```bash
python deployment/flask_app.py
```

Then POST JSON to `/predict` with all expected feature keys.
