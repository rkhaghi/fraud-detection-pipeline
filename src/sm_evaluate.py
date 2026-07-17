"""
SageMaker Evaluation Script

SageMaker mounts:
  /opt/ml/processing/input/model/   ← model.json + feature_columns.json
  /opt/ml/processing/input/test/    ← test.csv
  /opt/ml/processing/output/        ← evaluation.json (read by ConditionStep)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

MODEL_DIR = "/opt/ml/processing/input/model"
TEST_DIR  = "/opt/ml/processing/input/test"
OUTPUT_DIR = "/opt/ml/processing/output"

CATEGORICALS = ["merchant_category", "merchant_country", "currency"]
DROP_COLS = [
    "transaction_id", "cardholder_id", "timestamp",
    "merchant_name", "merchant_city", "merchant_state",
    "fraud_type", "is_fraud",
]


def main():
    # ── Load model ────────────────────────────────────────────────────────
    model = XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "model.json"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    # ── Load test data ────────────────────────────────────────────────────
    test_df = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    print(f"Test set: {len(test_df):,} rows")

    y_test = test_df["is_fraud"]
    X_test = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])
    X_test = pd.get_dummies(X_test, columns=[c for c in CATEGORICALS if c in X_test.columns])
    bool_cols = X_test.select_dtypes(include="bool").columns
    X_test[bool_cols] = X_test[bool_cols].astype(int)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1.argmax()
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"Test PR-AUC : {pr_auc:.4f}")
    print(f"Precision   : {tp/(tp+fp):.4f}")
    print(f"Recall      : {tp/(tp+fn):.4f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ── Write evaluation.json ─────────────────────────────────────────────
    # SageMaker ConditionStep reads this file to decide whether to register
    evaluation = {
        "binary_classification_metrics": {
            "average_precision": {
                "value": round(pr_auc, 4),
                "standard_deviation": "NaN",
            }
        },
        "test": {
            "pr_auc": round(pr_auc, 4),
            "precision": round(float(tp / (tp + fp)), 4),
            "recall": round(float(tp / (tp + fn)), 4),
            "f1": round(float(f1[best_idx]), 4),
            "threshold": round(threshold, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation saved → {out_path}")


if __name__ == "__main__":
    main()
