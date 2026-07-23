"""
SageMaker Training Script — XGBoost (Script Mode)

SageMaker mounts:
  /opt/ml/input/data/train/train.csv  ← training data
  /opt/ml/input/data/val/val.csv      ← validation data
  /opt/ml/model/                      ← model output (model.json + metrics.json)

Hyperparameters are passed via --arg flags by SageMaker.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

# SageMaker paths
TRAIN_DIR  = "/opt/ml/input/data/train"
VAL_DIR    = "/opt/ml/input/data/val"
MODEL_DIR  = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

CATEGORICALS = ["merchant_category", "merchant_country", "currency"]
DROP_COLS = [
    "transaction_id", "cardholder_id", "timestamp",
    "merchant_name", "merchant_city", "merchant_state",
    "fraud_type", "is_fraud",
]


def build_feature_matrix(df: pd.DataFrame):
    y = df["is_fraud"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = pd.get_dummies(X, columns=[c for c in CATEGORICALS if c in X.columns])
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    return X, y


def main(args):
    # ── Load data ─────────────────────────────────────────────────────────
    train_csv = os.path.join(TRAIN_DIR, "train.csv")
    val_csv   = os.path.join(VAL_DIR,   "val.csv")
    train_df  = pd.read_csv(train_csv)
    val_df    = pd.read_csv(val_csv)
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    X_train, y_train = build_feature_matrix(train_df)
    X_val,   y_val   = build_feature_matrix(val_df)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    # ── Class imbalance ───────────────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos)
    print(f"scale_pos_weight = {spw}")

    # ── Train ─────────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"Best iteration: {model.best_iteration}")

    # ── Evaluate on val ───────────────────────────────────────────────────
    y_prob = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1.argmax()
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using fixed threshold: {threshold}")
    else:
        threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        print(f"Using best F1 threshold: {threshold:.4f}")

    print(f"Val PR-AUC: {pr_auc:.4f}")

    # ── Save model ────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.json")
    model.save_model(model_path)
    print(f"Model saved → {model_path}")

    # ── Save metrics (picked up by evaluation step) ───────────────────────
    metrics = {
        "validation": {
            "pr_auc": round(pr_auc, 4),
            "threshold": round(threshold, 4),
            "best_iteration": int(model.best_iteration),
        }
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # ── Save feature column order (needed at inference time) ──────────────
    cols_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(cols_path, "w") as f:
        json.dump(list(X_train.columns), f)
    print(f"Feature columns saved → {cols_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",    type=int,   default=500)
    parser.add_argument("--max-depth",       type=int,   default=6)
    parser.add_argument("--learning-rate",   type=float, default=0.05)
    parser.add_argument("--subsample",       type=float, default=0.8)
    parser.add_argument("--colsample-bytree",type=float, default=0.8)
    parser.add_argument("--threshold",       type=float, default=None,
                        help="Fixed decision threshold (0-1). If not set, best F1 threshold is used.")
    args = parser.parse_args()
    main(args)
