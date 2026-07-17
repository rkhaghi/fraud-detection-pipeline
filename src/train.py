#%%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier

# Allow running from any working directory or interactively
_SRC = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd() / "src"
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
from engineering import compute_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH    = _ROOT / "Data"   / "synthetic_transactions.csv"
MODEL_PATH   = _ROOT / "models" / "model.json"
METRICS_PATH = _ROOT / "models" / "metrics.json"

# Time-based split boundaries  (dataset runs Jan–Jun 2024)
TRAIN_END = "2024-04-30"   # Jan–Apr → ~67% of data
VAL_END   = "2024-05-31"   # May     → ~17%
# Jun → test                         → ~17%

# Categorical columns to one-hot encode
CATEGORICALS = ["merchant_category", "merchant_country", "currency"]

# Columns to drop before training (identifiers, leakage, target)
DROP_COLS = [
    "transaction_id", "cardholder_id", "timestamp",
    "merchant_name", "merchant_city", "merchant_state",
    "fraud_type",   # ← target leakage: only exists for fraud rows
    "is_fraud",     # ← target
]


# ---------------------------------------------------------------------------
# Step 1: Load & engineer features
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("Loading raw data...")
    raw = pd.read_csv(DATA_PATH)
    print(f"  {len(raw):,} rows loaded")

    print("Engineering features...")
    df = compute_features(raw)
    print(f"  {len(df.columns)} total columns after feature engineering")
    return df


# ---------------------------------------------------------------------------
# Step 2: Time-based train / val / test split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    train = df[df["timestamp"] <= TRAIN_END]
    val   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
    test  = df[df["timestamp"] > VAL_END]

    print(f"\nSplit sizes:")
    print(f"  Train : {len(train):>7,} rows  |  fraud: {train['is_fraud'].sum():,}  ({train['is_fraud'].mean()*100:.2f}%)")
    print(f"  Val   : {len(val):>7,} rows  |  fraud: {val['is_fraud'].sum():,}  ({val['is_fraud'].mean()*100:.2f}%)")
    print(f"  Test  : {len(test):>7,} rows  |  fraud: {test['is_fraud'].sum():,}  ({test['is_fraud'].mean()*100:.2f}%)")
    return train, val, test


# ---------------------------------------------------------------------------
# Step 3: Build feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["is_fraud"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=[c for c in CATEGORICALS if c in X.columns])

    # Boolean columns → int
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X, y


# ---------------------------------------------------------------------------
# Step 4: Train
# ---------------------------------------------------------------------------
def train(X_train, y_train, X_val, y_val) -> XGBClassifier:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos)
    print(f"\nClass imbalance → scale_pos_weight = {spw}  (neg={neg:,}, pos={pos:,})")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# Step 5: Evaluate
# ---------------------------------------------------------------------------
def evaluate(model: XGBClassifier, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, y_prob)

    # Pick threshold that maximises F1
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    print(f"\n{'='*50}")
    print(f"{split_name} Results  (threshold={best_threshold:.3f})")
    print(f"{'='*50}")
    print(f"  PR-AUC    : {pr_auc:.4f}  (target ≥ 0.85)")
    print(f"  Precision : {tp/(tp+fp):.4f}")
    print(f"  Recall    : {tp/(tp+fn):.4f}")
    print(f"  F1        : {f1_scores[best_idx]:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(classification_report(y, y_pred, target_names=["legit", "fraud"]))

    return {
        "split": split_name,
        "pr_auc": round(pr_auc, 4),
        "precision": round(float(tp/(tp+fp)), 4),
        "recall": round(float(tp/(tp+fn)), 4),
        "f1": round(float(f1_scores[best_idx]), 4),
        "threshold": round(float(best_threshold), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load + engineer
    df = load_data()

    # Split
    train_df, val_df, test_df = split_data(df)

    # Feature matrices — fit encoding on train, apply to val/test
    X_train, y_train = build_feature_matrix(train_df)
    X_val,   y_val   = build_feature_matrix(val_df)
    X_test,  y_test  = build_feature_matrix(test_df)

    # Align columns (val/test may be missing some one-hot columns)
    X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")

    # Train
    model = train(X_train, y_train, X_val, y_val)

    # Evaluate
    val_metrics  = evaluate(model, X_val,  y_val,  "Validation")
    test_metrics = evaluate(model, X_test, y_test, "Test")

    # Save model + metrics
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"\nModel saved → {MODEL_PATH}")

    metrics = {"validation": val_metrics, "test": test_metrics}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved → {METRICS_PATH}")

# %%
