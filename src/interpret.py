import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

_SRC  = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd() / "src"
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

from engineering import compute_features
from train import (
    DROP_COLS,
    CATEGORICALS,
    TRAIN_END,
    VAL_END,
    DATA_PATH,
    MODEL_PATH,
    build_feature_matrix,
)

PLOTS_DIR = _ROOT / "plots"
SAMPLE_N  = 2000   # rows to use for SHAP (full 500K is slow)


# ---------------------------------------------------------------------------
# Load model + test data
# ---------------------------------------------------------------------------
def load_model_and_test() -> tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    print("Loading model...")
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print("Loading & engineering features...")
    raw = pd.read_csv(DATA_PATH)
    df  = compute_features(raw)

    # Reproduce the same train split to get column schema, then take test
    train_df = df[df["timestamp"] <= TRAIN_END]
    test_df  = df[df["timestamp"] > VAL_END]

    X_train, _ = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)

    # Align to training columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"  Test set: {len(X_test):,} rows, {X_test.shape[1]} features")
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# Plot 1: SHAP bar chart — mean absolute feature importance
# ---------------------------------------------------------------------------
def plot_shap_importance(model, X_test: pd.DataFrame) -> None:
    print(f"\nComputing SHAP values on {SAMPLE_N} samples...")
    X_sample = X_test.sample(SAMPLE_N, random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    PLOTS_DIR.mkdir(exist_ok=True)

    # Bar chart — overall feature importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_importance.png", dpi=150)
    plt.show()
    print(f"  Saved → {PLOTS_DIR / 'shap_importance.png'}")

    # Beeswarm — direction + magnitude per feature
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Beeswarm (feature value → impact on fraud probability)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_beeswarm.png", dpi=150)
    plt.show()
    print(f"  Saved → {PLOTS_DIR / 'shap_beeswarm.png'}")

    return shap_values, X_sample


# ---------------------------------------------------------------------------
# Plot 2: Precision-Recall curve
# ---------------------------------------------------------------------------
def plot_pr_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        y_test, y_prob, name=f"XGBoost (PR-AUC={pr_auc:.3f})", ax=ax
    )
    ax.axhline(y_test.mean(), color="red", linestyle="--", label="Baseline (random)")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curve.png", dpi=150)
    plt.show()
    print(f"  PR-AUC = {pr_auc:.4f}")
    print(f"  Saved → {PLOTS_DIR / 'pr_curve.png'}")


# ---------------------------------------------------------------------------
# Plot 3: Top-10 SHAP features printed as a table
# ---------------------------------------------------------------------------
def print_top_features(shap_values: np.ndarray, X_sample: pd.DataFrame, n: int = 10) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = (
        pd.Series(mean_abs, index=X_sample.columns)
        .sort_values(ascending=False)
        .head(n)
    )
    print(f"\nTop {n} features by mean |SHAP|:")
    print(top.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, X_test, y_test = load_model_and_test()

    shap_vals, X_sample = plot_shap_importance(model, X_test)
    plot_pr_curve(model, X_test, y_test)
    print_top_features(shap_vals, X_sample)
