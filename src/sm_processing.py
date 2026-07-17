"""
SageMaker Processing Script — Feature Engineering + Train/Val/Test Split

SageMaker mounts:
  /opt/ml/processing/input/data/   ← raw synthetic_transactions.csv
  /opt/ml/processing/output/train/ ← train.csv output
  /opt/ml/processing/output/val/   ← val.csv output
  /opt/ml/processing/output/test/  ← test.csv output
"""
import os
import subprocess
import sys

# Install dependencies not in the base sklearn container
subprocess.run(["pip", "install", "geopy", "--quiet"], check=True)

import pandas as pd

# engineering.py is uploaded alongside this script
sys.path.insert(0, "/opt/ml/processing/input/scripts")
from engineering import compute_features

INPUT_DIR  = "/opt/ml/processing/input/data"
OUTPUT_DIR = "/opt/ml/processing/output"

TRAIN_END = "2024-04-30"
VAL_END   = "2024-05-31"


def main():
    # ── Load ──────────────────────────────────────────────────────────────
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {INPUT_DIR}")
    if len(csv_files) > 1:
        raise ValueError(f"Expected exactly 1 CSV in {INPUT_DIR}, found: {csv_files}")
    csv_path = os.path.join(INPUT_DIR, csv_files[0])
    print(f"Loading {csv_path} ...")
    raw = pd.read_csv(csv_path)
    print(f"  {len(raw):,} rows loaded")

    # ── Feature engineering ───────────────────────────────────────────────
    print("Engineering features...")
    df = compute_features(raw)
    print(f"  {len(df.columns)} columns after engineering")

    # ── Split ─────────────────────────────────────────────────────────────
    train = df[df["timestamp"] <= TRAIN_END]
    val   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
    test  = df[df["timestamp"] > VAL_END]

    print(f"  Train : {len(train):,}  |  Val : {len(val):,}  |  Test : {len(test):,}")

    # ── Save ──────────────────────────────────────────────────────────────
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(OUTPUT_DIR, split_name, f"{split_name}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        split_df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}  ({len(split_df):,} rows)")


if __name__ == "__main__":
    main()
