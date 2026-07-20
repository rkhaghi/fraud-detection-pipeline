"""
Real-time inference helper for the Lambda function.

Converts a raw transaction dict → feature vector → endpoint CSV string.

Key design decision: rolling-window features (txn_count_last_1h, etc.)
default to 0 at inference time. In production these would come from
SageMaker Feature Store (online store). For Phase 3 this is acceptable —
the model degrades gracefully when behavioural features are missing.
"""
from __future__ import annotations

import os
from typing import Any

# Feature columns in the exact order used during training.
# These must match the columns produced by build_feature_matrix() in sm_train.py.
# Categorical one-hot columns are included with their exact dummy column names.
FEATURE_COLUMNS = [
    "amount",
    "is_online",
    "is_recurring",
    "is_international",
    "mcc",
    # Engineered behavioural features (default 0 at inference time)
    "time_since_last_txn_seconds",
    "txn_count_last_1h",
    "txn_count_last_6h",
    "txn_count_last_24h",
    "amount_vs_cardholder_avg",
    "geo_velocity_kmh",
    "is_new_merchant",
    "is_new_city",
    "online_ratio_last_7d",
    "spending_acceleration_7d",
    # One-hot: merchant_category (15 categories)
    "merchant_category_coffee",
    "merchant_category_entertainment",
    "merchant_category_fuel",
    "merchant_category_grocery",
    "merchant_category_hotel",
    "merchant_category_jewellery",
    "merchant_category_online_shopping",
    "merchant_category_pharmacy",
    "merchant_category_restaurant",
    "merchant_category_retail_clothing",
    "merchant_category_retail_discount",
    "merchant_category_rideshare",
    "merchant_category_subscription",
    "merchant_category_telecom",
    "merchant_category_transport_rail",
    # One-hot: merchant_country (3 countries)
    "merchant_country_GB",
    "merchant_country_IN",
    "merchant_country_US",
    # One-hot: currency (3 currencies)
    "currency_GBP",
    "currency_INR",
    "currency_USD",
]

FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))


def build_feature_vector(txn: dict[str, Any]) -> str:
    """
    Convert a raw transaction dict to a CSV string for the SageMaker endpoint.

    Args:
        txn: Raw transaction fields (matches synthetic_transactions.csv columns)

    Returns:
        Comma-separated feature values as a string
    """
    row: dict[str, float] = {}

    # ── Static features from the transaction ──────────────────────────
    row["amount"]           = float(txn.get("amount", 0))
    row["is_online"]        = int(str(txn.get("is_online", "False")).lower() == "true")
    row["is_recurring"]     = int(str(txn.get("is_recurring", "False")).lower() == "true")
    row["is_international"] = int(str(txn.get("is_international", "False")).lower() == "true")
    row["mcc"]              = float(txn.get("mcc", 0))

    # ── Behavioural features — default 0 (no history available) ───────
    row["time_since_last_txn_seconds"] = float(txn.get("time_since_last_txn_seconds", -1))
    row["txn_count_last_1h"]           = float(txn.get("txn_count_last_1h", 0))
    row["txn_count_last_6h"]           = float(txn.get("txn_count_last_6h", 0))
    row["txn_count_last_24h"]          = float(txn.get("txn_count_last_24h", 0))
    row["amount_vs_cardholder_avg"]    = float(txn.get("amount_vs_cardholder_avg", 1.0))
    row["geo_velocity_kmh"]            = float(txn.get("geo_velocity_kmh", 0))
    row["is_new_merchant"]             = int(txn.get("is_new_merchant", 0))
    row["is_new_city"]                 = int(txn.get("is_new_city", 0))
    row["online_ratio_last_7d"]        = float(txn.get("online_ratio_last_7d", 0))
    row["spending_acceleration_7d"]    = float(txn.get("spending_acceleration_7d", 1.0))

    # ── One-hot: merchant_category ─────────────────────────────────────
    category = str(txn.get("merchant_category", "")).lower()
    for col in FEATURE_COLUMNS:
        if col.startswith("merchant_category_"):
            row[col] = 1 if col == f"merchant_category_{category}" else 0

    # ── One-hot: merchant_country ──────────────────────────────────────
    country = str(txn.get("merchant_country", "")).upper()
    for col in FEATURE_COLUMNS:
        if col.startswith("merchant_country_"):
            row[col] = 1 if col == f"merchant_country_{country}" else 0

    # ── One-hot: currency ──────────────────────────────────────────────
    currency = str(txn.get("currency", "")).upper()
    for col in FEATURE_COLUMNS:
        if col.startswith("currency_"):
            row[col] = 1 if col == f"currency_{currency}" else 0

    # ── Build ordered CSV ──────────────────────────────────────────────
    values = [str(row.get(col, 0)) for col in FEATURE_COLUMNS]
    return ",".join(values)


def is_fraud(score: float) -> bool:
    return score >= FRAUD_THRESHOLD
