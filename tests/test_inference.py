import json
from src.serving.inference import build_feature_vector, is_fraud, FEATURE_COLUMNS


def test_feature_vector_length():
    """Feature vector should have correct number of comma-separated values."""
    txn = {
        "amount": 100.0,
        "merchant_category": "grocery",
        "merchant_country": "GB",
        "currency": "GBP",
        "is_online": True,
        "is_recurring": False,
        "is_international": False,
        "mcc": 5411,
    }
    csv = build_feature_vector(txn)
    values = csv.split(",")
    assert len(values) == len(FEATURE_COLUMNS), f"Expected {len(FEATURE_COLUMNS)}, got {len(values)}"


def test_one_hot_encoding():
    """Only one category should be 1, rest should be 0."""
    txn = {
        "amount": 50.0,
        "merchant_category": "fuel",
        "merchant_country": "US",
        "currency": "USD",
        "is_online": False,
        "is_recurring": False,
        "is_international": False,
        "mcc": 5541,
    }
    csv = build_feature_vector(txn)
    values = csv.split(",")

    # Check merchant_category one-hot (indices 15-29 based on FEATURE_COLUMNS)
    cat_indices = [i for i, c in enumerate(FEATURE_COLUMNS) if c.startswith("merchant_category_")]
    cat_values = [int(float(values[i])) for i in cat_indices]
    assert sum(cat_values) == 1, f"Expected exactly 1 category active, got {sum(cat_values)}"

    # fuel should be the active one
    fuel_idx = FEATURE_COLUMNS.index("merchant_category_fuel")
    assert int(float(values[fuel_idx])) == 1


def test_is_fraud_threshold():
    """Score above threshold = fraud, below = not fraud."""
    assert is_fraud(0.6) is True
    assert is_fraud(0.5) is True
    assert is_fraud(0.49) is False
    assert is_fraud(0.0) is False


def test_unknown_category_defaults_to_zero():
    """Unknown merchant category should result in all zeros."""
    txn = {
        "amount": 100.0,
        "merchant_category": "casino",  # not in training data
        "merchant_country": "GB",
        "currency": "GBP",
        "is_online": False,
        "is_recurring": False,
        "is_international": False,
        "mcc": 9999,
    }
    csv = build_feature_vector(txn)
    values = csv.split(",")

    cat_indices = [i for i, c in enumerate(FEATURE_COLUMNS) if c.startswith("merchant_category_")]
    cat_values = [int(float(values[i])) for i in cat_indices]
    assert sum(cat_values) == 0, "Unknown category should have all zeros"
