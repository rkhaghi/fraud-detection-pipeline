
#%%
import pandas as pd
import numpy as np
from geopy.distance import geodesic

# ---------------------------------------------------------------------------
# City coordinates lookup (all 24 cities in the dataset)
# ---------------------------------------------------------------------------
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Austin":        (30.2672,  -97.7431),
    "Bengaluru":     (12.9716,   77.5946),
    "Birmingham":    (52.4862,   -1.8904),
    "Boston":        (42.3601,  -71.0589),
    "Brighton":      (50.8225,   -0.1372),
    "Bristol":       (51.4545,   -2.5879),
    "Cambridge":     (52.2053,    0.1218),
    "Chicago":       (41.8781,  -87.6298),
    "Delhi":         (28.6139,   77.2090),
    "Denver":        (39.7392, -104.9903),
    "Edinburgh":     (55.9533,   -3.1883),
    "Hyderabad":     (17.3850,   78.4867),
    "Leeds":         (53.8008,   -1.5491),
    "London":        (51.5074,   -0.1278),
    "Los Angeles":   (34.0522, -118.2437),
    "Manchester":    (53.4808,   -2.2426),
    "Miami":         (25.7617,  -80.1918),
    "Mumbai":        (19.0760,   72.8777),
    "New York":      (40.7128,  -74.0060),
    "Oxford":        (51.7520,   -1.2577),
    "Reading":       (51.4543,   -0.9781),
    "San Francisco": (37.7749, -122.4194),
    "Seattle":       (47.6062, -122.3321),
    "Washington":    (38.9072,  -77.0369),
}


# ---------------------------------------------------------------------------
# Private helpers — each takes a sorted df and returns it with new columns
# ---------------------------------------------------------------------------


def _add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time since the cardholder's previous transaction (seconds).
    -1 for the very first transaction of each cardholder."""
    df["time_since_last_txn_seconds"] = (
        df.groupby("cardholder_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )
    return df


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count in the 1 h, 6 h, and 24 h window BEFORE each transaction.
    Requires timestamp to be the index (DatetimeIndex) for time-based rolling."""
    df = df.set_index("timestamp")
    for window in ("1h", "6h", "24h"):
        col = f"txn_count_last_{window.replace('h', 'h')}"
        df[col] = (
            df.groupby("cardholder_id")["amount"]
            .transform(lambda x: x.rolling(window).count() - 1)
        )
    df = df.reset_index()
    return df


def _add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of current amount to cardholder's expanding historical average.
    Uses shift(1) so the current transaction is excluded from its own average."""
    cardholder_avg = (
        df.groupby("cardholder_id")["amount"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    # Avoid division by zero on first transaction (avg is NaN → fill with amount)
    df["amount_vs_cardholder_avg"] = df["amount"] / cardholder_avg.fillna(df["amount"])
    return df


def _add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Speed in km/h between consecutive transactions per cardholder.
    Flags impossible travel (e.g. London → Mumbai in 30 min)."""
    df["lat"] = df["merchant_city"].map(lambda c: CITY_COORDS.get(c, (0.0, 0.0))[0])
    df["lon"] = df["merchant_city"].map(lambda c: CITY_COORDS.get(c, (0.0, 0.0))[1])

    df["prev_lat"]  = df.groupby("cardholder_id")["lat"].shift(1)
    df["prev_lon"]  = df.groupby("cardholder_id")["lon"].shift(1)
    df["prev_time"] = df.groupby("cardholder_id")["timestamp"].shift(1)

    def _velocity(row: pd.Series) -> float:
        if pd.isna(row["prev_lat"]):
            return 0.0
        dist_km = geodesic(
            (row["lat"], row["lon"]),
            (row["prev_lat"], row["prev_lon"])
        ).km
        hours = max((row["timestamp"] - row["prev_time"]).total_seconds() / 3600, 0.0167)
        return dist_km / hours

    df["geo_velocity_kmh"] = df.apply(_velocity, axis=1)
    df.drop(columns=["lat", "lon", "prev_lat", "prev_lon", "prev_time"], inplace=True)
    return df


def _add_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags: first time visiting this merchant / city for this cardholder."""
    df["merchant_cumcount"] = df.groupby(
        ["cardholder_id", "merchant_name"]
    ).cumcount()
    df["is_new_merchant"] = (df["merchant_cumcount"] == 0).astype(int)
    df.drop(columns=["merchant_cumcount"], inplace=True)

    df["city_cumcount"] = df.groupby(
        ["cardholder_id", "merchant_city"]
    ).cumcount()
    df["is_new_city"] = (df["city_cumcount"] == 0).astype(int)
    df.drop(columns=["city_cumcount"], inplace=True)

    return df


def _add_online_ratio_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of online transactions in the cardholder's last 7 days (rolling).
    Signals sudden channel shift → Card-Not-Present fraud."""
    df["is_online_int"] = df["is_online"].astype(int)
    df = df.set_index("timestamp")
    df["online_ratio_last_7d"] = (
        df.groupby("cardholder_id")["is_online_int"]
        .transform(lambda x: x.rolling("7d").mean().shift(1))
        .fillna(0.0)
    )
    df = df.reset_index()
    df.drop(columns=["is_online_int"], inplace=True)
    return df


def _add_spending_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """7-day spending acceleration: ratio of last-7d mean to prior-7d mean.
    Values > 1 mean spending is ramping up → bust-out signal."""
    df = df.set_index("timestamp")
    rolling_7d = (
        df.groupby("cardholder_id")["amount"]
        .transform(lambda x: x.rolling("7d").mean().shift(1))
    )
    rolling_14d = (
        df.groupby("cardholder_id")["amount"]
        .transform(lambda x: x.rolling("14d").mean().shift(1))
    )
    # Assign while timestamp is still the index so all three share the same index
    df["spending_acceleration_7d"] = (
        rolling_7d / rolling_14d.replace(0, np.nan).fillna(1.0)
    ).fillna(1.0)
    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering on a raw transactions DataFrame.

    Input columns required:
        transaction_id, cardholder_id, timestamp, merchant_name,
        merchant_city, amount, is_online, is_fraud (optional)

    Returns the same DataFrame with engineered feature columns appended.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cardholder_id", "timestamp"]).reset_index(drop=True)

    df = _add_recency_features(df)          # time_since_last_txn_seconds
    df = _add_velocity_features(df)         # txn_count_last_1h / 6h / 24h
    df = _add_amount_features(df)           # amount_vs_cardholder_avg
    df = _add_geo_features(df)              # geo_velocity_kmh
    df = _add_merchant_features(df)         # is_new_merchant, is_new_city
    df = _add_online_ratio_feature(df)      # online_ratio_last_7d
    df = _add_spending_acceleration(df)     # spending_acceleration_7d

    return df


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from pathlib import Path
    # resolve() makes __file__ absolute, so this works both when run directly
    # (python src/engineering.py) and when pasted into an interactive session
    _here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd() / "src"
    csv_path = str(_here.parent / "Data" / "synthetic_transactions.csv")
    raw = pd.read_csv(csv_path)
    print(f"Loaded {len(raw):,} rows")
    featured = compute_features(raw)
    new_cols = [c for c in featured.columns if c not in raw.columns]
    print(f"Engineered {len(new_cols)} new features: {new_cols}")
    print(featured[new_cols].describe().T[["mean", "min", "max"]])
# %%
