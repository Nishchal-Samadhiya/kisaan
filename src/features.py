"""
Feature engineering module for Kisan Mandi Intelligence.
Constructs lag features, rolling moving averages, and calendar features
as specified in Section 8 of the PRD:
- Price of 1 day ago (lag_1)
- Price of 7 days ago (lag_7)
- Price of 14 days ago (lag_14)
- Average price of the last 7 days (ma_7)
- Month of the year (1-12)
- Day of the week (0-6)
"""

import pandas as pd
import numpy as np

FEATURE_COLS = [
    "price_lag_1",
    "price_lag_7",
    "price_lag_14",
    "price_ma_7",
    "month",
    "day_of_week",
]
TARGET_COL = "modal_price"


def extract_single_series(df, commodity, state, market):
    """
    Filters and formats a continuous daily series for a single commodity in a mandi.
    Returns DataFrame indexed by date with 'modal_price'.
    """
    sub = df[
        (df["commodity"] == commodity)
        & (df["state"] == state)
        & (df["market"] == market)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").drop_duplicates(subset=["date"])
    sub = sub.set_index("date")

    # Ensure continuous daily index
    sub_daily = sub.resample("D").asfreq()
    sub_daily["modal_price"] = sub_daily["modal_price"].ffill().bfill()
    sub_daily["min_price"] = sub_daily["min_price"].ffill().bfill()
    sub_daily["max_price"] = sub_daily["max_price"].ffill().bfill()
    sub_daily["commodity"] = commodity
    sub_daily["state"] = state
    sub_daily["market"] = market

    return sub_daily


def build_features(df_series):
    """
    Builds the PRD-required features from the daily series:
    - price_lag_1, price_lag_7, price_lag_14
    - price_ma_7 (rolling 7-day average of past prices, excluding current day)
    - month (1-12)
    - day_of_week (0-6)
    """
    df = df_series.copy()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' missing from input series.")

    # Lag features
    df["price_lag_1"] = df[TARGET_COL].shift(1)
    df["price_lag_7"] = df[TARGET_COL].shift(7)
    df["price_lag_14"] = df[TARGET_COL].shift(14)

    # 7-day moving average of prior days (shift 1 so today's price is not leaked)
    df["price_ma_7"] = df[TARGET_COL].shift(1).rolling(window=7, min_periods=1).mean()

    # Calendar features
    dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["date"])
    df["month"] = dates.month
    df["day_of_week"] = dates.dayofweek

    # Drop rows that don't have sufficient historical lag (first 14 days)
    df_clean = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    return df_clean


def temporal_train_test_split(df_features, test_ratio=0.20):
    """
    Splits the data by time as required by PRD Section 9:
    Oldest 80% used for training, newest 20% for testing.
    No random split to prevent future leakage.
    """
    n = len(df_features)
    split_idx = int(n * (1.0 - test_ratio))

    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    return X_train, y_train, X_test, y_test, train_df, test_df


def build_next_feature_vector(recent_prices_list, target_date):
    """
    Constructs a feature row for a future step during multi-step recursive forecasting.
    recent_prices_list contains past prices ending at yesterday's price.
    """
    if len(recent_prices_list) < 14:
        raise ValueError("Need at least 14 days of history to compute features.")

    lag_1 = recent_prices_list[-1]
    lag_7 = recent_prices_list[-7]
    lag_14 = recent_prices_list[-14]
    ma_7 = float(np.mean(recent_prices_list[-7:]))
    month = target_date.month
    day_of_week = target_date.dayofweek

    features = pd.DataFrame([{
        "price_lag_1": lag_1,
        "price_lag_7": lag_7,
        "price_lag_14": lag_14,
        "price_ma_7": ma_7,
        "month": month,
        "day_of_week": day_of_week,
    }])
    return features[FEATURE_COLS]
