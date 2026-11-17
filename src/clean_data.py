"""
Data cleaning and preprocessing pipeline for Kisan Mandi Intelligence.
Reads raw commodity records, filters target commodities and states,
handles date parsing, deduplication, missing date gap-filling, and validation.
"""

import os
import pandas as pd
import numpy as np

KAGGLE_PARQUET_DIR = r"C:\Users\nishchal samadhiya\.cache\kagglehub\datasets\khandelwalmanas\daily-commodity-prices-india\versions\134\parquet"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TARGET_COMMODITIES = ["Wheat", "Onion", "Tomato", "Potato"]
TARGET_STATES = ["Punjab", "Haryana", "Uttar Pradesh"]
TARGET_YEARS = [2023, 2024, 2025]


def load_raw_dataset(years=TARGET_YEARS, parquet_dir=KAGGLE_PARQUET_DIR):
    """Loads parquet data from target years and filters target states & commodities."""
    dfs = []
    for y in years:
        fpath = os.path.join(parquet_dir, f"{y}.parquet")
        if os.path.exists(fpath):
            print(f"Loading {fpath}...")
            df_year = pd.read_parquet(
                fpath,
                columns=[
                    "State",
                    "District",
                    "Market",
                    "Commodity",
                    "Arrival_Date",
                    "Min_Price",
                    "Max_Price",
                    "Modal_Price",
                ],
            )
            # Filter early to keep memory optimal
            df_filtered = df_year[
                (df_year["State"].isin(TARGET_STATES))
                & (df_year["Commodity"].isin(TARGET_COMMODITIES))
            ]
            dfs.append(df_filtered)
        else:
            print(f"Warning: File {fpath} not found.")

    if not dfs:
        raise FileNotFoundError("No parquet data files found!")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def clean_mandi_data(df_raw):
    """
    Applies the cleaning steps specified in PRD Section 6:
    1. Standardize column names
    2. Convert date column to proper datetime format
    3. Remove duplicate records
    4. Remove zero, negative, or extreme outlier prices
    5. Clean and standardize strings (mandi names, commodity names)
    6. Fill gaps in date series using forward-fill per mandi-commodity series
    7. Sort data by date
    """
    df = df_raw.copy()

    # 1. Standardize column names
    df = df.rename(
        columns={
            "Arrival_Date": "date",
            "State": "state",
            "District": "district",
            "Market": "market",
            "Commodity": "commodity",
            "Min_Price": "min_price",
            "Max_Price": "max_price",
            "Modal_Price": "modal_price",
        }
    )

    # 2. Date parsing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Clean text columns
    for col in ["state", "district", "market", "commodity"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # 3. Numeric conversion and validation
    for col in ["min_price", "max_price", "modal_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter invalid prices
    df = df[(df["modal_price"] > 50) & (df["modal_price"] < 50000)]
    df["min_price"] = df["min_price"].fillna(df["modal_price"])
    df["max_price"] = df["max_price"].fillna(df["modal_price"])

    # 4. Remove exact duplicates on (date, state, market, commodity)
    # If multiple prices on the same day, take median modal price
    df_clean = (
        df.groupby(["date", "state", "district", "market", "commodity"], as_index=False)
        .agg({
            "min_price": "min",
            "max_price": "max",
            "modal_price": "median"
        })
    )

    # 5. Sort by market, commodity, and date
    df_clean = df_clean.sort_values(["state", "market", "commodity", "date"]).reset_index(drop=True)

    # 6. Date gap filling per (state, market, commodity)
    # To keep the dataset manageable and dense, we resample high-activity series
    # First, let's identify mandis that have sufficient active days (>= 120 days)
    mandi_counts = df_clean.groupby(["commodity", "state", "market"])["date"].count()
    active_mandi_keys = mandi_counts[mandi_counts >= 100].index

    print(f"Active mandi series with >= 100 days: {len(active_mandi_keys)}")

    cleaned_series_list = []
    for (comm, state, market) in active_mandi_keys:
        sub = df_clean[
            (df_clean["commodity"] == comm)
            & (df_clean["state"] == state)
            & (df_clean["market"] == market)
        ].copy()

        district_val = sub["district"].iloc[0]

        # Set date index and resample to daily frequency
        sub = sub.drop_duplicates(subset=["date"])
        sub = sub.set_index("date").sort_index()

        # Resample daily from its min date to max date
        sub_daily = sub.resample("D").asfreq()

        # Fill static columns
        sub_daily["commodity"] = comm
        sub_daily["state"] = state
        sub_daily["district"] = district_val
        sub_daily["market"] = market

        # Forward fill prices, then backward fill any initial missing
        sub_daily["modal_price"] = sub_daily["modal_price"].ffill().bfill()
        sub_daily["min_price"] = sub_daily["min_price"].ffill().bfill()
        sub_daily["max_price"] = sub_daily["max_price"].ffill().bfill()

        sub_daily = sub_daily.reset_index()
        cleaned_series_list.append(sub_daily)

    if cleaned_series_list:
        final_df = pd.concat(cleaned_series_list, ignore_index=True)
    else:
        final_df = df_clean

    final_df = final_df.sort_values(["state", "commodity", "market", "date"]).reset_index(drop=True)
    return final_df


def run_pipeline():
    """Runs data cleaning and writes raw_prices.csv and clean_prices.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_csv_path = os.path.join(DATA_DIR, "raw_prices.csv")
    clean_csv_path = os.path.join(DATA_DIR, "clean_prices.csv")

    print("Step 1: Loading raw Kaggle dataset subset...")
    df_raw = load_raw_dataset()
    print(f"Loaded {len(df_raw)} raw rows across target commodities & states.")

    # Save a representative raw sample for auditability
    df_raw_sample = df_raw.sample(n=min(50000, len(df_raw)), random_state=42)
    df_raw_sample.to_csv(raw_csv_path, index=False)
    print(f"Saved raw sample to {raw_csv_path}")

    print("Step 2: Cleaning, validating, and filling gaps...")
    df_clean = clean_mandi_data(df_raw)
    print(f"Cleaned dataset has {len(df_clean)} records across {df_clean['market'].nunique()} mandis.")
    print(f"Date range: {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}")
    print("Commodities:", df_clean["commodity"].value_counts().to_dict())

    df_clean.to_csv(clean_csv_path, index=False)
    print(f"Saved clean prices to {clean_csv_path}")
    return df_clean


if __name__ == "__main__":
    run_pipeline()
