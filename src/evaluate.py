"""
Model evaluation module for Kisan Mandi Intelligence.
Performs 80/20 temporal split evaluation on the 5 models,
computing MAE, RMSE, and MAPE, and highlighting the best model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    from .features import build_features, temporal_train_test_split, extract_single_series
    from .models import (
        NaiveModel,
        MovingAverageModel,
        ARIMAModelWrapper,
        LinearRegressionModelWrapper,
        RandomForestModelWrapper,
    )
except ImportError:
    from features import build_features, temporal_train_test_split, extract_single_series
    from models import (
        NaiveModel,
        MovingAverageModel,
        ARIMAModelWrapper,
        LinearRegressionModelWrapper,
        RandomForestModelWrapper,
    )


def compute_metrics(y_true, y_pred):
    """Calculates MAE (₹), RMSE (₹), and MAPE (%)."""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    # Avoid zero division
    valid_mask = (y_true_arr > 0) & (~np.isnan(y_pred_arr))
    if not np.any(valid_mask):
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    y_t = y_true_arr[valid_mask]
    y_p = y_pred_arr[valid_mask]

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    mape = float(np.mean(np.abs((y_t - y_p) / y_t)) * 100.0)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }


def evaluate_all_models_on_series(series_daily):
    """
    Given a daily price series for a single mandi & commodity:
    1. Extracts lag & calendar features
    2. Splits into 80% train and 20% test chronologically
    3. Fits all 5 models on train
    4. Evaluates on test
    5. Returns evaluation metrics table, trained model dictionary, and test predictions df
    """
    df_feat = build_features(series_daily)
    if len(df_feat) < 40:
        raise ValueError(f"Insufficient historical data ({len(df_feat)} rows) for reliable 80/20 evaluation.")

    X_train, y_train, X_test, y_test, train_df, test_df = temporal_train_test_split(df_feat, test_ratio=0.20)

    # Models dictionary
    models = {
        "Naive (Baseline)": NaiveModel().fit(X_train, y_train, series_train=train_df["modal_price"]),
        "Moving Average (7-Day)": MovingAverageModel(window=7).fit(X_train, y_train, series_train=train_df["modal_price"]),
        "ARIMA": ARIMAModelWrapper().fit(X_train, y_train, series_train=train_df["modal_price"]),
        "Linear Regression": LinearRegressionModelWrapper().fit(X_train, y_train),
        "Random Forest": RandomForestModelWrapper().fit(X_train, y_train),
    }

    test_preds = {"Actual": y_test.values}
    results = []

    # 1. Naive
    naive_pred = models["Naive (Baseline)"].predict(X_test)
    test_preds["Naive (Baseline)"] = naive_pred
    naive_metrics = compute_metrics(y_test, naive_pred)

    for name, model in models.items():
        if name == "ARIMA":
            pred = model.predict_test(len(y_test))
        else:
            pred = model.predict(X_test)

        test_preds[name] = pred
        metrics = compute_metrics(y_test, pred)

        beat_baseline = (metrics["mape"] < naive_metrics["mape"]) if name != "Naive (Baseline)" else False
        results.append({
            "Model": name,
            "Type": "Baseline" if "Naive" in name else ("Statistical" if "Moving" in name or "ARIMA" in name else "Machine Learning"),
            "MAE (Rs)": metrics["mae"],
            "RMSE (Rs)": metrics["rmse"],
            "MAPE (%)": metrics["mape"],
            "Beat Baseline": "-" if name == "Naive (Baseline)" else ("Yes" if beat_baseline else "No"),
        })

    eval_df = pd.DataFrame(results)

    # Best model determined by lowest test MAPE (or lowest RMSE)
    best_idx = eval_df["MAPE (%)"].idxmin()
    eval_df["Is Best"] = False
    eval_df.loc[best_idx, "Is Best"] = True

    # Build predictions dataframe with dates
    pred_df = pd.DataFrame(test_preds, index=test_df.index)

    return eval_df, models, pred_df, train_df, test_df


if __name__ == "__main__":
    import os
    clean_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clean_prices.parquet")
    if os.path.exists(clean_p):
        df_all = pd.read_parquet(clean_p)
        # Test on a major mandi
        s = extract_single_series(df_all, "Wheat", "Punjab", "Khanna")
        if not s.empty:
            eval_table, _, _, _, _ = evaluate_all_models_on_series(s)
            print("Evaluation Table:")
            print(eval_table.to_string())
