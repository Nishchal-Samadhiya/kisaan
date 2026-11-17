"""
Forecasting models for Kisan Mandi Intelligence.
Implements the 5 models specified in Section 8 of the PRD:
1. Baseline: Naive Model (tomorrow's price = today's price)
2. Statistical: Moving Average (7-day rolling window)
3. Statistical: ARIMA (statsmodels)
4. ML: Linear Regression (scikit-learn)
5. ML: Random Forest Regressor (scikit-learn, fixed random_state=42)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings

try:
    from .features import FEATURE_COLS, build_next_feature_vector
except ImportError:
    from features import FEATURE_COLS, build_next_feature_vector


class NaiveModel:
    """Baseline: Tomorrow's price equals today's price (lag 1)."""
    def __init__(self):
        self.last_price = None

    def fit(self, X_train, y_train, series_train=None):
        if len(y_train) > 0:
            self.last_price = float(y_train.iloc[-1])
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and "price_lag_1" in X.columns:
            return X["price_lag_1"].values
        # Fallback to last known price
        return np.full(len(X), self.last_price if self.last_price else 0.0)

    def forecast_steps(self, last_prices, horizon_days):
        last_val = last_prices[-1]
        return np.full(horizon_days, last_val)


class MovingAverageModel:
    """Statistical: Forecast is the average of the last 7 days."""
    def __init__(self, window=7):
        self.window = window
        self.last_ma = None

    def fit(self, X_train, y_train, series_train=None):
        if isinstance(X_train, pd.DataFrame) and "price_ma_7" in X_train.columns:
            self.last_ma = float(X_train["price_ma_7"].iloc[-1])
        elif len(y_train) >= self.window:
            self.last_ma = float(y_train.iloc[-self.window:].mean())
        elif len(y_train) > 0:
            self.last_ma = float(y_train.mean())
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and "price_ma_7" in X.columns:
            return X["price_ma_7"].values
        return np.full(len(X), self.last_ma if self.last_ma else 0.0)

    def forecast_steps(self, last_prices, horizon_days):
        current_history = list(last_prices[-self.window:])
        preds = []
        for _ in range(horizon_days):
            val = float(np.mean(current_history[-self.window:]))
            preds.append(val)
            current_history.append(val)
        return np.array(preds)


class ARIMAModelWrapper:
    """Statistical: ARIMA model from statsmodels library."""
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.fitted_model = None
        self.last_history = []

    def fit(self, X_train, y_train, series_train=None):
        if series_train is not None and len(series_train) > 0:
            train_series = series_train.values
        else:
            train_series = y_train.values

        self.last_history = list(train_series)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Fit ARIMA model with trend
                model = ARIMA(train_series, order=self.order, trend="t")
                self.fitted_model = model.fit()
        except Exception:
            # Fallback to simpler (1,1,0) if (1,1,1) has convergence issue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(train_series, order=(1, 1, 0))
                    self.fitted_model = model.fit()
            except Exception:
                self.fitted_model = None
        return self

    def predict_test(self, n_test_steps):
        if self.fitted_model is not None:
            try:
                forecast_res = self.fitted_model.forecast(steps=n_test_steps)
                return np.array(forecast_res)
            except Exception:
                pass
        # Fallback to mean/last
        return np.full(n_test_steps, np.mean(self.last_history[-7:]))

    def forecast_steps(self, last_prices, horizon_days):
        if self.fitted_model is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Refit or append latest series to forecast future
                    refitted = ARIMA(last_prices, order=self.order).fit()
                    preds = refitted.forecast(steps=horizon_days)
                    return np.array(preds)
            except Exception:
                pass
        # Fallback extrapolation
        return np.linspace(last_prices[-1], np.mean(last_prices[-7:]), horizon_days)


class LinearRegressionModelWrapper:
    """ML Model: Linear Regression predicting price from lag and date features."""
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train, series_train=None):
        self.model.fit(X_train[FEATURE_COLS], y_train)
        return self

    def predict(self, X):
        return self.model.predict(X[FEATURE_COLS])

    def forecast_steps(self, last_prices_list, start_date, horizon_days):
        history = list(last_prices_list)
        preds = []
        cur_date = pd.to_datetime(start_date)
        for d in range(horizon_days):
            target_date = cur_date + pd.Timedelta(days=d + 1)
            feat = build_next_feature_vector(history, target_date)
            p = float(self.model.predict(feat)[0])
            # Bound prediction to avoid unrealistic divergence
            p = max(50.0, min(p, history[-1] * 2.0))
            preds.append(p)
            history.append(p)
        return np.array(preds)


class RandomForestModelWrapper:
    """ML Model: Random Forest Regressor (reproducible with random_state=42)."""
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train, y_train, series_train=None):
        self.model.fit(X_train[FEATURE_COLS], y_train)
        return self

    def predict(self, X):
        return self.model.predict(X[FEATURE_COLS])

    def forecast_steps(self, last_prices_list, start_date, horizon_days):
        history = list(last_prices_list)
        preds = []
        cur_date = pd.to_datetime(start_date)
        for d in range(horizon_days):
            target_date = cur_date + pd.Timedelta(days=d + 1)
            feat = build_next_feature_vector(history, target_date)
            p = float(self.model.predict(feat)[0])
            p = max(50.0, min(p, history[-1] * 2.0))
            preds.append(p)
            history.append(p)
        return np.array(preds)


def train_all_models(X_train, y_train, series_train=None):
    """Initializes and trains all 5 PRD forecasting models."""
    models = {
        "Naive (Baseline)": NaiveModel().fit(X_train, y_train, series_train),
        "Moving Average (7-Day)": MovingAverageModel(window=7).fit(X_train, y_train, series_train),
        "ARIMA": ARIMAModelWrapper().fit(X_train, y_train, series_train),
        "Linear Regression": LinearRegressionModelWrapper().fit(X_train, y_train, series_train),
        "Random Forest": RandomForestModelWrapper().fit(X_train, y_train, series_train),
    }
    return models
