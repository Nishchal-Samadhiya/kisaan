# Kisan Mandi Intelligence (AI Farmer Market & Price Discovery)

**Smart India Hackathon 2026** | **Problem Statement ID:** `SIH26132`  
**Brand Identity:** Kisan Mandi Intelligence  
**Design System:** Functional Minimalism with Earthy Tactile Clarity (Forest Green `#1B4D3E`, Harvest Green `#2E7D32`, Sun Amber `#E65100`, Soft Cream `#F9FAF5`)

---

## 1. Overview & Problem Statement

Farmers across India frequently sell their crops without transparent visibility into which regulated APMC Mandi offers the highest price, or whether prices will rise or fall in the coming week. Mandi modal rates fluctuate daily based on arrivals, local demand, and regional arbitrage.

**Kisan Mandi Intelligence** is an AI-powered agricultural decision-support platform that:
1. Cleans and processes daily historical mandi modal prices (Agmarknet dataset).
2. Predicts future modal prices across **7, 15, and 30-day horizons**.
3. Benchmarks **5 distinct models** (Naive Baseline, Moving Average, ARIMA, Linear Regression, Random Forest) using an 80/20 temporal split.
4. Generates an instant, high-contrast **"Sell Now" vs. "Wait"** strategic recommendation.
5. Ranks all regional mandis and computes net arbitrage gains after transport/diesel costs.

---

## 2. Project Architecture

```
farmer-price-ai/
├── data/
│   ├── raw_prices.csv           # Raw extracted subset from Kaggle Agmarknet dataset
│   ├── clean_prices.csv         # Cleaned, gap-filled, validated dataset
│   └── clean_prices.parquet     # High-speed columnar cache for instant dashboard startup
├── src/
│   ├── __init__.py
│   ├── clean_data.py            # Data loading, date parsing, deduplication, gap filling
│   ├── features.py              # Feature engineering: lag_1, lag_7, lag_14, ma_7, month, dayofweek
│   ├── models.py                # 5 models: Naive, Moving Avg, ARIMA, Linear Reg, Random Forest
│   ├── evaluate.py              # 80/20 temporal evaluation, MAE, RMSE, MAPE metrics
│   └── mandi_compare.py         # Inter-mandi price rankings, arbitrage discovery, spread calculations
├── app.py                       # Full-featured Streamlit dashboard adhering to UI/UX specifications
├── requirements.txt             # Python package dependencies
└── README.md                    # System documentation
```

---

## 3. Forecasting Models & Features

### Features
- **Price Lags**: $y_{t-1}$ (1 day ago), $y_{t-7}$ (7 days ago), $y_{t-14}$ (14 days ago)
- **Rolling Stat**: 7-day Moving Average ($\text{MA}_7$)
- **Calendar Signals**: Month of year ($1-12$), Day of week ($0-6$)

### Models
1. **Naive Baseline**: Assumes tomorrow's price equals today's price ($y_{t+1} = y_t$). All models must beat this baseline.
2. **Moving Average (7-Day)**: Extrapolates recent 7-day trend rolling window.
3. **ARIMA (Statsmodels)**: Auto-Regressive Integrated Moving Average fitted with linear trend.
4. **Linear Regression (Scikit-Learn)**: Multi-variate linear model leveraging price lags and calendar features.
5. **Random Forest Regressor (Scikit-Learn)**: 100 decision trees (`random_state=42` for exact reproducibility).

---

## 4. Evaluation Methodology

- **Split**: 80% oldest records for training, 20% newest records for out-of-sample testing (chronological split, no random sampling to avoid lookahead bias).
- **Error Metrics**:
  - **MAE (₹/qtl)**: Mean Absolute Error in Rupees per quintal.
  - **RMSE (₹/qtl)**: Root Mean Squared Error (penalizes extreme deviations).
  - **MAPE (%)**: Mean Absolute Percentage Error (primary metric for selecting the winning model).

---

## 5. UI/UX Design Specifications

- **Typography**: Space Grotesk (numerical readouts, headings, metrics, tables) and Plus Jakarta Sans (body, labels, controls).
- **Color Palette**:
  - Primary (`#1B4D3E`): Deep Forest Green
  - Secondary (`#2E7D32`): Fresh Harvest Green
  - Tertiary (`#E65100`): Warm Sun Amber
  - Canvas (`#F9FAF5`): Soft Cream paper background
- **Semantic Badges**:
  - **SELL NOW**: `#E8F5E9` background, `#1B5E20` text, `#81C784` border with upward arrow icon.
  - **WAIT**: `#FFF3E0` background, `#BF360C` text, `#FFB74D` border with clock icon.

---

## 6. How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-run data cleaning pipeline
```bash
python src/clean_data.py
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.
