"""
Mandi price comparison and Sell-Now-or-Wait decision logic.
Implements PRD FR5 and FR6:
- Compares recent average prices across all mandis for a commodity
- Ranks mandis from highest to lowest price with spread analysis
- Evaluates Sell Now vs. Wait suggestion based on forecast trajectory
"""

import pandas as pd
import numpy as np


def get_mandi_comparison(df, commodity, selected_state=None, selected_market=None, lookback_days=14):
    """
    Computes comparative price metrics across all mandis for a given commodity.
    Returns:
    - comparison_df: ranked DataFrame of mandis
    - best_alt_mandi: dict of best alternative mandi details
    """
    # Filter commodity
    sub = df[df["commodity"] == commodity].copy()
    if sub.empty:
        return pd.DataFrame(), None

    # Determine max date in dataset for this commodity
    sub["date"] = pd.to_datetime(sub["date"])
    max_date = sub["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=lookback_days)

    recent_df = sub[sub["date"] >= cutoff_date]
    if recent_df.empty:
        recent_df = sub

    # Group by state, district, market
    stats = (
        recent_df.groupby(["state", "district", "market"])
        .agg(
            latest_price=("modal_price", "last"),
            avg_price=("modal_price", "mean"),
            min_price=("modal_price", "min"),
            max_price=("modal_price", "max"),
            records_count=("modal_price", "count"),
            last_date=("date", "max"),
        )
        .reset_index()
    )

    stats["latest_price"] = stats["latest_price"].round(1)
    stats["avg_price"] = stats["avg_price"].round(1)
    stats["min_price"] = stats["min_price"].round(1)
    stats["max_price"] = stats["max_price"].round(1)

    # Sort from highest to lowest average price
    stats = stats.sort_values(by="latest_price", ascending=False).reset_index(drop=True)

    # If current mandi is selected, compute price differentials
    current_price = None
    if selected_market:
        cur_match = stats[(stats["market"] == selected_market)]
        if selected_state:
            cur_match = cur_match[cur_match["state"] == selected_state]

        if not cur_match.empty:
            current_price = float(cur_match["latest_price"].iloc[0])

    if current_price is not None:
        stats["diff_vs_current"] = (stats["latest_price"] - current_price).round(1)
        stats["premium_pct"] = ((stats["diff_vs_current"] / current_price) * 100).round(1)
    else:
        stats["diff_vs_current"] = 0.0
        stats["premium_pct"] = 0.0

    # Best alternative mandi (highest price excluding current market)
    best_alt_mandi = None
    if selected_market:
        alt_df = stats[stats["market"] != selected_market]
        if not alt_df.empty:
            top_alt = alt_df.iloc[0]
            gain = top_alt["diff_vs_current"]
            best_alt_mandi = {
                "market": top_alt["market"],
                "state": top_alt["state"],
                "district": top_alt["district"],
                "price": top_alt["latest_price"],
                "diff": gain,
                "pct": top_alt["premium_pct"],
                "is_higher": gain > 0,
            }

    return stats, best_alt_mandi


def calculate_sell_now_or_wait(current_price, forecast_prices, horizon_days=7):
    """
    Implements PRD FR6:
    Compares today's price with the average predicted price of the next few days.
    If the forecast is clearly higher (> 2.5% gain), suggests WAITING.
    If lower or about the same, suggests SELLING NOW.
    """
    if current_price is None or current_price <= 0 or len(forecast_prices) == 0:
        return {
            "decision": "SELL NOW",
            "badge_color": "sell",
            "avg_forecast": current_price or 0.0,
            "diff_rs": 0.0,
            "diff_pct": 0.0,
            "peak_price": current_price or 0.0,
            "peak_day": 1,
            "recommendation": "Current market price provides stable immediate returns. Sell now to avoid holding risks.",
        }

    forecast_arr = np.array(forecast_prices, dtype=float)
    avg_forecast = float(np.mean(forecast_arr))
    max_forecast = float(np.max(forecast_arr))
    peak_day = int(np.argmax(forecast_arr)) + 1

    diff_rs = round(avg_forecast - current_price, 1)
    diff_pct = round((diff_rs / current_price) * 100, 2)
    peak_gain_pct = round(((max_forecast - current_price) / current_price) * 100, 2)

    # Threshold for Wait: expected price gain >= +2.0%
    if diff_pct >= 2.0 or peak_gain_pct >= 4.0:
        decision = "WAIT"
        badge_color = "wait"
        recommendation = (
            f"Forecast projects modal price to rise toward ₹{max_forecast:,.0f}/qtl "
            f"(+{peak_gain_pct}%) around Day {peak_day}. "
            f"Holding produce for {peak_day}–{min(horizon_days, peak_day + 3)} days is recommended to maximize revenue."
        )
    else:
        decision = "SELL NOW"
        badge_color = "sell"
        if diff_pct < -1.5:
            recommendation = (
                f"Forecast indicates a declining price trend (projected loss of -₹{abs(diff_rs):,.0f}/qtl, {diff_pct}%). "
                f"Liquidate current harvest immediately at ₹{current_price:,.0f}/qtl to avoid inventory depreciation."
            )
        else:
            recommendation = (
                f"Prices are expected to remain flat/steady (marginal delta: {diff_rs:+,.0f} ₹/qtl). "
                f"Sell now to capture existing liquidity without incurring extra storage or transit costs."
            )

    return {
        "decision": decision,
        "badge_color": badge_color,
        "avg_forecast": round(avg_forecast, 1),
        "diff_rs": diff_rs,
        "diff_pct": diff_pct,
        "peak_price": round(max_forecast, 1),
        "peak_day": peak_day,
        "recommendation": recommendation,
    }
