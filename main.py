"""
Main script for analyzing Polymarket price data to detect volatility regimes.

This script performs the following steps:
1.  Fetches market data for a given Polymarket event URL.
2.  Merges the price histories of all markets within that event.
3.  Prepares the data into a long format suitable for time-series analysis.
4.  Calculates a rolling standard deviation of prices to measure volatility.
5.  Uses the PELT algorithm to detect change points in the volatility signal,
    identifying distinct periods (regimes) of stable volatility.
6.  Generates and saves plots for each market, visualizing the price history,
    rolling volatility, and the detected volatility regimes.
"""
import pandas as pd
import polymarket_api as api
import analysis
from plotting import plot_regimes
import numpy as np
import warnings

# --- Configuration ---
EVENT_URL = "https://polymarket.com/event/poland-presidential-election"
# Higher fidelity means fewer data points. A value of 30 is high-resolution.
FIDELITY = 30
# The time window for calculating rolling volatility.
ROLLING_WINDOW = "1d"
# Penalty for the change point detection. Higher values lead to fewer regimes.
PENALTY = 10
# Use None to select all markets in the event
# Example: COLUMN_NAMES_TO_PICK = ['Will Donald Trump win the 2024 US Presidential Election?']
COLUMN_NAMES_TO_PICK = None 

def main():
    """
    Main function to run the Polymarket analysis pipeline.
    """
    # 1. Fetch data from Polymarket APIs
    print(f"Fetching markets for event: {EVENT_URL}")
    token_mapping = api.fetch_market_token_mapping(event_url=EVENT_URL)
    
    if not token_mapping:
        print("No markets with valid token IDs found. Exiting.")
        return

    print("Fetching and merging price histories...")
    df_prices = api.fetch_and_merge_price_histories(token_mapping=token_mapping, fidelity=FIDELITY)
    
    if COLUMN_NAMES_TO_PICK:
        df_prices = df_prices[COLUMN_NAMES_TO_PICK]

    # 2. Perform analysis
    print("Preparing data for analysis...")
    df_long = analysis.prepare_long_dataframe(df_prices)
    
    print("Calculating rolling volatility...")
    df_long = analysis.calculate_rolling_volatility(df_long, window_size_str=ROLLING_WINDOW)

    print("Finding convergence regimes...")
    first_nonzero = analysis.get_first_nonzero_volatility(df_long)
    df_regimes = analysis.find_convergence_regimes(
        df_long, first_nonzero, pen=PENALTY, model="rbf"
    )

    print("\n=== Convergence Regimes ===")
    print(df_regimes)

    # NOTE: The following detailed metrics are now replaced by the regime analysis.
    # They are kept here for reference but are commented out.
    # df_thresh = analysis.calculate_threshold_hits(df_long, first_nonzero, THRESHOLDS)
    # df_half = analysis.calculate_half_life(df_long, first_nonzero)
    # df_decay = analysis.calculate_decay_rate(df_long, first_nonzero)
    # df_auc = analysis.calculate_auc(df_long, first_nonzero)

    # # 3. Display results
    # print("\n=== Threshold Hits ===")
    # print(df_thresh)

    # print("\n=== Half-Life ===")
    # print(df_half)

    # print("\n=== Decay Rates ===")
    # print(df_decay)

    # print("\n=== Area Under Curve (AUC) ===")
    # print(df_auc)
    
    # # Merge for a summary view
    # df_summary = (
    #     df_thresh.query(f"threshold == {THRESHOLDS[0]}")[["market", "I_delta", "time_I_delta"]]
    #     .merge(df_half[["market", "i_half", "time_half"]], on="market")
    #     .merge(df_decay, on="market")
    #     .merge(df_auc, on="market")
    # )
    # print("\n=== Summary Table ===")
    # print(df_summary)

    # 4. Generate plots
    print("\nGenerating plots...")
    active_markets = df_regimes["market"].unique()
    if len(active_markets) == 0:
        print("No markets with significant price activity found. Skipping plots.")
    else:
        for market in active_markets:
            df_market_long = df_long[df_long["market"] == market]
            df_regimes_market = df_regimes[df_regimes["market"] == market]
            plot_regimes(market, df_market_long, df_regimes_market)

if __name__ == "__main__":
    # The 'trapz' function used in analysis.py is deprecated. This suppresses the warning.
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    main() 