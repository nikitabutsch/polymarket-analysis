"""
Main script for analyzing multiple election markets from Polymarket.
"""

import polymarket_api as api
import analysis
from plotting import plot_regimes
import numpy as np
import warnings
import os
from pathlib import Path
import subprocess

# --- Configuration ---
EVENT_URLS = [
    # "https://polymarket.com/event/presidential-election-winner-2024",  # USA
    # "https://polymarket.com/event/will-erdogan-win-the-2023-turkish-presidential-election",  # Turkey
    # "https://polymarket.com/event/who-will-win-the-2022-french-presidential-election",  # France
    "https://polymarket.com/event/poland-presidential-election"  # Poland
]

# Fidelity is the time step in minutes between price samples
FIDELITY = 800  # minutes
# The time window for calculating rolling volatility
ROLLING_WINDOW = "4d"
# Penalty for the change point detection. Higher values lead to fewer regimes.
PENALTY = 0.0006

def fetch_data(event_url: str) -> str:
    """
    Fetch market data for an event using fetch_market_data.py.
    Returns the event slug for use in data paths.
    """
    event_slug = api.extract_slug_from_event_url(event_url)
    print(f"\nFetching data for {event_slug}...")
    
    cmd = [
        "python", "fetch_market_data.py",
        "--event-slug", event_slug,
        "--fidelity", str(FIDELITY),
        "--chunk-days", "15"
    ]
    
    subprocess.run(cmd, check=True)
    return event_slug

def analyze_event(event_url: str):
    """
    Analyze a single Polymarket event.
    
    Args:
        event_url: The URL of the Polymarket event to analyze
    """
    print(f"\n{'='*80}")
    print(f"Analyzing event: {event_url}")
    print(f"{'='*80}\n")
    
    # 1. Fetch data using fetch_market_data.py
    event_slug = fetch_data(event_url)
    
    # 2. Load and process the data
    data_dir = Path("data") / event_slug
    csv_file = data_dir / "all_markets.csv"
    
    if not csv_file.exists():
        print(f"Error: No data file found at {csv_file}")
        return
        
    print("Loading and processing data...")
    print(f"CSV file size: {csv_file.stat().st_size / 1024 / 1024:.2f} MB")
    df_long = analysis.prepare_data_for_analysis(csv_file)
    print(f"Loaded {len(df_long)} rows, {len(df_long['market'].unique())} unique markets")
    
    # Calculate regimes
    print("\nCalculating regimes...")
    df_regimes = analysis.calculate_regimes(df_long, ROLLING_WINDOW, PENALTY)
    print(f"Calculated {len(df_regimes)} regimes")
    
    print("\n=== Convergence Regimes ===")
    print(df_regimes)

    # 3. Generate plots
    print("\nGenerating plots...")
    active_markets = df_regimes["market"].unique()
    if len(active_markets) == 0:
        print("No markets with significant price activity found. Skipping plots.")
    else:
        # Create a directory for this event's plots
        plots_dir = Path("plots") / event_slug
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for market in active_markets:
            df_market_long = df_long[df_long["market"] == market]
            df_regimes_market = df_regimes[df_regimes["market"] == market]
            plot_regimes(market, df_market_long, df_regimes_market, save_dir=str(plots_dir))

def main():
    """
    Main function to run the Polymarket analysis pipeline for multiple events.
    """
    for event_url in EVENT_URLS:
        try:
            analyze_event(event_url)
        except Exception as e:
            print(f"\nError analyzing event {event_url}:")
            print(f"{str(e)}\n")
            continue

if __name__ == "__main__":
    # The 'trapz' function used in analysis.py is deprecated. This suppresses the warning.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main() 