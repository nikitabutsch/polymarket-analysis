"""
Main script for analyzing multiple election markets from Polymarket.
"""

import polymarket_api as api
import analysis
from plotting import plot_regimes, plot_total_uncertainty_bar_chart, plot_volatility_by_context
import numpy as np
import warnings
import os
from pathlib import Path
import subprocess

# --- Configuration ---
EVENT_URLS = [
    "https://polymarket.com/event/presidential-election-winner-2024",  # USA
    "https://polymarket.com/event/will-erdogan-win-the-2023-turkish-presidential-election",  # Turkey
    # "https://polymarket.com/event/who-will-win-the-2022-french-presidential-election",  # France
    "https://polymarket.com/event/poland-presidential-election"  # Poland
]

# Fidelity is the time step in minutes between price samples
FIDELITY = 800  # minutes
# The time window for calculating rolling volatility
ROLLING_WINDOW = "4d"
# Penalty for the change point detection. Higher values lead to fewer regimes.
PENALTY = 0.0006

# Selected candidates for clean visualizations (manually curated)
SELECTED_CANDIDATES = {
    "presidential-election-winner-2024": [
        "Will Donald Trump win the 2024 US Presidential Election?",
        "Will Kamala Harris win the 2024 US Presidential Election?", 
        "Will Joe Biden win the 2024 US Presidential Election?"
    ],
    "poland-presidential-election": [
        "Will Karol Nawrocki be the next President of Poland?",
        "Will Rafał Trzaskowski be the next President of Poland?"
    ],
    "will-erdogan-win-the-2023-turkish-presidential-election": [
        "Will Erdoğan win the 2023 Turkish presidential election?"
    ]
}

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
    
    # Apply date filter for US election to focus on interesting period
    start_date = "2024-06-01" if "presidential-election-winner-2024" in str(csv_file) else None
    if start_date:
        print(f"Applying date filter for US election: data from {start_date} onwards")
    
    df_long = analysis.prepare_data_for_analysis(csv_file, start_date=start_date)
    print(f"Loaded {len(df_long)} rows, {len(df_long['market'].unique())} unique markets")
    
    # Calculate regimes for context analysis
    print("\nCalculating regimes for context analysis...")
    df_regimes = analysis.calculate_regimes(df_long, ROLLING_WINDOW, PENALTY)
    print(f"Calculated {len(df_regimes)} regimes")
    
    # Calculate AUC (Area Under Curve) for total uncertainty analysis
    print("Calculating total uncertainty (AUC)...")
    first_nonzero = analysis.get_first_nonzero_volatility(df_long)
    df_auc = analysis.calculate_auc(df_long, first_nonzero)
    print(f"Calculated AUC for {len(df_auc)} markets")
    
    # Calculate regime contexts for volatility by context analysis
    print("Analyzing regime contexts...")
    df_regime_contexts = analysis.calculate_regime_contexts(df_long, df_regimes)
    df_context_summary = analysis.aggregate_volatility_by_context(df_regime_contexts)
    print(f"Analyzed {len(df_regime_contexts)} regime contexts across {len(df_context_summary)} market-context combinations")

    # 3. Generate plots
    print("\nGenerating plots...")
    active_markets = df_long["market"].unique()  # Use df_long instead of df_regimes
    if len(active_markets) == 0:
        print("No markets with significant price activity found. Skipping plots.")
    else:
        # Create a directory for this event's plots
        plots_dir = Path("plots") / event_slug
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual regime plots for each market - commented out to save time
        # for market in active_markets:
        #     df_market_long = df_long[df_long["market"] == market]
        #     df_regimes_market = df_regimes[df_regimes["market"] == market]
        #     plot_regimes(market, df_market_long, df_regimes_market, save_dir=str(plots_dir))
        
        # Get selected candidates for this election
        selected_candidates = SELECTED_CANDIDATES.get(event_slug, None)
        
        # Generate the Total Uncertainty bar chart
        print("Generating Total Uncertainty bar chart...")
        election_name = event_slug.replace("-", " ").title()
        plot_total_uncertainty_bar_chart(df_auc, election_name=election_name, save_dir=str(plots_dir), 
                                        selected_candidates=selected_candidates)
        
        # Generate the Volatility by Context bar chart
        print("Generating Volatility by Context bar chart...")
        plot_volatility_by_context(df_context_summary, election_name=election_name, save_dir=str(plots_dir),
                                  selected_candidates=selected_candidates)

def main():
    """
    Main function to run the Polymarket analysis pipeline for multiple events.
    """
    for event_url in EVENT_URLS:
        try:
            analyze_event(event_url)
        except Exception as e:
            print(f"\nError analyzing event {event_url}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            continue

if __name__ == "__main__":
    # The 'trapz' function used in analysis.py is deprecated. This suppresses the warning.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main() 