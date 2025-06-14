"""
Polymarket Election Analysis System

This script analyzes multiple election markets from Polymarket to identify
volatility patterns, market regimes, and cross-election comparisons.
"""

import polymarket_api as api
import analysis
from plotting import plot_regimes, plot_total_uncertainty_bar_chart, plot_volatility_by_context, plot_frontrunner_vs_uncertainty_scatter, plot_half_life_comparison
import numpy as np
import warnings
import os
from pathlib import Path
import subprocess

# Configuration
EVENT_URLS = [
    "https://polymarket.com/event/presidential-election-winner-2024",
    "https://polymarket.com/event/will-erdogan-win-the-2023-turkish-presidential-election",
    "https://polymarket.com/event/poland-presidential-election"
]

FIDELITY = 800  # Time step in minutes between price samples
ROLLING_WINDOW = "4d"  # Time window for calculating rolling volatility
PENALTY = 0.0006  # Penalty for change point detection algorithm

# Selected candidates for focused visualizations
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
    
    Args:
        event_url: The URL of the Polymarket event
        
    Returns:
        str: The event slug for use in data paths
    """
    event_slug = api.extract_slug_from_event_url(event_url)
    print(f"Fetching data for {event_slug}...")
    
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
        
    Returns:
        tuple: (election_name, df_long, df_auc) for cross-election analysis
    """
    print(f"{'='*80}")
    print(f"Analyzing event: {event_url}")
    print(f"{'='*80}")
    
    event_slug = fetch_data(event_url)
    
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
    print("Calculating regimes for context analysis...")
    df_regimes = analysis.calculate_regimes(df_long, ROLLING_WINDOW, PENALTY)
    print(f"Calculated {len(df_regimes)} regimes")
    
    # Calculate AUC for total uncertainty analysis
    print("Calculating total uncertainty (AUC)...")
    first_nonzero = analysis.get_first_nonzero_volatility(df_long)
    df_auc = analysis.calculate_auc(df_long, first_nonzero)
    print(f"Calculated AUC for {len(df_auc)} markets")
    
    # Calculate regime contexts for volatility analysis
    print("Analyzing regime contexts...")
    df_regime_contexts = analysis.calculate_regime_contexts(df_long, df_regimes)
    df_context_summary = analysis.aggregate_volatility_by_context(df_regime_contexts)
    print(f"Analyzed {len(df_regime_contexts)} regime contexts across {len(df_context_summary)} market-context combinations")

    # Generate plots
    print("Generating plots...")
    active_markets = df_long["market"].unique()
    
    if len(active_markets) == 0:
        print("No markets with significant price activity found. Skipping plots.")
    else:
        plots_dir = Path("plots") / event_slug
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        selected_candidates = SELECTED_CANDIDATES.get(event_slug, None)
        election_name = event_slug.replace("-", " ").title()
        
        plot_total_uncertainty_bar_chart(df_auc, election_name=election_name, save_dir=str(plots_dir), 
                                        selected_candidates=selected_candidates)
        plot_volatility_by_context(df_context_summary, election_name=election_name, save_dir=str(plots_dir),
                                  selected_candidates=selected_candidates)
    
    return (election_name, df_long, df_auc)

def main():
    """
    Main function to run the Polymarket analysis pipeline for multiple events.
    """
    all_elections_data = []
    
    for event_url in EVENT_URLS:
        try:
            result = analyze_event(event_url)
            if result:
                all_elections_data.append(result)
        except Exception as e:
            print(f"Error analyzing event {event_url}: {e}")
            continue
    
    # Generate cross-election analysis
    if all_elections_data:
        print(f"{'='*80}")
        print("Generating Cross-Election Analysis: Front-Runner vs. Uncertainty")
        print(f"{'='*80}")
        
        df_scatter = analysis.calculate_frontrunner_vs_uncertainty(all_elections_data)
        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_frontrunner_vs_uncertainty_scatter(df_scatter, save_dir=str(plots_dir))
        
        print("Generating Half-Life Comparison chart...")
        plot_half_life_comparison(all_elections_data, save_dir=str(plots_dir))

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main() 