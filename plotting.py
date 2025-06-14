import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib
import os

def plot_price_histories(df_merged):
    """
    Plots the price histories of multiple markets.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    for col in df_merged.columns:
        ax.plot(df_merged.index, df_merged[col], label=col)
    
    # Place legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    ax.set_title("Price Histories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability / Price")
    ax.grid(True)
    
    # Improve date formatting
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    plt.show()

def plot_rolling_belief_spread(df_long, window):
    """
    Plots the rolling belief spread for all markets.
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    for m, grp in df_long.groupby("market"):
        ax.plot(
            grp["time"],
            grp["delta_b_roll"],
            label=m,
            alpha=0.7
        )
    
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Rolling-{window}-step belief spread")
    ax.set_title(f"Belief Spread over the Last {window} Snapshots")
    ax.legend(fontsize="small", ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Improve date formatting
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout
    plt.show()

def plot_annotated_rolling_belief_spread(df_long, window, df_thresh, df_half, df_decay):
    """
    Plots the rolling belief spread with convergence metric annotations.
    """
    markets_to_plot = df_long['market'].unique()
    fig, ax = plt.subplots(figsize=(20, 6))

    for m in markets_to_plot:
        grp = df_long[df_long['market'] == m]
        ax.plot(grp['time'], grp['delta_b_roll'], label=m, alpha=0.6)
    
    # --- Annotation collision avoidance ---
    last_t = None
    y_offset = 0

    for m in markets_to_plot:
        # --- Threshold I_{0.02} ---
        thr_row = df_thresh[(df_thresh.market == m) & (df_thresh.threshold == 0.02)]
        if not thr_row.empty and pd.notna(thr_row.iloc[0]['I_delta']):
            i_delta = int(thr_row.iloc[0]['I_delta'])
            t_delta = thr_row.iloc[0]['time_I_delta']
            val_delta = df_long[(df_long.market == m) & (df_long.i == i_delta)]['delta_b_roll'].iloc[0]
            ax.scatter(t_delta, val_delta, color='C0', marker='x', s=100, zorder=5)
            
            if last_t is not None and (t_delta - last_t).days < 1:
                y_offset += 0.05
            else:
                y_offset = 0
            
            ax.text(t_delta, val_delta + y_offset, f"I={i_delta}", color='C0', fontsize=8, ha='center')
            last_t = t_delta

        # --- Half‐life ---
        hl_row = df_half[df_half.market == m]
        if not hl_row.empty and pd.notna(hl_row.iloc[0]['i_half']):
            i_half = int(hl_row.iloc[0]['i_half'])
            t_half = hl_row.iloc[0]['time_half']
            val_half = df_long[(df_long.market == m) & (df_long.i == i_half)]['delta_b_roll'].iloc[0]
            ax.scatter(t_half, val_half, color='C1', marker='o', s=60, zorder=5)

            if last_t is not None and (t_half - last_t).days < 1:
                y_offset += 0.05
            else:
                y_offset = 0

            ax.text(t_half, val_half - 0.02 - y_offset, f"½@i={i_half}", color='C1', fontsize=8, ha='center', va='top')
            last_t = t_half

        # --- Decay fit ---
        dec_row = df_decay[df_decay.market == m]
        if not dec_row.empty and pd.notna(dec_row.iloc[0]['lambda']):
            λ = dec_row.iloc[0]['lambda']
            grp_fit = df_long[df_long['market'] == m]
            if grp_fit.empty: continue
            
            # Filter to relevant time range for fit
            first_val_row = grp_fit.iloc[0]
            first_delta = first_val_row['delta_b_roll']
            if pd.isna(first_delta) or first_delta == 0: continue

            alpha = np.log(first_delta) + λ * first_val_row['i']
            y_fit = np.exp(alpha - λ * grp_fit['i'])
            ax.plot(grp_fit['time'], y_fit, linestyle='--', color='gray', alpha=0.9, zorder=1)

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Rolling-{window}-step belief spread")
    ax.set_title("Rolling Belief Spread with Convergence Annotations")
    
    # Improve date formatting
    fig.autofmt_xdate(rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize="small", ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout
    plt.show()

def plot_regimes(market_name: str, df_market_long: pd.DataFrame, df_regimes: pd.DataFrame, save_dir: str = "output"):
    """
    Generates and saves a dual y-axis plot visualizing volatility regimes for a single market.

    The plot displays:
    - Price history (left y-axis, black line)
    - Rolling volatility (right y-axis, purple line)
    - Colored regime backgrounds (RdYlGn_r: green=low, red=high volatility)
    - Horizontal dashed lines showing mean volatility per regime
    - Numerical annotations of mean volatility values

    Args:
        market_name (str): Name of the market being plotted
        df_market_long (pd.DataFrame): Long-format DataFrame containing 'time', 
            price column ('p_t' or 'price'), and 'volatility' columns
        df_regimes (pd.DataFrame): Regime data containing 'start_time', 
            'end_time', and 'mean_volatility' columns
        save_dir (str): Directory to save the plot. Defaults to "output"
    """
    if df_regimes.empty:
        print(f"No regimes to plot for market: {market_name}")
        return

    # Create figure with dual y-axes and wide aspect ratio
    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Price plot (left y-axis)
    price_col = "p_t" if "p_t" in df_market_long.columns else "price"
    line1 = ax1.plot(df_market_long["time"], df_market_long[price_col], 
                     label="Price", color="black", alpha=0.8, linewidth=2)
    ax1.set_ylabel("Price", color="black", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="black")
    # Title will be set by suptitle instead to avoid overlap

    # Volatility plot (right y-axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(df_market_long["time"], df_market_long["volatility"], 
                     label="Volatility", color="#5D4E75", alpha=0.8, linewidth=1.5)
    ax2.set_ylabel("Volatility", color="#5D4E75", fontsize=11)
    ax2.tick_params(axis='y', labelcolor="#5D4E75")

    # Normalize volatility values for color mapping
    all_mean_vols = df_regimes["mean_volatility"].values
    if len(all_mean_vols) > 1:
        vol_min, vol_max = all_mean_vols.min(), all_mean_vols.max()
    else:
        vol_min, vol_max = 0, all_mean_vols[0] if len(all_mean_vols) > 0 else 1
    
    # Use RdYlGn_r colormap (green=low, red=high volatility)
    cmap = cm.get_cmap('RdYlGn_r')
    
    # Draw regime backgrounds and annotations
    for _, regime in df_regimes.iterrows():
        start_time = regime["start_time"]
        end_time = regime["end_time"]
        mean_vol = regime["mean_volatility"]

        # Normalize volatility for color mapping
        if vol_max > vol_min:
            normalized_vol = (mean_vol - vol_min) / (vol_max - vol_min)
        else:
            normalized_vol = 0.5
        
        regime_color = cmap(normalized_vol)
        
        # Colored background
        ax1.axvspan(start_time, end_time, alpha=0.3, color=regime_color, zorder=0)

        # Horizontal dashed line at mean volatility level
        ax2.hlines(y=mean_vol, xmin=start_time, xmax=end_time, 
                  colors='#7A6B8A', linestyles='dashed', linewidth=2, alpha=0.6, zorder=1)

        # Mean volatility annotation - temporarily disabled to reduce clutter
        # mid_time = start_time + (end_time - start_time) / 2
        # ax2.text(
        #     mid_time, mean_vol, f"μ = {mean_vol:.3f}",
        #     ha="center", va="bottom",
        #     bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.8, pad=2),
        #     fontsize=9, color='#333333'
        # )

    # Add legend and formatting
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax1.set_xlabel("Time", fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.suptitle(f"{market_name} - Volatility Regimes", y=0.98, fontsize=14, weight='bold')

    # Save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    safe_market_name = "".join(c for c in market_name if c.isalnum() or c in (" ", "_")).rstrip()
    plt.savefig(f"{save_dir}/regimes_{safe_market_name}.png", dpi=300)
    plt.close(fig) 

def plot_total_uncertainty_bar_chart(df_auc: pd.DataFrame, election_name: str = "", save_dir: str = "output", selected_candidates: list = None):
    """
    Creates a "Total Uncertainty" bar chart showing the Area Under the Curve (AUC) 
    of volatility for each market, sorted from highest to lowest uncertainty.
    
    This visualization provides a single, powerful number for each candidate that 
    represents the total amount of disagreement or uncertainty over the entire period.
    It immediately answers: "Which candidate's market was the most contentious overall?"
    
    Args:
        df_auc (pd.DataFrame): DataFrame with 'market' and 'AUC' columns from calculate_auc()
        election_name (str): Name of the election for the title
        save_dir (str): Directory to save the plot. Defaults to "output"
    """
    # Filter out None/NaN values and sort by AUC descending
    df_clean = df_auc.dropna(subset=['AUC']).copy()
    
    # Filter to selected candidates if provided
    if selected_candidates:
        df_clean = df_clean[df_clean['market'].isin(selected_candidates)]
        print(f"Filtered to {len(df_clean)} selected candidates for Total Uncertainty plot")
    
    df_clean = df_clean.sort_values('AUC', ascending=False)
    
    if df_clean.empty:
        print("No valid AUC data to plot")
        return
    
    # Use clean white background theme
    plt.style.use('default')
    
    # Create figure with clean white background
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create sophisticated color palette with muted tones
    n_bars = len(df_clean)
    # Use a professional gradient - from deep blue to light blue
    colors = cm.get_cmap('Blues')(np.linspace(0.4, 0.8, n_bars))
    
    # Create clean horizontal bars
    y_pos = np.arange(n_bars)
    bars = ax.barh(y_pos, df_clean['AUC'], color=colors, height=0.7, 
                   alpha=0.8, edgecolor='white', linewidth=1)
    
    # Style the labels with clean, professional font (extract key name parts)
    def clean_candidate_name(name):
        # Extract the main candidate name from long questions
        if "Will " in name:
            start = name.find("Will ") + 5
            # Handle different question formats
            if " be the" in name:
                # Polish format: "Will Rafał Trzaskowski be the next President of Poland?"
                end = name.find(" be the")
            elif " win the" in name:
                # US/Turkey format: "Will Donald Trump win the 2024 US Presidential Election?"
                end = name.find(" win the")
            else:
                # Fallback - take everything after "Will "
                end = len(name)
            
            if start < end:
                return name[start:end]
        
        # Fallback to basic cleaning
        return name.replace('_', ' ').title()
    
    candidate_labels = [clean_candidate_name(name) for name in df_clean['market']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(candidate_labels, fontsize=12, color='#2C2C2C', weight='400')
    
    # Style the x-axis
    ax.set_xlabel('Market Uncertainty (AUC)', fontsize=13, color='#2C2C2C', weight='500')
    ax.tick_params(axis='x', colors='#666666', labelsize=11)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    
    # Add subtle grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5, color='#E0E0E0')
    ax.set_axisbelow(True)
    
    # Add value labels with clean styling
    for i, (bar, auc_val) in enumerate(zip(bars, df_clean['AUC'])):
        width = bar.get_width()
        ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2,
                f'{auc_val:.3f}', ha='left', va='center', 
                fontsize=10, weight='500', color='#333333')
    
    # Clean, professional title
    if election_name:
        title = f"{election_name} - Market Uncertainty Index"        
        ax.set_title(title, fontsize=16, weight='600', color='#2C2C2C', pad=20)
    else:
        ax.set_title("Market Uncertainty Index", fontsize=16, weight='600', color='#2C2C2C', pad=20)
    
    # Adjust layout for clean appearance
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, top=0.9, right=0.95, bottom=0.1)
    
    # Save with high quality
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    safe_election_name = "".join(c for c in election_name if c.isalnum() or c in (" ", "_")).rstrip() if election_name else "election"
    plt.savefig(f"{save_dir}/total_uncertainty_{safe_election_name}.png", 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Print summary
    print(f"\nTotal Uncertainty Summary ({election_name}):")
    print("=" * 50)
    for i, row in df_clean.iterrows():
        print(f"{row['market']:<30} AUC: {row['AUC']:.3f}")
    print(f"\nMost uncertain market: {df_clean.iloc[0]['market']} (AUC: {df_clean.iloc[0]['AUC']:.3f})")
    if len(df_clean) > 1:
        print(f"Least uncertain market: {df_clean.iloc[-1]['market']} (AUC: {df_clean.iloc[-1]['AUC']:.3f})") 

def plot_volatility_by_context(df_context_summary: pd.DataFrame, election_name: str = "", save_dir: str = "output", selected_candidates: list = None):
    """
    Creates a grouped bar chart comparing volatility across different market contexts
    for each candidate. Shows how volatility changes when candidates are strong favorites
    vs when the race is contested vs when they're long-shots.
    
    Args:
        df_context_summary (pd.DataFrame): DataFrame from aggregate_volatility_by_context()
        election_name (str): Name of the election for the title
        save_dir (str): Directory to save the plot
    """
    if df_context_summary.empty:
        print("No context data to plot")
        return
    
    # Filter to selected candidates if provided
    if selected_candidates:
        df_context_summary = df_context_summary[df_context_summary['market'].isin(selected_candidates)]
        print(f"Filtered to {len(df_context_summary['market'].unique())} selected candidates for Context Volatility plot")
    
    # Pivot data for grouped bar chart
    pivot_data = df_context_summary.pivot(index='market', columns='context', values='volatility_mean')
    
    # Fill NaN values with 0 and ensure we have the expected columns
    expected_contexts = ['Long-shot', 'Contested', 'High-Confidence']
    for context in expected_contexts:
        if context not in pivot_data.columns:
            pivot_data[context] = 0
    
    # Reorder columns and filter out candidates with no data
    pivot_data = pivot_data[expected_contexts]
    pivot_data = pivot_data.dropna(how='all')
    
    if pivot_data.empty:
        print("No valid context data to plot")
        return
    
    # Clean candidate names (extract key name parts)
    def clean_candidate_name(name):
        # Extract the main candidate name from long questions
        if "Will " in name:
            start = name.find("Will ") + 5
            # Handle different question formats
            if " be the" in name:
                # Polish format: "Will Rafał Trzaskowski be the next President of Poland?"
                end = name.find(" be the")
            elif " win the" in name:
                # US/Turkey format: "Will Donald Trump win the 2024 US Presidential Election?"
                end = name.find(" win the")
            else:
                # Fallback - take everything after "Will "
                end = len(name)
            
            if start < end:
                return name[start:end]
        
        # Fallback to basic cleaning
        return name.replace('_', ' ').title()
    
    pivot_data.index = [clean_candidate_name(name) for name in pivot_data.index]
    
    # Create the grouped bar chart with larger figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define colors for each context (professional, distinctive color mapping)
    colors = {
        'Long-shot': '#8B5CF6',      # Purple - speculative, uncertain
        'Contested': '#EF4444',       # Red - heated, volatile competition  
        'High-Confidence': '#059669'  # Green - stable, confident
    }
    
    # Create grouped bars with better sizing
    bar_width = 0.28
    x_pos = np.arange(len(pivot_data))
    
    bars = {}
    for i, context in enumerate(expected_contexts):
        offset = (i - 1) * bar_width
        bars[context] = ax.bar(
            x_pos + offset, pivot_data[context], 
            bar_width, label=context, color=colors[context],
            alpha=0.9, edgecolor='white', linewidth=2
        )
    
    # Customize the chart with much larger fonts
    ax.set_xlabel('Candidate', fontsize=16, color='#2C2C2C', weight='600')
    ax.set_ylabel('Average Volatility', fontsize=16, color='#2C2C2C', weight='600')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pivot_data.index, rotation=0, ha='center', fontsize=14, weight='500')
    
    # Clean up appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#444444', which='both', labelsize=12)
    
    # Add subtle grid
    ax.grid(True, axis='y', alpha=0.4, linestyle='-', linewidth=0.8, color='#E8E8E8')
    ax.set_axisbelow(True)
    
    # Add value labels on bars with larger font
    for context, bar_group in bars.items():
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=12, weight='600', color='#2C2C2C')
    
    # Title and legend with larger fonts
    title = "Volatility by Market Context"
    if election_name:
        title = f"{election_name} - {title}"
    ax.set_title(title, fontsize=18, weight='600', color='#2C2C2C', pad=25)
    
    # Legend with explanation
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, 
                      facecolor='white', edgecolor='#CCCCCC', fontsize=12)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(1.5)
    
    # Add explanation text
    explanation = ("Long-shot: avg price < 35%  |  Contested: 35-65%  |  High-Confidence: > 65%")
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=11, 
             style='italic', color='#555555', weight='500')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.88, left=0.12, right=0.95)
    
    # Save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    safe_election_name = "".join(c for c in election_name if c.isalnum() or c in (" ", "_")).rstrip() if election_name else "election"
    plt.savefig(f"{save_dir}/volatility_by_context_{safe_election_name}.png", 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Print insights
    print(f"\nVolatility by Context Analysis ({election_name}):")
    print("=" * 60)
    
    for candidate in pivot_data.index:
        print(f"\n{candidate}:")
        for context in expected_contexts:
            vol = pivot_data.loc[candidate, context]
            if vol > 0:
                print(f"  {context:<15}: {vol:.4f}")
        
        # Calculate volatility ratio if both contested and high-confidence exist
        contested_vol = pivot_data.loc[candidate, 'Contested']
        confident_vol = pivot_data.loc[candidate, 'High-Confidence']
        if contested_vol > 0 and confident_vol > 0:
            ratio = contested_vol / confident_vol
            print(f"  {'Volatility Ratio':<15}: {ratio:.2f}x (Contested/High-Confidence)") 