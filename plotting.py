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
    ax1.set_title(market_name, fontsize=14, pad=20)

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

        # Mean volatility annotation
        mid_time = start_time + (end_time - start_time) / 2
        ax2.text(
            mid_time, mean_vol, f"μ = {mean_vol:.3f}",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.8, pad=2),
            fontsize=9, color='#333333'
        )

    # Add legend and formatting
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax1.set_xlabel("Time", fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.suptitle("Volatility Regimes (colors scaled locally)", y=0.95, fontsize=10)

    # Save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    safe_market_name = "".join(c for c in market_name if c.isalnum() or c in (" ", "_")).rstrip()
    plt.savefig(f"{save_dir}/regimes_{safe_market_name}.png", dpi=300)
    plt.close(fig) 