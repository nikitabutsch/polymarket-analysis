import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

def plot_price_histories(df_merged):
    """
    Plots the price histories of multiple markets.
    """
    plt.figure(figsize=(14, 8))
    for col in df_merged.columns:
        plt.plot(df_merged.index, df_merged[col], label=col)
    
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.title("Price Histories")
    plt.xlabel("Time")
    plt.ylabel("Probability / Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_rolling_belief_spread(df_long, window):
    """
    Plots the rolling belief spread for all markets.
    """
    plt.figure(figsize=(20, 6))
    for m, grp in df_long.groupby("market"):
        plt.plot(
            grp["time"],
            grp["delta_b_roll"],
            label=m,
            alpha=0.7
        )
    
    plt.xlabel("Time")
    plt.ylabel(f"Rolling-{window}-step belief spread")
    plt.title(f"Belief Spread over the Last {window} Snapshots")
    plt.legend(fontsize="small", ncol=2, bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
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

    for m in markets_to_plot:
        # --- Threshold I_{0.02} ---
        thr_row = df_thresh[(df_thresh.market == m) & (df_thresh.threshold == 0.02)]
        if not thr_row.empty and pd.notna(thr_row.iloc[0]['I_delta']):
            i_delta = int(thr_row.iloc[0]['I_delta'])
            t_delta = thr_row.iloc[0]['time_I_delta']
            val_delta = df_long[(df_long.market == m) & (df_long.i == i_delta)]['delta_b_roll'].iloc[0]
            ax.scatter(t_delta, val_delta, color='C0', marker='x', s=100)
            ax.text(t_delta, val_delta * 1.05, f"I={i_delta}", color='C0', fontsize=8, ha='center')

        # --- Half‐life ---
        hl_row = df_half[df_half.market == m]
        if not hl_row.empty and pd.notna(hl_row.iloc[0]['i_half']):
            i_half = int(hl_row.iloc[0]['i_half'])
            t_half = hl_row.iloc[0]['time_half']
            val_half = df_long[(df_long.market == m) & (df_long.i == i_half)]['delta_b_roll'].iloc[0]
            ax.scatter(t_half, val_half, color='C1', marker='o', s=100)
            ax.text(t_half, val_half * 0.95, f"½@i={i_half}", color='C1', fontsize=8, ha='center')

        # --- Decay fit ---
        dec_row = df_decay[df_decay.market == m]
        if not dec_row.empty and pd.notna(dec_row.iloc[0]['lambda']):
            λ = dec_row.iloc[0]['lambda']
            grp_fit = df_long[df_long['market'] == m]
            X = grp_fit['i']
            # compute alpha so that y_fit starts at actual first value
            first_delta = grp_fit['delta_b_roll'].iloc[0]
            alpha = np.log(first_delta + 1e-9) + λ * grp_fit['i'].iloc[0]
            y_fit = np.exp(alpha - λ * X)
            ax.plot(grp_fit['time'], y_fit, linestyle='--', color='gray', alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Rolling-{window}-step belief spread")
    ax.set_title("Rolling Belief Spread with Convergence Annotations")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize="small", ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show() 