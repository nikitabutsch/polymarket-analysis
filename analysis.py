import pandas as pd
import numpy as np
import statsmodels.api as sm
import ruptures as rpt
from typing import Optional, List, Tuple
from scipy import stats

def prepare_long_dataframe(df_prices):
    """
    Converts a wide-format price DataFrame to a long-format DataFrame.

    Args:
        df_prices (pd.DataFrame): Wide-format DataFrame with a DatetimeIndex
            and columns for each market's price.

    Returns:
        pd.DataFrame: A long-format DataFrame with columns:
            - 'time': Timestamp of the observation.
            - 'market': Identifier of the market.
            - 'p_t': Price at time t.
            - 'i': Snapshot index within each market's time series.
            - 'p_prev': Price at the previous time step.
            The resulting DataFrame is ready for time-series belief analysis.
    """
    df_long = (
        df_prices
        .reset_index()
        .melt(id_vars="time", var_name="market", value_name="p_t")
        .dropna(subset=["p_t"])
    )
    df_long = df_long.sort_values(["market", "time"]).reset_index(drop=True)
    df_long["p_prev"] = df_long.groupby("market")["p_t"].shift(1)
    df_long = df_long.dropna(subset=["p_prev"])
    
    # Ensure p_t and p_prev are floats
    df_long['p_t'] = df_long['p_t'].astype(float)
    df_long['p_prev'] = df_long['p_prev'].astype(float)
    
    # Re-calculate snapshot index `i` after dropping rows
    df_long["i"] = df_long.groupby("market").cumcount() + 1
    return df_long

def compute_belief_update(p0, p1):
    """
    Computes the belief update between two price points.

    The update is defined as the fraction of remaining belief resolved (if increasing),
    or the fraction of prior belief lost (if decreasing). This normalization makes
    belief changes comparable across time and markets, and reflects the conditional
    probability of an event occurring in the current step.

    Args:
        p0 (float): The previous price (0 to 1).
        p1 (float): The current price (0 to 1).

    Returns:
        float: The belief update b_i, a value in [0, 1].
    """
    if p1 > p0:
        # Normalize by the potential gain
        return (p1 - p0) / (1 - p0) if (1 - p0) != 0 else 0.0
    else:
        # Normalize by the potential loss
        return (p0 - p1) / p0 if p0 != 0 else 0.0



def calculate_rolling_belief_spread(df_long, window_size_str):
    """
    Calculates the belief update `b_i` and its spread over a rolling time window.

    This first computes the belief update `b_i` for each snapshot. Then, it
    captures recent volatility by finding the difference between the min
    and max belief updates within a time-based rolling window specified by
    `window_size_str`.

    Args:
        df_long (pd.DataFrame): DataFrame from `prepare_long_dataframe`.
        window_size_str (str): The size of the rolling window as a pandas
            time offset string (e.g., '1d', '12h').

    Returns:
        pd.DataFrame: The input DataFrame with added columns: 'b_i', 'b_roll_max',
            'b_roll_min', and 'delta_b_roll' (the rolling spread).
    """
    df_long["b_i"] = df_long.apply(lambda r: compute_belief_update(r.p_prev, r.p_t), axis=1)

    # For time-based rolling, we need a DatetimeIndex.
    df_with_time_index = df_long.set_index('time')

    # Use transform with a lambda to apply rolling on a time window for each market.
    b_roll_max = (
        df_with_time_index
        .groupby("market")["b_i"]
        .transform(lambda x: x.rolling(window_size_str, min_periods=1).max())
    )
    b_roll_min = (
        df_with_time_index
        .groupby("market")["b_i"]
        .transform(lambda x: x.rolling(window_size_str, min_periods=1).min())
    )

    df_long["b_roll_max"] = b_roll_max.values
    df_long["b_roll_min"] = b_roll_min.values

    df_long["delta_b_roll"] = df_long["b_roll_max"] - df_long["b_roll_min"]
    return df_long

def calculate_rolling_volatility(df_long, window_size_str):
    """
    Calculates the rolling standard deviation of prices as a measure of volatility.

    This function computes a time-based rolling standard deviation of the price (`p_t`)
    for each market, providing a robust measure of price volatility over time.

    Args:
        df_long (pd.DataFrame): DataFrame from `prepare_long_dataframe`.
        window_size_str (str): The size of the rolling window as a pandas
            time offset string (e.g., '1d', '12h').

    Returns:
        pd.DataFrame: The input DataFrame with an added 'volatility' column.
    """
    # For time-based rolling, we need a DatetimeIndex.
    df_with_time_index = df_long.set_index('time')

    # Calculate rolling standard deviation of the price for each market.
    rolling_std = (
        df_with_time_index
        .groupby("market")["p_t"]
        .transform(lambda x: x.rolling(window_size_str, min_periods=1).std())
    )
    df_long["volatility"] = rolling_std.values
    df_long["volatility"] = df_long["volatility"].fillna(0)  # First periods will be NaN
    return df_long

def get_first_nonzero_spread(df_long):
    """
    Finds the first snapshot index `i` where market volatility begins.

    Args:
        df_long (pd.DataFrame): DataFrame with 'market', 'i', and 'delta_b_roll'.

    Returns:
        dict: A map of {market_name: first_nonzero_index}.
    """
    return {
        market: int(grp.loc[grp["delta_b_roll"] > 0, "i"].min())
        if (grp["delta_b_roll"] > 0).any()
        else None
        for market, grp in df_long.groupby("market")
    }

def get_first_nonzero_volatility(df_long):
    """
    Finds the first snapshot index `i` where market volatility begins.

    Args:
        df_long (pd.DataFrame): DataFrame with 'market', 'i', and 'volatility'.

    Returns:
        dict: A map of {market_name: first_nonzero_index}.
    """
    return {
        market: int(grp.loc[grp["volatility"] > 1e-6, "i"].min())
        if (grp["volatility"] > 1e-6).any()
        else None
        for market, grp in df_long.groupby("market")
    }


def find_convergence_regimes(
    df_long: pd.DataFrame,
    first_nonzero: dict[str, Optional[int]],
    pen: int = 10,
    model: str = "l2",
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Partition each market's volatility into stable regimes using PELT with batch processing.

    Uses the PELT (Pruned Exact Linear Time) algorithm to detect change points
    in the `volatility` signal. For large datasets, processes data in batches
    to maintain performance while preserving full temporal resolution.

    Args:
        df_long (pd.DataFrame): Long-format DataFrame with 'market', 'i',
            'volatility', and 'time'.
        first_nonzero (dict): Mapping from market to the first index where
            `volatility` > 0, to ignore pre-activity data.
        pen (float): Penalty parameter to control the sensitivity of the
            change point detection. A higher penalty results in fewer regimes.
        model (str): The cost function for the PELT algorithm.
        batch_size (int): Maximum number of data points to process in each batch.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a detected regime,
            with columns: 'market', 'regime', 'start_time', 'end_time',
            'mean_volatility', etc.
    """
    regime_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        if i0 is None:
            continue
        # Subset to data from ignition point onward
        sub = grp.loc[grp["i"] >= i0].reset_index(drop=True)
        if sub.empty:
            continue

        series = sub["volatility"].values
        
        # Apply batch processing for large datasets
        if len(series) > batch_size:
            breakpoints = []
            offset = 0
            
            for batch_start in range(0, len(series), batch_size):
                batch_end = min(batch_start + batch_size, len(series))
                batch_series = series[batch_start:batch_end]
                
                algo = rpt.Pelt(model=model).fit(batch_series)
                batch_breakpoints = algo.predict(pen=pen)
                
                adjusted_breakpoints = [bp + offset for bp in batch_breakpoints[:-1]]
                breakpoints.extend(adjusted_breakpoints)
                offset += len(batch_series)
            
            breakpoints.append(len(series))
        else:
            algo = rpt.Pelt(model=model).fit(series)
            breakpoints = algo.predict(pen=pen)

        # Convert breakpoints to regimes
        start_idx = 0
        for regime_idx, bkp in enumerate(breakpoints):
            end_idx = bkp

            if start_idx >= end_idx:
                continue

            regime_slice = sub.iloc[start_idx:end_idx]
            
            start_row = regime_slice.iloc[0]
            end_row = regime_slice.iloc[-1]

            # Compute regime-level statistics
            regime_vol = regime_slice["volatility"]
            mean_vol = regime_vol.mean()
            std_vol = regime_vol.std(ddof=0)  # ddof=0 handles single-point regimes
            duration = end_row["time"] - start_row["time"]

            regime_rows.append({
                "market": market,
                "regime": regime_idx,
                "start_i": int(start_row["i"]),
                "end_i": int(end_row["i"]),
                "start_time": start_row["time"],
                "end_time": end_row["time"],
                "mean_volatility": mean_vol,
                "std_volatility": std_vol,
                "duration": duration,
            })
            start_idx = end_idx
    
    if not regime_rows:
        return pd.DataFrame()

    df_regimes = pd.DataFrame(regime_rows)

    # Make regime visualizations contiguous
    df_regimes = df_regimes.sort_values(["market", "start_time"])
    df_regimes["end_time"] = (
        df_regimes.groupby("market")["start_time"]
        .shift(-1)
        .fillna(df_regimes["end_time"])
    )
    
    df_regimes["duration"] = df_regimes["end_time"] - df_regimes["start_time"]

    return df_regimes

def calculate_threshold_hits(df_long, first_nonzero, thresholds):
    """
    Calculates when the rolling spread first drops below given thresholds.

    For each market, this finds the snapshot index `I_delta` at which the
    rolling belief spread falls to or below each specified `threshold`,
    signaling a return to consensus.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.
        thresholds (list[float]): A list of thresholds to check (e.g., [0.02, 0.01]).

    Returns:
        pd.DataFrame: A DataFrame detailing the `I_delta` and `time_I_delta`
            for each market-threshold pair.
    """
    thresh_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        for delta in thresholds:
            if i0 is None:
                i_hit, t_hit = None, None
            else:
                mask = (grp["i"] >= i0) & (grp["volatility"] <= delta)
                if mask.any():
                    hit_row = grp.loc[mask].iloc[0]
                    i_hit = int(hit_row["i"])
                    t_hit = hit_row["time"]
                else:
                    i_hit, t_hit = None, None
            thresh_rows.append({
                "market": market, "threshold": delta,
                "I_delta": i_hit, "time_I_delta": t_hit
            })
    return pd.DataFrame(thresh_rows)

def calculate_half_life(df_long, first_nonzero):
    """
    Calculates the half-life of convergence for the belief spread.

    For each market, it measures the initial spread value when volatility begins
    and finds the snapshot index `i_half` where the spread first drops
    to 50% of that initial value.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.

    Returns:
        pd.DataFrame: A DataFrame detailing the half-life metrics for each market.
    """
    half_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        if i0 is None:
            half_rows.append({"market": market, "initial_delta": None, "half_value": None, "i_half": None, "time_half": None})
            continue

        initial_vol = grp.loc[grp["i"] == i0, "volatility"].iloc[0]
        half_val = 0.5 * initial_vol
        mask = (grp["i"] >= i0) & (grp["volatility"] <= half_val)

        if mask.any():
            hit_row = grp.loc[mask].iloc[0]
            i_half = int(hit_row["i"])
            t_half = hit_row["time"]
        else:
            i_half, t_half = None, None

        half_rows.append({
            "market": market,
            "initial_delta": initial_vol,
            "half_value": half_val,
            "i_half": i_half,
            "time_half": t_half
        })
    return pd.DataFrame(half_rows)

def calculate_decay_rate(df_long, first_nonzero):
    """
    Estimates the exponential decay rate (lambda) of the belief spread.

    For each market, this performs a log-linear regression of the rolling
    belief spread against the snapshot index `i` to find how quickly the
    spread converges exponentially.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.

    Returns:
        pd.DataFrame: A DataFrame with the market, decay rate `lambda`, and `r2`
            of the regression fit.
    """
    decay_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        if i0 is None:
            decay_rows.append({"market": market, "lambda": None, "r2": None})
            continue

        sub = grp.loc[(grp["i"] >= i0) & (grp["volatility"] > 1e-6)].copy()
        if len(sub) < 5:
            decay_rows.append({"market": market, "lambda": None, "r2": None})
            continue

        sub["i_shifted"] = sub["i"] - i0
        X = sm.add_constant(sub["i_shifted"])
        Y = np.log(sub["volatility"])
        model = sm.OLS(Y, X).fit()
        
        λ = -model.params.get("i_shifted", np.nan)
        decay_rows.append({"market": market, "lambda": λ, "r2": model.rsquared})
    return pd.DataFrame(decay_rows)

def calculate_auc(df_long, first_nonzero):
    """
    Calculates the Area Under the Curve (AUC) for the rolling belief spread.

    This integrates the `volatility` with respect to the snapshot index `i`
    to quantify the total amount of disagreement/volatility over time for
    each market. A smaller AUC implies quicker convergence.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.

    Returns:
        pd.DataFrame: A DataFrame with the market and its `AUC`, sorted by AUC.
    """
    auc_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        sub = grp[grp["i"] >= i0] if i0 is not None else pd.DataFrame()
        
        if len(sub) < 2:
            auc = None
            normalized_auc = None
            time_span = None
        else:
            # Convert time to numerical values (days since start) for integration
            time_start = sub["time"].min()
            time_numeric = (sub["time"] - time_start).dt.total_seconds() / (24 * 3600)
            
            auc = np.trapz(y=sub["volatility"], x=time_numeric)
            time_span = (sub["time"].max() - sub["time"].min()).total_seconds() / (24 * 3600)
            normalized_auc = auc / max(time_span, 1)
            
        auc_rows.append({
            "market": market, 
            "AUC": auc,
            "AUC_normalized": normalized_auc,
            "time_span_days": time_span if len(sub) >= 2 else None
        })
    return pd.DataFrame(auc_rows).sort_values("AUC")

def filter_top_candidates(df_long: pd.DataFrame, top_k: int = 5, min_prob_threshold: float = 0.05) -> pd.DataFrame:
    """
    Filter to keep only the top-K candidates with highest maximum probability.
    
    This reduces the dataset size significantly while focusing on the most relevant candidates.
    
    Args:
        df_long: Long-format DataFrame with market data
        top_k: Number of top candidates to keep
        min_prob_threshold: Minimum probability threshold (candidates must exceed this at some point)
        
    Returns:
        Filtered DataFrame with only top-K candidates
    """
    # Calculate max probability for each candidate
    max_probs = df_long.groupby('market')['p_t'].max()
    
    # Filter candidates that exceed the minimum threshold
    eligible_candidates = max_probs[max_probs >= min_prob_threshold]
    
    # Get top-K candidates by max probability
    top_candidates = eligible_candidates.nlargest(top_k).index.tolist()
    

    
    # Filter the DataFrame
    df_filtered = df_long[df_long['market'].isin(top_candidates)].copy()
    
    return df_filtered

def prepare_data_for_analysis(csv_file: str, use_top_k_filter: bool = True, start_date: str = None) -> pd.DataFrame:
    """
    Load and prepare data from CSV for analysis.
    
    Handles different CSV formats (either 't,p' or 'timestamp,candidate,probability').
    
    Args:
        csv_file: Path to the CSV file
        use_top_k_filter: Whether to apply top-K filtering for large datasets
        start_date: Optional start date filter (e.g., '2024-06-01'). Only data after this date will be used.
        
    Returns:
        Prepared long-format DataFrame ready for analysis
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    
    if set(df.columns) == {'t', 'p'}:
        df['time'] = pd.to_datetime(df['t'])
        df['market'] = 'market_1'
        df['p_t'] = df['p']
        df_long = df[['time', 'market', 'p_t']].copy()
    else:
        if 'timestamp' in df.columns and 'candidate' in df.columns and 'probability' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'])
            df['market'] = df['candidate']
            df['p_t'] = df['probability']
            df_long = df[['time', 'market', 'p_t']].copy()
        else:
            raise ValueError(f"Unexpected CSV format. Columns: {list(df.columns)}")
    
    if start_date:
        start_datetime = pd.to_datetime(start_date, utc=True)
        df_long = df_long[df_long['time'] >= start_datetime].copy()
    
    if use_top_k_filter and len(df_long['market'].unique()) > 6:
        df_long = filter_top_candidates(df_long, top_k=5, min_prob_threshold=0.05)
    
    df_long = df_long.sort_values(['market', 'time']).reset_index(drop=True)
    df_long['p_prev'] = df_long.groupby('market')['p_t'].shift(1)
    df_long = df_long.dropna(subset=['p_prev'])
    
    df_long['p_t'] = df_long['p_t'].astype(float)
    df_long['p_prev'] = df_long['p_prev'].astype(float)
    
    df_long["i"] = df_long.groupby("market").cumcount() + 1
    
    return df_long

def calculate_regimes(df_long: pd.DataFrame, window_size_str: str, penalty: float) -> pd.DataFrame:
    """
    Calculate volatility regimes for all markets.
    
    Args:
        df_long: DataFrame in long format with timestamp, market, price, and volatility columns
        window_size_str: String specifying the window size for volatility calculation
        penalty: Penalty parameter for change point detection
        
    Returns:
        DataFrame containing regime information for each market
    """
    df_long = calculate_rolling_volatility(df_long, window_size_str)
    first_nonzero = get_first_nonzero_volatility(df_long)
    df_regimes = find_convergence_regimes(
        df_long, first_nonzero, pen=penalty, model="l2", batch_size=10000
    )
    
    return df_regimes

def calculate_regime_contexts(df_long: pd.DataFrame, df_regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze regime contexts by calculating average price during each regime period
    and categorizing regimes based on market sentiment.
    
    Args:
        df_long: Long-format DataFrame with time, market, price, and volatility data
        df_regimes: DataFrame with regime information including start_time, end_time, mean_volatility
        
    Returns:
        DataFrame with regime context analysis including price buckets and volatility by context
    """
    regime_contexts = []
    
    for _, regime in df_regimes.iterrows():
        market = regime['market']
        start_time = regime['start_time']
        end_time = regime['end_time']
        mean_volatility = regime['mean_volatility']
        
        market_data = df_long[
            (df_long['market'] == market) &
            (df_long['time'] >= start_time) &
            (df_long['time'] <= end_time)
        ]
        
        if len(market_data) == 0:
            continue
            
        price_col = "p_t" if "p_t" in market_data.columns else "price"
        avg_price = market_data[price_col].mean()
        

        if avg_price >= 0.65:
            context = "High-Confidence"
        elif avg_price >= 0.35:
            context = "Contested"
        else:
            context = "Long-shot"
            
        regime_contexts.append({
            'market': market,
            'start_time': start_time,
            'end_time': end_time,
            'mean_volatility': mean_volatility,
            'avg_price': avg_price,
            'context': context,
            'regime_duration': (end_time - start_time).total_seconds() / 3600
        })
    
    return pd.DataFrame(regime_contexts)

def aggregate_volatility_by_context(df_regime_contexts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate volatility statistics by market and context.
    
    Args:
        df_regime_contexts: DataFrame from calculate_regime_contexts
        
    Returns:
        DataFrame with aggregated volatility by market and context
    """
    context_summary = df_regime_contexts.groupby(['market', 'context']).agg({
        'mean_volatility': ['mean', 'std', 'count'],
        'regime_duration': 'sum',
        'avg_price': 'mean'
    }).reset_index()
    
    context_summary.columns = [
        'market', 'context', 'volatility_mean', 'volatility_std', 'regime_count',
        'total_duration_hours', 'avg_price_in_context'
    ]
    
    return context_summary

def calculate_frontrunner_vs_uncertainty(all_elections_data: list) -> pd.DataFrame:
    """
    Calculate front-runner status vs total uncertainty data across all elections.
    
    Args:
        all_elections_data: List of tuples (election_name, df_long, df_auc)
        
    Returns:
        DataFrame with candidate, election, max_price, total_uncertainty columns
    """
    scatter_data = []
    
    for election_name, df_long, df_auc in all_elections_data:
        # Calculate max price for each candidate in this election
        price_col = "p_t" if "p_t" in df_long.columns else "price" 
        max_prices = df_long.groupby('market')[price_col].max()
        
        df_auc_clean = df_auc.dropna(subset=['AUC_normalized'])
        
        for _, row in df_auc_clean.iterrows():
            candidate = row['market']
            total_uncertainty = row['AUC_normalized'] if pd.notna(row['AUC_normalized']) else row['AUC']
            max_price = max_prices.get(candidate, 0)
            
            def clean_candidate_name(name):
                """Extract candidate name from question format."""
                if "Will " in name:
                    start = name.find("Will ") + 5
                    if " be the" in name:
                        end = name.find(" be the")
                    elif " win the" in name:
                        end = name.find(" win the")
                    else:
                        end = len(name)
                    
                    if start < end:
                        return name[start:end]
                return name.replace('_', ' ').title()
            
            clean_name = clean_candidate_name(candidate)
            
            scatter_data.append({
                'candidate': candidate,
                'candidate_clean': clean_name,
                'election': election_name,
                'max_price': max_price,
                'total_uncertainty': total_uncertainty
            })
    
    return pd.DataFrame(scatter_data)
