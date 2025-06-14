import requests
import json
from urllib.parse import urlparse
import pandas as pd
import os
import datetime
from tqdm import tqdm



def extract_slug_from_event_url(event_url: str) -> str:
    """
    Given a Polymarket "/event/..." URL (possibly with query parameters), return just the slug part.
    Example: "https://polymarket.com/event/presidential-election-winner-2024?tid=abc123"
             → "presidential-election-winner-2024"
    Raises ValueError if the URL path does not look like /event/<slug>.
    """
    parsed = urlparse(event_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2 or parts[0] != "event":
        raise ValueError(f"URL does not look like a Polymarket /event/ URL: {event_url!r}")
    return parts[1]

def fetch_markets_from_slug(slug: str) -> list:
    """
    Fetch the event data from Polymarket's Gamma API using the slug, then return the list of markets.
    """
    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    resp = requests.get(api_url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise RuntimeError(f"No event found for slug '{slug}'")

    first_event = data[0]
    return first_event.get("markets", [])

def get_prices(market_id, fidelity, start_ts=None, end_ts=None, chunk_days=15, event_slug=None, data_dir=None):
    """
    Fetches price history for a given market token ID.
    If start_ts and end_ts are provided, fetches data in chunks and caches results.
    
    Args:
        market_id (str): The market's clobTokenId
        fidelity (int): Fidelity parameter for the API
        start_ts (int, optional): Unix timestamp for start of data
        end_ts (int, optional): Unix timestamp for end of data
        chunk_days (int, optional): Number of days per chunk when fetching historical data
        event_slug (str, optional): Event slug for caching purposes
        data_dir (str, optional): Directory to store the data files
    
    Returns:
        list: List of price history points
    """
    # If no time bounds specified, use the simple endpoint
    if start_ts is None or end_ts is None:
        url = "https://clob.polymarket.com/prices-history"
        params = {
            "market": market_id,
            "fidelity": fidelity
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("history", [])
    
    # For historical data, use chunked fetching with caching
    return fetch_price_history_chunked(
        market_id, start_ts, end_ts, fidelity, 
        chunk_days=chunk_days, event_slug=event_slug, data_dir=data_dir
    )

def fetch_price_history_chunked(token_id, start_ts, end_ts, fidelity, chunk_days=15, event_slug=None, data_dir=None, force_refresh=False):
    """
    Fetch price history in chunks for a given token_id, time range, and fidelity.
    Caches results in data_dir/event_slug/token_id.json if data_dir is provided.
    Shows a tqdm progress bar for chunk downloads.
    Prints full API response for debugging if not a list.
    """
    if data_dir is not None and event_slug is not None:
        os.makedirs(data_dir, exist_ok=True)
        cache_file = os.path.join(data_dir, f"{token_id}.json")
        if os.path.exists(cache_file) and not force_refresh:
            with open(cache_file, "r") as f:
                return json.load(f)
    all_prices = []
    chunk_seconds = chunk_days * 24 * 3600
    n_chunks = int((end_ts - start_ts) // chunk_seconds) + 1
    current_start = start_ts
    debugged = False
    with tqdm(total=n_chunks, desc=f"Market {token_id[:8]}...", unit="chunk") as pbar:
        while current_start < end_ts:
            current_end = min(current_start + chunk_seconds, end_ts)
            params = {
                "market": token_id,
                "startTs": current_start,
                "endTs": current_end,
                "fidelity": fidelity,
            }
            try:
                resp = requests.get("https://clob.polymarket.com/prices-history", params=params)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    all_prices.extend(data)
                else:
                    print(f"\n[DEBUG] Unexpected data format for market {token_id}")
                    print(f"  Params: {params}")
                    print(f"  Response: {data}")
                    if not debugged:
                        print("\n[DEBUG] Stopping after first unexpected response for inspection.")
                        debugged = True
            except Exception as e:
                print(f"Error fetching chunk {datetime.datetime.fromtimestamp(current_start, tz=datetime.timezone.utc)} to {datetime.datetime.fromtimestamp(current_end, tz=datetime.timezone.utc)} for market {token_id}: {e}")
            current_start = current_end
            pbar.update(1)
    # Only save to cache if we got valid data
    if data_dir is not None and event_slug is not None and all_prices:
        with open(cache_file, "w") as f:
            json.dump(all_prices, f)
    return all_prices

def get_trades(market_id):
    """
    NOTE: This endpoint (data-api.polymarket.com) appears to return the latest trades
    across ALL markets, not for a specific market_id. The official, supported endpoint for
    fetching trades by market is on clob.polymarket.com and requires authentication.
    See: https://docs.polymarket.com/developers/CLOB/trades/trades

    Returns a list of recent trades.
    Each trade is a dict with keys like: ['proxyWallet', 'side', 'asset', 'conditionId', 'size', 'price', 'timestamp', 'title', ...].
    """
    # The 'market' parameter in this URL doesn't seem to filter by market.
    url = f"https://data-api.polymarket.com/trades?market={market_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    return payload

def fetch_market_details(event_url: str) -> dict:
    """
    Given a full Polymarket event URL, returns a dict mapping the market question to its full market data object.

    The volume can be accessed via the 'volume' or 'volumeNum' key in the market data dict.
    Example:
    details = fetch_market_details(event_url)
    for question, market_data in details.items():
        print(f"{question}: {market_data['volumeNum']}")
    """
    slug = extract_slug_from_event_url(event_url)
    markets = fetch_markets_from_slug(slug)

    mapping = {}
    for market in markets:
        question = market.get("question", "<no question>")
        mapping[question] = market

    return mapping



def fetch_market_token_mapping(event_url: str) -> dict:
    """
    Given a full Polymarket event URL, returns a dict mapping the market question to its first clobTokenId.
    """
    slug = extract_slug_from_event_url(event_url)
    markets = fetch_markets_from_slug(slug)

    mapping = {}
    for market in markets:
        question = market.get("question", "<no question>")
        raw_clob_ids = market.get("clobTokenIds", "[]")
        try:
            clob_list = json.loads(raw_clob_ids)
        except json.JSONDecodeError:
            clob_list = []

        first_token = clob_list[0] if clob_list else None
        if first_token:
            mapping[question] = first_token

    return mapping

def fetch_and_merge_price_histories(token_mapping: dict, fidelity: int = 720, start_ts=None, end_ts=None, event_slug=None, data_dir=None):
    """
    Fetches and merges price histories for multiple markets into a single DataFrame, 
    aligning timestamps with an inner join to include only overlapping data points.
    Supports chunked fetching with caching for historical data.

    Args:
        token_mapping (dict): Maps market labels (str) to clobTokenIds (str)
        fidelity (int): Fidelity for the price history API
        start_ts (int, optional): Unix timestamp for start of data
        end_ts (int, optional): Unix timestamp for end of data
        event_slug (str, optional): Event slug for caching purposes
        data_dir (str, optional): Directory to store the data files

    Returns:
        pd.DataFrame: A DataFrame with a DatetimeIndex, and columns for each
            market's price, labeled from the token_mapping.
    """
    dfs = []
    for label, token_id in token_mapping.items():
        price_data = get_prices(
            token_id, fidelity, 
            start_ts=start_ts, 
            end_ts=end_ts,
            event_slug=event_slug,
            data_dir=data_dir
        )
        
        if not price_data:
            continue

        df = pd.DataFrame(price_data)
        df["time"] = pd.to_datetime(df["t"], unit="s")
        df = df.set_index("time").rename(columns={"p": label}).drop(columns=["t"])
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid market DataFrames were created.")

    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.join(df, how="inner")

    if df_merged.empty:
        raise RuntimeError("No overlapping timestamps between markets.")
    return df_merged

def fetch_event_start_end_from_slug(slug: str) -> tuple:
    """
    Fetch the start and end date (ISO8601) for a given event slug from the Gamma API.
    Returns (startDate, endDate) as ISO8601 strings.
    """
    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"No event found for slug '{slug}'")
    first_event = data[0]
    return first_event.get("startDate", []), first_event.get("endDate", [])
