"""
Fetch and process market data from Polymarket's CLOB API.

The Polymarket CLOB API has limitations on the amount of data that can be fetched in a single request.
When trying to fetch price history for a long time range (e.g., several months), the API will return an error
or incomplete data. This script solves this problem by implementing chunked data fetching, breaking down
the time range into smaller intervals (default: 15 days) and making multiple requests.

The script also implements caching to avoid re-fetching data that has already been downloaded,
making it efficient for repeated runs or when testing different analysis approaches.

Usage:
    python fetch_market_data.py --event-slug "event-slug" [--fidelity 60] [--chunk-days 15]

Arguments:
    --event-slug: The slug of the event to fetch data for (required, e.g. "presidential-election-winner-2024")
    --fidelity: Time interval between data points in minutes (default: 60)
    --chunk-days: Number of days to fetch in each chunk (default: 15)

Output:
    - Individual JSON files for each market in data/{event-slug}/
    - Combined CSV file at data/{event-slug}/all_markets.csv
    - Market metadata at data/{event-slug}/markets.json
"""

import os
import json
import time
import requests
from tqdm import tqdm
from pathlib import Path
import argparse
import datetime
import pandas as pd

def iso_to_unix(iso_str):
    dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    return int(dt.timestamp())

def fetch_markets_from_slug(slug):
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"No event found for slug '{slug}'")
    return data[0].get("markets", [])

def get_clob_token_id(market):
    raw = market.get("clobTokenIds", "[]")
    try:
        clob_list = json.loads(raw)
    except Exception:
        clob_list = []
    return clob_list[0] if clob_list else None

def fetch_price_history_chunked(clob_token_id, start_ts, end_ts, fidelity, chunk_days=15, event_slug=None):
    cache_file = Path("data") / event_slug / f"{clob_token_id}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    all_prices = []
    chunk_seconds = chunk_days * 24 * 3600
    current_start = start_ts
    
    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        params = {
            "market": clob_token_id,
            "startTs": current_start,
            "endTs": current_end,
            "fidelity": fidelity
        }
        
        resp = requests.get("https://clob.polymarket.com/prices-history", params=params)
        if resp.status_code != 200:
            break
            
        data = resp.json()
        if isinstance(data, dict) and "history" in data:
            all_prices.extend(data["history"])
            
        current_start = current_end
        time.sleep(0.5)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(all_prices, f)
    
    return all_prices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-slug", required=True)
    parser.add_argument("--fidelity", type=int, default=60)
    parser.add_argument("--chunk-days", type=int, default=15)
    args = parser.parse_args()

    event_slug = args.event_slug
    fidelity = args.fidelity
    chunk_days = args.chunk_days

    # Fetch event data to get time bounds
    url = f"https://gamma-api.polymarket.com/events?slug={event_slug}"
    resp = requests.get(url)
    resp.raise_for_status()
    event_data = resp.json()[0]
    
    # Get the first market to determine time bounds
    markets = event_data.get("markets", [])
    if not markets:
        raise RuntimeError("No markets found for this event")
        
    first_market = markets[0]
    start_iso = first_market.get("startDate")
    end_iso = first_market.get("endDate")
    
    if not start_iso or not end_iso:
        raise RuntimeError("Could not determine event time bounds")
        
    start_ts = iso_to_unix(start_iso)
    end_ts = iso_to_unix(end_iso)
    
    print(f"Event time bounds: {start_iso} to {end_iso}")

    # Fetch and save market data
    event_dir = Path("data") / event_slug
    event_dir.mkdir(parents=True, exist_ok=True)
    
    with open(event_dir / "markets.json", "w") as f:
        json.dump(markets, f, indent=2)

    # Prepare data for CSV
    all_data = []
    
    for market in tqdm(markets, desc="Markets"):
        clob_token_id = get_clob_token_id(market)
        if not clob_token_id:
            continue
            
        history = fetch_price_history_chunked(
            clob_token_id,
            start_ts,
            end_ts,
            fidelity,
            chunk_days,
            event_slug
        )
        
        if history:
            for point in history:
                timestamp = datetime.datetime.fromtimestamp(point["t"], tz=datetime.timezone.utc)
                all_data.append({
                    "timestamp": timestamp,
                    "candidate": market.get("question", "Unknown"),
                    "probability": point["p"]
                })
    
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values("timestamp")
        csv_file = event_dir / "all_markets.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved combined data to {csv_file}")
        print(f"Total data points: {len(df)}")
        print("Candidates and their data points:")
        print(df.groupby("candidate").size())

if __name__ == "__main__":
    main() 
    