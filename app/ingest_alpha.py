import os, json, requests, datetime as dt
import pandas as pd
from dotenv import load_dotenv
from datetime import timezone

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE = "https://www.alphavantage.co/query"

def fetch_historical_chain(symbol: str, date: str|None=None) -> dict:
    """
    Alpha Vantage HISTORICAL_OPTIONS:
      - previous trading session if date is None
      - returns IV and Greeks per contract
    """
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "apikey": API_KEY
    }
    if date:  # YYYY-MM-DD
        params["date"] = date
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Basic guardrails for throttling or errors
    if "Information" in data or "Note" in data or "Error Message" in data:
        raise RuntimeError(f"AlphaVantage response issue: {json.dumps(data)[:400]}")
    return data

def normalize_chain_json(raw: dict) -> pd.DataFrame:
    """
    The doc guarantees greeks and IV are present on historical options.
    We'll flatten out calls & puts into one DataFrame with consistent columns.
    """
    contracts = raw.get("option_chain", raw.get("data", raw))  # be resilient
    rows = []
    # Many community examples show structure grouped by expirations, each with calls/puts arrays
    for expiry_block in contracts if isinstance(contracts, list) else []:
        expiry = expiry_block.get("expiration_date") or expiry_block.get("expirationDate")
        for side in ("calls","puts"):
            for c in expiry_block.get(side, []):
                rows.append({
                    "symbol": raw.get("symbol") or c.get("underlyingSymbol"),
                    "right": "C" if side=="calls" else "P",
                    "expiration": expiry,
                    "strike": float(c.get("strike", c.get("strike_price"))),
                    "bid": float(c.get("bid", 0) or 0),
                    "ask": float(c.get("ask", 0) or 0),
                    "last": float(c.get("last", 0) or 0),
                    "volume": int(c.get("volume", 0) or 0),
                    "open_interest": int(c.get("openInterest", c.get("open_interest", 0)) or 0),
                    "implied_vol": float(c.get("impliedVolatility", c.get("implied_volatility", 0)) or 0),
                    "delta": float(c.get("delta", 0) or 0),
                    "gamma": float(c.get("gamma", 0) or 0),
                    "theta": float(c.get("theta", 0) or 0),
                    "vega": float(c.get("vega", 0) or 0),
                    "rho": float(c.get("rho", 0) or 0)
                })
    df = pd.DataFrame(rows)
    # Fallback: some responses may already be a flat list — handle that too
    if df.empty and isinstance(contracts, list):
        df = pd.DataFrame(contracts)
    return df

if __name__ == "__main__":
    symbol = "SPY"
    # date=None pulls previous trading day on free tier
    raw = fetch_historical_chain(symbol, date=None)
    df = normalize_chain_json(raw)
    if df.empty:
        raise SystemExit("No contracts parsed; print raw and inspect the structure.")
    # Save for reuse to conserve your 25/day limit
    stamp = dt.datetime.now(timezone.utc).strftime("%Y%m%d")
    out = f"data/{symbol}_chain_{stamp}.parquet"
    os.makedirs("data", exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print(df.head(10))