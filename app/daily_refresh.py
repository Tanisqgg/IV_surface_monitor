# app/daily_refresh.py
import os
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path

from app.ingest_alpha import fetch_historical_chain, normalize_chain_json
from app.features import add_time_and_moneyness
from app.iv import compute_iv_for_df
from app.surface import build_surface
from app.anomaly import calendar_violations, convexity_violations

DATA_DIR = Path(os.environ.get("DATA_DIR", "app/data"))

def run(symbol: str = None) -> dict:
    symbol = symbol or os.getenv("SYMBOL", "SPY")

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ALPHAVANTAGE_API_KEY (set it in Render → Environment).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Fetch previous trading session (AlphaVantage free tier behavior for date=None)
    print(f"[daily_refresh] fetching chain for {symbol} …")
    raw = fetch_historical_chain(symbol, date=None)
    df = normalize_chain_json(raw)
    if df.empty:
        raise RuntimeError("AlphaVantage returned no contracts.")

    # 2) Features → IV
    df_feat = add_time_and_moneyness(df)
    df_iv = compute_iv_for_df(df_feat)

    # 3) Save a stamped snapshot
    stamp = dt.datetime.utcnow().strftime("%Y%m%d")
    base = f"{symbol}_chain_{stamp}_feat_iv"
    out_parq = DATA_DIR / f"{base}.parquet"
    df_iv.to_parquet(out_parq, index=False)
    print(f"[daily_refresh] wrote {out_parq} rows={len(df_iv)}")

    # 4) Build surface + meta + anomalies
    k, T, IV, W, meta = build_surface(df_iv, n_k=61, max_exps=8, method="svi")
    npz = DATA_DIR / f"{base}_surface.npz"
    meta_csv = DATA_DIR / f"{base}_surface_meta.csv"
    np.savez_compressed(npz, k_grid=k, T_grid=T, IV=IV, W=W)
    pd.DataFrame(meta).to_csv(meta_csv, index=False)

    cal = calendar_violations(k, T, W, tol=1e-6)
    cal_csv = DATA_DIR / f"{base}_calendar_violations.csv"
    cal.to_csv(cal_csv, index=False)

    conv = convexity_violations(df_iv, max_exps=8, tol=0.0)
    conv_csv = DATA_DIR / f"{base}_convexity_violations.csv"
    conv.to_csv(conv_csv, index=False)

    print(f"[daily_refresh] surface/meta/anoms saved for {symbol}")
    return {
        "symbol": symbol,
        "rows": int(len(df_iv)),
        "feat_iv": str(out_parq),
        "surface_npz": str(npz),
        "meta_csv": str(meta_csv),
        "calendar_csv": str(cal_csv),
        "convexity_csv": str(conv_csv),
    }