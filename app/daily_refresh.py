# app/daily_refresh.py
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from app.ingest_alpha import fetch_historical_chain, normalize_chain_json
from app.features import add_time_and_moneyness
from app.iv import compute_iv_for_df
from app.surface import build_surface
from app.anomaly import calendar_violations, convexity_violations

DATA_DIR = Path("app/data")

def run(symbol: str = "SPY"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Pull previous trading session from Alpha Vantage
    raw = fetch_historical_chain(symbol, date=None)
    df_raw = normalize_chain_json(raw)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    chain_path = DATA_DIR / f"{symbol}_chain_{stamp}.parquet"
    df_raw.to_parquet(chain_path, index=False)

    # 2) Features -> IV
    df_feat = add_time_and_moneyness(df_raw)  # uses T≈(expiry - yesterday)/365
    feat_path = DATA_DIR / f"{symbol}_chain_{stamp}_feat.parquet"
    df_feat.to_parquet(feat_path, index=False)

    df_iv = compute_iv_for_df(df_feat)
    feat_iv_path = DATA_DIR / f"{symbol}_chain_{stamp}_feat_iv.parquet"
    df_iv.to_parquet(feat_iv_path, index=False)

    # 3) Surface + anomaly CSVs (optional, but nice to have)
    k_grid, T_grid, IV, W, meta = build_surface(df_iv, n_k=61, max_exps=8, method="svi")
    np.savez_compressed(DATA_DIR / f"{symbol}_chain_{stamp}_feat_iv_surface.npz",
                        k_grid=k_grid, T_grid=T_grid, IV=IV, W=W)
    pd.DataFrame(meta).to_csv(DATA_DIR / f"{symbol}_chain_{stamp}_feat_iv_surface_meta.csv", index=False)
    calendar_violations(k_grid, T_grid, W, tol=1e-6).to_csv(
        DATA_DIR / f"{symbol}_chain_{stamp}_feat_iv_calendar_violations.csv", index=False
    )
    convexity_violations(df_iv, max_exps=8, tol=0.0).to_csv(
        DATA_DIR / f"{symbol}_chain_{stamp}_feat_iv_convexity_violations.csv", index=False
    )

    print(f"✅ Wrote {feat_iv_path}")

if __name__ == "__main__":
    run()