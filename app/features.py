import pandas as pd
import numpy as np
import datetime as dt
import re
from typing import Optional

DATE_RE = re.compile(r"(\d{8})")  # e.g., 20250809

def infer_trade_date(df: pd.DataFrame, source_path: Optional[str] = None) -> dt.date:
    """
    Try to infer the trade/quote date from dataframe columns or filename.
    Priority:
      1) columns: 'trade_date', 'quote_date', 'date' (date-like)
      2) filename pattern: *_chain_YYYYMMDD.*
    """
    for cand in ("trade_date", "quote_date", "date"):
        if cand in df.columns:
            d = pd.to_datetime(df[cand], errors="coerce").dropna()
            if len(d):
                return d.iloc[0].date()

    if source_path:
        m = DATE_RE.search(str(source_path))
        if m:
            y, mth, day = m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8]
            return dt.date(int(y), int(mth), int(day))

    raise ValueError(
        "Cannot infer trade date. "
        "Provide trade_date explicitly or ensure filename contains YYYYMMDD (e.g. *_chain_20250809.parquet)."
    )

def add_time_and_moneyness(
    df: pd.DataFrame,
    trade_date: Optional[dt.date] = None,
    source_path: Optional[str] = None,
    max_spread_pct: float = 0.25,        # 25% relative to ask (tunable)
    use_last_if_inside_book: bool = False
) -> pd.DataFrame:
    """
    Compute: expiration(date), strike, mid (sanitized), T (ACT/365), forward F (robust), log-moneyness k.
    - trade_date is REQUIRED for determinism. If not provided, we try to infer it; else we raise.
    - We do NOT clip T<0 to 0. We drop non-positive T rows instead.
    - Mid: prefer good bid/ask; do not fall back to 'last' unless enabled AND inside the book.
    - Forward F: robust median of parity C-P+K on matched strikes after filtering.
    """
    df = df.copy()

    # Case-insensitive column lookup
    cmap = {c.lower(): c for c in df.columns}
    def col(name, default=None): return cmap.get(name, default)

    # Map likely column names
    exp = col("expiration") or col("expiration_date") or col("expiry") or col("maturity")
    strike = col("strike") or col("strike_price")
    bid = col("bid"); ask = col("ask"); last = col("last") or col("price")
    right = col("right") or col("type")
    oi = col("open_interest") or col("openinterest") or col("openinterestcount")
    delta = col("delta")

    if not exp or not strike:
        raise ValueError(f"Missing expiration/strike. Columns: {list(df.columns)}")

    # Parse types safely
    df["expiration"] = pd.to_datetime(df[exp], errors="coerce").dt.date
    df["strike"] = pd.to_numeric(df[strike], errors="coerce")
    if bid:  df["bid"]  = pd.to_numeric(df[bid],  errors="coerce")
    if ask:  df["ask"]  = pd.to_numeric(df[ask],  errors="coerce")
    if last: df["last"] = pd.to_numeric(df[last], errors="coerce")
    if oi:   df["open_interest"] = pd.to_numeric(df[oi], errors="coerce")
    if delta:df["delta"] = pd.to_numeric(df[delta], errors="coerce")
    if right:
        df["right"] = df[right].astype(str).str.upper().str[0]
    else:
        df["right"] = np.nan

    # --- TRADE DATE (deterministic) ---
    if trade_date is None:
        trade_date = infer_trade_date(df, source_path=source_path)

    # Time to expiry (ACT/365), *no clipping*; drop T<=0 later
    T_days = (pd.Series(df["expiration"]) - trade_date).apply(lambda x: x.days if pd.notnull(x) else np.nan)
    df["T"] = pd.to_numeric(T_days, errors="coerce") / 365.0

    # --- MID PRICE (hygiene first) ---
    mid = np.full(len(df), np.nan)
    if "bid" in df and "ask" in df:
        valid = np.isfinite(df["bid"]) & np.isfinite(df["ask"]) & (df["bid"] > 0) & (df["ask"] > df["bid"])
        spread = (df["ask"] - df["bid"]) / np.clip(df["ask"], 1e-12, None)
        valid &= (spread <= max_spread_pct)
        mid = np.where(valid, 0.5 * (df["bid"] + df["ask"]), np.nan)

        if use_last_if_inside_book and "last" in df:
            ok_last = (
                np.isfinite(df["last"]) &
                np.isfinite(df["bid"]) & np.isfinite(df["ask"]) &
                (df["last"] >= df["bid"]) & (df["last"] <= df["ask"])
            )
            mid = np.where(np.isfinite(mid), mid, df["last"].where(ok_last))

    elif "last" in df:
        # No quotes; only take 'last' if positive
        mid = np.where(np.isfinite(df["last"]) & (df["last"] > 0), df["last"], np.nan)

    df["mid"] = mid

    # --- FORWARD F (robust, from matched C/P on same strike) ---
    forwards = {}
    for exp_date, grp in df.groupby("expiration"):
        if pd.isnull(exp_date) or len(grp) == 0:
            continue

        F_est = None
        # Use only clean mids for parity
        if grp["right"].notna().any():
            calls = grp[(grp["right"] == "C") & np.isfinite(grp["mid"])][["strike","mid"]].rename(columns={"mid":"C"})
            puts  = grp[(grp["right"] == "P") & np.isfinite(grp["mid"])][["strike","mid"]].rename(columns={"mid":"P"})
            both = calls.merge(puts, on="strike", how="inner").dropna()
            if len(both) >= 5:
                both["F_est"] = both["C"] - both["P"] + both["strike"]
                # robust filter: trim extreme tails or MAD filter
                vals = both["F_est"].to_numpy()
                med = np.nanmedian(vals)
                mad = np.nanmedian(np.abs(vals - med))
                if np.isfinite(med):
                    if mad > 0:
                        keep = np.abs(vals - med) <= 5 * mad
                        vals = vals[keep]
                    else:
                        # fallback: 5-95% trimming
                        lo, hi = np.nanpercentile(vals, [5, 95])
                        vals = vals[(vals >= lo) & (vals <= hi)]
                    if len(vals) >= 3:
                        F_est = float(np.nanmedian(vals))

        # Fallback 1: call with delta ~ 0.5
        if (F_est is None or not np.isfinite(F_est)) and "delta" in grp.columns:
            calls_with_delta = grp[(grp["right"] == "C") & grp["delta"].notna()][["strike","delta"]]
            if len(calls_with_delta):
                at_idx = (calls_with_delta["delta"] - 0.5).abs().idxmin()
                F_est = float(calls_with_delta.loc[at_idx, "strike"])

        # Fallback 2: strike with max total OI
        if (F_est is None or not np.isfinite(F_est)) and "open_interest" in grp.columns:
            oi_by_strike = grp.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
            if len(oi_by_strike):
                F_est = float(oi_by_strike.index[0])

        # Fallback 3: median strike
        if F_est is None or not np.isfinite(F_est):
            if grp["strike"].notna().any():
                F_est = float(np.nanmedian(grp["strike"]))

        if F_est is not None and np.isfinite(F_est) and F_est > 0:
            forwards[exp_date] = F_est

    df["F"] = df["expiration"].map(forwards)
    df["k"] = np.log(df["strike"] / df["F"])

    # Final clean: drop rows with bad essentials
    df = df.dropna(subset=["expiration","strike","mid","T","F","k"]).copy()
    df = df[df["T"] > 0]  # deterministic: no clipping, just drop T<=0

    return df