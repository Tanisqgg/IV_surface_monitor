# app/features.py
import pandas as pd
import numpy as np
import datetime as dt

def add_time_and_moneyness(df: pd.DataFrame, trade_date: dt.date|None=None) -> pd.DataFrame:
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

    # Normalize right/type to 'C'/'P'
    if right:
        df["right"] = df[right].astype(str).str.upper().str[0]
    else:
        df["right"] = np.nan  # we’ll still have fallbacks

    # Mid price (prefer bid/ask, else last)
    if "bid" in df and "ask" in df:
        mid = np.where((df["bid"] > 0) & (df["ask"] > 0), (df["bid"] + df["ask"]) / 2.0, np.nan)
    else:
        mid = np.full(len(df), np.nan)
    if "last" in df:
        mid = np.where(np.isfinite(mid), mid, df["last"])
    df["mid"] = mid

    # Time to expiry (ACT/365)
    if trade_date is None:
        trade_date = dt.date.today() - dt.timedelta(days=1)
    T_days = (pd.Series(df["expiration"]) - trade_date).apply(lambda x: x.days if pd.notnull(x) else np.nan)
    df["T"] = np.clip(pd.to_numeric(T_days, errors="coerce"), 0, None) / 365.0

    # ---- Forward estimate per expiry ----
    forwards = {}
    for exp_date, grp in df.groupby("expiration"):
        if pd.isnull(exp_date) or len(grp) == 0:
            continue
        F_est = None

        # Parity from matched call/put strikes
        if grp["right"].notna().any():
            calls = grp[grp["right"] == "C"][["strike","mid"]].rename(columns={"mid":"C"}).dropna()
            puts  = grp[grp["right"] == "P"][["strike","mid"]].rename(columns={"mid":"P"}).dropna()
            both = calls.merge(puts, on="strike", how="inner").dropna()
            if len(both) >= 5:
                both["F_est"] = both["C"] - both["P"] + both["strike"]
                if both["F_est"].notna().any():
                    F_est = float(np.nanmedian(both["F_est"]))

        # Fallback 1: call with delta closest to 0.5
        if (F_est is None or not np.isfinite(F_est)) and "delta" in df.columns:
            calls_with_delta = grp[(grp["right"] == "C") & grp["delta"].notna()][["strike","delta"]]
            if len(calls_with_delta):
                at_idx = (calls_with_delta["delta"] - 0.5).abs().idxmin()
                F_est = float(calls_with_delta.loc[at_idx, "strike"])

        # Fallback 2: strike with max total OI
        if (F_est is None or not np.isfinite(F_est)) and "open_interest" in df.columns:
            oi_by_strike = grp.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
            if len(oi_by_strike):
                F_est = float(oi_by_strike.index[0])

        # Fallback 3: median strike
        if F_est is None or not np.isfinite(F_est):
            if grp["strike"].notna().any():
                F_est = float(np.nanmedian(grp["strike"]))

        if F_est is not None and np.isfinite(F_est):
            forwards[exp_date] = F_est

    df["F"] = df["expiration"].map(forwards)
    df["k"] = np.log(df["strike"] / df["F"])
    return df.dropna(subset=["T","k","F","mid"]).copy()