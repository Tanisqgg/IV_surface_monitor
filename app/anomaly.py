# app/anomaly.py
import numpy as np
import pandas as pd
from app.iv import black_call

def calendar_violations(k_grid, T_grid, W_grid, tol=1e-6):
    """
    total variance should be non-decreasing in T for fixed k.
    Returns DataFrame of violations with (k, T_small, T_large, w_small, w_large, gap).
    """
    rows = []
    for j, k in enumerate(k_grid):
        w = W_grid[:, j]
        # skip columns with too many NaNs
        if np.isnan(w).sum() > len(w) - 2:
            continue
        valid = ~np.isnan(w)
        Ts = T_grid[valid]; ws = w[valid]
        for a in range(len(ws) - 1):
            if ws[a+1] + tol < ws[a]:
                rows.append({
                    "k": float(k),
                    "T_small": float(Ts[a]),
                    "T_large": float(Ts[a+1]),
                    "w_small": float(ws[a]),
                    "w_large": float(ws[a+1]),
                    "gap": float(ws[a] - ws[a+1])
                })
    return pd.DataFrame(rows)

def convexity_violations_for_expiry(expiry_df: pd.DataFrame, tol=0.0):
    """
    Call price should be convex in K. We reconstruct call mids from quotes if possible,
    else from IV (model price), then check second divided difference >= -tol.
    """
    df = expiry_df.copy()
    # Build call mids from quotes if available
    if {"mid","F","strike","right"}.issubset(df.columns):
        calls = df[df["right"] == "C"][["strike","mid"]].rename(columns={"mid":"C"})
        puts  = df[df["right"] == "P"][["strike","mid"]].rename(columns={"mid":"P"})
        both = calls.merge(puts, on="strike", how="outer")
        K = both["strike"].to_numpy()
        C = both["C"].to_numpy()
        P = both["P"].to_numpy()
        F = float(df["F"].dropna().median()) if "F" in df else np.nan
        # parity fill for missing C
        if np.isfinite(F):
            C = np.where(np.isfinite(C), C, (P + (F - K)))
        mask = np.isfinite(K) & np.isfinite(C)
        K, C = K[mask], C[mask]
    else:
        K = df["strike"].to_numpy()
        F = float(df["F"].dropna().median()) if "F" in df else np.nan
        sigma = df["iv_est"].to_numpy()
        T = float(df["T"].dropna().median())
        C = np.array([black_call(F, k, T, s) if np.isfinite(s) else np.nan for k, s in zip(K, sigma)])
        mask = np.isfinite(K) & np.isfinite(C)
        K, C = K[mask], C[mask]

    if K.size < 3:
        return pd.DataFrame(columns=["strike1","strike2","strike3","violation"])

    order = np.argsort(K); K = K[order]; C = C[order]
    viols = []
    # second divided difference for non-uniform spacing
    for i in range(len(K) - 2):
        K1, K2, K3 = K[i], K[i+1], K[i+2]
        C1, C2, C3 = C[i], C[i+1], C[i+2]
        if not np.isfinite(C1) or not np.isfinite(C2) or not np.isfinite(C3):
            continue
        d1 = (C2 - C1) / (K2 - K1)
        d2 = (C3 - C2) / (K3 - K2)
        sec = 2 * (d2 - d1) / (K3 - K1)  # approx f''(K2)
        if sec < -tol:  # negative curvature => violates convexity
            viols.append({"strike1":K1,"strike2":K2,"strike3":K3,"violation":float(sec)})
    return pd.DataFrame(viols)

def convexity_violations(df: pd.DataFrame, max_exps=8, tol=0.0):
    out = []
    for (exp, T), grp in df[df["T"] > 1e-6].groupby(["expiration","T"]):
        if len(out) >= max_exps:  # only nearest few maturities to keep it light
            break
        v = convexity_violations_for_expiry(grp, tol=tol)
        v["expiration"] = exp; v["T"] = float(T)
        if not v.empty:
            out.append(v)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["strike1","strike2","strike3","violation","expiration","T"])