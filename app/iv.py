# app/iv.py
import numpy as np
from math import sqrt, log
from scipy.stats import norm

def black_call(F, K, T, sigma, D=1.0):
    # Black (forward) call with discount D (we’ll use D≈1 for equities)
    if T <= 0 or sigma <= 0:
        return D * max(F - K, 0.0)
    vol = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2))

def vega(F, K, T, sigma, D=1.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    vol = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / vol
    return D * F * norm.pdf(d1) * sqrt(T)

def implied_vol_from_mid(mid, F, K, T, right="C", D=1.0, tol=1e-6, max_iter=50):
    """
    Compute IV from mid price using Black (forward). If put, convert to call via parity: C = P + D*(F-K).
    Returns (sigma, ok, iters).
    """
    if not np.isfinite(mid) or not np.isfinite(F) or not np.isfinite(K) or T <= 0:
        return (np.nan, False, 0)

    # Convert put price to equivalent call price
    right = (right or "C").upper()
    price = float(mid)
    if right == "P":
        price = price + D * (F - K)

    # Intrinsic and trivial bounds
    intrinsic = D * max(F - K, 0.0)
    upper = D * F  # very loose
    if price <= intrinsic + 1e-10 or price >= upper:
        return (np.nan, False, 0)

    # Initial guess: Brenner–Subrahmanyam-ish; fallback constants
    try:
        sigma = max(1e-4, min(3.0, sqrt(2*np.pi/T) * price / (D*F)))
    except Exception:
        sigma = 0.2 if T > 0.03 else 0.4

    lo, hi = 1e-4, 5.0
    it = 0
    for it in range(1, max_iter+1):
        c = black_call(F, K, T, sigma, D)
        diff = c - price
        if abs(diff) < tol:
            return (sigma, True, it)
        v = vega(F, K, T, sigma, D)
        if v < 1e-8 or not np.isfinite(v):
            break
        sigma -= diff / v
        if sigma <= lo or sigma >= hi or not np.isfinite(sigma):
            break

    # Fallback: bisection
    a, b = lo, hi
    for it2 in range(1, 80):
        m = 0.5 * (a + b)
        c = black_call(F, K, T, m, D)
        if abs(c - price) < tol:
            return (m, True, it + it2)
        if c > price:
            b = m
        else:
            a = m
    return (m, False, it + 80)

def compute_iv_for_df(df):
    """Add columns: iv_est, iv_ok, iv_iters. Expects columns: mid, F, strike, T, right."""
    out = df.copy()
    sigmas = []
    oks = []
    iters = []
    for mid, F, K, T, r in zip(out["mid"], out["F"], out["strike"], out["T"], out.get("right", ["C"]*len(out))):
        s, ok, it = implied_vol_from_mid(mid, F, K, T, r)
        sigmas.append(s); oks.append(ok); iters.append(it)
    out["iv_est"] = sigmas
    out["iv_ok"] = oks
    out["iv_iters"] = iters
    return out