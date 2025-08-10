# app/iv.py
# Implied-vol utilities without SciPy: pure-Python N(·) using math.erf
from __future__ import annotations

import math
from typing import Iterable, Tuple
import pandas as pd


# -------- Normal PDF/CDF (no SciPy) --------
def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    # Abramowitz–Stegun via error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -------- Black (forward) model --------
def black_call(F: float, K: float, T: float, sigma: float, D: float = 1.0) -> float:
    """
    Forward Black call with discount factor D (≈ e^{-rT}). For equities you can take D≈1.
    """
    if T <= 0.0 or sigma <= 0.0:
        return D * max(F - K, 0.0)
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    return D * (F * norm_cdf(d1) - K * norm_cdf(d2))


def vega(F: float, K: float, T: float, sigma: float, D: float = 1.0) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol
    return D * F * norm_pdf(d1) * math.sqrt(T)


# -------- IV solver (Newton with bisection fallback) --------
def implied_vol_from_mid(
    mid: float,
    F: float,
    K: float,
    T: float,
    right: str = "C",
    D: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[float, bool, int]:
    """
    Compute implied vol from a mid price using Black (forward).
    If right == 'P', convert to call via parity: C = P + D*(F - K).

    Returns: (sigma, ok, iterations)
    """
    if not (math.isfinite(mid) and math.isfinite(F) and math.isfinite(K)) or T <= 0.0:
        return (math.nan, False, 0)

    price = float(mid)
    r = (right or "C").upper()
    if r == "P":
        price = price + D * (F - K)

    # Trivial bounds
    intrinsic = D * max(F - K, 0.0)
    upper = D * F  # very loose
    if price <= intrinsic + 1e-12 or price >= upper or not math.isfinite(price):
        return (math.nan, False, 0)

    # Initial guess (Brenner–Subrahmanyam-ish), clamped
    try:
        sigma = max(1e-4, min(3.0, math.sqrt(2.0 * math.pi / T) * price / (D * F)))
    except Exception:
        sigma = 0.2 if T > 0.03 else 0.4

    lo, hi = 1e-4, 5.0
    it = 0
    for it in range(1, max_iter + 1):
        c = black_call(F, K, T, sigma, D)
        diff = c - price
        if abs(diff) < tol:
            return (sigma, True, it)
        v = vega(F, K, T, sigma, D)
        if v < 1e-8 or not math.isfinite(v):
            break
        sigma -= diff / v
        if not math.isfinite(sigma) or sigma <= lo or sigma >= hi:
            break

    # Bisection fallback (robust)
    a, b = lo, hi
    m = (a + b) * 0.5
    for it2 in range(1, 100):
        m = 0.5 * (a + b)
        c = black_call(F, K, T, m, D)
        d = c - price
        if abs(d) < tol:
            return (m, True, it + it2)
        if d > 0.0:
            b = m
        else:
            a = m
    return (m, False, it + 100)


# -------- DataFrame helper --------
def compute_iv_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      - iv_est (float)
      - iv_ok (bool)
      - iv_iters (int)
    Expects columns: mid, F, strike, T, and optionally right.
    """
    need = {"mid", "F", "strike", "T"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"compute_iv_for_df: missing columns {missing}")

    out = df.copy()
    ivs, oks, iters = [], [], []

    # Use itertuples for speed & to avoid dtype surprises
    has_right = "right" in out.columns
    for row in out.itertuples(index=False):
        mid = getattr(row, "mid")
        F = getattr(row, "F")
        K = getattr(row, "strike")
        T = getattr(row, "T")
        r = getattr(row, "right") if has_right else "C"
        s, ok, it = implied_vol_from_mid(mid, F, K, T, r)
        ivs.append(s); oks.append(ok); iters.append(it)

    out["iv_est"] = ivs
    out["iv_ok"] = oks
    out["iv_iters"] = iters
    return out