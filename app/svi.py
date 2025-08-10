import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize

# Raw SVI (Gatheral) on total variance w = sigma^2 * T
def svi_w(k, a, b, rho, m, s):
    # s for "sigma" param (often called 'sigma' in SVI papers; renamed to s to avoid confusion)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))

@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float
    ok: bool
    n: int
    loss: float

def _init_guess(k: np.ndarray, w: np.ndarray) -> Tuple[float,float,float,float,float]:
    kmed = float(np.nanmedian(k))
    kstd = float(np.nanstd(k)) if np.nanstd(k) > 1e-6 else 0.1
    wmin = float(np.nanmin(w))
    wspan = float(np.nanpercentile(w, 95) - np.nanpercentile(w, 5))
    a0 = max(1e-6, wmin * 0.8)
    b0 = max(1e-6, 0.5 * (wspan / (kstd + 1e-6)))
    rho0 = 0.0
    m0 = kmed
    s0 = max(1e-3, kstd)
    return a0, b0, rho0, m0, s0

def fit_svi_smile(k: np.ndarray, w: np.ndarray) -> SVIParams:
    # Clean
    mask = np.isfinite(k) & np.isfinite(w)
    k = k[mask]; w = w[mask]
    n = len(k)
    if n < 6:
        return SVIParams(0,0,0,0,0,False,n,1e9)

    a0,b0,rho0,m0,s0 = _init_guess(k,w)

    # Weighted robust loss (Huber-like): downweight outliers
    def loss(theta):
        a,b,rho,m,s = theta
        # soft penalties to keep in feasible region & butterfly-free
        pen = 0.0
        # bounds-ish
        if b <= 0: pen += (1 - np.tanh(10*b))**2
        if s <= 0: pen += (1 - np.tanh(10*s))**2
        if abs(rho) >= 1: pen += (abs(rho) - 0.999)**2 * 1e3
        # minimum total variance >= 0 : a + b*s*sqrt(1-rho^2) >= 0
        pen += max(0.0, -(a + b * s * np.sqrt(max(0.0, 1 - rho*rho))))**2 * 1e3

        w_hat = svi_w(k, a,b,rho,m,s)
        r = w_hat - w
        # Huber-ish
        absr = np.abs(r)
        hub = np.where(absr < 0.02, 0.5 * r*r, 0.02*(absr - 0.01))
        return float(np.nanmean(hub) + pen)

    bounds = [
        (1e-9, 10.0),    # a
        (1e-6, 10.0),    # b
        (-0.999, 0.999), # rho
        (min(k)-1.0, max(k)+1.0),  # m
        (1e-6, 5.0)      # s
    ]
    x0 = np.array([a0,b0,rho0,m0,s0], dtype=float)

    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter":1000})
    a,b,rho,m,s = res.x
    ok = res.success and np.isfinite(res.fun)
    return SVIParams(a,b,rho,m,s, ok, n, float(res.fun))

def evaluate_svi_iv_on_grid(k_grid: np.ndarray, T: float, p: SVIParams) -> np.ndarray:
    w = svi_w(k_grid, p.a, p.b, p.rho, p.m, p.s)
    w = np.clip(w, 1e-8, None)
    iv = np.sqrt(w / max(T, 1e-8))
    return iv