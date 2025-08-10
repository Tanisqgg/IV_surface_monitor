import numpy as np
import pandas as pd
from app.svi import fit_svi_smile, evaluate_svi_iv_on_grid

def _per_expiry_interp(smile_df: pd.DataFrame, k_grid: np.ndarray) -> np.ndarray:
    """Linear interp IV(k) on a fixed k_grid for a single expiry. Returns array with NaNs outside range."""
    s = smile_df.dropna(subset=["k","iv_est"]).sort_values("k")
    if len(s) < 3:
        return np.full_like(k_grid, np.nan, dtype=float)
    k = s["k"].to_numpy(); iv = s["iv_est"].to_numpy()
    ivg = np.interp(k_grid, k, iv, left=np.nan, right=np.nan)
    # numpy.interp fills edges with left/right; we want NaNs outside
    ivg[k_grid < k.min()] = np.nan
    ivg[k_grid > k.max()] = np.nan
    return ivg

def build_surface(df: pd.DataFrame, n_k: int = 61, max_exps: int = 8, method: str = "interp"):
    """
    method: "interp" (raw linear interp) or "svi" (fit per expiry)
    """
    need = {"k","T","iv_est","expiration","F"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns: {need - set(df.columns)}")

    # nearest few expiries
    exps = (df[df["T"] > 1e-6]
            .groupby(["expiration","T"], as_index=False)
            .size()
            .sort_values("T")
            .head(max_exps))
    expirations = exps["expiration"].tolist()
    Ts         = exps["T"].to_numpy()

    # global k range
    slice_df = df[df["expiration"].isin(expirations)]
    kmin = float(np.nanpercentile(slice_df["k"], 5))
    kmax = float(np.nanpercentile(slice_df["k"], 95))
    k_grid = np.linspace(kmin, kmax, n_k)

    IV = np.empty((len(Ts), n_k), dtype=float); IV[:] = np.nan
    meta = []

    for i, (exp, T) in enumerate(zip(expirations, Ts)):
        smile = df[(df["expiration"] == exp) & (df["T"].between(T-1e-12, T+1e-12))].copy()
        smile = smile[(smile["iv_est"].between(0.01, 3.0)) & smile["k"].between(kmin-0.2, kmax+0.2)]
        F = float(smile["F"].dropna().median()) if "F" in smile else np.nan

        if method == "svi":
            # fit on total variance
            w_obs = (smile["iv_est"].to_numpy() ** 2) * T
            p = fit_svi_smile(smile["k"].to_numpy(), w_obs)
            iv_line = evaluate_svi_iv_on_grid(k_grid, T, p) if p.ok else np.full_like(k_grid, np.nan)
            meta.append({"expiration": exp, "T": T, "F": F, "n_obs": len(smile), "method":"svi", **p.__dict__})
        else:
            # raw interp
            s = smile.dropna(subset=["k","iv_est"]).sort_values("k")
            if len(s) >= 3:
                iv_line = np.interp(k_grid, s["k"], s["iv_est"], left=np.nan, right=np.nan)
                iv_line[k_grid < s["k"].min()] = np.nan
                iv_line[k_grid > s["k"].max()] = np.nan
            else:
                iv_line = np.full_like(k_grid, np.nan)
            meta.append({"expiration": exp, "T": T, "F": F, "n_obs": len(smile), "method":"interp"})

        IV[i, :] = iv_line

    meta = pd.DataFrame(meta)

    # Fill along T for plotting aesthetics
    for j in range(n_k):
        col = IV[:, j]
        if np.all(np.isnan(col)):
            continue
        IV[:, j] = pd.Series(col).ffill().bfill().to_numpy()

    W = (IV ** 2) * Ts.reshape(-1, 1)
    return k_grid, Ts, IV, W, meta