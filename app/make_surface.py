# app/make_surface.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from app.utils.io import read_table
from app.surface import build_surface
from app.anomaly import calendar_violations, convexity_violations

def main(inp_path: str):
    p = Path(inp_path)
    df = read_table(p)
    need = {"k","T","iv_est","expiration","F"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Missing {need - set(df.columns)}; run compute_iv first.")

    k_grid, T_grid, IV, W, meta = build_surface(df, n_k=61, max_exps=8)

    # Save surface snapshot (npz + csvs)
    out_npz = p.with_name(p.stem + "_surface.npz")
    np.savez_compressed(out_npz, k_grid=k_grid, T_grid=T_grid, IV=IV, W=W)
    meta_out = p.with_name(p.stem + "_surface_meta.csv")
    pd.DataFrame(meta).to_csv(meta_out, index=False)

    # Anomaly checks
    cal = calendar_violations(k_grid, T_grid, W, tol=1e-6)
    cal_out = p.with_name(p.stem + "_calendar_violations.csv")
    cal.to_csv(cal_out, index=False)

    conv = convexity_violations(df, max_exps=8, tol=0.0)
    conv_out = p.with_name(p.stem + "_convexity_violations.csv")
    conv.to_csv(conv_out, index=False)

    print(f"Surface -> {out_npz}")
    print(f"Meta    -> {meta_out}  (expiries used: {len(meta)})")
    print(f"Calendar violations: {len(cal)} -> {cal_out}")
    print(f"Convexity violations: {len(conv)} -> {conv_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to *_feat_iv (.csv or .parquet)")
    args = ap.parse_args()
    main(args.inp)