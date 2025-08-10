# app/compute_iv.py
import argparse
from pathlib import Path
from app.utils.io import read_table
from app.iv import compute_iv_for_df

def main(inp_path: str):
    p = Path(inp_path)
    df = read_table(p)
    need = {"mid","F","strike","T"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Missing {need - set(df.columns)}; run make_features first.")

    outdf = compute_iv_for_df(df)
    out = p.with_name(p.stem + "_iv" + p.suffix)
    if out.suffix.lower() == ".csv":
        outdf.to_csv(out, index=False)
    else:
        outdf.to_parquet(out, index=False)
    ok_rate = outdf["iv_ok"].mean() if len(outdf) else 0
    print(f"Wrote {out} | success={ok_rate:.2%} | rows={len(outdf)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to *_feat file (.csv or .parquet)")
    args = ap.parse_args()
    main(args.inp)