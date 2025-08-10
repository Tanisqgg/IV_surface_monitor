import argparse
from pathlib import Path
from app.utils.io import read_table
from app.features import add_time_and_moneyness

def main(inp_path: str):
    p = Path(inp_path)
    df = read_table(p)
    df2 = add_time_and_moneyness(df)

    out = p.with_name(p.stem + "_feat" + p.suffix)  # preserves .csv or .parquet
    if out.suffix.lower() == ".csv":
        df2.to_csv(out, index=False)
    else:
        df2.to_parquet(out, index=False)
    print(f"Wrote {out}  rows={len(df2)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to raw chain (.csv or .parquet)")
    args = ap.parse_args()
    main(args.inp)