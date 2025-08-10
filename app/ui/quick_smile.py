import argparse
from pathlib import Path
import plotly.express as px
from app.utils.io import read_table
from app.features import add_time_and_moneyness

def main(path_str: str):
    p = Path(path_str)
    df = read_table(p)

    # Auto-add features if missing
    if not {"T", "k"}.issubset(df.columns):
        df = add_time_and_moneyness(df)

    # Prefer our computed IV
    iv_col = "iv_est" if "iv_est" in df.columns else ("implied_vol" if "implied_vol" in df.columns else None)
    if not iv_col:
        raise SystemExit("No IV column yet. Run:  python -m app.compute_iv --in <*_feat file>")

    df = df[(df["T"] > 1e-6) & df[iv_col].between(0, 5)].copy()
    first_exp = df.sort_values("T")["expiration"].iloc[0]
    sm = df[df["expiration"] == first_exp].copy()

    hover = [c for c in ["strike","right","bid","ask","open_interest","T"] if c in sm.columns]
    fig = px.scatter(sm, x="k", y=iv_col, hover_data=hover)

    # --- fixed title ---
    if "symbol" in sm.columns and len(sm) > 0:
        sym = str(sm["symbol"].iloc[0])
    else:
        sym = p.stem.split("_")[0]
    fig.update_layout(title=f"Smile for {sym}  expiry {first_exp}")
    # --------------------

    fig.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file","-f", required=True, help="Path to raw or *_feat(_iv) file (.csv or .parquet)")
    args = ap.parse_args()
    main(args.file)