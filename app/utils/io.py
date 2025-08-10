import pandas as pd
from pathlib import Path

def read_table(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)
