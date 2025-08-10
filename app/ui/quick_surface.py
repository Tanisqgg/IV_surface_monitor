# app/ui/quick_surface.py
import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

def main(npz_path: str):
    p = Path(npz_path)
    with np.load(p) as z:
        k_grid = z["k_grid"]; T_grid = z["T_grid"]; IV = z["IV"]
    # Plotly expects 2D X,Y meshes
    K, TT = np.meshgrid(k_grid, T_grid)
    fig = go.Figure(data=[go.Surface(x=K, y=TT, z=IV)])
    fig.update_layout(
        title=f"IV Surface ({p.stem.replace('_surface','')})",
        scene=dict(
            xaxis_title="log-moneyness k",
            yaxis_title="T (years)",
            zaxis_title="IV"
        ),
        height=700
    )
    fig.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to *_surface.npz")
    args = ap.parse_args()
    main(args.npz)