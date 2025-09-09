# https://iv-surface-monitor.onrender.com
# IV Surface Monitor

IV Surface Monitor is a small quant toolkit + dashboard for options IV. It computes IV from quotes, derives moneyness and time to expiry, fits SVI or uses raw interpolation to render a 3D surface, and surfaces potential arbitrage violations like non-monotone total variance and non-convex calls. Ideal for learning, monitoring, and rapid IV diagnostics.

## Features
- Pull historical option chains from Alpha Vantage’s **HISTORICAL_OPTIONS** endpoint.
- Normalize raw contracts and engineer features (time to expiry, forward price,
  log-moneyness, etc.).
- Compute implied volatility via the Black forward model.
- Build IV surfaces by linear interpolation or per-expiry SVI fitting.
- Detect calendar and convexity violations in total variance or call prices.
- Interactive dashboard (Plotly Dash) for smiles, surfaces, and anomaly reports.

## Installation
```bash
git clone <this-repo-url>
cd IV_surface_monitor
python -m venv .venv
source .venv/bin/activate       # or use your platform’s equivalent
pip install -r requirements.txt
