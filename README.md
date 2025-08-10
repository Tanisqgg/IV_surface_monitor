# https://iv-surface-monitor.onrender.com
# IV Surface Monitor

Tools for building and monitoring option-implied volatility surfaces.  
The project ingests historical option chains, computes implied volatility,
constructs an IV surface, detects arbitrage anomalies, and exposes a
Dash-based dashboard for exploration.

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
