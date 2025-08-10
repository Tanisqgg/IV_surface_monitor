# app/index.py
# Exposes a WSGI callable for Vercel from your existing Dash app.
from app.ui.dashboard import app as dash_app

# Dash's underlying Flask server:
app = dash_app.server  # <-- Vercel looks for "app" (WSGI callable)

# Optionally set server name for Vercel envs
app.config.update(SERVER_NAME=None)
