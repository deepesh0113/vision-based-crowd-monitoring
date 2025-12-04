# server.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1) Dashboard backend (main.py)
from main import app as dashboard_app

# 2) Photo crowd counting backend (app.py)
from app import app as photo_app

# 3) Video → crowd txt backend (analytics.py)
from analytics import app as analytics_app
# server.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# import the 3 existing FastAPI apps
from main import app as dashboard_app      # dashboard backend
from app import app as photo_app          # photo crowd counting backend
from analytics import app as analytics_app  # video analytics backend


app = FastAPI(title="Unified Crowd Backend")

# Global CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000", "http://127.0.0.1:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount them at prefixes
app.mount("/dashboard", dashboard_app)
app.mount("/photo", photo_app)
app.mount("/analytics", analytics_app)


app = FastAPI(title="Unified Crowd Backend")

# Global CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or [ "http://localhost:3000", "http://127.0.0.1:3000" ]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount individual apps at their prefixes
app.mount("/dashboard", dashboard_app)
app.mount("/photo", photo_app)
app.mount("/analytics", analytics_app)
