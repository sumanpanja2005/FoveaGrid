"""
FoveaGrid Visualization Module: FastAPI + WebSocket Server and Three.js 3D Web Frontend Dashboard.
"""

from .backend import app, start_server
from .websocket import ConnectionManager

__all__ = ["app", "start_server", "ConnectionManager"]
