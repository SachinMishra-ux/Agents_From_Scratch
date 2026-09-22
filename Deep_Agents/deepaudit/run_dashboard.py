"""One-click launcher for the DeepAudit-AI Streaming Dashboard."""

import os
import sys
from pathlib import Path

# Add project root and virtual environment site-packages to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SITE_PACKAGES = ROOT_DIR / ".venv" / "lib" / "python3.12" / "site-packages"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if SITE_PACKAGES.exists() and str(SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(SITE_PACKAGES))

import uvicorn
from Deep_Agents.deepaudit.ui.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n========================================================")
    print(f"🚀 Launching DeepAudit-AI Live Streaming Dashboard")
    print(f"🌐 Access the UI at: http://localhost:{port}")
    print(f"========================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
