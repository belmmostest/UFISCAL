"""Convenience launcher for the DGCE Policy Simulator dashboard.

Run :

    python scripts/run_dashboard.py [--port 5000]

This starts the Flask API (via ``api.api_app_final``) and automatically opens
the browser to the front-end dashboard at ``/dashboard/index.html``.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgce_model.api.api_app_final import app  # noqa: E402 – import after path is set inside module


def _open_browser(port: int) -> None:
    """Wait briefly, then open default web browser."""

    time.sleep(1.5)  # give Flask a moment to start
    url = f"http://localhost:{port}/dashboard/index.html"
    try:
        webbrowser.open_new_tab(url)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Could not open browser automatically: {exc}\nVisit {url} manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DGCE dashboard")
    parser.add_argument("--port", type=int, default=8080, help="TCP port")
    args = parser.parse_args()

    # Spawn browser opener thread
    threading.Thread(target=_open_browser, args=(args.port,), daemon=True).start()

    print("🚀 Starting DGCE API & Dashboard → http://localhost:%d" % args.port)
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":  # pragma: no cover
    main()
