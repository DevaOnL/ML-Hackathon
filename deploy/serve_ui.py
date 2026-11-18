"""Optional Laptop-2 UI server. Never serves the repository or route files.

Run from the repository root: python -B deploy/serve_ui.py --port 8090
Prefer the backend's same-origin UI for the deployed public website.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.static import public_asset


class UIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._serve(head=False)

    def do_HEAD(self):
        self._serve(head=True)

    def _serve(self, *, head):
        path = unquote(urlsplit(self.path).path)
        asset = public_asset(ROOT / "frontend", path[1:] if path.startswith("/") else path)
        if asset is None:
            self.send_error(404, "This resource is not public.")
            return
        target, media_type = asset
        with target.open("rb") as source:
            self.send_response(200)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if not head:
                shutil.copyfileobj(source, self.wfile)

    def log_message(self, format, *args):
        # Avoid logging attacker-controlled paths or request lines.
        print("pathsnap_ui_request", self.command, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    server = ThreadingHTTPServer((args.host, args.port), UIHandler)
    print(f"PathSnap UI available on http://{args.host}:{args.port}; enter the API base URL in API settings.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
