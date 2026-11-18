#!/usr/bin/env python3
"""Capture sanitized, timestamped local/public hosting status without changing services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from launch_service import HOSTNAME, ORIGIN

HERE = Path(__file__).resolve().parent
PUBLIC = "https://" + HOSTNAME


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def probe(url: str, *, json_body: bool = False) -> dict:
    opener = build_opener(ProxyHandler({}), NoRedirect())
    request = Request(url, headers={"User-Agent": "PathSnap-hosting-status/1"})
    try:
        with opener.open(request, timeout=12) as response:
            payload = response.read(2 * 1024 * 1024)
            result = {"url": url, "http_status": response.status,
                      "content_type": response.headers.get("Content-Type"),
                      "tls_verified": url.startswith("https:"),
                      "bytes": len(payload), "response_sha256": hashlib.sha256(payload).hexdigest()}
            if json_body:
                data = json.loads(payload)
                result["health"] = {key: data.get(key) for key in ("status", "model_loaded", "model_metadata_loaded",
                                             "model_hash", "match_config_hash", "trip_count", "dataset_role")}
                result["health"]["error_codes"] = [item.get("code") for item in data.get("errors", [])]
            else:
                result["pathsnap_title_present"] = b"PathSnap" in payload
            return result
    except HTTPError as exc:
        return {"url": url, "http_status": exc.code, "tls_verified": url.startswith("https:"),
                "error": "http_error_or_redirect_not_followed"}
    except URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            kind = "tls_certificate_verification_failed"
        elif isinstance(exc.reason, ssl.SSLError):
            kind = "tls_sni_unrecognized_name" if getattr(exc.reason, "reason", None) == "TLSV1_UNRECOGNIZED_NAME" else "tls_handshake_failed"
        else:
            kind = "network_connection_failed"
        return {"url": url, "http_status": None, "tls_verified": False, "error": kind}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"url": url, "http_status": None, "tls_verified": False, "error": "invalid_response_or_connection_failed"}


def unit_status(name: str) -> dict:
    result = subprocess.run(["systemctl", "--user", "show", name,
                             "--property=Id,LoadState,ActiveState,SubState,MainPID,Restart,NRestarts,UnitFileState"],
                            text=True, capture_output=True, timeout=10)
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values if values else {"Id": name, "query_failed": True}


def dns_records() -> dict:
    answer = {}
    for record_type in ("A", "AAAA", "CNAME"):
        result = subprocess.run(["dig", "+time=2", "+tries=1", "+short", HOSTNAME, record_type],
                                text=True, capture_output=True, timeout=5)
        answer[record_type] = result.stdout.strip().splitlines() if result.returncode == 0 else None
    return answer


def capture() -> dict:
    cert_paths = [Path.home() / ".cloudflared/cert.pem", Path.home() / ".config/cloudflared/cert.pem",
                  Path("/etc/cloudflared/cert.pem"), Path("/usr/local/etc/cloudflared/cert.pem")]
    cert_available = any(path.is_file() and os.access(path, os.R_OK) for path in cert_paths)
    config = Path.home() / ".config/path-snap"
    credentials = (config / "tunnel-credentials.json").is_file()
    public_health = probe(PUBLIC + "/health", json_body=True)
    public_ui = probe(PUBLIC + "/")
    local_health = probe(ORIGIN + "/health", json_body=True)
    local_ui = probe(ORIGIN + "/")
    linger = subprocess.run(["loginctl", "show-user", str(os.getuid()), "--property=Linger", "--value"],
                            text=True, capture_output=True, timeout=10)
    blocker = None if cert_available or credentials else (
        "No Cloudflare origin certificate, scoped API-token authorization, or dedicated tunnel credentials were available "
        "in the bounded auth discovery. Cloudflare CLI tunnel-list failed for missing origin cert. "
        "DNS/tunnel creation cannot proceed safely until domain authorization is installed locally; no DNS record was overwritten.")
    return {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
            "hostname": HOSTNAME, "public_url": PUBLIC, "local_origin": ORIGIN,
            "local": {"ui": local_ui, "health": local_health},
            "public": {"ui": public_ui, "health": public_health, "dns_records": dns_records(),
                       "https_site_verified": bool(public_ui.get("http_status") == 200 and public_ui.get("tls_verified")
                                                     and public_ui.get("pathsnap_title_present")
                                                     and public_health.get("health", {}).get("status") == "ready"),
                       "live_match_verified": False,
                       "match_evidence": None},
            "cloudflare": {"origin_authorization_file_available": cert_available, "dedicated_tunnel_credentials_available": credentials,
                           "blocker": blocker, "unrelated_dns_or_services_modified": False},
            "user_services": [unit_status("path-snap-api.service"), unit_status("path-snap-tunnel.service")],
            "persistence": {"linger": linger.stdout.strip() == "yes", "reboot_actually_tested": False},
            "limitations": ["Public live-match success requires separate verify_site.py evidence; status capture does not infer it.",
                            "The provided student-test dataset is development data; hidden evaluator trips are not present."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "evidence/deployment-status.json")
    args = parser.parse_args()
    path = args.output.expanduser().absolute()
    if not path.resolve().is_relative_to(HERE):
        print("Status evidence must remain under infrastructure/.", file=sys.stderr)
        return 1
    try:
        content = json.dumps(capture(), indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(content, end="")
        return 0
    except (OSError, subprocess.SubprocessError):
        print("A bounded status probe failed; no service or DNS changes were attempted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
