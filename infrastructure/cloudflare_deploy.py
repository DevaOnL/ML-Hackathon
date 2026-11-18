#!/usr/bin/env python3
"""Preflight or deploy ONLY ml.d3va.dev through its own path-snap Cloudflare tunnel.

Default is read-only preflight. --apply is explicit; no existing DNS record,
unowned tunnel, Cloudflare account setting or unrelated service is overwritten.
Credentials stay in memory or private files outside the checkout, never stdout.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import stat
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import UUID

from launch_service import (HOSTNAME, ORIGIN, LaunchError, dedicated_config_dir, dedicated_directory,
                            private_file, read_runtime)
from install_user_services import operation_lock, write_private

ZONE_NAME = "d3va.dev"
TUNNEL_NAME = "path-snap"
INSTALLER_ID = "pathsnap-scoped-hosting-v1"
API_BASE = "https://api.cloudflare.com/client/v4"
ROOT = Path(__file__).resolve().parents[1]
HEX_ID = re.compile(r"^[0-9a-f]{32}$")


class DeploymentError(LaunchError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class CloudflareAPI:
    def __init__(self, token: str):
        self._token = token
        self._opener = build_opener(ProxyHandler({}), NoRedirect())

    def call(self, method: str, path: str, body: dict | None = None, **query) -> dict:
        if not path.startswith(("/zones", "/accounts/")) or ".." in path:
            raise DeploymentError("Refusing an unexpected Cloudflare API resource.")
        url = API_BASE + path + (("?" + urlencode(query)) if query else "")
        request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Authorization": "Bearer " + self._token,
                                   "Content-Type": "application/json",
                                   "User-Agent": "PathSnap-scoped-deployer/1"}, method=method)
        try:
            with self._opener.open(request, timeout=25) as response:
                data = json.loads(response.read(4 * 1024 * 1024))
        except HTTPError as exc:
            # Error bodies can contain echoed request values. Emit only numeric codes.
            codes = []
            try:
                payload = json.loads(exc.read(65536))
                codes = [item.get("code") for item in payload.get("errors", []) if isinstance(item.get("code"), int)]
            except (ValueError, OSError, AttributeError):
                pass
            raise DeploymentError(f"Cloudflare API {method} failed with HTTP {exc.code}; error codes {codes}. No credential values logged.") from None
        except (URLError, OSError, ValueError):
            raise DeploymentError("Cloudflare API network or response failure; inspect connectivity and retry preflight, not a blind mutation.") from None
        if not isinstance(data, dict) or data.get("success") is not True:
            raise DeploymentError("Cloudflare API did not confirm success; no response body is logged.")
        return data

    def list(self, path: str, **query) -> list[dict]:
        items = []
        for page in range(1, 101):
            data = self.call("GET", path, page=page, per_page=100, **query)
            rows = data.get("result")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise DeploymentError("Cloudflare returned an unexpected list schema.")
            items.extend(rows)
            info = data.get("result_info", {})
            if len(rows) < 100 or page >= info.get("total_pages", 100):
                return items
        raise DeploymentError("Cloudflare pagination exceeded the bounded discovery limit.")


def external_path(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink() or path.resolve().is_relative_to(ROOT):
        raise DeploymentError("All credential/state files must be regular paths outside the project checkout.")
    if any(parent.is_symlink() for parent in path.parents):
        raise DeploymentError("Credential/state paths must not traverse symlinked directories; refusing to follow them.")
    return path


def validate_runtime_identity(config_dir: Path, *, require_installed: bool) -> None:
    """Bind provisioning to the dedicated configuration the reviewed installer created for this checkout."""
    if not (config_dir / "runtime.json").exists():
        if require_installed:
            raise DeploymentError("Run the dedicated installer for this checkout before any Cloudflare mutation; nothing was changed.")
        return
    config = read_runtime(config_dir)
    if config.get("installer_id") != INSTALLER_ID or config.get("repo_root") != str(ROOT):
        raise DeploymentError("The dedicated configuration directory belongs to a different checkout or installer generation; refusing to use it.")


def validate_id(value: str | None, label: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not HEX_ID.fullmatch(value)):
        raise DeploymentError(f"The {label} must be a 32-character lowercase hexadecimal ID.")
    return value


def load_authorization(args: argparse.Namespace) -> tuple[str, str | None, str | None, str]:
    if args.token_file:
        path = external_path(args.token_file)
        private_file(path)
        token = path.read_text().strip()
        if not token or any(char.isspace() for char in token):
            raise DeploymentError("Private Cloudflare API-token file has an invalid format.")
        return token, validate_id(args.account_id, "account ID"), validate_id(args.zone_id, "zone ID"), "api-token-file"
    candidates = [args.origincert] if args.origincert else [
        Path.home() / ".cloudflared/cert.pem", Path.home() / ".config/cloudflared/cert.pem",
        Path("/etc/cloudflared/cert.pem"), Path("/usr/local/etc/cloudflared/cert.pem"),
    ]
    for candidate in candidates:
        path = external_path(candidate)
        if not path.is_file():
            continue
        # cloudflared origin-cert PEM embeds JSON authorization. Never print its body.
        private_file(path)
        match = re.search(rb"-----BEGIN ARGO TUNNEL TOKEN-----\s+([A-Za-z0-9+/=\r\n]+?)\s+-----END ARGO TUNNEL TOKEN-----", path.read_bytes())
        if not match:
            raise DeploymentError("Origin certificate contains no supported Cloudflare Tunnel authorization block.")
        try:
            data = json.loads(base64.b64decode(match.group(1)))
            token, account, zone = data["apiToken"], data["accountID"], data["zoneID"]
            if not isinstance(token, str) or not token or data.get("endpoint", "") not in {"", "cloudflare"}:
                raise ValueError
            return token, validate_id(account, "account ID"), validate_id(zone, "zone ID"), "cloudflared-origin-cert"
        except (KeyError, ValueError, TypeError):
            raise DeploymentError("Origin authorization metadata is incomplete or unsupported; no values logged.") from None
    raise DeploymentError("No readable Cloudflare authorization exists in the checked standard paths. Authorize cloudflared tunnel login for d3va.dev locally, or supply an outside-checkout mode-600 API-token file with the scoped account/zone permissions documented in infrastructure/HOSTING.md. Never paste credentials into chat.")


def resolve_zone(api: CloudflareAPI, account_id: str | None, zone_id: str | None) -> tuple[str, str]:
    if zone_id:
        zone = api.call("GET", "/zones/" + zone_id)["result"]
    else:
        zones = api.list("/zones", name=ZONE_NAME, status="active")
        exact = [zone for zone in zones if zone.get("name") == ZONE_NAME]
        if len(exact) != 1:
            raise DeploymentError("The authorization must identify exactly one active d3va.dev zone.")
        zone = exact[0]
    if zone.get("name") != ZONE_NAME or zone.get("status") != "active":
        raise DeploymentError("The selected zone is not the active authorized d3va.dev zone.")
    actual_account = zone.get("account", {}).get("id")
    if not actual_account or not zone.get("id"):
        raise DeploymentError("Cloudflare zone response omitted required account or zone IDs.")
    if account_id and account_id != actual_account:
        raise DeploymentError("Cloudflare account metadata conflicts with the selected zone.")
    return validate_id(actual_account, "account ID"), validate_id(zone.get("id"), "zone ID")


def load_owned_state(config_dir: Path, account_id: str, zone_id: str) -> tuple[dict | None, dict | None]:
    state_path, credentials_path = config_dir / "deployment.json", config_dir / "tunnel-credentials.json"
    if not state_path.exists() and not credentials_path.exists():
        return None, None
    if not state_path.exists() or not credentials_path.exists():
        raise DeploymentError("Dedicated local tunnel state is incomplete; inspect it privately before any mutation.")
    for path in (state_path, credentials_path):
        private_file(path)
    try:
        state = json.loads(state_path.read_text())
        credentials = json.loads(credentials_path.read_text())
        UUID(state["tunnel_id"])
        if (state.get("hostname") != HOSTNAME or state.get("tunnel_name") != TUNNEL_NAME
                or state.get("account_id") != account_id or state.get("zone_id") != zone_id
                or credentials.get("AccountTag") != account_id
                or credentials.get("TunnelID") != state["tunnel_id"]
                or len(base64.b64decode(credentials["TunnelSecret"], validate=True)) != 32):
            raise ValueError
        return state, credentials
    except (ValueError, TypeError, KeyError):
        raise DeploymentError("Local tunnel ownership/credential metadata does not match this deployment.") from None


def assert_records_safe(records: list[dict], owned_tunnel_id: str | None) -> None:
    if not records:
        return
    expected = (owned_tunnel_id or "") + ".cfargotunnel.com"
    if (not owned_tunnel_id or len(records) != 1 or records[0].get("name") != HOSTNAME
            or records[0].get("type") != "CNAME"
            or records[0].get("content", "").rstrip(".") != expected
            or records[0].get("proxied") is not True):
        types = sorted({str(record.get("type", "unknown")) for record in records})
        raise DeploymentError(f"Exact-hostname DNS conflict for {HOSTNAME} ({types}). Refusing overwrite; unrelated records remain unchanged.")


def write_state(config_dir: Path, state: dict) -> None:
    write_private(config_dir / "deployment.json", json.dumps(state, indent=2) + "\n", replace=True)


def write_tunnel_config(config_dir: Path, tunnel_id: str) -> None:
    # JSON is valid YAML and avoids a parser dependency or interpolation of paths.
    config = {"tunnel": tunnel_id, "credentials-file": str(config_dir / "tunnel-credentials.json"),
              "metrics": "127.0.0.1:0", "loglevel": "info",
              "ingress": [{"hostname": HOSTNAME, "service": ORIGIN,
                           "originRequest": {"connectTimeout": "10s"}},
                          {"service": "http_status:404"}]}
    write_private(config_dir / "cloudflared.yml", json.dumps(config, indent=2) + "\n", replace=True)


def read_deployment_state(api: CloudflareAPI, records_path: str, tunnels_path: str, owned_id: str | None) -> list[dict]:
    """Read the exact-hostname record and dedicated-name tunnel state, refusing every conflict."""
    records = api.list(records_path, name=HOSTNAME)
    if any(record.get("name") != HOSTNAME for record in records):
        raise DeploymentError("DNS API did not honor the exact-hostname filter; refusing mutation.")
    assert_records_safe(records, owned_id)
    tunnels = api.list(tunnels_path, name=TUNNEL_NAME, is_deleted="false")
    # The API already filters is_deleted=false; treat every same-name result as
    # a conflict/owned match, including legacy zero-time deleted_at encodings.
    exact = [tunnel for tunnel in tunnels if tunnel.get("name") == TUNNEL_NAME]
    if owned_id:
        if len(exact) != 1 or exact[0].get("id") != owned_id:
            raise DeploymentError("The locally owned tunnel does not match Cloudflare. No existing tunnel was modified.")
    elif exact:
        raise DeploymentError("A path-snap tunnel already exists without this deployment ownership state; refusing to adopt or modify it.")
    return records


def deploy(api: CloudflareAPI, args: argparse.Namespace, account_id: str, zone_id: str, auth_type: str) -> dict:
    config_dir = dedicated_config_dir(external_path(args.config_dir))
    # Fixed configuration/repository identity: apply only ever writes into the
    # dedicated directory the reviewed installer created for this exact checkout.
    validate_runtime_identity(config_dir, require_installed=args.apply)
    state, credentials = load_owned_state(config_dir, account_id, zone_id)
    owned_id = state["tunnel_id"] if state else None
    records_path = "/zones/" + zone_id + "/dns_records"
    tunnels_path = "/accounts/" + account_id + "/cfd_tunnel"
    records = read_deployment_state(api, records_path, tunnels_path, owned_id)
    summary = {"hostname": HOSTNAME, "origin": ORIGIN, "authorization": auth_type,
               "zone_status": "active", "exact_record_count": len(records),
               "tunnel": "owned" if owned_id else "absent",
               "mode": "apply" if args.apply else "read-only-preflight"}
    if not args.apply:
        return summary
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Verify the directory identity immediately before the fixed 0700 tightening.
    dedicated_directory(config_dir)
    # Bound the whole mutation section by the exclusive local apply lock so a
    # concurrent apply cannot race past the intent/ownership checks.
    with operation_lock(config_dir, "cloudflare-apply.lock"):
        # Re-read every ownership fact under the lock; nothing read before it is trusted.
        state, credentials = load_owned_state(config_dir, account_id, zone_id)
        owned_id = state["tunnel_id"] if state else None
        records = read_deployment_state(api, records_path, tunnels_path, owned_id)
        if not owned_id:
            # Persist a recovery intent before the one non-idempotent creation request.
            secret = base64.b64encode(secrets.token_bytes(32)).decode()
            intent_path = config_dir / "creation-intent.json"
            if intent_path.exists():
                raise DeploymentError("A previous tunnel-creation intent exists; inspect Cloudflare/local state before retrying.")
            intent = {"account_id": account_id, "zone_id": zone_id, "tunnel_name": TUNNEL_NAME,
                      "hostname": HOSTNAME, "tunnel_secret": secret}
            write_private(intent_path, json.dumps(intent) + "\n")
            result = api.call("POST", tunnels_path, {"name": TUNNEL_NAME, "tunnel_secret": secret, "config_src": "local"})["result"]
            owned_id = str(UUID(result["id"]))
            credentials = {"AccountTag": account_id, "TunnelID": owned_id, "TunnelSecret": secret}
            write_private(config_dir / "tunnel-credentials.json", json.dumps(credentials) + "\n")
            state = {"hostname": HOSTNAME, "origin": ORIGIN, "account_id": account_id,
                     "zone_id": zone_id, "tunnel_name": TUNNEL_NAME, "tunnel_id": owned_id,
                     "created_at": datetime.now(timezone.utc).isoformat(), "dns_record_id": None}
            write_state(config_dir, state)
        write_tunnel_config(config_dir, owned_id)
        # Repeat the exact-record read immediately before create. Never update/delete.
        records = api.list(records_path, name=HOSTNAME)
        assert_records_safe(records, owned_id)
        if not records:
            if auth_type == "cloudflared-origin-cert":
                # cloudflared route API has an explicit atomic no-overwrite flag.
                api.call("PUT", "/zones/" + zone_id + "/tunnels/" + owned_id + "/routes",
                         {"type": "dns", "user_hostname": HOSTNAME, "overwrite_existing": False})
            else:
                # Cloudflare rejects a CNAME colliding with an existing address record.
                api.call("POST", records_path, {"type": "CNAME", "name": HOSTNAME,
                         "content": owned_id + ".cfargotunnel.com", "proxied": True, "ttl": 1})
        records = api.list(records_path, name=HOSTNAME)
        assert_records_safe(records, owned_id)
        if not records:
            raise DeploymentError("Cloudflare did not confirm the dedicated DNS record; tunnel process was not started.")
        state["dns_record_id"] = records[0]["id"]
        write_state(config_dir, state)
    summary.update({"tunnel": "owned", "tunnel_id": owned_id, "dns": "dedicated proxied CNAME confirmed",
                    "service_started": False})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create only the dedicated tunnel/DNS if preflight is conflict-free.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--origincert", type=Path, help="Existing private cloudflared origin-certificate path (never its value).")
    group.add_argument("--token-file", type=Path, help="Existing mode-600 API-token file outside the checkout.")
    parser.add_argument("--account-id")
    parser.add_argument("--zone-id")
    parser.add_argument("--config-dir", type=Path, default=Path.home() / ".config/path-snap")
    args = parser.parse_args(argv)
    try:
        token, account_id, zone_id, auth_type = load_authorization(args)
        api = CloudflareAPI(token)
        account_id, zone_id = resolve_zone(api, account_id, zone_id)
        print(json.dumps(deploy(api, args, account_id, zone_id, auth_type), indent=2))
        return 0
    except (DeploymentError, LaunchError, OSError, ValueError, TypeError, KeyError) as exc:
        message = str(exc) if isinstance(exc, LaunchError) else "Local Cloudflare configuration or response metadata is invalid; no secret values are logged."
        print("PathSnap Cloudflare deployment refused: " + message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
