#!/usr/bin/env python3
"""Launch only the dedicated PathSnap processes with a small, explicit environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import stat
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

HOSTNAME = "ml.d3va.dev"
ORIGIN = "http://127.0.0.1:8000"
ALLOWED_SETTINGS = {
    "PATHSNAP_MODEL", "PATHSNAP_MODEL_METADATA", "PATHSNAP_MATCH_CONFIG",
    "PATHSNAP_DATA_DIR", "PATHSNAP_REFERENCE_DIR", "PATHSNAP_DATASET_ROLE",
    "PATHSNAP_GRAPH_CACHE_SIZE", "PATHSNAP_INFERENCE_CONCURRENCY",
    "PATHSNAP_CORS_ORIGINS",
}
THREAD_SETTINGS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


class LaunchError(Exception):
    """An actionable error safe to emit to the service journal."""


def dedicated_config_dir(path: Path) -> Path:
    path = path.expanduser().absolute()
    expected = Path.home().absolute() / ".config/path-snap"
    if path != expected:
        raise LaunchError("Configuration must use the dedicated ~/.config/path-snap directory, never a shared directory.")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise LaunchError("Dedicated configuration must not have symlinked path components.")
    if path.exists():
        info = path.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise LaunchError("Dedicated configuration directory must be owned by this user.")
    return path


def api_authorization(config_dir: Path, *, require_ready: bool = False) -> str:
    path = config_dir / "api-authorized"
    private_file(path)
    try:
        authorization = json.loads(path.read_text())
        phase = authorization.get("phase")
        if phase not in {"static", "ready"} or (require_ready and phase != "ready"):
            raise ValueError
        return phase
    except (ValueError, TypeError, AttributeError) as exc:
        raise LaunchError("Dedicated API authorization is invalid or not explicitly ready for public hosting.") from exc


def private_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise LaunchError("Dedicated runtime files must be regular files owned by this user.")
    if info.st_mode & 0o077:
        raise LaunchError("Dedicated runtime files must not be readable by other users (use mode 600).")
    if info.st_size > 1024 * 1024:
        raise LaunchError("Dedicated runtime file exceeds its size limit.")


def dedicated_directory(path: Path) -> None:
    """Verify a directory is this user's real, non-symlink directory before tightening it to 0700.

    The fixed 0700 mode is applied only to a path that was just re-verified with lstat,
    so no arbitrary or symlink-following chmod can occur.
    """
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise LaunchError("Dedicated runtime directories must be real directories owned by this user.")
    if info.st_mode & 0o077:
        os.chmod(path, 0o700)


def read_runtime(config_dir: Path) -> dict:
    path = config_dir / "runtime.json"
    private_file(path)
    try:
        config = json.loads(path.read_text())
        if config.get("hostname") != HOSTNAME or config.get("origin") != ORIGIN:
            raise LaunchError("Runtime is not scoped to the authorized PathSnap hostname and origin.")
        root = Path(config["repo_root"])
        python = Path(config["python"])
        cloudflared = Path(config["cloudflared"])
        if not root.is_absolute() or not root.is_dir() or not python.is_absolute() or not cloudflared.is_absolute():
            raise LaunchError("Runtime paths must be absolute and the project directory must exist.")
        return config
    except (KeyError, ValueError, TypeError) as exc:
        raise LaunchError("Dedicated runtime configuration is invalid.") from exc


def read_settings(path: Path) -> dict[str, str]:
    private_file(path)
    settings = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in ALLOWED_SETTINGS or key in settings:
            raise LaunchError("Runtime environment contains an unknown or repeated setting.")
        value = value.strip()
        if value.startswith(chr(34)):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise LaunchError("Quoted runtime values must be JSON strings.") from exc
        if not isinstance(value, str) or any(char in value for char in ("\n", "\r", "\0")):
            raise LaunchError("Runtime environment contains an invalid value.")
        settings[key] = value
    if settings.get("PATHSNAP_CORS_ORIGINS", "") not in {"", "https://" + HOSTNAME}:
        raise LaunchError("Only the exact dedicated HTTPS origin may be configured for public API access.")
    if settings.get("PATHSNAP_INFERENCE_CONCURRENCY", "1") not in {"1", "2"}:
        raise LaunchError("Public inference concurrency must be one or two.")
    if settings.get("PATHSNAP_DATASET_ROLE") not in {"development", "evaluator"}:
        raise LaunchError("Set the actual development/evaluator dataset role explicitly.")
    return settings


def limited_environment(settings: dict[str, str] | None = None) -> dict[str, str]:
    account = pwd.getpwuid(os.getuid())
    env = {"HOME": account.pw_dir, "USER": account.pw_name, "LOGNAME": account.pw_name,
           "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    runtime = f"/run/user/{os.getuid()}"
    if Path(runtime).is_dir():
        env["XDG_RUNTIME_DIR"] = runtime
    env.update({key: "1" for key in THREAD_SETTINGS})
    env.update(settings or {})
    return env


def check_api(config_dir: Path, config: dict, *, require_authorization: bool = True, require_artifacts: bool = True) -> dict[str, str]:
    if require_authorization:
        require_artifacts = api_authorization(config_dir) == "ready"
    root = Path(config["repo_root"])
    interpreter = Path(config["python"])
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise LaunchError("Project Python interpreter is unavailable; recreate the project virtual environment.")
    settings = read_settings(config_dir / "runtime.env")
    for key in ("PATHSNAP_MODEL", "PATHSNAP_MODEL_METADATA", "PATHSNAP_MATCH_CONFIG"):
        value = settings.get(key)
        if not value:
            raise LaunchError("Set explicit trained-model, metadata and decoder configuration paths before launch.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if require_artifacts and (not path.is_file() or path.stat().st_size == 0):
            raise LaunchError("A required trained-model artifact is missing or empty; backend was not started.")
    if not (root / "backend/main.py").is_file() or not (root / "frontend/index.html").is_file():
        raise LaunchError("API entry point or same-origin frontend is missing.")
    return settings


def check_tunnel(config_dir: Path) -> str:
    credentials_path = config_dir / "tunnel-credentials.json"
    state_path = config_dir / "deployment.json"
    yaml_path = config_dir / "cloudflared.yml"
    for path in (credentials_path, state_path, yaml_path):
        private_file(path)
    try:
        credentials = json.loads(credentials_path.read_text())
        state = json.loads(state_path.read_text())
        tunnel_id = credentials["TunnelID"]
        if (state.get("hostname") != HOSTNAME or state.get("tunnel_name") != "path-snap"
                or state.get("tunnel_id") != tunnel_id
                or state.get("account_id") != credentials.get("AccountTag")
                or not credentials.get("TunnelSecret")):
            raise LaunchError("Tunnel credentials do not match the dedicated deployment manifest.")
        # Parse our generated JSON-compatible YAML using the standard library only.
        config = json.loads(yaml_path.read_text())
        expected = [
            {"hostname": HOSTNAME, "service": ORIGIN, "originRequest": {"connectTimeout": "10s"}},
            {"service": "http_status:404"},
        ]
        if (config.get("tunnel") != tunnel_id
                or config.get("credentials-file") != str(credentials_path)
                or config.get("ingress") != expected
                or config.get("metrics") != "127.0.0.1:0"):
            raise LaunchError("Tunnel configuration is not scoped to the dedicated hostname and loopback API.")
        return tunnel_id
    except (KeyError, ValueError, TypeError) as exc:
        raise LaunchError("Dedicated tunnel configuration is invalid; no credential values are logged.") from exc


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def wait_for_api(timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}), NoRedirect())
    while True:
        try:
            with opener.open(ORIGIN + "/health", timeout=3) as response:
                data = json.loads(response.read(65536))
                if data.get("status") == "ready" and data.get("model_loaded") is True and data.get("trip_count") == 10:
                    return
        except (HTTPError, URLError, OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise LaunchError("The trained-model API did not become healthy; tunnel was not started.")
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process", choices=("api", "tunnel", "check-api", "check-tunnel"))
    parser.add_argument("--config-dir", type=Path, default=Path.home() / ".config/path-snap")
    args = parser.parse_args(argv)
    try:
        config_dir = dedicated_config_dir(args.config_dir)
        config = read_runtime(config_dir)
        if args.process in {"api", "check-api"}:
            settings = check_api(config_dir, config)
            if args.process == "check-api":
                print("PathSnap API runtime validated; no process started.")
                return 0
            command = [config["python"], "-B", "-m", "uvicorn", "backend.main:app",
                       "--host", "127.0.0.1", "--port", "8000", "--workers", "1",
                       "--no-access-log", "--limit-concurrency", "16",
                       "--timeout-keep-alive", "5", "--proxy-headers",
                       "--forwarded-allow-ips", "127.0.0.1"]
            os.chdir(config["repo_root"])
            os.execve(config["python"], command, limited_environment(settings))
        tunnel_id = check_tunnel(config_dir)
        if args.process == "check-tunnel":
            print("PathSnap tunnel runtime validated; no credential values displayed.")
            return 0
        api_authorization(config_dir, require_ready=True)
        wait_for_api()
        command = [config["cloudflared"], "--no-autoupdate", "--config",
                   str(config_dir / "cloudflared.yml"), "tunnel", "run", tunnel_id]
        os.chdir(config_dir)
        os.execve(config["cloudflared"], command, limited_environment())
    except (LaunchError, OSError) as exc:
        message = str(exc) if isinstance(exc, LaunchError) else "A required dedicated runtime file or executable is unavailable."
        print("PathSnap launch refused: " + message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
