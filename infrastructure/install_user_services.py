#!/usr/bin/env python3
"""Install or operate only the two dedicated PathSnap systemd user services.

Installation enables restart/reboot units but does not start the API or tunnel.
The explicit authorize-start-api action requires readiness unless --allow-notready authorizes the local static stage.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

from launch_service import (LaunchError, api_authorization, check_api, check_tunnel, dedicated_config_dir,
                            dedicated_directory, private_file, read_runtime, read_settings)

HERE = Path(__file__).resolve().parent
UNITS = ("path-snap-api.service", "path-snap-tunnel.service")


def quote_unit(value: Path | str) -> str:
    # ExecStart is a word list with quoting and dollar expansion.
    return json.dumps(str(value).replace("%", "%%").replace("$", "$$"))


def unit_path(value: Path | str) -> str:
    # Path-valued directives consume the whole string, including quote characters.
    text = str(value)
    if any(char in text for char in ("\n", "\r", "\0")):
        raise LaunchError("Unit paths must not contain control characters.")
    return text.replace("%", "%%")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_private(path: Path, content: str, *, replace: bool = False) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        private_file(path)
    if not replace:
        # An exclusive intent/credential claim must never silently succeed twice.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(path.parent)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pathsnap-" + path.name + "-", dir=path.parent)
    temporary = Path(temporary_name).absolute()
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            private_file(path)
        if temporary.parent != path.parent or not temporary.name.startswith(".pathsnap-" + path.name + "-"):
            raise LaunchError("Private atomic-write temporary path failed scope validation.")
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists() and temporary.parent == path.parent and temporary.name.startswith(".pathsnap-" + path.name + "-"):
            temporary.unlink()


@contextmanager
def operation_lock(config_dir: Path, name: str):
    if name not in {"cloudflare-apply.lock", "service-setup.lock"}:
        raise LaunchError("Unexpected deployment lock name.")
    path = config_dir / name
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        private_file(path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LaunchError("Another dedicated deployment operation is running; no duplicate work was started.") from exc
        yield
    finally:
        os.close(descriptor)


def systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", "--no-ask-password", *args], check=check)


def prepare_directories(config_dir: Path, unit_dir: Path) -> None:
    if any(path.is_symlink() for path in (config_dir, unit_dir, *unit_dir.parents)):
        raise LaunchError("Refusing symlinked dedicated configuration or user-unit directories.")
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Re-verify ownership/type immediately before the fixed 0700 tightening; never chmod an unverified path.
    dedicated_directory(config_dir)
    unit_dir.mkdir(mode=0o700, parents=True, exist_ok=True)


def rendered_units(root: Path, config_dir: Path, *, identity: bool = True) -> dict[str, str]:
    replacements = {
        "@REPO_PATH@": unit_path(root), "@CONFIG_PATH@": unit_path(config_dir),
        "@AUTH_PATH@": unit_path(config_dir / "api-authorized"),
        "@TUNNEL_CONFIG_PATH@": unit_path(config_dir / "cloudflared.yml"),
        "@CREDENTIALS_PATH@": unit_path(config_dir / "tunnel-credentials.json"),
        "@LAUNCHER_ARG@": quote_unit(config_dir / "launch_service.py"),
        "@CONFIG_ARG@": quote_unit(config_dir),
    }
    result = {}
    header = ("# PathSnap scoped hosting installer v1\n# RepoRoot=" + json.dumps(str(root))
              + "\n# ConfigDir=" + json.dumps(str(config_dir)) + "\n") if identity else ""
    for name in UNITS:
        content = (HERE / "templates" / (name + ".in")).read_text()
        for key, value in replacements.items():
            content = content.replace(key, value)
        result[name] = header + content
    return result


def validate_installed_units(config_dir: Path, *, verify_loaded: bool = True) -> dict:
    config = read_runtime(config_dir)
    root = Path(config["repo_root"])
    if config.get("installer_id") != "pathsnap-scoped-hosting-v1":
        raise LaunchError("Dedicated services have no matching installer identity; install them before lifecycle operations.")
    expected = rendered_units(root, config_dir)
    for name in UNITS:
        path = Path.home() / ".config/systemd/user" / name
        private_file(path)
        if path.read_text() != expected[name]:
            raise LaunchError("A fixed-name PathSnap unit does not match its dedicated checkout/runtime; no service action was performed.")
        if verify_loaded:
            result = subprocess.run(["systemctl", "--user", "show", name, "--property=FragmentPath,DropInPaths,WorkingDirectory"],
                                    text=True, capture_output=True, check=True)
            values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            working_directory = str(root if name == UNITS[0] else config_dir)
            if (values.get("FragmentPath") != str(path) or values.get("DropInPaths")
                    or values.get("WorkingDirectory") != working_directory):
                raise LaunchError("Loaded fixed-name PathSnap service differs or has unreviewed drop-ins; refusing service action.")
    return config


def install(args: argparse.Namespace, config_dir: Path, unit_dir: Path) -> None:
    root = args.repo_root.expanduser().resolve()
    python = args.python.expanduser().absolute() if args.python else root / ".venv/bin/python"
    cloudflared = args.cloudflared.expanduser().absolute()
    if not (root / "backend/main.py").is_file() or not python.is_file() or not cloudflared.is_file():
        raise LaunchError("API source, project interpreter or cloudflared executable is unavailable.")
    # Bind fixed unit names to exact generated bytes, paths and checkout identity.
    expected = rendered_units(root, config_dir)
    legacy = rendered_units(root, config_dir, identity=False)
    runtime_path = config_dir / "runtime.json"
    for name in UNITS:
        path = unit_dir / name
        if path.exists() or path.is_symlink():
            private_file(path)
            if not runtime_path.exists() or path.read_text() not in {expected[name], legacy[name]}:
                raise LaunchError("A same-named user service is not this dedicated installation; refusing overwrite.")
    if runtime_path.exists():
        current = read_runtime(config_dir)
        if current["repo_root"] != str(root):
            raise LaunchError("The dedicated PathSnap configuration belongs to a different checkout.")
    prepare_directories(config_dir, unit_dir)
    runtime = {"hostname": "ml.d3va.dev", "origin": "http://127.0.0.1:8000",
               "installer_id": "pathsnap-scoped-hosting-v1",
               "repo_root": str(root), "python": str(python), "cloudflared": str(cloudflared)}
    write_private(runtime_path, json.dumps(runtime, indent=2) + "\n", replace=True)
    write_private(config_dir / "launch_service.py", (HERE / "launch_service.py").read_text(), replace=True)
    if not (config_dir / "runtime.env").exists():
        write_private(config_dir / "runtime.env", (HERE / "templates/runtime.env.example").read_text())
    read_settings(config_dir / "runtime.env")
    for name in UNITS:
        write_private(unit_dir / name, expected[name], replace=True)
    systemctl("daemon-reload")
    systemctl("enable", *UNITS)
    validate_installed_units(config_dir)
    print("Installed and enabled only " + ", ".join(UNITS) + "; this action did not start or restart either unit.")
    print("Runtime directory: " + str(config_dir) + ". Explicit authorization controls the static/ready launch stage.")


def authorize_start_api(config_dir: Path, *, allow_notready: bool = False) -> None:
    config = validate_installed_units(config_dir)
    check_api(config_dir, config, require_authorization=False, require_artifacts=not allow_notready)
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNITS[0]]).returncode == 0
    if not active:
        # Never stop an existing listener in order to claim the origin port.
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", 8000))
            except OSError as exc:
                raise LaunchError("Port 8000 is already occupied; existing listener was not changed.") from exc
    authorization = {"phase": "static" if allow_notready else "ready",
                     "authorized_at": datetime.now(timezone.utc).isoformat()}
    write_private(config_dir / "api-authorized", json.dumps(authorization) + "\n", replace=True)
    systemctl("start", UNITS[0])
    print("Authorized and started only " + UNITS[0] + "; check /health before starting the tunnel.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "authorize-start-api", "start-tunnel", "status"))
    parser.add_argument("--allow-notready", action="store_true",
                        help="Explicitly authorize only the local static/notready stage before model artifacts exist.")
    parser.add_argument("--repo-root", type=Path, default=HERE.parent)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--cloudflared", type=Path, default=Path("/usr/bin/cloudflared"))
    parser.add_argument("--config-dir", type=Path, default=Path.home() / ".config/path-snap")
    args = parser.parse_args(argv)
    unit_dir = Path.home() / ".config/systemd/user"
    try:
        config_dir = dedicated_config_dir(args.config_dir)
        if args.action == "install":
            prepare_directories(config_dir, unit_dir)
            with operation_lock(config_dir, "service-setup.lock"):
                install(args, config_dir, unit_dir)
        elif args.action == "authorize-start-api":
            with operation_lock(config_dir, "service-setup.lock"):
                authorize_start_api(config_dir, allow_notready=args.allow_notready)
        elif args.action == "start-tunnel":
            with operation_lock(config_dir, "service-setup.lock"):
                validate_installed_units(config_dir)
                api_authorization(config_dir, require_ready=True)
                check_tunnel(config_dir)
                systemctl("start", UNITS[1])
                print("Started only " + UNITS[1] + ".")
        else:
            systemctl("show", *UNITS, "--property=Id,LoadState,ActiveState,SubState,MainPID,Restart,NRestarts,UnitFileState")
            result = subprocess.run(["loginctl", "show-user", str(os.getuid()), "--property=Linger"], check=False)
            return result.returncode
    except (LaunchError, OSError, subprocess.CalledProcessError) as exc:
        message = str(exc) if isinstance(exc, LaunchError) else "A local runtime file or user-manager operation failed; no unrelated service was changed."
        print("PathSnap setup refused: " + message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
