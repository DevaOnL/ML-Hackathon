#!/usr/bin/env python3
"""Fail-closed final-five activation of the existing dedicated API only.

Default is an OFFLINE validate-only dry run: no private runtime environment is
read, no HTTP request or service mutation occurs. --apply additionally requires
--lead-go FINAL_FIVE_ACTIVATE after the Lead's explicit GO. Exactly six managed
settings may change; all unrelated bytes are retained. No unit, tunnel, DNS,
credential, launcher or infrastructure installation changes are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "infrastructure") not in sys.path:
    sys.path.insert(0, str(ROOT / "infrastructure"))
from install_user_services import operation_lock, validate_installed_units, write_private
from launch_service import ALLOWED_SETTINGS, check_api, dedicated_config_dir, private_file
from scripts.verify_final_api import (
    FINAL, MODEL_SHA256, CONFIG_SHA256, COMPACT_CONFIG_SHA256, load_json, sha256,
    require, require_checks, load_offline_expected, validate_api_acceptance,
    source_manifest, verify_manifest, health_checks, utcnow,
)

REPLACEMENTS = {
    "PATHSNAP_MODEL": "artifacts/final_five/model.joblib",
    "PATHSNAP_MODEL_METADATA": "artifacts/final_five/model_metadata.json",
    "PATHSNAP_MATCH_CONFIG": "artifacts/final_five/match_config.json",
    "PATHSNAP_EVALUATOR_DATA_DIR": "evaluator_test",
    "PATHSNAP_EVALUATOR_REFERENCE_DIR": "evaluator_test",
    "PATHSNAP_EVALUATOR_EXPECTED_TRIP_COUNT": "10",
}
REQUIRED_ORIGINAL = {"PATHSNAP_MODEL", "PATHSNAP_MODEL_METADATA", "PATHSNAP_MATCH_CONFIG"}
API_UNIT = "path-snap-api.service"
TUNNEL_UNIT = "path-snap-tunnel.service"
BASES = ("http://127.0.0.1:8000", "https://ml.d3va.dev")
GO_TOKEN = "FINAL_FIVE_ACTIVATE"


def render_environment(original):
    """Operate on exact bytes, retain unrelated lines and reject duplicate keys."""
    require(isinstance(original, bytes), "environment renderer requires exact original bytes")
    text = original.decode("utf-8")
    require("\0" not in text, "runtime environment contains invalid bytes")
    seen = set()
    result = []
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        ending = line[len(content):]
        stripped = content.strip()
        if not stripped or stripped.startswith("#"):
            result.append(line)
            continue
        key, separator, _ = stripped.partition("=")
        require(separator and key and key == key.strip(), "invalid runtime environment setting syntax")
        require(key not in seen, "runtime environment contains a repeated key")
        seen.add(key)
        if key in REPLACEMENTS:
            result.append(key + "=" + json.dumps(REPLACEMENTS[key]) + ending)
        else:
            result.append(line)
    require(REQUIRED_ORIGINAL.issubset(seen), "explicit original artifact settings are missing")
    rendered = "".join(result)
    missing = [key for key in REPLACEMENTS if key not in seen]
    if missing and rendered and not rendered.endswith(("\n", "\r")):
        rendered += "\n"
    for key in missing:
        rendered += key + "=" + json.dumps(REPLACEMENTS[key]) + "\n"
    return rendered.encode("utf-8")


def require_unchanged_bytes(current, original):
    require(current == original, "runtime environment changed after validation; no mutation permitted")


def acceptance(root=ROOT):
    expected, datasets = load_offline_expected(root)
    report = load_json(Path(root) / FINAL / "api_integration_acceptance.json")
    validate_api_acceptance(report, datasets, expected)
    verify_manifest(report.get("source_sha256_after"), root)
    require(set(REPLACEMENTS).issubset(ALLOWED_SETTINGS), "reviewed launcher does not allow exactly required evaluator settings")
    return expected


def require_go(apply, lead_go):
    if apply:
        require(lead_go == GO_TOKEN, "--apply is forbidden before explicit Lead GO and its confirmation flag")


def write_exact_private(path, contents, replace=False):
    """Installer's safe atomic helper; decoded text preserves original UTF-8 bytes."""
    write_private(Path(path), contents.decode("utf-8"), replace=replace)
    require(Path(path).read_bytes() == contents, "private atomic write did not preserve exact bytes")


def restart_api():
    subprocess.run(["systemctl", "--user", "--no-ask-password", "restart", API_UNIT], check=True, capture_output=True, text=True, timeout=45)


def mutate_with_rollback(env_file, original, replacement, *, writer=None, restart=None, verify=None):
    """Testable mutation seam; production dependencies remain the real helpers."""
    writer = writer or (lambda path, data: write_exact_private(path, data, replace=True))
    restart = restart or restart_api
    require(verify is not None, "activation must supply an origin/readiness verifier")
    require_unchanged_bytes(Path(env_file).read_bytes(), original)
    try:
        writer(env_file, replacement)
        restart()
        checked = verify()
        return checked
    except Exception:
        try:
            writer(env_file, original)
            restart()
            require_unchanged_bytes(Path(env_file).read_bytes(), original)
        except Exception:
            raise RuntimeError("activation failed and exact rollback/restart requires Lead intervention") from None
        raise RuntimeError("activation failed; exact previous environment restored and dedicated API restarted") from None


def read_json_url(base, path):
    # Fixed origins only. Default TLS verification remains enabled.
    require(base in BASES, "unapproved API origin")
    request = urllib.request.Request(base + path, headers={"Cache-Control": "no-cache", "User-Agent": "PathSnap-final-guarded-activation/1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        require(response.status == 200 and response.geturl() == base + path, "activation HTTP redirect/status mismatch")
        return json.loads(response.read())


def check_existing_origins(expected):
    checked = {}
    for base in BASES:
        for attempt in range(8):
            try:
                health = read_json_url(base, "/health")
                catalogue = read_json_url(base, "/datasets")
                checks = health_checks(health, catalogue)
                require_checks(checks, "ready installed origin")
                require(health["model_hash"] == expected["model_sha256"] and health["match_config_hash"] == expected["match_config_hash"], "origin frozen identity mismatch")
                checked[base] = {"health": health, "datasets": catalogue, "checks": checks}
                break
            except (OSError, ValueError):
                if attempt == 7:
                    raise
                # Bounded readiness retry solely for this dedicated API restart.
                time.sleep(.4 * (attempt + 1))
    return checked


def installed_snapshot(config_dir):
    launcher = config_dir / "launch_service.py"
    private_file(launcher)
    require(sha256(launcher) == sha256(ROOT / "infrastructure/launch_service.py"), "installed dedicated launcher differs from reviewed source")
    unit_dir = Path.home() / ".config/systemd/user"
    return {"unit_sha256": {unit: sha256(unit_dir / unit) for unit in (API_UNIT, TUNNEL_UNIT)},
            "launcher_sha256": sha256(launcher),
            "runtime_config_sha256": sha256(config_dir / "runtime.json")}


def service_state():
    """Allowlisted nonsecret state; no ExecStart, environment, credentials or paths."""
    result = {}
    for unit in (API_UNIT, TUNNEL_UNIT):
        p = subprocess.run(["systemctl", "--user", "show", unit, "--property=Id,ActiveState,SubState,UnitFileState,MainPID"], check=True, capture_output=True, text=True, timeout=15)
        values = dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)
        require(values.get("Id") == unit and values.get("ActiveState") == "active" and values.get("UnitFileState") == "enabled", "dedicated services are not active and enabled")
        result[unit] = values
    return result


def activate(expected, sources):
    config_dir = dedicated_config_dir(Path.home() / ".config/path-snap")
    with operation_lock(config_dir, "service-setup.lock"):
        runtime = validate_installed_units(config_dir)
        require(Path(runtime["repo_root"]).resolve() == ROOT and runtime.get("origin") == BASES[0] and runtime.get("hostname") == "ml.d3va.dev", "installed runtime does not identify the authorized checkout/origin")
        expected_python = ROOT / ".venv/bin/python"
        require(Path(runtime["python"]).absolute() == expected_python.absolute(), "installed runtime does not use the original project interpreter")
        before = installed_snapshot(config_dir)
        state_before = service_state()
        original_settings = check_api(config_dir, runtime)
        require(original_settings.get("PATHSNAP_DATASET_ROLE") == "development", "original API dataset role is not development")
        data = Path(original_settings.get("PATHSNAP_DATA_DIR", "Student folder/student_test"))
        require((ROOT / data).resolve() == (ROOT / "Student folder/student_test").resolve(), "original development registry differs from the tested one")
        require(original_settings.get("PATHSNAP_INFERENCE_CONCURRENCY", "1") == "1", "original inference concurrency differs from tested global gate")
        env_file = config_dir / "runtime.env"
        private_file(env_file)
        original = env_file.read_bytes()
        replacement = render_environment(original)
        require(original != replacement, "final evaluator activation is already installed; refusing duplicate activation")
        # Private backup never enters reports or stdout. It contains exact original bytes.
        backup = config_dir / "runtime.env.before-final-five"
        if backup.exists() or backup.is_symlink():
            private_file(backup)
            require(backup.read_bytes() == original, "existing private rollback backup differs; no overwrite permitted")
        else:
            write_exact_private(backup, original)
        verify_manifest(sources)
        acceptance()  # Recheck immutable acceptance immediately before the write.
        require(installed_snapshot(config_dir) == before, "installed unit/launcher/runtime changed during validation")
        def verify():
            current = check_api(config_dir, runtime)
            require(all(current.get(k) == v for k, v in REPLACEMENTS.items()), "managed settings did not install exactly")
            require({k: v for k, v in current.items() if k not in REPLACEMENTS} == {k: v for k, v in original_settings.items() if k not in REPLACEMENTS}, "unrelated environment settings changed")
            origins = check_existing_origins(expected)
            require(installed_snapshot(config_dir) == before, "unit/launcher/runtime bytes changed during activation")
            verify_manifest(sources)
            return origins
        origins = mutate_with_rollback(env_file, original, replacement, verify=verify)
        after = installed_snapshot(config_dir)
        state_after = service_state()
        require(state_after[TUNNEL_UNIT] == state_before[TUNNEL_UNIT], "dedicated tunnel state changed unexpectedly")
        return {"checked_at_utc": utcnow(), "validation_passed": True, "applied": True,
                "lead_go_confirmed": True, "private_environment_values_emitted": False,
                "managed_keys": sorted(REPLACEMENTS), "unrelated_environment_preserved": True,
                "exact_private_backup_saved": True,
                "scope": "only six runtime settings and dedicated API restart; units, launcher, tunnel, DNS and credentials unchanged",
                **expected, "source_sha256": sources, "installed_before": before, "installed_after": after,
                "service_state_before": state_before, "service_state_after": state_after,
                "origins": origins}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--lead-go", choices=[GO_TOKEN])
    args = parser.parse_args(argv)
    require_go(args.apply, args.lead_go)
    expected = acceptance()
    sources = source_manifest(include_frontend=True)
    if not args.apply:
        print(json.dumps({"validation_passed": True, "applied": False, "private_environment_read": False,
                          "network_requests": 0, "service_mutations": 0,
                          "remaining_gate": "explicit Lead GO, installed-unit/env/launcher verification inside locked apply",
                          "managed_keys": sorted(REPLACEMENTS), **expected,
                          "source_sha256": sources}, indent=2, allow_nan=False))
        return
    target = ROOT / FINAL / "deployment_activation.json"
    require(not target.exists() and not target.is_symlink(), "refuse to overwrite activation evidence")
    report = activate(expected, sources)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation_passed": True, "applied": True, "model_sha256": MODEL_SHA256,
                      "config_file_sha256": CONFIG_SHA256, "match_config_hash": COMPACT_CONFIG_SHA256,
                      "base_url": BASES[1], "dataset_counts": {"development": 10, "evaluator": 10},
                      "unrelated_environment_preserved": True, "installed_bytes_unchanged": report["installed_before"] == report["installed_after"]}, indent=2))


if __name__ == "__main__":
    main()
