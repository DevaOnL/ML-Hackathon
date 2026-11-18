#!/usr/bin/env python3
"""Switch only the dedicated installed API after model and ten-route acceptance.

Default is validation only. --apply atomically updates only three allowlisted
artifact settings, restarts path-snap-api.service, and checks the existing local
and public origins. It never installs a unit, changes DNS/tunnel, or starts a
replacement server. On failure it restores the exact prior runtime environment.
No runtime configuration values or credentials are printed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infrastructure"))
from install_user_services import operation_lock, validate_installed_units, write_private
from launch_service import check_api, dedicated_config_dir

REPLACEMENTS = {
    "PATHSNAP_MODEL": "artifacts/phase_three/model.joblib",
    "PATHSNAP_MODEL_METADATA": "artifacts/phase_three/model_metadata.json",
    "PATHSNAP_MATCH_CONFIG": "artifacts/phase_three/match_config.json",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_environment(original):
    seen = set()
    lines = []
    for line in original.splitlines():
        stripped = line.strip()
        key, separator, _ = stripped.partition("=")
        if separator and not stripped.startswith("#") and key in REPLACEMENTS:
            if key in seen:
                raise ValueError("repeated artifact setting")
            seen.add(key)
            line = key + "=" + json.dumps(REPLACEMENTS[key])
        lines.append(line)
    if seen != set(REPLACEMENTS):
        raise ValueError("explicit artifact settings are missing")
    return "\n".join(lines) + "\n"


def acceptance():
    model = ROOT / REPLACEMENTS["PATHSNAP_MODEL"]
    config = ROOT / REPLACEMENTS["PATHSNAP_MATCH_CONFIG"]
    classifier = load_json(ROOT / "reports/phase_three/model_acceptance.json")
    routes = load_json(ROOT / "reports/phase_three/offline_delivery_verification.json")
    frozen = load_json(ROOT / "artifacts/phase_three/model_freeze.json")
    decoder = load_json(ROOT / "reports/phase_three/decoder/freeze.json")
    api = load_json(ROOT / "reports/phase_three/api_integration_acceptance.json")
    if not classifier.get("passed") or not routes.get("passed") or not api.get("passed"):
        raise ValueError("actual classifier, all-ten routes and API integration must all pass")
    if not classifier.get("selected_model_clean_environment", {}).get("passed"):
        raise ValueError("actual selected-model clean-environment probability equality must pass")
    if api.get("trip_count") != 10 or api.get("mocked_model_or_inference") is not False:
        raise ValueError("actual unmocked API inference must be tested on all ten trips")
    actual_hash = sha256(model)
    if not (actual_hash == classifier["frozen_model_sha256"] == routes["model_sha256"] == frozen["sha256"] == decoder["model_sha256"] == api["model_sha256"]):
        raise ValueError("tested frozen model identity mismatch")
    if not (sha256(config) == routes["config_file_sha256"] == decoder["config_file_sha256"]):
        raise ValueError("tested frozen decoder identity mismatch")
    families = {m["family"] for m in classifier["models"]}
    if families != {"logistic_regression", "knn_distance_weighted", "bagging"} or any(m["tested_trip_count"] != 10 for m in classifier["models"]):
        raise ValueError("requested three families were not all tested on ten trips")
    if api["match_config_hash"] != routes["api_compact_config_sha256"]:
        raise ValueError("actual API decoder identity mismatch")
    selected = classifier["selected_family"]
    metadata = load_json(ROOT / REPLACEMENTS["PATHSNAP_MODEL_METADATA"])
    if metadata["model_name"] != selected:
        raise ValueError("model sidecar is not the selected family")
    return {"model_sha256": actual_hash, "model_name": selected,
            "match_config_hash": routes["api_compact_config_sha256"],
            "config_file_sha256": sha256(config)}


def health_matches(health, expected):
    return (health.get("status") == "ready" and health.get("model_loaded") is True
            and health.get("model_hash") == expected["model_sha256"]
            and health.get("model_name") == expected["model_name"]
            and health.get("match_config_hash") == expected["match_config_hash"]
            and health.get("trip_count") == 10 and health.get("prediction_cache") is False)


def read_health(base):
    request = urllib.request.Request(base + "/health", headers={"Cache-Control":"no-cache", "User-Agent":"PathSnap-verified-model-activation/1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read())


def check_existing_origins(expected):
    checked = {}
    # Short retry only covers a real service restart, not training or background jobs.
    for base in ("http://127.0.0.1:8000", "https://ml.d3va.dev"):
        for attempt in range(8):
            try:
                health = read_health(base)
                if not health_matches(health, expected):
                    raise ValueError("ready health does not identify the tested artifact/config")
                checked[base] = {key:health[key] for key in ("status","model_name","model_hash","match_config_hash","trip_count","prediction_cache")}
                break
            except (OSError, ValueError):
                if attempt == 7:
                    raise
                time.sleep(.4 * (attempt+1))
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase_three/deployment_activation.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse to overwrite activation evidence")
    config_dir = dedicated_config_dir(Path.home() / ".config/path-snap")
    expected = acceptance()
    runtime = validate_installed_units(config_dir)
    if Path(runtime["repo_root"]).resolve() != ROOT:
        raise ValueError("dedicated services belong to another checkout")
    previous = check_api(config_dir, runtime)
    env_file = config_dir / "runtime.env"
    original = env_file.read_text()
    replacement = render_environment(original)
    report = {"checked_at_utc":datetime.now(timezone.utc).isoformat(), "validation_passed":True,
              "applied":False,"scope":"only dedicated API artifact paths and API restart; tunnel, DNS, credentials and unit bytes unchanged", **expected}
    if args.apply:
        with operation_lock(config_dir, "service-setup.lock"):
            if env_file.read_text() != original:
                raise RuntimeError("runtime changed after validation")
            validate_installed_units(config_dir)
            write_private(env_file, replacement, replace=True)
            try:
                installed = check_api(config_dir, runtime)
                if any(installed[k] != v for k,v in REPLACEMENTS.items()):
                    raise RuntimeError("artifact setting did not install exactly")
                subprocess.run(["systemctl","--user","restart","path-snap-api.service"],check=True)
                report["origins"] = check_existing_origins(expected)
                report["applied"] = True
            except Exception:
                write_private(env_file, original, replace=True)
                subprocess.run(["systemctl","--user","restart","path-snap-api.service"],check=True)
                raise RuntimeError("activation failed; exact previous runtime was restored and dedicated API restarted") from None
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
