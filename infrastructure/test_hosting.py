"""Offline deployment safety tests; all credentials and API responses are synthetic."""
from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cloudflare_deploy as cf
import install_user_services as installer
import launch_service as launch
import verify_site as verifier

ACCOUNT = "a" * 32
ZONE = "b" * 32
TUNNEL = "11111111-2222-3333-4444-555555555555"


class FakeAPI:
    def __init__(self, *, records=None, tunnels=None):
        self.records = records or []
        self.tunnels = tunnels or []
        self.mutations = []

    def list(self, path, **query):
        if path.endswith("dns_records"):
            if query.get("name") != cf.HOSTNAME:
                raise AssertionError("Not an exact-hostname DNS read")
            return list(self.records)
        if path.endswith("cfd_tunnel"):
            if query.get("name") != cf.TUNNEL_NAME:
                raise AssertionError("Not a dedicated-name tunnel read")
            return list(self.tunnels)
        raise AssertionError("Unexpected discovery resource")

    def call(self, method, path, body=None, **query):
        self.mutations.append((method, path, body))
        if method == "POST" and path.endswith("cfd_tunnel"):
            self.tunnels = [{"id": TUNNEL, "name": cf.TUNNEL_NAME}]
            return {"result": self.tunnels[0]}
        if method == "POST" and path.endswith("dns_records"):
            self.records = [{**body, "id": "record-owned"}]
            return {"result": self.records[0]}
        if method == "PUT" and path.endswith("/routes"):
            if body.get("overwrite_existing") is not False:
                raise AssertionError("Overwriting existing DNS was requested")
            self.records = [{"name": cf.HOSTNAME, "type": "CNAME",
                             "content": TUNNEL + ".cfargotunnel.com", "proxied": True, "id": "record-owned"}]
            return {"result": {"cname": "new", "name": cf.HOSTNAME}}
        raise AssertionError("Unexpected API mutation")


class HostingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pathsnap-hosting-test-")
        self.root = Path(self.temporary.name)
        self.config = self.root / ".config/path-snap"
        self.home_patch = patch.object(Path, "home", return_value=self.root)
        self.home_patch.start()

    def tearDown(self):
        self.home_patch.stop()
        self.temporary.cleanup()

    def args(self, *, apply=False):
        return argparse.Namespace(config_dir=self.config, apply=apply)

    def private(self, name, content):
        self.config.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.config / name
        installer.write_private(path, content, replace=True)
        return path

    def runtime(self, *, repo_root=None, installer_id="pathsnap-scoped-hosting-v1"):
        """Write a runtime.json bound to this checkout's installer identity."""
        content = {"hostname": "ml.d3va.dev", "origin": "http://127.0.0.1:8000",
                   "installer_id": installer_id,
                   "repo_root": repo_root if repo_root is not None else str(cf.ROOT),
                   "python": "/usr/bin/python3", "cloudflared": "/usr/bin/cloudflared"}
        return self.private("runtime.json", json.dumps(content, indent=2) + "\n")

    def settings(self, **changes):
        settings = {"PATHSNAP_MODEL": "artifacts/model.joblib",
                    "PATHSNAP_MODEL_METADATA": "artifacts/model_metadata.json",
                    "PATHSNAP_MATCH_CONFIG": "artifacts/match_config.json",
                    "PATHSNAP_DATASET_ROLE": "development",
                    "PATHSNAP_CORS_ORIGINS": "https://ml.d3va.dev"}
        settings.update(changes)
        return "\n".join(key + "=" + json.dumps(value) for key, value in settings.items()) + "\n"

    def test_preflight_no_mutations_or_files(self):
        api = FakeAPI()
        summary = cf.deploy(api, self.args(), ACCOUNT, ZONE, "api-token-file")
        self.assertEqual(summary["mode"], "read-only-preflight")
        self.assertEqual(api.mutations, [])
        self.assertFalse(self.config.exists())

    def test_dns_conflict_blocks_before_creating_tunnel(self):
        self.runtime()
        api = FakeAPI(records=[{"name": cf.HOSTNAME, "type": "A", "content": "192.0.2.1"}])
        with self.assertRaisesRegex(cf.DeploymentError, "DNS conflict"):
            cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "api-token-file")
        self.assertEqual(api.mutations, [])
        self.assertFalse((self.config / "deployment.json").exists())
        self.assertFalse((self.config / "creation-intent.json").exists())

    def test_unowned_same_named_tunnel_blocks(self):
        self.runtime()
        api = FakeAPI(tunnels=[{"id": TUNNEL, "name": cf.TUNNEL_NAME}])
        with self.assertRaisesRegex(cf.DeploymentError, "refusing to adopt"):
            cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "api-token-file")
        self.assertEqual(api.mutations, [])

    def test_api_apply_and_rerun_only_create_own_resources(self):
        self.runtime()
        api = FakeAPI()
        summary = cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "api-token-file")
        self.assertEqual([item[0] for item in api.mutations], ["POST", "POST"])
        self.assertFalse(summary["service_started"])
        self.assertNotIn("TunnelSecret", json.dumps(summary))
        self.assertEqual(launch.check_tunnel(self.config), TUNNEL)
        for path in self.config.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "api-token-file")
        self.assertEqual(len(api.mutations), 2)

    def test_origin_cert_apply_route_never_overwrites(self):
        self.runtime()
        api = FakeAPI()
        cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "cloudflared-origin-cert")
        self.assertEqual(api.mutations[-1][0], "PUT")
        self.assertFalse(api.mutations[-1][2]["overwrite_existing"])

    def test_incomplete_owned_state_blocks(self):
        self.runtime()
        self.private("deployment.json", "{}")
        with self.assertRaisesRegex(cf.DeploymentError, "incomplete"):
            cf.deploy(FakeAPI(), self.args(apply=True), ACCOUNT, ZONE, "api-token-file")

    def test_apply_requires_installer_runtime_identity(self):
        api = FakeAPI()
        with self.assertRaisesRegex(cf.DeploymentError, "installer"):
            cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "api-token-file")
        self.assertEqual(api.mutations, [])

    def test_apply_rejects_foreign_checkout_identity(self):
        self.runtime(repo_root=str(self.root))
        api = FakeAPI()
        with self.assertRaisesRegex(cf.DeploymentError, "different checkout or installer"):
            cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "api-token-file")
        self.assertEqual(api.mutations, [])
        self.assertFalse((self.config / "creation-intent.json").exists())

    def test_preflight_rejects_foreign_runtime_identity(self):
        self.runtime(installer_id="pathsnap-other-v0")
        with self.assertRaisesRegex(cf.DeploymentError, "different checkout or installer"):
            cf.deploy(FakeAPI(), self.args(), ACCOUNT, ZONE, "api-token-file")

    def test_apply_refuses_while_another_apply_holds_the_lock(self):
        self.runtime()
        lock_path = self.config / "cloudflare-apply.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            api = FakeAPI()
            with self.assertRaisesRegex(launch.LaunchError, "Another dedicated deployment operation"):
                cf.deploy(api, self.args(apply=True), ACCOUNT, ZONE, "api-token-file")
            self.assertEqual(api.mutations, [])
            self.assertFalse((self.config / "creation-intent.json").exists())
        finally:
            os.close(descriptor)

    def test_owned_record_must_be_proxied_and_exact(self):
        for record in [{"name": cf.HOSTNAME, "type": "CNAME", "content": TUNNEL + ".cfargotunnel.com", "proxied": False},
                       {"name": "other.d3va.dev", "type": "CNAME", "content": TUNNEL + ".cfargotunnel.com", "proxied": True}]:
            with self.assertRaises(cf.DeploymentError):
                cf.assert_records_safe([record], TUNNEL)

    def test_secret_files_not_allowed_in_checkout(self):
        with self.assertRaisesRegex(cf.DeploymentError, "outside"):
            cf.external_path(cf.ROOT / "infrastructure/forbidden-secret")

    def test_symlinked_ancestor_of_external_path_rejected(self):
        real = self.root / "real-credentials"
        real.mkdir()
        (real / "token").write_text("synthetic")
        (self.root / "linked-credentials").symlink_to(real)
        with self.assertRaisesRegex(cf.DeploymentError, "symlinked directories"):
            cf.external_path(self.root / "linked-credentials" / "token")
        self.assertEqual(cf.external_path(real / "token"), (real / "token").absolute())

    def test_shared_config_directory_rejected_without_permission_change(self):
        shared = self.root / ".config"
        shared.mkdir(mode=0o755)
        with self.assertRaises(launch.LaunchError):
            launch.dedicated_config_dir(shared)
        self.assertEqual(shared.stat().st_mode & 0o777, 0o755)

    def test_dedicated_directory_tightens_only_verified_directories(self):
        directory = self.root / "runtime-dir"
        directory.mkdir(mode=0o755)
        launch.dedicated_directory(directory)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        linked = self.root / "linked-dir"
        linked.symlink_to(directory)
        with self.assertRaises(launch.LaunchError):
            launch.dedicated_directory(linked)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)

    def test_static_authorization_never_allows_public_tunnel(self):
        self.private("api-authorized", json.dumps({"phase": "static"}))
        self.assertEqual(launch.api_authorization(self.config), "static")
        with self.assertRaises(launch.LaunchError):
            launch.api_authorization(self.config, require_ready=True)

    def test_secret_file_symlink_rejected(self):
        target = self.private("target", "synthetic")
        link = self.config / "link"
        link.symlink_to(target)
        with self.assertRaises(launch.LaunchError):
            launch.private_file(link)

    def test_world_readable_credential_rejected(self):
        path = self.private("credentials", "synthetic")
        path.chmod(0o644)
        with self.assertRaises(launch.LaunchError):
            launch.private_file(path)

    def test_environment_rejects_unknown_keys_wildcards_and_high_concurrency(self):
        for changes in [{"CF_API_TOKEN": "synthetic"}, {"PATHSNAP_CORS_ORIGINS": "*"},
                        {"PATHSNAP_CORS_ORIGINS": "https://other.invalid"}, {"PATHSNAP_INFERENCE_CONCURRENCY": "8"}]:
            path = self.private("runtime.env", self.settings())
            installer.write_private(path, self.settings(**changes), replace=True)
            with self.assertRaises(launch.LaunchError):
                launch.read_settings(path)

    def test_explicit_evaluator_settings_pass_the_narrow_allowlist(self):
        values = {"PATHSNAP_EVALUATOR_DATA_DIR": "/synthetic/evaluator",
                  "PATHSNAP_EVALUATOR_REFERENCE_DIR": "/synthetic/private-references",
                  "PATHSNAP_EVALUATOR_EXPECTED_TRIP_COUNT": "10"}
        path = self.private("runtime.env", self.settings(**values))
        actual = launch.read_settings(path)
        for key, value in values.items():
            self.assertEqual(actual[key], value)
        invalid = self.private("invalid-evaluator.env", self.settings(PATHSNAP_EVALUATOR_MODEL="forbidden"))
        with self.assertRaises(launch.LaunchError):
            launch.read_settings(invalid)

    def test_environment_values_are_never_executed(self):
        marker = self.root / "must-not-exist"
        path = self.private("runtime.env", self.settings(PATHSNAP_DATA_DIR="$(touch " + str(marker) + ")"))
        self.assertIn("$(touch", launch.read_settings(path)["PATHSNAP_DATA_DIR"])
        self.assertFalse(marker.exists())

    def test_environment_drops_inherited_credentials(self):
        with patch.dict(os.environ, {"CF_API_TOKEN": "synthetic", "OPENBLAS_NUM_THREADS": "99", "LD_PRELOAD": "/synthetic"}):
            env = launch.limited_environment({"PATHSNAP_DATASET_ROLE": "development"})
        self.assertNotIn("CF_API_TOKEN", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_static_phase_explicitly_allows_missing_artifacts(self):
        project = self.root / "project"
        (project / "backend").mkdir(parents=True)
        (project / "frontend").mkdir()
        (project / "backend/main.py").touch()
        (project / "frontend/index.html").touch()
        config = {"repo_root": str(project), "python": "/usr/bin/python3"}
        self.private("runtime.env", self.settings())
        self.private("api-authorized", json.dumps({"phase": "static"}))
        launch.check_api(self.config, config)
        installer.write_private(self.config / "api-authorized", json.dumps({"phase": "ready"}), replace=True)
        with self.assertRaisesRegex(launch.LaunchError, "missing or empty"):
            launch.check_api(self.config, config)

    def test_origin_cert_parse_never_emits_token(self):
        token = "synthetic-private-test-token"
        payload = base64.b64encode(json.dumps({"apiToken": token, "accountID": ACCOUNT, "zoneID": ZONE}).encode()).decode()
        path = self.private("origin-cert.pem", "-----BEGIN ARGO TUNNEL TOKEN-----\n" + payload + "\n-----END ARGO TUNNEL TOKEN-----\n")
        args = argparse.Namespace(token_file=None, origincert=path, account_id=None, zone_id=None)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = cf.load_authorization(args)
        self.assertEqual(result[:3], (token, ACCOUNT, ZONE))
        self.assertEqual(output.getvalue(), "")

    def test_unit_paths_not_shell_quoted(self):
        self.assertEqual(installer.unit_path("/home/user/project with space"), "/home/user/project with space")
        self.assertEqual(installer.quote_unit("/home/user/project with space"), chr(34) + "/home/user/project with space" + chr(34))
        with self.assertRaises(launch.LaunchError):
            installer.unit_path("/home/user/project\nExecStart=bad")

    def test_readiness_rejects_redirect_handler(self):
        handler = launch.NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, "redirect", {}, "http://127.0.0.1:8080/health"))

    def test_tunnel_launcher_requires_explicit_ready_phase(self):
        self.private("api-authorized", json.dumps({"phase": "static"}))
        with patch.object(launch, "read_runtime", return_value={}), patch.object(launch, "check_tunnel", return_value=TUNNEL), patch.object(launch, "wait_for_api") as wait:
            with contextlib.redirect_stderr(io.StringIO()):
                result = launch.main(["tunnel", "--config-dir", str(self.config)])
        self.assertEqual(result, 1)
        wait.assert_not_called()

    def test_verifier_detects_private_keys_recursively(self):
        found = verifier.forbidden_keys({"safe": [{"ground_truth_route": [1]}, {"TunnelSecret": "synthetic"}]})
        self.assertEqual(found, {"ground_truth_route", "TunnelSecret"})

    def test_identifier_validation_rejects_paths_and_missing_zone(self):
        with self.assertRaises(cf.DeploymentError):
            cf.validate_id("../../zones/unrelated", "zone ID")
        with self.assertRaises(cf.DeploymentError):
            cf.validate_id(123, "zone ID")


if __name__ == "__main__":
    unittest.main()
