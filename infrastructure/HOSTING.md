# PathSnap hosting: ml.d3va.dev

## Deployment contract

The current user instruction replaces the original physical two-laptop deployment. One process serves the evaluator UI and API on the same origin:

```text
https://ml.d3va.dev
  -> Cloudflare HTTPS edge
  -> dedicated named tunnel: path-snap
  -> http://127.0.0.1:8000
  -> backend.main:app (FastAPI + allowlisted frontend assets)
```

No router port forwarding or changes to existing web-server configurations are required. Ports 80, 8080, 4000 and 3001 belong to unrelated services and must not be stopped, rebound or reconfigured. The deployment scripts are fixed to this one hostname, tunnel name and local origin. They do not manage any other website.

The latest measured state is [deployment-status.json](<evidence/deployment-status.json>). Successful static-stage checks are recorded in [local-static.json](<evidence/local-static.json>). A configured service is not evidence of public reachability: check the timestamps and individual results. A public-live claim requires successful HTTPS, ready model health and a real POST match.

## Files and ownership

- [install_user_services.py](<install_user_services.py>) installs and operates only the two PathSnap user units.
- [launch_service.py](<launch_service.py>) validates private runtime files, clears inherited environment and starts the API or tunnel.
- [cloudflare_deploy.py](<cloudflare_deploy.py>) performs read-only Cloudflare preflight by default, or a narrowly scoped explicit apply.
- [verify_site.py](<verify_site.py>) checks the actual local/public website, API, learned inference and private-file denial.
- [status_snapshot.py](<status_snapshot.py>) captures a read-only status snapshot, including real public HTTPS probes.
- [test_hosting.py](<test_hosting.py>) exercises deployment safety with synthetic credentials and an offline fake API.

Private runtime data lives outside the checkout in [the dedicated configuration directory](</home/d3va/.config/path-snap>): mode 700 for the directory and mode 600 for its files. It contains runtime paths, non-secret environment settings, the installed launcher, readiness authorization and, only after Cloudflare provisioning succeeds, a tunnel manifest, ingress configuration and tunnel credentials. The account-wide origin certificate remains in its existing private authorization location. No credential belongs in Git, a command argument, the report, screenshots or chat. The dedicated directory must be an exact, non-symlinked path: credential and state files may not traverse symlinked directories, and the directory mode is tightened to 700 only after lstat re-verifies it is a real directory owned by the current user — the scripts never chmod an arbitrary or symlink-followed path.

The unit templates are [API](<templates/path-snap-api.service.in>) and [tunnel](<templates/path-snap-tunnel.service.in>); runtime defaults are in [runtime.env.example](<templates/runtime.env.example>). The installed units are named `path-snap-api.service` and `path-snap-tunnel.service`.

## Local installation and readiness

Run commands from the checkout root using the current user. Create the project virtual environment and install the pinned application dependencies using the project installation instructions first. The scripts require only standard-library Python, systemd user services, `cloudflared`, and `dig` for status capture.

```bash
python3 -B infrastructure/install_user_services.py install
python3 -B infrastructure/install_user_services.py status
loginctl --no-ask-password enable-linger "$USER"
```

Installation enables both units but starts neither. The API and tunnel have `Restart=always`, bounded restart delays, graceful shutdown and a fresh environment. No inherited Cloudflare token, shell setting, proxy or loader variable reaches either process. BLAS/OpenMP/NumExpr thread counts are fixed to one. Uvicorn uses one worker, one backend inference slot, a connection concurrency limit of 16, five-second keep-alive and trusted proxy headers only from 127.0.0.1. The API binds only 127.0.0.1:8000. Tunnel metrics bind an ephemeral loopback port, never an external interface.

On the deployed account, login linger was enabled successfully without a password prompt. This allows enabled user units to run after logout and at boot. A real machine reboot was not performed; enabled units and `Linger=yes` are the verified persistence evidence. The host must stay powered, awake and connected to the Internet.

An explicitly authorized static/notready stage can serve the UI while training runs:

```bash
python3 -B infrastructure/install_user_services.py authorize-start-api --allow-notready
python3 -B infrastructure/verify_site.py --local --static-only
```

This stage must not be described as a working trained-model deployment. Its health reports missing-model/config errors; inference does not substitute a baseline. A static authorization can never start the public tunnel, even if the API later discovers model files.

After the trained model, model metadata and final learned-emission decoder configuration exist and the integration lead confirms readiness:

```bash
python3 -B infrastructure/install_user_services.py authorize-start-api
systemctl --user --no-ask-password restart path-snap-api.service
python3 -B infrastructure/verify_site.py --local --all-trips \
  --output infrastructure/evidence/local-ready.json
```

The ready-stage authorization refuses missing/empty artifacts and does not evict another listener from port 8000. Local runtime settings are parsed as data, never sourced as shell commands. Their paths may point to a private evaluator package; set the actual dataset role (`development` or `evaluator`) accurately. Same-origin UI calls use relative API URLs. Only the exact optional HTTPS origin `https://ml.d3va.dev` is permitted in the deployment CORS setting, never a wildcard. The backend rejects untrusted Origin-bearing match requests before inference.

## Cloudflare authorization and conflict-safe provisioning

The initial machine inspection found no cloudflared origin certificate, Wrangler authorization, matching unlocked-keyring entry or CF token environment name. The authenticated tunnel-list CLI failed because no origin certificate existed. The authoritative Cloudflare nameservers for d3va.dev are `tanner.ns.cloudflare.com` and `cortney.ns.cloudflare.com`. At initial inspection, ml.d3va.dev and a fresh unrelated test subdomain both resolved to parking address 91.195.240.94; this suggests a wildcard, but does not prove an exact hostname record is absent. No parking or other DNS record was overwritten. As of 2026-11-18 a local origin authorization exists (recorded in [cloudflare-preflight.json](<evidence/cloudflare-preflight.json>)); the read-only preflight confirmed an active zone, no exact-hostname record and no existing `path-snap` tunnel. Apply was deliberately not run at that point because it must follow the ready-stage authorization and the coordinated network bring-up, not merely the existence of credentials.

Authorization must be installed locally, never pasted into chat. An account owner can perform the normal Cloudflare `cloudflared tunnel login` flow for d3va.dev on this computer. The deployment agent does not initiate an interactive login or request privilege prompts. Existing private origin authorization can then be used by path, without displaying its contents. Alternatively, use an outside-checkout mode-600 API-token file with:

- Zone Read for the d3va.dev zone (verify the actual zone name/account).
- Zone DNS Read and Edit for d3va.dev (inspect the exact ml.d3va.dev record before creating a CNAME).
- Account Cloudflare Tunnel Edit for the owning account (list/create the dedicated tunnel).

Cloudflare may scope DNS token permissions at the zone rather than a single record; the submitted client enforces the narrower hostname scope. Account and zone IDs are metadata, not token values. Do not grant or store a global API key just for this service. A cloudflared origin certificate may not permit the general zone/DNS read endpoints in some accounts: in that case preflight stops before all mutations; use a properly scoped token rather than bypassing the exact-record check.

```bash
# Read-only preflight; standard origin certificate is discovered automatically.
python3 -B infrastructure/cloudflare_deploy.py

# Alternative: existing private scoped API-token file; never put its value here.
python3 -B infrastructure/cloudflare_deploy.py \
  --token-file "$HOME/.config/path-snap/cloudflare-api-token" \
  --account-id ACCOUNT_ID --zone-id ZONE_ID

# Explicitly provision only if preflight is conflict-free.
python3 -B infrastructure/cloudflare_deploy.py --apply
```

With token-file authorization, repeat the same token-file/ID options for apply. The account-wide token is never written to the checkout or passed to cloudflared. The runtime tunnel credential grants only the dedicated tunnel connection.

Preflight checks active zone ownership, exact ml.d3va.dev DNS records and the dedicated `path-snap` tunnel. Any pre-existing conflicting DNS record, unowned same-name tunnel, wrong account, missing credential, unexpected filter result or partial local state stops deployment. There is no overwrite/delete DNS mode. Origin-certificate routing uses Cloudflare atomic `overwrite_existing:false`; token routing uses a new CNAME create and never update. A repeat run only accepts the locally owned tunnel and its exact proxied CNAME. No existing shared tunnel ingress is edited.

Apply additionally requires a prior `install` of this checkout: the dedicated directory must contain a `runtime.json` carrying `installer_id: pathsnap-scoped-hosting-v1` and this checkout's `repo_root`. Preflight validates that identity whenever the file exists. The entire mutation section runs under an exclusive local `cloudflare-apply.lock` (non-blocking flock), and every ownership fact — local state, exact DNS records and same-name tunnels — is re-read inside the lock, so a concurrent apply cannot race past the conflict checks. A held lock stops the run before any API mutation.

A private creation intent is saved before tunnel creation so an ambiguous network failure cannot silently cause duplicate creation. If apply is interrupted between tunnel creation and state persistence, the next run refuses to adopt the unknown tunnel; privately reconcile its identity and dedicated secret before retrying. Never delete unrelated tunnels or DNS to make preflight pass. The script does not attempt destructive automatic rollback.

## Public activation and verification

After successful Cloudflare provisioning and ready-stage local health:

```bash
python3 -B infrastructure/install_user_services.py start-tunnel
python3 -B infrastructure/verify_site.py --all-trips \
  --output infrastructure/evidence/public-ready.json
python3 -B infrastructure/status_snapshot.py
```

The tunnel launcher requires explicit ready authorization and waits for `status:ready`, `model_loaded:true` and ten trips before connecting. Ingress accepts only ml.d3va.dev, forwards to loopback8000 and ends with a 404 catchall. It does not expose the repository, dataset, artifacts, arbitrary local ports or general file listings. The same-origin backend serves only an explicit UI asset allowlist. TLS validation is never disabled for public checks.

The verifier checks the genuine UI and assets, ready model/config hashes, ten dynamic IDs, hostile Origin rejection, trained-emission use, nonempty continuous reconstructed route, server-private field absence and a real match request. Saved evidence omits raw route/GPS coordinates and credentials, recording request IDs, hashes, diagnostics, metrics and measured timings. It never invents public success when DNS/auth is unavailable.

## Operations and bounded rollback

```bash
systemctl --user --no-ask-password status path-snap-api.service path-snap-tunnel.service
journalctl --user -u path-snap-api.service -u path-snap-tunnel.service --no-pager -n 60
systemctl --user --no-ask-password restart path-snap-api.service
systemctl --user --no-ask-password restart path-snap-tunnel.service

# Stop this website only; do not restart unrelated servers.
systemctl --user --no-ask-password disable --now path-snap-tunnel.service path-snap-api.service
```

Cloudflared logging remains at info, never debug; debug logging can include request headers. Do not publish private logs or the configuration directory. Cloudflare deletion is deliberately not automated: stop the dedicated tunnel, privately verify the record/tunnel IDs in its deployment manifest, and remove only the created ml.d3va.dev record and dedicated tunnel through authorized account tooling if decommissioning is requested. Do not remove a wildcard, other subdomain, shared tunnel or existing system service.

Run the offline infrastructure tests without Cloudflare access:

```bash
python3 -B -m unittest discover -s infrastructure -p test_hosting.py -v
```
