# Original two-laptop evaluation alternative

The current requested installation uses [the public same-origin website](<DEPLOYMENT.md>)
instead of requiring a second physical laptop. This alternative retains the
original guideline topology without copying references to the browser machine.

## Laptop 1: prediction and private evaluation

Install [the serving requirements](<requirements.txt>), trained artifacts, graph/
GPS inputs and private references. Use exactly ten trips in the configured root.
On a trusted LAN only:

```bash
export PATHSNAP_HOST=0.0.0.0
export PATHSNAP_PORT=8000
export PATHSNAP_CORS_ORIGINS=http://LAPTOP2_LAN_IP:8090
bash deploy/start_backend.sh
```

Replace the origin placeholder with the exact browser origin, without a trailing
slash. If the browser opens localhost on Laptop 2 instead, allow that exact origin
instead. Do not use wildcard CORS. Permit inbound 8000 only from the trusted LAN
or Laptop 2 firewall address. Keep all weights/references server-side.

## Laptop 2: static UI only

Copy only the frontend and safe deployment/client files, not the datasets,
references, artifacts, outputs or server secrets. The helper below imports only
the stdlib static-asset policy in [the static policy module](<../backend/static.py>),
so include that module and [its package initializer](<../backend/__init__.py>)
with [the safe UI server](<serve_ui.py>) if deploying a minimal client folder.
At the matching directory root:

```bash
python3 -B deploy/serve_ui.py --host 0.0.0.0 --port 8090
```

Use an available client port; 8090 is the example because 8080 is already used by
an unrelated process on the current host. Open `http://LAPTOP2_LAN_IP:8090/`,
expand API settings, and enter `http://LAPTOP1_LAN_IP:8000`. The UI persists this
base URL locally; it has no training labels or route predictions embedded in it.
An HTTPS page must use an HTTPS API to avoid mixed-content failures.

The safe UI server exposes only exact UI HTML/CSS/modules/favicon and Leaflet
assets. Do not replace it with a repository-root generic HTTP file server.

## Demonstration checklist

1. Laptop 1 `GET /health` says ready and model loaded, with accurate dataset role.
2. The UI obtains dataset sections from Laptop 1 `GET /datasets`, then its ten IDs from `GET /datasets/{id}/trips`; no fixed ID list.
3. Selecting a trip sends only `trip_id` to `POST /datasets/{id}/match`.
4. Laptop 1 runs fresh classifier+HMM inference, then private scoring.
5. Laptop 2 displays raw GPS, predicted route and aggregate proxy-labelled scores.
6. Repeat requests produce distinct request IDs and fresh model work.
7. Capture the UI and Laptop 1 completion log for matching request IDs.

The ten subsequently supplied evaluator trips are a separate section, configured only after the final model and decoder freezes through the [documented evaluator handoff](<DEPLOYMENT.md>). Raw inputs are not included in the public checkout. Do not relabel student-development results as evaluator results. A physical two-laptop venue run remains a human task; the delivered public website is the user-authorized substitute.
