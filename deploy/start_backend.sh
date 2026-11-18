#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "${ROOT:?}"
PYTHON="${PATHSNAP_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Set PATHSNAP_PYTHON to an installed interpreter or create .venv first.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PATHSNAP_INFERENCE_CONCURRENCY="${PATHSNAP_INFERENCE_CONCURRENCY:-1}"
# Defaults bind only loopback. Use PATHSNAP_HOST=0.0.0.0 only for a trusted LAN.
exec "$PYTHON" -B -m uvicorn backend.main:app \
  --host "${PATHSNAP_HOST:-127.0.0.1}" --port "${PATHSNAP_PORT:-8000}" \
  --workers 1 --no-access-log --limit-concurrency 16 --timeout-keep-alive 5 \
  --proxy-headers --forwarded-allow-ips 127.0.0.1,::1
