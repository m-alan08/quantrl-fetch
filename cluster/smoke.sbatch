#!/bin/bash
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -n "${CLUSTER_SETUP:-}" ]; then
  eval "$CLUSTER_SETUP"
fi
exec scripts/cluster_smoke.sh
