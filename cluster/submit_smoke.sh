#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${PARTITION:?Set PARTITION to the name of a GPU partition. See docs/CLUSTER_RUNBOOK.md.}"
gpus="${GPUS:-1}"
time_limit="${TIME:-01:00:00}"
cpus="${CPUS:-8}"
memory="${MEM:-32G}"

if [ -n "${GPU_TYPE:-}" ]; then
  gpu_request="gpu:${GPU_TYPE}:${gpus}"
else
  gpu_request="gpu:${gpus}"
fi

arguments=(
  --job-name quantrl-smoke
  --partition "$PARTITION"
  --gres "$gpu_request"
  --nodes 1
  --ntasks 1
  --cpus-per-task "$cpus"
  --mem "$memory"
  --time "$time_limit"
  --output "runs/slurm-%j.out"
  --export ALL
)
if [ -n "${ACCOUNT:-}" ]; then
  arguments+=(--account "$ACCOUNT")
fi
if [ -n "${QOS:-}" ]; then
  arguments+=(--qos "$QOS")
fi

mkdir -p runs
echo "sbatch ${arguments[*]} cluster/smoke.sbatch"
sbatch "${arguments[@]}" cluster/smoke.sbatch
