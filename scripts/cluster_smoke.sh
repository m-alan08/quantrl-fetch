#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

torch_variant="${TORCH_VARIANT:-auto}"
render="${RENDER:-1}"
run_tests="${RUN_TESTS:-1}"
stamp="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
out_dir="${OUT_DIR:-runs/cluster_smoke_$stamp}"
mkdir -p "$out_dir"
summary="$out_dir/summary.txt"
: > "$summary"
failures=0

record() {
  printf '%-6s %s\n' "$1" "$2" | tee -a "$summary"
}

run_stage() {
  local name="$1"
  shift
  local log="$out_dir/$name.log"
  echo "===== $name: $*"
  if "$@" > "$log" 2>&1; then
    record PASS "$name (log: $log)"
  else
    record FAIL "$name (log: $log, last lines below)"
    tail -n 25 "$log" | sed 's/^/        /' | tee -a "$summary"
    failures=$((failures + 1))
  fi
}

gpu_tool=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  gpu_tool="nvidia-smi"
elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
  gpu_tool="rocm-smi"
fi

{
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "system: $(uname -a)"
  echo "slurm job: ${SLURM_JOB_ID:-none}, partition: ${SLURM_JOB_PARTITION:-none}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
  echo "MUJOCO_GL=${MUJOCO_GL:-unset} DISPLAY=${DISPLAY:-unset}"
  echo "uv: $(command -v uv >/dev/null 2>&1 && uv --version || echo missing)"
  echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "gpu tool: ${gpu_tool:-none found}"
  if [ -n "$gpu_tool" ]; then
    echo
    "$gpu_tool"
  fi
} > "$out_dir/machine.txt" 2>&1
cat "$out_dir/machine.txt"

run_stage install scripts/install.sh "$torch_variant"
if [ ! -x .venv/bin/python ]; then
  record FAIL "no environment was created, stopping"
  echo "CLUSTER SMOKE FAIL" | tee -a "$summary"
  exit 1
fi
python=".venv/bin/python"

run_stage torch_report "$python" - <<'PYTHON'
import json

import torch

from quantrl.utils.device import available_devices, describe_device, resolve_device

device = resolve_device("auto")
print(json.dumps(describe_device(device), indent=2))
print("cuda available:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
PYTHON
cat "$out_dir/torch_report.log"

if [ -n "$gpu_tool" ]; then
  run_stage gpu_visible_to_torch "$python" -c "
import torch
assert torch.cuda.is_available(), (
    'a GPU tool works on this node but torch cannot see a GPU: torch ' + torch.__version__
    + ', CUDA build ' + str(torch.version.cuda) + ', ROCm build ' + str(getattr(torch.version, 'hip', None))
)
x = torch.randn(256, 256, device='cuda')
print('GPU matmul ok on', torch.cuda.get_device_name(0), float((x @ x).sum()))
"
fi

if [ "$run_tests" = "1" ]; then
  run_stage unit_tests "$python" -m pytest -q -rs
fi

render_flags=()
if [ "$render" = "1" ]; then
  render_flags=(--render --render-backend auto)
fi
run_stage smoke_auto_device "$python" scripts/smoke_test.py --device auto ${render_flags[@]+"${render_flags[@]}"} \
  --output-dir "$out_dir/smoke_auto" --report "$out_dir/smoke_auto.json"
if [ -n "$gpu_tool" ]; then
  run_stage smoke_cuda "$python" scripts/smoke_test.py --device cuda \
    --output-dir "$out_dir/smoke_cuda" --report "$out_dir/smoke_cuda.json"
fi
run_stage smoke_cpu_forced "$python" scripts/smoke_test.py --device cpu \
  --output-dir "$out_dir/smoke_cpu" --report "$out_dir/smoke_cpu.json"
run_stage smoke_gpus_hidden env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
  "$python" scripts/smoke_test.py --device auto \
  --output-dir "$out_dir/smoke_hidden" --report "$out_dir/smoke_hidden.json"
run_stage hidden_gpus_fall_back_to_cpu "$python" -c "
import json, sys
report = json.load(open('$out_dir/smoke_hidden.json'))
assert report['device'] in ('cpu', 'mps'), 'with GPUs hidden the smoke test ran on ' + report['device']
print('with GPUs hidden, auto chose', report['device'])
"

echo
echo "Summary written to $summary"
if [ "$failures" -eq 0 ]; then
  echo "CLUSTER SMOKE PASS" | tee -a "$summary"
  exit 0
fi
echo "CLUSTER SMOKE FAIL ($failures stages)" | tee -a "$summary"
exit 1
