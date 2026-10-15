# Cluster runbook

This is the checklist for the first day on a Slurm GPU cluster (for example Georgia Tech PACE or the HCE cluster). It runs the same checks that pass on the development Mac: install, unit tests, the end-to-end smoke test on the GPU, a forced-CPU run, and a run with the GPUs hidden. Nothing here has been run on a GPU cluster yet; see "What has and has not been verified" at the end.

## Values you need to fill in

The scripts never contain cluster-specific values. You provide them as environment variables when you run them.

| Variable | Required | What to put | How to find it |
| --- | --- | --- | --- |
| `PARTITION` | yes | Name of a partition with GPUs | `sinfo -o "%P %G %l"` lists partitions, their GPUs (the `%G` column), and time limits |
| `ACCOUNT` | if your site requires one | Slurm account or charge code | `sacctmgr show associations user=$USER format=account,qos` |
| `QOS` | if your site requires one | Quality of service name | same command as `ACCOUNT` |
| `GPU_TYPE` | no | GPU type in `--gres`, for example `h100` or `a100` | the `%G` column of `sinfo` |
| `GPUS` | no | Number of GPUs, default `1` | one is enough for the smoke test |
| `TIME` | no | Time limit, default `01:00:00` | the smoke test takes a few minutes |
| `CPUS` | no | CPU cores, default `8` | |
| `MEM` | no | Memory, default `32G` | |
| `CLUSTER_SETUP` | no | Shell commands to run first on the node, for example `module load something` | Usually not needed: uv brings its own Python, and the torch wheels bundle the CUDA or ROCm runtime. Only the GPU driver must exist on the node. |
| `TORCH_VARIANT` | no | `auto` (default), `cu126`, `cu130`, `rocm`, or `cpu` | See "Choosing the torch build" below |
| `RENDER` | no | `1` (default) to also render a video with EGL or OSMesa, `0` to skip | Set `0` if the node has neither library |
| `RUN_TESTS` | no | `1` (default) to run the unit tests, `0` to skip | |

## Choosing the torch build

- `auto` installs `cu126` if `nvidia-smi` works on the node, `rocm` if `rocm-smi` works, and `cpu` otherwise.
- `cu126` (CUDA 12.6) is the safe default for NVIDIA GPUs. NVIDIA's compatibility rules for CUDA 12 builds require driver 525.60 or newer.
- `cu130` (CUDA 13.0) requires driver 580 or newer.
- `rocm` (ROCm 7.2) is for AMD GPUs and exists only for x86_64 nodes.

The driver version is the top line of `nvidia-smi`. If it is older than the minimum above, use the other CUDA build, or tell me the driver version.

## Step by step

### 1. Log in and get the code

Replace the angle-bracket parts with your own values.

```bash
ssh <your-username>@<login-node>
```

```bash
cd <a directory with a few GB of space, often a scratch or project directory>
```

```bash
git clone https://github.com/m-alan08/quantrl-fetch.git
```

```bash
cd quantrl-fetch
```

### 2. Install uv if it is missing

```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
```

If it was just installed, open a new shell or run `export PATH="$HOME/.local/bin:$PATH"`.

If your home directory has a small quota, point uv's download cache at the same large directory as the code:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
```

### 3. Install on the login node

Compute nodes often have no internet access, so install while you still have it. Use the variant you chose above; `cu126` is the usual choice for NVIDIA.

```bash
scripts/install.sh cu126
```

The last lines print the torch version and the devices torch can see. On a login node without a GPU, `available devices: ['cpu']` is expected.

### 4a. Run the smoke test as a batch job (recommended)

```bash
PARTITION=<gpu-partition> ACCOUNT=<account-if-needed> QOS=<qos-if-needed> TORCH_VARIANT=cu126 cluster/submit_smoke.sh
```

The script prints the exact `sbatch` command it ran. Then:

```bash
squeue -u $USER
```

When the job leaves the queue, read the summary (replace `<jobid>` with the number `sbatch` printed):

```bash
cat runs/slurm-<jobid>.out
```

```bash
cat runs/cluster_smoke_<jobid>/summary.txt
```

### 4b. Or run it interactively on a GPU node

```bash
salloc --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00
```

Some sites start a shell on the compute node automatically; others need `srun --pty bash` after `salloc`. Then:

```bash
TORCH_VARIANT=cu126 scripts/cluster_smoke.sh
```

## What the smoke test runs

| Stage | What it checks |
| --- | --- |
| `install` | `scripts/install.sh` creates the environment from the lockfile and prints the torch build and visible devices |
| `torch_report` | The device that `auto` chooses, the torch build, and the visible devices |
| `gpu_visible_to_torch` | Only when `nvidia-smi` or `rocm-smi` works: torch can see the GPU and run a matrix multiply on it |
| `unit_tests` | The full pytest suite, including the CUDA versions of the device tests that are skipped on the Mac |
| `smoke_auto_device` | `scripts/smoke_test.py --device auto`, which should pick the GPU: tiny fp32 training, checkpoint reload and evaluation, resume, PTQ at 8 and 4 bits, QAT at 8 and 4 bits, plots, and with `RENDER=1` a headless video |
| `smoke_cuda` | The same with `--device cuda` explicitly |
| `smoke_cpu_forced` | The same on the CPU, to confirm nothing depends on the GPU |
| `smoke_gpus_hidden` | The same with `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and `ROCR_VISIBLE_DEVICES` empty and `--device auto` |
| `hidden_gpus_fall_back_to_cpu` | That the hidden-GPU run really ran on the CPU |

## What a pass looks like

The last line of `summary.txt` is `CLUSTER SMOKE PASS`, and every line above it starts with `PASS`. On a GPU node you should also see:

- `gpu_visible_to_torch` and `smoke_cuda` in the list;
- in `torch_report.log`, a device of `cuda` and your GPU's name under `device_name`;
- in `smoke_auto_device.log`, a first step that starts with `device cuda`.

For reference, this is the summary from a run of `scripts/cluster_smoke.sh` on the development Mac. It has no GPU stages, because the Mac has no NVIDIA or AMD GPU.

```
PASS   install (log: runs/cluster_smoke_mac/install.log)
PASS   torch_report (log: runs/cluster_smoke_mac/torch_report.log)
PASS   unit_tests (log: runs/cluster_smoke_mac/unit_tests.log)
PASS   smoke_auto_device (log: runs/cluster_smoke_mac/smoke_auto_device.log)
PASS   smoke_cpu_forced (log: runs/cluster_smoke_mac/smoke_cpu_forced.log)
PASS   smoke_gpus_hidden (log: runs/cluster_smoke_mac/smoke_gpus_hidden.log)
PASS   hidden_gpus_fall_back_to_cpu (log: runs/cluster_smoke_mac/hidden_gpus_fall_back_to_cpu.log)
CLUSTER SMOKE PASS
```

A failing stage shows `FAIL` with the last 25 lines of its log underneath, and the last line is `CLUSTER SMOKE FAIL (<n> stages)`. For example, a simulated GPU node where torch could not see the GPU produced:

```
FAIL   gpu_visible_to_torch (log: ..., last lines below)
        AssertionError: a GPU tool works on this node but torch cannot see a GPU: torch 2.14.1, CUDA build None, ROCm build None
```

## If it fails: what to send back

Make one archive of everything the run wrote. Replace `<jobid>` with the job number, or with the timestamp in the folder name for an interactive run.

```bash
tar czf cluster_smoke_report.tgz runs/cluster_smoke_<jobid> runs/slurm-<jobid>.out
```

Send that file, plus the exact command you ran (including the variables in front of it). The archive already contains:

| File | Content |
| --- | --- |
| `summary.txt` | The PASS or FAIL line for every stage, with the last lines of each failing log |
| `machine.txt` | Date, host, Slurm job, visible-device variables, `MUJOCO_GL`, the git commit, and the full `nvidia-smi` or `rocm-smi` output |
| `torch_report.log` | torch version, its CUDA or ROCm build, and the devices it sees |
| `install.log` | The full installation output |
| `<stage>.log` | The full output of each stage, including complete tracebacks |
| `smoke_*.json` | Machine-readable results of each smoke run |

If the job never started, send the output of these two commands instead:

```bash
squeue -u $USER
```

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList,Reason
```

## Common problems

| Symptom | Likely cause and fix |
| --- | --- |
| `gpu_visible_to_torch` fails with `CUDA build None` | The CPU build of torch was installed. Rerun `scripts/install.sh cu126`, then run commands with `.venv/bin/python` or `uv run --no-sync`, never a plain `uv run`, which switches back to the CPU build. |
| `gpu_visible_to_torch` fails with a driver error | The driver is older than the torch build needs. Try `cu126` if you used `cu130`, or send `nvidia-smi` output. |
| `install` fails with a network error on a compute node | Run step 3 on the login node first. A compute node then only checks the existing environment. |
| A render stage fails with "neither the EGL nor the OSMesa library was found" | The node has no headless OpenGL library. Rerun with `RENDER=0` (training never renders), or ask the administrators for EGL or OSMesa. |
| Disk quota errors | Set `UV_CACHE_DIR` to a large directory, as in step 2, and clone into scratch or project space. |

## What has and has not been verified

- Verified on the development Mac (Apple M5, macOS, CPU and MPS): the whole `scripts/cluster_smoke.sh` run above, the install script, and `cluster/submit_smoke.sh`, which was given a stand-in `sbatch` that printed the arguments it received.
- Verified on GitHub's Ubuntu machines without a GPU or display: the unit tests, the smoke test on the CPU and with GPUs hidden, headless rendering with both EGL and OSMesa, and that the CUDA 12.6, CUDA 13.0, and ROCm 7.2 environments resolve from the lockfile on Linux.
- Not verified anywhere: anything running on an NVIDIA or AMD GPU, a real Slurm submission, the GPU builds of torch being downloaded and imported, and EGL rendering on a GPU node. The first cluster run is the test of those.
