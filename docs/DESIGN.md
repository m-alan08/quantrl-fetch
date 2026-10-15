# Design

This document describes what the code in this repository actually does, stage by stage, for a reader who knows reinforcement learning and quantization but does not want to read the code. Every statement about behavior was checked against the source on the `audit-hardening` branch, which starts from main at commit `d9fcb67`. Where something was measured, the command and result are given. Changes made on this branch so far: the unsigned activation grid after ReLU (concern B) and the straight-through estimator's gradient at the outermost grid levels (concern H), both fixed at your request.

Baseline at the start of the audit (run on the development Mac, Apple M5, 16 GB, macOS):

- `uv run pytest -q` printed `118 passed in 9.81s`.
- torch 2.14.1, Python 3.12.14, mujoco 3.11.0. `torch.cuda.is_available()` is `False` and `torch.backends.mps.is_available()` is `True`.
- Neither mypy nor pyright is installed in the project environment.

## Requirements checklist

Status is one of **present** (exists and is backed by a test or a run), **partial** (exists but incomplete or untested in an important way), or **missing**. The status at the start of the audit is in the second column; the evidence column says what proves the current status. "CI" means the GitHub Actions run on pull request #9 (Ubuntu, no GPU, no display); "Mac" means the development Mac (Apple M5, CPU and MPS).

### Reproducibility and runtime

| # | Requirement | Before | Now | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Seed Python, NumPy, torch, and the environment | present | present | `utils/seeding.py`; `tests/test_devices.py::test_same_seed_gives_the_same_training_metrics_on_device` passes on CPU and MPS (Mac). |
| 2 | Deterministic mode that can be switched on or off | partial | present | `deterministic` config field; on CUDA it sets `CUBLAS_WORKSPACE_CONFIG`. The CUDA path itself has not run (no GPU). |
| 3 | Same seed gives the same first N training metrics | partial | present | `test_same_seed_gives_the_same_training_metrics_on_device` compares every logged metric row (CPU and MPS). |
| 4 | One device-selection function, cuda then mps then cpu, with an override | partial | present | `utils/device.py: resolve_device`; `--device` on `train.py`, `sweep.py`, and `smoke_test.py`; `test_auto_prefers_cuda_then_mps_then_cpu`. |
| 5 | No hardcoded devices | present | present | Repo search in the audit report: only the list of valid device names and device-specific random-state saving remain. |
| 6 | float32 tensors on every device | present | present | `test_sac_update_runs_on_device` checks float32 parameters on CPU and MPS. |
| 7 | Headless MuJoCo rendering, EGL or OSMesa, selectable | missing | present | `rendering.py`, `make_video.py --render-backend`; CI renders a frame with EGL and with OSMesa on a machine with no display (`tests/test_rendering.py::test_headless_linux_rendering`). |
| 8 | Training never depends on rendering | present | present | Training creates no renderer; CI runs all training tests headless with no OpenGL libraries installed. |
| 9 | One typed, validated config that fails fast | partial | present | `config.py: validate_config`, called on every load; `tests/test_config.py` checks 23 bad values and that all problems are reported together; all 86 configs saved by earlier runs still pass. |
| 10 | Runtime checks on observation and action shapes, dtypes, and bounds | partial | present | `check_goal_env`, `check_goal_observation`, `check_action`, replay-buffer input checks; `tests/test_runtime_checks.py`. |
| 11 | NaN and inf checks on losses, Q values, and gradients | missing | present | `require_finite` in every SAC update (batch, targets, Q values, losses, gradient norms); `tests/test_runtime_checks.py` injects NaN weights and infinite rewards. |
| 12 | Clear errors when a checkpoint does not match the config | partial | present | `check_checkpoint_matches` for starting checkpoints and `check_resume_compatible` for resume; both have tests. |
| 13 | Checkpoint holds the full training state | partial | present | `resume_state.pt`: networks, targets, optimizers, temperature, normalizer, replay buffer, counters, schedule, all random states. `test_saved_state_holds_the_full_training_state`. |
| 14 | Resume matches an uninterrupted run | missing | present | `tests/test_resume.py` (stop and extend, and crash after a periodic save) and `test_resume_on_device_matches_an_uninterrupted_run` (CPU and MPS) compare every metric and all weights exactly. |
| 15 | Pinned dependencies with the Python version | partial | present | `uv.lock` pins torch 2.14.1 for CPU, CUDA 12.6, CUDA 13.0, and ROCm 7.2; CI resolves each GPU environment on Linux. |
| 16 | One install script for Mac and a Linux cluster | missing | present | `scripts/install.sh`; run on the Mac and in CI. Not yet run on a GPU node. |
| 17 | No absolute local paths | present | present | Repo search in the audit report finds none in code, configs, tests, or CI. |
| 18 | Metrics written to local files | present | present | `tests/test_logging.py::test_metrics_go_to_local_files_without_any_network`. |
| 19 | Optional Weights and Biases in offline mode | missing | present | `logging.wandb`, offline by default; tested with a stand-in wandb module. The real wandb package was not exercised. |
| 20 | Run metadata (git commit, versions, seed, device) | present | present | `metadata.json` now also records the torch CUDA and ROCm builds and visible devices. |

### Experiment content

| # | Requirement | Before | Now | Evidence |
| --- | --- | --- | --- | --- |
| 21 | Full-precision baseline | present | present | `results/` for both tasks. |
| 22 | Evaluation at each bit width on the same episodes | present | present | `test_sweep_uses_one_seed_list_for_every_method`. |
| 23 | Logged metrics that the plots need | present | present | `progress.csv` and sweep CSVs. |
| 24 | Plots generated from logs | partial | present | `tests/test_plot_script.py` runs `make_plots.py` on small fake logs for all three plot types; the smoke test draws plots from real logs. |

### Tests

| # | Requirement | Before | Now | Evidence |
| --- | --- | --- | --- | --- |
| 25 | Quantizer tests | partial | present | `tests/test_quantization.py`, including both grids, saturation, and the edge-gradient fix. |
| 26 | SAC tests | partial | present | `tests/test_sac.py`: output shapes and dtypes, exact Polyak formula, temperature direction, critic and actor parameters change, finite losses. |
| 27 | Replay buffer tests | partial | present | `tests/test_replay.py`: shapes and dtypes, no samples from unwritten slots, wrap-around order. |
| 28 | Environment contract tests | partial | present | `tests/test_environment.py` and `test_runtime_checks.py` for both tasks: spaces, reset and step types, goal handling, reward range. |
| 29 | Core tests on every available device | missing | present | `tests/test_devices.py` runs on CPU and MPS on the Mac; CUDA cases skip with a message and are expected to run on the cluster. |
| 30 | Resume test | missing | present | Row 14. |
| 31 | Plot script on a tiny fake log | partial | present | Row 24. |
| 32 | One-command smoke test under 5 minutes with PASS or FAIL | missing | present | `scripts/smoke_test.py`: 11 s on CPU and 27 s on MPS on the Mac; exits nonzero on failure (`tests/test_smoke_script.py` injects a failure). |

### Static checks and cluster readiness

| # | Requirement | Before | Now | Evidence |
| --- | --- | --- | --- | --- |
| 33 | ruff clean, in CI | present | present | CI. |
| 34 | mypy clean | missing | present | `uv run mypy` reports no issues in 57 files, locally and in CI. |
| 35 | Cluster smoke script and Slurm submission | missing | present | `scripts/cluster_smoke.sh`, `cluster/submit_smoke.sh`, `cluster/smoke.sbatch`; run on the Mac and in CI. Not run on Slurm. |
| 36 | Cluster runbook | missing | present | `docs/CLUSTER_RUNBOOK.md`. |
| 37 | CPU-forced and GPU-hidden fallback runs | missing | present | The smoke test with `--device cpu` and with all GPU visibility variables empty, on the Mac and in CI. Not yet on a machine that has a GPU to hide. |

## Pipeline at a glance

```
Fetch environment (MuJoCo)
  dict observation: observation, achieved_goal, desired_goal
        |
        v
Goal-observation normalizer  -> one float32 vector per step ("normalized input")
        |
        v
Actor (MLP, optionally fake-quantized) -> action in [-1, 1]^4
        |                                         |
        |                                   environment step
        v                                         |
Episode replay buffer with HER  <------------------+
        |
        v  batches of (input, action, reward, next input)
SAC update: twin critics, actor, entropy temperature, Polyak targets
        |
        v
Periodic evaluation on fixed seeds -> progress.csv
Final checkpoint + final evaluation -> checkpoint.pt, summary.json
        |
        v
PTQ, QAT fine-tuning, QAT from scratch (sweep) -> results/*.csv -> plots, videos
```

## Stage 1: environment

**Code:** `envs/factory.py`.

**What it does.** `make_env(config.env)` calls `gym.make(env_id)` for `FetchReach-v4` or `FetchPush-v4` from gymnasium-robotics 1.4.2. No wrappers are added besides Gymnasium's own time limit. `env_dimensions` reads the sizes below from the spaces.

**Inputs.** An action of shape (4,), float32, in [-1, 1].

- The first three components move the gripper. The environment multiplies them by 0.05, so each component is a target displacement of at most 5 cm along x, y, or z per step.
- The fourth component would open or close the gripper, but both FetchReach and FetchPush are created with `block_gripper=True`, so it is ignored in both tasks. The actor still outputs four numbers.

**Outputs.** `reset` and `step` return a dict observation:

| Key | FetchReach shape | FetchPush shape | Content and units |
| --- | --- | --- | --- |
| `observation` | (10,) | (25,) | Gripper position (m), gripper finger state, gripper velocity (velocities are multiplied by the 0.04 s control step, so they are meters per step). FetchPush adds block position, block position relative to the gripper, block rotation (Euler angles, rad), and block linear and angular velocity. |
| `achieved_goal` | (3,) | (3,) | Gripper position (Reach) or block position (Push), meters. |
| `desired_goal` | (3,) | (3,) | Target position, meters. |

`step` also returns a reward and `info["is_success"]`.

**Reward.** Sparse: 0 if the distance between `achieved_goal` and `desired_goal` is below 0.05 m, otherwise -1. The reward function is vectorized and can be called on a batch, which HER relies on.

**Episodes.** 50 steps at 25 Hz (2 seconds). Episodes never terminate early; they only end at the 50-step time limit.

**Assumptions.** mujoco is pinned below 3.12 because newer versions break gymnasium-robotics 1.4.2 (`docs/NOTES.md`).

## Stage 2: observation normalization

**Code:** `envs/normalizer.py`.

**Input.** A raw `observation` and `desired_goal`, or batches of them.

**Output.** One float32 vector, the "normalized input", of size 13 for FetchReach (10 + 3) and 28 for FetchPush (25 + 3).

**Math.** Observation and goal each have their own running normalizer. It keeps the exact sum and sum of squares of everything seen so far, in float64, so the mean and standard deviation are exact.

1. Clip raw values to [-200, 200].
2. Compute `(x - mean) / max(std, 0.01)`.
3. Clip to [-5, 5].

Normalizer statistics are updated once per collected episode, with that episode's observations and with both its desired goals and its achieved goals. Achieved goals are included because HER turns them into goals.

**Assumptions.** The normalizer is part of the policy: it is saved in every checkpoint, and a quantized actor uses the same frozen normalizer as the fp32 actor it came from. Its input range of [-5, 5] is what the first quantized layer sees.

## Stage 3: replay buffer with hindsight relabeling

**Code:** `algos/replay.py`.

**Storage.** Whole episodes in preallocated float32 arrays. With capacity C episodes and episode length T = 50:

| Array | Shape |
| --- | --- |
| observations | (C, T + 1, obs_dim) |
| achieved goals | (C, T + 1, 3) |
| desired goals | (C, T, 3) |
| actions | (C, T, 4) |

C is 20,000 by default, which is 1,000,000 transitions. When the buffer is full, the oldest episode is overwritten. Samples are only drawn from the first `size` slots, so unwritten (zero) slots are never returned.

**Sampling a batch of B transitions.**

1. Pick an episode uniformly from the stored ones and a time t uniformly from 0 to 49.
2. With probability 1 - 1/(1 + k) (k = 4, so 0.8), replace the goal with the achieved goal at a future index t' drawn uniformly from t + 1 to 50. This is HER's "future" strategy.
3. Recompute the reward with the environment's reward function on (achieved goal at t + 1, goal).

**Output.** Observation, next observation, goal, action, and reward, each of shape (B, ...) and float32, plus a boolean mask of which transitions were relabeled. The training loop then normalizes (observation, goal) and (next observation, goal) into the two input vectors.

## Stage 4: networks

**Code:** `algos/networks.py`.

Every linear layer is created through a `linear_factory`. By default this is a plain `nn.Linear`; for quantized runs it is a `QuantLinear` (Stage 6). That is the only hook between SAC and quantization.

**Actor (tanh-squashed Gaussian).** An MLP with ReLU activations, mapping input of size d to 2 × 4 outputs, split into a mean and a log standard deviation. The log standard deviation is clamped to [-20, 2].

- FetchReach: 13 -> 256 -> 256 -> 8.
- FetchPush: 28 -> 256 -> 256 -> 256 -> 8.

A sample is `a = tanh(mu + sigma * eps)` with standard normal eps. Its log probability is the Gaussian log density minus `sum log(1 - tanh(u)^2)`, computed as `2 (log 2 - u - softplus(-2u))` for numerical stability. The deterministic action is `tanh(mu)`.

**Critics.** Two independent MLPs, Q1 and Q2, each mapping (input, action) of size d + 4 through the same hidden sizes to one value. Target critics are deep copies with gradients switched off.

## Stage 5: the SAC update

**Code:** `algos/sac.py`, `SACAgent.update`.

**Input.** One batch: inputs (B, d), next inputs (B, d), actions (B, 4), rewards (B,). Defaults: B = 256 for FetchReach and 512 for FetchPush.

Each update has these steps, in order:

1. **Critic target**, without gradients: sample a' and log pi(a' | x') from the current actor, then compute `y = r + gamma * (min(Q1_target, Q2_target)(x', a') - alpha * log pi(a' | x'))`. There is no terminal mask, because Fetch episodes only end by time limit. If `sac.clip_q_target` is on (the default), y is clipped to [-1/(1 - gamma), 0], which is [-50, 0] for gamma = 0.98. See concern A below.
2. **Critic loss**: `MSE(Q1(x, a), y) + MSE(Q2(x, a), y)`, then one Adam step on both critics.
3. **Actor loss**, with critic parameters frozen: sample a fresh action with the reparameterization trick, then compute `mean(alpha * log pi(a | x) - min(Q1, Q2)(x, a))`. One Adam step on the actor.
4. **Temperature loss**: `-mean(log_alpha * (log pi(a | x) + target_entropy))`, using the log probabilities from step 3 without gradient. The target entropy is -4 (minus the action dimension). One Adam step on log alpha. When the policy's entropy is below the target, alpha grows; when above, it shrinks.
5. **Polyak update** of the target critics: `target = (1 - tau) * target + tau * online`, with tau = 0.005.

**Output.** The metrics `critic_loss`, `actor_loss`, `alpha_loss`, `alpha`, `entropy`, and `q_mean`, as Python floats.

**Defaults.** Learning rate 1e-3 for all three optimizers, gamma 0.98, initial alpha 1.0.

## Stage 6: quantization

**Code:** `quant/quantizer.py`, `quant/layers.py`, `quant/ptq.py`, `quant/qat.py`.

### What is quantized

| Part | Quantized? |
| --- | --- |
| Actor weights | Yes, every linear layer. |
| Actor layer inputs | Yes: the normalized input to the first layer (symmetric grid) and the ReLU output feeding every later layer (unsigned grid by default, see below). |
| Actor biases | No, kept in fp32. |
| Actor outputs (mean and log std) | No, kept in fp32. |
| Critics | No, by default. A flag (`quant.quantize_critics`) exists but no reported run used it. |

Weights and activations can have different bit widths. All reported runs used the same width for both.

### Bit widths

32 means "disabled": the layer is exactly an `nn.Linear`. Low bit widths are integers from 2 to 16. Anything else, such as 33 or 4.5, is rejected with an error. The reported runs used 8, 6, 5, 4, 3, and 2.

### The grids

Two uniform grids are used. Zero is on both.

- **Symmetric (signed):** integer levels -L to L where `L = 2^(b - 1) - 1`. That gives 2L + 1 levels: 255 for 8 bits, 15 for 4 bits, 7 for 3 bits, 3 for 2 bits. A value x with scale s becomes `clamp(round(x / s), -L, L) * s`. Used for all weights and for the first layer's input (the normalized observation, which can be negative).
- **Unsigned:** integer levels 0 to `H = 2^b - 1`, so all 2^b levels are used: 256 for 8 bits, 16 for 4 bits, 8 for 3 bits, 4 for 2 bits. A value becomes `clamp(round(x / s), 0, H) * s`. Used for every layer input that comes out of a ReLU, since those are never negative. This is the default (`quant.relu_activation_grid = "unsigned"`). Setting it to `"signed"` restores the old behavior, where post-ReLU inputs used the symmetric grid and only 4 of its 7 levels at 3 bits could ever be reached.

Inside the range the error is at most s / 2; outside it, values saturate at the outermost level.

**Old runs.** Every quantized result produced before this branch used the signed grid for everything. A saved run or checkpoint that used quantization and has no `relu_activation_grid` field is read as `"signed"`, so it rebuilds exactly as it was trained. Re-evaluating three old QAT checkpoints this way reproduced their recorded success rate, return, and final distance exactly, and rerunning PTQ with the signed grid reproduced all 36 recorded per-channel PTQ success rates (3 seeds, 6 bit widths, both tasks) exactly.

**Measured effect of the unsigned grid on PTQ** (per-channel weights, weights and activations at the same width, mean success over 3 seeds and 100 episodes; outputs in `runs/audit/`, not committed):

| Bits | FetchReach signed | FetchReach unsigned | FetchPush signed | FetchPush unsigned |
| --- | --- | --- | --- | --- |
| 8 | 100.0 | 100.0 | 98.7 | 99.0 |
| 6 | 100.0 | 100.0 | 95.0 | 98.0 |
| 5 | 97.7 | 98.0 | 82.7 | 86.7 |
| 4 | 82.0 | 88.0 | 14.7 | 20.7 |
| 3 | 43.0 | 36.0 | 8.0 | 7.3 |
| 2 | 0.0 | 0.0 | 8.0 | 8.0 |

The unsigned grid helps at 4 to 6 bits and does not help at 3 bits, where the first layer's input (still on the symmetric grid) is the likely limit.

### Scales

- **Weights:** `s = max|W| / L`, recomputed from the current weights at every forward pass. "Per tensor" uses one maximum for the whole matrix. "Per channel" uses one maximum per output row, so a row of small weights is not forced onto the coarse grid of a row of large weights. Weights are never clipped.
- **Activations (layer inputs):** one scale per layer, `s = running_max_abs / L` on the symmetric grid or `running_max_abs / H` on the unsigned grid. `running_max_abs` is a stored buffer that is updated in one of three ways:
  - During training, unless frozen: an exponential moving average of each batch's largest magnitude, `m * running + (1 - m) * max|x|` with m = 0.99. The first batch sets the value directly.
  - During PTQ calibration: the largest magnitude over all calibration inputs, after which the range is frozen.
  - In evaluation mode, or when frozen: no update.

  Until a range has been observed, inputs pass through unquantized.

The scale is floored at max(1e-8, the smallest normal number of the dtype), so all-zero tensors quantize to zero instead of dividing by zero.

### Gradients: the straight-through estimator

Training uses "fake quantization": values are quantized and immediately dequantized, so the arithmetic stays in float32 but every weight and activation lies on the grid.

Rounding has zero gradient almost everywhere, so its backward pass is replaced by the identity: `round_ste(v) = v + (round(v) - v).detach()`. The clamp keeps its normal gradient. The intended result, and what the documentation says, is:

- gradient 1 inside the range [-L s, L s];
- gradient 0 where values saturate.

**How the code implements it.** The gradient mask is taken from the rounded level: if `round(x / s)` lies inside [low, high] (edges included), the value gets gradient 1; otherwise it is saturated and gets gradient 0. Before this branch, the code clamped after rounding, and because PyTorch's clamp gives zero gradient exactly on its boundary, every value that rounded to the outermost level also got gradient 0 (concern H). The forward values are unchanged by the fix: re-evaluating old QAT checkpoints gives identical results.

Scales are treated as constants: weight scales are computed from detached weights and activation ranges are buffers, so no gradient flows to the scale.

### PTQ

1. Run the trained fp32 actor deterministically on 20 calibration episodes (seeds from 2,000,000, disjoint from evaluation seeds) and record the 1,000 normalized inputs it sees.
2. Build a quantized actor and copy the fp32 weights into it. Missing, extra, or wrongly shaped weights raise an error.
3. Run all calibration inputs through it in calibration mode, record each layer's maximum input magnitude, and freeze.

All layers are calibrated in one pass on unquantized activations.

### QAT fine-tuning

1. Load the whole trained fp32 agent into an agent with a quantized actor: weights, critics, target critics, temperature, optimizer states, and normalizer.
2. Reset the optimizer learning rates to the configured values.
3. Calibrate the actor exactly as PTQ does, then unfreeze the activation ranges. At step 0 the actor is therefore identical to the PTQ actor (`tests/test_qat.py::test_qat_fine_tune_starts_exactly_at_the_ptq_actor`).
4. Continue SAC with a fresh, empty replay buffer: actor learning rate 1e-4, temperature learning rate 0 (frozen), 10,000 environment steps for FetchReach and 100,000 for FetchPush.

### QAT from scratch

The same training as the fp32 baseline (same config, budget, and seeds), with a quantized actor from the first update.

### Train and eval modes

While acting in the environment and during evaluation, the actor is put in evaluation mode, so activation ranges move only on training batches.

## Stage 7: the training loop

**Code:** `training.py`, `train(config)`.

1. Seed everything, set the torch thread count, resolve the device, and write `config.json` and `metadata.json`.
2. Create a training environment and a separate evaluation environment. Seed the training environment once.
3. Build the normalizer, the replay buffer, and the agent. If `train.init_checkpoint` is set, initialize from it (QAT fine-tuning).
4. Loop until `train.total_steps` environment steps have been collected:
   1. Collect one full 50-step episode. Actions are uniform random for the first `train.random_steps` steps, and sampled from the stochastic actor after that.
   2. Store the episode and update the normalizer.
   3. Once `train.update_after` steps have been collected, run `updates_per_step × 50` SAC updates. Fractional rates carry over between episodes.
   4. Every `train.eval_interval` steps, evaluate deterministically on `train.eval_episodes` fixed seeds and append one row to `progress.csv`. The row holds the step, wall time, recent training success, evaluation results, and the mean of each update metric since the previous row.
   5. Optionally save `checkpoint_<step>.pt` every `train.checkpoint_interval` steps.
5. Save `checkpoint.pt`, evaluate on `eval.episodes` (100) fixed seeds, and write `summary.json`.

**Units.** Steps are environment steps (one step is 0.04 s of simulated time). Wall times come from `time.perf_counter`, which on macOS does not advance while the machine sleeps.

## Stage 8: evaluation and statistics

**Code:** `eval/evaluate.py`, `eval/stats.py`, `eval/quantized.py`.

**Episodes.** Episode i is reset with seed `1,000,000 + i`, so every policy is tested on exactly the same goals and starting states. Actions are deterministic (`tanh(mu)`).

**Per episode, the code records:**

- success: `is_success` at the last step;
- return: sum of rewards, between -50 and 0;
- final distance: meters between achieved goal and desired goal at the end.

**Within one training seed.** A 95 percent percentile bootstrap over episodes, with 10,000 resamples, gives intervals for success rate and return.

**Across training seeds.** The code reports the mean, sample standard deviation, minimum, maximum, and a bootstrap interval over the seed means. With 3 seeds that interval can only run from the lowest to the highest seed, so the minimum and maximum are reported as well.

## Stage 9: logs and output files

Each run directory `runs/<name>_seed<seed>/` contains:

| File | Content |
| --- | --- |
| `config.json` | Fully resolved config. |
| `metadata.json` | Seed, git commit and dirty flag, package versions, Python, platform, device, thread count, start time, and the initial checkpoint if any. |
| `progress.csv` | One row per periodic evaluation. |
| `summary.json` | Final evaluation and timings. |
| `checkpoint.pt` | The policy: config, actor and critic weights, optimizers, temperature, normalizer, dimensions, metadata, and step count. Used for evaluation, PTQ, and QAT fine-tuning. |
| `resume_state.pt` | The full training state for `--resume` (requirement 13), written atomically at every checkpoint interval and at the end. |
| `checkpoint_<step>.pt` | Periodic policy checkpoints when `train.checkpoint_interval` is set. |

The `runs/` folder is ignored by git; only small CSV and JSON summaries are copied into `results/`.

## Stage 10: the sweep

**Code:** `sweep.py`, `scripts/sweep.py`.

For each fp32 run and each bit width, the sweep builds a QAT fine-tuning config and a QAT-from-scratch config. It trains them in parallel worker processes, one torch thread each. A finished run is reused only if its saved config equals the requested one; otherwise the sweep stops with an error.

It then evaluates, on one shared list of seeds:

- each fp32 actor;
- a PTQ copy of each fp32 actor at every bit width;
- every QAT actor.

**Outputs.** A tidy CSV with one row per (seed, method, bit width), a summary across seeds, and per-run training times.

## Stage 11: plots and videos

**Code:** `viz/plots.py`, `viz/video.py`, `scripts/make_plots.py`, `scripts/make_video.py`.

**Plots.** Matplotlib with the non-interactive `Agg` backend, so no display is needed. There are three kinds: training curves (mean across seeds with min to max shading), PTQ ablations, and method comparisons (success rate and final distance against bit width, with distance on a log scale and the 5 cm radius marked).

**Videos.** These render the same evaluation seeds for one or two policies, label each frame with the policy and the success or failure, place two policies side by side, and write mp4 and gif files. This is the only code that needs a working OpenGL context.

## Devices and dtypes

- Every network tensor is float32.
- The device comes from `config.device` or `--device`, which defaults to `"auto"`: CUDA if available (including ROCm builds, which use the same API), then MPS, then CPU. Explicit `cpu`, `mps`, `cuda`, or `cuda:<index>` fail with a clear message if unavailable. Published results used CPU, so reproducing them exactly needs `--device cpu`.
- Device placement is consistent: networks are moved to the agent's device, batches are converted with `device=agent.device`, and actions come back with `.cpu().numpy()`.
- PTQ builds the quantized actor on the source actor's device and dtype.
- CPU with one thread is the default and the setup used for every reported result. Measured earlier in the project: one SAC update takes 1.6 ms on one CPU thread against 4.4 ms on MPS for the FetchReach network.

## Math and design concerns

Decisions so far:

| Concern | Decision |
| --- | --- |
| A. Critic target clipping | Kept as is. All reported results used it. |
| B. Symmetric grid after ReLU | Fixed on this branch: post-ReLU inputs use the unsigned grid by default. Results produced before the fix used the signed grid and are labeled as such. |
| C. What "an N-bit actor" means | Kept, and stated wherever results are quoted. |
| D. PTQ calibration choices | Kept as is. |
| E. Unequal training budgets | Intended: PTQ, a short fine-tune, and full training from scratch are the three points being compared. |
| F, G | Noted; no change. |
| H. Zero STE gradient at the outermost grid levels | Fixed on this branch. QAT results produced before the fix are stale and need rerunning. |

The original descriptions follow.

**A. Clipping the critic target to [-1/(1 - gamma), 0] in soft (entropy-regularized) SAC.**

The bounds are the possible range of a discounted sum of rewards in {-1, 0}. SAC's target also contains the entropy bonus `-alpha * log pi`, which the bounds do not account for.

- When the policy is wide, early in training with alpha near 1, the bonus is positive, the true soft value can be above 0, and clipping at 0 biases the target down.
- Once alpha is small and the policy narrow, the bonus is slightly negative and 0 is a valid upper bound.

Measured on the trained seed-0 agents with a scratch script (not committed), over 20,480 sampled targets each:

| Task | alpha | Targets clipped at the upper bound 0 | Targets clipped at the lower bound -50 | Mean entropy bonus per step |
| --- | --- | --- | --- | --- |
| FetchReach | 0.00066 | 54.18% | 0.00% | -0.0022 |
| FetchPush | 0.00129 | 5.34% | 0.00% | -0.0030 |

At the end of training the clip mostly removes small positive overestimates, which is its intended effect. Its effect early in training was not measured.

Options:

1. Keep it. All reported results used it.
2. Clip only the lower bound.
3. Turn it off.

Changing it would require rerunning every experiment.

**B. Symmetric activation grid after ReLU.**

Hidden-layer inputs are outputs of ReLU, so they are never negative, but the grid is symmetric, so half the levels are never used. At 3 bits a hidden activation can take only 4 distinct values (0, s, 2s, 3s) instead of 7. An unsigned grid for post-ReLU inputs would effectively give one extra bit. The symmetric grid matches the original specification. This probably makes the 3 and 4 bit results more pessimistic than an unsigned design would.

**C. What "an N-bit actor" means here.**

- Biases and the final layer's outputs stay in fp32.
- The first layer's input, the normalized observation, is quantized.
- The normalizer itself runs in float64 and is not quantized.

This is a common convention, but it should be stated wherever results are quoted. The FetchReach evidence suggests that the quantized first-layer input is what limits accuracy at 3 and 4 bits.

**D. PTQ calibration choices.**

- Activation ranges use the maximum, not a percentile, so one outlier stretches the grid.
- All layers are calibrated on unquantized activations in one pass.
- Calibration data comes from the fp32 policy, whose state distribution differs from the quantized policy's.

These are standard simple choices, and alternatives could change the PTQ numbers.

**E. Unequal training budgets across methods (intended).**

PTQ gets no extra training, QAT fine-tuning gets 10 percent of the baseline's steps, and QAT from scratch gets 100 percent. This is the intended comparison. When quoting the FetchPush ranking, it should be read as "with these budgets". The fine-tuning recipe was chosen from pilot runs on seed 0 at 3 and 4 bits on FetchReach.

**F. Statistical strength.**

Every comparison uses 3 training seeds and 100 evaluation episodes per seed. Interval estimates across seeds are coarse. Some FetchPush differences, such as 4-bit PTQ at 14.7 percent against QAT fine-tuning at 10.7 percent, are within the seed-to-seed spread.

**G. Success is measured at the final step only.**

This is the standard Fetch convention, but it means a policy that reaches the goal and then drifts away counts as a failure. The final-distance metric is reported alongside.

**H. The STE gives zero gradient at the outermost grid levels, not only beyond them.**

The quantizer computes `clamp(round_ste(x / s), low, high)`. A value that rounds to exactly `low` or `high` sits on the clamp's boundary, and PyTorch's clamp returns gradient 0 there. Checked directly with torch 2.14.1: clamp(0.0, 0, 3) and clamp(3.0, 0, 3) both have gradient 0, while clamp(1e-7, 0, 3) and clamp(2.9999, 0, 3) have gradient 1. The standard straight-through estimator, and PyTorch's own fake-quantize, pass gradient 1 for every value whose rounded level lies inside the range, edges included.

How much it matters, measured on the trained seed-0 actors:

| Bit width | Weights with zero gradient, per-channel scales (Reach / Push) | Same, per-tensor scales | Positive post-ReLU activations that round to 0 on the unsigned grid (Reach / Push) |
| --- | --- | --- | --- |
| 8 | 0.77% / 0.56% | 0.00% / 0.00% | 1.7% / 2.4% |
| 6 | 0.86% / 0.59% | 0.00% / 0.00% | 5.2% / 9.9% |
| 5 | 0.99% / 0.64% | 0.00% / 0.00% | 12.2% / 18.9% |
| 4 | 1.47% / 0.78% | 0.00% / 0.00% | 22.0% / 35.8% |
| 3 | 3.26% / 1.34% | 0.01% / 0.01% | 37.7% / 58.7% |
| 2 | 21.31% / 8.56% | 0.22% / 0.13% | 85.6% / 85.3% |

- **Effect on existing results:** the reported QAT runs used per-channel weights, so at 2 bits a fifth of the actor's weights on FetchReach (and 9 percent on FetchPush) never received a gradient, and at 3 bits 1 to 3 percent. At 4 bits and above the effect is about 1.5 percent of weights or less. On the old signed activation grid, activations were barely affected, because 0 was an interior level.
- **Interaction with fix B:** on the unsigned grid, 0 is the bottom edge. Every small positive activation that rounds to 0 would get no gradient: 22 percent of them at 4 bits on FetchReach and 36 percent on FetchPush. QAT with the unsigned grid should not be run until this is decided.
- PTQ is unaffected, since it uses no gradients.

**Decision: fixed.** The gradient mask now comes from the rounded level. Measured after the fix, on the same trained seed-0 actors: 0.00 percent of weights (per-channel) and 0.00 percent of positive post-ReLU activations (unsigned grid) get zero gradient from the quantizer, at every bit width from 8 to 2 on both tasks. Forward values did not change: two old QAT checkpoints re-evaluated after the fix gave exactly their recorded results. Every QAT result produced before the fix is stale. Rerunning the FetchReach sweep takes about 45 minutes and the FetchPush sweep about 6 hours on this Mac.

**Observation, not a concern.** While checking H, a full backward pass through the trained fp32 actors showed many weights with zero gradient even without quantization. The cause is ReLU units that never activate on the states the policy visits: 7.8 and 77.0 percent of the two hidden layers in the FetchReach actor, and 10.2, 40.2, and 25.4 percent of the three hidden layers in the FetchPush actor (seed 0, 1,000 calibration states). Saturated actions are rare: the deterministic action exceeds 0.999 in magnitude on at most 5 percent of visited states, and the tanh gradient is exactly zero in float32 on at most 0.2 percent. Inactive units are a normal property of trained ReLU networks, but they mean the actor uses fewer units than its nominal width.
