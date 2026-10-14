# Design

This document describes what the code in this repository actually does, stage by stage, for a reader who knows reinforcement learning and quantization but does not want to read the code. Every statement about behavior was checked against the source on the `audit-hardening` branch, which starts from main at commit `d9fcb67`. Where something was measured, the command and result are given.

Baseline at the start of the audit (run on the development Mac, Apple M5, 16 GB, macOS):

- `uv run pytest -q` printed `118 passed in 9.81s`.
- torch 2.14.1, Python 3.12.14, mujoco 3.11.0. `torch.cuda.is_available()` is `False` and `torch.backends.mps.is_available()` is `True`.
- Neither mypy nor pyright is installed in the project environment.

## Requirements checklist

Status is one of **present** (exists and is backed by a test or a run), **partial** (exists but incomplete or untested in an important way), or **missing**.

### Reproducibility and runtime

| # | Requirement | Status | Evidence and notes |
| --- | --- | --- | --- |
| 1 | Seed Python, NumPy, torch, and the environment | present | `utils/seeding.py: seed_everything` seeds `random`, `numpy`, and `torch`; `training.py` calls `env.reset(seed=...)` once and `env.action_space.seed(...)`; the replay buffer draws from a NumPy generator made from the same seed. `tests/test_training_smoke.py::test_training_is_reproducible_on_cpu` trains twice and checks bitwise identical actor weights. |
| 2 | Deterministic mode that can be switched on or off | partial | `torch.use_deterministic_algorithms(True)` is always switched on and cannot be turned off from the config. On CUDA this setting makes matrix multiplies raise an error unless the environment variable `CUBLAS_WORKSPACE_CONFIG` is set, which the code does not do. Not runnable here (no CUDA). |
| 3 | Determinism test on the first N training metrics | partial | The existing test compares final actor weights, not the logged metrics. Same intent, different check. |
| 4 | One device-selection function (cuda, then mps, then cpu) with an override | partial | `utils/seeding.py: resolve_device` handles `"auto"` as mps then cpu and never tries cuda. The config default is the fixed string `"cpu"`. The only override is `--set device=...`. |
| 5 | No hardcoded devices | present | Search for `"cuda"`, `.cuda(`, `"mps"`, `torch.backends` finds only `resolve_device` and one MPS test (search output is in the audit report). Tensors are created on `agent.device`. |
| 6 | float32 tensors on every device | present | Every tensor entering the networks goes through `torch.as_tensor(..., dtype=torch.float32)`. float64 appears only in NumPy code (normalizer statistics, bootstrap), which runs on the CPU. |
| 7 | Headless MuJoCo rendering on a Linux cluster (EGL or OSMesa, selectable) | missing | Nothing sets or reads `MUJOCO_GL`. Rendering works on macOS through the default backend. On a Linux node without a display, video rendering would most likely fail; not testable here. |
| 8 | Training never depends on rendering | present | Training and evaluation create environments without a render mode (`make_env(config.env)`). Only `viz/video.py` asks for `render_mode="rgb_array"`. |
| 9 | One typed, validated config that fails fast | partial | Configs are dataclasses loaded from JSON. Unknown keys and non-object sections are rejected (`tests/test_config.py`). Bit widths are validated, but only when a quantized layer is built. Nothing checks learning rates, sizes, gamma, tau, the device string, `quant.method`, or paths at startup. |
| 10 | Runtime checks on observation and action shapes, dtypes, and bounds | partial | Shapes are checked only when a run starts from another checkpoint (`quant/qat.py: check_dimensions`). Actions are in [-1, 1] by construction (tanh), but nothing asserts it. |
| 11 | NaN and inf checks on losses, Q values, and gradients | missing | No such checks. A NaN would propagate silently into the logs. |
| 12 | Clear errors when a checkpoint does not match the config | partial | Network weights load strictly (shape or key mismatch raises an error). There is no check that a checkpoint's config matches the run that loads it. |
| 13 | Checkpoint holds the full training state | partial | Saved: actor, critics, target critics, all three optimizers, the entropy temperature, the observation normalizer, the step counter, the config, and run metadata. Not saved: the replay buffer, the episode counter, the evaluation and checkpoint schedule, and the random number generator states (Python, NumPy generator, torch, environment). |
| 14 | Resume an interrupted run and match an uninterrupted run | missing | There is no resume path. `train.init_checkpoint` starts a new run from a checkpoint's weights, which is QAT fine-tuning, not resume. |
| 15 | Pinned dependencies with the Python version | partial | `uv.lock` pins every package and `.python-version` pins 3.12. On Linux, however, `pyproject.toml` forces the CPU-only torch build (`torch 2.14.1+cpu`, confirmed in `uv.lock`), which was added so CI does not download CUDA. On a GPU cluster this would make the GPU unusable. |
| 16 | One install script for Mac and a Linux cluster | missing | Installation is `uv sync` as described in the README. There is no script and no way to choose a CUDA or ROCm torch build. |
| 17 | No absolute local paths; all paths from config or arguments | present | Searching `src`, `scripts`, `tests`, `configs`, and CI for `/Users/`, `/home/`, `/private/` finds nothing. Output goes to `output_dir` from the config. |
| 18 | Metrics written to local files | present | Each run writes `config.json`, `metadata.json`, `progress.csv`, and `summary.json`. Sweeps write tidy CSV and JSON into `results/`. No network is needed. |
| 19 | Optional Weights and Biases in offline mode | missing | Not present. Marked optional in the request. |
| 20 | Run metadata (git commit, package versions, seed, device) | present | `utils/run_info.py`, written to `metadata.json` for every run. |

### Experiment content

| # | Requirement | Status | Evidence and notes |
| --- | --- | --- | --- |
| 21 | Full-precision baseline | present | fp32 runs for FetchReach and FetchPush, 3 seeds each, in `results/`. |
| 22 | Evaluation at each bit width on the same episodes | present | `sweep.py` builds one seed list and passes it to every evaluator (`tests/test_qat.py::test_sweep_uses_one_seed_list_for_every_method`). |
| 23 | Logged metrics that the plots need | present | `progress.csv` (training curves) and the sweep CSVs (bit-width plots). |
| 24 | Plots generated from logs | partial | `scripts/make_plots.py` works from the CSVs. The bit-width plot has a test (`tests/test_ptq_evaluation.py`); the training-curve plot does not. |

### Tests

| # | Requirement | Status | Evidence and notes |
| --- | --- | --- | --- |
| 25 | Quantizer: values on the grid, saturation, 2, 4, and 8 bits, exact identity at full precision, gradients through the STE, closeness at high bit width, per-tensor and per-channel scales | partial | All covered in `tests/test_quantization.py` except an explicit check that values beyond the range saturate at exactly plus or minus the largest level. |
| 26 | SAC: squashing and log-probability correction, output shapes, soft target update, temperature update direction, critic and actor parameters change, finite losses | partial | Covered: log-probability against `torch.distributions`, deterministic action, finite losses, actor and target parameters change. Missing: exact Polyak formula, temperature direction, critic parameter change, explicit output shapes. |
| 27 | Replay buffer: wrap-around, sample shapes and dtypes, no uninitialized data | partial | Wrap-around and relabeling are tested. Shapes and dtypes are not checked explicitly. Sampling from an empty buffer raises a NumPy error instead of a clear message. |
| 28 | Environment contracts: spaces, reset and step types, goal handling, reward range | partial | `tests/test_environment.py` covers FetchReach observation keys and shapes and the vectorized reward. FetchPush, reset and step return types, and the reward range are not covered. There is no wrapper class; observation flattening happens in the normalizer. |
| 29 | Device-parametrized core tests (cpu and mps here) | missing | One MPS test exists. The core tests run on CPU only. |
| 30 | Resume test | missing | No resume feature to test. |
| 31 | Plot script on a tiny fake log | partial | See row 24. |
| 32 | One-command smoke test under 5 minutes with PASS or FAIL | missing | No such script. |

### Static checks and cluster readiness

| # | Requirement | Status | Evidence and notes |
| --- | --- | --- | --- |
| 33 | ruff clean, in CI | present | `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, and `pytest` on every pull request. |
| 34 | mypy or pyright clean | missing | Neither tool is installed or configured. |
| 35 | Cluster smoke script and Slurm template | missing | |
| 36 | Cluster runbook | missing | |
| 37 | CPU-forced and GPU-hidden fallback runs | missing | No forced-CPU or hidden-GPU run has been done. |

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
- The fourth component opens or closes the gripper. FetchReach ignores it.

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
| Actor layer inputs | Yes: the normalized input to the first layer and the ReLU output feeding every later layer. |
| Actor biases | No, kept in fp32. |
| Actor outputs (mean and log std) | No, kept in fp32. |
| Critics | No, by default. A flag (`quant.quantize_critics`) exists but no reported run used it. |

Weights and activations can have different bit widths. All reported runs used the same width for both.

### Bit widths

32 means "disabled": the layer is exactly an `nn.Linear`. Low bit widths are integers from 2 to 16. Anything else, such as 33 or 4.5, is rejected with an error. The reported runs used 8, 6, 5, 4, 3, and 2.

### The grid

Symmetric and uniform. With b bits the integer levels are -L to L where `L = 2^(b - 1) - 1`. That gives 2L + 1 levels: 255 for 8 bits, 15 for 4 bits, 7 for 3 bits, 3 for 2 bits. Zero is always on the grid. A value x with scale s becomes `clamp(round(x / s), -L, L) * s`. Inside the range the error is at most s / 2; outside it, values saturate at plus or minus L * s.

### Scales

- **Weights:** `s = max|W| / L`, recomputed from the current weights at every forward pass. "Per tensor" uses one maximum for the whole matrix. "Per channel" uses one maximum per output row, so a row of small weights is not forced onto the coarse grid of a row of large weights. Weights are never clipped.
- **Activations (layer inputs):** one scale per layer, `s = running_max_abs / L`. `running_max_abs` is a stored buffer that is updated in one of three ways:
  - During training, unless frozen: an exponential moving average of each batch's largest magnitude, `m * running + (1 - m) * max|x|` with m = 0.99. The first batch sets the value directly.
  - During PTQ calibration: the largest magnitude over all calibration inputs, after which the range is frozen.
  - In evaluation mode, or when frozen: no update.

  Until a range has been observed, inputs pass through unquantized.

The scale is floored at max(1e-8, the smallest normal number of the dtype), so all-zero tensors quantize to zero instead of dividing by zero.

### Gradients: the straight-through estimator

Training uses "fake quantization": values are quantized and immediately dequantized, so the arithmetic stays in float32 but every weight and activation lies on the grid.

Rounding has zero gradient almost everywhere, so its backward pass is replaced by the identity: `round_ste(v) = v + (round(v) - v).detach()`. The clamp keeps its normal gradient. Overall, the gradient of a fake-quantized value with respect to its input is:

- 1 inside the range [-L s, L s];
- 0 where it saturates.

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
| `checkpoint.pt` | See requirement 13 for what is saved. |

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
- The device comes from `config.device`, which defaults to `"cpu"`. `"auto"` picks MPS if available, otherwise CPU, and never CUDA (requirement 4).
- Device placement is consistent: networks are moved to the agent's device, batches are converted with `device=agent.device`, and actions come back with `.cpu().numpy()`.
- PTQ builds the quantized actor on the source actor's device and dtype.
- CPU with one thread is the default and the setup used for every reported result. Measured earlier in the project: one SAC update takes 1.6 ms on one CPU thread against 4.4 ms on MPS for the FetchReach network.

## Math and design concerns for a decision

None of these were changed. Each needs your decision.

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

**E. Unequal training budgets across methods.**

PTQ gets no extra training, QAT fine-tuning gets 10 percent of the baseline's steps, and QAT from scratch gets 100 percent. On FetchPush the ranking (QAT from scratch first) may partly reflect budget. The fine-tuning recipe was also chosen from pilot runs on seed 0 at 3 and 4 bits on FetchReach.

**F. Statistical strength.**

Every comparison uses 3 training seeds and 100 evaluation episodes per seed. Interval estimates across seeds are coarse. Some FetchPush differences, such as 4-bit PTQ at 14.7 percent against QAT fine-tuning at 10.7 percent, are within the seed-to-seed spread.

**G. Success is measured at the final step only.**

This is the standard Fetch convention, but it means a policy that reaches the goal and then drifts away counts as a failure. The final-distance metric is reported alongside.
