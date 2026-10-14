# Notes

This file explains how the code is organized and why. It is updated in the same pull request as any code change it describes. Modules that are still stubs raise `NotImplementedError` and say which phase will fill them in.

## Environment findings

- Python: the system Python on the development Mac is 3.14, which is outside the supported range. The project pins Python 3.12 through `.python-version` and `requires-python = ">=3.10,<3.13"`, and uv installs that interpreter automatically.
- Fetch environments: gymnasium-robotics 1.4.2 registers `FetchReach-v4`, `FetchPush-v4`, `FetchSlide-v4`, `FetchPickAndPlace-v4`, the matching `Dense` variants, and the older `v1` ids. This project uses the `v4` ids. Episodes last 50 steps, actions are 4-dimensional in [-1, 1], and observations are dicts with `observation` (10 values for Reach, 25 for Push), `achieved_goal` (3), and `desired_goal` (3).
- MuJoCo version pin: with mujoco 3.12 and newer, creating any Fetch environment fails with an `AssertionError` in `gymnasium_robotics.utils.mujoco_utils.set_joint_qpos`. Newer mujoco returns joint types as `numpy.int32` while the enum members no longer compare equal inside a tuple membership test, so `joint_type in (mjJNT_HINGE, mjJNT_SLIDE)` is false for a valid slide joint. mujoco 3.11.0 works, so `pyproject.toml` pins `mujoco>=3.1,<3.12`.
- Rendering on macOS: `gym.make(env_id, render_mode="rgb_array")` renders offscreen through the default GLFW backend without setting `MUJOCO_GL`. Close environments explicitly with `env.close()`, otherwise the GL context destructor prints a harmless traceback at interpreter exit.
- Rewards: `env.unwrapped.compute_reward(achieved_goal, desired_goal, info)` is vectorized over a batch, which HER relabeling relies on. The sparse reward is 0 when the goal is within 5 cm and -1 otherwise.
- Importing gymnasium_robotics prints a notice about Adroit environments. It does not affect the Fetch tasks.
- CI installs the CPU-only torch wheel on Linux through a uv index rule in `pyproject.toml`, which avoids downloading CUDA packages.
- Hidden `.venv` on macOS: in the development checkout under `~/Documents`, every file inside `.venv` ended up with the macOS `hidden` file flag. Python 3.12.14 deliberately skips hidden `.pth` files, so the editable install of `quantrl` silently stopped being importable (`ModuleNotFoundError: No module named 'quantrl'`). A fresh clone in another directory did not show the problem. The fix is `chflags -R nohidden .venv`. The flag kept coming back within an hour, and `~/Documents` on the development Mac is synced by iCloud Drive, which is the likely cause. iCloud skips anything whose name ends in `.nosync`, so the development checkout keeps its environment in `.venv.nosync` with a `.venv` symlink pointing to it; uv follows the symlink. As a second line of defense, pytest adds `src` to `sys.path` through `pythonpath = ["src"]` in `pyproject.toml`.

## Module map

| Path | Purpose |
| --- | --- |
| `src/quantrl/config.py` | Dataclass configs (`RunConfig` with `env`, `sac`, `her`, `quant`, `train`, `eval` sections), JSON loading, and `section.key=value` overrides. Unknown keys and non-object sections raise errors so typos do not silently fall back to defaults. |
| `src/quantrl/envs/factory.py` | `make_env` wraps `gym.make` for a config, and `env_dimensions` reads observation, goal, and action sizes plus the episode length. |
| `src/quantrl/envs/normalizer.py` | Running mean and standard deviation normalizers for observations and goals. |
| `src/quantrl/algos/networks.py` | The MLP builder, the tanh-squashed Gaussian actor, and the twin Q critic. Every linear layer is created by a `linear_factory`, which is the hook that lets the quantization module swap in quantized layers without touching SAC. |
| `src/quantrl/algos/sac.py` | `SACAgent`: action selection and one gradient update of critics, actor, and temperature. |
| `src/quantrl/algos/replay.py` | `HindsightReplayBuffer`: stores whole episodes and relabels goals at sampling time. |
| `src/quantrl/algos/bc.py`, `post_training.py` | Stubs for Phase 5. |
| `src/quantrl/training.py` | The training loop: collect an episode, store it, update the normalizer, run gradient updates, evaluate periodically, save checkpoints and logs. |
| `src/quantrl/checkpoint.py` | Saving and loading checkpoints, rebuilding the actor and normalizer, and wrapping the actor as a deterministic policy function. |
| `src/quantrl/eval/evaluate.py` | Deterministic evaluation on a fixed list of environment seeds. |
| `src/quantrl/eval/quantized.py` | PTQ evaluation grid: `PTQSetting` (bit width, what to quantize, weight granularity) and `evaluate_ptq_settings`, which calibrates once per checkpoint and evaluates every setting on the fixed evaluation seeds. |
| `src/quantrl/eval/stats.py` | Percentile bootstrap confidence intervals and aggregation across training seeds. |
| `src/quantrl/viz/video.py` | Rendering labeled evaluation episodes and writing mp4 and gif files. |
| `src/quantrl/viz/plots.py` | Shared plot style and training curves across seeds. |
| `src/quantrl/utils/seeding.py` | Seeds Python, NumPy, and torch, enables deterministic torch algorithms, and resolves the device string. |
| `src/quantrl/utils/run_info.py` | Collects the git commit, whether the tree was dirty, package versions, Python version, and platform for each run. |
| `src/quantrl/quant/quantizer.py` | Pure functions: bit-width validation, scales, integer quantization, fake quantization with the straight-through estimator. |
| `src/quantrl/quant/layers.py` | `ActivationQuantizer` (range tracking and fake quantization of a layer input), `QuantLinear` (a drop-in `nn.Linear` with quantized weights and inputs), and `build_quantized_actor`. |
| `src/quantrl/quant/ptq.py` | Post-training quantization: copy fp32 weights into a quantized actor, calibrate activation ranges, and collect calibration inputs by running a policy. |
| `src/quantrl/quant/qat.py` | QAT fine-tuning setup: loads a trained agent into a quantized one, applies the configured learning rates, and calibrates activation ranges the same way PTQ does. |
| `src/quantrl/sweep.py` | The bit width by method sweep: builds QAT run configs, trains them in parallel worker processes (reusing finished runs), and evaluates fp32, PTQ, and QAT actors on the same episodes. |
| `scripts/train.py` | Trains one or more seeds from a config file with optional overrides. |
| `scripts/evaluate.py` | Evaluates checkpoints and writes a tidy per-seed CSV and a JSON summary across seeds. |
| `scripts/evaluate_ptq.py` | Runs the PTQ grid for a set of checkpoints and writes a tidy per-seed CSV and a summary CSV across seeds. |
| `scripts/make_plots.py` | Plots training curves (`--runs`), the PTQ grid (`--ptq-csv`), or the method comparison (`--sweep-csv`). |
| `scripts/make_video.py` | Renders labeled evaluation episodes of a checkpoint to mp4 and gif, or two policies side by side on the same episodes (`--compare-checkpoint`, or `--compare-ptq-bits` for a PTQ copy of the same actor). |
| `scripts/sweep.py` | Runs the sweep and writes a tidy per-seed CSV, a summary across seeds, and per-run training times. |
| `configs/` | JSON run configs. Any field not given keeps its dataclass default. |
| `results/` | Small CSV and JSON metric files that are committed. |
| `assets/` | A few small clips and plots used by the README. |

## How training works

### Observations and normalization

A Fetch observation is a dict. The policy sees `concat(normalize(observation), normalize(desired_goal))`, a 13-dimensional vector for FetchReach. Each part has its own running normalizer that keeps the sum and sum of squares of everything it has seen, so its mean and standard deviation are exact over all data so far. Raw values are first clipped to plus or minus 200, then normalized as `(x - mean) / max(std, 0.01)`, then clipped to plus or minus 5. The goal normalizer is updated with both the desired goals and the achieved goals of each episode, because relabeled goals come from achieved goals. The normalizer is saved inside every checkpoint since the actor is meaningless without it.

Normalized inputs live in a bounded range of plus or minus 5, which matters for quantization later: the first layer's input range is known in advance.

### Soft Actor-Critic

SAC learns a stochastic policy that maximizes reward plus an entropy bonus. Notation: `x` is the normalized input, `a` the action, `r` the reward, `gamma` the discount.

- Actor: an MLP outputs a mean `mu(x)` and a log standard deviation (clamped to [-20, 2]) for each of the 4 action dimensions. A sample is `a = tanh(mu + sigma * eps)` with `eps` from a standard normal, which keeps actions inside [-1, 1]. Its log probability is the Gaussian log density minus the log of the tanh derivative, `sum log(1 - tanh(u)^2)`, computed in the numerically stable form `2 * (log 2 - u - softplus(-2u))`. A test checks this against `torch.distributions` with a `TanhTransform`.
- Critics: two independent Q networks `Q1`, `Q2` take `(x, a)`. Each has a target copy that tracks it slowly through Polyak averaging, `target = (1 - tau) * target + tau * online`, after every update.
- Critic target: `y = r + gamma * (min(Q1_target, Q2_target)(x', a') - alpha * log pi(a' | x'))` with `a'` sampled from the current actor at the next input. Taking the minimum of two critics reduces overestimation. Fetch episodes never terminate early (they end only on the 50-step time limit), so there is no terminal mask: bootstrapping through a time limit is correct.
- Target clipping: with rewards in {-1, 0}, any discounted return lies in [-1 / (1 - gamma), 0]. With `sac.clip_q_target` on (the default), `y` is clipped to that range, a common stabilizer for sparse goal-reaching tasks. Because the entropy bonus can be positive, the clipped soft value is a slight approximation; this is a deliberate trade for stability.
- Actor loss: `mean(alpha * log pi(a | x) - min(Q1, Q2)(x, a))` with `a` resampled through the reparameterization trick so gradients flow through the critics into the actor. Critic parameters are frozen during this step.
- Temperature: `alpha = exp(log_alpha)` is learned by minimizing `-log_alpha * (log pi(a | x) + target_entropy)` with `target_entropy = -action_dim = -4`. When the policy is more random than the target, alpha shrinks, and the other way around.

Defaults (in `SACConfig`): two hidden layers of 256 units with ReLU, Adam with learning rate 1e-3 for actor, critics, and temperature, `gamma = 0.98`, `tau = 0.005`, batch size 256, initial alpha 1.0.

### Hindsight Experience Replay

With a sparse reward, almost every early episode fails and gives the same reward of -1 at every step, so there is nothing to learn from. HER fixes this by pretending, after the fact, that the goal had been somewhere the robot actually got to.

The buffer stores whole episodes: 51 observations and achieved goals (including the final state), 50 desired goals and 50 actions. Sampling a batch does the following for each element:

1. Pick a stored episode uniformly and a time step `t` uniformly in [0, 49].
2. With probability `1 - 1 / (1 + k)` (0.8 for the default `k = 4`), replace the goal with the achieved goal at a future index `t'` drawn uniformly from [t + 1, 50]. This is the "future" strategy.
3. Recompute the reward as `compute_reward(achieved_goal[t + 1], goal)`, which is 0 if the gripper ended that step within 5 cm of the (possibly new) goal.

Relabeling at sampling time rather than at storage time means each stored transition can be reused with many different goals. Tests check that relabeled goals always come from the same episode and from a strictly later index, that the relabel fraction matches `k`, that rewards match the relabeled goal, and that the buffer overwrites the oldest episode when full.

### Training loop

Each iteration collects one full 50-step episode. The first `train.random_steps` (1000) environment steps use uniform random actions; after that, actions are sampled from the stochastic actor. After each episode the loop runs `updates_per_step * 50` gradient updates (one update per environment step by default) once `train.update_after` steps have been collected. Every `train.eval_interval` steps the deterministic policy is evaluated on `train.eval_episodes` fixed seeds and a row is appended to `progress.csv`. At the end, `checkpoint.pt` is written and a final evaluation on `eval.episodes` seeds goes into `summary.json`.

Each run directory `runs/<name>_seed<seed>/` holds `config.json` (the fully resolved config), `metadata.json` (seed, git commit and dirty flag, package versions, Python version, platform, device, thread count, start time), `progress.csv`, `summary.json`, and `checkpoint.pt`.

### Reproducibility

`seed_everything` seeds Python, NumPy, and torch and turns on `torch.use_deterministic_algorithms`. The training environment is seeded once through `env.reset(seed=seed)` and its action space through `action_space.seed(seed)`. The replay buffer draws from a NumPy generator created from the same seed. Torch runs on one CPU thread by default (`torch_threads = 1`). A test trains twice with the same seed and checks that the two actors are bitwise identical.

### Evaluation and statistics

Evaluation uses the deterministic action `tanh(mu(x))`. Episode `i` of an evaluation is reset with seed `eval.seed_offset + i` (offset 1,000,000), so every policy is tested on exactly the same goals. Success is the environment's `is_success` flag at the last step. Return is the sum of rewards, so for FetchReach it is minus the number of steps spent more than 5 cm from the goal.

Within one training seed, the confidence interval for success rate or return is a percentile bootstrap over episodes: resample the episode outcomes with replacement 10,000 times, take the mean of each resample, and report the 2.5th and 97.5th percentiles. Across training seeds, `aggregate_across_seeds` reports the mean, sample standard deviation, minimum, maximum, and a bootstrap interval over the seed means. With only 3 seeds that interval is coarse (it can only take values between the smallest and largest seed), so the min and max are reported alongside it.

## How quantization works

### The quantizer

Quantization replaces a real number with one of a small set of evenly spaced values. This project uses symmetric uniform quantization with `b` bits:

- The integer range is `[-L, L]` with `L = 2^(b-1) - 1`, so 8 bits gives [-127, 127], 4 bits gives [-7, 7], 3 bits gives [-3, 3], and 2 bits gives [-1, 1]. That is `2L + 1 = 2^b - 1` levels, one fewer than `2^b`, because the range is kept symmetric around zero (the value `-2^(b-1)` is unused). Zero is always exactly representable.
- A scale `s` maps the real range onto the integers: `s = max_abs / L`, where `max_abs` is the largest magnitude the quantizer expects to see. The scale is floored at a tiny positive value (1e-8, or the smallest normal number of the tensor's dtype if that is larger, as in float16) so an all-zero tensor quantizes to zeros instead of dividing by zero.
- Quantize: `q = clamp(round(x / s), -L, L)`, an integer.
- Dequantize: `x_hat = q * s`.

Inside the range, the error `|x_hat - x|` is at most `s / 2`. Values beyond `max_abs` are clipped to `+-L * s`.

`bits = 32` means "disabled": the fake-quantization functions and the quantized layers return their input unchanged, and `QuantLinear` computes exactly what `nn.Linear` does. The integer-level functions (`quantize_symmetric`, `max_quant_level`) reject 32, because there is no integer grid to return. Bit widths must be integers: 32, or 2 to 16. Anything else, such as 33 or 4.5, raises an error instead of silently disabling quantization.

### Weights

A weight tensor's scale comes from its own largest magnitude at the moment of the forward pass, so nothing is ever clipped.

- Per tensor: one `max_abs` for the whole weight matrix.
- Per channel: one `max_abs` per output row (`weight.abs().amax(dim=1)`), so a row with small weights is not forced onto the coarse grid of a row with large weights. A test checks that per-channel quantization has lower error than per-tensor when one row is much larger than the rest.

Biases stay fp32. In integer inference engines biases are usually stored at 32 bits anyway, and they are a tiny fraction of the parameters.

### Activations

Each `QuantLinear` quantizes its input, so for the actor the quantized activations are the normalized observation (input of the first layer) and the output of each hidden ReLU. The final layer's output (the action mean and log standard deviation) is left in fp32, since the tanh squashing and the action itself are part of the controller, not the network.

Unlike weights, activation ranges cannot be read off a fixed tensor, because they depend on the data. `ActivationQuantizer` keeps a scalar `running_max_abs`:

- In training mode, and unless frozen, each batch updates it with an exponential moving average: `running = m * running + (1 - m) * max|x|` with `m = quant.activation_ema_momentum = 0.99`. Values of `m` outside [0, 1] are rejected. The very first batch initializes it directly.
- In evaluation mode, or after `freeze()`, the range is fixed.
- During PTQ calibration, the quantizer records the plain maximum of `|x|` over all calibration batches and passes activations through unquantized, then freezes.

Until a range has been observed, the quantizer passes values through unchanged.

### Fake quantization and the straight-through estimator

Training and evaluation both use fake quantization: values are quantized and immediately dequantized, so the arithmetic stays in fp32 but every weight and activation is restricted to the values an integer implementation could represent. The output of a fake-quantized network matches what an integer kernel would compute up to fp32 rounding of the accumulation.

Rounding is a step function, so its true derivative is zero almost everywhere and backpropagation through it would stop all learning. The straight-through estimator (STE) treats rounding as the identity in the backward pass. In code:

`round_ste(v) = v + (round(v) - v).detach()`

The forward value is `round(v)`. The `detach` removes the correction term from the graph, so the gradient of `round_ste(v)` with respect to `v` is exactly 1. The full fake quantizer is

`x_hat = clamp(round_ste(x / s), -L, L) * s`

and its gradient with respect to `x` is 1 where `x` is inside `[-L * s, L * s]` and 0 where the clamp is active. This "clipped STE" stops pushing on values that are already saturated. The scale `s` is treated as a constant (weights' scale is computed from detached values, and activation ranges are buffers), so there is no learned step size. Tests check the gradient is exactly 1 inside the range and 0 outside, and that gradients reach `QuantLinear` weights.

### Building a quantized actor

`build_quantized_actor` constructs the same `GaussianActor` used by SAC but passes a `linear_factory` that creates `QuantLinear` layers with the configured bit widths. Because `QuantLinear` subclasses `nn.Linear`, the weight and bias parameter names are identical, so an fp32 actor's state dict loads directly. The only extra entries are the activation quantizers' buffers. `load_full_precision_weights` checks that nothing else is missing, unexpected, or of the wrong shape before loading.

### Post-training quantization (PTQ)

1. Collect calibration inputs by running the trained fp32 policy deterministically on calibration episodes and recording the normalized inputs it sees (`collect_calibration_inputs`). Calibration episodes should use seeds disjoint from the evaluation seeds.
2. Build a quantized actor and copy the fp32 weights into it.
3. Run the calibration inputs through it with every activation quantizer in calibration mode, then freeze the ranges (`calibrate_activation_ranges`). An empty calibration set raises an error before any range is touched, rather than silently leaving activations unquantized. The quantized actor is created on the same device and dtype as the fp32 actor.
4. Evaluate it like any other policy.

Calibration episodes use seeds starting at `eval.calibration_seed_offset` (2,000,000), and there are `eval.calibration_episodes` of them (20, which gives 1,000 inputs for 50-step episodes). These seeds never overlap the evaluation seeds (starting at 1,000,000), so the quantization ranges are not fitted to the exact episodes they are scored on.

The PTQ evaluation grid has three targets: `both` (weights and activations at N bits), `weights` (weights at N bits, activations fp32), and `activations` (activations at N bits, weights fp32). Weight scales can be per tensor or per channel. Separating the targets shows which part of the network is the bottleneck.

All activation ranges are calibrated in a single pass over fp32 activations, so a later layer's range is measured on activations that were not yet quantized upstream. A test checks that PTQ at 32 bits gives bitwise identical outputs to the fp32 actor, and a check on the trained FetchReach checkpoint confirmed identical actions on 1,000 real calibration inputs.

### What PTQ on FetchReach showed

Measured on the 3 Phase 1 seeds (full numbers in the README and `results/`):

- No loss at 8 and 6 bits, a few points at most at 5 bits, about 85 percent success at 4 bits with weights and activations quantized, and failure at 2 bits.
- Weights alone are robust down to 4 bits, and down to 3 bits with per-channel scales. Activations alone cause most of the loss at 4 and 3 bits.
- The calibrated input range of the first layer is about plus or minus 2.4 to 2.6 standard deviations, so a 4-bit grid (7 positive levels) has a step of about 0.35 standard deviations, which is about 3 cm of gripper or goal position. That is the same scale as the policy's typical final distance (3.1 cm) and the 5 cm success radius, which makes input quantization a plausible cause of the 4-bit drop. It has not been isolated experimentally. A direct test would keep the first layer's input in higher precision.

### Quantization-aware training (QAT)

QAT keeps the fake quantizers in the actor while it trains, so the optimizer sees the rounded network and can move the weights to places where rounding hurts less. The straight-through estimator described above is what lets gradients pass through the rounding. The quantized actor is built through the same `linear_factory` hook as for PTQ, chosen automatically from `config.quant` by `actor_linear_factory`. `quant.method` records which method a run used (`fp32`, `ptq`, `qat_finetune`, `qat_scratch`) and only labels results; the bit widths are what change the network.

During training, the actor runs in two modes. Inside gradient updates it is in training mode, so each batch of replayed inputs updates the activation ranges by exponential moving average. When acting in the environment or being evaluated it is switched to evaluation mode (`evaluation_mode` in `algos/sac.py`), so ranges never move on single observations. For an fp32 actor the two modes compute exactly the same thing, so this does not change Phase 1 runs; a rerun of seed 0 reproduced its progress log exactly.

**QAT from scratch** (`quant.method = qat_scratch`) trains with the same config, budget (30,000 steps for FetchReach), and seeds as the fp32 baseline, only with a quantized actor. Weights are fake-quantized from the first update. Activation ranges start unset, so the very first acting steps pass activations through unquantized; the first gradient update initializes every range and from then on the actor is fully quantized.

**QAT fine-tuning** (`quant.method = qat_finetune`, with `train.init_checkpoint` pointing at a trained fp32 run) starts from the trained agent:

1. Load the fp32 actor weights into the quantized actor, plus the critics, target critics, temperature, optimizer states, and normalizer.
2. Calibrate the actor's activation ranges on the same calibration episodes PTQ uses, then unfreeze them so the moving average can follow the data. At step 0 the fine-tuned actor is therefore exactly the PTQ actor (a test checks this), and `train.eval_at_start` logs that starting point.
3. Continue SAC with a fresh replay buffer: no random actions, 1,000 steps of data from the quantized policy before the first update, then 10,000 environment steps in total.

The recipe uses an actor learning rate of 1e-4 (ten times lower than training) and a frozen temperature (`sac.alpha_lr = 0`). In pilot runs on seed 0 (scored on the 50 periodic evaluation episodes), the training learning rate made the 3-bit actor fall from 46 percent (its PTQ starting point) to about 20 percent while the temperature drifted upward. With the lower learning rate and frozen temperature it stayed between 28 and 54 percent over 10,000 steps, and 4 bits went from 78 to 100 percent within 1,000 steps either way.

One bug found on the way: `torch.optim.Optimizer.load_state_dict` also restores the saved learning rate, so a fine-tuning run silently used the old learning rates no matter what the config said. `apply_configured_learning_rates` now resets them after loading, and a test checks it.

### The sweep

`scripts/sweep.py` runs every combination of method and bit width for each fp32 training seed. Each QAT run's seed matches the fp32 seed it came from (fine-tuning) or uses the same seed as that baseline (from scratch). Runs go to worker processes, one torch thread each, and a run whose `summary.json` already exists is reused rather than retrained, so an interrupted sweep can be resumed. Afterwards every actor (fp32, PTQ copies of each fp32 actor, and every QAT actor) is evaluated on the same 100 evaluation episodes. Weights use per-channel scales in the sweep unless `--per-tensor` is passed, since PTQ showed per-channel scales are at least as good at 3 bits and above.

In the plot, the fp32 result is drawn as the leftmost point of every method's line so each line starts from the same reference.

### Side-by-side video

`compose_side_by_side` places two equally sized frame sequences next to each other with an 8-pixel divider (keeping the width a multiple of 8, which video encoders prefer). Both policies are rendered on the same environment seeds, so they see the same goal; each episode is 51 frames plus a short hold on the last frame, so the two sides stay in step. Labels show the policy (for example "fp32" or "3-bit PTQ") and success or failure on the last frame.

## Design decisions

- Configs are plain dataclasses serialized to JSON, so there is no hidden global state and every run can save exactly what it used.
- Training runs on CPU by default. The networks are small multilayer perceptrons, so CPU is fast enough and gives bitwise reproducible runs for a fixed seed. `device = "auto"` picks MPS when available.
- Only the actor is quantized at first, because the actor is the part that would be deployed on a robot. Critics stay fp32 unless `quant.quantize_critics` is turned on later.
- The actor has a single output layer that produces both the mean and the log standard deviation. This keeps the network a plain stack of linear layers, which makes swapping every layer for a quantized one straightforward.
- Run outputs (checkpoints, logs, raw videos) go to `runs/`, which is ignored by git. Only small summaries are copied into `results/` and `assets/`.

## Known limitations

- Activations are quantized symmetrically even after a ReLU, where they are never negative, so half of the integer range goes unused for hidden activations. An unsigned activation quantizer would give post-ReLU activations one extra bit of resolution. The symmetric version is kept for now because it is what the project specifies; this matters most at 2 and 3 bits, where a post-ReLU activation can only take 2 or 4 distinct values.
- The scale is the maximum magnitude, not a percentile or a learned step size, so a single outlier activation in the calibration data stretches the grid for everything else.

- The mujoco pin above should be revisited when gymnasium-robotics releases a fix.
- FetchReach is easy: all seeds solve it within a few thousand steps, so it can only reveal quantization damage that is large. FetchPush (Phase 2) is the more sensitive benchmark.
- The sparse reward only asks the gripper to get within 5 cm. A trained FetchReach policy ends episodes about 3 cm from the goal on average, so there is little margin before quantization noise flips a success into a failure, and success rate alone can hide changes in precision. The mean final distance is logged for that reason.
- Three training seeds give a rough estimate of spread across seeds, not a tight one.
