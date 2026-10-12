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
- Hidden `.venv` on macOS: in the development checkout under `~/Documents`, every file inside `.venv` ended up with the macOS `hidden` file flag. Python 3.12.14 deliberately skips hidden `.pth` files, so the editable install of `quantrl` silently stopped being importable (`ModuleNotFoundError: No module named 'quantrl'`). A fresh clone in another directory did not show the problem. The fix is `chflags -R nohidden .venv`. As a second line of defense, pytest adds `src` to `sys.path` through `pythonpath = ["src"]` in `pyproject.toml`.

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
| `src/quantrl/eval/stats.py` | Percentile bootstrap confidence intervals and aggregation across training seeds. |
| `src/quantrl/viz/video.py` | Rendering labeled evaluation episodes and writing mp4 and gif files. |
| `src/quantrl/viz/plots.py` | Shared plot style and training curves across seeds. |
| `src/quantrl/utils/seeding.py` | Seeds Python, NumPy, and torch, enables deterministic torch algorithms, and resolves the device string. |
| `src/quantrl/utils/run_info.py` | Collects the git commit, whether the tree was dirty, package versions, Python version, and platform for each run. |
| `src/quantrl/quant/` | Stubs for Phase 3 and Phase 4. |
| `scripts/train.py` | Trains one or more seeds from a config file with optional overrides. |
| `scripts/evaluate.py` | Evaluates checkpoints and writes a tidy per-seed CSV and a JSON summary across seeds. |
| `scripts/make_plots.py` | Plots training curves for a set of run directories. |
| `scripts/make_video.py` | Renders labeled evaluation episodes of a checkpoint to mp4 and gif. |
| `scripts/sweep.py` | Stub for Phase 4. |
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

## Design decisions

- Configs are plain dataclasses serialized to JSON, so there is no hidden global state and every run can save exactly what it used.
- Training runs on CPU by default. The networks are small multilayer perceptrons, so CPU is fast enough and gives bitwise reproducible runs for a fixed seed. `device = "auto"` picks MPS when available.
- Only the actor is quantized at first, because the actor is the part that would be deployed on a robot. Critics stay fp32 unless `quant.quantize_critics` is turned on later.
- The actor has a single output layer that produces both the mean and the log standard deviation. This keeps the network a plain stack of linear layers, which makes swapping every layer for a quantized one straightforward.
- Run outputs (checkpoints, logs, raw videos) go to `runs/`, which is ignored by git. Only small summaries are copied into `results/` and `assets/`.

## Known limitations

- The mujoco pin above should be revisited when gymnasium-robotics releases a fix.
- FetchReach is easy: all seeds solve it within a few thousand steps, so it can only reveal quantization damage that is large. FetchPush (Phase 2) is the more sensitive benchmark.
- The sparse reward only asks the gripper to get within 5 cm. A trained FetchReach policy ends episodes about 3 cm from the goal on average, so there is little margin before quantization noise flips a success into a failure, and success rate alone can hide changes in precision. The mean final distance is logged for that reason.
- Three training seeds give a rough estimate of spread across seeds, not a tight one.
