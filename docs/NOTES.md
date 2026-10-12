# Notes

This file explains how the code is organized and why. It is updated in the same pull request as any code change it describes. Modules that are still stubs raise `NotImplementedError` and say which phase will fill them in.

## Environment findings

- Python: the system Python on the development Mac is 3.14, which is outside the supported range. The project pins Python 3.12 through `.python-version` and `requires-python = ">=3.10,<3.13"`, and uv installs that interpreter automatically.
- Fetch environments: gymnasium-robotics 1.4.2 registers `FetchReach-v4`, `FetchPush-v4`, `FetchSlide-v4`, `FetchPickAndPlace-v4`, the matching `Dense` variants, and the older `v1` ids. This project uses the `v4` ids. Episodes last 50 steps, actions are 4-dimensional in [-1, 1], and observations are dicts with `observation` (10 values for Reach, 25 for Push), `achieved_goal` (3), and `desired_goal` (3).
- MuJoCo version pin: with mujoco 3.12 and newer, creating any Fetch environment fails with an `AssertionError` in `gymnasium_robotics.utils.mujoco_utils.set_joint_qpos`. Newer mujoco returns joint types as `numpy.int32` while the enum members no longer compare equal inside a tuple membership test, so `joint_type in (mjJNT_HINGE, mjJNT_SLIDE)` is false for a valid slide joint. mujoco 3.11.0 works, so `pyproject.toml` pins `mujoco>=3.1,<3.12`.
- Rendering on macOS: `gym.make(env_id, render_mode="rgb_array")` renders offscreen through the default GLFW backend without setting `MUJOCO_GL`. Close environments explicitly with `env.close()`, otherwise the GL context destructor prints a harmless traceback at interpreter exit.
- Rewards: `env.unwrapped.compute_reward(achieved_goal, desired_goal, info)` is vectorized over a batch, which HER relabeling relies on. The sparse reward is 0 when the goal is within 5 cm and -1 otherwise.
- CI installs the CPU-only torch wheel on Linux through a uv index rule in `pyproject.toml`, which avoids downloading CUDA packages.

## Module map

| Path | Purpose |
| --- | --- |
| `src/quantrl/config.py` | Dataclass configs (`RunConfig` with `env`, `sac`, `her`, `quant`, `train`, `eval` sections), JSON loading, and `section.key=value` overrides. Unknown keys raise an error so typos do not silently fall back to defaults. |
| `src/quantrl/envs/` | Environment creation and the goal-conditioned observation normalizer. |
| `src/quantrl/algos/` | SAC, its networks, the HER replay buffer, and later behavior cloning and post-training. |
| `src/quantrl/quant/` | Quantizer functions, fake-quant layers, PTQ calibration, and QAT utilities. |
| `src/quantrl/eval/` | Deterministic evaluation and bootstrap statistics. |
| `src/quantrl/viz/` | Video rendering and plotting. |
| `src/quantrl/utils/` | Seeding and run metadata (git commit, package versions). |
| `src/quantrl/training.py` | The training loop that ties the pieces together. |
| `scripts/` | Command line entry points: `train.py`, `evaluate.py`, `sweep.py`, `make_video.py`, `make_plots.py`. |
| `configs/` | JSON run configs. Any field not given keeps its dataclass default. |
| `results/` | Small CSV and JSON metric files that are committed. |
| `assets/` | A few small clips and plots used by the README. |

## Design decisions

- Configs are plain dataclasses serialized to JSON, so there is no hidden global state and every run can save exactly what it used.
- Training runs on CPU by default. The networks are small multilayer perceptrons, so CPU is fast enough and gives bitwise reproducible runs for a fixed seed.
- Only the actor is quantized at first, because the actor is the part that would be deployed on a robot. Critics stay fp32 unless `quant.quantize_critics` is turned on later.
- Run outputs (checkpoints, logs, raw videos) go to `runs/`, which is ignored by git. Only small summaries are copied into `results/` and `assets/`.

## Known limitations

- The mujoco pin above should be revisited when gymnasium-robotics releases a fix.
