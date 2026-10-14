# Roadmap

Each phase lands on main through its own pull requests. A box is checked only after the work is merged and, where it involves numbers, after those numbers come from a real run.

## Phase 0: setup

- [x] Verify the machine (Apple Silicon, Python 3.12 pinned through uv, uv and gh available)
- [x] Pin dependencies in pyproject.toml with a uv lockfile
- [x] Verify FetchReach-v4 and FetchPush-v4 (and the Dense variants) run with the installed versions
- [x] Verify offscreen rendering with `render_mode="rgb_array"` on macOS
- [x] Full repository skeleton with stub modules
- [x] README written from scratch, docs/NOTES.md, this roadmap
- [x] CI running ruff and pytest on pull requests

## Phase 1: SAC and HER at fp32 on FetchReach

- [x] Environment factory and goal-conditioned observation normalizer
- [x] SAC from scratch (twin critics, Polyak targets, automatic entropy tuning, tanh-squashed Gaussian actor)
- [x] Episode replay buffer with HER future relabeling
- [x] Training loop that saves the resolved config, git commit, seed, and package versions
- [x] Deterministic evaluation with bootstrap confidence intervals
- [x] FetchReach reaches a success rate near 100 percent on 3 training seeds (measured: 1.00 on 100 evaluation episodes for each of seeds 0, 1, 2)
- [x] Rendered rollout video of the learned policy (`assets/fetch_reach_fp32.mp4` and `.gif`)
- [x] Bitwise reproducibility on CPU checked by a test and by rerunning seed 0

## Phase 2: FetchPush at fp32

- [x] Hyperparameters for FetchPush (`configs/fetch_push.json`)
- [x] Train 3 seeds and record how long training takes on this machine (success 0.98, 0.99, 0.98; about 1 hour 47 minutes of compute per seed with other jobs sharing the CPU)

## Phase 3: quantization module and PTQ

- [x] Symmetric uniform quantizer, per-tensor and per-channel weights, EMA activation ranges
- [x] Straight-through estimator and QuantLinear layer, actor builder for any bit width
- [x] Unit tests (integer grid, at most 2^bits values, STE gradients, PTQ at 32 bits equals fp32)
- [x] PTQ calibration on observations from the trained policy
- [x] PTQ evaluation of the Phase 1 policy at 8, 6, 5, 4, 3, 2 bits (3 seeds, plus weights-only, activations-only, and per-channel ablations)

## Phase 4: QAT and the full sweep

- [x] QAT fine-tuning from an fp32 checkpoint (starts exactly at the PTQ actor, plus an fp32 fine-tuning control)
- [x] QAT from scratch
- [x] Sweep over bit width and method, tidy CSV in results/ (FetchReach)
- [x] PTQ grid on FetchPush
- [x] QAT sweep on FetchPush (QAT from scratch keeps 96 percent at 5 bits)
- [x] Success rate versus bit width plot with one line per method and shaded intervals
- [x] Side-by-side video of an fp32 policy and a low-bit policy

## Phase 5: post-training

- [ ] Expert dataset from the fp32 policy
- [ ] Behavior cloning student with a quantized actor
- [ ] Choose and document an online post-training method here before implementing it (residual RL first, RLPD if time allows)
- [ ] Measure whether post-training recovers low-bit performance

## Phase 6: final write-up

- [ ] README with results table, charts, embedded clips, reproduction steps, limitations, and citations
