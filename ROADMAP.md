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

- [ ] Environment factory and goal-conditioned observation normalizer
- [ ] SAC from scratch (twin critics, Polyak targets, automatic entropy tuning, tanh-squashed Gaussian actor)
- [ ] Episode replay buffer with HER future relabeling
- [ ] Training loop that saves the resolved config, git commit, seed, and package versions
- [ ] Deterministic evaluation with bootstrap confidence intervals
- [ ] FetchReach reaches a success rate near 100 percent on 3 training seeds
- [ ] Rendered rollout video of the learned policy

## Phase 2: FetchPush at fp32

- [ ] Hyperparameters for FetchPush
- [ ] Train 3 seeds and record how long training takes on this machine

## Phase 3: quantization module and PTQ

- [ ] Symmetric uniform quantizer, per-tensor and per-channel weights, EMA activation ranges
- [ ] Straight-through estimator and QuantLinear layer, actor builder for any bit width
- [ ] Unit tests (integer grid, at most 2^bits values, STE gradients, PTQ at 32 bits equals fp32)
- [ ] PTQ calibration on observations from the trained policy
- [ ] PTQ evaluation of the Phase 1 policy at 8, 6, 5, 4, 3, 2 bits

## Phase 4: QAT and the full sweep

- [ ] QAT fine-tuning from an fp32 checkpoint
- [ ] QAT from scratch
- [ ] Sweep over bit width and method, tidy CSV in results/
- [ ] Success rate versus bit width plot with one line per method and shaded intervals
- [ ] Side-by-side video of an fp32 policy and a low-bit policy

## Phase 5: post-training

- [ ] Expert dataset from the fp32 policy
- [ ] Behavior cloning student with a quantized actor
- [ ] Choose and document an online post-training method here before implementing it (residual RL first, RLPD if time allows)
- [ ] Measure whether post-training recovers low-bit performance

## Phase 6: final write-up

- [ ] README with results table, charts, embedded clips, reproduction steps, limitations, and citations
