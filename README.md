# quantrl-fetch

How few bits does a robot arm policy actually need? This project trains reinforcement learning policies for a simulated Fetch robot arm (MuJoCo, via Gymnasium Robotics) and then squeezes the policy's actor network down to low-bit integer precision: 8 bits, then 6, 5, 4, 3, and 2. It compares three ways of getting a low-bit actor (quantize after training, fine-tune with quantization in the loop, or train with quantization from the start) and later asks whether a post-training stage, behavior cloning followed by online post-training, can win back the performance that quantization costs. The motivation is deployment: a policy that runs on a microcontroller or a small accelerator has to live with low-precision arithmetic, and it is worth knowing where that starts to break a closed-loop controller and what it takes to repair it.

**Status:** Phase 0 (setup and skeleton) is done, with CI waiting on a token scope. Phase 1 (SAC and HER on FetchReach) is next. See [ROADMAP.md](ROADMAP.md).

## Key results

Pending. No numbers are reported here until they come from real runs.

## Demo

Pending. A side-by-side clip of an fp32 policy and a low-bit policy, plus the main success-versus-bit-width chart, will go here after Phase 4.

## Method

**Three ways to get a low-bit actor.**

1. Post-training quantization (PTQ): train the actor in fp32, then quantize its weights and activations to N bits. Activation ranges are calibrated on observations collected by running the trained policy.
2. QAT fine-tuning: train in fp32, then keep training the actor with fake quantization in its forward pass so it can adapt to the rounding.
3. QAT from scratch: fake quantization is in the actor for the entire training run.

**The quantizer.** Values are mapped to a symmetric uniform grid: divide by a scale, round to the nearest integer, clamp to the range an N-bit signed integer can hold, and multiply back. Weights use a scale from their largest absolute value (per tensor or per output channel), and activations use a range tracked with an exponential moving average during training and frozen for evaluation. Rounding has zero gradient almost everywhere, so training uses a straight-through estimator that treats rounding as the identity in the backward pass.

**The RL setup.** The agent is Soft Actor-Critic (SAC) with twin Q critics, Polyak-averaged target networks, automatic entropy tuning, and a tanh-squashed Gaussian actor. The Fetch tasks give a sparse reward (success or not), so the replay buffer uses Hindsight Experience Replay (HER) with the future strategy: some stored transitions are relabeled as if the goal had been a position the gripper or object actually reached later in the same episode. Observations, achieved goals, and desired goals are concatenated after running mean and standard deviation normalization with clipping. Only the actor is quantized; the critics stay fp32.

Everything (SAC, HER, the quantizer) is implemented from scratch in PyTorch.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv installs the pinned Python 3.12 automatically.

```bash
git clone https://github.com/m-alan08/quantrl-fetch.git
cd quantrl-fetch
uv sync
uv run pytest -q
```

Training and video commands arrive with Phase 1.

## Repository layout

```
src/quantrl/
  algos/     SAC, networks, HER replay buffer, later BC and post-training
  quant/     quantizers, fake-quant layers, PTQ calibration, QAT utilities
  envs/      environment creation and observation normalization
  eval/      deterministic evaluation and bootstrap statistics
  viz/       video rendering and plotting
  utils/     seeding and run metadata
  config.py  dataclass configs
configs/     JSON run configs
scripts/     train, evaluate, sweep, make_video, make_plots
tests/       pytest suite
results/     small metric files (CSV, JSON)
assets/      README clips and charts
docs/        NOTES.md, a plain-language guide to the code
```

## Reproducing results

Pending. Each phase will list its config files, seeds, and measured runtimes on an Apple Silicon Mac here.

## Limitations and caveats

- mujoco is pinned below 3.12 because newer releases break Fetch environment creation in gymnasium-robotics 1.4.2. Details are in [docs/NOTES.md](docs/NOTES.md).
- Quantization here is simulated (fake quantization in fp32 arithmetic). It models the accuracy effect of low-bit weights and activations but does not measure speed or energy on real integer hardware.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## References and credits

- Haarnoja, Zhou, Abbeel, Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018. Haarnoja et al. Soft Actor-Critic Algorithms and Applications. 2018.
- Andrychowicz et al. Hindsight Experience Replay. NeurIPS 2017.
- Plappert et al. Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research. 2018 (the Fetch tasks).
- Krishnan, Lam, Chitlangia, Wan, Barth-Maron, Faust, Reddi. QuaRL: Quantization for Fast and Environmentally Sustainable Reinforcement Learning. TMLR 2022 (earlier versions titled "Quantized Reinforcement Learning"). This project is inspired by QuaRL and its ActorQ system for quantized actors in distributed RL; all code here is my own.
- Bengio, Leonard, Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. 2013 (the straight-through estimator).
- Jacob et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018.
- Todorov, Erez, Tassa. MuJoCo: A physics engine for model-based control. IROS 2012. Gymnasium and Gymnasium Robotics from the Farama Foundation.

Released under the MIT License. See [LICENSE](LICENSE).
