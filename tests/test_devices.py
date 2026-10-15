import numpy as np
import pandas as pd
import pytest
import torch

from quantrl.algos.networks import GaussianActor
from quantrl.algos.sac import SACAgent
from quantrl.checkpoint import load_checkpoint, load_policy
from quantrl.config import QuantConfig, SACConfig, load_config
from quantrl.quant.layers import actor_linear_factory, build_quantized_actor
from quantrl.quant.ptq import quantize_actor_post_training
from quantrl.training import train
from quantrl.utils.device import mps_available, resolve_device


def device_available(name):
    if name == "cpu":
        return True
    if name == "mps":
        return mps_available()
    return torch.cuda.is_available()


DEVICES = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            not device_available(name), reason=f"{name} is not available on this machine"
        ),
    )
    for name in ("cpu", "mps", "cuda")
]

SMOKE = [
    "sac.hidden_sizes=[32, 32]",
    "sac.batch_size=32",
    "train.random_steps=100",
    "train.update_after=100",
    "train.updates_per_step=0.3",
    "train.eval_interval=150",
    "train.eval_episodes=2",
    "eval.episodes=2",
    "eval.bootstrap_resamples=20",
    "her.buffer_episodes=8",
]


def batch(rows=32, dim=13, action_dim=4):
    rng = np.random.default_rng(0)
    return {
        "inputs": rng.normal(size=(rows, dim)).astype(np.float32),
        "next_inputs": rng.normal(size=(rows, dim)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(rows, action_dim)).astype(np.float32),
        "rewards": -np.ones(rows, dtype=np.float32),
    }


@pytest.mark.parametrize("device", DEVICES)
def test_resolve_device_returns_the_requested_backend(device):
    assert resolve_device(device).type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bits", [32, 8, 4, 2])
def test_sac_update_runs_on_device(device, bits):
    torch.manual_seed(0)
    method = "fp32" if bits == 32 else "qat_scratch"
    quant = QuantConfig(method=method, weight_bits=bits, activation_bits=bits)
    agent = SACAgent(
        13,
        4,
        SACConfig(hidden_sizes=[32, 32]),
        device,
        actor_linear_factory=actor_linear_factory(quant),
    )
    for _ in range(3):
        metrics = agent.update(batch())
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(p.device.type == device for p in agent.actor.parameters())
    assert all(p.dtype == torch.float32 for p in agent.actor.parameters())
    actions = agent.act(batch()["inputs"], deterministic=True)
    assert actions.dtype == np.float32 and np.all(np.abs(actions) <= 1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_quantized_actor_gradients_flow_on_device(device):
    torch.manual_seed(0)
    actor = build_quantized_actor(
        13, 4, SACConfig(hidden_sizes=[32, 32]), QuantConfig(weight_bits=4, activation_bits=4)
    ).to(device)
    actor.train()
    inputs = torch.randn(64, 13, device=device)
    actions, log_probs = actor.sample(inputs)
    (actions.pow(2).sum() + log_probs.sum()).backward()
    for parameter in actor.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_ptq_at_32_bits_is_exact_on_device(device):
    torch.manual_seed(0)
    fp32_actor = GaussianActor(13, 4, [32, 32]).to(device)
    inputs = torch.randn(128, 13, device=device)
    quantized = quantize_actor_post_training(
        fp32_actor, SACConfig(hidden_sizes=[32, 32]), QuantConfig(), inputs
    )
    assert torch.equal(
        fp32_actor.deterministic_action(inputs), quantized.deterministic_action(inputs)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_training_on_device_writes_a_checkpoint_that_loads_on_cpu(device, tmp_path):
    config = load_config(
        "configs/fetch_reach.json",
        [*SMOKE, f"device={device}", f"output_dir={tmp_path}", "train.total_steps=300"],
    )
    summary = train(config, verbose=False)
    assert summary["total_steps"] == 300
    loaded = load_policy(config.run_dir / "checkpoint.pt")
    assert next(loaded.actor.parameters()).device.type == "cpu"
    metadata = pd.read_json(config.run_dir / "metadata.json", typ="series")
    assert str(metadata["device"]).startswith(device)


@pytest.mark.parametrize("device", DEVICES)
def test_same_seed_gives_the_same_training_metrics_on_device(device, tmp_path):
    runs = []
    for name in ("first", "second"):
        config = load_config(
            "configs/fetch_reach.json",
            [
                *SMOKE,
                f"device={device}",
                f"output_dir={tmp_path}",
                f"name={name}",
                "train.total_steps=450",
            ],
        )
        train(config, verbose=False)
        runs.append(pd.read_csv(config.run_dir / "progress.csv").drop(columns=["wall_seconds"]))
    pd.testing.assert_frame_equal(runs[0], runs[1])
    assert len(runs[0]) == 3


@pytest.mark.parametrize("device", DEVICES)
def test_resume_on_device_matches_an_uninterrupted_run(device, tmp_path):
    def config_for(name, steps, *extra):
        return load_config(
            "configs/fetch_reach.json",
            [
                *SMOKE,
                f"device={device}",
                f"output_dir={tmp_path}",
                f"name={name}",
                f"train.total_steps={steps}",
                *extra,
            ],
        )

    reference = config_for("reference", 450)
    train(reference, verbose=False)
    train(config_for("resumed", 300), verbose=False)
    resumed = config_for("resumed", 450, "train.resume=true")
    train(resumed, verbose=False)
    pd.testing.assert_frame_equal(
        pd.read_csv(reference.run_dir / "progress.csv").drop(columns=["wall_seconds"]),
        pd.read_csv(resumed.run_dir / "progress.csv").drop(columns=["wall_seconds"]),
    )
    first = load_checkpoint(reference.run_dir / "checkpoint.pt")["agent"]["actor"]
    second = load_checkpoint(resumed.run_dir / "checkpoint.pt")["agent"]["actor"]
    for key in first:
        assert torch.equal(first[key], second[key])
