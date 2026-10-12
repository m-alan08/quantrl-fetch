import json

import torch

from quantrl.checkpoint import load_policy
from quantrl.config import load_config
from quantrl.training import train

SMOKE_OVERRIDES = [
    "sac.hidden_sizes=[32, 32]",
    "sac.batch_size=32",
    "train.total_steps=400",
    "train.random_steps=100",
    "train.update_after=100",
    "train.updates_per_step=0.2",
    "train.eval_interval=200",
    "train.eval_episodes=2",
    "eval.episodes=2",
    "eval.bootstrap_resamples=100",
    "her.buffer_episodes=20",
]


def smoke_config(tmp_path, name):
    return load_config(
        "configs/fetch_reach.json",
        [*SMOKE_OVERRIDES, f"output_dir={tmp_path}", f"name={name}"],
    )


def test_training_runs_and_writes_outputs(tmp_path):
    config = smoke_config(tmp_path, "smoke")
    summary = train(config, verbose=False)
    run_dir = config.run_dir
    assert summary["total_steps"] == 400
    for file_name in ["config.json", "metadata.json", "progress.csv", "summary.json"]:
        assert (run_dir / file_name).exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["seed"] == 0
    assert "torch" in metadata["packages"]
    loaded = load_policy(run_dir / "checkpoint.pt")
    assert loaded.actor.input_dim == 13


def test_training_is_reproducible_on_cpu(tmp_path):
    first = smoke_config(tmp_path, "first")
    second = smoke_config(tmp_path, "second")
    train(first, verbose=False)
    train(second, verbose=False)
    first_actor = load_policy(first.run_dir / "checkpoint.pt").actor.state_dict()
    second_actor = load_policy(second.run_dir / "checkpoint.pt").actor.state_dict()
    for key, value in first_actor.items():
        assert torch.equal(value, second_actor[key])
