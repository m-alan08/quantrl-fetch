import json

import torch

from quantrl.checkpoint import load_policy
from quantrl.config import load_config
from quantrl.training import ProgressLog, UpdateScheduler, train

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


def test_update_scheduler_keeps_the_long_run_rate_for_fractional_values():
    scheduler = UpdateScheduler(0.01)
    counts = [scheduler.updates_for(50) for _ in range(10)]
    assert counts == [0, 1] * 5
    scheduler = UpdateScheduler(0.03)
    assert sum(scheduler.updates_for(50) for _ in range(100)) == 150
    assert UpdateScheduler(1.0).updates_for(50) == 50


def test_progress_log_accepts_columns_that_appear_later(tmp_path):
    log = ProgressLog(tmp_path / "progress.csv")
    log.write({"step": 1, "eval_success_rate": 0.0})
    log.write({"step": 2, "eval_success_rate": 0.5, "critic_loss": 0.1})
    log.write({"step": 3, "eval_success_rate": 1.0, "critic_loss": 0.2})
    lines = (tmp_path / "progress.csv").read_text().splitlines()
    assert lines[0] == "step,eval_success_rate,critic_loss"
    assert lines[1:] == ["1,0.0,", "2,0.5,0.1", "3,1.0,0.2"]


def test_training_with_evaluation_before_updates(tmp_path):
    config = load_config(
        "configs/fetch_reach.json",
        [
            *SMOKE_OVERRIDES,
            "train.eval_interval=50",
            "train.update_after=200",
            f"output_dir={tmp_path}",
            "name=early_eval",
        ],
    )
    train(config, verbose=False)
    header = (config.run_dir / "progress.csv").read_text().splitlines()[0]
    assert "critic_loss" in header
