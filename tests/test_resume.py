import pandas as pd
import pytest
import torch

import quantrl.training as training
from quantrl.checkpoint import load_checkpoint
from quantrl.config import load_config
from quantrl.resume import ResumeMismatchError, load_resume_state, resume_path
from quantrl.training import train

BASE = [
    "device=cpu",
    "sac.hidden_sizes=[32, 32]",
    "sac.batch_size=32",
    "train.random_steps=100",
    "train.update_after=100",
    "train.updates_per_step=0.3",
    "train.eval_interval=150",
    "train.eval_episodes=2",
    "eval.episodes=3",
    "eval.bootstrap_resamples=50",
    "her.buffer_episodes=8",
]


def config_for(tmp_path, name, total_steps, *extra):
    return load_config(
        "configs/fetch_reach.json",
        [
            *BASE,
            f"output_dir={tmp_path}",
            f"name={name}",
            f"train.total_steps={total_steps}",
            *extra,
        ],
    )


def progress_without_time(config):
    return pd.read_csv(config.run_dir / "progress.csv").drop(columns=["wall_seconds"])


def assert_same_training(first, second):
    pd.testing.assert_frame_equal(progress_without_time(first), progress_without_time(second))
    first_agent = load_checkpoint(first.run_dir / "checkpoint.pt")["agent"]
    second_agent = load_checkpoint(second.run_dir / "checkpoint.pt")["agent"]
    for network in ("actor", "critic", "critic_target"):
        for key, value in first_agent[network].items():
            assert torch.equal(value, second_agent[network][key]), f"{network}.{key} differs"
    assert torch.equal(first_agent["log_alpha"], second_agent["log_alpha"])


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("uninterrupted")
    config = config_for(tmp_path, "run", 600)
    summary = train(config, verbose=False)
    return config, summary


def test_stopping_and_extending_matches_an_uninterrupted_run(uninterrupted, tmp_path):
    reference, reference_summary = uninterrupted
    train(config_for(tmp_path, "run", 300), verbose=False)
    resumed = config_for(tmp_path, "run", 600, "train.resume=true")
    summary = train(resumed, verbose=False)
    assert_same_training(reference, resumed)
    assert summary["final_eval"] == reference_summary["final_eval"]
    assert summary["episodes"] == reference_summary["episodes"]


def test_resuming_after_a_crash_matches_an_uninterrupted_run(uninterrupted, tmp_path, monkeypatch):
    reference, reference_summary = uninterrupted
    crashing = config_for(tmp_path, "run", 600, "train.checkpoint_interval=200")
    original_collect = training.collect_episode
    calls = {"count": 0}

    def collect_then_crash(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 10:
            raise KeyboardInterrupt
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(training, "collect_episode", collect_then_crash)
    with pytest.raises(KeyboardInterrupt):
        train(crashing, verbose=False)
    monkeypatch.setattr(training, "collect_episode", original_collect)
    saved = load_resume_state(resume_path(crashing.run_dir))
    assert saved["counters"]["total_steps"] == 400
    assert pd.read_csv(crashing.run_dir / "progress.csv")["step"].max() == 450

    resumed = config_for(tmp_path, "run", 600, "train.checkpoint_interval=200", "train.resume=true")
    summary = train(resumed, verbose=False)
    assert_same_training(reference, resumed)
    assert summary["final_eval"] == reference_summary["final_eval"]


def test_resume_refuses_a_changed_config(tmp_path):
    train(config_for(tmp_path, "run", 150), verbose=False)
    changed = config_for(tmp_path, "run", 300, "train.resume=true", "sac.actor_lr=0.0005")
    with pytest.raises(ResumeMismatchError, match="sac.actor_lr"):
        train(changed, verbose=False)


def test_resume_without_saved_state_starts_fresh(tmp_path):
    config = config_for(tmp_path, "fresh", 150, "train.resume=true")
    summary = train(config, verbose=False)
    assert summary["total_steps"] == 150
    assert resume_path(config.run_dir).exists()


def test_saved_state_holds_the_full_training_state(uninterrupted):
    reference, _ = uninterrupted
    state = load_resume_state(resume_path(reference.run_dir))
    assert set(state) >= {
        "config",
        "agent",
        "update_count",
        "normalizer",
        "buffer",
        "counters",
        "recent_metrics",
        "recent_train_success",
        "rng",
    }
    assert set(state["agent"]) >= {
        "actor",
        "critic",
        "critic_target",
        "log_alpha",
        "actor_optimizer",
        "critic_optimizer",
        "alpha_optimizer",
    }
    assert set(state["rng"]) >= {"python", "numpy_global", "torch_cpu", "env", "action_space"}
    assert state["buffer"]["size"] == 8
    assert state["buffer"]["observations"].shape == (8, 51, 10)
    assert state["counters"]["total_steps"] == 600
