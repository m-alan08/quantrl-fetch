import sys
import types

import pandas as pd
import pytest

from quantrl.config import load_config
from quantrl.training import train
from quantrl.utils.wandb_logging import WandbUnavailableError

SMOKE = [
    "device=cpu",
    "sac.hidden_sizes=[16, 16]",
    "sac.batch_size=16",
    "train.total_steps=200",
    "train.random_steps=100",
    "train.update_after=100",
    "train.updates_per_step=0.2",
    "train.eval_interval=100",
    "train.eval_episodes=1",
    "eval.episodes=1",
    "eval.bootstrap_resamples=20",
    "her.buffer_episodes=5",
]


class FakeRun:
    def __init__(self, **settings):
        self.settings = settings
        self.logged = []
        self.summary = {}
        self.finished = False

    def log(self, row, step):
        self.logged.append((step, row))

    def finish(self):
        self.finished = True


def test_metrics_go_to_local_files_without_any_network(tmp_path):
    config = load_config("configs/fetch_reach.json", [*SMOKE, f"output_dir={tmp_path}"])
    train(config, verbose=False)
    progress = pd.read_csv(config.run_dir / "progress.csv")
    assert list(progress["step"]) == [100, 200]
    assert {"eval_success_rate", "critic_loss", "actor_grad_norm"} <= set(progress.columns)
    for file_name in ("config.json", "metadata.json", "summary.json"):
        assert (config.run_dir / file_name).stat().st_size > 0


def test_wandb_logging_is_offline_by_default_and_mirrors_progress(tmp_path, monkeypatch):
    runs = []

    def init(**settings):
        runs.append(FakeRun(**settings))
        return runs[-1]

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    config = load_config(
        "configs/fetch_reach.json", [*SMOKE, f"output_dir={tmp_path}", "logging.wandb=true"]
    )
    train(config, verbose=False)
    run = runs[0]
    assert run.settings["mode"] == "offline"
    assert run.settings["dir"] == str(config.run_dir)
    assert [step for step, _ in run.logged] == [100, 200]
    assert "success_rate" in run.summary
    assert run.finished


def test_missing_wandb_package_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    config = load_config(
        "configs/fetch_reach.json", [*SMOKE, f"output_dir={tmp_path}", "logging.wandb=true"]
    )
    with pytest.raises(WandbUnavailableError, match="uv sync --group wandb"):
        train(config, verbose=False)


def test_unknown_wandb_mode_is_rejected():
    with pytest.raises(ValueError, match="logging.wandb_mode"):
        load_config(None, ["logging.wandb_mode=cloud"])
