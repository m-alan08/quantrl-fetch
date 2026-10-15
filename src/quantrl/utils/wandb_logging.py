from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from quantrl.config import RunConfig


class WandbUnavailableError(RuntimeError):
    pass


class WandbLogger:
    def __init__(self, config: RunConfig, run_dir: Path):
        self.run = None
        if not config.logging.wandb:
            return
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise WandbUnavailableError(
                "logging.wandb is true but the wandb package is not installed; "
                "install it with 'uv sync --group wandb' or set logging.wandb=false"
            ) from error
        self.run = wandb.init(
            project=config.logging.wandb_project,
            name=f"{config.name}_seed{config.seed}",
            mode=config.logging.wandb_mode,
            dir=str(run_dir),
            config=config.to_dict(),
        )

    def log(self, row: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(row, step=int(row["step"]))

    def finish(self, summary: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(summary["final_eval"])
            self.run.finish()
