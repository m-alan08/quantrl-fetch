from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnvConfig:
    env_id: str = "FetchReach-v4"
    max_episode_steps: int | None = None
    obs_clip: float = 200.0
    normalized_clip: float = 5.0
    normalizer_eps: float = 0.01


@dataclass
class QuantConfig:
    method: str = "fp32"
    weight_bits: int = 32
    activation_bits: int = 32
    per_channel_weights: bool = False
    activation_ema_momentum: float = 0.99
    quantize_critics: bool = False
    relu_activation_grid: str = "unsigned"


@dataclass
class SACConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    gamma: float = 0.98
    tau: float = 0.005
    batch_size: int = 256
    initial_alpha: float = 1.0
    target_entropy: float | None = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    clip_q_target: bool = True


@dataclass
class HERConfig:
    replay_k: int = 4
    buffer_episodes: int = 20_000


@dataclass
class TrainConfig:
    total_steps: int = 30_000
    random_steps: int = 1_000
    update_after: int = 1_000
    updates_per_step: float = 1.0
    eval_interval: int = 2_500
    eval_episodes: int = 50
    checkpoint_interval: int = 0
    init_checkpoint: str | None = None
    eval_at_start: bool = False


@dataclass
class EvalConfig:
    episodes: int = 100
    seed_offset: int = 1_000_000
    calibration_seed_offset: int = 2_000_000
    calibration_episodes: int = 20
    bootstrap_resamples: int = 10_000
    confidence: float = 0.95


@dataclass
class RunConfig:
    name: str = "fetch_reach_fp32"
    seed: int = 0
    device: str = "cpu"
    torch_threads: int = 1
    output_dir: str = "runs"
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    her: HERConfig = field(default_factory=HERConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / f"{self.name}_seed{self.seed}"


def _build_dataclass(cls: type, values: dict[str, Any]) -> Any:
    known_fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known_fields)
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        default_value = getattr(cls(), key)
        if dataclasses.is_dataclass(default_value):
            if not isinstance(value, dict):
                raise TypeError(
                    f"Section {cls.__name__}.{key} must be an object, got {type(value).__name__}"
                )
            kwargs[key] = _build_dataclass(type(default_value), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(values: dict[str, Any]) -> RunConfig:
    return _build_dataclass(RunConfig, values)


def saved_config_from_dict(values: dict[str, Any]) -> RunConfig:
    quant = values.get("quant", {})
    quantized = quant.get("weight_bits", 32) != 32 or quant.get("activation_bits", 32) != 32
    if quantized and "relu_activation_grid" not in quant:
        values = {**values, "quant": {**quant, "relu_activation_grid": "signed"}}
    return config_from_dict(values)


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must look like section.key=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        target = values
        *parents, leaf = dotted_key.split(".")
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = _parse_override_value(raw_value)
    return values


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    values: dict[str, Any] = {}
    if path is not None:
        values = json.loads(Path(path).read_text())
    values = apply_overrides(values, overrides or [])
    return config_from_dict(values)


def save_config(config: RunConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2) + "\n")
