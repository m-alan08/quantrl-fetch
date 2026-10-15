from __future__ import annotations

import dataclasses
import json
import math
import re
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
    device: str = "auto"
    deterministic: bool = True
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


QUANT_METHODS = ("fp32", "fp32_finetune", "ptq", "qat_finetune", "qat_scratch")
FULL_PRECISION_METHODS = ("fp32", "fp32_finetune")
RELU_GRIDS = ("signed", "unsigned")
DEVICE_NAMES = ("auto", "cpu", "cuda", "mps")
FULL_PRECISION_BITS = 32
MIN_BITS = 2
MAX_BITS = 16


class ConfigError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (_is_int(value) or isinstance(value, float)) and math.isfinite(value)


class _Checker:
    def __init__(self) -> None:
        self.problems: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.problems.append(message)

    def integer(self, name: str, value: Any, minimum: int | None = None) -> None:
        if not _is_int(value):
            self.problems.append(f"{name} must be an integer, got {value!r}")
        elif minimum is not None and value < minimum:
            self.problems.append(f"{name} must be at least {minimum}, got {value}")

    def number(
        self,
        name: str,
        value: Any,
        low: float | None = None,
        high: float | None = None,
        low_inclusive: bool = True,
        high_inclusive: bool = True,
    ) -> None:
        if not _is_number(value):
            self.problems.append(f"{name} must be a finite number, got {value!r}")
            return
        if low is not None and (value < low or (value == low and not low_inclusive)):
            bound = ">=" if low_inclusive else ">"
            self.problems.append(f"{name} must be {bound} {low}, got {value}")
        if high is not None and (value > high or (value == high and not high_inclusive)):
            bound = "<=" if high_inclusive else "<"
            self.problems.append(f"{name} must be {bound} {high}, got {value}")

    def boolean(self, name: str, value: Any) -> None:
        if not isinstance(value, bool):
            self.problems.append(f"{name} must be true or false, got {value!r}")

    def bits(self, name: str, value: Any) -> None:
        if not (_is_int(value) and (value == FULL_PRECISION_BITS or MIN_BITS <= value <= MAX_BITS)):
            self.problems.append(
                f"{name} must be an integer from {MIN_BITS} to {MAX_BITS}, "
                f"or {FULL_PRECISION_BITS} for full precision, got {value!r}"
            )

    def text(self, name: str, value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            self.problems.append(f"{name} must be a non-empty string, got {value!r}")


def validate_config(config: RunConfig) -> RunConfig:
    check = _Checker()
    check.text("name", config.name)
    check.require(
        not isinstance(config.name, str) or ("/" not in config.name and "\\" not in config.name),
        f"name must not contain path separators, got {config.name!r}",
    )
    check.integer("seed", config.seed, 0)
    check.require(
        isinstance(config.device, str)
        and (config.device in DEVICE_NAMES or re.fullmatch(r"cuda:\d+", config.device) is not None),
        f"device must be one of {DEVICE_NAMES} or 'cuda:<index>', got {config.device!r}",
    )
    check.boolean("deterministic", config.deterministic)
    check.integer("torch_threads", config.torch_threads, 1)
    check.text("output_dir", config.output_dir)

    env = config.env
    check.text("env.env_id", env.env_id)
    if env.max_episode_steps is not None:
        check.integer("env.max_episode_steps", env.max_episode_steps, 1)
    check.number("env.obs_clip", env.obs_clip, 0, low_inclusive=False)
    check.number("env.normalized_clip", env.normalized_clip, 0, low_inclusive=False)
    check.number("env.normalizer_eps", env.normalizer_eps, 0, low_inclusive=False)

    quant = config.quant
    check.require(
        quant.method in QUANT_METHODS,
        f"quant.method must be one of {QUANT_METHODS}, got {quant.method!r}",
    )
    check.bits("quant.weight_bits", quant.weight_bits)
    check.bits("quant.activation_bits", quant.activation_bits)
    check.boolean("quant.per_channel_weights", quant.per_channel_weights)
    check.number("quant.activation_ema_momentum", quant.activation_ema_momentum, 0, 1)
    check.boolean("quant.quantize_critics", quant.quantize_critics)
    check.require(
        quant.relu_activation_grid in RELU_GRIDS,
        f"quant.relu_activation_grid must be one of {RELU_GRIDS}, "
        f"got {quant.relu_activation_grid!r}",
    )
    full_precision = (
        quant.weight_bits == FULL_PRECISION_BITS and quant.activation_bits == FULL_PRECISION_BITS
    )
    if quant.method in FULL_PRECISION_METHODS:
        check.require(
            full_precision,
            f"quant.method {quant.method!r} needs weight_bits and activation_bits of 32",
        )
    elif quant.method in ("qat_finetune", "qat_scratch"):
        check.require(
            not full_precision,
            f"quant.method {quant.method!r} needs weight_bits or activation_bits below 32",
        )
    if quant.method in ("qat_finetune", "fp32_finetune"):
        check.require(
            config.train.init_checkpoint is not None,
            f"quant.method {quant.method!r} needs train.init_checkpoint",
        )
    if quant.method in ("fp32", "qat_scratch"):
        check.require(
            config.train.init_checkpoint is None,
            f"quant.method {quant.method!r} trains from scratch, "
            "so train.init_checkpoint must be null",
        )

    sac = config.sac
    check.require(
        isinstance(sac.hidden_sizes, list)
        and len(sac.hidden_sizes) > 0
        and all(_is_int(size) and size > 0 for size in sac.hidden_sizes),
        f"sac.hidden_sizes must be a non-empty list of positive integers, got {sac.hidden_sizes!r}",
    )
    check.number("sac.actor_lr", sac.actor_lr, 0, low_inclusive=False)
    check.number("sac.critic_lr", sac.critic_lr, 0, low_inclusive=False)
    check.number("sac.alpha_lr", sac.alpha_lr, 0)
    check.number("sac.gamma", sac.gamma, 0, 1, low_inclusive=False, high_inclusive=False)
    check.number("sac.tau", sac.tau, 0, 1, low_inclusive=False)
    check.integer("sac.batch_size", sac.batch_size, 1)
    check.number("sac.initial_alpha", sac.initial_alpha, 0, low_inclusive=False)
    if sac.target_entropy is not None:
        check.number("sac.target_entropy", sac.target_entropy)
    check.number("sac.log_std_min", sac.log_std_min)
    check.number("sac.log_std_max", sac.log_std_max)
    if _is_number(sac.log_std_min) and _is_number(sac.log_std_max):
        check.require(
            sac.log_std_min < sac.log_std_max,
            f"sac.log_std_min ({sac.log_std_min}) must be below "
            f"sac.log_std_max ({sac.log_std_max})",
        )
    check.boolean("sac.clip_q_target", sac.clip_q_target)

    check.integer("her.replay_k", config.her.replay_k, 0)
    check.integer("her.buffer_episodes", config.her.buffer_episodes, 1)

    train = config.train
    check.integer("train.total_steps", train.total_steps, 1)
    check.integer("train.random_steps", train.random_steps, 0)
    check.integer("train.update_after", train.update_after, 0)
    check.number("train.updates_per_step", train.updates_per_step, 0)
    check.integer("train.eval_interval", train.eval_interval, 1)
    check.integer("train.eval_episodes", train.eval_episodes, 1)
    check.integer("train.checkpoint_interval", train.checkpoint_interval, 0)
    if train.init_checkpoint is not None:
        check.text("train.init_checkpoint", train.init_checkpoint)
    check.boolean("train.eval_at_start", train.eval_at_start)

    evaluation = config.eval
    check.integer("eval.episodes", evaluation.episodes, 1)
    check.integer("eval.seed_offset", evaluation.seed_offset, 0)
    check.integer("eval.calibration_seed_offset", evaluation.calibration_seed_offset, 0)
    check.integer("eval.calibration_episodes", evaluation.calibration_episodes, 1)
    check.integer("eval.bootstrap_resamples", evaluation.bootstrap_resamples, 1)
    check.number(
        "eval.confidence", evaluation.confidence, 0, 1, low_inclusive=False, high_inclusive=False
    )
    seed_fields = (
        evaluation.seed_offset,
        evaluation.episodes,
        evaluation.calibration_seed_offset,
        evaluation.calibration_episodes,
        train.eval_episodes,
    )
    if all(_is_int(value) for value in seed_fields):
        evaluation_end = evaluation.seed_offset + max(evaluation.episodes, train.eval_episodes)
        calibration_end = evaluation.calibration_seed_offset + evaluation.calibration_episodes
        check.require(
            calibration_end <= evaluation.seed_offset
            or evaluation_end <= evaluation.calibration_seed_offset,
            "calibration seeds and evaluation seeds overlap: "
            f"evaluation uses [{evaluation.seed_offset}, {evaluation_end}), "
            f"calibration uses [{evaluation.calibration_seed_offset}, {calibration_end})",
        )

    if check.problems:
        raise ConfigError(
            "Invalid config:\n" + "\n".join(f"  - {problem}" for problem in check.problems)
        )
    return config


def config_from_dict(values: dict[str, Any]) -> RunConfig:
    return validate_config(_build_dataclass(RunConfig, values))


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
