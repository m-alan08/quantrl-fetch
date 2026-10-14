from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantrl.checkpoint import LoadedPolicy, deterministic_policy_function
from quantrl.config import QuantConfig
from quantrl.envs.factory import make_env
from quantrl.eval.evaluate import evaluate_policy, evaluation_seeds
from quantrl.quant.ptq import collect_calibration_inputs, quantize_actor_post_training
from quantrl.quant.quantizer import FULL_PRECISION_BITS

QUANTIZATION_TARGETS = ("both", "weights", "activations")


@dataclass(frozen=True)
class PTQSetting:
    bits: int
    target: str
    per_channel_weights: bool

    def quant_config(self, base: QuantConfig) -> QuantConfig:
        if self.target not in QUANTIZATION_TARGETS:
            raise ValueError(f"target must be one of {QUANTIZATION_TARGETS}, got {self.target!r}")
        weight_bits = self.bits if self.target in ("both", "weights") else FULL_PRECISION_BITS
        activation_bits = (
            self.bits if self.target in ("both", "activations") else FULL_PRECISION_BITS
        )
        return QuantConfig(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            per_channel_weights=self.per_channel_weights,
            activation_ema_momentum=base.activation_ema_momentum,
            quantize_critics=False,
        )


def ptq_settings(
    bit_widths: Sequence[int],
    targets: Sequence[str] = ("both",),
    per_channel_options: Sequence[bool] = (False,),
) -> list[PTQSetting]:
    settings = []
    for target in targets:
        for per_channel in per_channel_options:
            for bits in bit_widths:
                if target == "activations" and per_channel:
                    continue
                settings.append(PTQSetting(bits, target, per_channel))
    return settings


def evaluate_ptq_settings(
    loaded: LoadedPolicy,
    settings: Sequence[PTQSetting],
    episodes: int | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    config = loaded.config
    env = make_env(config.env)
    try:
        calibration_inputs = collect_calibration_inputs(
            env,
            deterministic_policy_function(loaded.actor),
            loaded.normalizer,
            evaluation_seeds(config.eval.calibration_seed_offset, config.eval.calibration_episodes),
        )
        if seeds is None:
            seeds = evaluation_seeds(config.eval.seed_offset, episodes or config.eval.episodes)
        rows = []
        for setting in settings:
            quant_config = setting.quant_config(config.quant)
            actor = quantize_actor_post_training(
                loaded.actor, config.sac, quant_config, calibration_inputs
            )
            result, _ = evaluate_policy(
                env,
                deterministic_policy_function(actor),
                loaded.normalizer,
                seeds,
                config.eval.bootstrap_resamples,
                config.eval.confidence,
            )
            is_baseline = setting.bits == FULL_PRECISION_BITS
            rows.append(
                {
                    "env_id": config.env.env_id,
                    "name": config.name,
                    "seed": config.seed,
                    "method": "fp32" if is_baseline else "ptq",
                    "bits": setting.bits,
                    "target": setting.target,
                    "weight_granularity": "per_channel"
                    if setting.per_channel_weights
                    else "per_tensor",
                    "weight_bits": quant_config.weight_bits,
                    "activation_bits": quant_config.activation_bits,
                    "calibration_inputs": int(np.asarray(calibration_inputs).shape[0]),
                    **result.to_dict(),
                }
            )
    finally:
        env.close()
    return rows
