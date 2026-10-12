from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from quantrl.algos.networks import GaussianActor
from quantrl.config import QuantConfig, SACConfig
from quantrl.envs.normalizer import GoalObservationNormalizer
from quantrl.eval.evaluate import PolicyFunction
from quantrl.quant.layers import activation_quantizers, build_quantized_actor

QUANTIZER_BUFFER_SUFFIXES = (
    "input_quantizer.running_max_abs",
    "input_quantizer.initialized",
    "input_quantizer.frozen",
)


def load_full_precision_weights(quantized_actor: GaussianActor, fp32_state: dict) -> None:
    target_state = quantized_actor.state_dict()
    expected_keys = {key for key in target_state if not key.endswith(QUANTIZER_BUFFER_SUFFIXES)}
    missing = sorted(expected_keys - set(fp32_state))
    unexpected = sorted(set(fp32_state) - expected_keys)
    mismatched = sorted(
        key
        for key in expected_keys & set(fp32_state)
        if fp32_state[key].shape != target_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise KeyError(
            f"State dict mismatch: missing {missing}, unexpected {unexpected}, "
            f"shape mismatch {mismatched}"
        )
    quantized_actor.load_state_dict(fp32_state, strict=False)


@torch.no_grad()
def calibrate_activation_ranges(
    model: torch.nn.Module, calibration_inputs: np.ndarray | torch.Tensor, batch_size: int = 1024
) -> None:
    quantizers = list(activation_quantizers(model))
    for quantizer in quantizers:
        quantizer.start_calibration()
    inputs = torch.as_tensor(calibration_inputs, dtype=torch.float32)
    for start in range(0, inputs.shape[0], batch_size):
        model(inputs[start : start + batch_size])
    for quantizer in quantizers:
        quantizer.finish_calibration()


def quantize_actor_post_training(
    fp32_actor: GaussianActor,
    sac_config: SACConfig,
    quant_config: QuantConfig,
    calibration_inputs: np.ndarray | torch.Tensor,
) -> GaussianActor:
    quantized_actor = build_quantized_actor(
        fp32_actor.input_dim, fp32_actor.action_dim, sac_config, quant_config
    )
    load_full_precision_weights(quantized_actor, fp32_actor.state_dict())
    calibrate_activation_ranges(quantized_actor, calibration_inputs)
    quantized_actor.eval()
    return quantized_actor


def collect_calibration_inputs(
    env: gym.Env,
    policy: PolicyFunction,
    normalizer: GoalObservationNormalizer,
    seeds: Sequence[int],
) -> np.ndarray:
    collected = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            inputs = normalizer(observation["observation"], observation["desired_goal"])
            collected.append(inputs)
            action = policy(inputs[None])[0]
            observation, _, terminated, truncated, _ = env.step(action)
    return np.stack(collected).astype(np.float32)
