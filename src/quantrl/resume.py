from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from quantrl.algos.replay import HindsightReplayBuffer
from quantrl.config import RunConfig, saved_config_from_dict

RESUME_FILE_NAME = "resume_state.pt"
RESUME_FORMAT_VERSION = 1
FIELDS_ALLOWED_TO_CHANGE = (("train", "total_steps"), ("train", "resume"))


class ResumeMismatchError(ValueError):
    pass


def resume_path(run_dir: Path) -> Path:
    return run_dir / RESUME_FILE_NAME


def capture_rng_state(env: gym.Env, device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "env": env.unwrapped.np_random,
        "action_space": env.action_space.np_random,
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], env: gym.Env, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    env.unwrapped.np_random = state["env"]
    env.action_space._np_random = state["action_space"]
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if device.type == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])


def capture_buffer(buffer: HindsightReplayBuffer) -> dict[str, Any]:
    size = buffer.size
    return {
        "observations": buffer.observations[:size].copy(),
        "achieved_goals": buffer.achieved_goals[:size].copy(),
        "desired_goals": buffer.desired_goals[:size].copy(),
        "actions": buffer.actions[:size].copy(),
        "next_index": buffer.next_index,
        "size": size,
        "capacity": buffer.capacity,
        "rng": buffer.rng,
    }


def restore_buffer(buffer: HindsightReplayBuffer, state: dict[str, Any]) -> None:
    if state["capacity"] != buffer.capacity:
        raise ResumeMismatchError(
            f"replay buffer capacity is {state['capacity']} in the saved state, "
            f"{buffer.capacity} now"
        )
    size = state["size"]
    buffer.observations[:size] = state["observations"]
    buffer.achieved_goals[:size] = state["achieved_goals"]
    buffer.desired_goals[:size] = state["desired_goals"]
    buffer.actions[:size] = state["actions"]
    buffer.next_index = state["next_index"]
    buffer.size = size
    buffer.rng = state["rng"]


def check_resume_compatible(saved_config: dict[str, Any], config: RunConfig) -> None:
    saved = saved_config_from_dict(saved_config).to_dict()
    current = config.to_dict()
    for section, key in FIELDS_ALLOWED_TO_CHANGE:
        saved[section][key] = current[section][key]
    differences = []
    for section, value in current.items():
        if isinstance(value, dict):
            for key, item in value.items():
                if saved[section].get(key) != item:
                    differences.append(
                        f"{section}.{key}: {saved[section].get(key)!r} saved, {item!r} now"
                    )
        elif saved.get(section) != value:
            differences.append(f"{section}: {saved.get(section)!r} saved, {value!r} now")
    if differences:
        raise ResumeMismatchError(
            "Cannot resume: the config differs from the saved run in fields that must stay "
            "the same (only train.total_steps may change):\n"
            + "\n".join(f"  - {difference}" for difference in differences)
        )


def save_resume_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": RESUME_FORMAT_VERSION, **payload}, temporary)
    os.replace(temporary, path)


def load_resume_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != RESUME_FORMAT_VERSION:
        raise ResumeMismatchError(
            f"{path} has resume format {state.get('format_version')!r}, "
            f"expected {RESUME_FORMAT_VERSION}"
        )
    return state
