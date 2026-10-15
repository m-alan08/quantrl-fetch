from __future__ import annotations

import re
from typing import Any

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
CUDA_INDEX_PATTERN = re.compile(r"^cuda:(\d+)$")


class DeviceUnavailableError(RuntimeError):
    pass


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_devices() -> list[str]:
    devices = ["cpu"]
    if mps_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices


def validate_device_name(name: str) -> str:
    if not isinstance(name, str) or not (name in DEVICE_CHOICES or CUDA_INDEX_PATTERN.match(name)):
        raise ValueError(f"device must be one of {DEVICE_CHOICES} or 'cuda:<index>', got {name!r}")
    return name


def resolve_device(name: str) -> torch.device:
    validate_device_name(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not mps_available():
            raise DeviceUnavailableError(
                f"device 'mps' was requested but is not available; available: {available_devices()}"
            )
        return torch.device("mps")
    if not torch.cuda.is_available():
        raise DeviceUnavailableError(
            f"device {name!r} was requested but CUDA is not available "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
            f"ROCm {getattr(torch.version, 'hip', None)}); available: {available_devices()}"
        )
    match = CUDA_INDEX_PATTERN.match(name)
    if match and int(match.group(1)) >= torch.cuda.device_count():
        raise DeviceUnavailableError(
            f"device {name!r} was requested but only {torch.cuda.device_count()} CUDA "
            f"devices are visible; available: {available_devices()}"
        )
    return torch.device(name)


def describe_device(device: torch.device) -> dict[str, Any]:
    description: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_rocm_build": getattr(torch.version, "hip", None),
        "available_devices": available_devices(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        description["device_name"] = torch.cuda.get_device_name(index)
    return description
