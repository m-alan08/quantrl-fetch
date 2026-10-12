from __future__ import annotations

import torch

FULL_PRECISION_BITS = 32
MIN_BITS = 2
MAX_INTEGER_BITS = 16
SCALE_FLOOR = 1e-8


def is_full_precision(bits: int) -> bool:
    return bits == FULL_PRECISION_BITS


def validate_bits(bits: int) -> int:
    is_integer = isinstance(bits, int) and not isinstance(bits, bool)
    if is_integer and (bits == FULL_PRECISION_BITS or MIN_BITS <= bits <= MAX_INTEGER_BITS):
        return bits
    raise ValueError(
        f"bits must be an integer between {MIN_BITS} and {MAX_INTEGER_BITS}, "
        f"or {FULL_PRECISION_BITS} for full precision, got {bits!r}"
    )


def validate_integer_bits(bits: int) -> int:
    if is_full_precision(validate_bits(bits)):
        raise ValueError(
            f"Integer quantization needs a low bit width; {FULL_PRECISION_BITS} means disabled"
        )
    return bits


def max_quant_level(bits: int) -> int:
    return 2 ** (validate_integer_bits(bits) - 1) - 1


def round_straight_through(values: torch.Tensor) -> torch.Tensor:
    return values + (torch.round(values) - values).detach()


def scale_from_max_abs(max_abs: torch.Tensor, bits: int) -> torch.Tensor:
    return max_abs.clamp_min(SCALE_FLOOR) / max_quant_level(bits)


def weight_scale(weight: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    if per_channel:
        max_abs = weight.detach().abs().amax(dim=tuple(range(1, weight.dim())), keepdim=True)
    else:
        max_abs = weight.detach().abs().amax()
    return scale_from_max_abs(max_abs, bits)


def quantize_symmetric(values: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    level = max_quant_level(bits)
    return torch.clamp(torch.round(values / scale), -level, level)


def dequantize(integers: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return integers * scale


def fake_quantize(values: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    if is_full_precision(bits):
        return values
    level = max_quant_level(bits)
    integers = torch.clamp(round_straight_through(values / scale), -level, level)
    return integers * scale


def fake_quantize_weight(weight: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    if is_full_precision(bits):
        return weight
    return fake_quantize(weight, weight_scale(weight, bits, per_channel), bits)
