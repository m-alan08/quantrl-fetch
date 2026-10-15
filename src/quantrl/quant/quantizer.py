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


def max_unsigned_level(bits: int) -> int:
    return 2 ** validate_integer_bits(bits) - 1


def quantization_range(bits: int, signed: bool) -> tuple[int, int]:
    if signed:
        level = max_quant_level(bits)
        return -level, level
    return 0, max_unsigned_level(bits)


def round_straight_through(values: torch.Tensor) -> torch.Tensor:
    return values + (torch.round(values) - values).detach()


def scale_floor(dtype: torch.dtype) -> float:
    return max(SCALE_FLOOR, torch.finfo(dtype).tiny)


def scale_from_max_abs(max_abs: torch.Tensor, bits: int, signed: bool = True) -> torch.Tensor:
    _, high = quantization_range(bits, signed)
    return (max_abs / high).clamp_min(scale_floor(max_abs.dtype))


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


def fake_quantize(
    values: torch.Tensor, scale: torch.Tensor, bits: int, signed: bool = True
) -> torch.Tensor:
    if is_full_precision(validate_bits(bits)):
        return values
    low, high = quantization_range(bits, signed)
    integers = torch.clamp(round_straight_through(values / scale), low, high)
    return integers * scale


def fake_quantize_weight(weight: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    if is_full_precision(validate_bits(bits)):
        return weight
    return fake_quantize(weight, weight_scale(weight, bits, per_channel), bits)
