from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from quantrl.algos.networks import GaussianActor, LinearFactory
from quantrl.config import QuantConfig, SACConfig
from quantrl.quant.quantizer import (
    fake_quantize,
    fake_quantize_weight,
    is_full_precision,
    scale_from_max_abs,
    validate_bits,
)


class ActivationQuantizer(nn.Module):
    def __init__(self, bits: int, ema_momentum: float = 0.99):
        super().__init__()
        self.bits = validate_bits(bits)
        if not (isinstance(ema_momentum, (int, float)) and 0.0 <= ema_momentum <= 1.0):
            raise ValueError(f"ema_momentum must be a number in [0, 1], got {ema_momentum!r}")
        self.ema_momentum = float(ema_momentum)
        self.calibrating = False
        self.register_buffer("running_max_abs", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @property
    def enabled(self) -> bool:
        return not is_full_precision(self.bits)

    @property
    def scale(self) -> torch.Tensor:
        return scale_from_max_abs(self.running_max_abs, self.bits)

    def freeze(self) -> None:
        self.frozen.fill_(True)

    def unfreeze(self) -> None:
        self.frozen.fill_(False)

    def reset_range(self) -> None:
        self.running_max_abs.zero_()
        self.initialized.fill_(False)

    def start_calibration(self) -> None:
        self.reset_range()
        self.unfreeze()
        self.calibrating = True

    def finish_calibration(self) -> None:
        self.calibrating = False
        self.freeze()

    @torch.no_grad()
    def _observe(self, values: torch.Tensor) -> None:
        batch_max_abs = values.detach().abs().amax()
        if self.calibrating:
            if self.initialized:
                batch_max_abs = torch.maximum(self.running_max_abs, batch_max_abs)
            self.running_max_abs.copy_(batch_max_abs)
        elif self.initialized:
            self.running_max_abs.mul_(self.ema_momentum).add_(
                batch_max_abs, alpha=1.0 - self.ema_momentum
            )
        else:
            self.running_max_abs.copy_(batch_max_abs)
        self.initialized.fill_(True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        if self.calibrating:
            self._observe(values)
            return values
        if self.training and not self.frozen:
            self._observe(values)
        if not self.initialized:
            return values
        return fake_quantize(values, self.scale, self.bits)


class QuantLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits: int = 32,
        activation_bits: int = 32,
        per_channel_weights: bool = False,
        activation_ema_momentum: float = 0.99,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bits = validate_bits(weight_bits)
        self.per_channel_weights = per_channel_weights
        self.input_quantizer = ActivationQuantizer(activation_bits, activation_ema_momentum)

    def quantized_weight(self) -> torch.Tensor:
        return fake_quantize_weight(self.weight, self.weight_bits, self.per_channel_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(self.input_quantizer(inputs), self.quantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, weight_bits={self.weight_bits}, "
            f"activation_bits={self.input_quantizer.bits}, "
            f"per_channel_weights={self.per_channel_weights}"
        )


def quant_linear_factory(quant_config: QuantConfig) -> LinearFactory:
    def factory(in_features: int, out_features: int) -> nn.Module:
        return QuantLinear(
            in_features,
            out_features,
            weight_bits=quant_config.weight_bits,
            activation_bits=quant_config.activation_bits,
            per_channel_weights=quant_config.per_channel_weights,
            activation_ema_momentum=quant_config.activation_ema_momentum,
        )

    return factory


def build_quantized_actor(
    input_dim: int, action_dim: int, sac_config: SACConfig, quant_config: QuantConfig
) -> GaussianActor:
    return GaussianActor(
        input_dim,
        action_dim,
        sac_config.hidden_sizes,
        sac_config.log_std_min,
        sac_config.log_std_max,
        quant_linear_factory(quant_config),
    )


def activation_quantizers(module: nn.Module) -> Iterator[ActivationQuantizer]:
    for child in module.modules():
        if isinstance(child, ActivationQuantizer):
            yield child
