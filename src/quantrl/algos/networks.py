from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn


class LinearFactory(Protocol):
    def __call__(
        self, in_features: int, out_features: int, nonnegative_input: bool = ...
    ) -> nn.Module: ...


def plain_linear(in_features: int, out_features: int, nonnegative_input: bool = False) -> nn.Module:
    return nn.Linear(in_features, out_features)


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    linear_factory: LinearFactory = plain_linear,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for index, hidden_dim in enumerate(hidden_sizes):
        layers.append(linear_factory(previous_dim, hidden_dim, nonnegative_input=index > 0))
        layers.append(nn.ReLU())
        previous_dim = hidden_dim
    layers.append(linear_factory(previous_dim, output_dim, nonnegative_input=len(hidden_sizes) > 0))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        linear_factory: LinearFactory = plain_linear,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.network = build_mlp(input_dim, hidden_sizes, 2 * action_dim, linear_factory)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(inputs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(inputs)
        pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        gaussian_log_prob = (
            -0.5 * ((pre_tanh - mean) / log_std.exp()).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)
        tanh_log_det = (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)
        return torch.tanh(pre_tanh), gaussian_log_prob - tanh_log_det

    def deterministic_action(self, inputs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(inputs)
        return torch.tanh(mean)


class TwinQCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        linear_factory: LinearFactory = plain_linear,
    ):
        super().__init__()
        self.q1 = build_mlp(input_dim + action_dim, hidden_sizes, 1, linear_factory)
        self.q2 = build_mlp(input_dim + action_dim, hidden_sizes, 1, linear_factory)

    def forward(
        self, inputs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([inputs, actions], dim=-1)
        return self.q1(joint).squeeze(-1), self.q2(joint).squeeze(-1)
