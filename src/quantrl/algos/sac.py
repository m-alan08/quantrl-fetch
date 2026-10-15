from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quantrl.algos.networks import GaussianActor, LinearFactory, TwinQCritic, plain_linear
from quantrl.config import SACConfig

UPDATE_METRIC_NAMES = (
    "critic_loss",
    "actor_loss",
    "alpha_loss",
    "alpha",
    "entropy",
    "q_mean",
    "critic_grad_norm",
    "actor_grad_norm",
)


class NonFiniteTrainingError(RuntimeError):
    pass


def gradient_norm(parameters) -> torch.Tensor:
    norms = [
        parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None
    ]
    if not norms:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack(norms))


def require_finite(update_index: int, **values: torch.Tensor) -> None:
    bad = [name for name, value in values.items() if not bool(torch.isfinite(value).all())]
    if bad:
        raise NonFiniteTrainingError(
            f"Non-finite values in SAC update {update_index}: {', '.join(bad)}. "
            "Training stopped so the logs do not fill with NaN."
        )


@contextmanager
def evaluation_mode(module: torch.nn.Module) -> Iterator[None]:
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class SACAgent:
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        config: SACConfig,
        device: torch.device | str = "cpu",
        actor_linear_factory: LinearFactory = plain_linear,
        critic_linear_factory: LinearFactory = plain_linear,
    ):
        self.config = config
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.actor = GaussianActor(
            input_dim,
            action_dim,
            config.hidden_sizes,
            config.log_std_min,
            config.log_std_max,
            actor_linear_factory,
        ).to(self.device)
        self.critic = TwinQCritic(
            input_dim, action_dim, config.hidden_sizes, critic_linear_factory
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.log_alpha = torch.tensor(
            float(np.log(config.initial_alpha)), device=self.device, requires_grad=True
        )
        self.target_entropy = (
            config.target_entropy if config.target_entropy is not None else -float(action_dim)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.q_target_bounds = (-1.0 / (1.0 - config.gamma), 0.0)
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, inputs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        tensor_inputs = self._to_tensor(inputs)
        with evaluation_mode(self.actor):
            if deterministic:
                actions = self.actor.deterministic_action(tensor_inputs)
            else:
                actions, _ = self.actor.sample(tensor_inputs)
        return actions.cpu().numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        self.update_count += 1
        inputs = self._to_tensor(batch["inputs"])
        next_inputs = self._to_tensor(batch["next_inputs"])
        actions = self._to_tensor(batch["actions"])
        rewards = self._to_tensor(batch["rewards"])
        require_finite(
            self.update_count,
            inputs=inputs,
            next_inputs=next_inputs,
            actions=actions,
            rewards=rewards,
        )

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_inputs)
            target_q1, target_q2 = self.critic_target(next_inputs, next_actions)
            soft_value = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            q_target = rewards + self.config.gamma * soft_value
            if self.config.clip_q_target:
                q_target = q_target.clamp(*self.q_target_bounds)

        q1, q2 = self.critic(inputs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = gradient_norm(self.critic.parameters())
        require_finite(
            self.update_count,
            q_target=q_target,
            q1=q1,
            q2=q2,
            critic_loss=critic_loss,
            critic_grad_norm=critic_grad_norm,
        )
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        sampled_actions, log_probs = self.actor.sample(inputs)
        policy_q1, policy_q2 = self.critic(inputs, sampled_actions)
        actor_loss = (self.alpha.detach() * log_probs - torch.min(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = gradient_norm(self.actor.parameters())
        require_finite(
            self.update_count,
            log_probs=log_probs,
            actor_loss=actor_loss,
            actor_grad_norm=actor_grad_norm,
        )
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        require_finite(self.update_count, alpha_loss=alpha_loss, log_alpha_grad=self.log_alpha.grad)
        self.alpha_optimizer.step()

        self._polyak_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -log_probs.mean().item(),
            "q_mean": q1.mean().item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),
        }

    @torch.no_grad()
    def _polyak_update(self) -> None:
        tau = self.config.tau
        for parameter, target_parameter in zip(
            self.critic.parameters(), self.critic_target.parameters(), strict=True
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
