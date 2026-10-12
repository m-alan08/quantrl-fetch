import math

import numpy as np
import torch

from quantrl.algos.networks import GaussianActor
from quantrl.algos.sac import SACAgent
from quantrl.config import SACConfig


def test_tanh_gaussian_log_prob_matches_torch_distributions():
    torch.manual_seed(0)
    actor = GaussianActor(6, 3, [32])
    inputs = torch.randn(64, 6)
    torch.manual_seed(1)
    actions, log_probs = actor.sample(inputs)
    mean, log_std = actor(inputs)
    torch.manual_seed(1)
    pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
    base = torch.distributions.Normal(mean, log_std.exp())
    transformed = torch.distributions.TransformedDistribution(
        base, [torch.distributions.transforms.TanhTransform(cache_size=1)]
    )
    expected = transformed.log_prob(torch.tanh(pre_tanh)).sum(-1)
    torch.testing.assert_close(log_probs, expected, atol=1e-3, rtol=1e-4)
    assert torch.all(actions.abs() <= 1.0)


def test_update_changes_parameters_and_keeps_q_targets_in_bounds():
    torch.manual_seed(0)
    config = SACConfig(hidden_sizes=[32, 32], batch_size=16)
    agent = SACAgent(8, 2, config)
    rng = np.random.default_rng(0)
    batch = {
        "inputs": rng.normal(size=(16, 8)).astype(np.float32),
        "next_inputs": rng.normal(size=(16, 8)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(16, 2)).astype(np.float32),
        "rewards": -np.ones(16, dtype=np.float32),
    }
    actor_before = [p.clone() for p in agent.actor.parameters()]
    target_before = [p.clone() for p in agent.critic_target.parameters()]
    metrics = agent.update(batch)
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(target_before, agent.critic_target.parameters(), strict=True)
    )
    assert agent.q_target_bounds == (-1.0 / (1.0 - config.gamma), 0.0)


def test_deterministic_action_is_tanh_of_mean():
    torch.manual_seed(0)
    agent = SACAgent(4, 2, SACConfig(hidden_sizes=[16]))
    inputs = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    mean, _ = agent.actor(torch.as_tensor(inputs))
    np.testing.assert_allclose(
        agent.act(inputs, deterministic=True), torch.tanh(mean).detach().numpy()
    )
