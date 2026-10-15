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


def make_batch(rows=16, dim=8, action_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "inputs": rng.normal(size=(rows, dim)).astype(np.float32),
        "next_inputs": rng.normal(size=(rows, dim)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(rows, action_dim)).astype(np.float32),
        "rewards": -np.ones(rows, dtype=np.float32),
    }


def test_every_network_output_has_the_expected_shape_and_dtype():
    torch.manual_seed(0)
    agent = SACAgent(8, 2, SACConfig(hidden_sizes=[32, 32]))
    inputs = torch.randn(5, 8)
    mean, log_std = agent.actor(inputs)
    actions, log_probs = agent.actor.sample(inputs)
    q1, q2 = agent.critic(inputs, actions)
    target_q1, target_q2 = agent.critic_target(inputs, actions)
    for tensor, shape in [
        (mean, (5, 2)),
        (log_std, (5, 2)),
        (actions, (5, 2)),
        (log_probs, (5,)),
        (q1, (5,)),
        (q2, (5,)),
        (target_q1, (5,)),
        (target_q2, (5,)),
        (agent.actor.deterministic_action(inputs), (5, 2)),
    ]:
        assert tuple(tensor.shape) == shape
        assert tensor.dtype == torch.float32
    assert torch.all(log_std >= agent.config.log_std_min)
    assert torch.all(log_std <= agent.config.log_std_max)
    numpy_actions = agent.act(inputs.numpy())
    assert numpy_actions.shape == (5, 2) and numpy_actions.dtype == np.float32


def test_target_critics_follow_the_polyak_formula_exactly():
    torch.manual_seed(0)
    agent = SACAgent(8, 2, SACConfig(hidden_sizes=[16], tau=0.1))
    with torch.no_grad():
        for parameter in agent.critic_target.parameters():
            parameter.normal_()
    target_before = [p.clone() for p in agent.critic_target.parameters()]
    agent.update(make_batch())
    online_after = list(agent.critic.parameters())
    for before, online, after in zip(
        target_before, online_after, agent.critic_target.parameters(), strict=True
    ):
        torch.testing.assert_close(after, 0.9 * before + 0.1 * online)


def test_temperature_rises_when_entropy_is_below_target_and_falls_when_above():
    for target_entropy, should_rise in [(100.0, True), (-100.0, False)]:
        torch.manual_seed(0)
        agent = SACAgent(8, 2, SACConfig(hidden_sizes=[16], target_entropy=target_entropy))
        alpha_before = agent.alpha.item()
        agent.update(make_batch())
        assert (agent.alpha.item() > alpha_before) == should_rise


def test_critic_and_actor_parameters_change_after_one_update():
    torch.manual_seed(0)
    agent = SACAgent(8, 2, SACConfig(hidden_sizes=[16]))
    critic_before = [p.clone() for p in agent.critic.parameters()]
    actor_before = [p.clone() for p in agent.actor.parameters()]
    metrics = agent.update(make_batch())
    assert all(
        not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters(), strict=True)
    )
    assert all(np.isfinite(value) for value in metrics.values())


def test_critic_target_clip_range_is_the_discounted_return_range():
    torch.manual_seed(0)
    config = SACConfig(hidden_sizes=[16], gamma=0.5)
    agent = SACAgent(8, 2, config)
    assert agent.q_target_bounds == (-2.0, 0.0)
