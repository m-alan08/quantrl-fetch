import numpy as np
import pytest
import torch

from quantrl.algos.sac import SACAgent
from quantrl.checkpoint import deterministic_policy_function, load_checkpoint, load_policy
from quantrl.config import EnvConfig, QuantConfig, SACConfig, load_config
from quantrl.envs.factory import env_dimensions, make_env
from quantrl.envs.normalizer import GoalObservationNormalizer
from quantrl.eval.evaluate import evaluation_seeds
from quantrl.quant.layers import QuantLinear, activation_quantizers, actor_linear_factory
from quantrl.quant.ptq import collect_calibration_inputs, quantize_actor_post_training
from quantrl.quant.qat import initialize_agent_from_checkpoint
from quantrl.sweep import (
    StaleRunError,
    SweepPlan,
    baseline_rows_for_methods,
    evaluate_quantized_run,
    shared_evaluation_seeds,
    train_if_missing,
    training_configs,
)
from quantrl.training import train
from quantrl.viz.video import check_same_environment, compose_side_by_side

SMOKE_OVERRIDES = [
    "device=cpu",
    "sac.hidden_sizes=[32, 32]",
    "sac.batch_size=32",
    "train.total_steps=200",
    "train.random_steps=100",
    "train.update_after=100",
    "train.updates_per_step=0.2",
    "train.eval_interval=200",
    "train.eval_episodes=1",
    "eval.episodes=2",
    "eval.calibration_episodes=2",
    "eval.bootstrap_resamples=50",
    "her.buffer_episodes=10",
]


@pytest.fixture(scope="module")
def fp32_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("runs")
    config = load_config(
        "configs/fetch_reach.json", [*SMOKE_OVERRIDES, f"output_dir={output_dir}", "name=base"]
    )
    train(config, verbose=False)
    return config


def quantized_agent(config, quant_config, sac_overrides=None):
    sac_config = SACConfig(**{**config.sac.__dict__, **(sac_overrides or {})})
    env = make_env(config.env)
    dimensions = env_dimensions(env)
    normalizer = GoalObservationNormalizer(dimensions.observation_dim, dimensions.goal_dim)
    agent = SACAgent(
        normalizer.input_dim,
        dimensions.action_dim,
        sac_config,
        actor_linear_factory=actor_linear_factory(quant_config),
    )
    return agent, normalizer, env


def test_qat_fine_tune_starts_exactly_at_the_ptq_actor(fp32_run):
    checkpoint = load_checkpoint(fp32_run.run_dir / "checkpoint.pt")
    quant_config = QuantConfig(weight_bits=4, activation_bits=4, per_channel_weights=True)
    agent, normalizer, env = quantized_agent(fp32_run, quant_config)
    calibration_count = initialize_agent_from_checkpoint(
        agent, normalizer, checkpoint, fp32_run, env
    )
    assert calibration_count == 100
    assert not any(bool(q.frozen) for q in activation_quantizers(agent.actor))

    loaded = load_policy(fp32_run.run_dir / "checkpoint.pt")
    calibration_inputs = collect_calibration_inputs(
        env,
        deterministic_policy_function(loaded.actor),
        loaded.normalizer,
        evaluation_seeds(fp32_run.eval.calibration_seed_offset, fp32_run.eval.calibration_episodes),
    )
    env.close()
    ptq_actor = quantize_actor_post_training(
        loaded.actor, fp32_run.sac, quant_config, calibration_inputs
    )
    inputs = torch.as_tensor(calibration_inputs[:40])
    agent.actor.eval()
    torch.testing.assert_close(
        agent.actor.deterministic_action(inputs), ptq_actor.deterministic_action(inputs)
    )


def test_configured_learning_rates_override_checkpoint_optimizer_state(fp32_run):
    checkpoint = load_checkpoint(fp32_run.run_dir / "checkpoint.pt")
    agent, normalizer, env = quantized_agent(
        fp32_run, QuantConfig(), {"actor_lr": 1e-4, "alpha_lr": 0.0}
    )
    initialize_agent_from_checkpoint(agent, normalizer, checkpoint, fp32_run, env)
    env.close()
    assert agent.actor_optimizer.param_groups[0]["lr"] == 1e-4
    assert agent.alpha_optimizer.param_groups[0]["lr"] == 0.0
    assert agent.critic_optimizer.param_groups[0]["lr"] == fp32_run.sac.critic_lr


def test_acting_does_not_move_activation_ranges():
    agent = SACAgent(
        13,
        4,
        SACConfig(hidden_sizes=[16]),
        actor_linear_factory=actor_linear_factory(QuantConfig(weight_bits=4, activation_bits=4)),
    )
    first_quantizer = agent.actor.network[0].input_quantizer
    agent.actor(torch.ones(2, 13))
    before = first_quantizer.running_max_abs.clone()
    agent.act(np.full((1, 13), 50.0, dtype=np.float32))
    agent.act(np.full((1, 13), 50.0, dtype=np.float32), deterministic=True)
    assert torch.equal(first_quantizer.running_max_abs, before)
    assert agent.actor.training


def test_qat_from_scratch_checkpoint_round_trips(tmp_path):
    config = load_config(
        "configs/fetch_reach.json",
        [
            *SMOKE_OVERRIDES,
            f"output_dir={tmp_path}",
            "name=scratch",
            "quant.method=qat_scratch",
            "quant.weight_bits=4",
            "quant.activation_bits=4",
        ],
    )
    train(config, verbose=False)
    loaded = load_policy(config.run_dir / "checkpoint.pt")
    layers = [layer for layer in loaded.actor.network if isinstance(layer, QuantLinear)]
    assert len(layers) == 3
    assert all(bool(layer.input_quantizer.initialized) for layer in layers)
    assert loaded.config.quant.method == "qat_scratch"


def test_fine_tune_run_logs_the_starting_point(fp32_run, tmp_path):
    config = load_config(
        fp32_run.run_dir / "config.json",
        [
            f"output_dir={tmp_path}",
            "name=finetune",
            "quant.method=qat_finetune",
            "quant.weight_bits=4",
            "quant.activation_bits=4",
            f"train.init_checkpoint={fp32_run.run_dir / 'checkpoint.pt'}",
            "train.eval_at_start=true",
            "train.random_steps=0",
        ],
    )
    train(config, verbose=False)
    lines = (config.run_dir / "progress.csv").read_text().splitlines()
    assert lines[1].startswith("0,0,")
    assert len(lines) == 3


def test_compose_side_by_side_pads_and_validates():
    left = [np.zeros((4, 5, 3), np.uint8)] * 3
    right = [np.full((4, 5, 3), 255, np.uint8)] * 2
    composed = compose_side_by_side(left, right, divider_width=2)
    assert len(composed) == 3
    assert composed[0].shape == (4, 12, 3)
    assert composed[2][0, -1, 0] == 255
    with pytest.raises(ValueError):
        compose_side_by_side(left, [np.zeros((4, 6, 3), np.uint8)])
    with pytest.raises(ValueError):
        compose_side_by_side([], right)


def test_sweep_plan_builds_named_runs(fp32_run):
    plan = SweepPlan(
        base_config="configs/fetch_reach.json",
        fp32_run_dirs=[str(fp32_run.run_dir)],
        bit_widths=[4, 2],
        methods=["ptq", "qat_finetune", "qat_scratch"],
        output_dir="elsewhere",
        finetune_overrides=["train.total_steps=500"],
    )
    configs = training_configs(plan)
    names = sorted(config.name for config in configs)
    assert names == [
        "base_qat_finetune_w2a2",
        "base_qat_finetune_w4a4",
        "base_qat_scratch_w2a2",
        "base_qat_scratch_w4a4",
    ]
    finetune = next(c for c in configs if c.name == "base_qat_finetune_w4a4")
    assert finetune.train.total_steps == 500
    assert finetune.train.init_checkpoint.endswith("checkpoint.pt")
    assert finetune.quant.per_channel_weights
    scratch = next(c for c in configs if c.name == "base_qat_scratch_w2a2")
    assert scratch.train.init_checkpoint is None
    assert scratch.seed == fp32_run.seed
    assert scratch.output_dir == "elsewhere"


def test_sweep_plan_rejects_bad_inputs():
    with pytest.raises(ValueError):
        SweepPlan("configs/fetch_reach.json", [], [4], ["qat_magic"])
    with pytest.raises(ValueError):
        SweepPlan("configs/fetch_reach.json", [], [32, 4], ["ptq"])
    with pytest.raises(ValueError, match="bit widths"):
        SweepPlan("configs/fetch_reach.json", ["a"], [4, 4], ["ptq"])
    with pytest.raises(ValueError, match="methods"):
        SweepPlan("configs/fetch_reach.json", ["a"], [4], ["ptq", "ptq"])
    with pytest.raises(ValueError, match="fp32 run directories"):
        SweepPlan("configs/fetch_reach.json", ["runs/a", "runs/./a"], [4], ["ptq"])


def test_scratch_runs_never_inherit_an_initial_checkpoint(fp32_run, tmp_path):
    base = tmp_path / "continuation.json"
    base.write_text(
        '{"env": {"env_id": "FetchReach-v4"}, '
        '"train": {"init_checkpoint": "some/checkpoint.pt", "eval_at_start": true}}'
    )
    plan = SweepPlan(str(base), [str(fp32_run.run_dir)], [4], ["qat_scratch"])
    scratch = training_configs(plan)[0]
    assert scratch.train.init_checkpoint is None
    assert not scratch.train.eval_at_start


def test_side_by_side_requires_the_same_environment():
    check_same_environment(EnvConfig(), EnvConfig())
    with pytest.raises(ValueError):
        check_same_environment(EnvConfig(), EnvConfig(env_id="FetchPush-v4"))


def test_fp32_rows_are_repeated_for_each_method():
    rows = [
        {"method": "fp32", "bits": 32, "seed": 0},
        {"method": "ptq", "bits": 4, "seed": 0},
    ]
    expanded = baseline_rows_for_methods(rows, ["ptq", "qat_scratch"])
    assert [(row["method"], row["bits"]) for row in expanded] == [
        ("ptq", 32),
        ("qat_scratch", 32),
        ("ptq", 4),
    ]
    assert [row["is_fp32_reference"] for row in expanded] == [True, True, False]


def test_reusing_a_run_requires_an_identical_config(fp32_run):
    same = load_config(fp32_run.run_dir / "config.json")
    assert train_if_missing(same)["reused"]
    changed = load_config(fp32_run.run_dir / "config.json", ["quant.per_channel_weights=true"])
    with pytest.raises(StaleRunError, match="quant"):
        train_if_missing(changed)


def test_sweep_uses_one_seed_list_for_every_method(fp32_run, tmp_path):
    plan = SweepPlan(
        base_config="configs/fetch_reach.json",
        fp32_run_dirs=[str(fp32_run.run_dir)],
        bit_widths=[4],
        methods=["qat_scratch"],
    )
    seeds = shared_evaluation_seeds(plan)
    assert seeds == list(range(fp32_run.eval.seed_offset, fp32_run.eval.seed_offset + 2))
    assert len(shared_evaluation_seeds(plan, episodes=5)) == 5
    scratch = load_config(
        "configs/fetch_reach.json",
        [
            *SMOKE_OVERRIDES,
            f"output_dir={tmp_path}",
            "name=scratch_offset",
            "quant.method=qat_scratch",
            "quant.weight_bits=4",
            "quant.activation_bits=4",
            "eval.seed_offset=5000",
        ],
    )
    train(scratch, verbose=False)
    row = evaluate_quantized_run(scratch.run_dir, seeds)
    assert row["first_eval_seed"] == seeds[0]
    assert row["episodes"] == len(seeds)
