import json
import re

import pytest

from quantrl.config import ConfigError, RunConfig, load_config, save_config


def test_defaults_round_trip_through_json(tmp_path):
    config = RunConfig()
    path = tmp_path / "config.json"
    save_config(config, path)
    assert load_config(path) == config


def test_overrides_set_nested_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"sac": {"gamma": 0.9}}))
    config = load_config(
        path, ["sac.batch_size=64", "quant.weight_bits=4", "quant.method=qat_scratch", "name=demo"]
    )
    assert config.sac.gamma == 0.9
    assert config.sac.batch_size == 64
    assert config.quant.weight_bits == 4
    assert config.name == "demo"


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["sac.not_a_field=1"])


@pytest.mark.parametrize("section_value", [None, 3, "fast", [1]])
def test_non_object_sections_are_rejected(tmp_path, section_value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"sac": section_value}))
    with pytest.raises(TypeError):
        load_config(path)


def test_shipped_configs_load():
    assert load_config("configs/fetch_reach.json").env.env_id == "FetchReach-v4"


def test_saved_quantized_configs_without_a_grid_are_read_as_the_old_signed_grid():
    from quantrl.config import saved_config_from_dict

    legacy = RunConfig().to_dict()
    legacy["quant"].pop("relu_activation_grid")
    legacy["quant"].update(weight_bits=4, activation_bits=4, method="qat_scratch")
    assert saved_config_from_dict(legacy).quant.relu_activation_grid == "signed"
    fp32 = RunConfig().to_dict()
    fp32["quant"].pop("relu_activation_grid")
    assert saved_config_from_dict(fp32).quant.relu_activation_grid == "unsigned"
    current = RunConfig().to_dict()
    current["quant"].update(
        weight_bits=4, activation_bits=4, method="qat_scratch", relu_activation_grid="unsigned"
    )
    assert saved_config_from_dict(current).quant.relu_activation_grid == "unsigned"
    assert load_config(None, []).quant.relu_activation_grid == "unsigned"


INVALID_OVERRIDES = [
    ("seed=-1", "seed must be at least 0"),
    ("seed=true", "seed must be an integer"),
    ("device=gpu", "device must be one of"),
    ("torch_threads=0", "torch_threads must be at least 1"),
    ("sac.actor_lr=0", "sac.actor_lr must be > 0"),
    ("sac.critic_lr=-0.001", "sac.critic_lr must be > 0"),
    ("sac.gamma=1.0", "sac.gamma must be < 1"),
    ("sac.tau=0", "sac.tau must be > 0"),
    ("sac.batch_size=0", "sac.batch_size must be at least 1"),
    ("sac.hidden_sizes=[]", "sac.hidden_sizes must be a non-empty list"),
    ("sac.hidden_sizes=[256, 0]", "sac.hidden_sizes must be a non-empty list"),
    ("sac.log_std_min=3", "sac.log_std_min (3) must be below"),
    ("quant.weight_bits=1", "quant.weight_bits must be an integer from 2 to 16"),
    ("quant.activation_bits=4.5", "quant.activation_bits must be an integer from 2 to 16"),
    ("quant.relu_activation_grid=both", "quant.relu_activation_grid must be one of"),
    ("quant.method=magic", "quant.method must be one of"),
    ("quant.activation_ema_momentum=1.5", "quant.activation_ema_momentum must be <= 1"),
    ("her.buffer_episodes=0", "her.buffer_episodes must be at least 1"),
    ("train.total_steps=0", "train.total_steps must be at least 1"),
    ("train.updates_per_step=-1", "train.updates_per_step must be >= 0"),
    ("eval.confidence=1", "eval.confidence must be < 1"),
    ("eval.calibration_seed_offset=1000050", "calibration seeds and evaluation seeds overlap"),
    ("env.obs_clip=0", "env.obs_clip must be > 0"),
]


@pytest.mark.parametrize("override, message", INVALID_OVERRIDES)
def test_invalid_values_fail_fast_with_a_clear_message(override, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(None, [override])


def test_all_problems_are_reported_together():
    with pytest.raises(ConfigError) as error:
        load_config(None, ["sac.actor_lr=0", "sac.batch_size=0", "seed=-3"])
    text = str(error.value)
    assert "sac.actor_lr" in text and "sac.batch_size" in text and "seed" in text


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            ["quant.weight_bits=4"],
            "quant.method 'fp32' needs weight_bits and activation_bits of 32",
        ),
        (["quant.method=qat_scratch"], "needs weight_bits or activation_bits below 32"),
        (
            ["quant.method=qat_finetune", "quant.weight_bits=4"],
            "quant.method 'qat_finetune' needs train.init_checkpoint",
        ),
        (
            ["train.init_checkpoint=a.pt"],
            "trains from scratch, so train.init_checkpoint must be null",
        ),
    ],
)
def test_method_labels_must_match_the_bit_widths_and_initialization(overrides, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(None, overrides)


def test_frozen_temperature_is_allowed():
    assert load_config(None, ["sac.alpha_lr=0"]).sac.alpha_lr == 0


def test_training_refuses_a_missing_initial_checkpoint(tmp_path):
    from quantrl.training import train

    config = load_config(
        None,
        [
            "quant.method=fp32_finetune",
            f"train.init_checkpoint={tmp_path / 'missing.pt'}",
            f"output_dir={tmp_path}",
        ],
    )
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        train(config, verbose=False)
