import json

import pytest

from quantrl.config import RunConfig, load_config, save_config


def test_defaults_round_trip_through_json(tmp_path):
    config = RunConfig()
    path = tmp_path / "config.json"
    save_config(config, path)
    assert load_config(path) == config


def test_overrides_set_nested_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"sac": {"gamma": 0.9}}))
    config = load_config(path, ["sac.batch_size=64", "quant.weight_bits=4", "name=demo"])
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
