import pandas as pd
import pytest

from quantrl.checkpoint import load_policy
from quantrl.config import QuantConfig, load_config
from quantrl.eval.quantized import PTQSetting, evaluate_ptq_settings, ptq_settings
from quantrl.training import train
from quantrl.viz.plots import plot_success_vs_bits

SMOKE_OVERRIDES = [
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


def test_settings_map_targets_to_bit_widths():
    base = QuantConfig(activation_ema_momentum=0.9)
    both = PTQSetting(4, "both", True).quant_config(base)
    weights = PTQSetting(4, "weights", False).quant_config(base)
    activations = PTQSetting(4, "activations", False).quant_config(base)
    assert (both.weight_bits, both.activation_bits, both.per_channel_weights) == (4, 4, True)
    assert (weights.weight_bits, weights.activation_bits) == (4, 32)
    assert (activations.weight_bits, activations.activation_bits) == (32, 4)
    assert both.activation_ema_momentum == 0.9
    with pytest.raises(ValueError):
        PTQSetting(4, "critics", False).quant_config(base)


def test_settings_grid_skips_per_channel_for_activation_only():
    settings = ptq_settings([32, 4], ["both", "activations"], [False, True])
    assert len(settings) == 6
    assert not any(s.target == "activations" and s.per_channel_weights for s in settings)


def test_ptq_evaluation_rows_and_plot(tmp_path):
    config = load_config(
        "configs/fetch_reach.json", [*SMOKE_OVERRIDES, f"output_dir={tmp_path}", "name=ptq"]
    )
    train(config, verbose=False)
    loaded = load_policy(config.run_dir / "checkpoint.pt")
    rows = evaluate_ptq_settings(loaded, ptq_settings([32, 4], ["both", "weights"]))
    frame = pd.DataFrame(rows)
    assert list(frame["method"]) == ["fp32", "ptq", "fp32", "ptq"]
    assert set(frame["calibration_inputs"]) == {100}
    assert frame["episodes"].eq(2).all()
    frame = frame.assign(series=frame["target"])
    output = tmp_path / "bits.png"
    plot_success_vs_bits(frame, output, "smoke", "series")
    assert output.stat().st_size > 0
