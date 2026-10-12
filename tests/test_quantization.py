import numpy as np
import pytest
import torch

from quantrl.algos.networks import GaussianActor
from quantrl.config import QuantConfig, SACConfig
from quantrl.quant.layers import ActivationQuantizer, QuantLinear, build_quantized_actor
from quantrl.quant.ptq import (
    calibrate_activation_ranges,
    load_full_precision_weights,
    quantize_actor_post_training,
)
from quantrl.quant.quantizer import (
    fake_quantize,
    fake_quantize_weight,
    max_quant_level,
    quantize_symmetric,
    validate_bits,
    weight_scale,
)

LOW_BIT_WIDTHS = [8, 6, 5, 4, 3, 2]
SAC_CONFIG = SACConfig(hidden_sizes=[64, 64])


def random_tensor(*shape, seed=0, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator) * scale


def make_fp32_actor(seed=0):
    torch.manual_seed(seed)
    return GaussianActor(13, 4, SAC_CONFIG.hidden_sizes)


def calibration_batch(seed=1, rows=512):
    return random_tensor(rows, 13, seed=seed, scale=2.0).clamp(-5, 5)


def test_max_quant_level_is_symmetric_signed_range():
    assert [max_quant_level(bits) for bits in [2, 3, 4, 8]] == [1, 3, 7, 127]


@pytest.mark.parametrize("bits", [0, 1, 17, 31])
def test_invalid_bit_widths_are_rejected(bits):
    with pytest.raises(ValueError):
        validate_bits(bits)


@pytest.mark.parametrize("bits", LOW_BIT_WIDTHS)
def test_quantized_values_land_on_integer_grid(bits):
    values = random_tensor(1000, scale=3.0)
    scale = weight_scale(values, bits, per_channel=False)
    integers = quantize_symmetric(values, scale, bits)
    assert torch.equal(integers, torch.round(integers))
    assert integers.abs().max() <= max_quant_level(bits)
    fake = fake_quantize(values, scale, bits)
    torch.testing.assert_close(fake / scale, torch.round(fake / scale))


@pytest.mark.parametrize("bits", LOW_BIT_WIDTHS)
def test_rounding_error_within_range_is_at_most_half_a_step(bits):
    values = random_tensor(1000, scale=3.0)
    scale = weight_scale(values, bits, per_channel=False)
    error = (fake_quantize(values, scale, bits) - values).abs()
    assert error.max() <= scale / 2 + 1e-6


@pytest.mark.parametrize("bits", LOW_BIT_WIDTHS)
@pytest.mark.parametrize("per_channel", [False, True])
def test_weight_distinct_values_never_exceed_two_to_the_bits(bits, per_channel):
    weight = random_tensor(64, 32)
    quantized = fake_quantize_weight(weight, bits, per_channel)
    rows = quantized if per_channel else quantized.reshape(1, -1)
    for row in rows:
        assert torch.unique(row).numel() <= 2**bits
    integers = quantize_symmetric(weight, weight_scale(weight, bits, per_channel), bits)
    assert torch.unique(integers).numel() <= 2**bits


@pytest.mark.parametrize("bits", LOW_BIT_WIDTHS)
def test_activation_distinct_values_never_exceed_two_to_the_bits(bits):
    quantizer = ActivationQuantizer(bits)
    quantizer.train()
    outputs = quantizer(random_tensor(4096, scale=2.0))
    assert torch.unique(outputs).numel() <= 2**bits


def test_per_channel_scale_has_one_entry_per_output_row():
    weight = random_tensor(8, 5)
    weight[0] *= 100.0
    scale = weight_scale(weight, 4, per_channel=True)
    assert scale.shape == (8, 1)
    per_channel_error = (fake_quantize_weight(weight, 4, True) - weight)[1:].abs().mean()
    per_tensor_error = (fake_quantize_weight(weight, 4, False) - weight)[1:].abs().mean()
    assert per_channel_error < per_tensor_error


def test_straight_through_estimator_passes_gradients_inside_range_and_blocks_outside():
    values = torch.tensor([-3.0, -0.74, -0.2, 0.0, 0.31, 0.8, 3.0], requires_grad=True)
    scale = torch.tensor(1.0 / 3.0)
    output = fake_quantize(values, scale, bits=3)
    output.sum().backward()
    expected = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(values.grad, expected)


def test_straight_through_estimator_reaches_quant_linear_weights():
    layer = QuantLinear(6, 3, weight_bits=3, activation_bits=3)
    layer.train()
    layer(random_tensor(16, 6)).pow(2).sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0


def test_quant_linear_at_32_bits_matches_nn_linear_exactly():
    torch.manual_seed(0)
    reference = torch.nn.Linear(6, 3)
    layer = QuantLinear(6, 3)
    layer.load_state_dict(reference.state_dict(), strict=False)
    inputs = random_tensor(10, 6)
    assert torch.equal(layer(inputs), reference(inputs))


def test_activation_range_follows_exponential_moving_average():
    quantizer = ActivationQuantizer(8, ema_momentum=0.9)
    quantizer.train()
    quantizer(torch.tensor([2.0, -1.0]))
    quantizer(torch.tensor([0.5, -4.0]))
    torch.testing.assert_close(quantizer.running_max_abs, torch.tensor(0.9 * 2.0 + 0.1 * 4.0))


def test_activation_range_is_frozen_at_evaluation_and_after_freeze():
    quantizer = ActivationQuantizer(8)
    quantizer.train()
    quantizer(torch.tensor([1.0]))
    quantizer.eval()
    quantizer(torch.tensor([50.0]))
    assert quantizer.running_max_abs.item() == 1.0
    quantizer.train()
    quantizer.freeze()
    quantizer(torch.tensor([50.0]))
    assert quantizer.running_max_abs.item() == 1.0


def test_calibration_records_max_abs_and_freezes():
    actor = build_quantized_actor(13, 4, SAC_CONFIG, QuantConfig(weight_bits=4, activation_bits=4))
    inputs = calibration_batch()
    calibrate_activation_ranges(actor, inputs, batch_size=100)
    first_quantizer = actor.network[0].input_quantizer
    assert first_quantizer.running_max_abs.item() == pytest.approx(inputs.abs().max().item())
    assert bool(first_quantizer.frozen)
    assert not first_quantizer.calibrating


def test_ptq_at_32_bits_reproduces_fp32_actor_exactly():
    fp32_actor = make_fp32_actor()
    fp32_actor.eval()
    quantized_actor = quantize_actor_post_training(
        fp32_actor, SAC_CONFIG, QuantConfig(), calibration_batch()
    )
    inputs = calibration_batch(seed=7, rows=64)
    fp32_mean, fp32_log_std = fp32_actor(inputs)
    quantized_mean, quantized_log_std = quantized_actor(inputs)
    assert torch.equal(fp32_mean, quantized_mean)
    assert torch.equal(fp32_log_std, quantized_log_std)
    assert torch.equal(
        fp32_actor.deterministic_action(inputs), quantized_actor.deterministic_action(inputs)
    )


def test_ptq_error_shrinks_as_bits_grow():
    fp32_actor = make_fp32_actor()
    inputs = calibration_batch(seed=3, rows=256)
    reference = fp32_actor.deterministic_action(inputs)
    errors = []
    for bits in [2, 4, 8]:
        quantized_actor = quantize_actor_post_training(
            fp32_actor,
            SAC_CONFIG,
            QuantConfig(weight_bits=bits, activation_bits=bits),
            calibration_batch(),
        )
        errors.append((quantized_actor.deterministic_action(inputs) - reference).abs().mean())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.02


def test_ptq_weights_use_only_allowed_levels():
    fp32_actor = make_fp32_actor()
    quantized_actor = quantize_actor_post_training(
        fp32_actor,
        SAC_CONFIG,
        QuantConfig(weight_bits=3, activation_bits=3, per_channel_weights=True),
        calibration_batch(),
    )
    for layer in quantized_actor.network:
        if isinstance(layer, QuantLinear):
            for row in layer.quantized_weight():
                assert torch.unique(row).numel() <= 2**3


def test_loading_fp32_weights_rejects_mismatched_architecture():
    quantized_actor = build_quantized_actor(13, 4, SAC_CONFIG, QuantConfig(weight_bits=4))
    wrong_actor = GaussianActor(13, 4, [64, 64, 64])
    with pytest.raises(KeyError):
        load_full_precision_weights(quantized_actor, wrong_actor.state_dict())


def test_quantized_actor_trains_through_sac_style_loss():
    actor = build_quantized_actor(13, 4, SAC_CONFIG, QuantConfig(weight_bits=4, activation_bits=4))
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
    inputs = calibration_batch(rows=128)
    target = torch.as_tensor(np.tanh(np.linspace(-1, 1, 4)), dtype=torch.float32)
    losses = []
    for _ in range(50):
        actions, _ = actor.sample(inputs)
        loss = (actions - target).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
