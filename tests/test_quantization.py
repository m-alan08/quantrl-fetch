import numpy as np
import pytest
import torch

from quantrl.algos.networks import GaussianActor
from quantrl.checkpoint import deterministic_policy_function
from quantrl.config import QuantConfig, SACConfig
from quantrl.quant.layers import (
    ActivationQuantizer,
    QuantLinear,
    activation_quantizers,
    build_quantized_actor,
)
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
    scale_from_max_abs,
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


@pytest.mark.parametrize("bits", [0, 1, 17, 31, 33, 64, 4.5, 8.0, True, "8"])
def test_invalid_bit_widths_are_rejected(bits):
    with pytest.raises(ValueError):
        validate_bits(bits)


@pytest.mark.parametrize("bits", [32.0, 4.0, 33])
def test_fake_quantization_validates_bits_before_the_full_precision_shortcut(bits):
    with pytest.raises(ValueError):
        fake_quantize(torch.ones(3), torch.tensor(1.0), bits)
    with pytest.raises(ValueError):
        fake_quantize_weight(torch.ones(3, 2), bits, per_channel=False)


@pytest.mark.parametrize("momentum", [-0.1, 1.5, float("nan"), float("inf"), "0.9"])
def test_invalid_ema_momentum_is_rejected(momentum):
    with pytest.raises(ValueError):
        ActivationQuantizer(8, ema_momentum=momentum)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_all_zero_tensors_quantize_to_zero_without_nans(dtype):
    weight = torch.zeros(4, 3, dtype=dtype)
    for per_channel in [False, True]:
        quantized = fake_quantize_weight(weight, 8, per_channel)
        assert torch.equal(quantized, weight)
    quantizer = ActivationQuantizer(8).to(dtype)
    quantizer.train()
    assert torch.equal(quantizer(torch.zeros(5, dtype=dtype)), torch.zeros(5, dtype=dtype))


@pytest.mark.parametrize("batch_size", [0, -5])
def test_calibration_rejects_non_positive_batch_size(batch_size):
    actor = build_quantized_actor(13, 4, SAC_CONFIG, QuantConfig(weight_bits=4, activation_bits=4))
    with pytest.raises(ValueError):
        calibrate_activation_ranges(actor, calibration_batch(), batch_size=batch_size)
    assert not any(q.calibrating for q in activation_quantizers(actor))
    assert not any(bool(q.frozen) for q in activation_quantizers(actor))


def test_policy_wrapper_follows_actor_dtype():
    fp32_actor = make_fp32_actor().to(torch.float64)
    quantized_actor = quantize_actor_post_training(
        fp32_actor, SAC_CONFIG, QuantConfig(weight_bits=8, activation_bits=8), calibration_batch()
    )
    actions = deterministic_policy_function(quantized_actor)(calibration_batch(rows=3).numpy())
    assert actions.shape == (3, 4)
    assert actions.dtype == np.float32


def test_invalid_bit_widths_are_rejected_by_layers():
    with pytest.raises(ValueError):
        QuantLinear(4, 2, weight_bits=33)
    with pytest.raises(ValueError):
        ActivationQuantizer(4.5)


def test_integer_quantization_rejects_the_disabled_bit_width():
    with pytest.raises(ValueError):
        quantize_symmetric(torch.ones(3), torch.tensor(1.0), 32)
    assert torch.equal(fake_quantize(torch.ones(3), torch.tensor(1.0), 32), torch.ones(3))


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
    values = torch.tensor([-3.0, -1.2, -1.0, -0.74, -0.2, 0.0, 0.31, 0.8, 1.0, 3.0])
    values.requires_grad_(True)
    scale = torch.tensor(1.0 / 3.0)
    output = fake_quantize(values, scale, bits=3)
    output.sum().backward()
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(values.grad, expected)


@pytest.mark.parametrize("signed", [True, False])
def test_values_that_round_to_the_outermost_levels_still_get_gradient(signed):
    bits = 3
    low, high = (-3, 3) if signed else (0, 7)
    values = torch.tensor([low, low + 0.4, high - 0.4, high], dtype=torch.float32)
    values.requires_grad_(True)
    output = fake_quantize(values, torch.tensor(1.0), bits, signed=signed)
    assert torch.round(output).tolist() == [low, low, high, high]
    output.sum().backward()
    assert values.grad.tolist() == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("signed", [True, False])
def test_values_beyond_half_a_step_past_the_range_get_no_gradient(signed):
    low, high = (-3, 3) if signed else (0, 7)
    values = torch.tensor([low - 0.6, low - 5.0, high + 0.6, high + 5.0], dtype=torch.float32)
    values.requires_grad_(True)
    output = fake_quantize(values, torch.tensor(1.0), 3, signed=signed)
    assert output.tolist() == [low, low, high, high]
    output.sum().backward()
    assert values.grad.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_every_weight_inside_the_range_receives_gradient():
    weight = random_tensor(64, 32).requires_grad_(True)
    for bits in LOW_BIT_WIDTHS:
        for per_channel in (False, True):
            weight.grad = None
            fake_quantize_weight(weight, bits, per_channel).sum().backward()
            assert torch.all(weight.grad == 1.0)


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


@pytest.mark.parametrize("empty", [np.zeros((0, 13), np.float32), torch.zeros(0, 13)])
def test_calibration_rejects_empty_data_without_touching_ranges(empty):
    actor = build_quantized_actor(13, 4, SAC_CONFIG, QuantConfig(weight_bits=4, activation_bits=4))
    actor.train()
    actor(calibration_batch(rows=8))
    quantizer = actor.network[0].input_quantizer
    before = quantizer.running_max_abs.clone()
    with pytest.raises(ValueError):
        calibrate_activation_ranges(actor, empty)
    assert torch.equal(quantizer.running_max_abs, before)
    assert not bool(quantizer.frozen)


def test_ptq_actor_matches_source_dtype():
    fp32_actor = make_fp32_actor().to(torch.float64)
    quantized_actor = quantize_actor_post_training(
        fp32_actor, SAC_CONFIG, QuantConfig(weight_bits=8, activation_bits=8), calibration_batch()
    )
    assert next(quantized_actor.parameters()).dtype == torch.float64


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_ptq_actor_stays_on_source_device():
    fp32_actor = make_fp32_actor().to("mps")
    quantized_actor = quantize_actor_post_training(
        fp32_actor,
        SAC_CONFIG,
        QuantConfig(weight_bits=8, activation_bits=8),
        calibration_batch().to("mps"),
    )
    assert next(quantized_actor.parameters()).device.type == "mps"


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


@pytest.mark.parametrize("bits", [8, 4, 3, 2])
def test_unsigned_grid_uses_all_two_to_the_bits_levels_from_zero(bits):
    values = torch.linspace(-1.0, 10.0, 5000)
    high = 2**bits - 1
    scale = torch.tensor(10.0 / high)
    quantized = fake_quantize(values, scale, bits, signed=False)
    integers = torch.round(quantized / scale)
    assert integers.min() == 0
    assert integers.max() == high
    assert torch.unique(integers).numel() == 2**bits
    assert torch.all(quantized[values < 0] == 0)


def test_unsigned_grid_saturates_outside_its_range():
    values = torch.tensor([-0.5, 0.0, 0.2, 2.9, 4.0])
    quantized = fake_quantize(values, torch.tensor(1.0), bits=2, signed=False)
    assert quantized.tolist() == [0.0, 0.0, 0.0, 3.0, 3.0]


def test_signed_grid_saturates_at_the_largest_level():
    values = torch.tensor([-9.0, -1.4, 1.4, 9.0])
    quantized = fake_quantize(values, torch.tensor(1.0), bits=3)
    assert quantized.tolist() == [-3.0, -1.0, 1.0, 3.0]


def test_unsigned_scale_divides_by_the_unsigned_top_level():
    max_abs = torch.tensor(7.0)
    torch.testing.assert_close(scale_from_max_abs(max_abs, 3, signed=False), torch.tensor(1.0))
    torch.testing.assert_close(scale_from_max_abs(max_abs, 3, signed=True), torch.tensor(7.0 / 3.0))


@pytest.mark.parametrize("grid, hidden_signed", [("unsigned", False), ("signed", True)])
def test_only_inputs_that_come_out_of_a_relu_use_the_unsigned_grid(grid, hidden_signed):
    actor = build_quantized_actor(
        13, 4, SAC_CONFIG, QuantConfig(weight_bits=4, activation_bits=4, relu_activation_grid=grid)
    )
    layers = [layer for layer in actor.network if isinstance(layer, QuantLinear)]
    assert layers[0].input_quantizer.signed
    assert all(layer.input_quantizer.signed == hidden_signed for layer in layers[1:])


def test_unsigned_grid_gives_post_relu_activations_twice_the_levels():
    relu_outputs = torch.relu(random_tensor(4096, scale=2.0))
    counts = {}
    for signed in (True, False):
        quantizer = ActivationQuantizer(3, signed=signed)
        quantizer.train()
        counts[signed] = torch.unique(quantizer(relu_outputs)).numel()
    assert counts[True] == 4
    assert counts[False] == 8


def test_invalid_relu_grid_is_rejected():
    with pytest.raises(ValueError, match="relu_activation_grid"):
        build_quantized_actor(
            13, 4, SAC_CONFIG, QuantConfig(weight_bits=4, relu_activation_grid="unsigned8")
        )


def test_ptq_at_32_bits_is_still_exact_with_the_unsigned_grid():
    fp32_actor = make_fp32_actor()
    quantized_actor = quantize_actor_post_training(
        fp32_actor, SAC_CONFIG, QuantConfig(relu_activation_grid="unsigned"), calibration_batch()
    )
    inputs = calibration_batch(seed=7, rows=64)
    assert torch.equal(
        fp32_actor.deterministic_action(inputs), quantized_actor.deterministic_action(inputs)
    )
