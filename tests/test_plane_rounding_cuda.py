"""Independent scalar FP32 operation graph, validation and CUDA stream contracts."""

import math
import struct
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def library():
    torch.ops.load_library(str(ROOT / "build/libtensor_inference_engine_torch.so"))


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def add(lhs, rhs):
    return fp32(lhs + rhs)


def multiply(lhs, rhs):
    return fp32(lhs * rhs)


def norm(values):
    result = 0.0
    for value in values:
        result = add(result, multiply(value, value))
    return result


def reference(values, factor, diagonal, bits, group):
    source = values.cpu()
    directions, diagonals = factor.cpu().tolist(), diagonal.cpu().tolist()
    planes, rows, channels = source.shape
    rank = factor.shape[-1]
    limit = (1 << (bits - 1)) - 1
    maximum = source.reshape(planes, rows, channels // group, group).abs().amax(-1)
    exponents = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    scales = torch.ldexp(torch.ones_like(maximum), exponents).tolist()
    source = source.tolist()
    result = torch.empty(planes, rows, channels)
    selected = torch.empty(rows, channels, dtype=torch.uint8)
    for row in range(rows):
        for channel in range(channels):
            state, nearest_state = [0.] * rank, [0.] * rank
            candidate_diagonal = nearest_diagonal = 0.0
            candidates, nearest_codes = [], []
            for plane in range(planes):
                scale = scales[plane][row][channel // group]
                value = fp32(source[plane][row][channel] / scale)
                lower, upper = max(-limit, min(limit, math.floor(value))), max(-limit, min(limit, math.ceil(value)))
                nearest = max(-limit, min(limit, round(value)))
                low_error = multiply(fp32(lower - value), scale)
                high_error = multiply(fp32(upper - value), scale)
                near_error = multiply(fp32(nearest - value), scale)
                direction, d = directions[channel][plane], diagonals[channel][plane]
                low_state = [add(v, multiply(low_error, u)) for v, u in zip(state, direction)]
                high_state = [add(v, multiply(high_error, u)) for v, u in zip(state, direction)]
                low_cost = add(norm(low_state), multiply(d, multiply(low_error, low_error)))
                high_cost = add(norm(high_state), multiply(d, multiply(high_error, high_error)))
                choose_low = low_cost <= high_cost
                state = low_state if choose_low else high_state
                error = low_error if choose_low else high_error
                candidates.append(lower if choose_low else upper)
                nearest_codes.append(nearest)
                candidate_diagonal = add(candidate_diagonal, multiply(d, multiply(error, error)))
                nearest_diagonal = add(nearest_diagonal, multiply(d, multiply(near_error, near_error)))
                nearest_state = [add(v, multiply(near_error, u)) for v, u in zip(nearest_state, direction)]
            choose = add(candidate_diagonal, norm(state)) < add(nearest_diagonal, norm(nearest_state))
            selected[row, channel] = choose
            for plane in range(planes):
                code = candidates[plane] if choose else nearest_codes[plane]
                result[plane, row, channel] = multiply(code, scales[plane][row][channel // group])
    return result, exponents, selected


@torch.inference_mode()
@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group", [32, 128, 256])
def test_codes_scales_and_decisions_match_independent_scalar_graph(rank, bits, group):
    torch.manual_seed(721 + rank + group + bits)
    source = torch.randn(4, 2, 2 * group, device="cuda")
    source[:, :, ::7] = 0
    factor = torch.randn(2 * group, 4, rank, device="cuda") * 0.2
    diagonal = torch.rand(2 * group, 4, device="cuda")
    actual = torch.ops.tensor_inference_engine.plane_rounding(source, factor, diagonal, bits, group)
    expected = reference(source, factor, diagonal, bits, group)
    for value, target in zip(actual[:3], expected, strict=True):
        assert torch.equal(value.cpu(), target)
    assert actual[3].item() == 0
    assert torch.count_nonzero(actual[0][source == 0]) == 0


@torch.inference_mode()
@pytest.mark.parametrize("planes", [16, 36])
def test_long_plane_sequences_and_completed_nearest_fallback(planes):
    torch.manual_seed(14)
    source = torch.randn(planes, 2, 32, device="cuda")
    factor = torch.randn(32, planes, 4, device="cuda")
    diagonal = torch.rand(32, planes, device="cuda")
    actual = torch.ops.tensor_inference_engine.plane_rounding(source, factor, diagonal)
    expected = reference(source, factor, diagonal, 4, 32)
    for value, target in zip(actual[:3], expected, strict=True):
        assert torch.equal(value.cpu(), target)


@torch.inference_mode()
def test_exact_power_of_two_grids_and_zero_groups():
    source = torch.zeros(4, 1, 64, device="cuda")
    source[:, :, 0] = torch.tensor([127 * 2.**value for value in (-120, -5, 5, 120)], device="cuda")[:, None]
    factor = torch.ones(64, 4, 1, device="cuda")
    diagonal = torch.zeros(64, 4, device="cuda")
    actual = torch.ops.tensor_inference_engine.plane_rounding(source, factor, diagonal, 8)
    assert torch.equal(actual[0], source)
    assert torch.equal(actual[1][:, 0, 0].cpu(), torch.tensor([-120, -5, 5, 120]))
    assert not torch.count_nonzero(actual[1][:, :, 1])
    assert not torch.count_nonzero(actual[2])


@torch.inference_mode()
@pytest.mark.parametrize("which", ["source", "factor", "diagonal"])
def test_invalid_arithmetic_is_rejected_or_exposed_for_deferred_audit(which):
    source = torch.ones(4, 2, 32, device="cuda")
    factor = torch.ones(32, 4, 1, device="cuda")
    diagonal = torch.zeros(32, 4, device="cuda")
    {"source": source, "factor": factor, "diagonal": diagonal}[which].flatten()[0] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        torch.ops.tensor_inference_engine.plane_rounding(source, factor, diagonal)
    actual = torch.ops.tensor_inference_engine.plane_rounding(source, factor, diagonal, 4, 32, False)
    assert actual[3].item() > 0


@torch.inference_mode()
def test_metadata_shapes_empty_rows_and_nondefault_stream():
    empty = torch.empty(4, 0, 32, device="cuda")
    factor = torch.ones(32, 4, 1, device="cuda")
    diagonal = torch.zeros(32, 4, device="cuda")
    result = torch.ops.tensor_inference_engine.plane_rounding(empty, factor, diagonal)
    assert result[0].shape == empty.shape and not result[3].item()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.full((4, 1, 32), 0.4, device="cuda")
        result = torch.ops.tensor_inference_engine.plane_rounding(source, factor, diagonal)
    stream.synchronize()
    torch.testing.assert_close(result[0][:, 0, 0].cpu(), torch.tensor([.375, .4375, .375, .4375]), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="factor"):
        torch.ops.tensor_inference_engine.plane_rounding(source, factor.expand(32, 4, 2).contiguous(), diagonal)
    with pytest.raises(RuntimeError, match="aligned"):
        torch.ops.tensor_inference_engine.plane_rounding(source, factor, diagonal, 4, 256)
    meta = torch.ops.tensor_inference_engine.plane_rounding(empty.to("meta"), factor.to("meta"), diagonal.to("meta"))
    assert meta[1].shape == (4, 0, 1) and meta[2].dtype == torch.uint8
