"""Packed metadata arithmetic, causal emission and deferred validation."""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def library():
    torch.ops.load_library(str(ROOT / "build/libtensor_inference_engine_torch.so"))


def reference(source, metadata, bits, group):
    source, packed = source.cpu(), metadata.cpu().int()
    raw = packed & 255
    direction = torch.where(raw >= 128, raw - 256, raw)
    diagonal = (packed >> 8) & 255
    planes, rows, channels = source.shape
    limit = (1 << (bits - 1)) - 1
    maximum = source.reshape(planes, rows, -1, group).abs().amax(-1)
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    expanded = exponent.repeat_interleave(group, -1)
    state = torch.zeros(rows, channels)
    output = torch.empty_like(source)
    changed = torch.zeros(rows, channels, dtype=torch.uint8)
    for plane in range(planes):
        original = source[plane]
        scaled = torch.ldexp(original, -expanded[plane])
        lower, upper = scaled.floor().clamp(-limit, limit), scaled.ceil().clamp(-limit, limit)
        nearest = scaled.round().clamp(-limit, limit)
        low_value, high_value = torch.ldexp(lower, expanded[plane]), torch.ldexp(upper, expanded[plane])
        low_error, high_error = low_value - original, high_value - original
        u, h = direction[:, plane], diagonal[:, plane]
        projected = torch.where(u != 0, torch.ldexp(state, 1 - u.abs()), 0)
        projected = torch.where(u < 0, -projected, projected)
        term = torch.where(h != 0, torch.ldexp(low_error + high_error, 1 - h), 0)
        criterion = torch.ldexp(projected, torch.ones_like(u)) + term
        choose_low = (criterion > 0) | ((criterion == 0) & (nearest == lower))
        error = torch.where(choose_low, low_error, high_error)
        update = torch.where(u != 0, torch.ldexp(error, 1 - u.abs()), 0)
        state += torch.where(u < 0, -update, update)
        output[plane] = torch.where(choose_low, low_value, high_value)
        changed |= (torch.where(choose_low, lower, upper) != nearest).byte()
    return output, exponent, changed


@torch.inference_mode()
@pytest.mark.parametrize("planes", [4, 16, 36])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group", [32, 256])
def test_native_matches_cpu_shift_and_add_operation_graph(planes, bits, group):
    torch.manual_seed(713 + planes + group + bits)
    source = torch.randn(planes, 2, 2 * group, device="cuda")
    source[:, :, ::9] = 0
    directions = torch.randint(-5, 6, (2 * group, planes), device="cuda")
    diagonal = directions.abs()
    metadata = ((diagonal << 8) | (directions & 255)).short().contiguous()
    result = torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata, bits, group)
    expected = reference(source, metadata, bits, group)
    for actual, target in zip(result[:3], expected, strict=True):
        assert torch.equal(actual.cpu(), target)
    assert not result[3].item()
    assert not torch.count_nonzero(result[0][source == 0])


@torch.inference_mode()
def test_later_values_cannot_change_earlier_outputs():
    torch.manual_seed(61)
    source = torch.randn(16, 3, 128, device="cuda")
    metadata = torch.full((128, 16), 257, device="cuda", dtype=torch.int16)
    first = torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata, 4, 128)
    source[-1] *= 300
    second = torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata, 4, 128)
    assert torch.equal(first[0][:-1], second[0][:-1])


@torch.inference_mode()
@pytest.mark.parametrize("bad", [128 + 256, -32768, -1, 1, 769])
def test_invalid_metadata_is_rejected_or_exposed(bad):
    source = torch.ones(4, 1, 32, device="cuda")
    metadata = torch.full((32, 4), 257, device="cuda", dtype=torch.int16)
    metadata[0, 0] = bad
    with pytest.raises(RuntimeError, match="valid packed"):
        torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata)
    result = torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata, 4, 32, False)
    assert result[3].item() > 0


@torch.inference_mode()
def test_nearest_ties_zeros_empty_rows_stream_and_meta_contract():
    source = torch.zeros(4, 2, 32, device="cuda")
    source[:, :, 0] = 7
    source[:, :, 1] = 3.5
    metadata = torch.zeros(32, 4, dtype=torch.int16, device="cuda")
    result = torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata)
    assert torch.equal(result[0][:, :, 1], torch.full((4, 2), 4., device="cuda"))
    assert not torch.count_nonzero(result[2])
    empty = source[:, :0].contiguous()
    result = torch.ops.tensor_inference_engine.plane_shift_rounding(empty, metadata)
    assert result[0].shape == empty.shape and not result[3].item()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata)
    stream.synchronize()
    assert not result[3].item()
    meta = torch.ops.tensor_inference_engine.plane_shift_rounding(empty.to("meta"), metadata.to("meta"))
    assert meta[1].shape == (4, 0, 1) and meta[2].dtype == torch.uint8


@torch.inference_mode()
def test_nonfinite_source_does_not_silently_produce_an_accepted_result():
    source = torch.ones(4, 1, 32, device="cuda")
    source[0, 0, 0] = float("nan")
    metadata = torch.zeros(32, 4, dtype=torch.int16, device="cuda")
    with pytest.raises(RuntimeError, match="finite"):
        torch.ops.tensor_inference_engine.plane_shift_rounding(source, metadata)
