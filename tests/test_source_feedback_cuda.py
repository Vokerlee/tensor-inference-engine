"""Independent CPU recurrence and CUDA metadata/streaming contract checks."""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def library():
    torch.ops.load_library(str(ROOT / "build/libtensor_inference_engine_torch.so"))


def reference(values, coefficient, bits, preserve):
    source = values.cpu().float()
    alpha = coefficient.cpu()
    streams, width, channels = source.shape
    limit = (1 << (bits - 1)) - 1
    maximum = source.reshape(streams, width, channels // 32, 32).abs().amax(-1)
    exponent = torch.where(maximum > 0, torch.ceil(torch.log2(maximum / limit)), 0).int()
    scales = torch.ldexp(torch.ones_like(maximum), exponent).repeat_interleave(32, -1)
    carry = torch.zeros(streams, channels)
    output = torch.empty_like(source)
    clipped = introduced = 0
    for column in range(width):
        original = source[:, column]
        incoming = original + alpha * carry
        code = torch.round(incoming / scales[:, column])
        current = code.clamp(-limit, limit) * scales[:, column]
        carry = incoming - current
        if preserve:
            current = torch.where(original == 0, 0, current)
            carry = torch.where(original == 0, 0, carry)
        output[:, column] = current
        clipped += int((code.abs() > limit).sum())
        introduced += int(((original == 0) & (current != 0)).sum())
    return output.bfloat16(), carry, exponent, [clipped, introduced, 0]


@torch.inference_mode()
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("alpha", [-0.5, 0.0, 0.5, "mixed"])
def test_native_recurrence_and_original_group_scales_match_cpu(bits, preserve, alpha):
    torch.manual_seed(943)
    values = torch.randn(6, 17, 96, device="cuda").bfloat16()
    values[:, ::3, ::5] = 0
    coefficient = (torch.linspace(-0.5, 0.5, 96, device="cuda") if alpha == "mixed"
                   else torch.full((96,), alpha, device="cuda"))
    expected = reference(values, coefficient, bits, preserve)
    actual = torch.ops.tensor_inference_engine.source_feedback(values, coefficient, bits, preserve)
    assert torch.equal(actual[0].cpu().view(torch.int16), expected[0].view(torch.int16))
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])
    assert actual[3].cpu().tolist() == expected[3]


@torch.inference_mode()
def test_clipping_zero_introduction_and_row_reset_are_observable():
    values = torch.tensor([6.5, 7, 0, 7.25, 0, 0], device="cuda").bfloat16()
    values = values[None, :, None].expand(2, -1, 32).contiguous()
    alpha = torch.ones(32, device="cuda")
    for preserve in (False, True):
        expected = reference(values, alpha, 4, preserve)
        actual = torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, preserve)
        assert torch.equal(actual[0].cpu(), expected[0])
        assert actual[3].cpu().tolist() == expected[3]
        assert actual[3][0] > 0
        if preserve:
            assert actual[3][1] == 0


@torch.inference_mode()
@pytest.mark.parametrize("shape", [(0, 9, 32), (3, 0, 64)])
def test_empty_stream_or_width_has_defined_zero_terminal_state(shape):
    values = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    alpha = torch.zeros(shape[-1], device="cuda")
    output, terminal, exponents, counts = torch.ops.tensor_inference_engine.source_feedback(values, alpha)
    assert output.shape == values.shape and output.dtype == torch.bfloat16
    assert not torch.count_nonzero(terminal) and not torch.count_nonzero(counts)
    assert exponents.shape == (*shape[:-1], shape[-1] // 32)


@torch.inference_mode()
def test_invalid_values_are_rejected_or_exposed_for_explicit_deferred_audit():
    values = torch.ones(2, 3, 32, device="cuda", dtype=torch.bfloat16)
    alpha = torch.zeros(32, device="cuda")
    values[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        torch.ops.tensor_inference_engine.source_feedback(values, alpha)
    result = torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, False, False)
    assert result[3][2] > 0
    values.fill_(1)
    alpha[2] = 1.25
    with pytest.raises(RuntimeError, match="finite"):
        torch.ops.tensor_inference_engine.source_feedback(values, alpha)


@torch.inference_mode()
def test_deferred_validation_supports_cuda_graph_replay_and_resets_counts():
    values = torch.full((2, 4, 32), 6.5, device="cuda", dtype=torch.bfloat16)
    alpha = torch.ones(32, device="cuda")
    for _ in range(3):
        torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, False, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, False, False)
    values.mul_(2)
    graph.replay()
    first_counts = actual[3].clone()
    expected = reference(values, alpha, 4, False)
    assert torch.equal(actual[0].cpu(), expected[0])
    assert actual[3].cpu().tolist() == expected[3]
    graph.replay()
    assert torch.equal(actual[3], first_counts)


def test_meta_allocation_preserves_static_contract():
    output, terminal, exponent, counts = torch.ops.tensor_inference_engine.source_feedback(
        torch.empty(2, 7, 64, device="meta", dtype=torch.bfloat16), torch.empty(64, device="meta"))
    assert output.shape == (2, 7, 64) and output.dtype == torch.bfloat16
    assert terminal.shape == (2, 64) and terminal.dtype == torch.float32
    assert exponent.shape == (2, 7, 2) and exponent.dtype == torch.int32
    assert counts.shape == (3,) and counts.dtype == torch.int64


@torch.inference_mode()
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("preserve", [False, True])
def test_external_scales_match_independent_recurrence_and_reset_state(bits, preserve):
    torch.manual_seed(401)
    values = torch.randn(3, 9, 64, device="cuda").bfloat16()
    values[:, 3, 1::2] = 0
    alpha = torch.linspace(-0.5, 0.5, 64, device="cuda")
    exponent = torch.randint(-6, 0, (3, 9, 2), device="cuda", dtype=torch.int32)
    source, coefficients = values.cpu().float(), alpha.cpu()
    scales = torch.ldexp(torch.ones_like(exponent.cpu(), dtype=torch.float32), exponent.cpu())
    scales = scales.repeat_interleave(32, -1)
    carry = torch.zeros(3, 64)
    expected = torch.empty_like(source)
    clipped = introduced = 0
    limit = (1 << (bits - 1)) - 1
    for column in range(9):
        incoming = source[:, column] + coefficients * carry
        code = (incoming / scales[:, column]).round()
        current = code.clamp(-limit, limit) * scales[:, column]
        carry = incoming - current
        if preserve:
            current = torch.where(source[:, column] == 0, 0, current)
            carry = torch.where(source[:, column] == 0, 0, carry)
        expected[:, column] = current
        clipped += int((code.abs() > limit).sum())
        introduced += int(((source[:, column] == 0) & (current != 0)).sum())
    actual = torch.ops.tensor_inference_engine.source_feedback(values, alpha, bits, preserve, True, exponent)
    assert torch.equal(actual[0].cpu().view(torch.int16), expected.bfloat16().view(torch.int16))
    assert torch.equal(actual[1].cpu(), carry)
    assert torch.equal(actual[2], exponent)
    assert actual[3].cpu().tolist() == [clipped, introduced, 0]


@torch.inference_mode()
def test_live_and_external_matching_scales_are_bit_identical():
    values = torch.randn(2, 13, 64, device="cuda").bfloat16()
    alpha = torch.ones(64, device="cuda") * 0.5
    live = torch.ops.tensor_inference_engine.source_feedback(values, alpha)
    replay = torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, False, True, live[2])
    for actual, expected in zip(replay, live):
        assert torch.equal(actual, expected)


@torch.inference_mode()
def test_invalid_external_scale_exposes_failure_and_validates_metadata():
    values = torch.ones(1, 3, 32, device="cuda", dtype=torch.bfloat16)
    alpha = torch.zeros(32, device="cuda")
    exponent = torch.full((1, 3, 1), 200, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="finite"):
        torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, False, True, exponent)
    result = torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, False, False, exponent)
    assert int(result[3][2]) > 0
    with pytest.raises(RuntimeError, match="INT32"):
        torch.ops.tensor_inference_engine.source_feedback(values, alpha, 4, False, True, exponent.float())
