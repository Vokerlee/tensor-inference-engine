"""Compare native tile decisions with independent reference candidate losses."""

import importlib.util
from itertools import product
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("selection", ROOT / "src/selection_reference.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def library():
    torch.ops.load_library(str(ROOT / "build/libtensor_inference_engine_torch.so"))


def data(channels=96):
    x = torch.randn(197, channels, device="cuda", dtype=torch.bfloat16)
    gram = selection.group_gram(torch.randn(channels, 29, device="cuda"))
    upper = torch.triu_indices(4, 4, device="cuda")
    return x, gram, gram[:, upper[0], upper[1]].contiguous()


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("tolerance", [1e-20, .0001, .01, .1])
@pytest.mark.parametrize("channels", [96, 256])
def test_candidate_decisions(fp4, diagonal, tolerance, channels):
    torch.manual_seed(73)
    x, gram, packed = data(channels)
    output, modes = torch.ops.tensor_inference_engine.adaptive_precision(
        x, packed, tolerance, fp4, diagonal)
    low = (selection.quantize_e2m1(x) if fp4 else selection.quantize_symmetric(x, 4)).bfloat16()
    high = selection.quantize_symmetric(x, 8).bfloat16()
    for block, column in product(range(2), range((channels + 127) // 128)):
        rows = slice(block * 128, (block + 1) * 128)
        columns = slice(column * 128, (column + 1) * 128)
        local_gram = gram[column * 32:(column + 1) * 32]
        original = x[rows, columns].float()

        def loss(error):
            if diagonal:
                return (error.reshape(-1, local_gram.shape[0], 4).square()
                        * local_gram.diagonal(dim1=-2, dim2=-1)).sum()
            return selection.group_loss(error, local_gram).sum()

        denominator = loss(original).clamp_min(1e-30)
        expected = (0 if loss(low[rows, columns].float() - original) / denominator <= tolerance
                    else (1 if loss(high[rows, columns].float() - original) / denominator <= tolerance else 2))
        assert modes[block, column] == expected
        expected_values = [low, high, x][expected][rows, columns]
        torch.testing.assert_close(output[rows, columns], expected_values, rtol=0, atol=0)


def test_stream_graph_contracts_and_empty_inputs():
    x, _, packed = data()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        values = x.clone()
        output, modes = torch.ops.tensor_inference_engine.adaptive_precision(values, packed)
    stream.synchronize()
    expected, expected_modes = torch.ops.tensor_inference_engine.adaptive_precision(x, packed)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(modes, expected_modes)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, _ = torch.ops.tensor_inference_engine.adaptive_precision(x, packed)
    graph.replay()
    torch.testing.assert_close(result, expected)
    torch.library.opcheck(torch.ops.tensor_inference_engine.adaptive_precision.default, (x, packed))
    empty, empty_modes = torch.ops.tensor_inference_engine.adaptive_precision(x[:0], packed)
    assert empty.numel() == empty_modes.numel() == 0
    with pytest.raises(RuntimeError, match="positive finite"):
        torch.ops.tensor_inference_engine.adaptive_precision(x, packed, float("nan"))


@pytest.mark.parametrize("geometry", [2, 3])
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("tolerance", [1e-20, .0001, .01])
def test_separate_error_and_signal_geometry(geometry, fp4, tolerance):
    x, gram, packed = data()
    low = (selection.quantize_e2m1(x) if fp4 else selection.quantize_symmetric(x, 4)).bfloat16()
    high = selection.quantize_symmetric(x, 8).bfloat16()
    output, modes = torch.ops.tensor_inference_engine.adaptive_precision_geometry(
        x, packed, tolerance, fp4, geometry)
    for start in [0, 128]:
        rows = slice(start, start + 128)
        original = x[rows].float()

        def loss(error, full):
            if full:
                return selection.group_loss(error, gram).sum()
            return (error.reshape(-1, 24, 4).square() * gram.diagonal(dim1=-2, dim2=-1)).sum()

        denominator = loss(original, geometry == 3).clamp_min(1e-30)
        expected = (0 if loss(low[rows].float() - original, geometry == 2) / denominator <= tolerance
                    else (1 if loss(high[rows].float() - original, False) / denominator <= tolerance else 2))
        assert modes[start // 128, 0] == expected
        torch.testing.assert_close(output[rows], [low, high, x][expected][rows], rtol=0, atol=0)
    torch.library.opcheck(torch.ops.tensor_inference_engine.adaptive_precision_geometry.default,
                          (x, packed, tolerance, fp4, geometry))
