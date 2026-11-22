"""CUDA selector checks against direct receiving products and stream contracts."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("selection", ROOT / "src/selection_reference.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@pytest.fixture(scope="module", autouse=True)
def library():
    torch.ops.load_library(str(ROOT / "build/libtensor_inference_engine_torch.so"))


def inputs(rows, channels):
    x = torch.randn(rows, channels, device="cuda")
    q = selection.quantize_symmetric(x, 4, 4)
    weight = torch.randn(channels, 29, device="cuda")
    gram = selection.group_gram(weight)
    upper = torch.triu_indices(4, 4, device="cuda")
    return x, q, weight, gram, gram[:, upper[0], upper[1]].contiguous()


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("shape", [(1, 4), (17, 32), (128, 768), (197, 3072)])
def test_direct_product_optimum(shape, diagonal):
    torch.manual_seed(7)
    x, q, weight, gram, packed = inputs(*shape)
    actual, pairs = torch.ops.tensor_inference_engine.select_2of4(x, q, packed, diagonal)
    expected, _ = selection.select_2of4(x, q, gram, "diagonal" if diagonal else "gram")
    # FP32 FMA can change a near-tied pair; compare the actual objective instead.
    if diagonal:
        norms = weight.square().sum(-1)
        loss = ((actual - x).square() * norms).reshape(shape[0], -1, 4).sum(-1)
        best = ((expected - x).square() * norms).reshape(shape[0], -1, 4).sum(-1)
    else:
        loss = selection.group_loss(actual - x, gram)
        best = selection.group_loss(expected - x, gram)
    torch.testing.assert_close(loss, best, rtol=3e-5, atol=3e-5)
    assert pairs.dtype == torch.uint8
    assert pairs.shape == (shape[0], shape[1] // 4)
    assert torch.all((actual == 0) | (actual == q))


def test_stream_graph_empty_and_invalid():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        x, q, _, gram, packed = inputs(17, 32)
        output, _ = torch.ops.tensor_inference_engine.select_2of4(x, q, packed)
        reference, _ = selection.select_2of4(x, q, gram)
    stream.synchronize()
    torch.testing.assert_close(selection.group_loss(output - x, gram),
                               selection.group_loss(reference - x, gram), rtol=3e-5, atol=3e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = torch.ops.tensor_inference_engine.select_2of4(x, q, packed)
    graph.replay()
    torch.library.opcheck(torch.ops.tensor_inference_engine.select_2of4.default, (x, q, packed))
    empty, indices = torch.ops.tensor_inference_engine.select_2of4(x[:0], q[:0], packed)
    assert empty.numel() == indices.numel() == 0
    with pytest.raises(RuntimeError, match="packed Gram"):
        torch.ops.tensor_inference_engine.select_2of4(x, q, gram)
    with pytest.raises(RuntimeError, match="contiguous FP32"):
        torch.ops.tensor_inference_engine.select_2of4(x.bfloat16(), q, packed)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("policy", [0, 1, 2])
def test_fused_preparation(bits, policy):
    torch.manual_seed(11)
    x = torch.randn(197, 96, device="cuda", dtype=torch.bfloat16)
    x[:3] = 0
    weight = torch.randn(96, 29, device="cuda")
    gram = selection.group_gram(weight)
    upper = torch.triu_indices(4, 4, device="cuda")
    packed = gram[:, upper[0], upper[1]].contiguous()
    quantized = selection.quantize_symmetric(x, bits).bfloat16()
    output, pairs = torch.ops.tensor_inference_engine.prepare_activations(
        x, None if policy == 0 else packed, bits, policy)
    if policy == 0:
        torch.testing.assert_close(output, quantized, rtol=0, atol=0)
        assert torch.all(pairs == 255)
    else:
        expected, _ = selection.select_2of4(x, quantized, gram, "diagonal" if policy == 1 else "gram")
        if policy == 1:
            diagonal = gram.diagonal(dim1=-2, dim2=-1)
            actual_loss = ((output.float() - x.float()).reshape(197, -1, 4).square() * diagonal).sum(-1)
            expected_loss = ((expected - x.float()).reshape(197, -1, 4).square() * diagonal).sum(-1)
        else:
            actual_loss = selection.group_loss(output.float() - x.float(), gram)
            expected_loss = selection.group_loss(expected - x.float(), gram)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=3e-5, atol=3e-5)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        new_x = x.clone()
        streamed, _ = torch.ops.tensor_inference_engine.prepare_activations(
            new_x, None if policy == 0 else packed, bits, policy)
    stream.synchronize()
    torch.testing.assert_close(streamed, output)
    torch.library.opcheck(torch.ops.tensor_inference_engine.prepare_activations.default,
                          (x, None if policy == 0 else packed, bits, policy))
