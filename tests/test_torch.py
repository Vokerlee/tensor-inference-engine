"""Differential and dispatcher checks for the inference-only CUDA GEMM."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@pytest.fixture(scope="module", autouse=True)
def operator_library():
    library = Path(os.environ.get("TIE_TORCH_LIBRARY", Path(__file__).resolve().parents[1]
                                  / "build/libtensor_inference_engine_torch.so"))
    if not library.is_file():
        pytest.fail(f"Build the Torch adapter first: {library}")
    torch.ops.load_library(str(library))
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(0)
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced


def matrix(rows, columns, transpose=False, padding=0):
    shape = (columns, rows + padding) if transpose else (rows, columns + padding)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return x[:, :rows].t() if transpose else x[:, :columns]


@pytest.mark.parametrize("shape", [(0, 7, 4), (8, 0, 4), (5, 7, 0), (1, 1, 1),
                                   (3, 5, 7), (65, 47, 33), (128, 128, 128),
                                   (128, 256, 1024), (257, 513, 8)])
@pytest.mark.parametrize("transposes", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_gemm(shape, transposes, output_fp32):
    m, n, k = shape
    a = matrix(m, k, transposes[0])
    b = matrix(k, n, transposes[1])
    actual = torch.ops.tensor_inference_engine.gemm(a, b, output_fp32)
    reference = a.float() @ b.float()
    if not output_fp32:
        reference = reference.bfloat16()
    torch.testing.assert_close(actual, reference, rtol=3e-4 if output_fp32 else .008,
                               atol=3e-4 if output_fp32 else .02)


@pytest.mark.parametrize("transpose", [False, True])
def test_pitched_and_offset(transpose):
    a = matrix(19, 33, transpose, padding=3)
    b = matrix(33, 47, not transpose, padding=5)
    pitched = torch.ops.tensor_inference_engine.gemm(a, b, True)
    torch.testing.assert_close(pitched, a.float() @ b.float(), rtol=3e-4, atol=3e-4)
    # Exercise an input storage offset that is only BF16-aligned.
    backing = torch.randn(a.numel() + 1, device="cuda", dtype=torch.bfloat16)
    a = backing[1:].view(a.shape)
    actual = torch.ops.tensor_inference_engine.gemm(a, b, True)
    torch.testing.assert_close(actual, a.float() @ b.float(), rtol=3e-4, atol=3e-4)


def test_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)  # Delay producers so an incorrect default-stream launch races.
        a = matrix(65, 33)
        b = matrix(33, 47)
        actual = torch.ops.tensor_inference_engine.gemm(a, b, True)
        reference = a.float() @ b.float()
    stream.synchronize()
    torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-4)


def test_cuda_graph_and_registration():
    a, b = matrix(32, 64), matrix(64, 48)
    torch.library.opcheck(torch.ops.tensor_inference_engine.gemm.default, (a, b, False))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = torch.ops.tensor_inference_engine.gemm(a, b)
    graph.replay()
    torch.testing.assert_close(output, a @ b, rtol=.008, atol=.02)


def test_linear_dataflow():
    layer = torch.nn.Linear(33, 47, device="cuda", dtype=torch.bfloat16).eval()
    a = matrix(19, 33)
    with torch.inference_mode():
        actual = torch.tanh(torch.ops.tensor_inference_engine.gemm(a, layer.weight.t()) + layer.bias)
        reference = torch.tanh(layer(a))
    torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)


def test_invalid_contracts():
    op = torch.ops.tensor_inference_engine.gemm
    with pytest.raises(RuntimeError, match="BF16"):
        op(matrix(4, 8).float(), matrix(8, 4))
    with pytest.raises(RuntimeError, match="inner dimensions"):
        op(matrix(4, 8), matrix(7, 4))
    with pytest.raises(RuntimeError, match="two-dimensional"):
        op(matrix(4, 8).unsqueeze(0), matrix(8, 4))
    with pytest.raises(RuntimeError, match="row/column-major"):
        op(matrix(4, 16)[:, ::2], matrix(8, 4))
    with pytest.raises(RuntimeError, match="inference-only"):
        op(matrix(4, 8).requires_grad_(), matrix(8, 4))
