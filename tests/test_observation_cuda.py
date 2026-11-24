"""Directed norm enclosures and explicit BF16 rounding-cell boundaries."""

from fractions import Fraction
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def library():
    torch.ops.load_library(str(ROOT / "build/libtensor_inference_engine_torch.so"))


@torch.inference_mode()
@pytest.mark.parametrize("columns", [False, True])
def test_norms_enclose_exact_positive_sums_and_squares(columns):
    torch.manual_seed(280)
    values = (torch.randn(7, 131) * 2).bfloat16().cuda()
    actual = torch.ops.tensor_inference_engine.bf16_norms(values, columns).cpu()
    rows = values.cpu().float().t() if columns else values.cpu().float()
    for row, bound in zip(rows, actual):
        exact = [abs(Fraction(float(value))) for value in row]
        assert Fraction(float(bound[0])) >= sum(exact)
        assert Fraction(float(bound[1])) == max(exact)
        assert Fraction(float(bound[2])) ** 2 >= sum(value * value for value in exact)
        assert bound[3] == 0


@torch.inference_mode()
def test_zero_subnormal_nonfinite_and_empty_norm_inputs_are_defined():
    values = torch.tensor([[0, 0], [2.0 ** -133, 1], [float('inf'), float('nan')]], device="cuda").bfloat16()
    norms = torch.ops.tensor_inference_engine.bf16_norms(values)
    assert not norms[0].count_nonzero()
    assert norms[1:, 3].tolist() == [1, 1]
    for shape in ((0, 3), (3, 0)):
        source = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        for columns in (False, True):
            actual = torch.ops.tensor_inference_engine.bf16_norms(source, columns)
            assert not actual.count_nonzero()


@torch.inference_mode()
def test_observation_strictly_excludes_ties_tiny_extreme_and_invalid_domains():
    values = torch.tensor([[1, 1 + 2.0 ** -8, -1, -1 - 2.0 ** -8, 2.0 ** -133, float('inf')]], device="cuda")
    a = torch.tensor([[1, 1, 1, 0]], device="cuda", dtype=torch.float32)
    b = a.expand(6, -1).contiguous()
    observed, accepted, error = torch.ops.tensor_inference_engine.bf16_observe(values, a, b, 0.0, 0.0)
    assert accepted.tolist() == [[True, False, True, False, False, False]]
    assert torch.equal(observed.view(torch.int16), values.bfloat16().view(torch.int16))
    assert not error.count_nonzero()
    b[:, 3] = 1
    _, accepted, _ = torch.ops.tensor_inference_engine.bf16_observe(values, a, b, 0.0, 0.0)
    assert not accepted.any()


@torch.inference_mode()
def test_zero_product_observation_has_exact_positive_zero_and_bias():
    product = torch.tensor([[-0.0, 99.0, -10.0]], device="cuda")
    a = torch.zeros(1, 4, device="cuda")
    b = torch.ones(3, 4, device="cuda")
    b[:, 3] = 0
    observed, accepted, _ = torch.ops.tensor_inference_engine.bf16_observe(product, a, b, 1e-4, 1e-40)
    assert accepted.all() and not observed.view(torch.int16).count_nonzero()
    bias = torch.tensor([1, -2, 3], device="cuda", dtype=torch.float32)
    observed, accepted, _ = torch.ops.tensor_inference_engine.bf16_observe(product, a, b, 1e-4, 1e-40, bias)
    assert accepted.all() and torch.equal(observed, bias[None].bfloat16())


def test_meta_shapes_and_bad_coefficients_are_rejected():
    source = torch.empty(3, 7, device="meta", dtype=torch.bfloat16)
    norms = torch.ops.tensor_inference_engine.bf16_norms(source)
    assert norms.shape == (3, 4)
    product = torch.empty(3, 5, device="meta", dtype=torch.float32)
    weights = torch.empty(5, 4, device="meta", dtype=torch.float32)
    observed, accepted, error = torch.ops.tensor_inference_engine.bf16_observe(product, norms, weights, 1e-4, 1e-40)
    assert observed.dtype == torch.bfloat16 and accepted.dtype == torch.bool and error.shape == product.shape
    with pytest.raises(RuntimeError, match="nonnegative"):
        torch.ops.tensor_inference_engine.bf16_observe(product, norms, weights, -1.0, 0.0)
