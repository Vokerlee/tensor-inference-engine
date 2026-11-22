"""Independent finite-mask and numerical contract checks."""

import importlib.util
from pathlib import Path

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "selection_reference", Path(__file__).resolve().parents[1] / "src/selection_reference.py"
)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("policy", ["diagonal", "gram", "gated"])
def test_matches_independent_six_mask_oracle(bits, policy):
    torch.manual_seed(31)
    weight = torch.randn(32, 19)
    x = torch.randn(13, 32)
    q = selection.quantize_symmetric(x, bits)
    gram = selection.group_gram(weight)
    selected, _ = selection.select_2of4(x, q, gram, policy)
    for group in range(8):
        block = slice(group * 4, (group + 1) * 4)
        dense_geometry = policy == "gram" or (
            policy == "gated" and selection.coherence(gram)[group] > 0.1 / 2.1
        )
        losses = []
        for i, j in selection.PAIRS:
            candidate = torch.zeros_like(q[:, block])
            candidate[:, [i, j]] = q[:, block][:, [i, j]]
            error = candidate.double() - x[:, block].double()
            if dense_geometry:
                loss = (error @ weight[block].double()).square().sum(-1)
            else:
                loss = (error.square() * weight[block].double().square().sum(-1)).sum(-1)
            losses.append(loss)
        actual = selected[:, block].double() - x[:, block].double()
        actual_loss = ((actual @ weight[block].double()).square().sum(-1) if dense_geometry
                       else (actual.square() * weight[block].double().square().sum(-1)).sum(-1))
        torch.testing.assert_close(actual_loss, torch.stack(losses).amin(0), rtol=2e-6, atol=2e-6)


def test_cancellation_example_and_zero_groups():
    weight = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., -1.]])
    x = torch.tensor([[.9, 1., 1.1, 1.2]])
    gram = selection.group_gram(weight)
    actual, index = selection.select_2of4(x, x, gram)
    assert index.item() == 0
    torch.testing.assert_close(((actual - x) @ weight).square().sum(), torch.tensor(.01))
    assert selection.coherence(torch.zeros_like(gram)).item() == 0
    assert torch.isfinite(selection.quantize_symmetric(torch.zeros(2, 32), 4)).all()


def test_rounding_and_contracts():
    x = torch.tensor([[0., .5, 1.5, 7.]])
    torch.testing.assert_close(selection.quantize_symmetric(x, 4, 4), torch.tensor([[0., 0., 2., 7.]]))
    with pytest.raises(ValueError):
        selection.quantize_symmetric(torch.zeros(1, 31), 4)
    with pytest.raises(ValueError):
        selection.select_2of4(torch.zeros(4), torch.zeros(4), torch.zeros(2, 4, 4))


def test_affine_and_fp4_formats():
    positive = torch.tensor([[0., 1., 7., 15.]])
    torch.testing.assert_close(selection.quantize_affine(positive, 4, 4), positive)
    negative = -positive
    torch.testing.assert_close(selection.quantize_affine(negative, 4, 4), negative)
    values = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.]])
    expected = torch.tensor([[0., 1., 1., 2., 2., 4., 4., 6.]])
    torch.testing.assert_close(selection.quantize_e2m1(values, 8), expected)
    torch.testing.assert_close(selection.quantize_e2m1(-values, 8), -expected)
    for function in [selection.quantize_affine, selection.quantize_e2m1]:
        assert torch.count_nonzero(function(torch.zeros(2, 32))) == 0


@pytest.mark.parametrize("shape", [(0, 32), (2, 0), (0,)])
def test_empty_reference_shapes(shape):
    x = torch.empty(shape)
    for function in [lambda value: selection.quantize_symmetric(value, 4),
                     selection.quantize_affine, selection.quantize_e2m1]:
        assert function(x).shape == x.shape
    gram = selection.group_gram(torch.zeros(shape[-1], 19))
    output, pairs = selection.select_2of4(x, x, gram)
    assert output.shape == x.shape
    assert pairs.numel() == 0
    assert selection.group_loss(output - x, gram).numel() == 0
    for function in [selection.quantize_affine, selection.quantize_e2m1]:
        with pytest.raises(ValueError):
            function(torch.tensor(1.))


@pytest.mark.parametrize("both_sparse", [False, True])
@pytest.mark.parametrize("power_two", [False, True])
def test_sparse_residual_representation(both_sparse, power_two):
    torch.manual_seed(47)
    x = torch.randn(5, 64)
    result = selection.sparse_residual_e2m1(x, both_sparse=both_sparse, power_two=power_two)
    assert ((result.coarse.reshape(-1, 4) != 0).sum(-1) <= 2).all()
    if both_sparse:
        assert ((result.residual.reshape(-1, 4) != 0).sum(-1) <= 2).all()
    coarse_error = (x - result.coarse).square().sum()
    assert (x - result.coarse - result.residual).square().sum() < coarse_error
    if power_two:
        assert result.residual_shift.min() >= 0
        assert result.residual_shift.max() <= 8
        scale = x.reshape(5, 2, 32).abs().amax(-1, keepdim=True) / 6
        levels = result.residual.reshape(5, 2, 32) / (scale * 2.0 ** (-result.residual_shift))
        allowed = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        assert torch.cdist(levels.abs().reshape(-1, 1), allowed[:, None]).amin(-1).max() < 1e-5
    else:
        assert result.residual_shift is None
    zero = selection.sparse_residual_e2m1(torch.zeros_like(x), both_sparse=both_sparse,
                                            power_two=power_two)
    assert torch.count_nonzero(zero.coarse + zero.residual) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("residual", [False, True])
def test_fp4_reference_graph_replay_after_table_warmup(residual):
    x = torch.randn(17, 64, device="cuda")
    def evaluate():
        if residual:
            parts = selection.sparse_residual_e2m1(x, both_sparse=True, power_two=True)
            return parts.coarse + parts.residual
        return selection.quantize_e2m1(x)
    expected = evaluate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = evaluate()
    x.mul_(.5)
    graph.replay()
    torch.testing.assert_close(output, evaluate(), rtol=0, atol=0)
