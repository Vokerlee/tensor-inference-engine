"""Known operation-order contracts, sparse repairs and borrowed input safety."""

import ctypes
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
libm = ctypes.CDLL("libm.so.6")
libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
libm.fmaf.restype = ctypes.c_float


@pytest.fixture(scope="module", autouse=True)
def library():
    torch.ops.load_library(str(ROOT / "build/libtensor_inference_engine_torch.so"))


def cpu_reference(a, b, primitive, chains):
    av, bv = a.cpu().float().tolist(), b.cpu().float().tolist()
    result = torch.empty(a.shape[0], b.shape[1], dtype=torch.float32)
    for row in range(a.shape[0]):
        for column in range(b.shape[1]):
            total = 0.0
            for start in range(0, a.shape[1], primitive):
                partial = [0.0] * chains
                for k in range(start, min(start + primitive, a.shape[1])):
                    lane = (k - start) % chains
                    partial[lane] = libm.fmaf(av[row][k], bv[k][column], partial[lane])
                current = partial[0] if chains == 1 else ctypes.c_float(
                    ctypes.c_float(partial[0] + partial[1]).value
                    + ctypes.c_float(partial[2] + partial[3]).value).value
                total = ctypes.c_float(total + current).value
            result[row, column] = total
    return result


@torch.inference_mode()
@pytest.mark.parametrize("primitive", [1, 7, 128])
@pytest.mark.parametrize("chains", [1, 4])
def test_contract_matches_independent_libm_fma_including_primitive_tail(primitive, chains):
    torch.manual_seed(247)
    a = torch.randn(3, 131, device="cuda").bfloat16()
    b = torch.randn(131, 5, device="cuda").bfloat16()
    output, invalid = torch.ops.tensor_inference_engine.ordered_gemm(a, b, primitive, chains)
    expected = cpu_reference(a, b, primitive, chains)
    assert torch.equal(output.cpu().view(torch.int32), expected.view(torch.int32))
    assert invalid.item() == 0


@torch.inference_mode()
def test_operation_order_difference_is_preserved_instead_of_algebraically_reassociated():
    a = torch.tensor([[1e8, 1, -1e8, 1]], device="cuda").bfloat16()
    b = torch.ones(4, 1, device="cuda", dtype=torch.bfloat16)
    sequential, _ = torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 1)
    split, _ = torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 4)
    assert sequential.item() == 1 and split.item() == 0


@torch.inference_mode()
def test_indexed_repair_changes_only_selected_positions_and_accepts_empty_selection():
    torch.manual_seed(91)
    a = torch.randn(2, 129, device="cuda").bfloat16()
    b = torch.randn(129, 3, device="cuda").bfloat16()
    initial = torch.full((2, 3), -999.0, device="cuda")
    indices = torch.tensor([0, 4], device="cuda")
    reference, _ = torch.ops.tensor_inference_engine.ordered_gemm(a, b)
    output, invalid = torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 1, indices, initial)
    expected = initial.clone().flatten()
    expected[indices] = reference.flatten()[indices]
    assert torch.equal(output.flatten(), expected) and not invalid.item()
    assert (initial == -999).all()
    empty, _ = torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 1, indices[:0], initial)
    assert torch.equal(empty, initial)


@torch.inference_mode()
@pytest.mark.parametrize("indices", [[-1, 0, 6], [1, 1], [2, 1]])
def test_invalid_index_stream_is_rejected_or_audited(indices):
    a = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(4, 3, device="cuda", dtype=torch.bfloat16)
    initial = torch.zeros(2, 3, device="cuda")
    index = torch.tensor(indices, device="cuda")
    with pytest.raises(RuntimeError, match="strictly increasing"):
        torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 1, index, initial)
    _, invalid = torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 1, index, initial, False)
    assert invalid.item() > 0


@pytest.mark.parametrize("shape", [((0, 5), (5, 3)), ((2, 0), (0, 3)), ((2, 3), (3, 0))])
@torch.inference_mode()
def test_empty_matrix_axes_are_defined(shape):
    a, b = [torch.empty(dimensions, device="cuda", dtype=torch.bfloat16) for dimensions in shape]
    result, invalid = torch.ops.tensor_inference_engine.ordered_gemm(a, b)
    assert result.shape == (a.shape[0], b.shape[1]) and not invalid.item()
    if a.shape[1] == 0:
        assert not result.count_nonzero()


def test_meta_contract_rejects_wrong_dtype_and_missing_sparse_initial():
    a = torch.empty(2, 4, device="meta", dtype=torch.bfloat16)
    b = torch.empty(4, 3, device="meta", dtype=torch.bfloat16)
    result, invalid = torch.ops.tensor_inference_engine.ordered_gemm(a, b)
    assert result.shape == (2, 3) and result.dtype == torch.float32 and invalid.shape == (1,)
    with pytest.raises(RuntimeError, match="BF16"):
        torch.ops.tensor_inference_engine.ordered_gemm(a.float(), b)
    with pytest.raises(RuntimeError, match="initial"):
        torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 1, torch.empty(0, device="meta", dtype=torch.int64))


@torch.inference_mode()
@pytest.mark.parametrize("primitive", [7, 127, 128])
@pytest.mark.parametrize("chains", [1, 4])
def test_full_tiling_matches_indexed_contract_across_row_column_and_k_tails(primitive, chains):
    torch.manual_seed(452)
    a = torch.randn(9, 259, device="cuda").bfloat16()
    b = torch.randn(259, 131, device="cuda").bfloat16()
    dense, _ = torch.ops.tensor_inference_engine.ordered_gemm(a, b, primitive, chains)
    indices = torch.arange(dense.numel(), device="cuda")
    indexed, invalid = torch.ops.tensor_inference_engine.ordered_gemm(
        a, b, primitive, chains, indices, torch.zeros_like(dense))
    assert not invalid.item()
    assert torch.equal(dense.view(torch.int32), indexed.view(torch.int32))
