"""Tensor-only numerical references for grouped activation representations.

These Torch operations define semantics, not a low-bit execution backend or
an estimate of a custom hardware controller's cost.
"""

from itertools import combinations

import torch


PAIRS = tuple(combinations(range(4), 2))


def group_gram(weight: torch.Tensor) -> torch.Tensor:
    """Return four-input-channel Gram blocks for a weight matrix [K, N]."""
    if weight.ndim != 2 or weight.shape[0] % 4:
        raise ValueError("weight must have shape [K, N] with K divisible by four")
    rows = weight.float().reshape(-1, 4, weight.shape[1])
    return rows @ rows.transpose(-1, -2)


def quantize_symmetric(x: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    """Reconstruct symmetric signed INT values; max scaling, ties to even.

    Scales are independent per last-axis group, shared by its quartets and
    fixed before mask selection. All-zero groups use scale one. This is not
    FP4, NVFP4 or a packed integer tensor. Inputs must be finite.
    """
    if bits not in (4, 8) or group_size <= 0 or group_size % 4:
        raise ValueError("use 4/8 bits and a positive group size divisible by four")
    if x.ndim == 0 or x.shape[-1] % group_size:
        raise ValueError("the last dimension must be divisible by group_size")
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    maximum = grouped.abs().amax(dim=-1, keepdim=True)
    limit = (1 << (bits - 1)) - 1
    scale = torch.where(maximum > 0, maximum / limit, 1.0)
    return (torch.round(grouped / scale).clamp(-limit, limit) * scale).reshape(x.shape)


def coherence(gram: torch.Tensor) -> torch.Tensor:
    """Maximum absolute row sum of normalized off-diagonal Gram entries."""
    diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp_min(0)
    denominator = (diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)).sqrt()
    normalized = torch.where(denominator > 0, gram.abs() / denominator.clamp_min(1e-30), 0)
    eye = torch.eye(4, device=gram.device, dtype=torch.bool)
    return normalized.masked_fill(eye, 0).sum(dim=-1).amax(dim=-1)


def select_2of4(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    gram: torch.Tensor,
    policy: str = "gram",
    gate_delta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one of six masks and return reconstruction and pair indices.

    x and reconstructed have the same shape [..., K], gram is [K/4, 4, 4].
    Policies: magnitude, norm (weight-norm times squared activation), diagonal
    (quantization-aware keep/drop gain), gram (signed pair interactions), gated
    (diagonal where the local coherence bound permits it). Lowest pair index
    breaks score ties. Gated reference computes both paths; its GPU runtime
    does not measure a selective hardware implementation.
    """
    if x.shape != reconstructed.shape or x.ndim == 0 or x.shape[-1] % 4:
        raise ValueError("equal input shapes with last dimension divisible by four required")
    if gram.shape != (x.shape[-1] // 4, 4, 4):
        raise ValueError("gram shape must be [K/4, 4, 4]")
    if x.device != reconstructed.device or x.device != gram.device:
        raise ValueError("all tensors must share a device")
    if policy not in {"magnitude", "norm", "diagonal", "gram", "gated"}:
        raise ValueError(f"unsupported policy: {policy}")
    if gate_delta <= 0:
        raise ValueError("gate_delta must be positive")
    original_shape = x.shape
    values = x.float().reshape(-1, gram.shape[0], 4)
    quantized = reconstructed.float().reshape_as(values)
    diagonal = gram.diagonal(dim1=-2, dim2=-1)

    if policy == "magnitude":
        terms = -values.abs()
    elif policy == "norm":
        terms = -diagonal * values.square()
    elif policy == "diagonal":
        terms = diagonal * (quantized.square() - 2 * quantized * values)
    else:
        # Shared scalar subexpressions; no receiving product is evaluated.
        response = torch.einsum("gij,bgj->bgi", gram, -values)
        terms = 2 * quantized * response + quantized.square() * diagonal

    scores = torch.stack([terms[..., i] + terms[..., j] for i, j in PAIRS], dim=-1)
    if policy in {"gram", "gated"}:
        cross = torch.stack(
            [2 * quantized[..., i] * quantized[..., j] * gram[:, i, j] for i, j in PAIRS],
            dim=-1,
        )
        scores = scores + cross
    if policy == "gated":
        simple = diagonal * (quantized.square() - 2 * quantized * values)
        simple_scores = torch.stack([simple[..., i] + simple[..., j] for i, j in PAIRS], dim=-1)
        rich = coherence(gram) > gate_delta / (2 + gate_delta)
        scores = torch.where(rich[None, :, None], scores, simple_scores)

    indices = scores.argmin(dim=-1)
    masks = torch.zeros((6, 4), device=x.device, dtype=torch.bool)
    for pair, (i, j) in enumerate(PAIRS):
        masks[pair, i] = True
        masks[pair, j] = True
    selected = quantized * masks[indices]
    return selected.reshape(original_shape), indices.reshape(*original_shape[:-1], -1)


def group_loss(error: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """Exact per-quartet output error; its sum omits inter-quartet terms."""
    values = error.float().reshape(-1, gram.shape[0], 4)
    return torch.einsum("bgi,gij,bgj->bg", values, gram, values).clamp_min(0)
