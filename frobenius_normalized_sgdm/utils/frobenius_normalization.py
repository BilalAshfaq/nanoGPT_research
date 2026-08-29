"""Numerically safe Frobenius normalization for SGDM momentum matrices."""

import math
from dataclasses import dataclass
from numbers import Real

import torch


SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@dataclass(frozen=True)
class FrobeniusNormalizationResult:
    """Normalized matrix and detached scalar evidence for its construction."""

    normalized_matrix: torch.Tensor
    raw_frobenius_norm: torch.Tensor
    denominator: torch.Tensor
    retained_rank: int
    epsilon: float
    fixed_shape_factor: float
    nominal_target_norm: torch.Tensor
    epsilon_adjusted_expected_norm: torch.Tensor
    applied_normalization_multiplier: torch.Tensor
    zero_momentum: torch.Tensor
    epsilon_dominated: torch.Tensor


def _positive_finite_scalar(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite, strictly positive number")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite, strictly positive number")
    return scalar


def _validate_momentum_matrix(momentum_matrix):
    if not torch.is_tensor(momentum_matrix):
        raise TypeError("momentum_matrix must be a torch.Tensor")
    if momentum_matrix.layout != torch.strided:
        raise ValueError("momentum_matrix must be a dense strided tensor")
    if momentum_matrix.ndim != 2:
        raise ValueError("momentum_matrix must be two-dimensional")
    if min(momentum_matrix.shape) <= 0:
        raise ValueError("momentum_matrix dimensions must be positive")
    if momentum_matrix.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            "momentum_matrix must use float16, bfloat16, float32, or float64"
        )


@torch.no_grad()
def normalize_frobenius_momentum(
    momentum_matrix,
    epsilon,
    fixed_shape_factor=1.0,
):
    """Apply the exact additive-epsilon Frobenius normalization rule.

    Scalar arithmetic uses float32 for float16, bfloat16, and float32 inputs,
    and float64 for float64 inputs. Scalar evidence remains as detached
    zero-dimensional tensors on the input device, avoiding forced host
    synchronization in a future optimizer integration.
    """

    _validate_momentum_matrix(momentum_matrix)
    epsilon = _positive_finite_scalar("epsilon", epsilon)
    fixed_shape_factor = _positive_finite_scalar(
        "fixed_shape_factor", fixed_shape_factor
    )

    calculation_dtype = (
        torch.float64
        if momentum_matrix.dtype == torch.float64
        else torch.float32
    )
    calculation_limits = torch.finfo(calculation_dtype)
    if not calculation_limits.tiny <= epsilon <= calculation_limits.max:
        raise ValueError(
            "epsilon must be representable as a positive normal value in "
            f"{calculation_dtype}"
        )
    retained_rank = min(momentum_matrix.shape)
    nominal_target = fixed_shape_factor * math.sqrt(retained_rank)
    if (
        not math.isfinite(nominal_target)
        or not calculation_limits.tiny
        <= nominal_target
        <= calculation_limits.max
    ):
        raise ValueError(
            "fixed_shape_factor produces a target norm that is not "
            f"representable in {calculation_dtype}"
        )
    minimum_safe_denominator = nominal_target / calculation_limits.max
    if epsilon < minimum_safe_denominator:
        raise ValueError(
            "epsilon is too small to keep the normalization multiplier "
            f"finite in {calculation_dtype}"
        )

    working_matrix = momentum_matrix.to(dtype=calculation_dtype)
    raw_norm = torch.linalg.vector_norm(working_matrix)
    epsilon_tensor = torch.as_tensor(
        epsilon,
        dtype=calculation_dtype,
        device=momentum_matrix.device,
    )
    denominator = raw_norm + epsilon_tensor
    nominal_target_norm = torch.as_tensor(
        nominal_target,
        dtype=calculation_dtype,
        device=momentum_matrix.device,
    )

    zero_momentum = raw_norm == 0
    epsilon_dominated = raw_norm <= epsilon_tensor
    formula_multiplier = nominal_target_norm / denominator
    applied_multiplier = torch.where(
        zero_momentum,
        torch.zeros_like(formula_multiplier),
        formula_multiplier,
    )
    expected_norm = nominal_target_norm * raw_norm / denominator
    normalized_matrix = (working_matrix * applied_multiplier).to(
        dtype=momentum_matrix.dtype
    )

    return FrobeniusNormalizationResult(
        normalized_matrix=normalized_matrix,
        raw_frobenius_norm=raw_norm.detach(),
        denominator=denominator.detach(),
        retained_rank=retained_rank,
        epsilon=epsilon,
        fixed_shape_factor=fixed_shape_factor,
        nominal_target_norm=nominal_target_norm.detach(),
        epsilon_adjusted_expected_norm=expected_norm.detach(),
        applied_normalization_multiplier=applied_multiplier.detach(),
        zero_momentum=zero_momentum.detach(),
        epsilon_dominated=epsilon_dominated.detach(),
    )
