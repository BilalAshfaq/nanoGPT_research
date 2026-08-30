"""EMA-momentum SGDM with dynamic Frobenius normalization per matrix."""

import math
from numbers import Real

import torch

from frobenius_normalized_sgdm.utils.frobenius_normalization import (
    normalize_frobenius_momentum,
    validate_frobenius_normalization_inputs,
)
from shared_utils.ema_momentum import update_ema_momentum_buffer


FROBENIUS_NORMALIZATION_VERSION = "additive_epsilon_v1"
FROBENIUS_NORMALIZATION_EQUATION = (
    "q*sqrt(min(d_out,d_in))*M/(frobenius_norm(M)+epsilon)"
)


def _finite_nonnegative(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite, non-negative number")
    return float(value)


class FrobeniusNormalizedSGDM(torch.optim.Optimizer):
    """Normalize each eligible matrix after shared EMA momentum formation.

    The momentum-derived update is
    ``-lr * q * sqrt(rank) * M / (||M||_F + epsilon)``. Decoupled
    weight decay uses only the scheduled base ``lr`` and is applied before the
    normalized momentum update.
    """

    def __init__(
        self,
        params,
        lr,
        momentum,
        epsilon,
        fixed_shape_factor=1.0,
        weight_decay=0.0,
    ):
        lr = _finite_nonnegative("learning rate", lr)
        weight_decay = _finite_nonnegative("weight decay", weight_decay)
        if (
            isinstance(momentum, bool)
            or not isinstance(momentum, Real)
            or not math.isfinite(float(momentum))
            or not 0.0 <= momentum < 1.0
        ):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        momentum = float(momentum)

        parameters = tuple(params)
        if not parameters:
            raise ValueError("FrobeniusNormalizedSGDM requires parameters")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("FrobeniusNormalizedSGDM parameters must be unique")

        validated_epsilon = None
        validated_shape_factor = None
        for parameter in parameters:
            (
                parameter_epsilon,
                parameter_shape_factor,
                _,
                _,
                _,
            ) = validate_frobenius_normalization_inputs(
                parameter,
                epsilon,
                fixed_shape_factor,
            )
            validated_epsilon = parameter_epsilon
            validated_shape_factor = parameter_shape_factor

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "epsilon": validated_epsilon,
            "fixed_shape_factor": validated_shape_factor,
            "weight_decay": weight_decay,
        }
        super().__init__(parameters, defaults)
        self._diagnostic_parameter_ids = frozenset()
        self._diagnostic_scalars = {}

    def request_step_diagnostics(self, parameters):
        """Capture scalar normalization evidence during the next step only."""

        if self._diagnostic_parameter_ids or self._diagnostic_scalars:
            raise RuntimeError("Frobenius step diagnostics are already pending")
        requested = tuple(parameters)
        requested_ids = {id(parameter) for parameter in requested}
        if len(requested_ids) != len(requested):
            raise ValueError("diagnostic parameters must be unique")
        owned_ids = {
            id(parameter)
            for group in self.param_groups
            for parameter in group["params"]
        }
        if not requested_ids <= owned_ids:
            raise ValueError(
                "diagnostic parameters must be owned by FrobeniusNormalizedSGDM"
            )
        self._diagnostic_parameter_ids = frozenset(requested_ids)

    def pop_step_diagnostics(self, parameter):
        """Return and release captured scalar evidence for one parameter."""

        parameter_id = id(parameter)
        try:
            return self._diagnostic_scalars.pop(parameter_id)
        except KeyError as exc:
            raise RuntimeError(
                "Frobenius normalization diagnostics were not captured"
            ) from exc

    def clear_step_diagnostics(self):
        """Release all transient diagnostic requests and scalar evidence."""

        self._diagnostic_parameter_ids = frozenset()
        self._diagnostic_scalars.clear()

    def has_pending_step_diagnostics(self):
        return bool(self._diagnostic_parameter_ids or self._diagnostic_scalars)

    @staticmethod
    def _normalization_diagnostic_scalars(
        momentum_buffer,
        normalization,
        learning_rate,
    ):
        calculation_dtype = normalization.raw_frobenius_norm.dtype
        working_momentum = momentum_buffer.to(dtype=calculation_dtype)
        working_normalized = normalization.normalized_matrix.to(
            dtype=calculation_dtype
        )
        normalized_norm = torch.linalg.vector_norm(working_normalized)
        cosine_denominator = (
            normalization.raw_frobenius_norm * normalized_norm
        )
        cosine = torch.where(
            cosine_denominator == 0,
            torch.full_like(cosine_denominator, float("nan")),
            -torch.sum(working_normalized * working_momentum)
            / cosine_denominator,
        )
        return {
            "momentum_frobenius_norm": normalization.raw_frobenius_norm,
            "normalization_denominator": normalization.denominator,
            "retained_rank": normalization.retained_rank,
            "frobenius_epsilon": normalization.epsilon,
            "frobenius_shape_factor": normalization.fixed_shape_factor,
            "nominal_normalized_matrix_target_norm": (
                normalization.nominal_target_norm
            ),
            "epsilon_adjusted_expected_normalized_matrix_norm": (
                normalization.epsilon_adjusted_expected_norm
            ),
            "scheduled_frobenius_learning_rate": learning_rate,
            "expected_momentum_derived_step_norm": (
                abs(learning_rate)
                * normalization.epsilon_adjusted_expected_norm
            ),
            "normalization_multiplier": (
                normalization.applied_normalization_multiplier
            ),
            "full_effective_multiplier": (
                learning_rate
                * normalization.applied_normalization_multiplier
            ),
            "zero_momentum": normalization.zero_momentum,
            "epsilon_dominated": normalization.epsilon_dominated,
            "momentum_update_frobenius_cosine": cosine.detach(),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            momentum = group["momentum"]
            epsilon = group["epsilon"]
            fixed_shape_factor = group["fixed_shape_factor"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        "FrobeniusNormalizedSGDM does not support sparse gradients"
                    )

                momentum_buffer = update_ema_momentum_buffer(
                    self.state[parameter], parameter, gradient, momentum
                )
                normalization = normalize_frobenius_momentum(
                    momentum_buffer,
                    epsilon=epsilon,
                    fixed_shape_factor=fixed_shape_factor,
                )
                parameter_id = id(parameter)
                if parameter_id in self._diagnostic_parameter_ids:
                    self._diagnostic_scalars[parameter_id] = (
                        self._normalization_diagnostic_scalars(
                            momentum_buffer,
                            normalization,
                            learning_rate,
                        )
                    )

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(
                    normalization.normalized_matrix,
                    alpha=-learning_rate,
                )

        return loss
