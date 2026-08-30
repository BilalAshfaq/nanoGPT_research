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

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(
                    normalization.normalized_matrix,
                    alpha=-learning_rate,
                )

        return loss
