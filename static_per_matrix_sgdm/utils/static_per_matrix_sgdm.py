"""EMA-momentum SGDM with one fixed scale per eligible matrix."""

import math
from numbers import Real

import torch

from shared_utils.ema_momentum import update_ema_momentum_buffer


class StaticPerMatrixSGDM(torch.optim.Optimizer):
    """Apply fixed per-matrix scaling without changing SGDM momentum.

    For matrix ``l``, the update is
    ``W_l <- (1 - lr * weight_decay) * W_l - lr * a_l * M_l``.
    The fixed multiplier ``a_l`` affects only the momentum-derived update;
    decoupled weight decay always uses the common scheduled base learning rate.
    """

    def __init__(
        self,
        named_parameters,
        resolved_multipliers,
        lr,
        momentum,
        weight_decay=0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight decay: {weight_decay}")

        named_parameters = tuple(named_parameters)
        parameter_names = [item.name for item in named_parameters]
        if not parameter_names:
            raise ValueError("StaticPerMatrixSGDM requires parameters")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("static optimizer parameter names must be unique")
        if len({id(item.parameter) for item in named_parameters}) != len(
            named_parameters
        ):
            raise ValueError("static optimizer parameters must be unique")
        if set(resolved_multipliers) != set(parameter_names):
            missing = sorted(set(parameter_names) - set(resolved_multipliers))
            unknown = sorted(set(resolved_multipliers) - set(parameter_names))
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                "resolved static multiplier mapping does not match parameters ("
                + "; ".join(details)
                + ")"
            )

        multipliers = []
        for name in parameter_names:
            multiplier = resolved_multipliers[name]
            if (
                isinstance(multiplier, bool)
                or not isinstance(multiplier, Real)
                or not math.isfinite(multiplier)
                or multiplier <= 0.0
            ):
                raise ValueError(
                    f"static multiplier for {name!r} must be finite and positive"
                )
            multipliers.append(float(multiplier))

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        parameter_group = {
            "params": [item.parameter for item in named_parameters],
            "parameter_names": parameter_names,
            "static_multipliers": multipliers,
        }
        super().__init__([parameter_group], defaults)

    def multiplier_for(self, parameter):
        """Return the configured fixed multiplier for ``parameter``."""

        for group in self.param_groups:
            for candidate, multiplier in zip(
                group["params"], group["static_multipliers"]
            ):
                if candidate is parameter:
                    return multiplier
        raise KeyError("parameter is not owned by StaticPerMatrixSGDM")

    def effective_learning_rate_for(self, parameter):
        """Return the scheduled base LR multiplied by the matrix scale."""

        for group in self.param_groups:
            for candidate, multiplier in zip(
                group["params"], group["static_multipliers"]
            ):
                if candidate is parameter:
                    return group["lr"] * multiplier
        raise KeyError("parameter is not owned by StaticPerMatrixSGDM")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            parameters = group["params"]
            multipliers = group["static_multipliers"]
            if len(parameters) != len(multipliers):
                raise ValueError(
                    "static multiplier count must match the parameter count"
                )
            for parameter, multiplier in zip(parameters, multipliers):
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        "StaticPerMatrixSGDM does not support sparse gradients"
                    )

                momentum_buffer = update_ema_momentum_buffer(
                    self.state[parameter], parameter, gradient, momentum
                )

                # The per-matrix multiplier intentionally does not affect decay.
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(
                    momentum_buffer,
                    alpha=-learning_rate * multiplier,
                )

        return loss
