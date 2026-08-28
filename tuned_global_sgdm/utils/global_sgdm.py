"""Exact EMA-momentum SGDM used by the tuned global baseline."""

import torch

from shared_utils.ema_momentum import update_ema_momentum_buffer


class GlobalSGDM(torch.optim.Optimizer):
    """SGDM with EMA momentum and decoupled weight decay.

    Momentum follows ``M_t = beta * M_(t-1) + (1 - beta) * G_t``. The
    parameter update is ``W <- (1 - lr * weight_decay) * W - lr * M_t``.
    Nesterov momentum is intentionally unsupported.
    """

    def __init__(self, params, lr, momentum, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight decay: {weight_decay}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

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
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("GlobalSGDM does not support sparse gradients")

                momentum_buffer = update_ema_momentum_buffer(
                    self.state[parameter], parameter, gradient, momentum
                )

                # Decoupled decay acts directly on the pre-update parameter.
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(momentum_buffer, alpha=-learning_rate)

        return loss
