"""Shared EMA momentum-buffer update for SGDM optimizer variants."""

import torch


def update_ema_momentum_buffer(state, parameter, gradient, momentum):
    """Apply ``M_t = beta * M_(t-1) + (1 - beta) * G_t`` in place."""

    momentum_buffer = state.get("momentum_buffer")
    if momentum_buffer is None:
        momentum_buffer = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )
        state["momentum_buffer"] = momentum_buffer
    momentum_buffer.mul_(momentum).add_(gradient, alpha=1.0 - momentum)
    return momentum_buffer
