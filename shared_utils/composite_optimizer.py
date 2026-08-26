"""One training-loop-facing interface for disjoint child optimizers."""

import torch


class CompositeOptimizer:
    """Coordinate matrix and auxiliary optimizers over disjoint parameters."""

    STATE_FORMAT_VERSION = 1

    def __init__(self, matrix_optimizer, auxiliary_optimizer):
        self.matrix_optimizer = matrix_optimizer
        self.auxiliary_optimizer = auxiliary_optimizer
        self._validate_disjoint_parameters()

    @property
    def param_groups(self):
        # Resolve dynamically because Optimizer.load_state_dict replaces groups.
        return (
            self.matrix_optimizer.param_groups
            + self.auxiliary_optimizer.param_groups
        )

    def _validate_disjoint_parameters(self):
        matrix_ids = {
            id(parameter)
            for group in self.matrix_optimizer.param_groups
            for parameter in group["params"]
        }
        auxiliary_ids = {
            id(parameter)
            for group in self.auxiliary_optimizer.param_groups
            for parameter in group["params"]
        }
        if matrix_ids & auxiliary_ids:
            raise ValueError("child optimizers must own disjoint parameters")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.matrix_optimizer.step()
        self.auxiliary_optimizer.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.matrix_optimizer.zero_grad(set_to_none=set_to_none)
        self.auxiliary_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "format_version": self.STATE_FORMAT_VERSION,
            "matrix_optimizer": self.matrix_optimizer.state_dict(),
            "auxiliary_optimizer": self.auxiliary_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("format_version") != self.STATE_FORMAT_VERSION:
            raise ValueError("unsupported composite optimizer state format")
        try:
            matrix_state = state_dict["matrix_optimizer"]
            auxiliary_state = state_dict["auxiliary_optimizer"]
        except KeyError as exc:
            raise ValueError(
                f"composite optimizer state is missing {exc.args[0]!r}"
            ) from exc
        self.matrix_optimizer.load_state_dict(matrix_state)
        self.auxiliary_optimizer.load_state_dict(auxiliary_state)
        self._validate_disjoint_parameters()
