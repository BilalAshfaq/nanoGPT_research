"""Selected-step diagnostics for Frobenius-normalized SGDM."""

import math

from shared_utils.sgdm_diagnostics import SGDMUpdateDiagnostics


def _scalar_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


class FrobeniusNormalizedSGDMDiagnostics(SGDMUpdateDiagnostics):
    """Merge observed parameter deltas with optimizer-produced scalar evidence."""

    @staticmethod
    def _matrix_scale(matrix_optimizer, parameter, base_learning_rate):
        # A dynamic multiplier does not exist until momentum has been formed.
        return None, None

    def begin_step(self, step, eligible_named_parameters, matrix_optimizer):
        context = super().begin_step(
            step,
            eligible_named_parameters,
            matrix_optimizer,
        )
        if context is not None:
            matrix_optimizer.request_step_diagnostics(
                entry["parameter"] for entry in context["entries"]
            )
        return context

    def _momentum_diagnostics(
        self,
        entry,
        matrix_optimizer,
        momentum_buffer,
        optimizer_step_applied,
    ):
        if not optimizer_step_applied:
            values = super()._momentum_diagnostics(
                entry,
                matrix_optimizer,
                momentum_buffer,
                optimizer_step_applied=False,
            )
            values.update({
                "normalization_denominator": None,
                "retained_rank": min(entry["parameter"].shape),
                "frobenius_epsilon": None,
                "frobenius_shape_factor": None,
                "nominal_normalized_matrix_target_norm": None,
                "epsilon_adjusted_expected_normalized_matrix_norm": None,
                "scheduled_frobenius_learning_rate": entry[
                    "scheduled_learning_rate"
                ],
                "expected_momentum_derived_step_norm": 0.0,
                "normalization_multiplier": 0.0,
                "full_effective_multiplier": 0.0,
                "zero_momentum": None,
                "epsilon_dominated": None,
                "momentum_update_frobenius_cosine": None,
            })
            return values

        captured = matrix_optimizer.pop_step_diagnostics(entry["parameter"])
        values = {
            name: _scalar_value(value) for name, value in captured.items()
        }
        if values["zero_momentum"]:
            values["momentum_update_frobenius_cosine"] = None
        elif not math.isfinite(values["momentum_update_frobenius_cosine"]):
            raise ValueError("nonzero momentum produced an invalid direction cosine")
        values["momentum_update_frobenius_norm"] = values[
            "expected_momentum_derived_step_norm"
        ]
        return values

    def end_step(self, context, matrix_optimizer, optimizer_step_applied=True):
        try:
            result = super().end_step(
                context,
                matrix_optimizer,
                optimizer_step_applied=optimizer_step_applied,
            )
            if result is None:
                return None
            for record in result["matrices"]:
                record["observed_total_parameter_delta_norm"] = record[
                    "applied_update_frobenius_norm"
                ]
                record[
                    "decoupled_weight_decay_update_frobenius_norm"
                ] = record["weight_decay_update_frobenius_norm"]
            return result
        finally:
            matrix_optimizer.clear_step_diagnostics()
