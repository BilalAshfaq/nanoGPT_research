"""Optional update diagnostics for tuned global SGDM."""

import json
import math
import os

import torch


def parse_diagnostic_steps(value):
    if not value.strip():
        return ()
    try:
        steps = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise ValueError(
            "diagnostic_steps must be a comma-separated list of integers"
        ) from exc
    if any(step < 0 for step in steps):
        raise ValueError("diagnostic steps must be non-negative")
    return steps


def parse_diagnostic_matrix_names(value):
    if not value.strip():
        return ()
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(names) != len(set(names)):
        raise ValueError("diagnostic spectral matrix names must be unique")
    return names


def _frobenius_norm(tensor):
    return torch.linalg.vector_norm(tensor.detach().float()).item()


def _spectral_norm(matrix):
    return torch.linalg.matrix_norm(
        matrix.detach().float(), ord=2
    ).item()


class GlobalSGDMDiagnostics:
    """Measure observed global-SGDM matrix updates at selected steps."""

    def __init__(
        self,
        enabled=False,
        steps=(),
        spectral_matrix_names=(),
        epsilon=1e-12,
    ):
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("diagnostics epsilon must be positive")
        self.enabled = enabled
        self.steps = frozenset(steps)
        self.spectral_matrix_names = frozenset(spectral_matrix_names)
        self.epsilon = epsilon

    def should_collect(self, step):
        return self.enabled and step in self.steps

    def validate_parameter_names(self, eligible_named_parameters):
        eligible_names = {item.name for item in eligible_named_parameters}
        unknown_spectral_names = self.spectral_matrix_names - eligible_names
        if unknown_spectral_names:
            raise ValueError(
                "unknown diagnostic spectral matrix names: "
                + ", ".join(sorted(unknown_spectral_names))
            )

    @torch.no_grad()
    def begin_step(self, step, eligible_named_parameters, matrix_optimizer):
        if not self.should_collect(step):
            return None

        self.validate_parameter_names(eligible_named_parameters)

        groups_by_parameter_id = {}
        for group in matrix_optimizer.param_groups:
            for parameter in group["params"]:
                groups_by_parameter_id[id(parameter)] = group

        entries = []
        scheduled_learning_rates = set()
        for item in eligible_named_parameters:
            parameter = item.parameter
            if parameter.grad is None:
                raise ValueError(
                    f"eligible matrix {item.name!r} has no gradient at diagnostic step"
                )
            try:
                group = groups_by_parameter_id[id(parameter)]
            except KeyError as exc:
                raise ValueError(
                    f"eligible matrix {item.name!r} is not owned by GlobalSGDM"
                ) from exc
            learning_rate = group["lr"]
            scheduled_learning_rates.add(learning_rate)
            entries.append(
                {
                    "name": item.name,
                    "parameter": parameter,
                    "parameter_before": parameter.detach().clone(),
                    "gradient_frobenius_norm": _frobenius_norm(parameter.grad),
                    "scheduled_learning_rate": learning_rate,
                    "weight_decay": group["weight_decay"],
                }
            )

        if len(scheduled_learning_rates) != 1:
            raise ValueError(
                "all eligible matrices must use the same global SGDM learning rate"
            )
        return {"step": step, "entries": entries}

    @torch.no_grad()
    def end_step(self, context, matrix_optimizer, optimizer_step_applied=True):
        if context is None:
            return None

        records = []
        for entry in context["entries"]:
            parameter = entry["parameter"]
            before = entry["parameter_before"]
            state = matrix_optimizer.state.get(parameter, {})
            momentum_buffer = state.get("momentum_buffer")
            if momentum_buffer is None:
                momentum_norm = 0.0
                momentum_update_norm = 0.0
            else:
                momentum_norm = _frobenius_norm(momentum_buffer)
                momentum_update_norm = (
                    abs(entry["scheduled_learning_rate"]) * momentum_norm
                    if optimizer_step_applied
                    else 0.0
                )

            applied_update = parameter.detach() - before
            weight_norm = _frobenius_norm(before)
            applied_update_norm = _frobenius_norm(applied_update)
            weight_decay_update_norm = (
                abs(
                    entry["scheduled_learning_rate"]
                    * entry["weight_decay"]
                )
                * weight_norm
                if optimizer_step_applied
                else 0.0
            )
            record = {
                "name": entry["name"],
                "weight_frobenius_norm": weight_norm,
                "gradient_frobenius_norm": entry["gradient_frobenius_norm"],
                "momentum_frobenius_norm": momentum_norm,
                "momentum_update_frobenius_norm": momentum_update_norm,
                "applied_update_frobenius_norm": applied_update_norm,
                "update_to_weight_ratio": (
                    applied_update_norm / (weight_norm + self.epsilon)
                ),
                "scheduled_learning_rate": entry["scheduled_learning_rate"],
                "weight_decay": entry["weight_decay"],
                "weight_decay_update_frobenius_norm": weight_decay_update_norm,
                "optimizer_step_applied": optimizer_step_applied,
                "gradient_stage": "post_clipping_unscaled",
            }
            if entry["name"] in self.spectral_matrix_names:
                record["applied_update_spectral_norm"] = _spectral_norm(
                    applied_update
                )
            records.append(record)

        records.sort(key=lambda record: record["name"])
        return {"step": context["step"], "matrices": records}


def append_diagnostic_record(output_directory, record):
    path = os.path.join(output_directory, "optimizer_diagnostics.jsonl")
    with open(path, "a", encoding="utf-8") as output_file:
        json.dump(record, output_file, sort_keys=True)
        output_file.write("\n")


def initialize_diagnostic_log(output_directory, resume=False):
    """Start a clean log for a new run while preserving resumed records."""

    path = os.path.join(output_directory, "optimizer_diagnostics.jsonl")
    if not resume:
        with open(path, "w", encoding="utf-8"):
            pass
    return path
