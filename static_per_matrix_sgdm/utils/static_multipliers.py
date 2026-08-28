"""Configuration and deterministic resolution of static matrix multipliers."""

import math
from collections.abc import Mapping
from numbers import Real


MATRIX_TYPE_SUFFIXES = {
    "attention_qkv": "attn.c_attn.weight",
    "attention_output": "attn.c_proj.weight",
    "mlp_input": "mlp.c_fc.weight",
    "mlp_output": "mlp.c_proj.weight",
}

# The log-space geometric mean must be this close to log(1) == 0. This avoids
# overflow while enforcing a single, unambiguous scale split between the base
# learning rate and the per-matrix multipliers.
GEOMETRIC_MEAN_LOG_TOLERANCE = 1e-9


def _validate_mapping(name, value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return dict(value)


def _validate_multiplier(label, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite, strictly positive number")
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            f"{label} must be a finite, strictly positive number"
        )
    return multiplier


def _matrix_type_for_name(parameter_name):
    matches = [
        matrix_type
        for matrix_type, suffix in MATRIX_TYPE_SUFFIXES.items()
        if parameter_name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"eligible matrix {parameter_name!r} must match exactly one "
            "documented matrix type"
        )
    return matches[0]


def resolve_static_multipliers(
    eligible_named_parameters,
    *,
    default_multiplier,
    matrix_type_multipliers=None,
    exact_parameter_multipliers=None,
):
    """Resolve and validate one fixed multiplier for every eligible matrix.

    Precedence is exact parameter name, then documented matrix type, then the
    explicit default. The returned dictionaries are sorted and contain plain
    floats so they are deterministic and directly serializable in run and
    checkpoint metadata.
    """

    eligible = tuple(eligible_named_parameters)
    eligible_names = [item.name for item in eligible]
    if not eligible_names:
        raise ValueError("static per-matrix SGDM requires eligible matrices")
    if len(eligible_names) != len(set(eligible_names)):
        raise ValueError("eligible matrix names must be unique")
    if len({id(item.parameter) for item in eligible}) != len(eligible):
        raise ValueError("eligible matrices must not contain duplicate parameters")

    default_value = _validate_multiplier(
        "static default multiplier", default_multiplier
    )
    type_values = _validate_mapping(
        "static matrix-type multipliers", matrix_type_multipliers
    )
    exact_values = _validate_mapping(
        "static exact-parameter multipliers", exact_parameter_multipliers
    )

    unknown_types = sorted(set(type_values) - set(MATRIX_TYPE_SUFFIXES))
    if unknown_types:
        raise ValueError(
            "unknown static matrix types: " + ", ".join(unknown_types)
        )
    unknown_names = sorted(set(exact_values) - set(eligible_names))
    if unknown_names:
        raise ValueError(
            "unknown static exact parameter names: " + ", ".join(unknown_names)
        )

    validated_type_values = {
        matrix_type: _validate_multiplier(
            f"static matrix-type multiplier {matrix_type!r}", value
        )
        for matrix_type, value in sorted(type_values.items())
    }
    validated_exact_values = {
        parameter_name: _validate_multiplier(
            f"static exact-parameter multiplier {parameter_name!r}", value
        )
        for parameter_name, value in sorted(exact_values.items())
    }

    resolved = {}
    for parameter_name in sorted(eligible_names):
        matrix_type = _matrix_type_for_name(parameter_name)
        if parameter_name in validated_exact_values:
            multiplier = validated_exact_values[parameter_name]
        elif matrix_type in validated_type_values:
            multiplier = validated_type_values[matrix_type]
        else:
            multiplier = default_value
        resolved[parameter_name] = multiplier

    if set(resolved) != set(eligible_names):
        missing = sorted(set(eligible_names) - set(resolved))
        raise ValueError(
            "static multiplier mapping is missing eligible matrices: "
            + ", ".join(missing)
        )

    mean_log_multiplier = math.fsum(
        math.log(multiplier) for multiplier in resolved.values()
    ) / len(resolved)
    if abs(mean_log_multiplier) > GEOMETRIC_MEAN_LOG_TOLERANCE:
        geometric_mean = math.exp(mean_log_multiplier)
        raise ValueError(
            "resolved static multipliers must have geometric mean 1.0 "
            f"within log tolerance {GEOMETRIC_MEAN_LOG_TOLERANCE}; "
            f"got {geometric_mean:.17g}"
        )

    return {
        "specification": {
            "default_multiplier": default_value,
            "matrix_type_multipliers": validated_type_values,
            "exact_parameter_multipliers": validated_exact_values,
        },
        "resolved_multipliers": resolved,
        "geometric_mean": math.exp(mean_log_multiplier),
        "geometric_mean_log_tolerance": GEOMETRIC_MEAN_LOG_TOLERANCE,
    }
