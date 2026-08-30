"""Optimizer selection shared by the nanoGPT training entry point."""

import inspect

import torch

from shared_utils.composite_optimizer import CompositeOptimizer
from shared_utils.parameter_partition import (
    build_parameter_group_audit,
    partition_optimizer_parameters,
)
from frobenius_normalized_sgdm.utils.frobenius_normalized_sgdm import (
    FROBENIUS_NORMALIZATION_EQUATION,
    FROBENIUS_NORMALIZATION_VERSION,
    FrobeniusNormalizedSGDM,
)
from static_per_matrix_sgdm.utils.static_multipliers import (
    resolve_static_multipliers,
)
from static_per_matrix_sgdm.utils.static_per_matrix_sgdm import (
    StaticPerMatrixSGDM,
)
from tuned_global_sgdm.utils.global_sgdm import GlobalSGDM


SUPPORTED_OPTIMIZERS = (
    "adamw",
    "global_sgdm",
    "static_per_matrix_sgdm",
    "frobenius_normalized_sgdm",
)
GLOBAL_SGDM_MOMENTUM_CONVENTION = "ema"
GLOBAL_SGDM_WEIGHT_DECAY_MODE = "decoupled"
AUXILIARY_ADAMW_WEIGHT_DECAY_MODE = "adamw_decoupled"


def validate_optimizer_name(optimizer_name):
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        choices = ", ".join(SUPPORTED_OPTIMIZERS)
        raise ValueError(
            f"unknown optimizer_name {optimizer_name!r}; expected one of: {choices}"
        )


def _configure_auxiliary_adamw(
    partition,
    learning_rate,
    weight_decay,
    betas,
    device_type,
):
    decay_parameters = [
        item.parameter
        for item in partition.auxiliary
        if item.parameter.ndim >= 2
    ]
    no_decay_parameters = [
        item.parameter
        for item in partition.auxiliary
        if item.parameter.ndim < 2
    ]
    optimizer_groups = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )
    for group in optimizer.param_groups:
        group["optimizer_role"] = "auxiliary"
        group["base_lr"] = learning_rate
    print(f"using fused auxiliary AdamW: {use_fused}")
    return optimizer


def configure_optimizer(
    model,
    optimizer_name,
    device_type,
    adamw_learning_rate,
    adamw_weight_decay,
    adamw_betas,
    matrix_learning_rate,
    matrix_momentum,
    matrix_weight_decay,
    auxiliary_learning_rate,
    auxiliary_weight_decay,
    auxiliary_betas,
    matrix_momentum_convention=GLOBAL_SGDM_MOMENTUM_CONVENTION,
    matrix_weight_decay_mode=GLOBAL_SGDM_WEIGHT_DECAY_MODE,
    auxiliary_weight_decay_mode=AUXILIARY_ADAMW_WEIGHT_DECAY_MODE,
    matrix_nesterov=False,
    static_default_multiplier=1.0,
    static_matrix_type_multipliers=None,
    static_exact_parameter_multipliers=None,
    frobenius_learning_rate=None,
    frobenius_epsilon=1e-12,
    frobenius_shape_factor=1.0,
):
    """Construct the selected optimizer and its optional partition audit."""

    validate_optimizer_name(optimizer_name)
    if optimizer_name == "adamw":
        optimizer = model.configure_optimizers(
            adamw_weight_decay,
            adamw_learning_rate,
            adamw_betas,
            device_type,
        )
        partition = partition_optimizer_parameters(model)
        audit = build_parameter_group_audit(
            partition,
            eligible_optimizer="adamw",
            eligible_weight_decay=adamw_weight_decay,
            auxiliary_weight_decay=adamw_weight_decay,
            auxiliary_optimizer="adamw",
            eligible_weight_decay_semantics=AUXILIARY_ADAMW_WEIGHT_DECAY_MODE,
            auxiliary_weight_decay_semantics=AUXILIARY_ADAMW_WEIGHT_DECAY_MODE,
        )
        return optimizer, audit

    if matrix_momentum_convention != GLOBAL_SGDM_MOMENTUM_CONVENTION:
        raise ValueError("experimental SGDM variants require EMA momentum convention")
    if matrix_weight_decay_mode != GLOBAL_SGDM_WEIGHT_DECAY_MODE:
        raise ValueError(
            "experimental SGDM variants require decoupled matrix weight decay"
        )
    if auxiliary_weight_decay_mode != AUXILIARY_ADAMW_WEIGHT_DECAY_MODE:
        raise ValueError("auxiliary parameters require AdamW decoupled weight decay")
    if matrix_nesterov:
        raise ValueError("Nesterov momentum is disabled for experimental SGDM")

    partition = partition_optimizer_parameters(model)
    static_multiplier_configuration = None
    if optimizer_name == "static_per_matrix_sgdm":
        static_multiplier_configuration = resolve_static_multipliers(
            partition.eligible_matrices,
            default_multiplier=static_default_multiplier,
            matrix_type_multipliers=static_matrix_type_multipliers,
            exact_parameter_multipliers=static_exact_parameter_multipliers,
        )
    if optimizer_name == "static_per_matrix_sgdm":
        matrix_optimizer = StaticPerMatrixSGDM(
            partition.eligible_matrices,
            static_multiplier_configuration["resolved_multipliers"],
            lr=matrix_learning_rate,
            momentum=matrix_momentum,
            weight_decay=matrix_weight_decay,
        )
    elif optimizer_name == "frobenius_normalized_sgdm":
        if frobenius_learning_rate is None:
            raise ValueError(
                "frobenius_learning_rate is required for "
                "frobenius_normalized_sgdm"
            )
        matrix_optimizer = FrobeniusNormalizedSGDM(
            [item.parameter for item in partition.eligible_matrices],
            lr=frobenius_learning_rate,
            momentum=matrix_momentum,
            epsilon=frobenius_epsilon,
            fixed_shape_factor=frobenius_shape_factor,
            weight_decay=matrix_weight_decay,
        )
    else:
        matrix_optimizer = GlobalSGDM(
            [item.parameter for item in partition.eligible_matrices],
            lr=matrix_learning_rate,
            momentum=matrix_momentum,
            weight_decay=matrix_weight_decay,
        )
    matrix_base_learning_rate = (
        frobenius_learning_rate
        if optimizer_name == "frobenius_normalized_sgdm"
        else matrix_learning_rate
    )
    for group in matrix_optimizer.param_groups:
        group["optimizer_role"] = "matrix"
        group["base_lr"] = matrix_base_learning_rate
    auxiliary_optimizer = _configure_auxiliary_adamw(
        partition,
        learning_rate=auxiliary_learning_rate,
        weight_decay=auxiliary_weight_decay,
        betas=auxiliary_betas,
        device_type=device_type,
    )
    optimizer = CompositeOptimizer(matrix_optimizer, auxiliary_optimizer)
    if static_multiplier_configuration is not None:
        optimizer.static_multiplier_configuration = (
            static_multiplier_configuration
        )
    if optimizer_name == "frobenius_normalized_sgdm":
        optimizer.frobenius_configuration = {
            "normalization_version": FROBENIUS_NORMALIZATION_VERSION,
            "normalization_equation": FROBENIUS_NORMALIZATION_EQUATION,
            "frobenius_learning_rate": float(frobenius_learning_rate),
            "frobenius_epsilon": matrix_optimizer.param_groups[0]["epsilon"],
            "frobenius_shape_factor": matrix_optimizer.param_groups[0][
                "fixed_shape_factor"
            ],
        }
    audit = build_parameter_group_audit(
        partition,
        eligible_optimizer=optimizer_name,
        eligible_weight_decay=matrix_weight_decay,
        auxiliary_weight_decay=auxiliary_weight_decay,
        eligible_weight_decay_semantics=matrix_weight_decay_mode,
        auxiliary_weight_decay_semantics=auxiliary_weight_decay_mode,
    )
    return optimizer, audit


def set_optimizer_learning_rates(
    optimizer,
    optimizer_name,
    adamw_learning_rate,
    experimental_schedule_scale,
):
    """Apply one schedule while preserving experimental group LR ratios."""

    validate_optimizer_name(optimizer_name)
    if optimizer_name == "adamw":
        for group in optimizer.param_groups:
            group["lr"] = adamw_learning_rate
        return

    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * experimental_schedule_scale


def get_effective_learning_rates(optimizer, optimizer_name):
    """Return the currently applied matrix and auxiliary learning rates."""

    validate_optimizer_name(optimizer_name)
    if optimizer_name == "adamw":
        learning_rates = {group["lr"] for group in optimizer.param_groups}
        if len(learning_rates) != 1:
            raise ValueError("protected AdamW groups must share one learning rate")
        return {"matrix": None, "auxiliary": learning_rates.pop()}

    matrix_rates = {
        group["lr"]
        for group in optimizer.param_groups
        if group["optimizer_role"] == "matrix"
    }
    auxiliary_rates = {
        group["lr"]
        for group in optimizer.param_groups
        if group["optimizer_role"] == "auxiliary"
    }
    if len(matrix_rates) != 1 or len(auxiliary_rates) != 1:
        raise ValueError("matrix and auxiliary groups must each share one rate")
    return {
        "matrix": matrix_rates.pop(),
        "auxiliary": auxiliary_rates.pop(),
    }
