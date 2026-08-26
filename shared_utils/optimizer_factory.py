"""Optimizer selection shared by the nanoGPT training entry point."""

import inspect

import torch

from shared_utils.composite_optimizer import CompositeOptimizer
from shared_utils.parameter_partition import (
    build_parameter_group_audit,
    partition_optimizer_parameters,
)
from tuned_global_sgdm.utils.global_sgdm import GlobalSGDM


SUPPORTED_OPTIMIZERS = ("adamw", "global_sgdm")
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
        return optimizer, None

    if matrix_momentum_convention != GLOBAL_SGDM_MOMENTUM_CONVENTION:
        raise ValueError("global_sgdm requires EMA momentum convention")
    if matrix_weight_decay_mode != GLOBAL_SGDM_WEIGHT_DECAY_MODE:
        raise ValueError("global_sgdm requires decoupled matrix weight decay")
    if auxiliary_weight_decay_mode != AUXILIARY_ADAMW_WEIGHT_DECAY_MODE:
        raise ValueError("auxiliary parameters require AdamW decoupled weight decay")
    if matrix_nesterov:
        raise ValueError("Nesterov momentum is disabled for global_sgdm")

    partition = partition_optimizer_parameters(model)
    matrix_optimizer = GlobalSGDM(
        [item.parameter for item in partition.eligible_matrices],
        lr=matrix_learning_rate,
        momentum=matrix_momentum,
        weight_decay=matrix_weight_decay,
    )
    for group in matrix_optimizer.param_groups:
        group["optimizer_role"] = "matrix"
        group["base_lr"] = matrix_learning_rate
    auxiliary_optimizer = _configure_auxiliary_adamw(
        partition,
        learning_rate=auxiliary_learning_rate,
        weight_decay=auxiliary_weight_decay,
        betas=auxiliary_betas,
        device_type=device_type,
    )
    optimizer = CompositeOptimizer(matrix_optimizer, auxiliary_optimizer)
    audit = build_parameter_group_audit(
        partition,
        eligible_optimizer="global_sgdm",
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
