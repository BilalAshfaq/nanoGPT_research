"""Deterministic parameter partitioning for optimizer experiments."""

from dataclasses import dataclass


ELIGIBLE_MATRIX_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


@dataclass(frozen=True)
class NamedParameter:
    """A parameter paired with its canonical ``named_parameters`` name."""

    name: str
    parameter: object


@dataclass(frozen=True)
class ParameterPartition:
    """Disjoint optimizer groups and the model's tied embedding aliases."""

    eligible_matrices: tuple
    auxiliary: tuple
    tied_parameter_names: tuple
    tied_parameter_canonical_name: str


def _expected_eligible_names(model):
    try:
        num_layers = len(model.transformer.h)
    except (AttributeError, TypeError) as exc:
        raise ValueError("expected a GPT model with transformer.h blocks") from exc

    return {
        f"transformer.h.{layer_index}.{suffix}"
        for layer_index in range(num_layers)
        for suffix in ELIGIBLE_MATRIX_SUFFIXES
    }


def _validate_tied_embeddings(model, all_named_parameters):
    try:
        token_embedding = model.transformer.wte.weight
        output_head = model.lm_head.weight
    except AttributeError as exc:
        raise ValueError(
            "expected transformer.wte.weight and lm_head.weight parameters"
        ) from exc

    if token_embedding is not output_head:
        raise ValueError(
            "transformer.wte.weight and lm_head.weight must share one tensor"
        )

    canonical_names = [
        name
        for name, parameter in all_named_parameters
        if parameter is token_embedding
    ]
    if len(canonical_names) != 1:
        raise ValueError(
            "the tied token embedding/output head must appear exactly once in "
            "model.named_parameters()"
        )
    return canonical_names[0]


def partition_optimizer_parameters(model):
    """Split trainable GPT parameters into eligible matrices and auxiliaries.

    The eligible group contains the four hidden projection weights in every
    Transformer block. Embeddings (including the tied token embedding/output
    head tensor), biases, and LayerNorm parameters remain auxiliary. PyTorch's
    canonical ``named_parameters`` traversal reports the tied tensor once as
    ``transformer.wte.weight``; both aliases are recorded on the partition.
    Frozen parameters are intentionally excluded from both groups.
    """

    all_named_parameters = list(model.named_parameters())
    canonical_tied_name = _validate_tied_embeddings(model, all_named_parameters)
    expected_eligible_names = _expected_eligible_names(model)
    actual_names = {name for name, _ in all_named_parameters}

    missing_names = sorted(expected_eligible_names - actual_names)
    if missing_names:
        raise ValueError(
            "missing expected hidden projection parameters: "
            + ", ".join(missing_names)
        )

    eligible = []
    auxiliary = []
    for name, parameter in all_named_parameters:
        if not parameter.requires_grad:
            continue
        named_parameter = NamedParameter(name, parameter)
        if name in expected_eligible_names:
            if parameter.ndim != 2:
                raise ValueError(
                    f"eligible hidden projection {name!r} must be two-dimensional"
                )
            eligible.append(named_parameter)
        else:
            auxiliary.append(named_parameter)

    eligible.sort(key=lambda item: item.name)
    auxiliary.sort(key=lambda item: item.name)
    _validate_partition(model, eligible, auxiliary)

    return ParameterPartition(
        eligible_matrices=tuple(eligible),
        auxiliary=tuple(auxiliary),
        tied_parameter_names=("transformer.wte.weight", "lm_head.weight"),
        tied_parameter_canonical_name=canonical_tied_name,
    )


def _validate_partition(model, eligible, auxiliary):
    grouped_parameters = eligible + auxiliary
    grouped_ids = [id(item.parameter) for item in grouped_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("a trainable parameter was assigned to multiple groups")

    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    actual_ids = set(grouped_ids)
    if actual_ids != expected_ids:
        raise ValueError(
            "every trainable parameter must be assigned to exactly one optimizer group"
        )


def build_parameter_group_audit(
    partition,
    eligible_optimizer,
    eligible_weight_decay,
    auxiliary_weight_decay,
    auxiliary_optimizer="adamw",
):
    """Return a deterministic, serializable optimizer-assignment audit.

    Auxiliary AdamW follows nanoGPT's existing rule: tensors with two or more
    dimensions receive the configured decay and lower-dimensional tensors do
    not. The eligible optimizer's decay is recorded independently.
    """

    if eligible_weight_decay < 0 or auxiliary_weight_decay < 0:
        raise ValueError("weight decay values must be non-negative")

    entries = []
    for group_name, named_parameters in (
        ("eligible_matrix", partition.eligible_matrices),
        ("auxiliary", partition.auxiliary),
    ):
        for item in named_parameters:
            if group_name == "eligible_matrix":
                optimizer_name = eligible_optimizer
                weight_decay = eligible_weight_decay
            else:
                optimizer_name = auxiliary_optimizer
                weight_decay = (
                    auxiliary_weight_decay if item.parameter.ndim >= 2 else 0.0
                )
            entries.append(
                {
                    "name": item.name,
                    "shape": tuple(item.parameter.shape),
                    "parameter_count": item.parameter.numel(),
                    "group": group_name,
                    "optimizer": optimizer_name,
                    "weight_decay": weight_decay,
                    "weight_decay_treatment": (
                        "decay" if weight_decay != 0.0 else "no_decay"
                    ),
                }
            )

    entries.sort(key=lambda entry: entry["name"])
    return {
        "parameters": entries,
        "totals": {
            "parameter_tensors": len(entries),
            "parameter_count": sum(
                entry["parameter_count"] for entry in entries
            ),
        },
        "tied_parameters": {
            "aliases": partition.tied_parameter_names,
            "canonical_name": partition.tied_parameter_canonical_name,
            "assignment": "auxiliary",
            "optimizer": auxiliary_optimizer,
            "counted_once": True,
        },
    }
