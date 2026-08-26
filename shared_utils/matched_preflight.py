"""Deterministic preflight checks for matched optimizer comparisons."""

import hashlib

import torch


def _update_tensor_fingerprint(digest, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())


def _update_batch_fingerprint(digest, value):
    if torch.is_tensor(value):
        _update_tensor_fingerprint(digest, value)
    elif isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("utf-8"))
        for item in value:
            _update_batch_fingerprint(digest, item)
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_batch_fingerprint(digest, value[key])
    else:
        raise TypeError(
            "preflight batches may contain only tensors, tuples, lists, or dicts"
        )


def fingerprint_model_parameters(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        _update_tensor_fingerprint(digest, parameter)
    return digest.hexdigest()


def fingerprint_batch(batch):
    digest = hashlib.sha256()
    _update_batch_fingerprint(digest, batch)
    return digest.hexdigest()


def capture_matched_preflight(
    *,
    seed,
    model_factory,
    optimizer_factory,
    batch_sampler_factory,
    num_batches,
):
    """Capture initialization and pre-update batch fingerprints for one path."""

    if num_batches <= 0:
        raise ValueError("preflight num_batches must be positive")
    torch.manual_seed(seed)
    model = model_factory()
    optimizer_factory(model)
    model_fingerprint = fingerprint_model_parameters(model)
    batch_sampler = batch_sampler_factory()
    batch_fingerprints = [
        fingerprint_batch(batch_sampler()) for _ in range(num_batches)
    ]
    return {
        "seed": seed,
        "model_parameter_fingerprint": model_fingerprint,
        "batch_fingerprints": batch_fingerprints,
        "num_batches": num_batches,
    }


def assert_preflights_match(first, second):
    if first["seed"] != second["seed"]:
        raise AssertionError("preflight seeds do not match")
    if (
        first["model_parameter_fingerprint"]
        != second["model_parameter_fingerprint"]
    ):
        raise AssertionError("initialized model parameters do not match")
    if first["batch_fingerprints"] != second["batch_fingerprints"]:
        raise AssertionError("sampled training batches do not match")
