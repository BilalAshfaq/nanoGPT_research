"""Reproducibility metadata and measurements for optimizer experiments."""

import copy
import json
import os
import platform
import subprocess
import time


def get_git_commit_hash(repository_directory):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_directory,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def get_hardware_metadata(torch_module, device, device_type):
    metadata = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": str(torch_module.__version__),
        "device": device,
        "device_type": device_type,
    }
    if device_type == "cuda" and torch_module.cuda.is_available():
        properties = torch_module.cuda.get_device_properties(device)
        metadata["cuda"] = {
            "device_count": torch_module.cuda.device_count(),
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    else:
        metadata["cuda"] = "not_applicable"
    return metadata


def get_peak_gpu_memory_metadata(torch_module, device, device_type):
    if device_type != "cuda":
        return {"bytes": None, "status": "not_applicable"}
    return {
        "bytes": torch_module.cuda.max_memory_allocated(device),
        "status": "measured",
    }


def build_optimizer_group_signature(model, optimizer):
    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    signature = []
    seen_parameter_ids = set()
    for index, group in enumerate(optimizer.param_groups):
        group_parameters = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                raise ValueError("optimizer parameter groups must be disjoint")
            seen_parameter_ids.add(parameter_id)
            if parameter_id not in parameter_names:
                raise ValueError("optimizer contains an unnamed model parameter")
            group_parameters.append(
                {
                    "name": parameter_names[parameter_id],
                    "shape": tuple(parameter.shape),
                }
            )
        group_parameters.sort(key=lambda entry: entry["name"])
        signature.append(
            {
                "index": index,
                "optimizer_role": group.get("optimizer_role", "adamw"),
                "parameters": group_parameters,
            }
        )

    expected_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if seen_parameter_ids != expected_parameter_ids:
        raise ValueError(
            "optimizer groups must contain every trainable model parameter exactly once"
        )
    return signature


def validate_resume_compatibility(
    checkpoint,
    optimizer_name,
    optimizer_group_signature,
    optimizer_settings=None,
):
    try:
        saved_optimizer = checkpoint["run_metadata"]["optimizer"]
    except KeyError as exc:
        raise ValueError(
            "checkpoint lacks optimizer compatibility metadata and cannot be resumed"
        ) from exc

    saved_name = saved_optimizer.get("name")
    if saved_name != optimizer_name:
        raise ValueError(
            "checkpoint optimizer mismatch: "
            f"saved {saved_name!r}, requested {optimizer_name!r}"
        )
    saved_signature = saved_optimizer.get("group_signature")
    if saved_signature != optimizer_group_signature:
        raise ValueError(
            "checkpoint optimizer parameter-group structure does not match "
            "the current model and configuration"
        )
    if (
        optimizer_settings is not None
        and saved_optimizer.get("settings") != optimizer_settings
    ):
        raise ValueError(
            "checkpoint optimizer settings do not match the current configuration"
        )


def build_run_metadata(
    *,
    config,
    git_commit,
    model_args,
    trainable_parameter_count,
    optimizer_name,
    optimizer_settings,
    optimizer_group_signature,
    parameter_group_audit,
    seed,
    dataset,
    tokenizer,
    block_size,
    batch_size,
    gradient_accumulation_steps,
    ddp_world_size,
    tokens_per_iteration,
    precision,
    hardware,
):
    return {
        "config": copy.deepcopy(config),
        "git_commit": git_commit,
        "model": {
            "architecture": copy.deepcopy(model_args),
            "trainable_parameter_count": trainable_parameter_count,
        },
        "optimizer": {
            "name": optimizer_name,
            "settings": copy.deepcopy(optimizer_settings),
            "group_signature": copy.deepcopy(optimizer_group_signature),
            "parameter_group_audit": copy.deepcopy(parameter_group_audit),
        },
        "random_seed": seed,
        "data": {
            "dataset": dataset,
            "tokenizer": tokenizer,
            "sequence_length": block_size,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps_per_process": (
                gradient_accumulation_steps
            ),
            "ddp_world_size": ddp_world_size,
            "effective_batch_size": (
                batch_size * gradient_accumulation_steps * ddp_world_size
            ),
            "tokens_per_iteration": tokens_per_iteration,
        },
        "precision": precision,
        "hardware": copy.deepcopy(hardware),
    }


def snapshot_run_metadata(base_metadata, progress, metrics):
    metadata = copy.deepcopy(base_metadata)
    metadata["progress"] = copy.deepcopy(progress)
    metadata["metrics"] = copy.deepcopy(metrics)
    return metadata


def write_run_summary(output_directory, run_metadata):
    summary_path = os.path.join(output_directory, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(run_metadata, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")


class OptimizerStepTimer:
    """Accumulate CPU or CUDA optimizer-step wall time without per-step sync."""

    def __init__(
        self,
        torch_module,
        device_type,
        initial_total_seconds=0.0,
        initial_step_count=0,
    ):
        self.torch = torch_module
        self.device_type = device_type
        self.total_seconds = initial_total_seconds
        self.step_count = initial_step_count
        self._cpu_start = None
        self._cuda_start = None
        self._pending_cuda_events = []

    def start(self):
        if self.device_type == "cuda":
            self._cuda_start = self.torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        else:
            self._cpu_start = time.perf_counter()

    def stop(self):
        if self.device_type == "cuda":
            end_event = self.torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._pending_cuda_events.append((self._cuda_start, end_event))
            self._cuda_start = None
        else:
            self.total_seconds += time.perf_counter() - self._cpu_start
            self._cpu_start = None
        self.step_count += 1

    def flush(self):
        if not self._pending_cuda_events:
            return
        self._pending_cuda_events[-1][1].synchronize()
        self.total_seconds += sum(
            start_event.elapsed_time(end_event) / 1000.0
            for start_event, end_event in self._pending_cuda_events
        )
        self._pending_cuda_events.clear()

    @property
    def mean_seconds(self):
        if self.step_count == 0:
            return 0.0
        return self.total_seconds / self.step_count
