"""Optimizer-agnostic manifest validation and execution."""

import argparse
import datetime
import difflib
import hashlib
import json
import os
import subprocess
import sys


SCHEMA_VERSION = 1
FINAL_OUTCOMES = {"completed", "failed", "divergent", "interrupted"}
INTERRUPTED_RETURN_CODES = {-15, -9, -2, 130, 137, 143}


def repository_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _absolute_repository_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(repository_root(), path)


def resolve_repository_path(path):
    """Resolve a configured path relative to the repository root."""

    expanded = os.path.expandvars(os.path.expanduser(path))
    return os.path.abspath(_absolute_repository_path(expanded))


def _read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    manifest_path = _absolute_repository_path(path)
    manifest = _read_json(manifest_path)
    validate_manifest(manifest)
    for run in manifest["runs"]:
        resolve_run_config(manifest, run)
    manifest["_manifest_path"] = manifest_path
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported experiment manifest schema version")
    if manifest.get("purpose") not in {"pilot", "study", "confirmation"}:
        raise ValueError("manifest purpose must be pilot, study, or confirmation")
    if not isinstance(manifest.get("launch_authorized"), bool):
        raise ValueError("manifest launch_authorized must be a boolean")
    if not isinstance(manifest.get("counts_toward_study_budget"), bool):
        raise ValueError("counts_toward_study_budget must be a boolean")
    if (
        manifest["purpose"] == "pilot"
        and manifest["counts_toward_study_budget"]
    ):
        raise ValueError("pilot runs cannot count toward the study budget")
    if manifest["counts_toward_study_budget"]:
        selection = manifest.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("study manifests require a selection rule")
        required_selection = {
            "group_by",
            "expected_groups",
            "metric",
            "mode",
            "tie_break",
        }
        if required_selection - set(selection):
            raise ValueError("study selection rule is incomplete")

    config_path = _absolute_repository_path(manifest.get("config_file", ""))
    if not os.path.isfile(config_path):
        raise ValueError(f"manifest config file does not exist: {config_path}")

    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("manifest runs must be a list")
    if manifest.get("expected_run_count") != len(runs):
        raise ValueError("manifest run count does not match expected_run_count")
    resources = manifest.get("resources")
    if not isinstance(resources, dict) or resources.get("world_size", 0) <= 0:
        raise ValueError("manifest resources require a positive world_size")

    run_ids = set()
    run_names = set()
    for run in runs:
        required = {
            "run_id",
            "run_name",
            "optimizer_name",
            "seed",
            "max_iters",
            "tokens_per_update",
            "selection_tokens",
            "max_processed_tokens",
            "evaluation_steps",
            "selection_step",
            "selection_values",
            "overrides",
        }
        missing = sorted(required - set(run))
        if missing:
            raise ValueError(
                f"manifest run is missing required fields: {', '.join(missing)}"
            )
        if run["run_id"] in run_ids or run["run_name"] in run_names:
            raise ValueError("manifest run ids and names must be unique")
        run_ids.add(run["run_id"])
        run_names.add(run["run_name"])
        if not isinstance(run["optimizer_name"], str):
            raise ValueError("optimizer_name must be a string")
        if not isinstance(run["overrides"], dict):
            raise ValueError("run overrides must be an object")
        if run["max_iters"] < 0:
            raise ValueError("run max_iters must be nonnegative")
        if run["tokens_per_update"] <= 0:
            raise ValueError("tokens_per_update must be positive")
        expected_max_tokens = (run["max_iters"] + 1) * run["tokens_per_update"]
        if run["max_processed_tokens"] != expected_max_tokens:
            raise ValueError("max_processed_tokens does not match the run budget")
        evaluation_steps = run["evaluation_steps"]
        if (
            not isinstance(evaluation_steps, list)
            or not evaluation_steps
            or evaluation_steps != sorted(set(evaluation_steps))
        ):
            raise ValueError("evaluation_steps must be sorted and unique")
        if evaluation_steps[-1] != run["max_iters"]:
            raise ValueError("final evaluation step must equal max_iters")
        if (
            run["selection_step"] is not None
            and run["selection_step"] not in evaluation_steps
        ):
            raise ValueError("selection_step must be an evaluation step")
        if manifest["counts_toward_study_budget"] and run["selection_step"] is None:
            raise ValueError("study runs require a selection_step")
        expected_selection_tokens = (
            None
            if run["selection_step"] is None
            else run["selection_step"] * run["tokens_per_update"]
        )
        if run["selection_tokens"] != expected_selection_tokens:
            raise ValueError("selection_tokens does not match selection_step")
        forbidden = {"out_dir", "init_from", "optimizer_name", "seed", "max_iters"}
        if forbidden & set(run["overrides"]):
            raise ValueError("reserved run settings cannot appear in overrides")
        if manifest["counts_toward_study_budget"]:
            missing_ties = set(manifest["selection"]["tie_break"]) - set(
                run["selection_values"]
            )
            if missing_ties:
                raise ValueError("run selection_values omit tie-break fields")


def _normalized_freeze_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def validate_environment(manifest):
    lock_path = _absolute_repository_path(manifest["environment_lock"])
    if not os.path.isfile(lock_path):
        raise FileNotFoundError(
            f"missing environment lock {lock_path}; run setup_nanogpt_env.sh"
        )
    with open(lock_path, encoding="utf-8") as lock_file:
        expected = _normalized_freeze_lines(lock_file.read())
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = _normalized_freeze_lines(result.stdout)
    if actual != expected:
        difference = "\n".join(
            difflib.unified_diff(expected, actual, "locked", "current")
        )
        raise RuntimeError(f"environment does not match lock:\n{difference}")
    return {"path": lock_path, "sha256": sha256_file(lock_path)}


def fingerprint_dataset(manifest):
    fingerprints = {}
    for relative_path in manifest["dataset"]["files"]:
        path = _absolute_repository_path(relative_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing dataset file: {path}")
        fingerprints[relative_path] = {
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        }
    return fingerprints


def write_dataset_lock(manifest):
    lock_path = _absolute_repository_path(manifest["dataset"]["lock_file"])
    lock = {
        "schema_version": 1,
        "dataset": manifest["dataset"]["name"],
        "files": fingerprint_dataset(manifest),
    }
    _write_json(lock_path, lock)
    return lock_path


def validate_dataset(manifest):
    actual = fingerprint_dataset(manifest)
    dataset = manifest["dataset"]
    lock_path = _absolute_repository_path(dataset["lock_file"])
    if dataset.get("require_lock", False):
        if not os.path.isfile(lock_path):
            raise FileNotFoundError(
                f"missing dataset lock {lock_path}; run fingerprint-data first"
            )
        lock = _read_json(lock_path)
        if lock.get("dataset") != dataset["name"] or lock.get("files") != actual:
            raise RuntimeError("prepared dataset does not match its fingerprint lock")
    return {"lock_path": lock_path, "files": actual}


def validate_allocated_gpus(manifest):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    resources = manifest["resources"]
    if len(names) != resources["world_size"]:
        raise RuntimeError(
            f"expected {resources['world_size']} allocated GPUs, found {len(names)}"
        )
    required_name = resources.get("device_name_contains")
    if required_name and any(required_name not in name for name in names):
        raise RuntimeError("allocated GPU hardware does not match the manifest")
    return names


def _command_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def resolve_run_config(manifest, run):
    config_path = _absolute_repository_path(manifest["config_file"])
    namespace = {}
    with open(config_path, encoding="utf-8") as config_file:
        exec(compile(config_file.read(), config_path, "exec"), namespace)
    configurable_types = (bool, float, int, str, dict)
    values = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("_") and isinstance(value, configurable_types)
    }
    applied = {
        "optimizer_name": run["optimizer_name"],
        "seed": run["seed"],
        "max_iters": run["max_iters"],
        **run["overrides"],
    }
    for key, value in applied.items():
        if key not in values:
            raise ValueError(f"manifest override is not a config setting: {key}")
        if type(value) is not type(values[key]):
            raise ValueError(f"manifest override type does not match config: {key}")
        values[key] = value

    tokens_per_update = (
        values["gradient_accumulation_steps"]
        * values["batch_size"]
        * values["block_size"]
    )
    if tokens_per_update != run["tokens_per_update"]:
        raise ValueError("resolved config does not match tokens_per_update")
    if values["gradient_accumulation_steps"] % manifest["resources"]["world_size"]:
        raise ValueError("gradient accumulation must divide across DDP ranks")
    evaluation_steps = list(
        range(0, values["max_iters"] + 1, values["eval_interval"])
    )
    if evaluation_steps != run["evaluation_steps"]:
        raise ValueError("resolved config does not match evaluation_steps")
    return values


def _load_evaluation_steps(output_directory):
    path = os.path.join(output_directory, "evaluation_metrics.jsonl")
    if not os.path.isfile(path):
        return set()
    steps = set()
    with open(path, encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                steps.add(json.loads(line)["step"])
    return steps


def _outcome_path(output_directory):
    return os.path.join(output_directory, "outcome.json")


def _read_outcome(output_directory):
    path = _outcome_path(output_directory)
    return _read_json(path) if os.path.isfile(path) else None


def _write_outcome(output_directory, run, status, **extra):
    outcome = {
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "optimizer_name": run["optimizer_name"],
        "status": status,
        "updated_at": _utc_now(),
    }
    outcome.update(extra)
    _write_json(_outcome_path(output_directory), outcome)
    return outcome


def _classify_completed_process(output_directory, run, return_code):
    if return_code in INTERRUPTED_RETURN_CODES:
        return "interrupted"
    summary_path = os.path.join(output_directory, "run_summary.json")
    if os.path.isfile(summary_path):
        summary = _read_json(summary_path)
        if summary.get("metrics", {}).get("divergence_status") == "observed":
            return "divergent"
    if return_code != 0:
        return "failed"
    if run["evaluation_steps"][-1] not in _load_evaluation_steps(output_directory):
        return "failed"
    return "completed"


def run_manifest_entry(manifest, run, environment, dataset):
    output_root = resolve_repository_path(manifest["output_root"])
    output_directory = os.path.join(output_root, run["run_name"])
    outcome = _read_outcome(output_directory)
    checkpoint_path = os.path.join(output_directory, "ckpt.pt")

    if outcome and outcome.get("status") in FINAL_OUTCOMES:
        print(f"Skipping final outcome {run['run_id']}: {outcome['status']}")
        return outcome

    resume = False
    if outcome and outcome.get("status") == "running":
        if os.path.isfile(checkpoint_path):
            resume = True
        else:
            return _write_outcome(
                output_directory,
                run,
                "interrupted",
                reason="previous execution ended before its first checkpoint",
            )
    elif os.path.isdir(output_directory) and os.listdir(output_directory):
        raise RuntimeError(
            f"output directory exists without a tracked outcome: {output_directory}"
        )

    os.makedirs(output_directory, exist_ok=True)
    resolved_config = resolve_run_config(manifest, run)
    resolved = {
        "manifest_id": manifest["manifest_id"],
        "purpose": manifest["purpose"],
        "counts_toward_study_budget": manifest["counts_toward_study_budget"],
        "config_file": manifest["config_file"],
        "run": run,
        "resolved_config": resolved_config,
        "environment": environment,
        "dataset": dataset,
    }
    _write_json(os.path.join(output_directory, "resolved_run.json"), resolved)
    _write_outcome(output_directory, run, "running", resumed=resume)

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={manifest['resources']['world_size']}",
        "train.py",
        manifest["config_file"],
        f"--out_dir={output_directory}",
        f"--optimizer_name={run['optimizer_name']}",
        f"--seed={run['seed']}",
        f"--max_iters={run['max_iters']}",
        f"--init_from={'resume' if resume else 'scratch'}",
    ]
    for key in sorted(run["overrides"]):
        command.append(f"--{key}={_command_value(run['overrides'][key])}")

    print("Launching:", " ".join(command), flush=True)
    started_at = _utc_now()
    result = subprocess.run(command, cwd=repository_root(), check=False)
    status = _classify_completed_process(output_directory, run, result.returncode)
    return _write_outcome(
        output_directory,
        run,
        status,
        resumed=resume,
        started_at=started_at,
        return_code=result.returncode,
    )


def run_manifest(manifest, run_id=None):
    if not manifest["launch_authorized"]:
        raise RuntimeError(
            f"manifest {manifest['manifest_id']} is not authorized for launch"
        )
    gpu_names = validate_allocated_gpus(manifest)
    environment = validate_environment(manifest)
    dataset = validate_dataset(manifest)
    for test_module in manifest.get("preflight_tests", []):
        subprocess.run(
            [sys.executable, "-m", "unittest", test_module, "-v"],
            cwd=repository_root(),
            check=True,
        )
    selected_runs = manifest["runs"]
    if run_id is not None:
        selected_runs = [run for run in selected_runs if run["run_id"] == run_id]
        if not selected_runs:
            raise ValueError(f"unknown manifest run id: {run_id}")

    outcomes = [
        run_manifest_entry(
            manifest,
            run,
            {**environment, "allocated_gpu_names": gpu_names},
            dataset,
        )
        for run in selected_runs
    ]
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("manifest")

    fingerprint_parser = subparsers.add_parser("fingerprint-data")
    fingerprint_parser.add_argument("manifest")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("manifest")
    run_parser.add_argument("--run-id")

    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "validate":
        print(f"valid manifest: {manifest['manifest_id']}")
        return 0
    if args.command == "fingerprint-data":
        print(write_dataset_lock(manifest))
        return 0
    outcomes = run_manifest(manifest, run_id=args.run_id)
    return 0 if all(item["status"] == "completed" for item in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
