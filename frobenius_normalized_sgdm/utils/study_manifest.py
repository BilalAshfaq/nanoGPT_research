"""Materialize the frozen, unauthorized Frobenius-normalized study grid."""

import argparse
import copy
import json
import os

from shared_utils.experiment_manifest import sha256_file, validate_manifest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DESIGN_PATH = os.path.join(
    REPOSITORY_ROOT,
    "frobenius_normalized_sgdm",
    "task_3_4_candidate_design.json",
)
BASELINE_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT, "experiment_manifests", "task_1_6_exploratory.json"
)


def _read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _repository_path(path):
    return os.path.join(REPOSITORY_ROOT, *path.split("/"))


def _validate_comparator_locks(design):
    snapshots = {}
    for family, lock in design["comparators"].items():
        manifest_path = _repository_path(lock["source_manifest_path"])
        report_path = _repository_path(lock["selection_report_path"])
        if sha256_file(manifest_path) != lock["source_manifest_sha256"]:
            raise ValueError(f"locked {family} source manifest changed")
        if sha256_file(report_path) != lock["selection_report_sha256"]:
            raise ValueError(f"locked {family} selection report changed")

        source_manifest = _read_json(manifest_path)
        report = _read_json(report_path)
        if source_manifest.get("manifest_id") != lock["source_manifest_id"]:
            raise ValueError(f"locked {family} source manifest id changed")
        if report.get("manifest_id") != lock["source_manifest_id"]:
            raise ValueError(f"locked {family} selection report id changed")
        winner = report.get("winners", {}).get(lock["winner_group"])
        if not isinstance(winner, dict) or winner.get("status") != "completed":
            raise ValueError(f"locked {family} winner is unavailable")
        for field in ("run_id", "run_name", "selection_values", "selection_record"):
            if winner.get(field) != lock[field]:
                raise ValueError(f"locked {family} winner {field} changed")
        if winner.get("run_summary", {}).get("git_commit") != lock["git_commit"]:
            raise ValueError(f"locked {family} winner git commit changed")
        source_runs = {
            run["run_id"]: run for run in source_manifest.get("runs", [])
        }
        source_run = source_runs.get(lock["run_id"])
        if not isinstance(source_run, dict):
            raise ValueError(f"locked {family} source run is unavailable")
        if source_run.get("run_name") != lock["run_name"]:
            raise ValueError(f"locked {family} source run name changed")
        if source_run.get("overrides") != lock["overrides"]:
            raise ValueError(f"locked {family} source run overrides changed")
        if family == "static_per_matrix_sgdm":
            configuration = winner.get("static_multiplier_configuration", {})
            if configuration.get("mapping_id") != lock["mapping_id"]:
                raise ValueError("locked static mapping id changed")
            if configuration.get("fingerprint_sha256") != lock[
                "mapping_fingerprint_sha256"
            ]:
                raise ValueError("locked static mapping fingerprint changed")
        snapshots[family] = copy.deepcopy(lock)
    return snapshots


def materialize_manifest(design=None, baseline_manifest=None):
    """Return the exact 12-run manifest, always without launch authorization."""

    design = copy.deepcopy(design if design is not None else _read_json(DESIGN_PATH))
    baseline = copy.deepcopy(
        baseline_manifest
        if baseline_manifest is not None
        else _read_json(BASELINE_MANIFEST_PATH)
    )
    comparators = _validate_comparator_locks(design)
    audit = design["shared_code_audit"]
    audit_path = _repository_path(audit["artifact_path"])
    if sha256_file(audit_path) != audit["artifact_sha256"]:
        raise ValueError("locked shared-code audit artifact changed")
    global_lock = comparators["global_sgdm"]
    if baseline.get("manifest_id") != global_lock["source_manifest_id"]:
        raise ValueError("locked common-control source manifest is unavailable")
    if sha256_file(BASELINE_MANIFEST_PATH) != global_lock["source_manifest_sha256"]:
        raise ValueError("locked common-control source manifest changed")

    normalization = design["normalization"]
    budget = design["budget"]
    runs = []
    for learning_rate in design["learning_rate_design"]["values"]:
        for momentum in design["momentum_values"]:
            lr_label = format(learning_rate, "g")
            momentum_label = format(momentum, "g")
            rule_id = normalization["rule_id"]
            runs.append(
                {
                    "run_id": f"frobenius_lr{lr_label}_mom{momentum_label}_{rule_id}",
                    "run_name": (
                        "frobenius_normalized_sgdm_gpt2-124m_"
                        f"lr{lr_label}_mom{momentum_label}_wd0.1_"
                        f"seed{design['exploratory_seed']}_scale{rule_id}"
                    ),
                    "optimizer_name": "frobenius_normalized_sgdm",
                    "seed": design["exploratory_seed"],
                    "max_iters": budget["max_iters"],
                    "tokens_per_update": budget["tokens_per_update"],
                    "selection_tokens": budget["selection_tokens"],
                    "max_processed_tokens": budget["max_processed_tokens"],
                    "evaluation_steps": copy.deepcopy(budget["evaluation_steps"]),
                    "selection_step": budget["selection_step"],
                    "selection_values": {
                        "frobenius_learning_rate": learning_rate,
                        "momentum": momentum,
                        "weight_decay": 0.1,
                        "normalization_rule_id": rule_id,
                        "frobenius_epsilon": normalization["frobenius_epsilon"],
                        "frobenius_shape_factor": normalization[
                            "frobenius_shape_factor"
                        ],
                    },
                    "overrides": {
                        "frobenius_learning_rate": learning_rate,
                        "matrix_momentum": momentum,
                        "frobenius_epsilon": normalization["frobenius_epsilon"],
                        "frobenius_shape_factor": normalization[
                            "frobenius_shape_factor"
                        ],
                    },
                }
            )

    if len(runs) != design["expected_run_count"]:
        raise ValueError("frozen candidate design does not produce 12 runs")
    manifest = {
        "schema_version": baseline["schema_version"],
        "manifest_id": "task-3.4-frobenius-exploratory-v1",
        "purpose": "study",
        "launch_authorized": False,
        "counts_toward_study_budget": True,
        "config_file": "config/task_3_4_frobenius.py",
        "environment_lock": baseline["environment_lock"],
        "preflight_tests": [
            "tests.test_baseline_preflight",
            "tests.test_frobenius_normalization",
            "tests.test_frobenius_normalized_sgdm",
            "tests.test_frobenius_normalized_diagnostics",
            "tests.test_frobenius_study_protocol",
        ],
        "dataset": copy.deepcopy(baseline["dataset"]),
        "resources": copy.deepcopy(baseline["resources"]),
        "output_root": "nanogpt-study-runs/task-3.5",
        "expected_run_count": len(runs),
        "selection": {
            "group_by": "optimizer_name",
            "expected_groups": ["frobenius_normalized_sgdm"],
            "metric": design["selection"]["metric"],
            "mode": design["selection"]["mode"],
            "tie_break": copy.deepcopy(design["selection"]["tie_break"]),
        },
        "confirmation": {
            "manifest_id": "task-3.5-frobenius-confirmation-v1",
            "additional_seeds": copy.deepcopy(design["confirmation_seeds"]),
            "run_name_template": (
                "{optimizer_name}_gpt2-124m_lr{frobenius_learning_rate}_"
                "mom{momentum}_wd{weight_decay}_seed{seed}_"
                "scale{normalization_rule_id}"
            ),
        },
        "candidate_design": {
            "design_id": design["design_id"],
            "path": os.path.relpath(DESIGN_PATH, REPOSITORY_ROOT).replace("\\", "/"),
            "sha256": sha256_file(DESIGN_PATH),
        },
        "comparator_locks": comparators,
        "shared_code_audit": copy.deepcopy(design["shared_code_audit"]),
        "claim_gate": copy.deepcopy(design["claim_gate"]),
        "outcome_policy": copy.deepcopy(design["outcome_policy"]),
        "runs": runs,
    }
    validate_manifest(manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = materialize_manifest()
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"materialized unauthorized manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
