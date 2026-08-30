"""Report matched Global, Static, and Frobenius SGDM study results."""

import argparse
import json
import math
import os
import statistics

from shared_utils.experiment_manifest import (
    FINAL_OUTCOMES,
    load_manifest,
    repository_root,
    resolve_run_config,
    sha256_file,
)
from shared_utils.experiment_results import collect_results, report_output_path
from shared_utils.study_comparison import (
    COMMON_MATCHED_CONFIG_KEYS,
    completed_winner,
    manifest_run,
    read_json,
    runtime_locks,
    verify_matched_controls,
)


FAMILIES = (
    "global_sgdm",
    "static_per_matrix_sgdm",
    "frobenius_normalized_sgdm",
)
CONFIRMATION_SEEDS = (2027, 4099)


def _repository_path(relative_path):
    return os.path.join(repository_root(), *relative_path.split("/"))


def _verify_comparator_locks(
    frobenius_manifest,
    global_manifest,
    global_selection,
    static_manifest,
    static_selection,
):
    candidate = frobenius_manifest["candidate_design"]
    design_path = _repository_path(candidate["path"])
    if sha256_file(design_path) != candidate["sha256"]:
        raise ValueError("locked Frobenius candidate design changed")
    design = read_json(design_path)
    if frobenius_manifest["comparator_locks"] != design["comparators"]:
        raise ValueError("Frobenius manifest comparator locks changed")
    audit = frobenius_manifest["shared_code_audit"]
    if sha256_file(_repository_path(audit["artifact_path"])) != audit[
        "artifact_sha256"
    ]:
        raise ValueError("locked Task 3.4 shared-code audit changed")
    supplied = {
        "global_sgdm": (global_manifest, global_selection),
        "static_per_matrix_sgdm": (static_manifest, static_selection),
    }
    for family, lock in frobenius_manifest["comparator_locks"].items():
        manifest, selection = supplied[family]
        if sha256_file(_repository_path(lock["source_manifest_path"])) != lock[
            "source_manifest_sha256"
        ]:
            raise ValueError(f"locked {family} source manifest changed")
        if sha256_file(_repository_path(lock["selection_report_path"])) != lock[
            "selection_report_sha256"
        ]:
            raise ValueError(f"locked {family} selection report changed")
        if manifest["manifest_id"] != lock["source_manifest_id"]:
            raise ValueError(f"supplied {family} manifest is not the locked source")
        if selection.get("manifest_id") != lock["source_manifest_id"]:
            raise ValueError(f"supplied {family} selection is not the locked report")
        winner = selection.get("winners", {}).get(lock["winner_group"], {})
        for key in ("run_id", "run_name", "selection_values", "selection_record"):
            if winner.get(key) != lock[key]:
                raise ValueError(f"locked {family} winner {key} changed")
        if winner.get("run_summary", {}).get("git_commit") != lock["git_commit"]:
            raise ValueError(f"locked {family} winner git commit changed")
        run = manifest_run(manifest, winner)
        if run.get("overrides") != lock["overrides"]:
            raise ValueError(f"locked {family} winner overrides changed")
        if family == "static_per_matrix_sgdm":
            configuration = run.get("static_multiplier_configuration", {})
            if configuration.get("mapping_id") != lock["mapping_id"]:
                raise ValueError("locked static mapping id changed")
            if configuration.get("fingerprint_sha256") != lock[
                "mapping_fingerprint_sha256"
            ]:
                raise ValueError("locked static mapping fingerprint changed")


def _selected_runs_and_results(
    global_manifest,
    global_selection,
    static_manifest,
    static_selection,
    frobenius_manifest,
    frobenius_selection,
):
    _verify_comparator_locks(
        frobenius_manifest,
        global_manifest,
        global_selection,
        static_manifest,
        static_selection,
    )
    manifests = {
        "global_sgdm": global_manifest,
        "static_per_matrix_sgdm": static_manifest,
        "frobenius_normalized_sgdm": frobenius_manifest,
    }
    selections = {
        "global_sgdm": global_selection,
        "static_per_matrix_sgdm": static_selection,
        "frobenius_normalized_sgdm": frobenius_selection,
    }
    if frobenius_selection.get("manifest_id") != frobenius_manifest["manifest_id"]:
        raise ValueError("Frobenius selection is not from the frozen manifest")
    winners = {
        family: completed_winner(selections[family], family)
        for family in FAMILIES
    }
    runs = {
        family: manifest_run(manifests[family], winners[family])
        for family in FAMILIES
    }
    design = read_json(_repository_path(frobenius_manifest["candidate_design"]["path"]))
    normalization = design["normalization"]
    frobenius_values = runs["frobenius_normalized_sgdm"]["selection_values"]
    expected_normalization = {
        "normalization_rule_id": normalization["rule_id"],
        "frobenius_epsilon": normalization["frobenius_epsilon"],
        "frobenius_shape_factor": normalization["frobenius_shape_factor"],
    }
    for key, expected in expected_normalization.items():
        if frobenius_values.get(key) != expected:
            raise ValueError(f"selected Frobenius {key} changed")
    runtime = verify_matched_controls(
        ("global_sgdm", global_manifest, runs["global_sgdm"], winners["global_sgdm"]),
        [
            (
                "static_per_matrix_sgdm",
                static_manifest,
                runs["static_per_matrix_sgdm"],
                winners["static_per_matrix_sgdm"],
            ),
            (
                "frobenius_normalized_sgdm",
                frobenius_manifest,
                runs["frobenius_normalized_sgdm"],
                winners["frobenius_normalized_sgdm"],
            ),
        ],
        COMMON_MATCHED_CONFIG_KEYS,
    )
    return manifests, winners, runs, runtime


def _diagnostic_summary(output_directory):
    path = os.path.join(output_directory, "optimizer_diagnostics.jsonl")
    if not os.path.isfile(path):
        return {
            "collected": False,
            "reason": "diagnostics_disabled_for_matched_study_run",
        }
    step_count = 0
    matrix_count = 0
    epsilon_dominated_count = 0
    zero_momentum_count = 0
    with open(path, encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            step_count += 1
            for matrix in record.get("matrices", []):
                matrix_count += 1
                epsilon_dominated_count += int(
                    matrix.get("epsilon_dominated") is True
                )
                zero_momentum_count += int(matrix.get("zero_momentum") is True)
    return {
        "collected": True,
        "diagnostic_step_count": step_count,
        "matrix_record_count": matrix_count,
        "epsilon_dominated_count": epsilon_dominated_count,
        "zero_momentum_count": zero_momentum_count,
    }


def _measurement(family, result, run):
    values = result["selection_values"]
    measurement = {
        "run_id": result["run_id"],
        "run_name": result["run_name"],
        "seed": result["seed"],
        "status": result["status"],
        "resumed": result.get("resumed", False),
        "selection_record": result["selection_record"],
        "selection_values": values,
        "run_metrics": (
            result.get("run_summary", {}).get("metrics")
            if result.get("run_summary")
            else None
        ),
    }
    if family == "static_per_matrix_sgdm":
        measurement["static_multiplier_configuration"] = run[
            "static_multiplier_configuration"
        ]
    if family == "frobenius_normalized_sgdm":
        measurement["normalization"] = {
            "rule_id": values["normalization_rule_id"],
            "frobenius_learning_rate": values["frobenius_learning_rate"],
            "momentum": values["momentum"],
            "frobenius_epsilon": values["frobenius_epsilon"],
            "frobenius_shape_factor": values["frobenius_shape_factor"],
        }
        measurement["normalization_events"] = _diagnostic_summary(
            result["output_directory"]
        )
    return measurement


def exploratory_comparison(
    global_manifest,
    global_selection,
    static_manifest,
    static_selection,
    frobenius_manifest,
    frobenius_selection,
):
    manifests, winners, runs, runtime = _selected_runs_and_results(
        global_manifest,
        global_selection,
        static_manifest,
        static_selection,
        frobenius_manifest,
        frobenius_selection,
    )
    measurements = {
        family: _measurement(family, winners[family], runs[family])
        for family in FAMILIES
    }
    losses = {
        family: winners[family]["selection_record"]["validation_loss"]
        for family in FAMILIES
    }
    measurements["validation_loss_differences"] = {
        "static_minus_global": (
            losses["static_per_matrix_sgdm"] - losses["global_sgdm"]
        ),
        "frobenius_minus_global": (
            losses["frobenius_normalized_sgdm"] - losses["global_sgdm"]
        ),
        "frobenius_minus_static": (
            losses["frobenius_normalized_sgdm"]
            - losses["static_per_matrix_sgdm"]
        ),
    }
    return {
        "report_type": "exploratory_seed_1337_three_family",
        "claim_status": (
            "confirmation seeds 2027 and 4099 are required for every family; "
            "no final H3 claim is permitted"
        ),
        "observed_measurements": measurements,
        "matched_controls": {
            "status": "verified",
            "seed": 1337,
            "selection_step": 999,
            "selection_tokens": 491_028_480,
            "environment_lock": manifests["global_sgdm"]["environment_lock"],
            "dataset_lock": manifests["global_sgdm"]["dataset"]["lock_file"],
            "runtime_fingerprints": runtime,
            "config_keys": list(COMMON_MATCHED_CONFIG_KEYS),
            "optimizer_hyperparameters_are_family_specific": [
                "matrix_learning_rate",
                "frobenius_learning_rate",
                "matrix_momentum",
                "static_multiplier_configuration",
                "frobenius_normalization_rule",
            ],
        },
        "interpretation": None,
    }


def _confirmation_results(manifest, family):
    results = [
        result
        for result in collect_results(manifest)
        if result["optimizer_name"] == family
    ]
    if {result["seed"] for result in results} != set(CONFIRMATION_SEEDS):
        raise ValueError(f"{family} confirmation seeds must be 2027 and 4099")
    return sorted(results, key=lambda result: result["seed"])


def _is_successful_matched_result(result):
    if (
        result["status"] != "completed"
        or result.get("resumed", False)
        or result.get("selection_record") is None
        or result.get("run_summary") is None
    ):
        return False
    metrics = result["run_summary"].get("metrics", {})
    return (
        metrics.get("numerical_status", "ok") == "ok"
        and metrics.get("divergence_status", "not_observed") == "not_observed"
    )


def _numeric_metric_aggregates(results):
    completed = [
        result
        for result in results
        if _is_successful_matched_result(result)
    ]
    if not completed:
        return {}
    common = set.intersection(
        *[
            {
                key
                for key, value in result["run_summary"]["metrics"].items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value is not None
                and math.isfinite(value)
            }
            for result in completed
        ]
    )
    aggregates = {}
    for key in sorted(common):
        values = [result["run_summary"]["metrics"][key] for result in completed]
        aggregates[key] = {
            "mean": statistics.mean(values),
            "sample_standard_deviation": (
                statistics.stdev(values) if len(values) >= 2 else None
            ),
        }
    return aggregates


def confirmed_comparison(
    global_manifest,
    global_selection,
    static_manifest,
    static_selection,
    frobenius_manifest,
    frobenius_selection,
    confirmation_manifests,
):
    manifests, winners, runs, reference_runtime = _selected_runs_and_results(
        global_manifest,
        global_selection,
        static_manifest,
        static_selection,
        frobenius_manifest,
        frobenius_selection,
    )
    family_reports = {}
    unsuccessful = []
    all_complete = True
    for family in FAMILIES:
        confirmation_manifest = confirmation_manifests[family]
        for key in ("environment_lock", "dataset", "resources"):
            if confirmation_manifest[key] != manifests[family][key]:
                raise ValueError(f"{family} confirmation {key} changed")
        confirmations = _confirmation_results(
            confirmation_manifest, family
        )
        seed_results = [winners[family], *confirmations]
        selected_config = resolve_run_config(manifests[family], runs[family])
        for result in confirmations:
            if result["status"] not in FINAL_OUTCOMES:
                raise RuntimeError(
                    f"nonfinal {family} confirmation outcome: {result['run_id']}"
                )
            if result["selection_values"] != winners[family]["selection_values"]:
                raise ValueError(f"{family} confirmation configuration changed")
            confirmation_run = manifest_run(confirmation_manifest, result)
            confirmation_config = resolve_run_config(
                confirmation_manifest, confirmation_run
            )
            selected_comparable = dict(selected_config)
            confirmation_comparable = dict(confirmation_config)
            selected_comparable.pop("seed")
            confirmation_comparable.pop("seed")
            if confirmation_comparable != selected_comparable:
                raise ValueError(f"{family} confirmation resolved config changed")
            if _is_successful_matched_result(result):
                record = result.get("selection_record")
                if (
                    not isinstance(record, dict)
                    or record.get("step") != 999
                    or record.get("processed_tokens") != 491_028_480
                ):
                    raise ValueError(
                        f"{family} confirmation lacks the matched checkpoint"
                    )
                if runtime_locks(result) != reference_runtime:
                    raise ValueError(f"{family} confirmation runtime locks differ")
            else:
                unsuccessful.append(result)
        eligible = [
            result
            for result in seed_results
            if _is_successful_matched_result(result)
        ]
        values = [
            result["selection_record"]["validation_loss"] for result in eligible
        ]
        all_complete = all_complete and len(values) == 3
        family_reports[family] = {
            "selected_configuration": winners[family]["selection_values"],
            "seed_results": [
                _measurement(
                    family,
                    result,
                    (
                        runs[family]
                        if result["seed"] == 1337
                        else manifest_run(
                            confirmation_manifest, result
                        )
                    ),
                )
                for result in seed_results
            ],
            "completed_matched_seed_count": len(values),
            "validation_loss_mean": statistics.mean(values) if values else None,
            "validation_loss_sample_standard_deviation": (
                statistics.stdev(values) if len(values) >= 2 else None
            ),
            "run_metric_aggregates": _numeric_metric_aggregates(seed_results),
        }
    return {
        "report_type": "confirmed_three_family_comparison",
        "claim_status": (
            "three matched successful seeds are available for every family; "
            "H3 interpretation may be assessed"
            if all_complete
            else "H3 claim blocked: every family requires three matched successful seeds"
        ),
        "claim_gate_passed": all_complete,
        "optimizer_results": family_reports,
        "unsuccessful_or_ineligible_runs": unsuccessful,
        "matched_controls": {
            "status": "verified",
            "runtime_fingerprints": reference_runtime,
            "config_keys": list(COMMON_MATCHED_CONFIG_KEYS),
        },
        "interpretation": None,
    }


def _add_exploratory_arguments(parser):
    parser.add_argument("--global-manifest", required=True)
    parser.add_argument("--global-selection", required=True)
    parser.add_argument("--static-manifest", required=True)
    parser.add_argument("--static-selection", required=True)
    parser.add_argument("--frobenius-manifest", required=True)
    parser.add_argument("--frobenius-selection", required=True)
    parser.add_argument("--output", required=True)


def _load_exploratory_inputs(args):
    return (
        load_manifest(args.global_manifest),
        read_json(args.global_selection),
        load_manifest(args.static_manifest),
        read_json(args.static_selection),
        load_manifest(args.frobenius_manifest),
        read_json(args.frobenius_selection),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    exploratory_parser = subparsers.add_parser("exploratory")
    _add_exploratory_arguments(exploratory_parser)
    confirmed_parser = subparsers.add_parser("confirmed")
    _add_exploratory_arguments(confirmed_parser)
    confirmed_parser.add_argument("--global-confirmation-manifest", required=True)
    confirmed_parser.add_argument("--static-confirmation-manifest", required=True)
    confirmed_parser.add_argument("--frobenius-confirmation-manifest", required=True)
    args = parser.parse_args(argv)

    inputs = _load_exploratory_inputs(args)
    if args.command == "exploratory":
        report = exploratory_comparison(*inputs)
    else:
        report = confirmed_comparison(
            *inputs,
            confirmation_manifests={
                "global_sgdm": load_manifest(args.global_confirmation_manifest),
                "static_per_matrix_sgdm": load_manifest(
                    args.static_confirmation_manifest
                ),
                "frobenius_normalized_sgdm": load_manifest(
                    args.frobenius_confirmation_manifest
                ),
            },
        )
    output_path = report_output_path(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
