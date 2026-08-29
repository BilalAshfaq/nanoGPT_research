"""Deterministic selection and reporting for optimizer manifests."""

import argparse
import copy
import json
import os
import statistics

from shared_utils.experiment_manifest import FINAL_OUTCOMES, load_manifest


def _read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _output_directory(manifest, run):
    root = os.path.expandvars(os.path.expanduser(manifest["output_root"]))
    return os.path.join(root, run["run_name"])


def evaluation_records(output_directory):
    path = os.path.join(output_directory, "evaluation_metrics.jsonl")
    if not os.path.isfile(path):
        return {}
    by_step = {}
    with open(path, encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                record = json.loads(line)
                by_step[record["step"]] = record
    return by_step


def collect_results(manifest):
    results = []
    for run in manifest["runs"]:
        output_directory = _output_directory(manifest, run)
        outcome_path = os.path.join(output_directory, "outcome.json")
        summary_path = os.path.join(output_directory, "run_summary.json")
        outcome = _read_json(outcome_path) if os.path.isfile(outcome_path) else None
        summary = _read_json(summary_path) if os.path.isfile(summary_path) else None
        summary_view = None
        if summary is not None:
            summary_view = {
                key: summary.get(key)
                for key in (
                    "git_commit",
                    "progress",
                    "metrics",
                    "hardware",
                    "precision",
                    "data",
                )
            }
        records = evaluation_records(output_directory)
        selection_record = records.get(run["selection_step"])
        result = {
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "optimizer_name": run["optimizer_name"],
            "seed": run["seed"],
            "status": outcome["status"] if outcome else "pending",
            "selection_values": copy.deepcopy(run["selection_values"]),
            "selection_record": selection_record,
            "run_summary": summary_view,
            "run_summary_path": summary_path,
            "output_directory": output_directory,
        }
        if "static_multiplier_configuration" in run:
            result["static_multiplier_configuration"] = copy.deepcopy(
                run["static_multiplier_configuration"]
            )
        results.append(result)
    return results


def select_winners(manifest):
    if not manifest["counts_toward_study_budget"]:
        raise ValueError("pilot manifests are ineligible for study selection")
    selection = manifest["selection"]
    metric = selection["metric"]
    reverse = selection["mode"] == "max"
    if selection["mode"] not in {"min", "max"}:
        raise ValueError("selection mode must be min or max")

    results = collect_results(manifest)
    nonfinal = [
        result["run_id"]
        for result in results
        if result["status"] not in FINAL_OUTCOMES
    ]
    if nonfinal:
        raise RuntimeError(
            "selection requires a final outcome for every run; nonfinal: "
            + ", ".join(nonfinal)
        )

    eligible = []
    for result in results:
        record = result["selection_record"]
        if result["status"] == "completed" and record is not None:
            result["selection_metric"] = record[metric]
            eligible.append(result)

    winners = {}
    groups = sorted({item[selection["group_by"]] for item in eligible})
    for group in groups:
        candidates = [
            item for item in eligible if item[selection["group_by"]] == group
        ]
        candidates.sort(
            key=lambda item: tuple(
                item["selection_values"][key]
                for key in selection["tie_break"]
            )
        )
        candidates.sort(
            key=lambda item: item["selection_metric"], reverse=reverse
        )
        winners[group] = candidates[0]
    if set(winners) != set(selection["expected_groups"]):
        missing = sorted(set(selection["expected_groups"]) - set(winners))
        raise RuntimeError(
            "cannot select every required optimizer family; missing: "
            + ", ".join(missing)
        )
    return winners


def generate_confirmation_manifest(exploratory_manifest, winners):
    confirmation = exploratory_manifest["confirmation"]
    runs_by_id = {
        run["run_id"]: run for run in exploratory_manifest["runs"]
    }
    runs = []
    for group in sorted(winners):
        winning_run = runs_by_id[winners[group]["run_id"]]
        for seed in confirmation["additional_seeds"]:
            values = dict(winning_run["selection_values"])
            values.update(
                {
                    "optimizer_name": winning_run["optimizer_name"],
                    "seed": seed,
                }
            )
            run_name = confirmation["run_name_template"].format(**values)
            confirmation_run = {
                "run_id": f"confirm_{group}_seed{seed}",
                "run_name": run_name,
                "optimizer_name": winning_run["optimizer_name"],
                "seed": seed,
                "max_iters": winning_run["max_iters"],
                "tokens_per_update": winning_run["tokens_per_update"],
                "selection_tokens": winning_run["selection_tokens"],
                "max_processed_tokens": winning_run["max_processed_tokens"],
                "evaluation_steps": winning_run["evaluation_steps"],
                "selection_step": winning_run["selection_step"],
                "selection_values": copy.deepcopy(
                    winning_run["selection_values"]
                ),
                "overrides": copy.deepcopy(winning_run["overrides"]),
            }
            if "static_multiplier_configuration" in winning_run:
                confirmation_run["static_multiplier_configuration"] = (
                    copy.deepcopy(
                        winning_run["static_multiplier_configuration"]
                    )
                )
            runs.append(confirmation_run)

    return {
        "schema_version": exploratory_manifest["schema_version"],
        "manifest_id": confirmation["manifest_id"],
        "purpose": "confirmation",
        "launch_authorized": False,
        "counts_toward_study_budget": True,
        "config_file": exploratory_manifest["config_file"],
        "environment_lock": exploratory_manifest["environment_lock"],
        "preflight_tests": copy.deepcopy(
            exploratory_manifest.get("preflight_tests", [])
        ),
        "dataset": copy.deepcopy(exploratory_manifest["dataset"]),
        "resources": copy.deepcopy(exploratory_manifest["resources"]),
        "output_root": exploratory_manifest["output_root"],
        "expected_run_count": len(runs),
        "selection": copy.deepcopy(exploratory_manifest["selection"]),
        "runs": runs,
    }


def selection_report(manifest):
    results = collect_results(manifest)
    winners = select_winners(manifest)
    return {
        "manifest_id": manifest["manifest_id"],
        "results": results,
        "winners": winners,
    }


def final_report(exploratory_manifest, confirmation_manifest):
    winners = select_winners(exploratory_manifest)
    exploratory_results = {
        result["run_id"]: result for result in collect_results(exploratory_manifest)
    }
    confirmation_results = collect_results(confirmation_manifest)
    nonfinal_confirmation = [
        result["run_id"]
        for result in confirmation_results
        if result["status"] not in FINAL_OUTCOMES
    ]
    if nonfinal_confirmation:
        raise RuntimeError(
            "final report requires every confirmation outcome; nonfinal: "
            + ", ".join(nonfinal_confirmation)
        )
    report = {
        "exploratory_manifest_id": exploratory_manifest["manifest_id"],
        "confirmation_manifest_id": confirmation_manifest["manifest_id"],
        "optimizer_results": {},
        "unsuccessful_runs": [],
    }
    for result in list(exploratory_results.values()) + confirmation_results:
        if result["status"] not in {"completed", "pending"}:
            report["unsuccessful_runs"].append(result)

    metric = exploratory_manifest["selection"]["metric"]
    for optimizer_name, winner in winners.items():
        seed_results = [exploratory_results[winner["run_id"]]]
        seed_results.extend(
            result
            for result in confirmation_results
            if result["optimizer_name"] == optimizer_name
        )
        completed = [
            result
            for result in seed_results
            if result["status"] == "completed"
            and result["selection_record"] is not None
        ]
        values = [result["selection_record"][metric] for result in completed]
        metric_names = set.intersection(
            *[
                {
                    key
                    for key, value in result["run_summary"]["metrics"].items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value is not None
                }
                for result in completed
                if result["run_summary"] is not None
            ]
        ) if completed and all(
            result["run_summary"] is not None for result in completed
        ) else set()
        metric_aggregates = {}
        for metric_name in sorted(metric_names):
            metric_values = [
                result["run_summary"]["metrics"][metric_name]
                for result in completed
            ]
            metric_aggregates[metric_name] = {
                "mean": statistics.mean(metric_values),
                "sample_standard_deviation": (
                    statistics.stdev(metric_values)
                    if len(metric_values) >= 2
                    else None
                ),
            }
        optimizer_result = {
            "winning_configuration": winner["selection_values"],
            "winning_run": exploratory_results[winner["run_id"]],
            "seed_results": seed_results,
            "completed_seed_count": len(values),
            "mean": statistics.mean(values) if values else None,
            "sample_standard_deviation": (
                statistics.stdev(values) if len(values) >= 2 else None
            ),
            "run_metric_aggregates": metric_aggregates,
        }
        if "static_multiplier_configuration" in winner:
            optimizer_result["static_multiplier_configuration"] = copy.deepcopy(
                winner["static_multiplier_configuration"]
            )
        report["optimizer_results"][optimizer_name] = optimizer_result
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("manifest")
    select_parser.add_argument("--report", required=True)
    select_parser.add_argument("--confirmation-manifest", required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("exploratory_manifest")
    report_parser.add_argument("confirmation_manifest")
    report_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "select":
        manifest = load_manifest(args.manifest)
        report = selection_report(manifest)
        confirmation = generate_confirmation_manifest(
            manifest, report["winners"]
        )
        _write_json(args.report, report)
        _write_json(args.confirmation_manifest, confirmation)
        return 0

    exploratory = load_manifest(args.exploratory_manifest)
    confirmation = load_manifest(args.confirmation_manifest)
    _write_json(args.output, final_report(exploratory, confirmation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
