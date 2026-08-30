import copy
import importlib.util
import json
import os
import tempfile
import unittest

from shared_utils.experiment_manifest import (
    _classify_completed_process,
    load_manifest,
    resolve_run_config,
)
from shared_utils.experiment_results import (
    generate_confirmation_manifest,
    select_winners,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
GLOBAL_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT, "experiment_manifests", "task_1_6_exploratory.json"
)
STATIC_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT,
    "experiment_manifests",
    "task_2_5_static_exploratory.json",
)
FROBENIUS_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT,
    "experiment_manifests",
    "task_3_4_frobenius_exploratory.json",
)
SMOKE_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT,
    "experiment_manifests",
    "task_3_5_frobenius_smoke.json",
)
STUDY_RESULTS_PATH = os.path.join(
    REPOSITORY_ROOT,
    "frobenius_normalized_sgdm",
    "utils",
    "study_results.py",
)


def read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def load_study_results_module():
    specification = importlib.util.spec_from_file_location(
        "frobenius_study_results", STUDY_RESULTS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


STUDY_RESULTS = load_study_results_module()


def write_runtime_lock(output_directory):
    os.makedirs(output_directory, exist_ok=True)
    with open(
        os.path.join(output_directory, "resolved_run.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "environment": {
                    "sha256": "locked-environment",
                    "allocated_gpu_names": [
                        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    ],
                },
                "dataset": {
                    "files": {
                        "train.bin": {"sha256": "train"},
                        "val.bin": {"sha256": "val"},
                    }
                },
            },
            output_file,
        )


def write_completed_run(output_root, run, validation_loss):
    output_directory = os.path.join(output_root, run["run_name"])
    write_runtime_lock(output_directory)
    with open(
        os.path.join(output_directory, "outcome.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump({"status": "completed", "resumed": False}, output_file)
    with open(
        os.path.join(output_directory, "evaluation_metrics.jsonl"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "step": 999,
                "processed_tokens": 491_028_480,
                "validation_loss": validation_loss,
            },
            output_file,
        )
        output_file.write("\n")
    with open(
        os.path.join(output_directory, "run_summary.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "metrics": {
                    "latest_validation_loss": validation_loss,
                    "gradient_clipping_frequency": 0.5,
                    "total_wall_time_seconds": 10.0,
                    "peak_gpu_memory_bytes": 1000,
                    "numerical_event_count": 0,
                    "numerical_status": "ok",
                    "divergence_status": "not_observed",
                }
            },
            output_file,
        )


def frobenius_winner(run, output_directory, validation_loss=5.0):
    return {
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "optimizer_name": run["optimizer_name"],
        "seed": run["seed"],
        "status": "completed",
        "resumed": False,
        "selection_values": copy.deepcopy(run["selection_values"]),
        "selection_record": {
            "step": 999,
            "processed_tokens": 491_028_480,
            "train_loss": validation_loss - 0.01,
            "validation_loss": validation_loss,
            "matrix_learning_rate": 0.001,
            "auxiliary_learning_rate": 0.00006,
        },
        "run_summary": {
            "git_commit": "frobenius-commit",
            "metrics": {
                "gradient_clipping_frequency": 0.5,
                "total_wall_time_seconds": 10.0,
                "peak_gpu_memory_bytes": 1000,
                "numerical_event_count": 0,
                "numerical_status": "ok",
                "divergence_status": "not_observed",
            },
        },
        "output_directory": output_directory,
    }


def write_valid_smoke(smoke_manifest, output_root):
    manifest = copy.deepcopy(smoke_manifest)
    manifest["output_root"] = output_root
    run = manifest["runs"][0]
    output_directory = os.path.join(output_root, run["run_name"])
    os.makedirs(output_directory)
    resolved_config = resolve_run_config(manifest, run)
    environment_sha256 = STUDY_RESULTS.sha256_file(
        os.path.join(REPOSITORY_ROOT, manifest["environment_lock"])
    )
    dataset_files = read_json(
        os.path.join(REPOSITORY_ROOT, manifest["dataset"]["lock_file"])
    )["files"]
    with open(
        os.path.join(output_directory, "outcome.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump({"status": "completed", "resumed": False}, output_file)
    with open(
        os.path.join(output_directory, "resolved_run.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "manifest_id": manifest["manifest_id"],
                "run": run,
                "resolved_config": resolved_config,
                "environment": {
                    "sha256": environment_sha256,
                    "allocated_gpu_names": [
                        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    ],
                },
                "dataset": {
                    "files": dataset_files,
                },
            },
            output_file,
        )
    with open(
        os.path.join(output_directory, "run_summary.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "optimizer": {
                    "name": "frobenius_normalized_sgdm",
                    "settings": {
                        "normalization_version": "additive_epsilon_v1",
                        "normalization_equation": (
                            "q*sqrt(min(d_out,d_in))*M/"
                            "(frobenius_norm(M)+epsilon)"
                        ),
                        "matrix_momentum": 0.95,
                        "frobenius_learning_rate": 0.01,
                        "frobenius_epsilon": 1e-12,
                        "frobenius_shape_factor": 1.0,
                    },
                },
                "metrics": {
                    "numerical_status": "ok",
                    "divergence_status": "not_observed",
                },
            },
            output_file,
        )
    with open(
        os.path.join(output_directory, "evaluation_metrics.jsonl"),
        "w",
        encoding="utf-8",
    ) as output_file:
        for step in (0, 1):
            output_file.write(json.dumps({"step": step}) + "\n")
    names = [f"matrix.{index}.weight" for index in range(48)]
    with open(
        os.path.join(output_directory, "optimizer_diagnostics.jsonl"),
        "w",
        encoding="utf-8",
    ) as output_file:
        for step in (0, 1):
            matrices = [
                {
                    "name": name,
                    "momentum_frobenius_norm": float(index + step),
                    "epsilon_dominated": index == 0 and step == 0,
                    "zero_momentum": index == 0 and step == 0,
                }
                for index, name in enumerate(names)
            ]
            output_file.write(
                json.dumps({"step": step, "matrices": matrices}) + "\n"
            )
    with open(os.path.join(output_directory, "ckpt.pt"), "wb") as output_file:
        output_file.write(b"checkpoint")
    return manifest


class FrobeniusStudyExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.global_manifest = load_manifest(GLOBAL_MANIFEST_PATH)
        cls.static_manifest = load_manifest(STATIC_MANIFEST_PATH)
        cls.frobenius_manifest = load_manifest(FROBENIUS_MANIFEST_PATH)
        cls.smoke_manifest = load_manifest(SMOKE_MANIFEST_PATH)
        cls.global_selection = read_json(
            os.path.join(REPOSITORY_ROOT, "reports", "task_1_6_selection.json")
        )
        cls.static_selection = read_json(
            os.path.join(
                REPOSITORY_ROOT, "reports", "task_2_5_static_selection.json"
            )
        )

    def test_smoke_is_short_representative_nonbudget_and_unauthorized(self):
        self.assertFalse(self.smoke_manifest["launch_authorized"])
        self.assertFalse(self.smoke_manifest["counts_toward_study_budget"])
        self.assertEqual(self.smoke_manifest["expected_run_count"], 1)
        run = self.smoke_manifest["runs"][0]
        self.assertEqual(run["max_iters"], 1)
        self.assertEqual(run["evaluation_steps"], [0, 1])
        self.assertEqual(run["tokens_per_update"], 491_520)
        path = os.path.join(REPOSITORY_ROOT, self.smoke_manifest["config_file"])
        namespace = {"compile": True}
        with open(path, encoding="utf-8") as config_file:
            exec(config_file.read(), namespace)
        self.assertEqual(namespace["optimizer_name"], "frobenius_normalized_sgdm")
        self.assertEqual(namespace["frobenius_learning_rate"], 0.01)
        self.assertEqual(namespace["matrix_momentum"], 0.95)
        self.assertEqual(namespace["frobenius_epsilon"], 1e-12)
        self.assertEqual(namespace["frobenius_shape_factor"], 1.0)
        self.assertTrue(namespace["diagnostics_enabled"])
        self.assertEqual(namespace["diagnostic_steps"], "0,1")

    def test_frozen_exploratory_study_remains_unauthorized(self):
        self.assertFalse(self.frobenius_manifest["launch_authorized"])
        self.assertEqual(len(self.frobenius_manifest["runs"]), 12)

    def test_smoke_normalization_evidence_is_validated_and_summarized(self):
        with tempfile.TemporaryDirectory() as output_root:
            manifest = write_valid_smoke(self.smoke_manifest, output_root)
            evidence = STUDY_RESULTS.smoke_normalization_evidence(manifest)
        self.assertEqual(evidence["validation_status"], "verified")
        self.assertEqual(evidence["matrix_record_count"], 96)
        self.assertEqual(evidence["epsilon_dominated_event_count"], 1)
        self.assertEqual(evidence["zero_momentum_event_count"], 1)
        self.assertEqual(evidence["momentum_frobenius_norm_min"], 0.0)
        self.assertEqual(evidence["momentum_frobenius_norm_max"], 48.0)
        self.assertIn("not a broad-study outcome", evidence["scope"])

    def test_smoke_evidence_rejects_incomplete_matrix_diagnostics(self):
        with tempfile.TemporaryDirectory() as output_root:
            manifest = write_valid_smoke(self.smoke_manifest, output_root)
            run = manifest["runs"][0]
            path = os.path.join(
                output_root, run["run_name"], "optimizer_diagnostics.jsonl"
            )
            with open(path, encoding="utf-8") as input_file:
                records = [json.loads(line) for line in input_file]
            records[0]["matrices"].pop()
            with open(path, "w", encoding="utf-8") as output_file:
                for record in records:
                    output_file.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "all 48 eligible matrices"):
                STUDY_RESULTS.smoke_normalization_evidence(manifest)

    def test_resumed_completed_run_is_selection_ineligible(self):
        manifest = copy.deepcopy(self.frobenius_manifest)
        manifest["runs"] = manifest["runs"][:2]
        manifest["expected_run_count"] = 2
        with tempfile.TemporaryDirectory() as output_root:
            manifest["output_root"] = output_root
            for index, run in enumerate(manifest["runs"]):
                output_directory = os.path.join(output_root, run["run_name"])
                os.makedirs(output_directory)
                with open(
                    os.path.join(output_directory, "outcome.json"),
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    json.dump(
                        {"status": "completed", "resumed": index == 0},
                        output_file,
                    )
                with open(
                    os.path.join(output_directory, "evaluation_metrics.jsonl"),
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    json.dump(
                        {
                            "step": 999,
                            "processed_tokens": 491_028_480,
                            "validation_loss": 1.0 + index,
                        },
                        output_file,
                    )
                    output_file.write("\n")
            winner = select_winners(manifest)["frobenius_normalized_sgdm"]
        self.assertEqual(winner["run_id"], manifest["runs"][1]["run_id"])
        self.assertFalse(winner["resumed"])

    def test_numerical_event_gets_distinct_final_outcome(self):
        run = self.frobenius_manifest["runs"][0]
        with tempfile.TemporaryDirectory() as output_directory:
            with open(
                os.path.join(output_directory, "run_summary.json"),
                "w",
                encoding="utf-8",
            ) as output_file:
                json.dump(
                    {
                        "metrics": {
                            "divergence_status": "not_observed",
                            "numerical_status": "nonfinite_gradient_norm",
                        }
                    },
                    output_file,
                )
            self.assertEqual(
                _classify_completed_process(output_directory, run, 0),
                "numerically_unstable",
            )

    def test_confirmation_preserves_frobenius_protocol_metadata(self):
        winning_run = self.frobenius_manifest["runs"][3]
        confirmation = generate_confirmation_manifest(
            self.frobenius_manifest,
            {"frobenius_normalized_sgdm": {"run_id": winning_run["run_id"]}},
        )
        self.assertFalse(confirmation["launch_authorized"])
        self.assertEqual(
            confirmation["source_exploratory_manifest_id"],
            self.frobenius_manifest["manifest_id"],
        )
        for key in (
            "candidate_design",
            "comparator_locks",
            "shared_code_audit",
            "claim_gate",
            "outcome_policy",
        ):
            self.assertEqual(confirmation[key], self.frobenius_manifest[key])
        self.assertEqual(
            {run["seed"] for run in confirmation["runs"]}, {2027, 4099}
        )
        for run in confirmation["runs"]:
            self.assertEqual(run["overrides"], winning_run["overrides"])
            self.assertEqual(
                run["selection_values"], winning_run["selection_values"]
            )

    def test_seed_1337_three_family_comparison_verifies_locks_and_controls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            smoke_manifest = write_valid_smoke(
                self.smoke_manifest,
                os.path.join(temporary_directory, "smoke"),
            )
            directories = {
                family: os.path.join(temporary_directory, family)
                for family in STUDY_RESULTS.FAMILIES
            }
            for output_directory in directories.values():
                write_runtime_lock(output_directory)
            global_selection = copy.deepcopy(self.global_selection)
            static_selection = copy.deepcopy(self.static_selection)
            global_selection["winners"]["global_sgdm"]["output_directory"] = (
                directories["global_sgdm"]
            )
            static_selection["winners"]["static_per_matrix_sgdm"][
                "output_directory"
            ] = directories["static_per_matrix_sgdm"]
            frobenius_run = self.frobenius_manifest["runs"][0]
            frobenius_selection = {
                "manifest_id": self.frobenius_manifest["manifest_id"],
                "winners": {
                    "frobenius_normalized_sgdm": frobenius_winner(
                        frobenius_run,
                        directories["frobenius_normalized_sgdm"],
                        validation_loss=5.2,
                    )
                },
            }
            report = STUDY_RESULTS.exploratory_comparison(
                self.global_manifest,
                global_selection,
                self.static_manifest,
                static_selection,
                self.frobenius_manifest,
                frobenius_selection,
                smoke_manifest,
            )
        self.assertEqual(
            report["report_type"], "exploratory_seed_1337_three_family"
        )
        self.assertEqual(report["matched_controls"]["status"], "verified")
        self.assertIn("no final H3 claim", report["claim_status"])
        measurements = report["observed_measurements"]
        self.assertEqual(
            measurements["static_per_matrix_sgdm"][
                "static_multiplier_configuration"
            ]["fingerprint_sha256"],
            "2633929b6bafc6556341f36a0c5318003a647fe2795298f030b9e26330c6b600",
        )
        self.assertEqual(
            measurements["frobenius_normalized_sgdm"]["normalization"][
                "rule_id"
            ],
            "frobsqrtr-eps1em12-v1",
        )
        self.assertFalse(
            measurements["frobenius_normalized_sgdm"]["normalization_events"][
                "collected"
            ]
        )
        self.assertEqual(
            report["mechanical_smoke_normalization_evidence"][
                "epsilon_dominated_event_count"
            ],
            1,
        )

    def test_confirmed_report_requires_and_aggregates_three_matched_seeds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            smoke_manifest = write_valid_smoke(
                self.smoke_manifest,
                os.path.join(temporary_directory, "smoke"),
            )
            selected_directories = {
                family: os.path.join(temporary_directory, "selected", family)
                for family in STUDY_RESULTS.FAMILIES
            }
            for output_directory in selected_directories.values():
                write_runtime_lock(output_directory)
            global_selection = copy.deepcopy(self.global_selection)
            static_selection = copy.deepcopy(self.static_selection)
            global_winner = global_selection["winners"]["global_sgdm"]
            static_winner = static_selection["winners"]["static_per_matrix_sgdm"]
            global_winner["output_directory"] = selected_directories["global_sgdm"]
            static_winner["output_directory"] = selected_directories[
                "static_per_matrix_sgdm"
            ]
            frobenius_run = self.frobenius_manifest["runs"][0]
            frobenius_selected = frobenius_winner(
                frobenius_run,
                selected_directories["frobenius_normalized_sgdm"],
                validation_loss=5.2,
            )
            frobenius_selection = {
                "manifest_id": self.frobenius_manifest["manifest_id"],
                "winners": {
                    "frobenius_normalized_sgdm": frobenius_selected
                },
            }

            confirmation_manifests = {
                "global_sgdm": generate_confirmation_manifest(
                    self.global_manifest,
                    {"global_sgdm": {"run_id": global_winner["run_id"]}},
                ),
                "static_per_matrix_sgdm": generate_confirmation_manifest(
                    self.static_manifest,
                    {
                        "static_per_matrix_sgdm": {
                            "run_id": static_winner["run_id"]
                        }
                    },
                ),
                "frobenius_normalized_sgdm": generate_confirmation_manifest(
                    self.frobenius_manifest,
                    {
                        "frobenius_normalized_sgdm": {
                            "run_id": frobenius_run["run_id"]
                        }
                    },
                ),
            }
            for family_index, (family, manifest) in enumerate(
                confirmation_manifests.items()
            ):
                manifest["output_root"] = os.path.join(
                    temporary_directory, "confirmation", family
                )
                for seed_index, run in enumerate(manifest["runs"]):
                    write_completed_run(
                        manifest["output_root"],
                        run,
                        5.0 + family_index * 0.1 + seed_index * 0.01,
                    )

            report = STUDY_RESULTS.confirmed_comparison(
                self.global_manifest,
                global_selection,
                self.static_manifest,
                static_selection,
                self.frobenius_manifest,
                frobenius_selection,
                smoke_manifest,
                confirmation_manifests,
            )
        self.assertTrue(report["claim_gate_passed"])
        self.assertEqual(report["unsuccessful_or_ineligible_runs"], [])
        for family in STUDY_RESULTS.FAMILIES:
            family_report = report["optimizer_results"][family]
            self.assertEqual(family_report["completed_matched_seed_count"], 3)
            self.assertIsNotNone(
                family_report["validation_loss_sample_standard_deviation"]
            )
            self.assertIn(
                "gradient_clipping_frequency",
                family_report["run_metric_aggregates"],
            )


if __name__ == "__main__":
    unittest.main()
