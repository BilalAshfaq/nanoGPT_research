import copy
import hashlib
import json
import os
import unittest

from shared_utils.experiment_manifest import resolve_run_config, validate_manifest
from shared_utils.experiment_results import generate_confirmation_manifest
from zero_matrix_decay_followup.utils.study_manifest import (
    DESIGN_PATH,
    materialize_manifest,
    verify_protected_artifacts,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT,
    "experiment_manifests",
    "zero_matrix_decay_followup_exploratory.json",
)


def read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ZeroMatrixDecayFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = read_json(DESIGN_PATH)
        cls.manifest = read_json(MANIFEST_PATH)

    def test_original_artifacts_are_hash_locked_and_unchanged(self):
        verify_protected_artifacts(self.design)
        for relative_path, expected_hash in self.design[
            "protected_artifacts"
        ].items():
            path = os.path.join(REPOSITORY_ROOT, *relative_path.split("/"))
            self.assertEqual(sha256(path), expected_hash, relative_path)

    def test_original_baseline_retains_weight_decay_and_followup_sets_zero(self):
        baseline_namespace = {}
        with open(
            os.path.join(REPOSITORY_ROOT, "config", "task_1_6_baseline.py"),
            encoding="utf-8",
        ) as config_file:
            exec(config_file.read(), baseline_namespace)
        followup_namespace = {}
        with open(
            os.path.join(
                REPOSITORY_ROOT, "config", "zero_matrix_decay_followup.py"
            ),
            encoding="utf-8",
        ) as config_file:
            exec(config_file.read(), followup_namespace)

        self.assertEqual(baseline_namespace["matrix_weight_decay"], 0.1)
        self.assertEqual(followup_namespace["matrix_weight_decay"], 0.0)
        self.assertEqual(followup_namespace["auxiliary_weight_decay"], 0.1)
        self.assertEqual(followup_namespace["auxiliary_learning_rate"], 6e-4)
        self.assertEqual(followup_namespace["auxiliary_beta1"], 0.9)
        self.assertEqual(followup_namespace["auxiliary_beta2"], 0.95)

    def test_materialized_manifest_is_frozen_valid_and_unauthorized(self):
        expected = materialize_manifest(self.design)
        self.assertEqual(self.manifest, expected)
        validate_manifest(self.manifest)
        self.assertFalse(self.manifest["launch_authorized"])
        self.assertEqual(self.manifest["expected_run_count"], 36)
        self.assertEqual(
            self.manifest["output_root"],
            "nanogpt-study-runs/zero-matrix-decay-followup-v1",
        )
        self.assertEqual(
            self.manifest["report_names"],
            {
                "selection": "zero_matrix_decay_followup_selection.json",
                "final": "zero_matrix_decay_followup_final.json",
            },
        )

    def test_each_family_has_equal_budget_and_expected_grid(self):
        families = self.design["families"]
        for optimizer_name, family in families.items():
            with self.subTest(optimizer_name=optimizer_name):
                runs = [
                    run
                    for run in self.manifest["runs"]
                    if run["optimizer_name"] == optimizer_name
                ]
                self.assertEqual(len(runs), 12)
                observed = {
                    (
                        run["selection_values"]["learning_rate"],
                        run["selection_values"]["momentum"],
                    )
                    for run in runs
                }
                expected = {
                    (learning_rate, momentum)
                    for learning_rate in family["learning_rates"]
                    for momentum in family["momenta"]
                }
                self.assertEqual(observed, expected)

    def test_every_run_resolves_zero_matrix_decay_and_matched_controls(self):
        reference = resolve_run_config(self.manifest, self.manifest["runs"][0])
        matched_keys = (
            "dataset",
            "tokenizer",
            "gradient_accumulation_steps",
            "batch_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "bias",
            "auxiliary_learning_rate",
            "auxiliary_weight_decay",
            "auxiliary_beta1",
            "auxiliary_beta2",
            "max_iters",
            "decay_lr",
            "warmup_iters",
            "lr_decay_iters",
            "min_lr",
            "grad_clip",
            "eval_interval",
            "eval_iters",
            "dtype",
            "compile",
            "diagnostics_enabled",
        )
        for run in self.manifest["runs"]:
            with self.subTest(run_id=run["run_id"]):
                resolved = resolve_run_config(self.manifest, run)
                self.assertEqual(resolved["matrix_weight_decay"], 0.0)
                self.assertNotIn("matrix_weight_decay", run["overrides"])
                self.assertIn("mwd0", run["run_name"])
                self.assertTrue(run["run_name"].startswith("zero_matrix_decay_"))
                for key in matched_keys:
                    self.assertEqual(resolved[key], reference[key], key)

    def test_static_mapping_is_fixed_and_frobenius_settings_are_fixed(self):
        static_runs = [
            run
            for run in self.manifest["runs"]
            if run["optimizer_name"] == "static_per_matrix_sgdm"
        ]
        fingerprints = {
            run["static_multiplier_configuration"]["fingerprint_sha256"]
            for run in static_runs
        }
        self.assertEqual(
            fingerprints,
            {
                "2633929b6bafc6556341f36a0c5318003a647fe2795298f030b9e26330c6b600"
            },
        )
        frobenius_runs = [
            run
            for run in self.manifest["runs"]
            if run["optimizer_name"] == "frobenius_normalized_sgdm"
        ]
        self.assertEqual(
            {run["overrides"]["frobenius_epsilon"] for run in frobenius_runs},
            {1e-12},
        )
        self.assertEqual(
            {
                run["overrides"]["frobenius_shape_factor"]
                for run in frobenius_runs
            },
            {1.0},
        )

    def test_confirmation_uses_new_names_and_two_additional_seeds_per_family(self):
        winners = {}
        for optimizer_name in self.manifest["selection"]["expected_groups"]:
            run = next(
                run
                for run in self.manifest["runs"]
                if run["optimizer_name"] == optimizer_name
            )
            winners[optimizer_name] = {"run_id": run["run_id"]}
        confirmation = generate_confirmation_manifest(self.manifest, winners)

        self.assertFalse(confirmation["launch_authorized"])
        self.assertEqual(confirmation["expected_run_count"], 6)
        self.assertEqual({run["seed"] for run in confirmation["runs"]}, {2027, 4099})
        self.assertEqual(
            confirmation["output_root"],
            "nanogpt-study-runs/zero-matrix-decay-followup-v1",
        )
        for run in confirmation["runs"]:
            self.assertIn("zero_matrix_decay_", run["run_name"])
            self.assertIn("mwd0", run["run_name"])

    def test_protection_rejects_a_changed_original_hash(self):
        changed = copy.deepcopy(self.design)
        first_path = next(iter(changed["protected_artifacts"]))
        changed["protected_artifacts"][first_path] = "0" * 64
        with self.assertRaisesRegex(ValueError, "protected artifact changed"):
            verify_protected_artifacts(changed)


if __name__ == "__main__":
    unittest.main()
