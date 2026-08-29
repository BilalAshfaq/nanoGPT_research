import json
import os
import tempfile
import unittest

from shared_utils.experiment_manifest import load_manifest
from shared_utils.experiment_results import generate_confirmation_manifest
from static_per_matrix_sgdm.utils.study_results import exploratory_comparison


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
GLOBAL_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT, "experiment_manifests", "task_1_6_exploratory.json"
)
STATIC_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT, "experiment_manifests", "task_2_5_static_exploratory.json"
)
SMOKE_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT, "experiment_manifests", "task_2_5_static_smoke.json"
)


def winner(run, validation_loss, output_directory):
    return {
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "optimizer_name": run["optimizer_name"],
        "seed": run["seed"],
        "status": "completed",
        "selection_values": run["selection_values"],
        "selection_record": {
            "step": 999,
            "processed_tokens": 491_028_480,
            "validation_loss": validation_loss,
        },
        "output_directory": output_directory,
    }


class StaticStudyExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.global_manifest = load_manifest(GLOBAL_MANIFEST_PATH)
        cls.static_manifest = load_manifest(STATIC_MANIFEST_PATH)
        cls.smoke_manifest = load_manifest(SMOKE_MANIFEST_PATH)

    def test_materialized_study_uses_actual_variant_1_winner(self):
        self.assertTrue(self.static_manifest["launch_authorized"])
        self.assertEqual(len(self.static_manifest["runs"]), 12)
        self.assertEqual(
            {run["selection_values"]["momentum"] for run in self.static_manifest["runs"]},
            {0.99},
        )
        self.assertEqual(
            self.static_manifest["variant_1_winner"]["run_id"],
            "global_sgdm_lr0.03_mom0.99",
        )

    def test_real_configuration_smoke_is_short_and_nonbudget(self):
        self.assertTrue(self.smoke_manifest["launch_authorized"])
        self.assertFalse(self.smoke_manifest["counts_toward_study_budget"])
        self.assertEqual(
            self.smoke_manifest["manifest_id"], "task-2.5-static-smoke-v2"
        )
        run = self.smoke_manifest["runs"][0]
        self.assertIn("retry-v2", run["run_name"])
        self.assertEqual(run["max_iters"], 1)
        self.assertEqual(run["tokens_per_update"], 491_520)
        self.assertEqual(run["evaluation_steps"], [0, 1])

    def test_configs_load_when_train_compile_setting_shadows_builtin(self):
        for relative_path in (
            "config/task_2_4_static.py",
            "config/task_2_5_static_smoke.py",
        ):
            with self.subTest(relative_path=relative_path):
                path = os.path.join(REPOSITORY_ROOT, relative_path)
                namespace = {"compile": True}
                with open(path, encoding="utf-8") as config_file:
                    exec(config_file.read(), namespace)
                self.assertTrue(namespace["compile"])
                self.assertEqual(
                    namespace["optimizer_name"], "static_per_matrix_sgdm"
                )

    def test_confirmation_preserves_complete_static_mapping(self):
        winning_run = self.static_manifest["runs"][0]
        confirmation = generate_confirmation_manifest(
            self.static_manifest,
            {"static_per_matrix_sgdm": {"run_id": winning_run["run_id"]}},
        )
        self.assertFalse(confirmation["launch_authorized"])
        self.assertEqual(
            {run["seed"] for run in confirmation["runs"]}, {2027, 4099}
        )
        for run in confirmation["runs"]:
            self.assertEqual(
                run["static_multiplier_configuration"],
                winning_run["static_multiplier_configuration"],
            )

    def test_seed_1337_comparison_is_labeled_exploratory(self):
        global_run = next(
            run
            for run in self.global_manifest["runs"]
            if run["run_id"] == "global_sgdm_lr0.03_mom0.99"
        )
        static_run = self.static_manifest["runs"][0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            global_output = os.path.join(temporary_directory, "global")
            static_output = os.path.join(temporary_directory, "static")
            runtime = {
                "environment": {"sha256": "environment-lock"},
                "dataset": {
                    "files": {
                        "train.bin": {"sha256": "train"},
                        "val.bin": {"sha256": "val"},
                    }
                },
            }
            for output_directory in (global_output, static_output):
                os.makedirs(output_directory)
                with open(
                    os.path.join(output_directory, "resolved_run.json"),
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    json.dump(runtime, output_file)
            report = exploratory_comparison(
                self.global_manifest,
                {
                    "winners": {
                        "global_sgdm": winner(
                            global_run, 5.41, global_output
                        )
                    }
                },
                self.static_manifest,
                {
                    "winners": {
                        "static_per_matrix_sgdm": winner(
                            static_run, 5.2, static_output
                        )
                    }
                },
            )
        self.assertEqual(report["report_type"], "exploratory_seed_1337_only")
        self.assertIn("no final improvement claim", report["claim_status"])
        self.assertEqual(report["matched_controls"]["status"], "verified")
        self.assertAlmostEqual(
            report["observed_measurements"][
                "validation_loss_difference_static_minus_global"
            ],
            -0.21,
        )


if __name__ == "__main__":
    unittest.main()
