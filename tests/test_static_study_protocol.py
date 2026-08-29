import copy
import json
import os
import unittest

from shared_utils.experiment_manifest import resolve_run_config
from static_per_matrix_sgdm.utils.study_manifest import materialize_manifest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
DESIGN_PATH = os.path.join(
    REPOSITORY_ROOT,
    "static_per_matrix_sgdm",
    "task_2_4_candidate_design.json",
)
BASELINE_PATH = os.path.join(
    REPOSITORY_ROOT,
    "experiment_manifests",
    "task_1_6_exploratory.json",
)


def read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def selection_report(momentum=0.95):
    return {
        "manifest_id": "task-1.6-exploratory-v1",
        "winners": {
            "global_sgdm": {
                "run_id": f"global_sgdm_winner_mom{momentum}",
                "status": "completed",
                "selection_values": {
                    "learning_rate": 0.1,
                    "momentum": momentum,
                },
            }
        },
    }


class StaticStudyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = read_json(DESIGN_PATH)
        cls.baseline = read_json(BASELINE_PATH)
        cls.manifest = materialize_manifest(
            selection_report(), cls.design, cls.baseline
        )

    def test_design_freezes_exact_twelve_candidate_grid(self):
        self.assertEqual(self.design["expected_run_count"], 12)
        self.assertEqual(len(self.manifest["runs"]), 12)
        self.assertEqual(
            {
                (
                    run["selection_values"]["learning_rate"],
                    run["selection_values"]["mapping_name"],
                )
                for run in self.manifest["runs"]
            },
            {
                (learning_rate, profile["name"])
                for learning_rate in (0.03, 0.1, 0.3, 1.0)
                for profile in self.design["mapping_profiles"]
            },
        )
        self.assertEqual(
            self.manifest["selection"]["tie_break"],
            ["learning_rate", "mapping_order"],
        )

    def test_mappings_and_run_names_are_frozen_and_complete(self):
        expected_fingerprints = {
            profile["fingerprint_sha256"]
            for profile in self.design["mapping_profiles"]
        }
        self.assertEqual(
            {
                run["static_multiplier_configuration"]["fingerprint_sha256"]
                for run in self.manifest["runs"]
            },
            expected_fingerprints,
        )
        for run in self.manifest["runs"]:
            configuration = run["static_multiplier_configuration"]
            self.assertEqual(len(configuration["resolved_multipliers"]), 48)
            self.assertEqual(configuration["geometric_mean"], 1.0)
            self.assertIn(configuration["mapping_id"], run["run_name"])

    def test_variant_1_momentum_is_fixed_not_tuned(self):
        self.assertEqual(
            {run["selection_values"]["momentum"] for run in self.manifest["runs"]},
            {0.95},
        )
        self.assertFalse(self.manifest["launch_authorized"])
        self.assertEqual(
            self.manifest["confirmation"]["additional_seeds"], [2027, 4099]
        )
        with self.assertRaisesRegex(ValueError, "undeclared momentum"):
            materialize_manifest(selection_report(0.8), self.design, self.baseline)

    def test_static_mapping_overrides_resolve_through_shared_manifest(self):
        run = self.manifest["runs"][0]
        resolved = resolve_run_config(self.manifest, run)
        self.assertEqual(
            resolved["static_matrix_type_multipliers"],
            run["overrides"]["static_matrix_type_multipliers"],
        )

    def test_matched_budget_and_schedule_controls(self):
        for run in self.manifest["runs"]:
            self.assertEqual(run["seed"], 1337)
            self.assertEqual(run["selection_tokens"], 491_028_480)
            self.assertEqual(run["max_processed_tokens"], 491_520_000)
            self.assertEqual(run["evaluation_steps"], [0, 333, 666, 999])

        wrong_report = selection_report()
        wrong_report["manifest_id"] = "some-other-study"
        with self.assertRaisesRegex(ValueError, "not from the locked"):
            materialize_manifest(wrong_report, self.design, self.baseline)


if __name__ == "__main__":
    unittest.main()
