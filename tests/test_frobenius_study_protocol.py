import copy
import importlib.util
import json
import os
import unittest

from shared_utils.experiment_manifest import resolve_run_config
from shared_utils.experiment_results import generate_confirmation_manifest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
DESIGN_PATH = os.path.join(
    REPOSITORY_ROOT,
    "frobenius_normalized_sgdm",
    "task_3_4_candidate_design.json",
)
BASELINE_PATH = os.path.join(
    REPOSITORY_ROOT,
    "experiment_manifests",
    "task_1_6_exploratory.json",
)
STUDY_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT,
    "frobenius_normalized_sgdm",
    "utils",
    "study_manifest.py",
)


def _load_materializer():
    specification = importlib.util.spec_from_file_location(
        "frobenius_study_manifest", STUDY_MANIFEST_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.materialize_manifest


materialize_manifest = _load_materializer()


def read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


class FrobeniusStudyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = read_json(DESIGN_PATH)
        cls.baseline = read_json(BASELINE_PATH)
        cls.manifest = materialize_manifest(cls.design, cls.baseline)

    def test_design_freezes_exact_deterministic_twelve_candidate_grid(self):
        self.assertEqual(self.design["expected_run_count"], 12)
        self.assertEqual(len(self.manifest["runs"]), 12)
        expected = [
            (learning_rate, momentum)
            for learning_rate in (0.001, 0.003, 0.01, 0.03)
            for momentum in (0.9, 0.95, 0.99)
        ]
        actual = [
            (
                run["selection_values"]["frobenius_learning_rate"],
                run["selection_values"]["momentum"],
            )
            for run in self.manifest["runs"]
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            materialize_manifest(self.design, self.baseline), self.manifest
        )

    def test_normalization_settings_and_names_are_frozen(self):
        rule_id = "frobsqrtr-eps1em12-v1"
        for run in self.manifest["runs"]:
            self.assertEqual(run["overrides"]["frobenius_epsilon"], 1e-12)
            self.assertEqual(run["overrides"]["frobenius_shape_factor"], 1.0)
            self.assertEqual(
                run["selection_values"]["normalization_rule_id"], rule_id
            )
            self.assertIn("frobenius_normalized_sgdm_gpt2-124m", run["run_name"])
            self.assertIn(f"seed{run['seed']}", run["run_name"])
            self.assertIn("wd0.1", run["run_name"])
            self.assertIn(rule_id, run["run_name"])

    def test_budget_selection_and_authorization_are_frozen(self):
        self.assertFalse(self.manifest["launch_authorized"])
        self.assertTrue(self.manifest["counts_toward_study_budget"])
        self.assertEqual(
            self.manifest["selection"]["tie_break"],
            ["frobenius_learning_rate", "momentum"],
        )
        self.assertEqual(
            self.manifest["confirmation"]["additional_seeds"], [2027, 4099]
        )
        for run in self.manifest["runs"]:
            self.assertEqual(run["seed"], 1337)
            self.assertEqual(run["max_iters"], 999)
            self.assertEqual(run["selection_step"], 999)
            self.assertEqual(run["selection_tokens"], 491_028_480)
            self.assertEqual(run["max_processed_tokens"], 491_520_000)
            self.assertEqual(run["evaluation_steps"], [0, 333, 666, 999])

    def test_common_controls_resolve_to_task_1_6_values(self):
        candidate_values = resolve_run_config(
            self.manifest, self.manifest["runs"][0]
        )
        global_run = next(
            run
            for run in self.baseline["runs"]
            if run["run_id"] == "global_sgdm_lr0.03_mom0.99"
        )
        baseline_values = resolve_run_config(self.baseline, global_run)
        for key in self.design["matched_control_keys"]:
            self.assertEqual(candidate_values[key], baseline_values[key], key)
        self.assertEqual(self.manifest["dataset"], self.baseline["dataset"])
        self.assertEqual(self.manifest["resources"], self.baseline["resources"])
        self.assertEqual(
            self.manifest["environment_lock"], self.baseline["environment_lock"]
        )
        self.assertFalse(candidate_values["diagnostics_enabled"])

    def test_comparator_artifacts_and_static_mapping_are_locked(self):
        locks = self.manifest["comparator_locks"]
        self.assertEqual(
            locks["global_sgdm"]["run_id"], "global_sgdm_lr0.03_mom0.99"
        )
        static = locks["static_per_matrix_sgdm"]
        self.assertEqual(static["mapping_id"], "2633929b6baf")
        self.assertEqual(
            static["mapping_fingerprint_sha256"],
            "2633929b6bafc6556341f36a0c5318003a647fe2795298f030b9e26330c6b600",
        )
        changed = copy.deepcopy(self.design)
        changed["comparators"]["global_sgdm"]["run_id"] = "different"
        with self.assertRaisesRegex(ValueError, "winner run_id changed"):
            materialize_manifest(changed, self.baseline)
        changed = copy.deepcopy(self.design)
        changed["shared_code_audit"]["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "audit artifact changed"):
            materialize_manifest(changed, self.baseline)

    def test_confirmation_preserves_winner_and_is_unauthorized(self):
        winning_run = self.manifest["runs"][4]
        winner = {
            "run_id": winning_run["run_id"],
            "status": "completed",
            "selection_values": copy.deepcopy(winning_run["selection_values"]),
        }
        confirmation = generate_confirmation_manifest(
            self.manifest, {"frobenius_normalized_sgdm": winner}
        )
        self.assertFalse(confirmation["launch_authorized"])
        self.assertEqual(confirmation["expected_run_count"], 2)
        self.assertEqual([run["seed"] for run in confirmation["runs"]], [2027, 4099])
        for run in confirmation["runs"]:
            self.assertEqual(run["overrides"], winning_run["overrides"])
            self.assertEqual(
                run["selection_values"], winning_run["selection_values"]
            )
            self.assertIn("frobsqrtr-eps1em12-v1", run["run_name"])

    def test_resume_and_claim_gates_are_explicit(self):
        policy = self.manifest["outcome_policy"]
        self.assertEqual(policy["selection_eligibility"], "completed_uninterrupted_only")
        self.assertIn("resumed", policy["record"])
        self.assertIn("numerically_unstable", policy["record"])
        self.assertTrue(self.manifest["shared_code_audit"]["required_before_launch"])
        self.assertEqual(
            self.manifest["claim_gate"]["required_successful_matched_seeds_per_family"],
            3,
        )
        self.assertTrue(self.manifest["claim_gate"]["h3_blocked_until_gate_passes"])


if __name__ == "__main__":
    unittest.main()
