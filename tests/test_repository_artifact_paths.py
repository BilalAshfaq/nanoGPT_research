import json
import os
import unittest

from shared_utils.experiment_manifest import (
    repository_root,
    resolve_repository_path,
)
from shared_utils.experiment_results import report_output_path


REPOSITORY_ROOT = repository_root()
MANIFEST_DIRECTORY = os.path.join(REPOSITORY_ROOT, "experiment_manifests")
EXPECTED_RUN_ROOTS = {
    "pilot_global_sgdm.json": "nanogpt-pilot-runs",
    "task_1_6_exploratory.json": "nanogpt-study-runs/task-1.6",
    "task_1_6_confirmation.json": "nanogpt-study-runs/task-1.6",
    "task_2_5_static_smoke.json": "nanogpt-smoke-runs/task-2.5",
    "task_2_5_static_exploratory.json": "nanogpt-study-runs/task-2.5",
    "task_2_5_static_confirmation.json": "nanogpt-study-runs/task-2.5",
}


class RepositoryArtifactPathTests(unittest.TestCase):
    def test_checked_in_run_roots_are_repository_relative(self):
        for filename, expected_root in EXPECTED_RUN_ROOTS.items():
            with self.subTest(filename=filename):
                path = os.path.join(MANIFEST_DIRECTORY, filename)
                with open(path, encoding="utf-8") as manifest_file:
                    manifest = json.load(manifest_file)

                self.assertEqual(manifest["output_root"], expected_root)
                self.assertFalse(os.path.isabs(manifest["output_root"]))
                self.assertEqual(
                    resolve_repository_path(manifest["output_root"]),
                    os.path.join(REPOSITORY_ROOT, *expected_root.split("/")),
                )

    def test_absolute_paths_remain_absolute(self):
        absolute = os.path.abspath(os.path.join("outside", "runs"))
        self.assertEqual(resolve_repository_path(absolute), absolute)

    def test_generated_reports_are_forced_into_root_reports_directory(self):
        expected = os.path.join(
            REPOSITORY_ROOT,
            "reports",
            "task_1_6_selection.json",
        )

        self.assertEqual(
            report_output_path("task_1_6_selection.json"),
            expected,
        )
        self.assertEqual(
            report_output_path("some/other/task_1_6_selection.json"),
            expected,
        )
        self.assertEqual(
            report_output_path(
                os.path.abspath(
                    os.path.join("somewhere-else", "task_1_6_selection.json")
                )
            ),
            expected,
        )

    def test_existing_named_reports_are_in_the_reports_directory(self):
        report_names = (
            "task_1_6_selection.json",
            "task_2_5_seed1337_comparison.json",
            "task_2_5_static_selection.json",
        )
        for report_name in report_names:
            with self.subTest(report_name=report_name):
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(REPOSITORY_ROOT, "reports", report_name)
                    )
                )
                self.assertFalse(
                    os.path.exists(os.path.join(REPOSITORY_ROOT, report_name))
                )

    def test_report_output_requires_a_filename(self):
        for invalid_path in ("", ".", ".."):
            with self.subTest(invalid_path=invalid_path):
                with self.assertRaisesRegex(ValueError, "include a filename"):
                    report_output_path(invalid_path)


if __name__ == "__main__":
    unittest.main()
