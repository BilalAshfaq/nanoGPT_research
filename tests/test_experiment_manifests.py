import copy
import json
import os
import tempfile
import unittest

from shared_utils.experiment_manifest import (
    load_manifest,
    run_manifest,
    validate_manifest,
)
from shared_utils.experiment_results import (
    final_report,
    generate_confirmation_manifest,
    select_winners,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
PILOT_MANIFEST = os.path.join(
    REPOSITORY_ROOT,
    'experiment_manifests',
    'pilot_global_sgdm.json',
)
STUDY_MANIFEST = os.path.join(
    REPOSITORY_ROOT,
    'experiment_manifests',
    'task_1_6_exploratory.json',
)


class ExperimentManifestTests(unittest.TestCase):
    def test_checked_in_manifests_are_valid_and_launch_guard_is_enforced(self):
        pilot = load_manifest(PILOT_MANIFEST)
        study = load_manifest(STUDY_MANIFEST)

        self.assertEqual(len(pilot['runs']), 1)
        self.assertEqual(len(study['runs']), 24)
        self.assertTrue(pilot['launch_authorized'])
        self.assertTrue(study['launch_authorized'])
        unauthorized_study = copy.deepcopy(study)
        unauthorized_study['launch_authorized'] = False
        with self.assertRaisesRegex(RuntimeError, 'not authorized'):
            run_manifest(unauthorized_study)

    def test_study_materializes_exact_candidate_grids_and_budgets(self):
        study = load_manifest(STUDY_MANIFEST)
        global_runs = [
            run for run in study['runs']
            if run['optimizer_name'] == 'global_sgdm'
        ]
        adamw_runs = [
            run for run in study['runs']
            if run['optimizer_name'] == 'adamw'
        ]

        self.assertEqual(len(global_runs), 12)
        self.assertEqual(len(adamw_runs), 12)
        self.assertEqual(
            {
                (run['selection_values']['learning_rate'],
                 run['selection_values']['momentum'])
                for run in global_runs
            },
            {
                (learning_rate, momentum)
                for learning_rate in (0.03, 0.1, 0.3, 1.0)
                for momentum in (0.9, 0.95, 0.99)
            },
        )
        self.assertEqual(
            {
                (run['selection_values']['learning_rate'],
                 run['selection_values']['momentum'])
                for run in adamw_runs
            },
            {
                (learning_rate, momentum)
                for learning_rate in (0.0003, 0.0006, 0.001, 0.002)
                for momentum in (0.85, 0.9, 0.95)
            },
        )
        for run in study['runs']:
            self.assertEqual(run['selection_tokens'], 491_028_480)
            self.assertEqual(run['max_processed_tokens'], 491_520_000)
            self.assertEqual(run['evaluation_steps'], [0, 333, 666, 999])

    def test_invalid_budget_is_rejected(self):
        manifest = load_manifest(PILOT_MANIFEST)
        manifest.pop('_manifest_path')
        manifest['runs'][0]['max_processed_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'max_processed_tokens'):
            validate_manifest(manifest)

    def test_manifest_layer_does_not_whitelist_optimizer_names(self):
        manifest = load_manifest(PILOT_MANIFEST)
        manifest.pop('_manifest_path')
        manifest['runs'][0]['optimizer_name'] = 'future_optimizer'
        validate_manifest(manifest)

    def test_selection_and_confirmation_are_deterministic(self):
        manifest = load_manifest(STUDY_MANIFEST)
        manifest = copy.deepcopy(manifest)
        manifest['runs'] = [
            manifest['runs'][0],
            manifest['runs'][1],
            manifest['runs'][12],
            manifest['runs'][15],
        ]
        manifest['expected_run_count'] = 4

        with tempfile.TemporaryDirectory() as output_root:
            manifest['output_root'] = output_root
            losses = {
                manifest['runs'][0]['run_id']: 3.0,
                manifest['runs'][1]['run_id']: 3.0,
                manifest['runs'][2]['run_id']: 3.2,
                manifest['runs'][3]['run_id']: 2.9,
            }
            for run in manifest['runs']:
                output_directory = os.path.join(output_root, run['run_name'])
                os.makedirs(output_directory)
                with open(
                    os.path.join(output_directory, 'outcome.json'),
                    'w',
                    encoding='utf-8',
                ) as outcome_file:
                    json.dump({'status': 'completed'}, outcome_file)
                with open(
                    os.path.join(output_directory, 'evaluation_metrics.jsonl'),
                    'w',
                    encoding='utf-8',
                ) as evaluation_file:
                    json.dump(
                        {
                            'step': 999,
                            'validation_loss': losses[run['run_id']],
                        },
                        evaluation_file,
                    )
                    evaluation_file.write('\n')
                with open(
                    os.path.join(output_directory, 'run_summary.json'),
                    'w',
                    encoding='utf-8',
                ) as summary_file:
                    json.dump(
                        {
                            'metrics': {
                                'latest_validation_loss': losses[run['run_id']],
                                'total_wall_time_seconds': 10.0,
                            }
                        },
                        summary_file,
                    )

            winners = select_winners(manifest)
            self.assertEqual(
                winners['global_sgdm']['run_id'],
                'global_sgdm_lr0.03_mom0.90',
            )
            self.assertEqual(
                winners['adamw']['run_id'],
                'adamw_lr0.0006_beta1_0.85',
            )
            confirmation = generate_confirmation_manifest(manifest, winners)

            for index, run in enumerate(confirmation['runs']):
                output_directory = os.path.join(output_root, run['run_name'])
                os.makedirs(output_directory)
                validation_loss = 2.8 + index * 0.1
                with open(
                    os.path.join(output_directory, 'outcome.json'),
                    'w',
                    encoding='utf-8',
                ) as outcome_file:
                    json.dump({'status': 'completed'}, outcome_file)
                with open(
                    os.path.join(output_directory, 'evaluation_metrics.jsonl'),
                    'w',
                    encoding='utf-8',
                ) as evaluation_file:
                    json.dump(
                        {'step': 999, 'validation_loss': validation_loss},
                        evaluation_file,
                    )
                    evaluation_file.write('\n')
                with open(
                    os.path.join(output_directory, 'run_summary.json'),
                    'w',
                    encoding='utf-8',
                ) as summary_file:
                    json.dump(
                        {
                            'metrics': {
                                'latest_validation_loss': validation_loss,
                                'total_wall_time_seconds': 12.0,
                            }
                        },
                        summary_file,
                    )

            report = final_report(manifest, confirmation)

        self.assertFalse(confirmation['launch_authorized'])
        self.assertEqual(confirmation['expected_run_count'], 4)
        self.assertEqual(
            {run['seed'] for run in confirmation['runs']},
            {2027, 4099},
        )
        self.assertEqual(
            report['optimizer_results']['global_sgdm']['completed_seed_count'],
            3,
        )
        self.assertIsNotNone(
            report['optimizer_results']['global_sgdm'][
                'sample_standard_deviation'
            ]
        )
        self.assertIn(
            'total_wall_time_seconds',
            report['optimizer_results']['adamw']['run_metric_aggregates'],
        )


if __name__ == '__main__':
    unittest.main()
