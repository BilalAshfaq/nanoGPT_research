import ast
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from shared_utils.run_metadata import (
    OptimizerStepTimer,
    append_evaluation_record,
    build_optimizer_group_signature,
    build_run_metadata,
    get_git_commit_hash,
    get_hardware_metadata,
    get_peak_gpu_memory_metadata,
    initialize_evaluation_log,
    snapshot_run_metadata,
    validate_resume_compatibility,
    write_run_summary,
)


class FakeParameter:
    def __init__(self, shape, requires_grad=True):
        self.shape = shape
        self.requires_grad = requires_grad

    def numel(self):
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count


class FakeModel:
    def __init__(self):
        self.first = FakeParameter((2, 3))
        self.second = FakeParameter((3,))

    def named_parameters(self):
        return iter((('first', self.first), ('second', self.second)))

    def parameters(self):
        return iter((self.first, self.second))


class FakeOptimizer:
    def __init__(self, model):
        self.param_groups = [
            {'params': [model.first], 'optimizer_role': 'matrix'},
            {'params': [model.second], 'optimizer_role': 'auxiliary'},
        ]


class FakeTorch:
    __version__ = 'test-version'

    class cuda:
        @staticmethod
        def is_available():
            return False


class RunMetadataTests(unittest.TestCase):
    def test_group_signature_is_deterministic_and_complete(self):
        model = FakeModel()
        signature = build_optimizer_group_signature(
            model, FakeOptimizer(model)
        )

        self.assertEqual(
            signature,
            [
                {
                    'index': 0,
                    'optimizer_role': 'matrix',
                    'parameters': [{'name': 'first', 'shape': (2, 3)}],
                },
                {
                    'index': 1,
                    'optimizer_role': 'auxiliary',
                    'parameters': [{'name': 'second', 'shape': (3,)}],
                },
            ],
        )

    def test_resume_rejects_optimizer_name_and_group_mismatches(self):
        signature = [{'index': 0, 'parameters': []}]
        checkpoint = {
            'run_metadata': {
                'optimizer': {
                    'name': 'global_sgdm',
                    'group_signature': signature,
                    'settings': {'matrix_momentum': 0.9},
                }
            }
        }
        validate_resume_compatibility(
            checkpoint,
            'global_sgdm',
            signature,
            optimizer_settings={'matrix_momentum': 0.9},
        )

        with self.assertRaisesRegex(ValueError, 'optimizer mismatch'):
            validate_resume_compatibility(checkpoint, 'adamw', signature)
        with self.assertRaisesRegex(ValueError, 'parameter-group structure'):
            validate_resume_compatibility(
                checkpoint,
                'global_sgdm',
                [{'index': 1, 'parameters': []}],
            )
        with self.assertRaisesRegex(ValueError, 'settings do not match'):
            validate_resume_compatibility(
                checkpoint,
                'global_sgdm',
                signature,
                optimizer_settings={'matrix_momentum': 0.8},
            )

    def test_run_metadata_contains_reconstruction_and_progress_fields(self):
        base = build_run_metadata(
            config={'optimizer_name': 'global_sgdm', 'seed': 123},
            git_commit='abc123',
            model_args={'n_layer': 2, 'n_embd': 8},
            trainable_parameter_count=42,
            optimizer_name='global_sgdm',
            optimizer_settings={'matrix_momentum': 0.9},
            optimizer_group_signature=[{'index': 0}],
            parameter_group_audit={'totals': {'parameter_count': 42}},
            seed=123,
            dataset='dataset-id',
            tokenizer='tokenizer-id',
            block_size=16,
            batch_size=2,
            gradient_accumulation_steps=3,
            ddp_world_size=4,
            tokens_per_iteration=384,
            precision={'dtype': 'float32'},
            hardware={'device_type': 'cpu'},
        )
        snapshot = snapshot_run_metadata(
            base,
            progress={'optimizer_steps': 5, 'processed_tokens': 1920},
            metrics={'latest_validation_loss': 2.5},
        )

        self.assertEqual(snapshot['git_commit'], 'abc123')
        self.assertEqual(snapshot['model']['trainable_parameter_count'], 42)
        self.assertEqual(snapshot['optimizer']['name'], 'global_sgdm')
        self.assertEqual(snapshot['random_seed'], 123)
        self.assertEqual(snapshot['data']['effective_batch_size'], 24)
        self.assertEqual(snapshot['data']['tokens_per_iteration'], 384)
        self.assertEqual(snapshot['progress']['processed_tokens'], 1920)
        self.assertEqual(snapshot['metrics']['latest_validation_loss'], 2.5)

    def test_cpu_hardware_and_step_timing_are_recorded(self):
        hardware = get_hardware_metadata(FakeTorch, 'cpu', 'cpu')
        self.assertEqual(hardware['cuda'], 'not_applicable')
        self.assertEqual(hardware['torch_version'], 'test-version')
        self.assertEqual(
            get_peak_gpu_memory_metadata(FakeTorch, 'cpu', 'cpu'),
            {'bytes': None, 'status': 'not_applicable'},
        )

        timer = OptimizerStepTimer(FakeTorch, device_type='cpu')
        timer.start()
        time.sleep(0.001)
        timer.stop()
        self.assertEqual(timer.step_count, 1)
        self.assertGreater(timer.total_seconds, 0.0)
        self.assertGreater(timer.mean_seconds, 0.0)

    def test_git_commit_hash_is_recorded_when_repository_is_available(self):
        repository_root = os.path.dirname(os.path.dirname(__file__))
        commit_hash = get_git_commit_hash(repository_root)
        self.assertEqual(len(commit_hash), 40)
        int(commit_hash, 16)

    def test_run_summary_is_json_serializable(self):
        metadata = {
            'optimizer': {'name': 'adamw'},
            'progress': {'processed_tokens': 16},
        }
        with tempfile.TemporaryDirectory() as output_directory:
            write_run_summary(output_directory, metadata)
            summary_path = os.path.join(output_directory, 'run_summary.json')
            with open(summary_path, encoding='utf-8') as summary_file:
                self.assertEqual(json.load(summary_file), metadata)

    def test_evaluation_log_is_structured_and_resume_appends(self):
        first = {
            'step': 0,
            'processed_tokens': 0,
            'train_loss': 4.0,
            'validation_loss': 4.1,
        }
        second = {
            'step': 10,
            'processed_tokens': 160,
            'train_loss': 3.0,
            'validation_loss': 3.1,
        }
        with tempfile.TemporaryDirectory() as output_directory:
            path = initialize_evaluation_log(output_directory, resume=False)
            append_evaluation_record(output_directory, first)
            initialize_evaluation_log(output_directory, resume=True)
            append_evaluation_record(output_directory, second)

            with open(path, encoding='utf-8') as evaluation_file:
                records = [json.loads(line) for line in evaluation_file]

        self.assertEqual(records, [first, second])

    def test_train_defaults_and_validation_precede_first_batch(self):
        repository_root = os.path.dirname(os.path.dirname(__file__))
        train_path = os.path.join(repository_root, 'train.py')
        with open(train_path, encoding='utf-8') as train_file:
            tree = ast.parse(train_file.read())

        scalar_defaults = {}
        validation_line = None
        first_batch_line = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    scalar_defaults[target.id] = node.value.value
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                function = node.value.func
                if isinstance(function, ast.Name):
                    if function.id == 'validate_optimizer_name':
                        validation_line = node.lineno
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                function = node.value.func
                if isinstance(function, ast.Name) and function.id == 'get_batch':
                    first_batch_line = node.lineno

        self.assertEqual(scalar_defaults['optimizer_name'], 'adamw')
        self.assertEqual(scalar_defaults['seed'], 1337)
        self.assertIsNotNone(validation_line)
        self.assertIsNotNone(first_batch_line)
        self.assertLess(validation_line, first_batch_line)

    def test_omitted_and_explicit_adamw_configuration_resolve_identically(self):
        repository_root = os.path.dirname(os.path.dirname(__file__))
        configurator_path = os.path.join(repository_root, 'configurator.py')
        with open(configurator_path, encoding='utf-8') as configurator_file:
            configurator_code = compile(
                configurator_file.read(), configurator_path, 'exec'
            )

        def resolve(arguments):
            namespace = {'optimizer_name': 'adamw'}
            with mock.patch('sys.argv', ['train.py'] + arguments):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(configurator_code, namespace)
            return namespace['optimizer_name']

        self.assertEqual(resolve([]), 'adamw')
        self.assertEqual(resolve(['--optimizer_name=adamw']), 'adamw')
        self.assertEqual(
            resolve(['--optimizer_name=global_sgdm']), 'global_sgdm'
        )


if __name__ == '__main__':
    unittest.main()
