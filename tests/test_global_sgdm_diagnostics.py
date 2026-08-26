import json
import tempfile
import unittest
from unittest import mock

import torch

from shared_utils.parameter_partition import NamedParameter
from tuned_global_sgdm.utils.diagnostics import (
    GlobalSGDMDiagnostics,
    append_diagnostic_record,
    initialize_diagnostic_log,
    parse_diagnostic_matrix_names,
    parse_diagnostic_steps,
)
from tuned_global_sgdm.utils.global_sgdm import GlobalSGDM


def frobenius_norm(tensor):
    return torch.linalg.vector_norm(tensor.float()).item()


class GlobalSGDMDiagnosticsTests(unittest.TestCase):
    def test_reported_values_match_direct_tensor_calculations(self):
        name = 'transformer.h.0.attn.c_attn.weight'
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )
        gradient = torch.tensor(
            [[0.2, -0.4], [0.6, -0.8]], dtype=torch.float64
        )
        optimizer = GlobalSGDM(
            [parameter], lr=0.2, momentum=0.5, weight_decay=0.1
        )
        diagnostics = GlobalSGDMDiagnostics(
            enabled=True,
            steps=(3,),
            spectral_matrix_names=(name,),
            epsilon=1e-12,
        )
        before = parameter.detach().clone()
        parameter.grad = gradient.clone()

        context = diagnostics.begin_step(
            3, (NamedParameter(name, parameter),), optimizer
        )
        optimizer.step()
        result = diagnostics.end_step(context, optimizer)
        record = result['matrices'][0]

        expected_momentum = 0.5 * gradient
        expected_after = before * (1.0 - 0.2 * 0.1) - 0.2 * expected_momentum
        expected_update = expected_after - before
        expected_decay_update = -0.2 * 0.1 * before
        torch.testing.assert_close(parameter.detach(), expected_after)
        self.assertAlmostEqual(
            record['weight_frobenius_norm'], frobenius_norm(before)
        )
        self.assertAlmostEqual(
            record['gradient_frobenius_norm'], frobenius_norm(gradient)
        )
        self.assertAlmostEqual(
            record['momentum_frobenius_norm'],
            frobenius_norm(expected_momentum),
        )
        self.assertAlmostEqual(
            record['applied_update_frobenius_norm'],
            frobenius_norm(expected_update),
        )
        self.assertAlmostEqual(
            record['weight_decay_update_frobenius_norm'],
            frobenius_norm(expected_decay_update),
        )
        self.assertAlmostEqual(
            record['update_to_weight_ratio'],
            frobenius_norm(expected_update)
            / (frobenius_norm(before) + 1e-12),
        )
        self.assertAlmostEqual(
            record['applied_update_spectral_norm'],
            torch.linalg.matrix_norm(expected_update.float(), ord=2).item(),
        )
        self.assertEqual(record['scheduled_learning_rate'], 0.2)

    def test_actual_update_norm_comes_from_observed_parameter_change(self):
        name = 'transformer.h.0.mlp.c_fc.weight'
        parameter = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float64))
        optimizer = GlobalSGDM(
            [parameter], lr=0.1, momentum=0.25, weight_decay=0.0
        )
        diagnostics = GlobalSGDMDiagnostics(enabled=True, steps=(0,))
        before = parameter.detach().clone()
        parameter.grad = torch.full_like(parameter, 0.5)

        context = diagnostics.begin_step(
            0, (NamedParameter(name, parameter),), optimizer
        )
        optimizer.step()
        record = diagnostics.end_step(context, optimizer)['matrices'][0]

        observed_update = parameter.detach() - before
        self.assertAlmostEqual(
            record['applied_update_frobenius_norm'],
            frobenius_norm(observed_update),
        )

    def test_spectral_norm_is_only_computed_for_selected_names(self):
        selected_name = 'transformer.h.0.attn.c_proj.weight'
        ordinary_name = 'transformer.h.0.mlp.c_proj.weight'
        selected = torch.nn.Parameter(torch.ones((2, 2)))
        ordinary = torch.nn.Parameter(torch.ones((2, 2)))
        optimizer = GlobalSGDM(
            [selected, ordinary], lr=0.1, momentum=0.0
        )
        diagnostics = GlobalSGDMDiagnostics(
            enabled=True,
            steps=(1,),
            spectral_matrix_names=(selected_name,),
        )
        selected.grad = torch.ones_like(selected)
        ordinary.grad = torch.ones_like(ordinary)
        named_parameters = (
            NamedParameter(selected_name, selected),
            NamedParameter(ordinary_name, ordinary),
        )

        context = diagnostics.begin_step(1, named_parameters, optimizer)
        optimizer.step()
        records = {
            record['name']: record
            for record in diagnostics.end_step(context, optimizer)['matrices']
        }

        self.assertIn('applied_update_spectral_norm', records[selected_name])
        self.assertNotIn('applied_update_spectral_norm', records[ordinary_name])

    def test_disabled_and_unselected_steps_do_no_norm_work(self):
        parameter = torch.nn.Parameter(torch.ones((2, 2)))
        parameter.grad = torch.ones_like(parameter)
        optimizer = GlobalSGDM([parameter], lr=0.1, momentum=0.0)
        named_parameters = (NamedParameter('matrix.weight', parameter),)

        with mock.patch(
            'tuned_global_sgdm.utils.diagnostics._frobenius_norm',
            side_effect=AssertionError('norm should not run'),
        ):
            disabled = GlobalSGDMDiagnostics(enabled=False, steps=(0,))
            self.assertIsNone(
                disabled.begin_step(0, named_parameters, optimizer)
            )
            selected_elsewhere = GlobalSGDMDiagnostics(
                enabled=True, steps=(5,)
            )
            self.assertIsNone(
                selected_elsewhere.begin_step(0, named_parameters, optimizer)
            )

    def test_collection_does_not_change_optimizer_results(self):
        baseline = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        measured = torch.nn.Parameter(baseline.detach().clone())
        baseline_optimizer = GlobalSGDM(
            [baseline], lr=0.1, momentum=0.8, weight_decay=0.01
        )
        measured_optimizer = GlobalSGDM(
            [measured], lr=0.1, momentum=0.8, weight_decay=0.01
        )
        gradient = torch.tensor([[0.5, -0.25], [0.125, -0.75]])
        baseline.grad = gradient.clone()
        measured.grad = gradient.clone()
        diagnostics = GlobalSGDMDiagnostics(enabled=True, steps=(0,))

        baseline_optimizer.step()
        context = diagnostics.begin_step(
            0,
            (NamedParameter('transformer.h.0.attn.c_proj.weight', measured),),
            measured_optimizer,
        )
        measured_optimizer.step()
        diagnostics.end_step(context, measured_optimizer)

        torch.testing.assert_close(baseline, measured)
        torch.testing.assert_close(
            baseline_optimizer.state[baseline]['momentum_buffer'],
            measured_optimizer.state[measured]['momentum_buffer'],
        )

    def test_mismatched_matrix_learning_rates_are_rejected(self):
        first = torch.nn.Parameter(torch.ones((2, 2)))
        second = torch.nn.Parameter(torch.ones((2, 2)))
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        optimizer = GlobalSGDM(
            [
                {'params': [first], 'lr': 0.1},
                {'params': [second], 'lr': 0.2},
            ],
            lr=0.1,
            momentum=0.0,
        )
        diagnostics = GlobalSGDMDiagnostics(enabled=True, steps=(0,))

        with self.assertRaisesRegex(ValueError, 'same global SGDM learning rate'):
            diagnostics.begin_step(
                0,
                (
                    NamedParameter('first.weight', first),
                    NamedParameter('second.weight', second),
                ),
                optimizer,
            )

    def test_configuration_parsing_and_jsonl_output(self):
        self.assertEqual(parse_diagnostic_steps('5, 1,5'), (1, 5))
        self.assertEqual(
            parse_diagnostic_matrix_names('a.weight,b.weight'),
            ('a.weight', 'b.weight'),
        )
        record = {
            'step': 1,
            'matrices': [{'name': 'a.weight', 'applied_update_frobenius_norm': 1.0}],
        }
        with tempfile.TemporaryDirectory() as output_directory:
            initialize_diagnostic_log(output_directory, resume=False)
            append_diagnostic_record(output_directory, record)
            path = f'{output_directory}/optimizer_diagnostics.jsonl'
            with open(path, encoding='utf-8') as input_file:
                self.assertEqual(json.loads(input_file.readline()), record)


if __name__ == '__main__':
    unittest.main()
