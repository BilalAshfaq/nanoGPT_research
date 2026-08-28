import unittest
from unittest import mock

import torch

from shared_utils.parameter_partition import NamedParameter
from shared_utils.sgdm_diagnostics import SGDMUpdateDiagnostics
from static_per_matrix_sgdm.utils.static_per_matrix_sgdm import (
    StaticPerMatrixSGDM,
)


def frobenius_norm(tensor):
    return torch.linalg.vector_norm(tensor.float()).item()


class StaticPerMatrixDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_match_direct_static_update_calculations(self):
        first_name = "transformer.h.0.attn.c_attn.weight"
        second_name = "transformer.h.0.attn.c_proj.weight"
        first = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )
        second = torch.nn.Parameter(
            torch.tensor([[2.0, -1.0], [4.0, -3.0]], dtype=torch.float64)
        )
        first_gradient = torch.tensor(
            [[0.2, -0.4], [0.6, -0.8]], dtype=torch.float64
        )
        second_gradient = torch.tensor(
            [[-0.3, 0.5], [-0.7, 0.9]], dtype=torch.float64
        )
        first.grad = first_gradient.clone()
        second.grad = second_gradient.clone()
        optimizer = StaticPerMatrixSGDM(
            (
                NamedParameter(first_name, first),
                NamedParameter(second_name, second),
            ),
            {first_name: 2.0, second_name: 0.5},
            lr=0.2,
            momentum=0.5,
            weight_decay=0.0,
        )
        diagnostics = SGDMUpdateDiagnostics(enabled=True, steps=(3,))
        named_parameters = (
            NamedParameter(first_name, first),
            NamedParameter(second_name, second),
        )
        before = {
            first_name: first.detach().clone(),
            second_name: second.detach().clone(),
        }

        context = diagnostics.begin_step(3, named_parameters, optimizer)
        optimizer.step()
        records = {
            record["name"]: record
            for record in diagnostics.end_step(context, optimizer)["matrices"]
        }

        for name, parameter, gradient, multiplier in (
            (first_name, first, first_gradient, 2.0),
            (second_name, second, second_gradient, 0.5),
        ):
            momentum = 0.5 * gradient
            update = parameter.detach() - before[name]
            record = records[name]
            self.assertEqual(record["matrix_multiplier"], multiplier)
            self.assertEqual(
                record["effective_matrix_learning_rate"], 0.2 * multiplier
            )
            self.assertAlmostEqual(
                record["momentum_update_frobenius_norm"],
                0.2 * multiplier * frobenius_norm(momentum),
            )
            self.assertAlmostEqual(
                record["applied_update_frobenius_norm"],
                frobenius_norm(update),
            )
            self.assertAlmostEqual(
                record["update_to_weight_ratio"],
                frobenius_norm(update)
                / (frobenius_norm(before[name]) + diagnostics.epsilon),
            )

    def test_disabled_diagnostics_perform_no_norm_work(self):
        parameter = torch.nn.Parameter(torch.ones((2, 2)))
        parameter.grad = torch.ones_like(parameter)
        optimizer = StaticPerMatrixSGDM(
            (NamedParameter("matrix", parameter),),
            {"matrix": 1.0},
            lr=0.1,
            momentum=0.0,
        )

        with mock.patch(
            "shared_utils.sgdm_diagnostics._frobenius_norm",
            side_effect=AssertionError("norm should not run"),
        ):
            diagnostics = SGDMUpdateDiagnostics(enabled=False, steps=(0,))
            self.assertIsNone(
                diagnostics.begin_step(
                    0,
                    (NamedParameter("matrix", parameter),),
                    optimizer,
                )
            )


if __name__ == "__main__":
    unittest.main()
