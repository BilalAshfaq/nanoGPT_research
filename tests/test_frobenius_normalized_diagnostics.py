import math
import unittest
from unittest import mock

import torch

from frobenius_normalized_sgdm.utils.diagnostics import (
    FrobeniusNormalizedSGDMDiagnostics,
)
from frobenius_normalized_sgdm.utils.frobenius_normalized_sgdm import (
    FrobeniusNormalizedSGDM,
)
from shared_utils.parameter_partition import NamedParameter


def frobenius_norm(tensor):
    return torch.linalg.vector_norm(tensor.detach().float()).item()


def frobenius_cosine(first, second):
    first = first.detach().double()
    second = second.detach().double()
    return (
        torch.sum(first * second)
        / (
            torch.linalg.vector_norm(first)
            * torch.linalg.vector_norm(second)
        )
    ).item()


class FrobeniusNormalizedDiagnosticsTests(unittest.TestCase):
    def test_all_scalars_match_square_tall_wide_zero_and_epsilon_cases(self):
        epsilon = 0.25
        learning_rate = 0.2
        shape_factor = 1.5
        specifications = (
            (
                "square",
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64
                ),
            ),
            (
                "tall",
                torch.tensor(
                    [[0.2, -0.4], [0.6, -0.8], [1.0, -1.2]],
                    dtype=torch.float64,
                ),
            ),
            (
                "wide",
                torch.tensor(
                    [[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]],
                    dtype=torch.float64,
                ),
            ),
            (
                "epsilon_dominated",
                torch.tensor([[0.05, 0.0]], dtype=torch.float64),
            ),
            ("zero", torch.zeros((2, 1), dtype=torch.float64)),
        )
        named_parameters = []
        gradients = {}
        before = {}
        for index, (label, gradient) in enumerate(specifications):
            parameter = torch.nn.Parameter(
                torch.arange(
                    1,
                    gradient.numel() + 1,
                    dtype=torch.float64,
                ).reshape(gradient.shape)
                + index
            )
            parameter.grad = gradient.clone()
            name = f"{label}.weight"
            named_parameters.append(NamedParameter(name, parameter))
            gradients[name] = gradient
            before[name] = parameter.detach().clone()

        optimizer = FrobeniusNormalizedSGDM(
            [item.parameter for item in named_parameters],
            lr=learning_rate,
            momentum=0.0,
            epsilon=epsilon,
            fixed_shape_factor=shape_factor,
            weight_decay=0.0,
        )
        diagnostics = FrobeniusNormalizedSGDMDiagnostics(
            enabled=True,
            steps=(4,),
            epsilon=1e-12,
        )

        context = diagnostics.begin_step(4, named_parameters, optimizer)
        optimizer.step()
        result = diagnostics.end_step(context, optimizer)
        records = {record["name"]: record for record in result["matrices"]}

        for item in named_parameters:
            name = item.name
            gradient = gradients[name]
            record = records[name]
            raw_norm = torch.linalg.vector_norm(gradient).item()
            denominator = raw_norm + epsilon
            retained_rank = min(gradient.shape)
            nominal_target = shape_factor * math.sqrt(retained_rank)
            exact_normalized_norm = (
                nominal_target * raw_norm / denominator
            )
            zero_momentum = raw_norm == 0.0
            normalization_multiplier = (
                0.0 if zero_momentum else nominal_target / denominator
            )
            expected_step_norm = learning_rate * exact_normalized_norm
            observed_delta = item.parameter.detach() - before[name]

            self.assertAlmostEqual(
                record["momentum_frobenius_norm"], raw_norm
            )
            self.assertAlmostEqual(
                record["normalization_denominator"], denominator
            )
            self.assertEqual(record["retained_rank"], retained_rank)
            self.assertEqual(record["frobenius_epsilon"], epsilon)
            self.assertEqual(record["frobenius_shape_factor"], shape_factor)
            self.assertAlmostEqual(
                record["nominal_normalized_matrix_target_norm"],
                nominal_target,
            )
            self.assertAlmostEqual(
                record[
                    "epsilon_adjusted_expected_normalized_matrix_norm"
                ],
                exact_normalized_norm,
            )
            self.assertEqual(
                record["scheduled_frobenius_learning_rate"], learning_rate
            )
            self.assertAlmostEqual(
                record["expected_momentum_derived_step_norm"],
                expected_step_norm,
            )
            self.assertAlmostEqual(
                record["momentum_update_frobenius_norm"],
                expected_step_norm,
            )
            self.assertAlmostEqual(
                record["observed_total_parameter_delta_norm"],
                frobenius_norm(observed_delta),
            )
            self.assertAlmostEqual(
                record["applied_update_frobenius_norm"],
                frobenius_norm(observed_delta),
            )
            self.assertAlmostEqual(
                record["update_to_weight_ratio"],
                frobenius_norm(observed_delta)
                / (frobenius_norm(before[name]) + diagnostics.epsilon),
            )
            self.assertAlmostEqual(
                record["normalization_multiplier"],
                normalization_multiplier,
            )
            self.assertAlmostEqual(
                record["full_effective_multiplier"],
                learning_rate * normalization_multiplier,
            )
            self.assertEqual(record["zero_momentum"], zero_momentum)
            self.assertEqual(
                record["epsilon_dominated"], raw_norm <= epsilon
            )
            self.assertEqual(record["matrix_multiplier"], None)
            self.assertEqual(
                record["weight_decay_update_frobenius_norm"], 0.0
            )
            if zero_momentum:
                self.assertIsNone(
                    record["momentum_update_frobenius_cosine"]
                )
            else:
                self.assertAlmostEqual(
                    record["momentum_update_frobenius_cosine"],
                    -1.0,
                    places=12,
                )
                self.assertAlmostEqual(
                    frobenius_norm(observed_delta),
                    expected_step_norm,
                    places=6,
                )
                self.assertAlmostEqual(
                    frobenius_cosine(observed_delta, gradient),
                    -1.0,
                    places=12,
                )

    def test_weight_decay_and_normalization_contributions_are_separate(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
        )
        gradient = torch.tensor(
            [[0.0, 1.0], [-2.0, 0.0]], dtype=torch.float64
        )
        parameter.grad = gradient.clone()
        before = parameter.detach().clone()
        learning_rate = 0.2
        weight_decay = 0.1
        epsilon = 1e-6
        optimizer = FrobeniusNormalizedSGDM(
            [parameter],
            lr=learning_rate,
            momentum=0.0,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )
        diagnostics = FrobeniusNormalizedSGDMDiagnostics(
            enabled=True, steps=(0,)
        )
        named_parameters = (NamedParameter("matrix.weight", parameter),)

        context = diagnostics.begin_step(0, named_parameters, optimizer)
        optimizer.step()
        record = diagnostics.end_step(context, optimizer)["matrices"][0]

        raw_norm = torch.linalg.vector_norm(gradient)
        normalized = math.sqrt(2.0) * gradient / (raw_norm + epsilon)
        momentum_update = -learning_rate * normalized
        decay_update = -learning_rate * weight_decay * before
        total_update = momentum_update + decay_update
        torch.testing.assert_close(parameter.detach() - before, total_update)
        self.assertAlmostEqual(
            record["expected_momentum_derived_step_norm"],
            frobenius_norm(momentum_update),
        )
        self.assertAlmostEqual(
            record["decoupled_weight_decay_update_frobenius_norm"],
            frobenius_norm(decay_update),
        )
        self.assertAlmostEqual(
            record["observed_total_parameter_delta_norm"],
            frobenius_norm(total_update),
        )
        self.assertNotEqual(
            record["observed_total_parameter_delta_norm"],
            record["expected_momentum_derived_step_norm"],
        )
        self.assertAlmostEqual(
            record["momentum_update_frobenius_cosine"], -1.0, places=12
        )

    def test_disabled_diagnostics_add_only_the_inherent_normalization_norm(self):
        parameter = torch.nn.Parameter(torch.ones((2, 2)))
        parameter.grad = torch.full_like(parameter, 0.5)
        optimizer = FrobeniusNormalizedSGDM(
            [parameter], lr=0.1, momentum=0.0, epsilon=1e-6
        )
        diagnostics = FrobeniusNormalizedSGDMDiagnostics(
            enabled=False, steps=(0,)
        )

        with mock.patch(
            "torch.linalg.vector_norm",
            wraps=torch.linalg.vector_norm,
        ) as vector_norm:
            self.assertIsNone(
                diagnostics.begin_step(
                    0,
                    (NamedParameter("matrix.weight", parameter),),
                    optimizer,
                )
            )
            optimizer.step()

        self.assertEqual(vector_norm.call_count, 1)
        self.assertFalse(optimizer.has_pending_step_diagnostics())

    def test_selected_step_releases_all_optimizer_diagnostic_tensors(self):
        parameter = torch.nn.Parameter(torch.ones((2, 3)))
        parameter.grad = torch.full_like(parameter, 0.5)
        optimizer = FrobeniusNormalizedSGDM(
            [parameter], lr=0.1, momentum=0.0, epsilon=1e-6
        )
        diagnostics = FrobeniusNormalizedSGDMDiagnostics(
            enabled=True, steps=(0,)
        )

        context = diagnostics.begin_step(
            0,
            (NamedParameter("matrix.weight", parameter),),
            optimizer,
        )
        optimizer.step()
        for captured in optimizer._diagnostic_scalars.values():
            for value in captured.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.ndim, 0)
                    self.assertFalse(value.requires_grad)
        result = diagnostics.end_step(context, optimizer)

        self.assertFalse(optimizer.has_pending_step_diagnostics())
        self.assertEqual(optimizer._diagnostic_scalars, {})
        for value in result["matrices"][0].values():
            self.assertFalse(torch.is_tensor(value))

    def test_skipped_step_clears_pending_capture_and_reports_no_candidate(self):
        parameter = torch.nn.Parameter(torch.ones((2, 2)))
        parameter.grad = torch.ones_like(parameter)
        optimizer = FrobeniusNormalizedSGDM(
            [parameter], lr=0.1, momentum=0.0, epsilon=1e-6
        )
        diagnostics = FrobeniusNormalizedSGDMDiagnostics(
            enabled=True, steps=(0,)
        )
        context = diagnostics.begin_step(
            0,
            (NamedParameter("matrix.weight", parameter),),
            optimizer,
        )

        record = diagnostics.end_step(
            context,
            optimizer,
            optimizer_step_applied=False,
        )["matrices"][0]

        self.assertFalse(optimizer.has_pending_step_diagnostics())
        self.assertEqual(record["observed_total_parameter_delta_norm"], 0.0)
        self.assertEqual(record["expected_momentum_derived_step_norm"], 0.0)
        self.assertIsNone(record["normalization_denominator"])


if __name__ == "__main__":
    unittest.main()
