import unittest

import torch

from model import GPT, GPTConfig
from shared_utils.optimizer_factory import (
    configure_optimizer,
    set_optimizer_learning_rates,
)
from shared_utils.parameter_partition import NamedParameter
from static_per_matrix_sgdm.utils.static_per_matrix_sgdm import (
    StaticPerMatrixSGDM,
)
from tuned_global_sgdm.utils.global_sgdm import GlobalSGDM


def named(name, parameter):
    return NamedParameter(name, parameter)


def frobenius_cosine(first, second):
    return torch.sum(first * second) / (
        torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    )


def make_model():
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
        )
    )


def configure_static(model, **overrides):
    settings = {
        "device_type": "cpu",
        "adamw_learning_rate": 1e-3,
        "adamw_weight_decay": 0.1,
        "adamw_betas": (0.9, 0.95),
        "matrix_learning_rate": 0.2,
        "matrix_momentum": 0.5,
        "matrix_weight_decay": 0.0,
        "auxiliary_learning_rate": 1e-3,
        "auxiliary_weight_decay": 0.1,
        "auxiliary_betas": (0.9, 0.95),
        "static_default_multiplier": 1.0,
        "static_matrix_type_multipliers": {
            "attention_qkv": 2.0,
            "attention_output": 0.5,
            "mlp_input": 4.0,
            "mlp_output": 0.25,
        },
        "static_exact_parameter_multipliers": {},
    }
    settings.update(overrides)
    return configure_optimizer(
        model=model,
        optimizer_name="static_per_matrix_sgdm",
        **settings,
    )


class StaticPerMatrixSGDMUpdateTests(unittest.TestCase):
    def test_synthetic_updates_match_fixed_scaling_for_several_shapes(self):
        specifications = (
            ("square", (2, 2), 2.0),
            ("tall", (3, 2), 0.5),
            ("wide", (2, 4), 1.5),
        )
        named_parameters = []
        multipliers = {}
        gradients = {}
        before = {}
        for index, (label, shape, multiplier) in enumerate(specifications):
            parameter = torch.nn.Parameter(
                torch.arange(1, 1 + shape[0] * shape[1], dtype=torch.float64)
                .reshape(shape)
            )
            name = f"{label}.weight"
            gradient = torch.full_like(parameter, 0.2 * (index + 1))
            parameter.grad = gradient.clone()
            named_parameters.append(named(name, parameter))
            multipliers[name] = multiplier
            gradients[name] = gradient
            before[name] = parameter.detach().clone()

        optimizer = StaticPerMatrixSGDM(
            named_parameters,
            multipliers,
            lr=0.1,
            momentum=0.5,
            weight_decay=0.0,
        )
        optimizer.step()

        for item in named_parameters:
            expected_momentum = 0.5 * gradients[item.name]
            expected_update = (
                -0.1 * multipliers[item.name] * expected_momentum
            )
            observed_update = item.parameter.detach() - before[item.name]
            torch.testing.assert_close(observed_update, expected_update)
            torch.testing.assert_close(
                optimizer.state[item.parameter]["momentum_buffer"],
                expected_momentum,
            )
            self.assertAlmostEqual(
                frobenius_cosine(observed_update, expected_momentum).item(),
                -1.0,
            )

    def test_update_norm_ratios_match_multiplier_ratios(self):
        first = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float64))
        second = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float64))
        first.grad = torch.full_like(first, 0.4)
        second.grad = first.grad.clone()
        optimizer = StaticPerMatrixSGDM(
            (named("first", first), named("second", second)),
            {"first": 2.0, "second": 0.5},
            lr=0.1,
            momentum=0.0,
        )
        first_before = first.detach().clone()
        second_before = second.detach().clone()

        optimizer.step()

        first_norm = torch.linalg.vector_norm(first.detach() - first_before)
        second_norm = torch.linalg.vector_norm(second.detach() - second_before)
        self.assertAlmostEqual((first_norm / second_norm).item(), 4.0)

    def test_momentum_evolution_is_identical_to_global_sgdm(self):
        global_parameter = torch.nn.Parameter(
            torch.ones((2, 2), dtype=torch.float64)
        )
        static_parameter = torch.nn.Parameter(
            global_parameter.detach().clone()
        )
        global_optimizer = GlobalSGDM(
            [global_parameter], lr=0.1, momentum=0.8
        )
        static_optimizer = StaticPerMatrixSGDM(
            (named("matrix", static_parameter),),
            {"matrix": 2.0},
            lr=0.1,
            momentum=0.8,
        )
        gradients = (
            torch.tensor([[0.1, -0.2], [0.3, -0.4]], dtype=torch.float64),
            torch.tensor([[-0.5, 0.6], [0.7, -0.8]], dtype=torch.float64),
        )

        for gradient in gradients:
            global_parameter.grad = gradient.clone()
            static_parameter.grad = gradient.clone()
            global_optimizer.step()
            static_optimizer.step()
            torch.testing.assert_close(
                global_optimizer.state[global_parameter]["momentum_buffer"],
                static_optimizer.state[static_parameter]["momentum_buffer"],
            )

    def test_all_ones_matches_global_sgdm_step_for_step(self):
        global_parameters = [
            torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64)),
            torch.nn.Parameter(torch.full((3, 2), 2.0, dtype=torch.float64)),
        ]
        static_parameters = [
            torch.nn.Parameter(parameter.detach().clone())
            for parameter in global_parameters
        ]
        global_optimizer = GlobalSGDM(
            global_parameters, lr=0.15, momentum=0.7, weight_decay=0.03
        )
        static_optimizer = StaticPerMatrixSGDM(
            tuple(
                named(f"matrix_{index}", parameter)
                for index, parameter in enumerate(static_parameters)
            ),
            {"matrix_0": 1.0, "matrix_1": 1.0},
            lr=0.15,
            momentum=0.7,
            weight_decay=0.03,
        )

        for gradient_value in (0.25, -0.125, 0.5):
            for parameter in global_parameters:
                parameter.grad = torch.full_like(parameter, gradient_value)
            for parameter in static_parameters:
                parameter.grad = torch.full_like(parameter, gradient_value)
            global_optimizer.step()
            static_optimizer.step()

            for global_parameter, static_parameter in zip(
                global_parameters, static_parameters
            ):
                torch.testing.assert_close(global_parameter, static_parameter)
                torch.testing.assert_close(
                    global_optimizer.state[global_parameter]["momentum_buffer"],
                    static_optimizer.state[static_parameter]["momentum_buffer"],
                )

    def test_weight_decay_uses_base_lr_not_static_multiplier(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )
        gradient = torch.tensor(
            [[0.2, 0.4], [-0.6, -0.8]], dtype=torch.float64
        )
        parameter.grad = gradient.clone()
        optimizer = StaticPerMatrixSGDM(
            (named("matrix", parameter),),
            {"matrix": 2.5},
            lr=0.2,
            momentum=0.5,
            weight_decay=0.1,
        )
        before = parameter.detach().clone()

        optimizer.step()

        expected_momentum = 0.5 * gradient
        expected_after = (
            before * (1.0 - 0.2 * 0.1)
            - 0.2 * 2.5 * expected_momentum
        )
        incorrectly_scaled_decay = (
            before * (1.0 - 0.2 * 2.5 * 0.1)
            - 0.2 * 2.5 * expected_momentum
        )
        torch.testing.assert_close(parameter.detach(), expected_after)
        self.assertFalse(torch.equal(parameter.detach(), incorrectly_scaled_decay))

    def test_multiplier_and_effective_lr_are_exposed_and_remain_fixed(self):
        parameter = torch.nn.Parameter(torch.ones((2, 2)))
        optimizer = StaticPerMatrixSGDM(
            (named("matrix", parameter),),
            {"matrix": 2.0},
            lr=0.1,
            momentum=0.5,
        )

        self.assertEqual(optimizer.multiplier_for(parameter), 2.0)
        self.assertEqual(optimizer.effective_learning_rate_for(parameter), 0.2)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 0.025
        parameter.grad = torch.full_like(parameter, -3.0)
        optimizer.step()

        self.assertEqual(optimizer.multiplier_for(parameter), 2.0)
        self.assertEqual(optimizer.effective_learning_rate_for(parameter), 0.05)


class StaticPerMatrixSGDMIntegrationTests(unittest.TestCase):
    def test_factory_uses_one_matrix_optimizer_group_and_shared_schedule(self):
        model = make_model()
        optimizer, audit = configure_static(model)
        matrix_optimizer = optimizer.matrix_optimizer

        self.assertIsInstance(matrix_optimizer, StaticPerMatrixSGDM)
        self.assertEqual(len(matrix_optimizer.param_groups), 1)
        resolved = optimizer.static_multiplier_configuration[
            "resolved_multipliers"
        ]
        names_by_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        for group in matrix_optimizer.param_groups:
            for parameter in group["params"]:
                name = names_by_id[id(parameter)]
                self.assertEqual(
                    matrix_optimizer.multiplier_for(parameter), resolved[name]
                )

        set_optimizer_learning_rates(
            optimizer,
            optimizer_name="static_per_matrix_sgdm",
            adamw_learning_rate=999.0,
            experimental_schedule_scale=0.25,
        )
        for parameter in matrix_optimizer.param_groups[0]["params"]:
            name = names_by_id[id(parameter)]
            self.assertEqual(
                matrix_optimizer.effective_learning_rate_for(parameter),
                0.2 * 0.25 * resolved[name],
            )
        self.assertEqual(
            {
                entry["optimizer"]
                for entry in audit["parameters"]
                if entry["group"] == "eligible_matrix"
            },
            {"static_per_matrix_sgdm"},
        )


if __name__ == "__main__":
    unittest.main()
