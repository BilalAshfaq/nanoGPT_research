import copy
import unittest

import torch

from model import GPT, GPTConfig
from shared_utils.composite_optimizer import CompositeOptimizer
from shared_utils.optimizer_factory import (
    configure_optimizer,
    set_optimizer_learning_rates,
    validate_optimizer_name,
)
from shared_utils.parameter_partition import partition_optimizer_parameters
from tuned_global_sgdm.utils.global_sgdm import GlobalSGDM


def make_model(num_layers=1):
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=num_layers,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
        )
    )


def configure(model, optimizer_name="global_sgdm", **overrides):
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
    }
    settings.update(overrides)
    return configure_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        **settings,
    )


class GlobalSGDMUpdateTests(unittest.TestCase):
    def test_zero_momentum_matches_observed_parameter_change(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )
        optimizer = GlobalSGDM(
            [parameter], lr=0.25, momentum=0.0, weight_decay=0.0
        )
        gradient = torch.tensor(
            [[0.2, -0.4], [0.6, -0.8]], dtype=torch.float64
        )
        before = parameter.detach().clone()
        parameter.grad = gradient.clone()

        optimizer.step()

        torch.testing.assert_close(parameter.detach() - before, -0.25 * gradient)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], gradient
        )

    def test_nonzero_momentum_matches_first_and_subsequent_changes(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        )
        optimizer = GlobalSGDM(
            [parameter], lr=0.1, momentum=0.75, weight_decay=0.0
        )
        first_gradient = torch.tensor(
            [[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64
        )
        second_gradient = torch.tensor(
            [[-0.5, 1.5], [2.0, -3.0]], dtype=torch.float64
        )

        before_first = parameter.detach().clone()
        parameter.grad = first_gradient.clone()
        optimizer.step()
        first_momentum = 0.25 * first_gradient
        torch.testing.assert_close(
            parameter.detach() - before_first, -0.1 * first_momentum
        )

        before_second = parameter.detach().clone()
        parameter.grad = second_gradient.clone()
        optimizer.step()
        second_momentum = 0.75 * first_momentum + 0.25 * second_gradient
        torch.testing.assert_close(
            parameter.detach() - before_second, -0.1 * second_momentum
        )
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], second_momentum
        )

    def test_weight_decay_is_decoupled_and_precedes_momentum_update(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )
        optimizer = GlobalSGDM(
            [parameter], lr=0.2, momentum=0.5, weight_decay=0.1
        )
        gradient = torch.tensor(
            [[0.2, 0.4], [-0.6, -0.8]], dtype=torch.float64
        )
        before = parameter.detach().clone()
        parameter.grad = gradient.clone()

        optimizer.step()

        expected_momentum = 0.5 * gradient
        expected_after = before * (1.0 - 0.2 * 0.1) - 0.2 * expected_momentum
        torch.testing.assert_close(parameter.detach(), expected_after)

    def test_state_round_trip_matches_uninterrupted_next_update(self):
        first = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
        first_optimizer = GlobalSGDM(
            [first], lr=0.1, momentum=0.8, weight_decay=0.02
        )
        first.grad = torch.full_like(first, 0.5)
        first_optimizer.step()

        restored = torch.nn.Parameter(first.detach().clone())
        restored_optimizer = GlobalSGDM(
            [restored], lr=0.1, momentum=0.8, weight_decay=0.02
        )
        restored_optimizer.load_state_dict(
            copy.deepcopy(first_optimizer.state_dict())
        )

        next_gradient = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64
        )
        first.grad = next_gradient.clone()
        restored.grad = next_gradient.clone()
        first_optimizer.step()
        restored_optimizer.step()

        torch.testing.assert_close(first, restored)
        torch.testing.assert_close(
            first_optimizer.state[first]["momentum_buffer"],
            restored_optimizer.state[restored]["momentum_buffer"],
        )


class OptimizerIntegrationTests(unittest.TestCase):
    def test_experimental_groups_are_disjoint_and_owned_by_intended_optimizers(self):
        model = make_model(num_layers=2)
        partition = partition_optimizer_parameters(model)
        optimizer, audit = configure(model)

        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertIsInstance(optimizer.matrix_optimizer, GlobalSGDM)
        self.assertIsInstance(optimizer.auxiliary_optimizer, torch.optim.AdamW)
        self.assertEqual(
            {
                id(parameter)
                for group in optimizer.matrix_optimizer.param_groups
                for parameter in group["params"]
            },
            {id(item.parameter) for item in partition.eligible_matrices},
        )
        self.assertEqual(
            {
                id(parameter)
                for group in optimizer.auxiliary_optimizer.param_groups
                for parameter in group["params"]
            },
            {id(item.parameter) for item in partition.auxiliary},
        )
        self.assertEqual(
            {
                entry["weight_decay_semantics"]
                for entry in audit["parameters"]
                if entry["group"] == "eligible_matrix"
            },
            {"decoupled"},
        )

    def test_zero_grad_set_to_none_covers_both_parameter_sets(self):
        model = make_model()
        optimizer, _ = configure(model)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)

        optimizer.zero_grad(set_to_none=True)

        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

    def test_each_parameter_set_changes_only_through_its_assigned_optimizer(self):
        model = make_model()
        partition = partition_optimizer_parameters(model)
        optimizer, _ = configure(
            model,
            matrix_learning_rate=0.2,
            matrix_momentum=0.5,
            matrix_weight_decay=0.0,
            auxiliary_weight_decay=0.0,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)

        optimizer.step()

        for item in partition.eligible_matrices:
            observed_change = item.parameter.detach() - before[item.name]
            torch.testing.assert_close(
                observed_change,
                torch.full_like(item.parameter, -0.025),
            )
            self.assertIn(item.parameter, optimizer.matrix_optimizer.state)
            self.assertNotIn(item.parameter, optimizer.auxiliary_optimizer.state)
        for item in partition.auxiliary:
            self.assertIn(item.parameter, optimizer.auxiliary_optimizer.state)
            self.assertNotIn(item.parameter, optimizer.matrix_optimizer.state)

    def test_composite_state_round_trip_matches_uninterrupted_update(self):
        torch.manual_seed(123)
        uninterrupted_model = make_model()
        uninterrupted_optimizer, _ = configure(uninterrupted_model)
        for parameter in uninterrupted_model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        uninterrupted_optimizer.step()

        restored_model = make_model()
        restored_model.load_state_dict(uninterrupted_model.state_dict())
        restored_optimizer, _ = configure(restored_model)
        restored_optimizer.load_state_dict(
            copy.deepcopy(uninterrupted_optimizer.state_dict())
        )

        for parameter in uninterrupted_model.parameters():
            parameter.grad = torch.full_like(parameter, -0.125)
        for parameter in restored_model.parameters():
            parameter.grad = torch.full_like(parameter, -0.125)
        uninterrupted_optimizer.step()
        restored_optimizer.step()

        for uninterrupted, restored in zip(
            uninterrupted_model.parameters(), restored_model.parameters()
        ):
            torch.testing.assert_close(uninterrupted, restored)

    def test_scheduler_preserves_matrix_to_auxiliary_lr_ratio(self):
        model = make_model()
        optimizer, _ = configure(
            model,
            matrix_learning_rate=0.2,
            auxiliary_learning_rate=0.005,
        )

        set_optimizer_learning_rates(
            optimizer,
            optimizer_name="global_sgdm",
            adamw_learning_rate=999.0,
            experimental_schedule_scale=0.25,
        )

        matrix_lrs = {
            group["lr"]
            for group in optimizer.param_groups
            if group["optimizer_role"] == "matrix"
        }
        auxiliary_lrs = {
            group["lr"]
            for group in optimizer.param_groups
            if group["optimizer_role"] == "auxiliary"
        }
        self.assertEqual(matrix_lrs, {0.05})
        self.assertEqual(auxiliary_lrs, {0.00125})

    def test_adamw_selection_uses_protected_model_construction(self):
        model = make_model()
        optimizer, audit = configure(model, optimizer_name="adamw")

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsNone(audit)
        self.assertEqual(
            {
                id(parameter)
                for parameter in optimizer.param_groups[0]["params"]
            },
            {
                id(parameter)
                for parameter in model.parameters()
                if parameter.ndim >= 2
            },
        )
        self.assertEqual(
            {
                id(parameter)
                for parameter in optimizer.param_groups[1]["params"]
            },
            {
                id(parameter)
                for parameter in model.parameters()
                if parameter.ndim < 2
            },
        )

        set_optimizer_learning_rates(
            optimizer,
            optimizer_name="adamw",
            adamw_learning_rate=0.0003,
            experimental_schedule_scale=999.0,
        )
        self.assertEqual(
            {group["lr"] for group in optimizer.param_groups}, {0.0003}
        )

    def test_nesterov_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nesterov momentum is disabled"):
            configure(make_model(), matrix_nesterov=True)

    def test_unknown_optimizer_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown optimizer_name"):
            validate_optimizer_name("not_an_optimizer")


if __name__ == "__main__":
    unittest.main()
