import copy
import io
import math
import unittest

import torch

from frobenius_normalized_sgdm.utils.frobenius_normalized_sgdm import (
    FROBENIUS_NORMALIZATION_EQUATION,
    FROBENIUS_NORMALIZATION_VERSION,
    FrobeniusNormalizedSGDM,
)
from model import GPT, GPTConfig
from shared_utils.optimizer_factory import (
    configure_optimizer,
    get_effective_learning_rates,
    set_optimizer_learning_rates,
)
from shared_utils.parameter_partition import partition_optimizer_parameters
from shared_utils.run_metadata import (
    build_optimizer_group_signature,
    validate_resume_compatibility,
)
from tuned_global_sgdm.utils.global_sgdm import GlobalSGDM


def frobenius_cosine(first, second):
    first = first.double()
    second = second.double()
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


def configure_frobenius(model, **overrides):
    settings = {
        "device_type": "cpu",
        "adamw_learning_rate": 1e-3,
        "adamw_weight_decay": 0.1,
        "adamw_betas": (0.9, 0.95),
        "matrix_learning_rate": 999.0,
        "matrix_momentum": 0.5,
        "matrix_weight_decay": 0.0,
        "auxiliary_learning_rate": 1e-3,
        "auxiliary_weight_decay": 0.1,
        "auxiliary_betas": (0.9, 0.95),
        "frobenius_learning_rate": 0.2,
        "frobenius_epsilon": 1e-6,
        "frobenius_shape_factor": 1.0,
    }
    settings.update(overrides)
    return configure_optimizer(
        model=model,
        optimizer_name="frobenius_normalized_sgdm",
        **settings,
    )


class FrobeniusNormalizedSGDMUpdateTests(unittest.TestCase):
    def test_nonzero_updates_match_exact_additive_epsilon_rule(self):
        for shape, shape_factor in (((2, 2), 1.0), ((3, 2), 1.5), ((2, 4), 0.75)):
            with self.subTest(shape=shape, shape_factor=shape_factor):
                parameter = torch.nn.Parameter(
                    torch.arange(
                        1,
                        shape[0] * shape[1] + 1,
                        dtype=torch.float64,
                    ).reshape(shape)
                )
                gradient = torch.linspace(
                    -0.4,
                    0.6,
                    steps=parameter.numel(),
                    dtype=torch.float64,
                ).reshape(shape)
                parameter.grad = gradient.clone()
                before = parameter.detach().clone()
                learning_rate = 0.2
                momentum = 0.5
                epsilon = 0.25
                optimizer = FrobeniusNormalizedSGDM(
                    [parameter],
                    lr=learning_rate,
                    momentum=momentum,
                    epsilon=epsilon,
                    fixed_shape_factor=shape_factor,
                )

                optimizer.step()

                expected_momentum = (1.0 - momentum) * gradient
                raw_norm = torch.linalg.vector_norm(expected_momentum)
                nominal_target = shape_factor * math.sqrt(min(shape))
                expected_normalized = (
                    nominal_target * expected_momentum / (raw_norm + epsilon)
                )
                expected_delta = -learning_rate * expected_normalized
                observed_delta = parameter.detach() - before
                exact_expected_step_norm = (
                    learning_rate
                    * nominal_target
                    * raw_norm
                    / (raw_norm + epsilon)
                )

                torch.testing.assert_close(observed_delta, expected_delta)
                torch.testing.assert_close(
                    torch.linalg.vector_norm(observed_delta),
                    exact_expected_step_norm,
                )
                self.assertLess(
                    torch.linalg.vector_norm(observed_delta).item(),
                    learning_rate * nominal_target,
                )
                self.assertAlmostEqual(
                    frobenius_cosine(observed_delta, expected_momentum).item(),
                    -1.0,
                    places=12,
                )

    def test_nominal_target_is_reached_when_epsilon_is_negligible(self):
        parameter = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float64))
        parameter.grad = torch.full_like(parameter, 10.0)
        before = parameter.detach().clone()
        optimizer = FrobeniusNormalizedSGDM(
            [parameter],
            lr=0.3,
            momentum=0.0,
            epsilon=1e-12,
            fixed_shape_factor=1.25,
        )

        optimizer.step()

        observed_norm = torch.linalg.vector_norm(parameter.detach() - before)
        nominal_step_norm = 0.3 * 1.25 * math.sqrt(2.0)
        self.assertAlmostEqual(observed_norm.item(), nominal_step_norm, places=12)

    def test_zero_momentum_is_finite_and_decay_remains_independent(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )
        parameter.grad = torch.zeros_like(parameter)
        before = parameter.detach().clone()
        optimizer = FrobeniusNormalizedSGDM(
            [parameter],
            lr=0.2,
            momentum=0.7,
            epsilon=1e-6,
            weight_decay=0.1,
        )

        optimizer.step()

        torch.testing.assert_close(parameter, before * (1.0 - 0.2 * 0.1))
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertEqual(
            torch.linalg.vector_norm(
                optimizer.state[parameter]["momentum_buffer"]
            ).item(),
            0.0,
        )

    def test_weight_decay_uses_only_scheduled_base_learning_rate(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )
        gradient = torch.tensor(
            [[0.2, 0.4], [-0.6, -0.8]], dtype=torch.float64
        )
        parameter.grad = gradient.clone()
        before = parameter.detach().clone()
        optimizer = FrobeniusNormalizedSGDM(
            [parameter],
            lr=0.2,
            momentum=0.5,
            epsilon=1e-6,
            fixed_shape_factor=2.5,
            weight_decay=0.1,
        )

        optimizer.step()

        expected_momentum = 0.5 * gradient
        normalized = (
            2.5
            * math.sqrt(2.0)
            * expected_momentum
            / (torch.linalg.vector_norm(expected_momentum) + 1e-6)
        )
        expected = before * (1.0 - 0.2 * 0.1) - 0.2 * normalized
        incorrectly_scaled_decay = (
            before * (1.0 - 0.2 * 2.5 * 0.1) - 0.2 * normalized
        )
        torch.testing.assert_close(parameter, expected)
        self.assertFalse(torch.equal(parameter, incorrectly_scaled_decay))

    def test_momentum_and_step_skipping_match_global_sgdm(self):
        global_parameter = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float64))
        normalized_parameter = torch.nn.Parameter(global_parameter.detach().clone())
        global_optimizer = GlobalSGDM(
            [global_parameter], lr=0.1, momentum=0.8
        )
        normalized_optimizer = FrobeniusNormalizedSGDM(
            [normalized_parameter],
            lr=0.1,
            momentum=0.8,
            epsilon=1e-6,
        )
        gradients = (
            torch.tensor(
                [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]],
                dtype=torch.float64,
            ),
            None,
            torch.tensor(
                [[-0.6, 0.5, -0.4], [0.3, -0.2, 0.1]],
                dtype=torch.float64,
            ),
        )

        for gradient in gradients:
            global_parameter.grad = (
                None if gradient is None else gradient.clone()
            )
            normalized_parameter.grad = (
                None if gradient is None else gradient.clone()
            )
            global_optimizer.step()
            normalized_optimizer.step()
            torch.testing.assert_close(
                global_optimizer.state[global_parameter]["momentum_buffer"],
                normalized_optimizer.state[normalized_parameter][
                    "momentum_buffer"
                ],
            )

    def test_invalid_parameters_settings_and_sparse_gradients_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            FrobeniusNormalizedSGDM(
                [torch.nn.Parameter(torch.ones(2))],
                lr=0.1,
                momentum=0.9,
                epsilon=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "epsilon must be"):
            FrobeniusNormalizedSGDM(
                [torch.nn.Parameter(torch.ones((2, 2)))],
                lr=0.1,
                momentum=0.9,
                epsilon=0.0,
            )
        with self.assertRaisesRegex(ValueError, "fixed_shape_factor must be"):
            FrobeniusNormalizedSGDM(
                [torch.nn.Parameter(torch.ones((2, 2)))],
                lr=0.1,
                momentum=0.9,
                epsilon=1e-6,
                fixed_shape_factor=float("nan"),
            )

        parameter = torch.nn.Parameter(torch.ones((2, 2)))
        parameter.grad = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]),
            torch.tensor([1.0]),
            size=(2, 2),
        )
        optimizer = FrobeniusNormalizedSGDM(
            [parameter], lr=0.1, momentum=0.9, epsilon=1e-6
        )
        with self.assertRaisesRegex(RuntimeError, "sparse gradients"):
            optimizer.step()


class FrobeniusNormalizedSGDMIntegrationTests(unittest.TestCase):
    def test_factory_preserves_partition_and_records_configuration(self):
        model = make_model()
        partition = partition_optimizer_parameters(model)
        optimizer, audit = configure_frobenius(model)

        self.assertIsInstance(
            optimizer.matrix_optimizer, FrobeniusNormalizedSGDM
        )
        self.assertIsInstance(optimizer.auxiliary_optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.matrix_optimizer.param_groups), 1)
        self.assertEqual(
            {
                id(parameter)
                for parameter in optimizer.matrix_optimizer.param_groups[0][
                    "params"
                ]
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
            optimizer.frobenius_configuration,
            {
                "normalization_version": FROBENIUS_NORMALIZATION_VERSION,
                "normalization_equation": FROBENIUS_NORMALIZATION_EQUATION,
                "frobenius_learning_rate": 0.2,
                "frobenius_epsilon": 1e-6,
                "frobenius_shape_factor": 1.0,
            },
        )
        self.assertEqual(
            {
                entry["optimizer"]
                for entry in audit["parameters"]
                if entry["group"] == "eligible_matrix"
            },
            {"frobenius_normalized_sgdm"},
        )

    def test_shared_schedule_preserves_frobenius_to_auxiliary_ratio(self):
        optimizer, _ = configure_frobenius(
            make_model(),
            frobenius_learning_rate=0.2,
            auxiliary_learning_rate=0.005,
        )

        for scale, expected_matrix, expected_auxiliary in (
            (0.25, 0.05, 0.00125),
            (0.5, 0.1, 0.0025),
        ):
            set_optimizer_learning_rates(
                optimizer,
                optimizer_name="frobenius_normalized_sgdm",
                adamw_learning_rate=999.0,
                experimental_schedule_scale=scale,
            )
            self.assertEqual(
                get_effective_learning_rates(
                    optimizer, "frobenius_normalized_sgdm"
                ),
                {
                    "matrix": expected_matrix,
                    "auxiliary": expected_auxiliary,
                },
            )

    def test_state_and_metadata_validation_match_uninterrupted_next_update(self):
        torch.manual_seed(123)
        uninterrupted_model = make_model()
        uninterrupted_optimizer, _ = configure_frobenius(uninterrupted_model)
        for parameter in uninterrupted_model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        uninterrupted_optimizer.step()

        settings = {
            **uninterrupted_optimizer.frobenius_configuration,
            "matrix_momentum": 0.5,
            "matrix_weight_decay": 0.0,
        }
        signature = build_optimizer_group_signature(
            uninterrupted_model, uninterrupted_optimizer
        )
        checkpoint = {
            "model": copy.deepcopy(uninterrupted_model.state_dict()),
            "optimizer": copy.deepcopy(uninterrupted_optimizer.state_dict()),
            "run_metadata": {
                "optimizer": {
                    "name": "frobenius_normalized_sgdm",
                    "settings": copy.deepcopy(settings),
                    "group_signature": copy.deepcopy(signature),
                }
            },
        }

        restored_model = make_model()
        restored_model.load_state_dict(checkpoint["model"])
        restored_optimizer, _ = configure_frobenius(restored_model)
        restored_signature = build_optimizer_group_signature(
            restored_model, restored_optimizer
        )
        validate_resume_compatibility(
            checkpoint,
            optimizer_name="frobenius_normalized_sgdm",
            optimizer_group_signature=restored_signature,
            optimizer_settings=settings,
        )
        restored_optimizer.load_state_dict(checkpoint["optimizer"])

        changed_settings = copy.deepcopy(settings)
        changed_settings["frobenius_epsilon"] = 1e-5
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            validate_resume_compatibility(
                checkpoint,
                optimizer_name="frobenius_normalized_sgdm",
                optimizer_group_signature=restored_signature,
                optimizer_settings=changed_settings,
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
        uninterrupted_partition = partition_optimizer_parameters(
            uninterrupted_model
        )
        restored_partition = partition_optimizer_parameters(restored_model)
        for uninterrupted_item, restored_item in zip(
            uninterrupted_partition.eligible_matrices,
            restored_partition.eligible_matrices,
        ):
            torch.testing.assert_close(
                uninterrupted_optimizer.matrix_optimizer.state[
                    uninterrupted_item.parameter
                ]["momentum_buffer"],
                restored_optimizer.matrix_optimizer.state[
                    restored_item.parameter
                ]["momentum_buffer"],
            )
        self.assertEqual(
            len(uninterrupted_optimizer.auxiliary_optimizer.state),
            len(restored_optimizer.auxiliary_optimizer.state),
        )

    def test_cpu_bfloat16_smoke_forward_step_eval_save_and_resume(self):
        if not hasattr(torch, "autocast"):
            self.skipTest("torch.autocast is unavailable")
        torch.manual_seed(456)
        model = make_model()
        optimizer, _ = configure_frobenius(model)
        inputs = torch.randint(0, model.config.vocab_size, (2, 8))
        targets = torch.randint(0, model.config.vocab_size, (2, 8))

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(), torch.autocast(
            device_type="cpu", dtype=torch.bfloat16
        ):
            _, evaluation_loss = model(inputs, targets)
        self.assertTrue(torch.isfinite(evaluation_loss))

        buffer = io.BytesIO()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=False)
        restored_model = make_model()
        restored_optimizer, _ = configure_frobenius(restored_model)
        restored_model.load_state_dict(checkpoint["model"])
        restored_optimizer.load_state_dict(checkpoint["optimizer"])

        restored_model.train()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, resumed_loss = restored_model(inputs, targets)
        resumed_loss.backward()
        torch.nn.utils.clip_grad_norm_(restored_model.parameters(), 1.0)
        restored_optimizer.step()
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for parameter in restored_model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
