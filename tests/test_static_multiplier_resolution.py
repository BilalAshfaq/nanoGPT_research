import copy
import unittest

from model import GPT, GPTConfig
from shared_utils.optimizer_factory import (
    configure_optimizer,
    validate_optimizer_name,
)
from shared_utils.parameter_partition import partition_optimizer_parameters
from shared_utils.run_metadata import (
    build_optimizer_group_signature,
    validate_resume_compatibility,
)
from static_per_matrix_sgdm.utils.static_multipliers import (
    MATRIX_TYPE_SUFFIXES,
    resolve_static_multipliers,
)


def make_model(num_layers=2):
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


def configure(model, optimizer_name="static_per_matrix_sgdm", **overrides):
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
        "static_matrix_type_multipliers": {},
        "static_exact_parameter_multipliers": {},
    }
    settings.update(overrides)
    return configure_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        **settings,
    )


class StaticMultiplierResolutionTests(unittest.TestCase):
    def test_matrix_type_assignment_covers_every_matrix_in_multiple_layers(self):
        model = make_model(num_layers=3)
        partition = partition_optimizer_parameters(model)
        type_values = {
            "attention_qkv": 2.0,
            "attention_output": 0.5,
            "mlp_input": 4.0,
            "mlp_output": 0.25,
        }

        configuration = resolve_static_multipliers(
            partition.eligible_matrices,
            default_multiplier=1.0,
            matrix_type_multipliers=type_values,
        )

        resolved = configuration["resolved_multipliers"]
        self.assertEqual(
            list(resolved), sorted(item.name for item in partition.eligible_matrices)
        )
        self.assertEqual(len(resolved), 12)
        for parameter_name, multiplier in resolved.items():
            matching_types = [
                matrix_type
                for matrix_type, suffix in MATRIX_TYPE_SUFFIXES.items()
                if parameter_name.endswith(suffix)
            ]
            self.assertEqual(len(matching_types), 1)
            self.assertEqual(multiplier, type_values[matching_types[0]])
        self.assertAlmostEqual(configuration["geometric_mean"], 1.0)

    def test_exact_name_override_takes_precedence_over_matrix_type(self):
        model = make_model(num_layers=2)
        partition = partition_optimizer_parameters(model)
        first_name = "transformer.h.0.attn.c_attn.weight"
        second_name = "transformer.h.1.attn.c_attn.weight"

        configuration = resolve_static_multipliers(
            partition.eligible_matrices,
            default_multiplier=1.0,
            matrix_type_multipliers={"attention_qkv": 1.0},
            exact_parameter_multipliers={first_name: 2.0, second_name: 0.5},
        )

        resolved = configuration["resolved_multipliers"]
        self.assertEqual(resolved[first_name], 2.0)
        self.assertEqual(resolved[second_name], 0.5)
        self.assertTrue(
            all(
                multiplier == 1.0
                for name, multiplier in resolved.items()
                if name not in {first_name, second_name}
            )
        )

    def test_unnormalized_mapping_is_rejected_instead_of_renormalized(self):
        partition = partition_optimizer_parameters(make_model(num_layers=1))

        with self.assertRaisesRegex(ValueError, "geometric mean 1.0"):
            resolve_static_multipliers(
                partition.eligible_matrices,
                default_multiplier=2.0,
            )

    def test_invalid_values_unknown_names_and_unknown_types_are_rejected(self):
        partition = partition_optimizer_parameters(make_model(num_layers=1))
        invalid_values = (
            0.0,
            -1.0,
            float("inf"),
            float("nan"),
            True,
            "1.0",
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    resolve_static_multipliers(
                        partition.eligible_matrices,
                        default_multiplier=invalid_value,
                    )

        with self.assertRaisesRegex(ValueError, "unknown static matrix types"):
            resolve_static_multipliers(
                partition.eligible_matrices,
                default_multiplier=1.0,
                matrix_type_multipliers={"attention_qkv_typo": 1.0},
            )
        with self.assertRaisesRegex(
            ValueError, "unknown static exact parameter names"
        ):
            resolve_static_multipliers(
                partition.eligible_matrices,
                default_multiplier=1.0,
                exact_parameter_multipliers={
                    "transformer.h.0.attn.misspelled.weight": 1.0
                },
            )

    def test_duplicate_eligible_names_are_rejected(self):
        partition = partition_optimizer_parameters(make_model(num_layers=1))
        duplicated = partition.eligible_matrices + (
            partition.eligible_matrices[0],
        )

        with self.assertRaisesRegex(ValueError, "names must be unique"):
            resolve_static_multipliers(
                duplicated,
                default_multiplier=1.0,
            )

    def test_missing_eligible_matrices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires eligible matrices"):
            resolve_static_multipliers((), default_multiplier=1.0)

    def test_all_ones_resolves_to_global_sgdm_scaling(self):
        partition = partition_optimizer_parameters(make_model(num_layers=2))
        configuration = resolve_static_multipliers(
            partition.eligible_matrices,
            default_multiplier=1.0,
        )

        self.assertEqual(
            set(configuration["resolved_multipliers"].values()), {1.0}
        )

    def test_construction_is_stable_and_resume_settings_validate(self):
        first_model = make_model(num_layers=2)
        first_optimizer, first_audit = configure(
            first_model,
            static_matrix_type_multipliers={
                "attention_qkv": 2.0,
                "attention_output": 0.5,
                "mlp_input": 4.0,
                "mlp_output": 0.25,
            },
        )
        second_model = make_model(num_layers=2)
        second_optimizer, second_audit = configure(
            second_model,
            static_matrix_type_multipliers={
                "attention_qkv": 2.0,
                "attention_output": 0.5,
                "mlp_input": 4.0,
                "mlp_output": 0.25,
            },
        )

        first_configuration = first_optimizer.static_multiplier_configuration
        second_configuration = second_optimizer.static_multiplier_configuration
        self.assertEqual(first_configuration, second_configuration)
        self.assertEqual(first_audit, second_audit)
        signature = build_optimizer_group_signature(first_model, first_optimizer)
        second_signature = build_optimizer_group_signature(
            second_model, second_optimizer
        )
        self.assertEqual(signature, second_signature)
        checkpoint = {
            "run_metadata": {
                "optimizer": {
                    "name": "static_per_matrix_sgdm",
                    "group_signature": signature,
                    "settings": {
                        "static_multiplier_configuration": copy.deepcopy(
                            first_configuration
                        )
                    },
                }
            }
        }
        validate_resume_compatibility(
            checkpoint,
            optimizer_name="static_per_matrix_sgdm",
            optimizer_group_signature=second_signature,
            optimizer_settings={
                "static_multiplier_configuration": second_configuration
            },
        )

        changed_settings = copy.deepcopy(second_configuration)
        changed_settings["resolved_multipliers"] = dict(
            changed_settings["resolved_multipliers"]
        )
        changed_settings["resolved_multipliers"][
            "transformer.h.0.attn.c_attn.weight"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            validate_resume_compatibility(
                checkpoint,
                optimizer_name="static_per_matrix_sgdm",
                optimizer_group_signature=signature,
                optimizer_settings={
                    "static_multiplier_configuration": changed_settings
                },
            )

    def test_variant_is_registered_and_adamw_path_remains_protected(self):
        validate_optimizer_name("static_per_matrix_sgdm")
        optimizer, audit = configure(make_model(), optimizer_name="adamw")

        self.assertEqual(optimizer.__class__.__name__, "AdamW")
        self.assertEqual(
            {entry["optimizer"] for entry in audit["parameters"]}, {"adamw"}
        )


if __name__ == "__main__":
    unittest.main()
