import unittest

from model import GPT, GPTConfig
from shared_utils.parameter_partition import (
    ELIGIBLE_MATRIX_SUFFIXES,
    build_parameter_group_audit,
    partition_optimizer_parameters,
)


def make_model(num_layers, bias=True):
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=num_layers,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=bias,
        )
    )


class ParameterPartitionTests(unittest.TestCase):
    def test_every_trainable_parameter_is_assigned_exactly_once(self):
        model = make_model(num_layers=2)
        partition = partition_optimizer_parameters(model)
        grouped = partition.eligible_matrices + partition.auxiliary

        grouped_ids = [id(item.parameter) for item in grouped]
        trainable_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }

        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), trainable_ids)

    def test_all_hidden_projection_matrices_are_eligible_at_multiple_depths(self):
        for num_layers in (1, 3):
            with self.subTest(num_layers=num_layers):
                model = make_model(num_layers=num_layers)
                partition = partition_optimizer_parameters(model)
                actual_names = {
                    item.name for item in partition.eligible_matrices
                }
                expected_names = {
                    f"transformer.h.{layer_index}.{suffix}"
                    for layer_index in range(num_layers)
                    for suffix in ELIGIBLE_MATRIX_SUFFIXES
                }

                self.assertEqual(actual_names, expected_names)
                self.assertTrue(
                    all(
                        item.parameter.ndim == 2
                        for item in partition.eligible_matrices
                    )
                )

    def test_embeddings_head_layernorms_and_biases_are_auxiliary(self):
        model = make_model(num_layers=2, bias=True)
        partition = partition_optimizer_parameters(model)
        eligible_names = {
            item.name for item in partition.eligible_matrices
        }
        auxiliary_names = {item.name for item in partition.auxiliary}

        self.assertIn("transformer.wte.weight", auxiliary_names)
        self.assertIn("transformer.wpe.weight", auxiliary_names)
        self.assertNotIn("lm_head.weight", eligible_names)
        self.assertNotIn("lm_head.weight", auxiliary_names)
        self.assertTrue(
            all(
                name in auxiliary_names
                for name, _ in model.named_parameters()
                if ".ln_" in name or name.startswith("transformer.ln_f")
            )
        )
        self.assertTrue(
            all(
                name in auxiliary_names
                for name, _ in model.named_parameters()
                if name.endswith(".bias")
            )
        )
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        self.assertEqual(
            partition.tied_parameter_canonical_name,
            "transformer.wte.weight",
        )

    def test_frozen_parameters_are_excluded(self):
        model = make_model(num_layers=1)
        model.transformer.wpe.weight.requires_grad_(False)
        partition = partition_optimizer_parameters(model)
        grouped_names = {
            item.name
            for item in partition.eligible_matrices + partition.auxiliary
        }

        self.assertNotIn("transformer.wpe.weight", grouped_names)

    def test_audit_is_deterministic_and_totals_match_the_model(self):
        model = make_model(num_layers=2)
        partition = partition_optimizer_parameters(model)
        audit_a = build_parameter_group_audit(
            partition,
            eligible_optimizer="global_sgdm",
            eligible_weight_decay=0.01,
            auxiliary_weight_decay=0.1,
        )
        audit_b = build_parameter_group_audit(
            partition_optimizer_parameters(model),
            eligible_optimizer="global_sgdm",
            eligible_weight_decay=0.01,
            auxiliary_weight_decay=0.1,
        )

        self.assertEqual(audit_a, audit_b)
        self.assertEqual(
            audit_a["totals"]["parameter_count"],
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )
        self.assertEqual(
            audit_a["totals"]["parameter_tensors"],
            sum(1 for parameter in model.parameters() if parameter.requires_grad),
        )
        self.assertTrue(audit_a["tied_parameters"]["counted_once"])
        self.assertEqual(
            audit_a["tied_parameters"]["assignment"], "auxiliary"
        )
        self.assertEqual(
            [entry["name"] for entry in audit_a["parameters"]],
            sorted(entry["name"] for entry in audit_a["parameters"]),
        )
        entries_by_name = {
            entry["name"]: entry for entry in audit_a["parameters"]
        }
        self.assertEqual(
            entries_by_name["transformer.h.0.attn.c_attn.weight"]["optimizer"],
            "global_sgdm",
        )
        self.assertEqual(
            entries_by_name["transformer.h.0.attn.c_attn.weight"]["weight_decay"],
            0.01,
        )
        self.assertEqual(
            entries_by_name["transformer.wte.weight"]["weight_decay"], 0.1
        )
        self.assertEqual(
            entries_by_name["transformer.ln_f.weight"]["weight_decay"], 0.0
        )

    def test_default_adamw_groups_keep_the_existing_dimension_rule(self):
        model = make_model(num_layers=2)
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        expected_decay_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim >= 2
        }
        expected_no_decay_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim < 2
        }

        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.1)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)
        self.assertEqual(
            {id(parameter) for parameter in optimizer.param_groups[0]["params"]},
            expected_decay_ids,
        )
        self.assertEqual(
            {id(parameter) for parameter in optimizer.param_groups[1]["params"]},
            expected_no_decay_ids,
        )


if __name__ == "__main__":
    unittest.main()
