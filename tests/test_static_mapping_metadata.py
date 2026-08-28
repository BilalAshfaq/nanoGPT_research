import copy
import re
import unittest
from types import SimpleNamespace

from static_per_matrix_sgdm.utils.resume_validation import (
    validate_static_multiplier_resume,
)
from static_per_matrix_sgdm.utils.static_multipliers import (
    resolve_static_multipliers,
    static_scaling_rule,
)


def eligible_items():
    suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    return tuple(
        SimpleNamespace(
            name=f"transformer.h.0.{suffix}",
            parameter=object(),
        )
        for suffix in suffixes
    )


def resolve(type_multipliers):
    return resolve_static_multipliers(
        eligible_items(),
        default_multiplier=1.0,
        matrix_type_multipliers=type_multipliers,
    )


class StaticMappingMetadataTests(unittest.TestCase):
    def test_fingerprint_summary_and_run_name_component_are_deterministic(self):
        type_multipliers = {
            "attention_qkv": 2.0,
            "attention_output": 0.5,
            "mlp_input": 4.0,
            "mlp_output": 0.25,
        }
        first = resolve(type_multipliers)
        second = resolve_static_multipliers(
            tuple(reversed(eligible_items())),
            default_multiplier=1.0,
            matrix_type_multipliers=dict(reversed(tuple(type_multipliers.items()))),
        )

        self.assertEqual(first, second)
        self.assertRegex(first["fingerprint_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            first["mapping_id"], first["fingerprint_sha256"][:12]
        )
        self.assertTrue(
            re.fullmatch(r"static-[0-9a-f]{12}", static_scaling_rule(first))
        )
        self.assertEqual(
            first["summary"],
            {
                "eligible_matrix_count": 4,
                "minimum_multiplier": 0.25,
                "maximum_multiplier": 4.0,
                "unique_multiplier_count": 4,
            },
        )

        changed = resolve(
            {
                "attention_qkv": 4.0,
                "attention_output": 0.25,
                "mlp_input": 2.0,
                "mlp_output": 0.5,
            }
        )
        self.assertNotEqual(first["fingerprint_sha256"], changed["fingerprint_sha256"])
        self.assertNotEqual(static_scaling_rule(first), static_scaling_rule(changed))

    def test_resume_validation_accepts_identical_resolved_mapping(self):
        configuration = resolve(
            {
                "attention_qkv": 2.0,
                "attention_output": 0.5,
                "mlp_input": 4.0,
                "mlp_output": 0.25,
            }
        )
        checkpoint = {
            "run_metadata": {
                "optimizer": {
                    "settings": {
                        "static_multiplier_configuration": copy.deepcopy(
                            configuration
                        )
                    }
                }
            }
        }

        validate_static_multiplier_resume(checkpoint, configuration)

    def test_resume_validation_reports_changed_matrix_before_loading_state(self):
        saved = resolve(
            {
                "attention_qkv": 2.0,
                "attention_output": 0.5,
                "mlp_input": 4.0,
                "mlp_output": 0.25,
            }
        )
        current = resolve(
            {
                "attention_qkv": 4.0,
                "attention_output": 0.25,
                "mlp_input": 2.0,
                "mlp_output": 0.5,
            }
        )
        checkpoint = {
            "run_metadata": {
                "optimizer": {
                    "settings": {
                        "static_multiplier_configuration": saved
                    }
                }
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            "static multiplier mapping mismatch.*changed:",
        ):
            validate_static_multiplier_resume(checkpoint, current)


if __name__ == "__main__":
    unittest.main()
