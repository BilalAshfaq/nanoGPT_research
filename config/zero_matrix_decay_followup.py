# Zero-matrix-decay follow-up. Reuse every non-decay control from Task 1.6.
with open('config/task_1_6_baseline.py', encoding='utf-8') as base_config_file:
    exec(base_config_file.read())

# The original Task 1.6, 2.5, and 3.5 studies used this value:
# matrix_weight_decay = 0.1
# This follow-up removes eligible-matrix decay only. Auxiliary AdamW remains
# unchanged at auxiliary_weight_decay = 0.1.
matrix_weight_decay = 0.0

# All three optimizer families share this config; manifests select the family.
static_default_multiplier = 1.0
static_matrix_type_multipliers = {}
static_exact_parameter_multipliers = {}

frobenius_learning_rate = 0.003
frobenius_epsilon = 1e-12
frobenius_shape_factor = 1.0
