# Task 2.4 reuses every common Task 1.6 control. Candidate-specific static
# settings are supplied only by the materialized, reviewed manifest.
with open('config/task_1_6_baseline.py', encoding='utf-8') as base_config_file:
    exec(compile(base_config_file.read(), base_config_file.name, 'exec'))

optimizer_name = 'static_per_matrix_sgdm'
static_default_multiplier = 1.0
static_matrix_type_multipliers = {}
static_exact_parameter_multipliers = {}
