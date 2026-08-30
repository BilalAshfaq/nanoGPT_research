# Task 3.4 reuses every common Task 1.6 control. Candidate-specific
# Frobenius settings are supplied only by the frozen manifest.
with open('config/task_1_6_baseline.py', encoding='utf-8') as base_config_file:
    exec(base_config_file.read())

optimizer_name = 'frobenius_normalized_sgdm'
frobenius_learning_rate = 0.01
frobenius_epsilon = 1e-12
frobenius_shape_factor = 1.0
