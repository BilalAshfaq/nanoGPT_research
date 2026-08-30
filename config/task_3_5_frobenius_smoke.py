# Short two-GPU preflight using the real Task 3.5 model, data, and optimizer.
with open('config/task_3_4_frobenius.py', encoding='utf-8') as study_config_file:
    exec(study_config_file.read())

out_dir = 'out-task-3-5-frobenius-smoke-placeholder'
eval_interval = 1
eval_iters = 2
log_interval = 1
max_iters = 1
warmup_iters = 0
lr_decay_iters = 1

frobenius_learning_rate = 0.01
matrix_momentum = 0.95
frobenius_epsilon = 1e-12
frobenius_shape_factor = 1.0

diagnostics_enabled = True
diagnostic_steps = '0,1'
diagnostic_spectral_matrix_names = 'transformer.h.0.attn.c_proj.weight'
