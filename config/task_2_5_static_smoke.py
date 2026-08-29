# Short two-GPU preflight using the real Task 2.5 model, data, and optimizer.
with open('config/task_2_4_static.py', encoding='utf-8') as study_config_file:
    exec(compile(study_config_file.read(), study_config_file.name, 'exec'))

out_dir = 'out-task-2-5-static-smoke-placeholder'
eval_interval = 1
eval_iters = 2
log_interval = 1
max_iters = 1
warmup_iters = 0
lr_decay_iters = 1

matrix_learning_rate = 0.03
matrix_momentum = 0.99
static_matrix_type_multipliers = {
    'attention_qkv': 1.4142135623730951,
    'attention_output': 1.4142135623730951,
    'mlp_input': 0.7071067811865476,
    'mlp_output': 0.7071067811865476,
}

diagnostics_enabled = True
diagnostic_steps = '0,1'
diagnostic_spectral_matrix_names = 'transformer.h.0.attn.c_proj.weight'
