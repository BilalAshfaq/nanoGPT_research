# Exact common controls for the Task 1.6 AdamW/global-SGDM comparison.
# Candidate-specific settings are supplied only by the approved manifest.

out_dir = 'out-task-1-6-placeholder'
eval_interval = 333
eval_iters = 200
log_interval = 10
always_save_checkpoint = True
wandb_log = False

dataset = 'openwebtext'
tokenizer = 'gpt2'
gradient_accumulation_steps = 40
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

optimizer_name = 'adamw'
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

matrix_learning_rate = 0.1
matrix_momentum = 0.95
matrix_weight_decay = 0.1
matrix_momentum_convention = 'ema'
matrix_weight_decay_mode = 'decoupled'
matrix_nesterov = False

auxiliary_learning_rate = 6e-4
auxiliary_weight_decay = 0.1
auxiliary_beta1 = 0.9
auxiliary_beta2 = 0.95
auxiliary_weight_decay_mode = 'adamw_decoupled'

max_iters = 999
decay_lr = True
warmup_iters = 100
lr_decay_iters = 999
min_lr = 6e-5
grad_clip = 1.0

diagnostics_enabled = False
diagnostic_steps = ''
diagnostic_spectral_matrix_names = ''

seed = 1337
device = 'cuda'
dtype = 'bfloat16'
compile = True
