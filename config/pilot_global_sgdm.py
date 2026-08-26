# Two-GPU Global SGDM pilot for observing a meaningful Tiny Shakespeare trend.
# This is an implementation/stability check, not a Task 1.6 tuning result.

out_dir = 'out-global-sgdm-pilot'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = True
wandb_log = False

dataset = 'shakespeare_char'
tokenizer = 'character'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

optimizer_name = 'global_sgdm'
learning_rate = 6e-4
matrix_learning_rate = 0.1
matrix_momentum = 0.95
matrix_weight_decay = 0.1
auxiliary_learning_rate = 6e-4
auxiliary_weight_decay = 0.1
auxiliary_beta1 = 0.9
auxiliary_beta2 = 0.95

max_iters = 500
decay_lr = True
warmup_iters = 50
lr_decay_iters = 500
min_lr = 6e-5
grad_clip = 1.0

diagnostics_enabled = True
diagnostic_steps = '0,100,250,500'
diagnostic_spectral_matrix_names = 'transformer.h.0.attn.c_proj.weight'

seed = 1337
device = 'cuda'
dtype = 'bfloat16'
compile = True
