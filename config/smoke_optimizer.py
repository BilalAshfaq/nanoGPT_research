# Tiny repeatable CPU smoke run for optimizer integration and checkpoint resume.

out_dir = 'out-optimizer-smoke'
eval_interval = 1
eval_iters = 1
log_interval = 1
always_save_checkpoint = True
wandb_log = False

dataset = 'shakespeare_char'
tokenizer = 'character'
gradient_accumulation_steps = 1
batch_size = 2
block_size = 8

n_layer = 1
n_head = 1
n_embd = 8
dropout = 0.0
bias = True

learning_rate = 1e-3
matrix_learning_rate = 1e-3
auxiliary_learning_rate = 1e-3
max_iters = 1
decay_lr = False
warmup_iters = 0
lr_decay_iters = 1
min_lr = 1e-4

seed = 1337
device = 'cpu'
dtype = 'float32'
compile = False
