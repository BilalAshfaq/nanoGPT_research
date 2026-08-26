"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from shared_utils.optimizer_factory import (
    configure_optimizer,
    get_effective_learning_rates,
    set_optimizer_learning_rates,
    validate_optimizer_name,
)
from shared_utils.run_metadata import (
    OptimizerStepTimer,
    append_evaluation_record,
    build_optimizer_group_signature,
    build_run_metadata,
    get_git_commit_hash,
    get_hardware_metadata,
    get_peak_gpu_memory_metadata,
    initialize_evaluation_log,
    snapshot_run_metadata,
    validate_resume_compatibility,
    write_run_summary,
)
from shared_utils.parameter_partition import partition_optimizer_parameters
from tuned_global_sgdm.utils.diagnostics import (
    GlobalSGDMDiagnostics,
    append_diagnostic_record,
    initialize_diagnostic_log,
    parse_diagnostic_matrix_names,
    parse_diagnostic_steps,
)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
tokenizer = 'gpt2'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# optimizer selection; AdamW remains the protected default
optimizer_name = 'adamw'
# original AdamW baseline
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# global SGDM eligible-matrix settings
matrix_learning_rate = 6e-4
matrix_momentum = 0.9
matrix_weight_decay = 0.0
matrix_momentum_convention = 'ema'
matrix_weight_decay_mode = 'decoupled'
matrix_nesterov = False
# auxiliary AdamW settings used with experimental matrix optimizers
auxiliary_learning_rate = 6e-4
auxiliary_weight_decay = 1e-1
auxiliary_beta1 = 0.9
auxiliary_beta2 = 0.95
auxiliary_weight_decay_mode = 'adamw_decoupled'
# optional global-SGDM matrix diagnostics
diagnostics_enabled = False
diagnostic_steps = '' # comma-separated optimizer-step indices
diagnostic_spectral_matrix_names = '' # comma-separated exact parameter names
diagnostics_epsilon = 1e-12
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
validate_optimizer_name(optimizer_name) # fail before model construction or data loading
parsed_diagnostic_steps = parse_diagnostic_steps(diagnostic_steps)
parsed_diagnostic_spectral_names = parse_diagnostic_matrix_names(
    diagnostic_spectral_matrix_names
)
if diagnostics_enabled and optimizer_name != 'global_sgdm':
    raise ValueError(
        "global SGDM diagnostics require optimizer_name='global_sgdm'"
    )
if optimizer_name != 'adamw' and decay_lr and learning_rate <= 0.0:
    raise ValueError(
        "learning_rate must be positive because it defines the shared "
        "experimental LR schedule scale"
    )
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer, optimizer_audit = configure_optimizer(
    model=model,
    optimizer_name=optimizer_name,
    device_type=device_type,
    adamw_learning_rate=learning_rate,
    adamw_weight_decay=weight_decay,
    adamw_betas=(beta1, beta2),
    matrix_learning_rate=matrix_learning_rate,
    matrix_momentum=matrix_momentum,
    matrix_weight_decay=matrix_weight_decay,
    auxiliary_learning_rate=auxiliary_learning_rate,
    auxiliary_weight_decay=auxiliary_weight_decay,
    auxiliary_betas=(auxiliary_beta1, auxiliary_beta2),
    matrix_momentum_convention=matrix_momentum_convention,
    matrix_weight_decay_mode=matrix_weight_decay_mode,
    auxiliary_weight_decay_mode=auxiliary_weight_decay_mode,
    matrix_nesterov=matrix_nesterov,
)
optimizer_group_signature = build_optimizer_group_signature(model, optimizer)
if master_process:
    print("optimizer parameter audit:")
    print(json.dumps(optimizer_audit, indent=2, sort_keys=True))
if optimizer_name == 'adamw':
    optimizer_settings = {
        'learning_rate': learning_rate,
        'betas': [beta1, beta2],
        'weight_decay': weight_decay,
        'weight_decay_mode': 'adamw_decoupled',
    }
else:
    optimizer_settings = {
        'matrix_learning_rate': matrix_learning_rate,
        'matrix_momentum': matrix_momentum,
        'matrix_momentum_convention': matrix_momentum_convention,
        'matrix_nesterov': matrix_nesterov,
        'matrix_weight_decay': matrix_weight_decay,
        'matrix_weight_decay_mode': matrix_weight_decay_mode,
        'auxiliary_learning_rate': auxiliary_learning_rate,
        'auxiliary_betas': [auxiliary_beta1, auxiliary_beta2],
        'auxiliary_weight_decay': auxiliary_weight_decay,
        'auxiliary_weight_decay_mode': auxiliary_weight_decay_mode,
    }

diagnostics = GlobalSGDMDiagnostics(
    enabled=diagnostics_enabled,
    steps=parsed_diagnostic_steps,
    spectral_matrix_names=parsed_diagnostic_spectral_names,
    epsilon=diagnostics_epsilon,
)
diagnostic_eligible_parameters = (
    partition_optimizer_parameters(model).eligible_matrices
    if diagnostics_enabled
    else ()
)
if diagnostics_enabled:
    diagnostics.validate_parameter_names(diagnostic_eligible_parameters)
    if master_process:
        initialize_diagnostic_log(out_dir, resume=init_from == 'resume')
if master_process:
    initialize_evaluation_log(out_dir, resume=init_from == 'resume')

if init_from == 'resume':
    validate_resume_compatibility(
        checkpoint,
        optimizer_name=optimizer_name,
        optimizer_group_signature=optimizer_group_signature,
        optimizer_settings=optimizer_settings,
    )
    optimizer.load_state_dict(checkpoint['optimizer'])
    previous_run_metadata = checkpoint['run_metadata']
else:
    previous_run_metadata = {}

base_run_metadata = build_run_metadata(
    config=config,
    git_commit=get_git_commit_hash(os.getcwd()),
    model_args=model_args,
    trainable_parameter_count=sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ),
    optimizer_name=optimizer_name,
    optimizer_settings=optimizer_settings,
    optimizer_group_signature=optimizer_group_signature,
    parameter_group_audit=optimizer_audit,
    seed=seed,
    dataset=dataset,
    tokenizer=tokenizer,
    block_size=block_size,
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    ddp_world_size=ddp_world_size,
    tokens_per_iteration=tokens_per_iter,
    precision={
        'dtype': dtype,
        'grad_scaler_enabled': scaler.is_enabled(),
        'allow_tf32': True,
    },
    hardware=get_hardware_metadata(torch, device, device_type),
)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# measurements persisted in checkpoints and the run summary
previous_metrics = previous_run_metadata.get('metrics', {})
previous_progress = previous_run_metadata.get('progress', {})
completed_wall_time = previous_metrics.get('total_wall_time_seconds', 0.0)
gradient_clipping_count = previous_metrics.get('gradient_clipping_count', 0)
numerical_event_count = previous_metrics.get('numerical_event_count', 0)
numerical_status = previous_metrics.get('numerical_status', 'ok')
divergence_status = previous_metrics.get('divergence_status', 'not_observed')
latest_training_loss = previous_metrics.get('latest_training_loss')
latest_evaluation_train_loss = previous_metrics.get('latest_evaluation_train_loss')
latest_validation_loss = previous_metrics.get('latest_validation_loss')
latest_evaluation_step = previous_metrics.get('latest_evaluation_step')
successful_optimizer_steps = previous_progress.get('optimizer_steps', iter_num)
optimizer_step_timer = OptimizerStepTimer(
    torch,
    device_type=device_type,
    initial_total_seconds=previous_metrics.get(
        'optimizer_step_time_total_seconds', 0.0
    ),
    initial_step_count=previous_metrics.get('optimizer_step_time_samples', 0),
)
if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
run_session_start = time.perf_counter()

def current_run_metadata():
    optimizer_step_timer.flush()
    effective_lrs = get_effective_learning_rates(optimizer, optimizer_name)
    peak_memory = get_peak_gpu_memory_metadata(torch, device, device_type)
    total_wall_time = (
        completed_wall_time + time.perf_counter() - run_session_start
    )
    clipping_frequency = (
        gradient_clipping_count / iter_num if iter_num > 0 else 0.0
    )
    progress = {
        'optimizer_step_attempts': iter_num,
        'optimizer_steps': successful_optimizer_steps,
        'processed_tokens': iter_num * tokens_per_iter,
        'configured_max_iters': max_iters,
        # Preserve nanoGPT's existing inclusive max_iters termination behavior.
        'planned_optimizer_step_attempts': max_iters + 1,
        'configured_token_budget': (max_iters + 1) * tokens_per_iter,
    }
    metrics = {
        'latest_training_loss': latest_training_loss,
        'latest_evaluation_train_loss': latest_evaluation_train_loss,
        'latest_validation_loss': latest_validation_loss,
        'latest_evaluation_step': latest_evaluation_step,
        'effective_matrix_learning_rate': effective_lrs['matrix'],
        'effective_auxiliary_learning_rate': effective_lrs['auxiliary'],
        'total_wall_time_seconds': total_wall_time,
        'mean_optimizer_step_time_seconds': optimizer_step_timer.mean_seconds,
        'optimizer_step_time_total_seconds': optimizer_step_timer.total_seconds,
        'optimizer_step_time_samples': optimizer_step_timer.step_count,
        'gradient_clipping_enabled': grad_clip != 0.0,
        'gradient_clipping_count': gradient_clipping_count,
        'gradient_clipping_frequency': clipping_frequency,
        'numerical_event_count': numerical_event_count,
        'numerical_status': numerical_status,
        'divergence_status': divergence_status,
        'peak_gpu_memory_bytes': peak_memory['bytes'],
        'peak_gpu_memory_status': peak_memory['status'],
    }
    return snapshot_run_metadata(base_run_metadata, progress, metrics)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    # Experimental matrix and auxiliary rates share this dimensionless shape.
    schedule_scale = (
        lr / learning_rate
        if optimizer_name != 'adamw' and decay_lr
        else 1.0
    )
    set_optimizer_learning_rates(
        optimizer,
        optimizer_name=optimizer_name,
        adamw_learning_rate=lr,
        experimental_schedule_scale=schedule_scale,
    )
    effective_lrs = get_effective_learning_rates(optimizer, optimizer_name)

    if iter_num % eval_interval == 0:
        optimizer_step_timer.flush()

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        latest_evaluation_train_loss = float(losses['train'])
        latest_validation_loss = float(losses['val'])
        latest_evaluation_step = iter_num
        if not (
            math.isfinite(latest_evaluation_train_loss)
            and math.isfinite(latest_validation_loss)
        ):
            numerical_status = 'nonfinite_evaluation_loss'
            divergence_status = 'observed'
            numerical_event_count += 1
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        append_evaluation_record(
            out_dir,
            {
                'step': iter_num,
                'processed_tokens': iter_num * tokens_per_iter,
                'train_loss': latest_evaluation_train_loss,
                'validation_loss': latest_validation_loss,
                'matrix_learning_rate': effective_lrs['matrix'],
                'auxiliary_learning_rate': effective_lrs['auxiliary'],
            },
        )
        if wandb_log:
            wandb_metrics = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "optimizer/auxiliary_lr": effective_lrs['auxiliary'],
                "mfu": running_mfu*100, # convert to percentage
            }
            if effective_lrs['matrix'] is not None:
                wandb_metrics["optimizer/matrix_lr"] = effective_lrs['matrix']
            wandb.log(wandb_metrics)
        run_metadata = current_run_metadata()
        write_run_summary(out_dir, run_metadata)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'run_metadata': run_metadata,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    collect_diagnostics = master_process and diagnostics.should_collect(iter_num)
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip
        )
        total_gradient_norm_value = float(total_gradient_norm)
        if math.isfinite(total_gradient_norm_value):
            if total_gradient_norm_value > grad_clip:
                gradient_clipping_count += 1
        else:
            numerical_status = 'nonfinite_gradient_norm'
            numerical_event_count += 1
    elif collect_diagnostics:
        # Diagnostics must observe the same unscaled gradient used by the optimizer.
        scaler.unscale_(optimizer)
    diagnostic_context = (
        diagnostics.begin_step(
            iter_num,
            diagnostic_eligible_parameters,
            optimizer.matrix_optimizer,
        )
        if collect_diagnostics
        else None
    )
    # step the optimizer and scaler if training in fp16
    previous_scale = scaler.get_scale() if scaler.is_enabled() else None
    optimizer_step_timer.start()
    scaler.step(optimizer)
    scaler.update()
    optimizer_step_timer.stop()
    step_was_skipped = (
        previous_scale is not None
        and scaler.get_scale() < previous_scale
    )
    if step_was_skipped:
        numerical_status = 'grad_scaler_backoff'
        numerical_event_count += 1
    else:
        successful_optimizer_steps += 1
    if diagnostic_context is not None:
        diagnostic_record = diagnostics.end_step(
            diagnostic_context,
            optimizer.matrix_optimizer,
            optimizer_step_applied=not step_was_skipped,
        )
        diagnostic_record['processed_tokens_before_step'] = (
            iter_num * tokens_per_iter
        )
        diagnostic_record['processed_tokens_after_step'] = (
            (iter_num + 1) * tokens_per_iter
        )
        append_diagnostic_record(out_dir, diagnostic_record)
        diagnostic_context = None # release temporary parameter snapshots promptly
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        latest_training_loss = lossf
        if not math.isfinite(lossf):
            numerical_status = 'nonfinite_training_loss'
            divergence_status = 'observed'
            numerical_event_count += 1
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        matrix_lr_text = (
            'not_applicable'
            if effective_lrs['matrix'] is None
            else f"{effective_lrs['matrix']:.6g}"
        )
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%, matrix lr {matrix_lr_text}, "
            f"auxiliary lr {effective_lrs['auxiliary']:.6g}"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

optimizer_step_timer.flush()
if master_process:
    write_run_summary(out_dir, current_run_metadata())

if ddp:
    destroy_process_group()
