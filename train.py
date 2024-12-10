import os, sys
import time
import math
from contextlib import nullcontext
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from seq2seq import Seq2SeqConfig, Seq2Seq

import h5py
from hdf5dataloader import HDF5Dataset, HDF5DataLoader
from helper import decode

os.environ['PATH'] += ':/sbin'

# -----------------------------------------------------------------------------
# I/O default settings (can be overriden by config file)
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'

bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
gradient_accumulation_steps = 1 # accumulate gradients across this many batches before updating weights
# learning rate decay settings
decay_lr = True # whether to decay the learning rate

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    print(f'hello from gpu {ddp_rank+1} of {ddp_world_size}',flush=True)
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# -----------------------------------------------------------------------------
# config file parsing (first argument is the python config file to load)
exec(open(sys.argv[1]).read())

# save all config variables in a dictionary for logging
job_id = os.environ.get('SLURM_JOB_ID')
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging

if master_process:
    for k,v in config.items():
        print(f"{k} = {v}")
# -----------------------------------------------------------------------------

#include task directory in path
import sys
sys.path.append(task_dir)

try:
    from data_main.graphdata import print_diff
except:
    def print_diff(src, tgt, pred, token_dict):
        print('print_diff failed')


tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * (src_len + tgt_len)
samples_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
if master_process: print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    # create the output directory
    os.makedirs(out_dir, exist_ok=True)
    #save the config as python file
    with open(os.path.join(out_dir, 'config.py'), 'w') as f:
        for k,v in config.items():
            f.write(f"{k} = {v}\n")

torch.manual_seed(137 + seed_offset) # approx 1/fine-structure-constant
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, src_len=src_len, tgt_len=tgt_len,
                  bias=bias, src_vocab_size=None, tgt_vocab_size=None, dropout=dropout) # start with model_args from command line

# determine the vocab size we'll use for from-scratch training
# load token json from 'token_dict'
src_token_dict = json.load(open(src_token_dict))
tgt_token_dict = json.load(open(tgt_token_dict))
model_args['src_vocab_size'] = (len(src_token_dict)//64 + 1)*64 # round up to nearest multiple of 64
model_args['tgt_vocab_size'] = (len(tgt_token_dict)//64 + 1)*64 # round up to nearest multiple of 64
model_args['src_pad_token_id'] = src_token_dict['<PAD>']
model_args['tgt_pad_token_id'] = tgt_token_dict['<PAD>']

# data loader
val_data = HDF5Dataset(os.path.join(data_dir, val_filename), max_src_len=src_len, max_tgt_len=tgt_len, pad_token=0, minus_token=src_token_dict['-'], src_key=src_key, tgt_key=tgt_key)
val_dataloader = HDF5DataLoader(val_data, batch_size, ddp_local_rank, ddp_world_size, device, shuffle=False)

if init_from == 'scratch':
    if master_process: print("Initializing a new model from scratch")
    seq2seqconf = Seq2SeqConfig(**model_args)
    model = Seq2Seq(seq2seqconf)
elif init_from == 'resume':
    if master_process: print(f"Resuming training from {cp_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(cp_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    for k in ['n_layer', 'n_head', 'n_embd', 'src_len', 'tgt_len', 'bias', 'src_vocab_size', 'tgt_vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    modelconf = Seq2SeqConfig(**model_args)
    model = Seq2Seq(modelconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, device_type)
if init_from == 'resume': optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
checkpoint_model_args = None

# compile the model
if compile:
    print("compiling the model...")
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
    for split, dataloader in {'train': train_dataloader, 'val': val_dataloader}.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            S, X, Y = dataloader.get_batch()
            with ctx:
                logits, loss = model(S, X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging on w&b
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed

while True:
    # the training data is split into multiple files, load them one by one to save time on I/O
    for data_ind in range(split_train):
        if master_process: print(f'loading data {data_ind}')
        train_data = HDF5Dataset(os.path.join(data_dir, f'{traindata_prefix}{data_ind}.hdf5'), max_src_len=src_len, max_tgt_len=tgt_len, pad_token=0, minus_token=src_token_dict['-'], src_key=src_key, tgt_key=tgt_key)
        train_dataloader = HDF5DataLoader(train_data, batch_size, ddp_local_rank, ddp_world_size, device, shuffle=True)
        if local_iter_num == 0: S, X, Y = train_dataloader.get_next()

        for _ in range(train_dataloader.n_batches):
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if (iter_num % eval_interval == 0 and master_process and iter_num > 0) or (master_process and iter_num == 50):
                print('evaluating loss...')
                tt = time.time()
                losses = estimate_loss()
                print(f"evaluation done in {time.time() - tt:.2f}s")
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)

                # generate some samples from the model to see how it's doing
                for ii in range(3):
                    print(f'--- sample {ii} ---')
                    # # generate a prediction from the model
                    src, _, tgt = val_dataloader.get_one()
                    tgt = tgt.tolist()[0]
                    pred = raw_model.generate(src, start_token_id=1, end_token_id=2)
                    src = src.cpu().numpy().tolist()[0]
                    pred = pred.cpu().numpy().tolist()[0]

                    print('src')
                    print(src)
                    print(decode(src, src_token_dict))

                    print('pred')
                    print(pred)
                    print(decode(pred, tgt_token_dict))

                    print('tgt')
                    print(tgt)
                    print(decode(tgt, tgt_token_dict))

                    print_diff(src, tgt, pred, src_token_dict)

                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                    })
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
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, 
                    # but it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(S, X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                S, X, Y = train_dataloader.get_next()
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
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
                if master_process: print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.4e}, samples processed {iter_num*samples_per_iter:,}, samples per second {samples_per_iter/dt:.2f}")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "loss": lossf,
                        "lr": lr,
                        "samples_processed": iter_num*samples_per_iter,
                        "samples_per_second": samples_per_iter/dt,
                    })
            iter_num += 1
            local_iter_num += 1
