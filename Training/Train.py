import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
from Model.GPTModel import GPT, GPTConfig
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import Evaluators.hellaswag as hellaswag
import platform
import Tokenization.cl100k_enc as cl100k_enc


OS = platform.system()

# ----------------------------------------------------------------------
## Training setup
# ----------------------------------------------------------------------
enc = cl100k_enc.enc
total_batch_size = 32768 # 2**19, number of tokens
B = 16 # batch size
T = 64 # sequence length

# Shard Training
shard_size = 1e8 # 100M tokens per shard
shard_training_data_root = "./Data/Shards/inputData"
# HW_dataset = "HuggingFaceFW/fineweb-edu"
# file_dataset = "input.txt"


# Pretrained Model
pretrained_model = None # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# Gradient accumulation
use_grad_accum = True # prevents backend running out of memory

# Torch Compile (Linux + CUDA required)
use_torch_compile = False and OS == "Linux"

# Learning rate
max_lr = 6e-4
min_lr = max_lr * 0.1 # 10%
warmup_steps = 20
max_steps = 19073
running_mfu = -1.0

# Model Generation
generate_every_n_steps = 3 # sample every 3 iters
num_samples = 4 # how many samples should it generate
sample_max_length = 32 # max length how generation
topk_precision = 30 # lower: more precise/accurate | higher: more diverse and creative output

# Validation Evaluation
eval_val_loss_every_n_steps = 100
val_loss_steps = 20

# Hellaswag Evaluation
eval_hellaswag_every_n_steps = 2

# Device
device = "cpu"

# Logs
log_dir = "log"
log_file = "log.txt"

# Manual seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)



# ----------------------------------------------------------------------
# Set up DDP(Distributed Data Parallel)
cuda_available = torch.cuda.is_available()
rank_str = os.environ.get('RANK')
ddp = int(rank_str) != 1 if rank_str is not None else False # Default to False if 'RANK' is not set
if ddp:
    # Use of DDP atm demands CUDA, we set the device according to the rank
    assert cuda_available, "CUDA required to run with DDP"
    init_process_group(backend=('nccl' if OS=="Linux" else 'gloo')) # nccl on linux
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # Vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    
    # Attempt to autodetect device
    if cuda_available:
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


# ----------------------------------------------------------------------
# Gradient accumulation
if use_grad_accum:
    assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*ddp_world_size"
    grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
else:
    grad_accum_steps = 1



# ----------------------------------------------------------------------
# Model Creation
model = GPT(GPTConfig(vocab_size=50304)) if pretrained_model==None else GPT.from_pretrained(pretrained_model)
model.to(device)

# Torch Compile (Linux + CUDA Required)
if use_torch_compile:
    model = torch.compile(model)

# DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# Optimizer
optimizer = raw_model.configure_optimizers(weight_decay=.1, learning_rate=6e-3, device_type=device, betas=(.9,.95))

# Create the log directory for writing checkpoints and logs
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, log_file)
with open(log_file, "w") as f: # open for writing to clear the file
    pass
sys.path.append(os.path.abspath(os.path.join(os.path.curdir, log_dir)))


# Data Loader
from Training.Data.DataLoader import DataLoader
train_loader = DataLoader(B, T, shard_training_data_root, master_process, 
                        process_rank=ddp_rank, num_processes=ddp_world_size, split='train')

val_loader = DataLoader(B, T, shard_training_data_root, 
                        master_process, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')


# ---------------------------- Helper functions ------------------------------------------
from Helpers import get_most_likely_row
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def eval_val_loss():
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x,y = val_loader.next_batch()
            x,y = x.to(device), y.to(device)
            if device == 'cuda':
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x,y)
            else:
                logits, loss = model(x,y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"Validation Loss: {val_loss_accum.item():.4f}")

def eval_hellaswag():
    num_corrent_norm = 0
    num_total = 0
    for i,example in enumerate(hellaswag.iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        
        # render the example into tokens and labels
        _, tokens, mask, label = hellaswag.render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        pred_norm = 0
        with torch.no_grad():
            if device=='cuda':
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm = int(pred_norm == label)
    # reduce thestats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")


# ------------------------------------------------------------------------------------------


for step in range(max_steps):
    last_step = (step == max_steps-1)

    # Evaluate Validation Loss
    if step % eval_val_loss_every_n_steps == 0 or last_step:
        eval_val_loss()

    # Evaluate HellaSwag
    if (step % eval_hellaswag_every_n_steps == 0 or last_step) and (not use_torch_compile):
        eval_hellaswag()

    # Once in a while generate from the model (except step 0, which is a noise)
    # Not working with torch.compile for some reason
    if ((step>0 and step & generate_every_n_steps == 0) or last_step) and (not use_torch_compile):
        tokens = enc.encode("<|startoftext|>Today I was", cl100k_enc.allowed_special)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_samples, 1) # (num_samples, len(original_tokens))
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device)
        sample_rng.manual_seed(42+ddp_rank)
        
        while xgen.size(1) < sample_max_length:
            # Forward the model to get the logits
            while torch.no_grad():
                logits, loss = model(xgen) # (B,T,vocab_size)
               
                # Take the logits at the last position
                logits = logits[:,-1,:]/1.0 # (B, vocab_size)
               
                # Get the probabilities
                probs = F.softmax(logits, dim=-1)
                # Get the top 'topk_precision' probabilities and their corresponding indices (vocab tokens)
                # Use probs to pick 1 result and then pluck a token out of indicies depending on that result
                topk_probs, topk_indicies = torch.topk(probs, topk_precision, dim=-1)  # (B, 50)
                # Sample one token from the top 50 probabilities for each batch element
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # Map the sampled indices back to the original vocab indices
                xcol = torch.gather(topk_indicies, -1, ix)  # (B, 1)
                # Append the newly generated token to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)  # (B, T+1)
        
        # Print the generated text
        for i in range(num_samples):
            tokens = xgen[i, :sample_max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # Training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    t0 = time.perf_counter() 
    
    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        
        if device == 'cuda':
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x,y)
        else:
            logits, loss = model(x,y)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        # to avoid gradient sync for every micro_step
        if ddp: 
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        
        loss.backward()

 


