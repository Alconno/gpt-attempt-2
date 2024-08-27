from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2);
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True) # flash attention
        y = y.transpose(1,2).contiguous().view(B,T,C) 
        y = self.c_proj(y)
        return y
    

"""


Example:
- block_size=128
- vocab_size=50304
- n_layer=4
- n_head=4
- n_embd=128
- B=16
- T=64
- total_batch_size=32768

B*T arrays length C   ==>  c_attn  ==>  B*T arrays length C*3
qkv (B, T, C*3) or (batch_size, sequence length, tokens*3)
q,k,v (B,T,C)
q,k,v (16,64,384)

k.view(B,T,n_head,C/n_head).transpose(2,1)
    = (16,64,4,32).transpose(2,1)
    = (16,4,64,32)

after flash attention y   3*(16,4,64,32)  ==>  (16,4,64,32)
y.transpose(1,2)  ==>  (16,64,4,32)
y.contigous()  ==>  make tensor contigous in memory
y.view(B,T,C)  ==>  (16,64,128)

c_proj(y)  ==>  (16,64,128)
   
"""