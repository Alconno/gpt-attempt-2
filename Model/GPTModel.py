from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Block import Block
import inspect, math
import Tokenization.encodings as enc

# ----------------------------------------------------------------------
## Model setup
# ----------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 256 # Sequence length
    vocab_size: int = 50304
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True






# ----------------------------------------------------------------------
## Model functionality
# ----------------------------------------------------------------------
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd), # Each token's embedding
            wpe=nn.Embedding(config.block_size, config.n_embd), # Each token's positional embedding depending on sequence index (0-block_size)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Hidden layer
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        # initialize weights
        self.apply(self._init_weights)

        # Report nubmer of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def _init_weights(self, module):
        # The standard deviation controls the spread or scale of the initialized weights.
        std = 0.02 
        # The mean in weight initialization is generally set to zero in order to center the distribution of weights. 
        mean = .0

        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -.5
            
            # The mean in weight initialization is generally set to zero in order to center the distribution of weights. 
            torch.nn.init.normal_(module.weight, mean, std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean, std)

        # Apply special scaled init to residual projections
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=.0, std=.02/math.sqrt(2*self.config.n_layer))


    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    

    def forward(self, token_indexes, targets=None):
        device = token_indexes.device
        B, T = token_indexes.size()
        assert T <= self.config.block_size, f"Sequence length (T: {T}) has to be <= to block size {self.config.block_size}"

        # Run token indexes through token embeddings layer
        token_embeddings = self.transformer.wte(token_indexes) # (B,T,n_embd)

        # Get default position embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=device) # The position embeddings are the same for every sequence because
        # they are designed to reflect the position in a generic sense, not relative to the specific content of the sequence. 
        position_embeddings = self.transformer.wpe(positions) # (T,n_embd)
        
        # Add position embeddings to every batch of token embeddings (B,(T,n_embd)+(T,n_embd))
        x = self.transformer.drop(token_embeddings + position_embeddings) # (B, T, n_embd)
        
        # Run embedding results through hidden layer Blocks (SelfAttention, MLP)
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm to stabilize and improve the training
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we're given some desired targets to calculate the loss
            logits = self.lm_head(x) # output layer (B, T, vocab_size)
            # targest are size (B, T)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # ignore padded sequences - This argument 
            # tells the loss function to ignore any target values that are equal to -1. 
        else:
            # inference-time mini-optimization: only forward lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        # only dropout can be overridden, see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel # type: ignore
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args =  {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch innitialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # just a buffer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use "Conv1D" module, but we only want to use vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all parameters
        param_dict = {pn: p for pn,p in self.named_parameters()}
        # Filter out those that do not require gradient
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that are 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms won't be decayed.
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters.")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = ('fused' in inspect.signature(torch.optim.AdamW).parameters) and device_type == 'cuda'
        # Fused AdamW minimizes the number of times data needs to be read from and written to memory. 
        # This can lead to better utilization of GPU memory bandwidth.
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {fused_available}")

        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
