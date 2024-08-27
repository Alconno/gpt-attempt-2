import torch
import numpy as np
import os

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, shard_training_data_root, master_process,
                process_rank=0, num_processes=1, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.advancement = B*T*num_processes
        assert split in {'train', 'val'}

        # Shard Training
        shards = os.listdir(shard_training_data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(shard_training_data_root,s) for s in shards]
        
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
    
    def reset(self):
        # initialize at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T + self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        
        x = (buf[:-1]).view(B, T) # inputs (everything except last character)
        y = (buf[1:]).view(B, T) # targets (everything except first character)
        
        # adva  nce the position in the tensor
        self.current_position += self.advancement

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard+1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x,y


