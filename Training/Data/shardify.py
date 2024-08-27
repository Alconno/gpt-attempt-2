"""
Run simply as:
$ python fineweb.py
Will save shards to the local directory.
"""


import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm
import sys


# Add main directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# init the tokenizer
from Tokenization.cl100k_enc import enc
sot = enc._special_tokens['<|startoftext|>'] # start of text token
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokens = [sot] # the special <|startoftext|> token 
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens.append(eot) # the special <|endoftext|> token 
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint32 = tokens_np.astype(np.uint32)
    return tokens_np_uint32

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def create_shards(local_dir_name, HW_dataset, remote_name, file_dataset, shard_size=1e8):
    # Check if local_dir is valid and if it exists
    assert len(local_dir_name) > 0, f"Shard directory cannot be empty."
    local_dir = f"Shards/{local_dir_name}"

    # If HW dataset is empty, it should be a file dataset run
    if len(HW_dataset) == 0:
        assert len(file_dataset) > 0, f"File dataset name cannot be empty."
        file_dataset = f"./InputFiles/{file_dataset}"

    # If file dataset is empty, it should be a HW dataset run
    elif len(file_dataset) == 0:
        assert len(HW_dataset) > 0, f"HW_dataset name cannot be empty."
        assert len(remote_name) > 0, f"remote_name cannot be empty."
    
    # Create the cache directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    if len(HW_dataset) > 0 and len(remote_name) > 0:
        # Download the dataset from HW
        fw = load_dataset(HW_dataset, name=remote_name, split="train")
    elif len(file_dataset) > 0:
        # Get dataset from txt file
        fw = load_dataset('text', data_files={'train': file_dataset})["train"]

    # Convert shard_size to an integer
    shard_size = int(shard_size)

    # Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{local_dir_name}_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{local_dir_name}_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == '__main__':
    mp.freeze_support()        
    
    # text file shardifying
    create_shards("inputData", "", "", "input.txt", 1e4)

    # Hugging face dataset shardifying
    #create_shards("edu_fineweb10B", "HuggingFaceFW/fineweb-edu", "sample-10BT", "")

    print("Shard creation completed")

