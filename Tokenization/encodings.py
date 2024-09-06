import tiktoken

cl100k_base = tiktoken.get_encoding('cl100k_base')
gpt2_base = tiktoken.get_encoding('gpt2')

cl100k_enc = tiktoken.Encoding(
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
    }
)

gpt2_enc = tiktoken.Encoding(
    name="gpt2",
    pat_str=gpt2_base._pat_str,
    mergeable_ranks=gpt2_base._mergeable_ranks,
    special_tokens={
        **gpt2_base._special_tokens,
    }
)

enc = gpt2_enc

"""
allowed_special = {
    '<|startoftext|>', 
    '<|endoftext|>', 
    'system', 
    'user', 
    'assistant'
}

special_tokens={
    **cl100k_base._special_tokens,
    "<|startoftext|>": 100264,
    "<|endoftext|>": 100265,
    "system": 100266,
    "user": 100267,
    "assistant": 100268,
}"""