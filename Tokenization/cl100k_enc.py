import tiktoken

cl100k_base = tiktoken.get_encoding('cl100k_base')

enc = tiktoken.Encoding(
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|startoftext|>": 100264,
        "<|endoftext|>": 100265,
        "system": 100266,
        "user": 100267,
        "assistant": 100268,
        "context": 100269,
    }
)

allowed_special = {
    '<|startoftext|>', 
    '<|endoftext|>', 
    'system', 
    'user', 
    'assistant'
}