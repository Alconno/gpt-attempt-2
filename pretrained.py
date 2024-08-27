from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()

# print params in gpt-2 and their shape 
for k,v in sd_hf.items():
    print(k, v.shape)

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generated = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=2, truncation=True)

print(generated)